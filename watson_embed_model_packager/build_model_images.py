"""
This script builds one or more model images based on a configuration
spreadsheet. The spreadsheet should be formatted as:

<model_source>,<model_name>,<target_image_name>

* <model_source>: A fully-qualified path in artifactory or a path on disk
* <model_name>: The string name that will be used to release the model
* <target_image_name>: The target image repository
    (e.g. icr.io/cp/watson-ai/nlp-keywords-en:1.2.3)
"""

# Standard
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse
import csv
import glob
import os
import shlex
import shutil
import subprocess
import tempfile

# First Party
import alog

# Local
from .common_args import bool_from_env, get_command_parser, handle_logging_args
from .constants import CONFIG_CSV_COL_HEADERS

## Constants ###################################################################

log = alog.use_channel("BUILD")

# File constants pointing to the docker build assets
RESOURCES_DIR = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        "resources",
    )
)
ARTIFACTORY_DOCKERFILE = os.path.join(RESOURCES_DIR, "artifactory.dockerfile")
LOCAL_DOCKERFILE = os.path.join(RESOURCES_DIR, "local.dockerfile")

# We'll figure out which docker build command to use on the first build
# invocation
DOCKER_BUILD_COMMAND = None

## Helpers #####################################################################


@dataclass
class ModelBuildConfig:
    """Class holding the config for an individual model build"""

    model_source: str
    model_name: str
    target_image_name: str
    image_labels: str


def parse_config_csv(config_csv: str) -> List[ModelBuildConfig]:
    """Parse the config csv into a list of build configs"""
    with open(config_csv, "r") as handle:
        csv_rows = list(csv.reader(handle))

    # Make sure there's a header and at least one value row
    if len(csv_rows) < 2:
        raise ValueError(
            f"CSV Config [{config_csv}] must include a header row and at least one value row"
        )
    header_row, value_rows = csv_rows[0], csv_rows[1:]

    # Make sure the header has the right columns
    if any(expected_col not in header_row for expected_col in CONFIG_CSV_COL_HEADERS):
        raise ValueError(
            "Missing required column headers: {}".format(
                set(CONFIG_CSV_COL_HEADERS) - set(header_row)
            )
        )

    # Parse each row into a ModelBuildConfig
    col_idxs = {col: header_row.index(col) for col in CONFIG_CSV_COL_HEADERS}
    return [
        ModelBuildConfig(**{col: row[col_idxs[col]] for col in CONFIG_CSV_COL_HEADERS})
        for row in value_rows
    ]


def do_docker_build(
    dockerfile: str,
    working_dir: str,
    image_tag: str,
    build_args: Dict[str, str],
    build_labels: Dict[str, str],
):
    """Helper to perform a docker build in a given working dir"""
    # If this is the first build, figure out the right command
    global DOCKER_BUILD_COMMAND
    if DOCKER_BUILD_COMMAND is None:
        check_proc = subprocess.Popen(
            shlex.split("docker buildx build --help"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = check_proc.communicate()
        if "--platform" in out.decode("utf-8"):
            log.info("Building with --platform")
            DOCKER_BUILD_COMMAND = "docker buildx build --platform linux/amd64"
        else:
            log.info("Building without --platform")
            DOCKER_BUILD_COMMAND = "docker build"

    build_cmd = f"{DOCKER_BUILD_COMMAND} . -f {dockerfile} -t {image_tag}"
    for arg_name, arg_val in build_args.items():
        build_cmd += f" --build-arg {arg_name}={arg_val}"
    for label_name, label_val in build_labels.items():
        build_cmd += f' --label "{label_name}={label_val}"'
    proc = subprocess.Popen(shlex.split(build_cmd), cwd=working_dir)
    proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("Docker build failed!")


def link_or_copy(source: str, target: str):
    """Helper to try to hardlink a file and fall back to copying if not
    possible. This handles the case when the source code is mounted and the
    tempfile is created on a non-mounted volume, but we can't easily ensure that
    it's covered by tests since it's runtime dependent.
    """
    try:
        os.link(source, target)
    except OSError:  # pragma: no cover
        shutil.copy(source, target)


def build_model_image(
    model_config: ModelBuildConfig,
    artifactory_username: Optional[str],
    artifactory_api_key: Optional[str],
    strict: bool,
) -> bool:
    """Perform a single model build"""

    # If the model source is a file, use the local docker build and set up the
    # working dir to contain the local file and the build files
    build_args = {"MODEL_NAME": model_config.model_name}
    build_labels = {}
    for label in model_config.image_labels.split(";"):
        label_key_value = label.split("=")
        label_key = label_key_value[0]
        label_val = label_key_value[1]
        build_labels[label_key] = label_val
    try:
        if os.path.exists(model_config.model_source):
            log.debug("Using local model for [%s]", model_config.model_name)
            dockerfile = LOCAL_DOCKERFILE

            with tempfile.TemporaryDirectory() as working_dir:
                build_args[
                    "MODEL_DEST"
                ] = f"/model_landing_zone/{model_config.model_name}"

                # Hard link all of the scripts from the resources dir
                for fname in glob.glob(f"{RESOURCES_DIR}/*.sh"):
                    link_or_copy(
                        fname, os.path.join(working_dir, os.path.basename(fname))
                    )

                # Hard link all files in the model directory
                log.debug2(f"Looking at model dir: {model_config.model_source}")
                if os.path.isdir(model_config.model_source):
                    log.debug2("Linking all files in model directory...")
                    model_files = {
                        os.path.realpath(fname): os.path.relpath(fname, os.getcwd())
                        for fname in glob.glob(
                            f"{model_config.model_source}/**", recursive=True
                        )
                        if not os.path.isdir(fname)
                    }
                    # Set the model path to copy from, need a relative path to {working_dir}
                    build_args["MODEL_PATH"] = os.path.relpath(
                        model_config.model_source, os.getcwd()
                    )
                else:
                    # TODO: this is probably a zip and zips probably won't work :D
                    log.debug2("Linking single model file...")
                    model_files = {
                        os.path.realpath(model_config.model_source): os.path.basename(
                            model_config.model_source
                        ),
                    }
                    # Set the model path to copy from
                    build_args["MODEL_PATH"] = os.path.basename(
                        model_config.model_source
                    )

                log.debug4(f"Model files: {model_files}")

                for file_source_path, file_target_path in model_files.items():
                    target = os.path.join(working_dir, file_target_path)
                    parent_dir = os.path.dirname(file_target_path)
                    if parent_dir:
                        log.debug2("Making parent dir %s", parent_dir)
                        os.makedirs(
                            os.path.join(working_dir, parent_dir), exist_ok=True
                        )
                    log.debug2("Linking %s -> %s", file_source_path, target)
                    link_or_copy(file_source_path, target)

                # Do the docker build
                do_docker_build(
                    dockerfile=dockerfile,
                    working_dir=working_dir,
                    image_tag=model_config.target_image_name,
                    build_args=build_args,
                    build_labels=build_labels,
                )

        # Otherwise, use the artifactory build and validate the credentials
        else:
            log.debug("Using artifactory model for [%s]", model_config.model_name)
            dockerfile = ARTIFACTORY_DOCKERFILE
            if None in [artifactory_username, artifactory_api_key]:
                raise ValueError("Missing or incomplete artifactory credentials")
            build_args["ARTIFACTORY_USERNAME"] = artifactory_username
            build_args["ARTIFACTORY_API_KEY"] = artifactory_api_key
            build_args["MODEL_URL"] = model_config.model_source
            do_docker_build(
                dockerfile=dockerfile,
                working_dir=RESOURCES_DIR,
                image_tag=model_config.target_image_name,
                build_args=build_args,
                build_labels=build_labels,
            )
    except RuntimeError:
        log.error("Failed to build [%s]!", model_config.model_name)
        if strict:
            raise
        return False
    return True


## Main ########################################################################


def main(parent_parser: Optional[argparse.ArgumentParser] = None):
    """Run the core model construction script"""
    parser = get_command_parser(__doc__, parent_parser)

    # Add the args for this script specifically
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Configuration CSV file",
    )
    parser.add_argument(
        "--push",
        "-p",
        action="store_true",
        default=bool_from_env("PUSH"),
    )
    parser.add_argument(
        "--artifactory-username",
        "-u",
        default=os.environ.get("ARTIFACTORY_USERNAME"),
        help="Username for artifactory access",
    )
    parser.add_argument(
        "--artifactory-api-key",
        "-k",
        default=os.environ.get("ARTIFACTORY_API_KEY"),
        help="API key for artifactory access",
    )
    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        default=bool_from_env("STRICT"),
        help="Any build/push failures are fatal",
    )
    args = parser.parse_args()

    # Configure logging
    handle_logging_args(args)

    # Parse the config spreadsheet
    model_configs = parse_config_csv(args.config)

    # Iterate each build in the sheet and perform the build
    for model_config in model_configs:
        log.info("Building model [%s]", model_config.model_name)
        success = build_model_image(
            model_config=model_config,
            artifactory_username=args.artifactory_username,
            artifactory_api_key=args.artifactory_api_key,
            strict=args.strict,
        )

        # If requested, push the image
        if success and args.push:
            log.debug("Pushing [%s]", model_config.target_image_name)
            proc = subprocess.Popen(
                shlex.split(f"docker push {model_config.target_image_name}")
            )
            proc.communicate()
            if proc.returncode != 0:
                msg = f"Failed to push [{model_config.target_image_name}]"
                log.error(msg)
                if args.strict:
                    raise RuntimeError(msg)


if __name__ == "__main__":  # pragma: no cover
    main()
