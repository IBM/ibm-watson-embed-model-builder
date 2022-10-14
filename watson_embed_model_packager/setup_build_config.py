"""
This script takes a set of Watson Core module IDs that will be supported by a
given watson_runtime image and produces the configuration CSV needed to publish
the latest models supporting the target Watson Core libraries.
"""

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, TextIO, Tuple
import argparse
import asyncio
import csv
import os
import re
import sys
import zipfile

# Third Party
import httpx
import semver
import yaml

# First Party
import alog

# Local
from .common_args import (
    get_command_parser,
    handle_logging_args,
    number_from_env,
    str_list_from_env,
)
from .constants import (
    CONFIG_CSV_COL_HEADERS,
    IMAGE_LABELS,
    LABEL_PREFIX,
    MODEL_NAME,
    MODEL_SOURCE,
    TARGET_IMAGE_NAME,
)

## Constants ###################################################################

log = alog.use_channel("SETUP")

# Some models have outdated module names that need to be mapped forward in order
# to determine the correct domain for the target model image
LIBRARY_PACKAGE_ALIASES = {"lego": "watson_nlp"}

# Regexes to help parse model names
DATE_EXPR = re.compile(
    r"(?P<year>\d\d\d\d)-(?P<month>\d\d)-(?P<day>\d\d)-(?P<hms>\d\d\d\d\d\d)"
)
VERSION_EXPR = re.compile(r"v(?P<major>\d+)-(?P<minor>\d+)-(?P<patch>\d+)")

# Timestamp format for watson core models
# https://github.ibm.com/ai-foundation/watson_nlp/blob/main/scripts/utils/package_and_upload_workflows.py#L138
TIME_PARSE_FORMAT = "%Y-%m-%d-%H%M%S"

# Global singleton client populated at start time
CLIENT = None

## Helpers #####################################################################


@dataclass
class ModelInfo:
    """Common information about a model that is needed to perform filtering and
    lookups
    """

    name: str
    url: str
    guid: str
    parent_library: str
    parent_library_version: semver.VersionInfo
    created: str
    module_class: str


@contextmanager
def output_handle(output_csv: Optional[str]) -> TextIO:
    """Wrapper around open() as a context manager to yield either an open file
    handle or sys.stdout
    """
    if not output_csv:
        yield sys.stdout
    else:
        with open(output_csv, "w") as handle:
            yield handle


def split_artifactory_repo_url(artifactory_repo: str) -> Tuple[str, str]:
    """Split the url and repo name from a fully-qualified repo url"""
    parts = artifactory_repo.rpartition("/")
    return parts[0], parts[2]


async def get_models_in_repo(
    artifactory_repo: str,
    artifactory_username: str,
    artifactory_api_key: str,
) -> List[str]:
    """Get a list of all model zip file URLs from the target artifactory repo"""
    # Find all zip files in the repo
    url, repo = split_artifactory_repo_url(artifactory_repo)
    search_url = f"{url}/api/search/pattern"
    result = await CLIENT.get(
        url=search_url,
        params={"pattern": f"{repo}:/*/*/*.zip"},
        auth=(artifactory_username, artifactory_api_key),
    )
    log.debug("Completed GET request [%s]", search_url)
    return result.json().get("files", [])


def group_by_module_type(model_paths: List[str]) -> Dict[str, List[str]]:
    """Parse the model paths to group individual models by their module types"""
    models_by_type = {}
    invalid_models = []
    for model_path in model_paths:
        # Split each model path into path and name
        path, _, name = model_path.rpartition("/")

        # 🌶️🌶️🌶️ Filter out any "dummy" models
        if "dummy" in name:
            log.warning("Found dummy model [%s]", name)
            invalid_models.append(model_path)
            continue

        # 🌶️🌶️🌶️ Filter out any `bert` models that aren't for `multi` language
        if "bert" in name and "multi" not in name:
            log.warning("Found bert model without multi-language support [%s]", name)
            invalid_models.append(model_path)
            continue

        # 🌶️🌶️🌶️ Filter out any classifiers for emotion:
        if "classification" in name and "emotion" in name:
            log.warning("Skipping emotion classifier [%s]", name)
            invalid_models.append(model_path)
            continue

        # Parse the model names to remove the date stamps and version info
        name_parts = name.split(".")[0].split("_")
        version_matches = [bool(VERSION_EXPR.match(part)) for part in name_parts]
        date_matches = [bool(DATE_EXPR.match(part)) for part in name_parts]

        if version_matches.count(True) != 1:
            log.warning("Found zero or multiple version fields in [%s]", name)
            invalid_models.append(model_path)
            continue
        if date_matches.count(True) != 1:
            log.warning("Found zero or multiple date fields in [%s]", name)
            invalid_models.append(model_path)
            continue
        clean_parts = [
            name_parts[i]
            for i in range(len(name_parts))
            if not version_matches[i] and not date_matches[i]
        ]
        model_type_name = "_".join(clean_parts)
        model_type_name = enforce_naming_conventions(model_type_name)
        log.debug2(
            "Clean name for [%s]: [%s] (path: [%s])", name, model_type_name, path
        )

        # Bin by path and un-versioned model name
        models_by_type.setdefault(model_type_name, []).append(model_path)

    # TODO: Should we do anything meaningful with the invalid models?
    log.debug3("All invalid model names: %s", invalid_models)
    return models_by_type


async def get_model_info_from_remote_config(
    model_name: str,
    model_path: str,
    artifactory_username: str,
    artifactory_api_key: str,
    artifactory_repo: str,
) -> Optional[ModelInfo]:
    """Parse the information about the target model from the name and the
    enclosed config.yml
    """
    # Get the content of config.yml
    model_url = f"{artifactory_repo}/{model_path}"
    full_url = f"{model_url}!/config.yml"
    log.debug3("Full [%s] config.yml url: %s", model_path, full_url)
    try:
        result = await CLIENT.get(
            url=full_url,
            auth=(artifactory_username, artifactory_api_key),
        )
        log.debug("Completed GET request [%s]", full_url)
        log.debug4("Full [%s] config.yml result:\n%s", model_path, result.text)
    except httpx.HTTPStatusError:
        log.warning("No config.yml found for [%s]", model_path)
        return None

    # Parse the yaml
    config = yaml.safe_load(result.text)

    # Update guid, parent_library, parent_library_version, module_class
    model_info = update_model_info_from_config(
        config=config, model_name=model_name, model_path=model_path, model_url=model_url
    )

    return model_info


def enforce_naming_conventions(model_name: str) -> str:
    """
    Enforces model naming conventions, fixing well-known errors in stock watson_nlp model names.

    Args:
        model_name (str): a model name in 5 underscore-delimited sections

    Returns: str
        A model name that conforms to the `$task_$impl_lang_$lang_$desc` format

    """
    # 🌶️🌶️🌶️ Munge the names around so they make sense
    name_parts = model_name.split("_")
    # sentiment-aggregated_foo -> sentiment_aggregated-foo
    if name_parts[0] == "sentiment-aggregated":
        log.debug("Renaming sentiment-aggregated model %s", model_name)
        name_parts[0] = "sentiment"
        name_parts[1] = "aggregated-" + name_parts[1]

    # ensemble_classification-foo -> classification_ensemble-foo
    if name_parts[0] == "ensemble":
        log.debug("Renaming classification ensemble model %s", model_name)
        name_parts[0] = "classification"
        name_parts[1] = name_parts[1].replace("classification", "ensemble")

    # task_impl-wf_blah -> task_impl-workflow_blah
    if name_parts[1].endswith("-wf"):
        log.debug("expanding -wf on implementation tag for %s", model_name)
        name_parts[1] = name_parts[1].replace("-wf", "-workflow")

    # task_impl_lang_code_desc-wf -> task_impl-workflow_lang_code_desc
    if name_parts[-1].endswith("-wf"):
        log.debug(
            "moving -wf flag on description tag to the implementation tag for %s",
            model_name,
        )
        name_parts[-1] = name_parts[-1].replace("-wf", "")
        name_parts[1] = name_parts[1] + "-workflow"
    model_name = "_".join(name_parts)
    log.debug("model name: %s", model_name)
    return model_name


async def get_latest_valid_model(
    model_name: str,
    model_artifacts: List[str],
    artifactory_username: str,
    artifactory_api_key: str,
    artifactory_repo: str,
    module_guids: List[str],
    library_versions: Dict[str, semver.VersionInfo],
) -> Optional[ModelInfo]:
    """Get the latest valid model within the given list of model artifacts for
    the given logical model. If no artifact bundle matches the valid guids, None
    is returned.
    """

    log.debug(
        "Getting latest model for type: %s, models: %s", model_name, model_artifacts
    )

    # Sort the artifacts by date, newest to oldest
    model_artifacts = sorted(
        model_artifacts,
        key=lambda model_name: datetime.strptime(
            model_name.split(".")[0].rpartition("_")[-1],
            TIME_PARSE_FORMAT,
        ),
        reverse=True,
    )

    # Starting with the newest, look up the config.yml and parse it into a
    # ModelInfo
    for model_artifact in model_artifacts:
        model_info = await get_model_info_from_remote_config(
            model_name=model_name,
            model_path=model_artifact,
            artifactory_username=artifactory_username,
            artifactory_api_key=artifactory_api_key,
            artifactory_repo=artifactory_repo,
        )
        if model_info is None:
            log.warning("Failed to create a ModelInfo for %s", model_artifact)
            continue

        # If the guild is not one being released, keep looking
        if model_info.guid not in module_guids:
            log.info(
                "Skipping model %s for unreleased module with guid [%s]",
                model_artifact,
                model_info.guid,
            )
            continue

        # If the library version is not ahead of the supported library versions,
        # return this ModelInfo
        supported_lib_version = library_versions.get(model_info.parent_library)
        if supported_lib_version is None:
            log.warning(
                "Skipping model %s for unreleased library [%s]",
                model_artifact,
                model_info.parent_library,
            )
            continue

        if (
            model_info.parent_library_version
            and model_info.parent_library_version <= supported_lib_version
        ):
            log.info("Found latest model [%s]", model_artifact)
            return model_info
        else:
            log.debug(
                "Skipping model %s with too-new library version %s",
                model_artifact,
                model_info.parent_library_version,
            )


async def get_models_from_repo(
    artifactory_repo: str,
    artifactory_username: str,
    artifactory_api_key: str,
    module_guids: List[str],
    library_versions: Dict[str, semver.VersionInfo],
    path_exprs: Optional[List[str]],
) -> List[ModelInfo]:
    """Get all models from the given repo that should be built into images"""

    # Get a list of all candidate models from the target repo
    candidate_models = await get_models_in_repo(
        artifactory_repo,
        artifactory_username,
        artifactory_api_key,
    )
    log.debug3("Raw candidate models for [%s]: %s", artifactory_repo, candidate_models)

    # Limit models to path expressions
    if path_exprs:
        candidate_models = list(
            filter(
                lambda candidate: any(
                    re.match(path_expr, candidate) for path_expr in path_exprs
                ),
                candidate_models,
            )
        )
        log.debug3("Candidate models after path filtering: %s", candidate_models)

    # Group models by module type and qualifiers (e.g. language)
    models_by_type = group_by_module_type(candidate_models)
    log.debug2("All model types: %s", models_by_type.keys())

    # For each model type, get the latest model that has a supported guid and
    # matches the versioning constraints
    all_futures = []
    for model_type_name, model_artifacts in models_by_type.items():
        fut = get_latest_valid_model(
            model_name=model_type_name,
            model_artifacts=model_artifacts,
            artifactory_username=artifactory_username,
            artifactory_api_key=artifactory_api_key,
            artifactory_repo=artifactory_repo,
            module_guids=module_guids,
            library_versions=library_versions,
        )
        all_futures.append(fut)
    model_results = await asyncio.gather(*all_futures)
    results = list(filter(None, model_results))
    log.debug3("Models from [%s]: %s", artifactory_repo, results)
    return results


def update_model_info_from_config(
    config, model_name: str, model_path="", model_url="", local_model=False
) -> Optional[ModelInfo]:
    # Get the guid for this model's module
    id_fields = [key for key in config.keys() if key.endswith("_id")]
    if len(id_fields) != 1:
        log.warning("No single module guid found for model with config [%s]", config)
        return None
    id_field = id_fields[0]
    module_guid = config[id_field]

    # Determine the parent library name
    module_flavor = id_field[: id_field.index("_id")]
    class_field = f"{module_flavor}_class"
    if class_field not in config:
        log.warning("No %s found in config [%s]", class_field, config)
        return None
    raw_parent_library = config[class_field].split(".")[0]
    parent_library = LIBRARY_PACKAGE_ALIASES.get(raw_parent_library, raw_parent_library)
    fixed_module_class = config[class_field].replace(raw_parent_library, parent_library)

    # Determine the parent library version
    parent_lib_version_key = f"{raw_parent_library}_version"
    parent_lib_version = config.get(parent_lib_version_key)
    if (
        parent_lib_version is None and not local_model
    ):  # we check version in model_path if this is the case of artifactory repo models
        log.debug("No %s found in config [%s]", parent_lib_version_key, config)

        # Look in the model name if not found in the config.yml
        version_match = VERSION_EXPR.search(model_path)
        assert (
            version_match
        ), f"Programming Error: model names with missing versions should be eliminated earlier. Model: {model_path}"
        parent_lib_version = "{}.{}.{}".format(
            version_match.group("major"),
            version_match.group("minor"),
            version_match.group("patch"),
        )
        log.debug("Using library version [%s] found in model name", parent_lib_version)
    try:
        parent_lib_version = semver.VersionInfo.parse(parent_lib_version)
    except ValueError:
        log.warning(
            "Could not parse a valid version for model %s from parent library version: %s",
            config,
            parent_lib_version,
        )
        return None

    # Parse the date this thing was created:
    if "created" in config:
        created = config.get("created")
    else:
        if local_model:
            created = datetime.now().isoformat().replace("T", " ")
        else:  # using artifactory model, so parse the date from model name
            # Need to look in the model name
            created_datetime = datetime.strptime(
                model_path.split(".")[0].rpartition("_")[-1],
                TIME_PARSE_FORMAT,
            )
            created = created_datetime.isoformat().replace("T", " ")

    return ModelInfo(
        name=model_name,
        guid=module_guid,
        parent_library=parent_library,
        parent_library_version=parent_lib_version,
        module_class=fixed_module_class,
        created=created,
        url=model_url,
    )


def get_model_info_from_local_config_yml(
    model_name: str, config_path: str
) -> Optional[ModelInfo]:
    # Parse the yaml
    config = {}
    with open(config_path) as f:
        config_text = f.read()
        config = yaml.safe_load(config_text)

    model_url = config_path.replace("/config.yml", "").replace("/config.yaml", "")

    model_info = update_model_info_from_config(
        config=config, model_name=model_name, model_url=model_url, local_model=True
    )

    # Construct the ModelInfo and return
    return model_info


def get_models_from_local_dir(model_dir_path: str) -> List[ModelInfo]:
    """Get all models from the given local directory that should be built into images"""

    # Get a list of all models from the local dir
    local_models = []

    # unzip first if they give us zip files
    for zip_file in os.listdir(model_dir_path):
        path_to_zip_file = os.path.join(model_dir_path, zip_file)
        if os.path.isfile(path_to_zip_file) and path_to_zip_file.endswith(".zip"):
            dest_path = path_to_zip_file.replace(".zip", "")
            log.debug(
                "Extracting zip file %s into folder %s", path_to_zip_file, dest_path
            )
            with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                zip_ref.extractall(dest_path)
                zip_ref.close()

    for child_dir in os.listdir(model_dir_path):
        full_model_dir = os.path.join(model_dir_path, child_dir)
        if os.path.isdir(full_model_dir) and "config.yml" in os.listdir(full_model_dir):
            model = get_model_info_from_local_config_yml(
                child_dir, os.path.join(full_model_dir, "config.yml")
            )
            local_models.append(model)

    log.debug3("Models for local dir [%s]: %s", model_dir_path, local_models)

    return local_models


def get_target_image_name(
    model_info: ModelInfo, target_registry: Optional[str], image_tag: Optional[str]
) -> str:
    """Get the image name that should be used to tag the built image"""
    lib_prefix = model_info.parent_library.replace("_", "-")
    image_name = f"{lib_prefix}_{model_info.name}"
    image_version = image_tag if image_tag else str(model_info.parent_library_version)
    full_name = f"{image_name}:{image_version}"
    if target_registry:
        full_name = "{}/{}".format(target_registry.rstrip("/"), full_name)
    return full_name


def get_image_labels(model_info: ModelInfo) -> str:
    """Get the image labels in 'label=value;label2=value2' format"""
    labels = []
    labels.append(f"{LABEL_PREFIX}.watson_library={model_info.parent_library}")
    labels.append(f"{LABEL_PREFIX}.library_version={model_info.parent_library_version}")
    labels.append(f"{LABEL_PREFIX}.module_guid={model_info.guid}")
    labels.append(f"{LABEL_PREFIX}.module_class={model_info.module_class}")
    labels.append(f"{LABEL_PREFIX}.created={model_info.created}")

    return ";".join(labels)


def parse_pip_semver(version_str: str) -> semver.VersionInfo:
    """Handle converting pip version qualifier conventions back to standard
    semver in order to allow qualified versions (e.g. release candidates) to be
    parsed.
    """
    semver_str = re.sub(r"([0-9])rc([0-9]+)", r"\1-rc.\2", version_str)
    return semver.VersionInfo.parse(semver_str)


class RetriedAsyncClient:
    """Wrapper for the httpx.AsyncClient that performs backed off retries"""

    def __init__(
        self,
        timeout_base: float,
        timeout_retries: int,
        timeout_backoff: float,
        max_connections: int,
    ):
        self._client = httpx.AsyncClient(
            timeout=timeout_base,
            limits=httpx.Limits(max_connections=max_connections),
        )
        self._timeout_retries = timeout_retries
        self._timeout_backoff = timeout_backoff

    async def get(self, *args, **kwargs) -> httpx.Response:
        """Perform a retried GET call"""
        return await self._retried_call(self._client.get, *args, **kwargs)

    async def _retried_call(
        self, call: Callable, *args, retry_number=0, **kwargs
    ) -> httpx.Response:
        """Perform the given call with retries"""
        try:
            result = await call(*args, **kwargs)
            try:
                result.raise_for_status()
            except httpx.HTTPStatusError as err:
                log.error("Got exception [%s] on call %s (%s)", err, args, kwargs)
                raise
            return result
        except httpx.TimeoutException:
            if retry_number <= self._timeout_retries:
                next_try = retry_number + 1
                next_timeout = (
                    kwargs.pop("timeout", self._client.timeout.read)
                    * self._timeout_backoff
                )
                log.debug2(
                    "Retrying. Attempt %d with timeout %f: %s (%s)",
                    next_try,
                    next_timeout,
                    args,
                    kwargs,
                )
                return await self._retried_call(
                    call,
                    *args,
                    retry_number=next_try,
                    timeout=next_timeout,
                    **kwargs,
                )
            raise


## Main ########################################################################


async def async_main(args: argparse.Namespace):
    """Async implementation of the main functionality"""

    # Set up the global client
    global CLIENT
    CLIENT = RetriedAsyncClient(
        timeout_base=args.timeout_base,
        timeout_retries=args.timeout_retries,
        timeout_backoff=args.timeout_backoff,
        max_connections=args.max_connections,
    )

    # Parse the library versions dict
    try:
        library_versions = {
            entry.split(":")[0]: parse_pip_semver(entry.split(":")[1])
            for entry in args.library_version
        }
    except (IndexError, ValueError):
        log.error("Invalid --library-version: %s", args.library_version)
        sys.exit(1)

    # Parse the local model dir, make sure it's not path to a model itself
    if args.local_model_dir:
        if not os.path.isdir(args.local_model_dir):
            log.error(
                "Invalid --local-model-dir: %s. The path is not a directory.",
                args.local_model_dir,
            )
            sys.exit(1)
        if os.path.exists(os.path.join(args.local_model_dir, "config.yml")):
            log.error(
                "Invalid --local-model-dir: %s. The path should not be a model itself.",
                args.local_model_dir,
            )
            sys.exit(1)

    log.info("Running SETUP")
    log.info("Library Versions: %s", library_versions)
    if args.artifactory_repo:
        log.info("Artifactory Repos: %s", args.artifactory_repo)
    if args.local_model_dir:
        log.info("Local Model Dir: %s", args.local_model_dir)
    log.info("Module GUIDs: %s", args.module_guid)
    if args.image_tag:
        log.info("Image tag version: %s", args.image_tag)

    all_model_infos = []

    # Gather the model infos from local model dir (if given)
    if args.local_model_dir:
        all_model_infos.extend(get_models_from_local_dir(args.local_model_dir))
        log.debug(
            "Found a total of %d model infos from local dir", len(all_model_infos)
        )
    # Gather the model infos from each repo (if given)
    if args.artifactory_repo:
        model_lookup_futures = {
            repo.rstrip("/"): get_models_from_repo(
                artifactory_repo=repo.rstrip("/"),
                artifactory_username=args.artifactory_username,
                artifactory_api_key=args.artifactory_api_key,
                module_guids=args.module_guid,
                library_versions=library_versions,
                path_exprs=args.path_expr,
            )
            for repo in args.artifactory_repo
        }
        model_lookup_results = {
            repo: await lookup_future
            for repo, lookup_future in model_lookup_futures.items()
        }
        all_model_infos.extend(
            model_info
            for model_infos in model_lookup_results.values()
            for model_info in model_infos
        )
        log.debug(
            "Found a total of %d model infos from artifactory",
            len(model_lookup_results),
        )

    # Construct the CSV file from all of the found models
    csv_cols = {
        MODEL_NAME: [],
        TARGET_IMAGE_NAME: [],
        MODEL_SOURCE: [],
        IMAGE_LABELS: [],
    }
    for model_info in all_model_infos:
        csv_cols[MODEL_NAME].append(model_info.name)
        csv_cols[MODEL_SOURCE].append(model_info.url)
        csv_cols[TARGET_IMAGE_NAME].append(
            get_target_image_name(model_info, args.target_registry, args.image_tag)
        )
        csv_cols[IMAGE_LABELS].append(get_image_labels(model_info))

    with output_handle(args.output_csv) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CONFIG_CSV_COL_HEADERS)
        for i in range(len(all_model_infos)):
            row = []
            for header in CONFIG_CSV_COL_HEADERS:
                row.append(csv_cols[header][i])
            writer.writerow(row)
            log.debug3("CSV ROW: %s", row)


def main(parent_parser: Optional[argparse.ArgumentParser] = None):
    """Run the main entrypoint for the setup command"""
    parser = get_command_parser(__doc__, parent_parser)

    # Add the args for this script
    parser.add_argument(
        "--module-guid",
        "-m",
        nargs="+",
        default=str_list_from_env("MODULE_GUID"),
        help="Module GUID(s) that will be exposed",
    )
    parser.add_argument(
        "--artifactory-username",
        "-u",
        default=os.environ.get("ARTIFACTORY_USERNAME", ""),
        help="Username for artifactory access",
    )
    parser.add_argument(
        "--artifactory-api-key",
        "-k",
        default=os.environ.get("ARTIFACTORY_API_KEY", ""),
        help="API key for artifactory access",
    )
    parser.add_argument(
        "--artifactory-repo",
        "-r",
        nargs="+",
        default=str_list_from_env("ARTIFACTORY_REPO"),
        help="Artifactory repo to get models from. At least one of --artifactory-repo or --local-model-dir has to be set.",
    )
    parser.add_argument(
        "--local-model-dir",
        "-md",
        default=str_list_from_env("LOCAL_MODEL_DIR"),
        help="Local directory that has the models to package from. At least one of --artifactory-repo or --local-model-dir has to be set.",
    )
    parser.add_argument(
        "--target-registry",
        "-t",
        default=None,
        help="The target docker registry to use for pushing model images to a remote registry",
    )
    parser.add_argument(
        "--library-version",
        "-v",
        nargs="+",
        default=str_list_from_env("LIBRARY_VERSIONS"),
        help="lib:version pair(s) for watson libraries being released",
    )
    parser.add_argument(
        "--image-tag",
        "-it",
        default=os.environ.get("IMAGE_TAG", ""),
        help="What to tag the version of the model images",
    )
    parser.add_argument(
        "--path-expr",
        "-e",
        nargs="*",
        default=None,
        help="Regular expression(s) to limit which model paths are considered",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        default=None,
        help="CSV file to output to (defaults to stdout)",
    )
    parser.add_argument(
        "--timeout-base",
        "-tb",
        type=float,
        default=number_from_env("TIMEOUT_BASE", default_val=120.0),
        help="The base timeout to use for http requests",
    )
    parser.add_argument(
        "--timeout-retries",
        "-tr",
        type=int,
        default=number_from_env("TIMEOUT_RETRIES", number_type=int, default_val=3),
        help="The number of times to retry on timeout",
    )
    parser.add_argument(
        "--timeout-backoff",
        "-tk",
        type=float,
        default=number_from_env("TIMEOUT_BACKOFF", default_val=1.5),
        help="Multiplier to use for the timeout when retrying",
    )
    parser.add_argument(
        "--max-connections",
        "-j",
        type=int,
        default=number_from_env("MAX_CONNECTIONS", number_type=int, default_val=30),
        help="Max concurrent open connections",
    )
    args = parser.parse_args()

    # Configure logging
    handle_logging_args(args)

    # Make sure if repos are given, then guids are also given
    if args.artifactory_repo and not args.module_guid:
        parser.print_usage()
        print(
            "error: if --artifactory-repo is set, then the following argument must have at least one entry: {}/{}".format(
                "--module-guid",
                "-m",
            )
        )
        sys.exit(2)
    if not args.artifactory_repo and not args.local_model_dir:
        parser.print_usage()
        print(
            "error: at least one of the following arguments must have a value: {}/{} and {}/{}".format(
                "--artifactory-repo", "-r", "--local-model-dir", "-md"
            )
        )
        sys.exit(2)
    if not args.library_version:
        parser.print_usage()
        print(
            "error: the following arguments must have at least one entry: {}/{}".format(
                "--library-version",
                "-v",
            )
        )
        sys.exit(2)

    # Run the main async entrypoint
    asyncio.run(async_main(args))


if __name__ == "__main__":  # pragma: no cover
    main()
