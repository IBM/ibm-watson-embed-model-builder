"""
Tests for setup command
"""

# Standard
from contextlib import closing, contextmanager
from datetime import datetime
from fnmatch import fnmatch
from http import HTTPStatus
from typing import Dict, List, Optional, TextIO
import copy
import csv
import io
import os
import random
import shutil
import socket
import tempfile
import threading
import time

# Third Party
from flask import Flask, request
from flask_basicauth import BasicAuth
from werkzeug.serving import make_server
import httpx
import pytest
import yaml

# First Party
from watson_embed_model_packager import constants
from watson_embed_model_packager import setup_build_config as command
import alog

# Local
from tests.helpers import cli_args, env

## Constants ###################################################################

log = alog.use_channel("TEST")

# Shared test creds for the mock
ARTIFACTORY_USERNAME = "testy@somewhere.com"
ARTIFACTORY_API_KEY = "supersecret!"

# Default supported library version
SUPPORTED_LIBS = ["watson_nlp:3.0.0"]

## Helpers #####################################################################


def port_open(port: int) -> bool:
    """Check whether the given port is open"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def random_port():
    """Grab a random port number"""
    return int(random.uniform(12345, 55555))


def get_available_port():
    """Look for random ports until an open one is found"""
    port = random_port()
    while not port_open(port):
        port = random_port
    return port


class ArtifactoryMock:
    """Simple mock artifactory server that serves a collection of models"""

    def __init__(
        self,
        repo_data: dict,
        username: str,
        apikey: str,
        inject_latency: Optional[Dict[str, float]] = None,
    ):
        self.app = Flask(self.__class__.__name__)
        self.app.config["BASIC_AUTH_USERNAME"] = username
        self.app.config["BASIC_AUTH_PASSWORD"] = apikey
        self.auth = BasicAuth(self.app)
        self.repo_data = repo_data
        self.inject_latency = inject_latency or {}
        log.debug3("Full repo data:\n%s", self.repo_data)

        # Define the handler routes
        @self.app.route("/artifactory/api/search/pattern", methods=["GET"])
        @self.auth.required
        def search():
            self._inject_latency("search")
            return self._handle_search()

        @self.app.route("/artifactory/<string:repo>/<path:subpath>", methods=["GET"])
        @self.auth.required
        def storage(repo: str, subpath: str):
            self._inject_latency("storage")
            return self._handle_storage(repo, subpath)

        # Boot up the server on a random open port
        self.port = get_available_port()
        self.server = make_server("localhost", self.port, self.app, threaded=True)
        self.serve_thread = threading.Thread(target=self.server.serve_forever)
        self.serve_thread.start()

    def stop(self):
        self.server.shutdown()
        self.serve_thread.join()

    def _inject_latency(self, handler_name: str):
        latency = self.inject_latency.get(handler_name)
        if latency:
            time.sleep(latency)

    def _handle_search(self):
        """Generic handler to yield content of a search request"""
        result = {}
        pattern = request.args.to_dict().get("pattern")
        if not pattern:
            return "Missing argument 'pattern'", HTTPStatus.BAD_REQUEST

        pattern_parts = pattern.split(":")
        if len(pattern_parts) != 2:
            return (
                f"Malformed pattern [{pattern}]. Should be <repo>:<path expr>",
                HTTPStatus.BAD_REQUEST,
            )
        repo, path_expr = pattern_parts

        repo_files = self.repo_data.get(repo)
        if repo_files is None:
            return f"Repo {repo} Not Found", HTTPStatus.NOT_FOUND
        log.debug3("Search repo data: %s", repo_files)

        files = result.setdefault("files", [])
        for path_name in repo_files:
            log.debug3("Comparing path %s to pattern %s", path_name, path_expr)
            if fnmatch(path_name, path_expr):
                log.debug("Found file match: %s", path_name)
                files.append(path_name.lstrip("/"))
        return result

    def _handle_storage(self, repo: str, path: str):
        """Generic handler to yield content of a storage request"""
        # Get the repo content
        repo_content = self.repo_data.get(repo)
        if repo_content is None:
            return f"Repo {repo} Not Found", HTTPStatus.NOT_FOUND

        # Split on ! to get files
        parts = path.split("!")
        if len(parts) > 2:
            return "Too many '!' in request", HTTPStatus.BAD_REQUEST
        key = parts[0]
        if not key.startswith("/"):
            key = f"/{key}"

        # If we have the key, fetch its content
        model_obj = repo_content.get(key)
        if not model_obj:
            return f"Path {key} Not Found", HTTPStatus.NOT_FOUND

        # If a sub_file is requested, grab it
        result = model_obj
        if len(parts) == 2:
            result = model_obj.get(parts[1])
            if result is None:
                return f"File {parts[1]} not found in {key}", HTTPStatus.NOT_FOUND
        return result


@contextmanager
def artifactory_repo(
    models: dict,
    repo_name: Optional[str] = "test-repo",
    **kwargs,
) -> ArtifactoryMock:
    """Context manager to stand up a local mock server for an individual test"""
    repo_data = {repo_name: models} if repo_name is not None else models
    server = ArtifactoryMock(
        repo_data,
        ARTIFACTORY_USERNAME,
        ARTIFACTORY_API_KEY,
        **kwargs,
    )
    try:
        repo_urls = [
            f"http://localhost:{server.port}/artifactory/{rname}"
            for rname in repo_data.keys()
        ]
        yield repo_urls
    finally:
        server.stop()


def make_model_content(model_cfg: dict, **other_files) -> dict:
    """Helper to create the body of a 'model blob'"""
    content = other_files or {}
    content["config.yml"] = yaml.safe_dump(model_cfg)
    return {
        key if key.startswith("/") else f"/{key}": val for key, val in content.items()
    }


def make_model_name(
    module_type: str = "sample",
    model_label: str = "testing",
    module_flavor: str = "block",
    library_version: Optional[str] = "v1-2-3",
    qualifier: str = "",
    skip_timestamp: bool = False,
) -> str:
    """Helper to make a model name"""
    name_parts = [
        module_type,
        model_label,
    ]
    # Blocks predated including the flavor in the name
    if module_flavor != "block":
        name_parts.append(module_flavor)
    if library_version is not None:
        name_parts.append(library_version)
    if qualifier:
        name_parts.append(qualifier)
    if not skip_timestamp:
        name_parts.append(datetime.now().strftime(command.TIME_PARSE_FORMAT))
    return "_".join(name_parts) + ".zip"


def parse_csv(handle: TextIO) -> dict:
    """Parse an open file-like object as a CSV"""
    csv_rows = list(csv.reader(handle))
    header, data_rows = csv_rows[0], csv_rows[1:]
    data = []
    for row in data_rows:
        model_entry = {}
        for i, val in enumerate(row):
            model_entry[header[i]] = val
        data.append(model_entry)
    return data


def parse_csv_file(fname: str) -> List[dict]:
    """Helper to parse the created CSV file"""
    with open(fname, "r") as handle:
        return parse_csv(handle)


@contextmanager
def cli_test_harness(
    models: dict,
    *cliargs,
    include_output_csv=True,
    include_creds=True,
    include_lib_version=True,
    skip_default_repo=False,
    inject_latency=None,
    local_model=False,
    zipped_model=False,
    **envkwargs,
):
    """Helper to avoid nested with statements in tests"""
    cliargs = list(cliargs)
    if include_lib_version and not ("-v" in cliargs or "--library-version" in cliargs):
        cliargs.append("-v")
        cliargs.extend(SUPPORTED_LIBS)

    if local_model:
        with tempfile.TemporaryDirectory() as model_dir:
            if not ("-md" in cliargs or "--local-model-dir" in cliargs):
                cliargs.extend(["-md", model_dir])

            for model_name, model_files in models.items():
                current_model_dir = os.path.join(model_dir, model_name)
                os.mkdir(current_model_dir)
                for (
                    file_name,
                    file_data,
                ) in model_files.items():  # file_name is "/config.yml"
                    print(
                        "writing file name: ",
                        os.path.join(current_model_dir, file_name[1:]),
                    )
                    with open(os.path.join(current_model_dir, file_name[1:]), "w") as f:
                        f.write(file_data)
                if zipped_model:
                    shutil.make_archive(
                        base_name=os.path.join(model_dir, model_name),
                        format="zip",
                        root_dir=current_model_dir,
                    )
                    shutil.rmtree(current_model_dir)
            if include_output_csv:
                with tempfile.NamedTemporaryFile(suffix=".csv") as output_csv:
                    if not ("-o" in cliargs or "--output-csv" in cliargs):
                        cliargs.extend(["-o", output_csv.name])
                    with cli_args(*cliargs):
                        yield output_csv.name
            else:
                with cli_args(*cliargs):
                    yield None
    else:
        artifactory_repo_args = (models, None) if skip_default_repo else (models,)
        with artifactory_repo(
            *artifactory_repo_args,
            inject_latency=inject_latency,
        ) as art_repo:
            with env(**envkwargs):
                if not ("-r" in cliargs or "--artifactory-repo" in cliargs):
                    cliargs.append("-r")
                    cliargs.extend(art_repo)
                if include_creds:
                    if not ("-u" in cliargs or "--artifactory-username" in cliargs):
                        cliargs.extend(["-u", ARTIFACTORY_USERNAME])
                    if not ("-k" in cliargs or "--artifactory-apikey" in cliargs):
                        cliargs.extend(["-k", ARTIFACTORY_API_KEY])
                if include_output_csv:
                    with tempfile.NamedTemporaryFile(suffix=".csv") as output_csv:
                        if not ("-o" in cliargs or "--output-csv" in cliargs):
                            cliargs.extend(["-o", output_csv.name])
                        with cli_args(*cliargs):
                            yield output_csv.name
                else:
                    with cli_args(*cliargs):
                        yield None


## Reusable Test Content #######################################################

MODEL_NAME = make_model_name()
MODULE_GUID = "asdf1234"
REPO_DATA = {
    f"/blocks/sample/{MODEL_NAME}": make_model_content(
        {
            "block_class": "lego.blocks.sample.testing.Tester",
            "block_id": MODULE_GUID,
            "lego_version": "1.2.3",
        }
    )
}
LOCAL_DATA = {
    f"my_local_model": make_model_content(
        {
            "block_class": "lego.blocks.sample.testing.Tester",
            "block_id": MODULE_GUID,
            "lego_version": "1.2.3",
        }
    )
}

## Tests #######################################################################


def test_single_model():
    """Test that the simple case of running the command with a single model
    yields the desired result
    """
    with cli_test_harness(
        REPO_DATA,
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1


def test_local_model():
    """Test that the simple case of running the command with a single model locally
    yields the desired result
    """
    with cli_test_harness(
        LOCAL_DATA,
        "-it",
        "1.3.2",
        local_model=True,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        model_source = model_entries[0]["model_source"]
        assert "config.yml" not in model_source


def test_model_omitted_by_guid():
    """Test that a model with an unsupported guid is eliminated from
    consideration
    """
    with cli_test_harness(
        REPO_DATA,
        "-m",
        "something else",
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_output_to_stdout(capsys):
    """Make sure that by default the output goes to stdout"""
    with cli_test_harness(
        REPO_DATA,
        "-m",
        MODULE_GUID,
        include_output_csv=False,
    ):
        command.main()
        captured = capsys.readouterr()
        out_stream = io.StringIO(captured.out)
        model_entries = parse_csv(out_stream)
        assert len(model_entries) == 1


def test_missing_credentials():
    """Make sure that the credentials are passed to the requests correctly"""
    with cli_test_harness(
        REPO_DATA,
        "-m",
        MODULE_GUID,
        include_creds=False,
    ):
        with pytest.raises(httpx.HTTPStatusError):
            command.main()


def test_bad_local_model_dir_not_a_dir():
    """Make sure that the local model dir is passed as a path to a directory"""
    with cli_test_harness(
        LOCAL_DATA,
        "-it",
        "1.3.2",
        "-md",
        __file__,
        local_model=True,
    ):
        with pytest.raises(SystemExit) as exit_err:
            command.main()
        assert exit_err.value.code == 1


def test_bad_local_model_dir_not_a_model():
    """Make sure that the local model dir doesn't include a config.yml file, aka it's not a path to a model"""
    with tempfile.TemporaryDirectory() as model_dir:
        with open(os.path.join(model_dir, "config.yml"), "w") as f:
            f.write("foo")
        with cli_test_harness(
            LOCAL_DATA,
            "-it",
            "1.3.2",
            "-md",
            model_dir,
            local_model=True,
        ):
            with pytest.raises(SystemExit) as exit_err:
                command.main()
            assert exit_err.value.code == 1


def test_local_model_as_zip():
    """Test that the simple case of running the command with a single model as a zip file locally
    yields the desired result
    """
    with cli_test_harness(
        LOCAL_DATA,
        "-it",
        "1.3.2",
        local_model=True,
        zipped_model=True,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1


def test_missing_version_in_name():
    """Make sure that model names without library versions are discarded"""
    model_name = make_model_name(library_version=None)
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_metadata_are_included_in_csv_for_old_format():
    """Make sure that model metadata info are included in csv file, the old model config yaml format does not have library version"""
    model_name = make_model_name(
        module_type="syntax",
        model_label="izumo",
        module_flavor="block",
        library_version="v1-2-1",
    )
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        assert len(model_entries[0]) == 4  # we should have a labels column
        assert "watson_library=watson_nlp" in model_entries[0]["image_labels"]
        assert "library_version=1.2.1" in model_entries[0]["image_labels"]
        assert f"module_guid={MODULE_GUID}" in model_entries[0]["image_labels"]
        assert (
            "created=20" in model_entries[0]["image_labels"]
        )  # this test will fail in 2100s (good luck!)
        assert (
            "module_class=watson_nlp.blocks.sample.testing.Tester"
            in model_entries[0]["image_labels"]
        )


def test_metadata_are_included_in_csv_for_new_format():
    """Make sure that model metadata info are included in csv file, the new format has both watson_nlp_version and version field in the config yaml"""
    model_name = make_model_name(
        module_type="classification",
        model_label="ensemble-workflow",
        module_flavor="workflow",
    )
    with cli_test_harness(
        {
            f"/workflows/classification/{model_name}": make_model_content(
                {
                    "created": "2021-04-21 17:31:33.343251",
                    "doc_embed_style": "raw_text",
                    "module_paths": {
                        "cnn_model": "./cnn_model",
                        "ensemble_model": "./ensemble_model",
                        "stopwords": "./stopwords",
                        "syntax_model": "./syntax_model",
                        "tf_idf_model": "./tf_idf_model",
                        "tf_idf_svm_model": "./tf_idf_svm_model",
                        "use_model": "./use_model",
                        "use_svm_model": "./use_svm_model",
                    },
                    "name": "Classifier Workflow",
                    "train_params": {
                        "cnn_epochs": 30,
                        "random_seed": 1001,
                        "shuffle": True,
                        "tf_idf_svm_epochs": 3000,
                        "use_svm_epochs": 3000,
                    },
                    "version": "0.0.1",
                    "watson_nlp_version": "1.12.0",
                    "workflow_class": "watson_nlp.workflows.classification.ensemble.Ensemble",
                    "workflow_id": MODULE_GUID,
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        assert len(model_entries[0]) == 4  # we should have an image_labels column
        assert (
            "com.ibm.watson.embed.watson_library=watson_nlp"
            in model_entries[0]["image_labels"]
        )
        assert (
            "com.ibm.watson.embed.library_version=1.12.0"
            in model_entries[0]["image_labels"]
        )
        assert (
            f"com.ibm.watson.embed.module_guid={MODULE_GUID}"
            in model_entries[0]["image_labels"]
        )
        assert (
            "com.ibm.watson.embed.created=2021-04-21 17:31:33.343251"
            in model_entries[0]["image_labels"]
        )
        assert (
            "com.ibm.watson.embed.module_class=watson_nlp.workflows.classification.ensemble.Ensemble"
            in model_entries[0]["image_labels"]
        )


def test_image_tag_is_included_in_csv_if_passed_in():
    """Make sure that model image tag is included in csv file, coming from the passed in flag"""
    with cli_test_harness(
        REPO_DATA,
        "-m",
        MODULE_GUID,
        "-it",
        "0.0.5",
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        print(model_entries[0])
        assert "0.0.5" in model_entries[0][constants.TARGET_IMAGE_NAME]


def test_image_tag_is_included_in_csv_coming_from_library_version():
    """Make sure that model image tag is included in csv file, when there's no --image-tag flag set"""
    model_name = make_model_name(
        module_type="ensemble", model_label="classification-workflow"
    )
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        print(model_entries[0])
        assert "1.2.3" in model_entries[0][constants.TARGET_IMAGE_NAME]


def test_image_tag_is_included_in_csv_coming_from_flag_not_library_version():
    """Make sure that model image tag is included in csv file, and the image-tag flag takes precedence"""
    model_name = make_model_name(
        module_type="ensemble", model_label="classification-workflow"
    )
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
        "-it",
        "0.0.10",
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        print(model_entries[0])
        assert "1.2.3" not in model_entries[0][constants.TARGET_IMAGE_NAME]
        assert "0.0.10" in model_entries[0][constants.TARGET_IMAGE_NAME]


def test_sentiment_names_are_fixed():
    """Make sure that models with invalid task type names are fixed, specifically for sentiment-aggregated"""
    model_name = make_model_name(
        module_type="sentiment-aggregated", model_label="workflow"
    )
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        assert model_entries[0]["model_name"] == "sentiment_aggregated-workflow"


def test_classification_names_are_fixed():
    """Make sure that models with invalid task type names are fixed, specifically for classification"""
    model_name = make_model_name(
        module_type="ensemble", model_label="classification-workflow"
    )
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        assert model_entries[0]["model_name"] == "classification_ensemble-workflow"


def test_workflow_tags_are_expanded():
    """Make sure that any '-wf' bits in model names are expanded to '-workflow'
    Also tests that -wf tags in the qualifier are moved to the label"""
    model_name = make_model_name(
        module_type="widget", model_label="test", qualifier="stock-wf"
    )
    model_name_2 = make_model_name(
        module_type="fidget", model_label="test-wf", qualifier="stock"
    )
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            ),
            f"/blocks/sample/{model_name_2}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            ),
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 2
        assert model_entries[0]["model_name"] == "widget_test-workflow_stock"
        assert model_entries[1]["model_name"] == "fidget_test-workflow_stock"


def test_dummy_models_are_skipped():
    """Make sure that model names with the word 'dummy' in them are not shipped"""
    model_name = make_model_name(qualifier="dummy")
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_single_language_bert_models_are_skipped():
    """Make sure that any bert models without multi-language support are not shipped"""
    model_name = make_model_name(model_label="bert-workflow", qualifier="lang_en_stock")
    model_name_2 = make_model_name(
        model_label="bert-workflow", qualifier="lang_multi_stock"
    )
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            ),
            f"/blocks/sample/{model_name_2}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            ),
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        assert model_entries[0]["model_name"] == "sample_bert-workflow_lang_multi_stock"


def test_emotion_classifiers_are_skipped():
    """Make sure that the old-style emotion classifiers are not shipped"""
    model_name = make_model_name(model_label="emotion", qualifier="classification")
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_missing_date_in_name():
    """Make sure that model names without date stamps are discarded"""
    model_name = make_model_name(skip_timestamp=True)
    with cli_test_harness(
        {
            f"/blocks/sample/{model_name}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_missing_guid_in_config():
    """Make sure that a model without a guid in the config.yml is discarded"""
    with cli_test_harness(
        {
            f"/blocks/sample/{MODEL_NAME}": make_model_content(
                {
                    "block_class": "watson_nlp.blocks.sample.testing.Tester",
                    "watson_nlp_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_missing_parent_lib_version_in_config():
    """Make sure that a model without a parent lib version in the config.yml
    falls back to the parent lib version in the name
    """
    with cli_test_harness(
        {
            f"/blocks/sample/{MODEL_NAME}": make_model_content(
                {
                    "block_class": "watson_nlp.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1


def test_missing_module_class_in_config():
    """Make sure that a model without a <module>_class field in the config.yml
    is discarded
    """
    with cli_test_harness(
        {
            f"/blocks/sample/{MODEL_NAME}": make_model_content(
                {
                    "block_id": MODULE_GUID,
                    "watson_nlp_version": "1.2.3",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_version_not_a_semver():
    """Make sure that a bad version causes a model to be discarded"""
    with cli_test_harness(
        {
            f"/blocks/sample/{MODEL_NAME}": make_model_content(
                {
                    "block_class": "watson_nlp.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "watson_nlp_version": "not a version!",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_unreleased_model_version_skipped():
    """Make sure that model built with a too-new version of the library is
    discarded
    """
    with cli_test_harness(
        {
            f"/blocks/sample/{MODEL_NAME}": make_model_content(
                {
                    "block_class": "watson_nlp.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "watson_nlp_version": "100.0.0",
                }
            )
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_unreleased_model_version_fall_back_to_released_version():
    """Make sure that model built with a too-new version of the library is
    discarded
    """
    new_qualifier = "new-new-new"
    unreleased_model_name = make_model_name(qualifier=new_qualifier)
    with cli_test_harness(
        {
            f"/blocks/sample/{MODEL_NAME}": make_model_content(
                {
                    "block_class": "watson_nlp.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "watson_nlp_version": "1.2.3",
                }
            ),
            f"/blocks/sample/{unreleased_model_name}": make_model_content(
                {
                    "block_class": "watson_nlp.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "watson_nlp_version": "100.0.0",
                }
            ),
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        assert new_qualifier not in model_entries[0][constants.MODEL_NAME]


def test_unreleased_library():
    """Make sure that a model for a library that is not being released is
    discarded
    """
    with cli_test_harness(
        {
            f"/blocks/sample/{MODEL_NAME}": make_model_content(
                {
                    "block_class": "watson_unreleased.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "watson_unreleased_version": "1.2.3",
                }
            ),
        },
        "-m",
        MODULE_GUID,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 0


def test_path_expression():
    """Make sure that a path expression can be used to limit which models are
    considered
    """
    workflow_guid = "qwer5678"
    workflow_name = make_model_name(module_flavor="workflow")
    with cli_test_harness(
        {
            f"/blocks/sample/{MODEL_NAME}": make_model_content(
                {
                    "block_class": "lego.blocks.sample.testing.Tester",
                    "block_id": MODULE_GUID,
                    "lego_version": "1.2.3",
                }
            ),
            f"/workflows/sample/{workflow_name}": make_model_content(
                {
                    "workflow_class": "watson_nlp.workflows.sample.testing.Tester",
                    "workflow_id": workflow_guid,
                    "watson_nlp_version": "1.2.3",
                }
            ),
        },
        "-m",
        MODULE_GUID,
        workflow_guid,
        "-e",
        "workflows/.*",
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        assert workflow_name.startswith(model_entries[0][constants.MODEL_NAME])


def test_with_target_repo():
    """Make sure that a target repository is added to the output image name"""
    reg_prefix = "foo.bar.com/reg/"
    with cli_test_harness(
        REPO_DATA,
        "-m",
        MODULE_GUID,
        "-t",
        reg_prefix,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        assert model_entries[0][constants.TARGET_IMAGE_NAME].startswith(reg_prefix)
        assert "//" not in model_entries[0][constants.TARGET_IMAGE_NAME]


def test_library_rc_version_arg():
    """Make sure that a release candidate library can be parsed"""
    # Missing version field
    with cli_test_harness(
        REPO_DATA,
        "-m",
        MODULE_GUID,
        "-v",
        "watson_nlp:3.1.0rc26",
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1


def test_bad_library_version_arg():
    """Make sure that an invalid library version arg is caught"""
    # Missing version field
    with cli_test_harness(
        REPO_DATA,
        "-m",
        MODULE_GUID,
        "-v",
        "foobar_no_version",
    ) as output_csv:
        with pytest.raises(SystemExit) as exit_err:
            command.main()
        assert exit_err.value.code != 0

    # Non-semver version field
    with cli_test_harness(
        REPO_DATA,
        "-m",
        MODULE_GUID,
        "-v",
        "foobar:not_a_version",
    ) as output_csv:
        with pytest.raises(SystemExit) as exit_err:
            command.main()
        assert exit_err.value.code != 0


def test_missing_required_args():
    """Make sure that required list args are correctly checked"""
    # No --module-guids
    with env():
        with cli_args("-r", "some repo", "-v", SUPPORTED_LIBS):
            with pytest.raises(SystemExit) as exit_err:
                command.main()
            assert exit_err.value.code != 0

    # No artifactory repo
    with env():
        with cli_args("-m", MODULE_GUID, "-v", SUPPORTED_LIBS):
            with pytest.raises(SystemExit) as exit_err:
                command.main()
            assert exit_err.value.code != 0

    # No library versions
    with env():
        with cli_args("-m", MODULE_GUID, "-r", "some repo"):
            with pytest.raises(SystemExit) as exit_err:
                command.main()
            assert exit_err.value.code != 0


def test_multi_repo():
    """Make sure that the script can handle multiple repositories at once"""
    model_name1 = make_model_name("model1")
    model_name2 = make_model_name("model2", library_version="v1-2-3")
    model_name3 = make_model_name("model2", library_version="v100-0-0")
    module_guidA = "A"
    module_guidB = "B"
    with cli_test_harness(
        {
            "repo-one": {
                f"/blocks/sample/{model_name1}": make_model_content(
                    {
                        "block_class": "lego.blocks.sample.testing.Tester",
                        "block_id": module_guidA,
                        "lego_version": "1.2.3",
                    }
                )
            },
            "repo-two": {
                f"/blocks/sample/{model_name2}": make_model_content(
                    {
                        "block_class": "lego.blocks.sample.testing.Tester",
                        "block_id": module_guidB,
                        "lego_version": "2.3.4",
                    }
                ),
                f"/blocks/sample/{model_name3}": make_model_content(
                    {
                        "block_class": "watson_nlp.blocks.sample.testing.Tester",
                        "block_id": module_guidB,
                        "watson_nlp_version": "100.0.0",
                    }
                ),
            },
        },
        "-m",
        module_guidA,
        module_guidB,
        skip_default_repo=True,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 2
        model_source_fnames = [
            entry[constants.MODEL_SOURCE].rpartition("/")[-1] for entry in model_entries
        ]
        assert model_name1 in model_source_fnames
        assert model_name2 in model_source_fnames
        assert model_name3 not in model_source_fnames


def test_module_guids_from_env():
    """Test that list arguments (such as --module-guid) can be taken from the
    environment correctly
    """
    with cli_test_harness(
        REPO_DATA,
        MODULE_GUID=f"{MODULE_GUID},other-module-guid",
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1


def test_missing_config_yml_in_model():
    """Make sure that a model without a config.yml in the right place is skipped
    without triggering a fatal error
    """
    repo_data = copy.deepcopy(REPO_DATA)
    model_key = list(repo_data.keys())[0]
    model_data = {f"/nested_dir{key}": val for key, val in repo_data[model_key].items()}
    other_model_name = make_model_name("other")
    other_module_guid = "qwer5678"
    repo_data[f"/blocks/other/{other_model_name}"] = make_model_content(
        {
            "block_class": "watson_nlp.blocks.other.testing.Tester",
            "block_id": other_module_guid,
            "watson_nlp_version": "2.3.4",
        }
    )
    repo_data[model_key] = model_data
    with cli_test_harness(
        repo_data,
        "-m",
        MODULE_GUID,
        other_module_guid,
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1
        model_source_fnames = [
            entry[constants.MODEL_SOURCE].rpartition("/")[-1] for entry in model_entries
        ]
        assert other_model_name in model_source_fnames
        assert MODEL_NAME not in model_source_fnames


def test_search_retries():
    """Test that with injected latency, a search call retries when the first
    first pass fails,
    """
    with cli_test_harness(
        REPO_DATA,
        "-tb",
        "0.1",
        "-tr",
        "1",
        "-tk",
        "10",
        "-m",
        MODULE_GUID,
        inject_latency={"search": 0.2},
    ) as output_csv:
        command.main()
        model_entries = parse_csv_file(output_csv)
        assert len(model_entries) == 1


def test_search_timeout():
    """Test that a timeout on the search call raises the correct exception after
    retrying
    """
    with cli_test_harness(
        REPO_DATA,
        "-tb",
        "0.1",
        "-tr",
        "1",
        "-tk",
        "1.01",
        "-m",
        MODULE_GUID,
        inject_latency={"search": 0.5},
    ):
        with pytest.raises(httpx.TimeoutException):
            command.main()
