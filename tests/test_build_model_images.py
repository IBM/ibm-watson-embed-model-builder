"""
Unit tests for the build_model_images command
"""

# Standard
from contextlib import contextmanager
from typing import List, Union
import csv
import os
import tempfile

# Third Party
import pytest

# Local
from watson_embed_model_packager import build_model_images as command
from watson_embed_model_packager.constants import CONFIG_CSV_COL_HEADERS
from tests.helpers import (
    TEST_CONFIG,
    TEST_LABELS,
    TEST_MODEL,
    cli_args,
    env,
    subproc_mock_fixture_base,
)

## Helpers #####################################################################


@pytest.fixture
def subproc_mock():
    """Fixture to mock out the subprocess module"""
    with subproc_mock_fixture_base([command]) as fake_subproc:
        yield fake_subproc


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global variables for each test"""
    command.DOCKER_BUILD_COMMAND = None


@contextmanager
def temp_config_file(content: Union[List[List[str]], str]):
    """Localized config file scoped to a test. The content can be either a list
    of lists (rows to use to create a csv) or a plain string to dump to the
    file.
    """
    if isinstance(content, str):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w") as csv_file:
            csv_file.write(content)
            csv_file.flush()
            yield csv_file.name
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", newline=""
        ) as csv_file:
            writer = csv.writer(csv_file, lineterminator="\n")
            for row in content:
                writer.writerow(row)
            csv_file.flush()
            yield csv_file.name


# This is the real help string generated with docker buildx build --help on my
# M1 Mac (with an arm chip)
DOCKER_DESKTOP_M1_HELP = """
Usage:  docker buildx build [OPTIONS] PATH | URL | -

Start a build

Aliases:
  build, b

Options:
      --add-host strings              Add a custom host-to-IP mapping (format: "host:ip")
      --allow strings                 Allow extra privileged entitlement (e.g., "network.host", "security.insecure")
      --build-arg stringArray         Set build-time variables
      --build-context stringArray     Additional build contexts (e.g., name=path)
      --builder string                Override the configured builder instance
      --cache-from stringArray        External cache sources (e.g., "user/app:cache", "type=local,src=path/to/dir")
      --cache-to stringArray          Cache export destinations (e.g., "user/app:cache", "type=local,dest=path/to/dir")
      --cgroup-parent string          Optional parent cgroup for the container
  -f, --file string                   Name of the Dockerfile (default: "PATH/Dockerfile")
      --iidfile string                Write the image ID to the file
      --label stringArray             Set metadata for an image
      --load                          Shorthand for "--output=type=docker"
      --metadata-file string          Write build result metadata to the file
      --network string                Set the networking mode for the "RUN" instructions during build (default "default")
      --no-cache                      Do not use cache when building the image
      --no-cache-filter stringArray   Do not cache specified stages
  -o, --output stringArray            Output destination (format: "type=local,dest=path")
      --platform stringArray          Set target platform for build
      --progress string               Set type of progress output ("auto", "plain", "tty"). Use plain to show container output
                                      (default "auto")
      --pull                          Always attempt to pull all referenced images
      --push                          Shorthand for "--output=type=registry"
  -q, --quiet                         Suppress the build output and print image ID on success
      --secret stringArray            Secret to expose to the build (format: "id=mysecret[,src=/local/secret]")
      --shm-size bytes                Size of "/dev/shm"
      --ssh stringArray               SSH agent socket or keys to expose to the build (format:
                                      "default|<id>[=<socket>|<key>[,<key>]]")
  -t, --tag stringArray               Name and optionally a tag (format: "name:tag")
      --target string                 Set the target build stage to build
      --ulimit ulimit                 Ulimit options (default [])
"""

## Tests #######################################################################


def test_gen_multi_image(subproc_mock):
    """Test that multiple images can be generated using the sample config"""
    with cli_args(
        "--config",
        TEST_CONFIG,
        "--artifactory-username",
        "foobar@us.ibm.com",
        "--artifactory-api-key",
        "asdf1234",
    ):
        command.main()
        # 4 subprocess calls: 1 to check the docker capabilities and 3 to build
        # the individual images
        assert subproc_mock.proc_count == 4


def test_missing_header_row(subproc_mock):
    """Make sure that a ValueError is raised if the csv is missing the header
    row
    """
    with temp_config_file(
        [
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", "somewhere"],
        ]
    ) as config_file:
        with cli_args("--config", config_file):
            with pytest.raises(ValueError):
                command.main()
            assert subproc_mock.proc_count == 0


def test_bad_header_row(subproc_mock):
    """Make sure that a ValueError is raised if the names of the headers are
    wrong
    """
    with temp_config_file(
        [
            ["foo", "bar", "baz"],
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", "somewhere"],
        ]
    ) as config_file:
        with cli_args("--config", config_file):
            with pytest.raises(ValueError):
                command.main()
            assert subproc_mock.proc_count == 0


def test_no_data_rows(subproc_mock):
    """Make sure that a ValueError is raised if there are no data rows"""
    with temp_config_file([CONFIG_CSV_COL_HEADERS]) as config_file:
        with cli_args("--config", config_file):
            with pytest.raises(ValueError):
                command.main()
            assert subproc_mock.proc_count == 0


def test_missing_artifactory_creds_artifactory_build(subproc_mock):
    """Make sure that a ValueError is raised if there are no artifactory creds
    given and the config targets artifactory
    """
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            [
                "dummy-model",
                "us.icr.io/cp/dummy/dummy:latest",
                "https://some.artifactory/model",
                TEST_LABELS,
            ],
        ]
    ) as config_file:
        with cli_args("--config", config_file):
            with env():
                with pytest.raises(ValueError):
                    command.main()
            assert subproc_mock.proc_count == 0


def test_missing_artifactory_creds_local_build(subproc_mock):
    """Make sure that a ValueError is NOT raised if there are no data rows and the
    config targets artifactory
    """
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file):
            with env():
                command.main()
                assert subproc_mock.proc_count == 2  # docker check + build


def test_push(subproc_mock):
    """Make sure that enabling --push will run a docker push"""
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file, "--push"):
            command.main()
            assert subproc_mock.proc_count == 3  # docker check + build + push


def test_docker_build_fail_strict(subproc_mock):
    """Make sure that with --strict, any failure in a subproces results in an
    error
    """
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file, "--strict"):
            subproc_mock.returncodes = 1
            with pytest.raises(RuntimeError):
                command.main()


def test_docker_build_fail_strict(subproc_mock):
    """Make sure that with --strict, any failure in a subproces results in an
    error
    """
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file, "--strict"):
            subproc_mock.returncodes = 1
            with pytest.raises(RuntimeError):
                command.main()


def test_docker_build_fail_non_strict(subproc_mock):
    """Make sure that without --strict, any failure in a subproces results in a
    non-fatal warning only
    """
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file):
            subproc_mock.returncodes = 1
            command.main()
            assert subproc_mock.proc_count == 2  # docker check + build


def test_docker_no_push_build_fail(subproc_mock):
    """Make sure that without strict, push is not called if the build fails"""
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file, "--push"):
            subproc_mock.returncodes = 1
            command.main()
            assert subproc_mock.proc_count == 2  # docker check + build


def test_docker_build_fail_non_strict_multi(subproc_mock):
    """Make sure that without --strict, any failure in docker build results in a
    non-fatal warning only and non-errored builds proceed as expected
    """
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file):
            subproc_mock.returncodes = [0, 1, 0]
            command.main()
            assert subproc_mock.proc_count == 3  # docker check + 2x build


def test_docker_push_fail_non_strict_multi(subproc_mock):
    """Make sure that without --strict, any failure in docker push results in a
    non-fatal warning only and non-errored builds proceed as expected
    """
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file, "--push"):
            subproc_mock.returncodes = [0, 0, 1, 0, 0]
            command.main()
            assert subproc_mock.proc_count == 5  # docker check + 2x build + 2x push


def test_docker_push_fail_strict_multi(subproc_mock):
    """Make sure that with --strict, any failure in docker push results in a
    fatal error
    """
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
            ["dummy-model", "us.icr.io/cp/dummy/dummy:latest", TEST_MODEL, TEST_LABELS],
        ]
    ) as config_file:
        with cli_args("--config", config_file, "--push", "--strict"):
            subproc_mock.returncodes = [0, 0, 1, 0, 0]
            with pytest.raises(RuntimeError):
                command.main()
            assert subproc_mock.proc_count == 3  # docker check + build + push


def test_single_file_model(subproc_mock):
    """Make sure that a local model which is a single file can be built"""
    with temp_config_file(
        [
            CONFIG_CSV_COL_HEADERS,
            [
                "dummy-model",
                "us.icr.io/cp/dummy/dummy:latest",
                os.path.join(TEST_MODEL, "config.yml"),
                TEST_LABELS,
            ],
        ]
    ) as config_file:
        with cli_args("--config", config_file):
            command.main()


def test_docker_build_with_platform():
    """Make sure that if --platform is found in the docker build help, we build
    with the right target --platform argument
    """
    with subproc_mock_fixture_base(
        [command],
        stdout=DOCKER_DESKTOP_M1_HELP,
    ) as subproc_mock:
        with cli_args(
            "--config",
            TEST_CONFIG,
            "--artifactory-username",
            "foobar@us.ibm.com",
            "--artifactory-api-key",
            "asdf1234",
        ):
            command.main()
            # 4 subprocess calls: 1 to check the docker capabilities and 3 to
            # build the individual images
            assert subproc_mock.proc_count == 4

            # Make sure that each of the real docker build commands have
            # --platform set
            assert all("--platform" in cmd for cmd in subproc_mock.commands[1:])


def test_docker_build_model_images_with_metadata(subproc_mock):
    with cli_args(
        "--config",
        TEST_CONFIG,
        "--artifactory-username",
        "foobar@us.ibm.com",
        "--artifactory-api-key",
        "asdf1234",
    ):
        command.main()
        # 4 subprocess calls: 1 to check the docker capabilities and 3 to build
        # the individual images
        assert subproc_mock.proc_count == 4
        assert all(
            "--label" in cmd
            and "com.ibm.watson.embed.created=2020-06-04 19:00:00.000000" in cmd
            for cmd in subproc_mock.commands[1:]
        )
        assert (
            "com.ibm.watson.embed.module_class=watson_doc_understanding.workflows"
            in subproc_mock.commands[3]
        )
