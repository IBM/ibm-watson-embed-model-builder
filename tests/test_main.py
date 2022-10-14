"""
Tests for the main module entrypoint
"""

# Standard
import os
import shutil
import tempfile

# Third Party
import pytest

# First Party
from watson_embed_model_packager import build_model_images
from watson_embed_model_packager.__main__ import main

# Local
from tests.helpers import (
    TEST_CONFIG,
    TEST_CONFIG_ARTIFACTORY_ONLY,
    TEST_MODELS_DIR,
    cli_args,
    subproc_mock_fixture_base,
)


@pytest.fixture
def subproc_mock():
    """Fixture to mock out the subprocess module"""
    with subproc_mock_fixture_base([build_model_images]) as fake_subproc:
        yield fake_subproc


def test_main_build(subproc_mock):
    """Make sure that main can run with the build command"""
    with cli_args(
        "build",
        "--config",
        TEST_CONFIG,
        "--artifactory-username",
        "foobar@us.ibm.com",
        "--artifactory-api-key",
        "asdf1234",
    ):
        main()


def test_main_real_artifactory_build_with_bad_creds():
    with cli_args(
        "build",
        "--config",
        TEST_CONFIG_ARTIFACTORY_ONLY,
        "--artifactory-username",
        "foobar@us.ibm.com",
        "--artifactory-api-key",
        "asdf1234",
        "--strict",
    ):
        with pytest.raises(RuntimeError):
            main()


def test_main_real_artifactory_build():
    with cli_args(
        "build",
        "--config",
        TEST_CONFIG,
        "--artifactory-username",
        "foobar@us.ibm.com",
        "--artifactory-api-key",
        "asdf1234",
    ):
        main()


@pytest.mark.xfail
def test_setup_and_build_with_local_data_outside_working_path():
    """This test creates a manifest with local models in a tempdir outside the working path,
    and ensures the build phase can correctly handle those local models."""
    with tempfile.TemporaryDirectory() as tempdir:
        # Copy over our test models into this tmpdir
        models_dir = os.path.join(tempdir, "models")
        # os.mkdir(models_dir)
        shutil.copytree(TEST_MODELS_DIR, models_dir)
        manifest_path = os.path.join(tempdir, "manifest.csv")

        # Build the manifest:
        with cli_args(
            "setup",
            "--output-csv",
            manifest_path,
            "--library-version",
            "watson_nlp:3.2.0",
            "--local-model-dir",
            models_dir,
        ):
            main()

        # Show the manifest if the test fails
        os.system(f"cat {manifest_path}")

        # Build the image(s)
        with cli_args(
            "build",
            "--config",
            manifest_path,
        ):
            main()


def test_setup_and_build_with_local_data_in_relative_path():
    """This tests the simple case of creating a manifest with our local test models, and building
    model images out of them"""
    with tempfile.TemporaryDirectory() as tempdir:
        # Put the manifest in this new temp dir, but build the models from the local test data as is
        manifest_path = os.path.join(tempdir, "manifest.csv")

        # Build the manifest:
        with cli_args(
            "setup",
            "--output-csv",
            manifest_path,
            "--library-version",
            "watson_nlp:3.2.0",
            "--local-model-dir",
            TEST_MODELS_DIR,
        ):
            main()

        # Show the manifest if the test fails
        os.system(f"cat {manifest_path}")

        # Build the image(s)
        with cli_args(
            "build",
            "--config",
            manifest_path,
        ):
            main()


def test_main_bad_command(subproc_mock):
    """Make sure that a ValueError is raised if a bad command is given"""
    with cli_args("foobar"):
        with pytest.raises(ValueError):
            main()


def test_main_with_help_no_command(capsys):
    """Make sure that help is displayed if --help is given without a command and
    that the help content does not include the command-specific arguments
    """
    with cli_args("--help"):
        with pytest.raises(SystemExit) as exit_err:
            main()
        assert exit_err.value.code == 0
        captured = capsys.readouterr()
        assert "--log-level" in captured.out
        assert "artifactory" not in captured.out


def test_main_with_help_with_command(capsys):
    """Make sure that help is displayed if --help is given without a command"""
    with cli_args("build", "--help"):
        with pytest.raises(SystemExit) as exit_err:
            main()
        assert exit_err.value.code == 0
        captured = capsys.readouterr()
        assert "--log-level" in captured.out
        assert "artifactory" in captured.out
