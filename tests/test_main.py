"""
Tests for the main module entrypoint
"""

# Third Party
import pytest

# First Party
from watson_embed_model_packager import build_model_images
from watson_embed_model_packager.__main__ import main

# Local
from tests.helpers import TEST_CONFIG, cli_args, subproc_mock_fixture_base


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
        assert 1 == 0


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
