"""
Shared test helpers
"""

# Standard
from contextlib import contextmanager
from types import ModuleType
from typing import List, Optional, Tuple, Union
import copy
import os
import subprocess
import sys

TEST_DATA_DIR = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        "data",
    )
)
TEST_CONFIG = os.path.join(TEST_DATA_DIR, "model_config.csv")
TEST_CONFIG_ARTIFACTORY_ONLY = os.path.join(TEST_DATA_DIR, "artifactory_model_config.csv")
TEST_MODEL = os.path.join(TEST_DATA_DIR, "sample_models", "doc_conversion")
TEST_MODELS_DIR = os.path.join(TEST_DATA_DIR, "sample_models")
TEST_LABELS = "com.ibm.watson.embed.library_version=1.2.3;com.ibm.watson.embed.watson_library=watson_nlp;com.ibm.watson.embed.created=2020-06-04 19:00:00.000000;com.ibm.watson.embed.module_class=watson_nlp.workflows.keywords.text_rank.Text_Rank;com.ibm.watson.embed.module_guid=asdf1234"


class FakeProcess:
    """A fake process object that has the same basic API as subprocess.Process"""

    def __init__(
        self,
        returncode: int,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._completed = False

    @property
    def returncode(self) -> Optional[int]:
        """Return the configured code IFF the process is complete"""
        if self._completed:
            return self._returncode

    def communicate(self) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Emulate blocking on completion"""
        self._completed = True
        out = self._stdout.encode("utf-8") if self._stdout is not None else None
        err = self._stderr.encode("utf-8") if self._stderr is not None else None
        return (out, err)


class FakeSubprocess:
    """Patch replacement for the subprocess module that can be configured to
    emulate success or failure
    """

    PIPE = subprocess.PIPE

    def __init__(
        self,
        returncodes: Union[int, List[int]] = 0,
        stdout: str = "",
        stderr: str = "",
    ):
        """Initialize with a returncode that will be used for processes"""
        # NOTE: These are intentionally exposed as settable attributes so that
        #   they can be updated after initialization
        self.returncodes = returncodes
        self.stdout = stdout
        self.stderr = stderr
        self._commands = []

    @property
    def proc_count(self) -> int:
        return len(self._commands)

    @property
    def commands(self) -> List[List[str]]:
        return self._commands

    def Popen(
        self,
        args: List[str],
        cwd: Optional[str] = None,  # NOTE: Unused, but here to emulate subprocess
        stdout: Optional[int] = None,
        stderr: Optional[int] = None,
    ):
        """Emulate the suprocess.Popen command"""
        if isinstance(self.returncodes, int):
            returncode = self.returncodes
        else:
            assert self.proc_count < len(
                self.returncodes
            ), "Not enough return codes given!"
            returncode = self.returncodes[self.proc_count]
        self._commands.append(args)
        return FakeProcess(
            returncode=returncode,
            stdout=self.stdout if stdout == subprocess.PIPE else None,
            stderr=self.stderr if stderr == subprocess.PIPE else None,
        )


@contextmanager
def subproc_mock_fixture_base(commands: List[ModuleType], *args, **kwargs):
    """Fixture to mock out the subprocess module"""
    fake_subproc = FakeSubprocess(*args, **kwargs)
    for command in commands:
        setattr(command, "subprocess", fake_subproc)
    yield fake_subproc
    for command in commands:
        setattr(command, "subprocess", subprocess)


@contextmanager
def cli_args(*args):
    """Mock out the sys.argv set so that argparse gets the desired values"""
    real_args = copy.deepcopy(sys.argv)
    sys.argv = sys.argv[:1] + list(args)
    yield
    sys.argv = real_args


@contextmanager
def env(keep_logging=True, **kwargs):
    """Overwrite the real os.environ entries to match the given kwargs"""
    real_env = {key: val for key, val in os.environ.items()}
    os.environ.clear()
    os.environ.update(**kwargs)
    if keep_logging:
        os.environ.update(
            **{key: val for key, val in real_env.items() if key.startswith("LOG_")}
        )
    yield
    os.environ.clear()
    os.environ.update(**real_env)
