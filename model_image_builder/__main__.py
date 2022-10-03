"""
Tools for building watson_core model images
"""

# Standard
import argparse
import sys

# Local
from .build_model_images import main as build_command
from .common_args import add_logging_args
from .setup_build_config import main as setup_command

CMD_MAP = {
    "build": build_command,
    "setup": setup_command,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=None)
    add_logging_args(parser)
    parser.add_argument(
        "command",
        default=None,
        help=f"The sub-command to run. Options: {list(CMD_MAP.keys())}",
    )

    # Emulate the --help argument so it shows up in help!
    parser.add_argument(
        "--help",
        "-h",
        action="store_true",
        default=False,
        help="show this help message and exit",
    )

    # Before parsing the args with argparse, do a minimal parse to look for a
    # command and help request, and if found, run the help for only the parent
    # parser.
    if ("--help" in sys.argv or "-h" in sys.argv) and not [
        arg for arg in sys.argv[1:] if not arg.startswith("-")
    ]:
        parser.print_help()
        sys.exit(0)

    # Strip out the help arg so that help for the parent doesn't mask help from
    # the child command
    parser._actions = [action for action in parser._actions if action.dest != "help"]

    args = parser.parse_known_args()[0]
    command = CMD_MAP.get(args.command)
    if not command:
        raise ValueError(f"Invalid command: {args.command}")
    command(parser)


if __name__ == "__main__":  # pragma: no cover
    main()
