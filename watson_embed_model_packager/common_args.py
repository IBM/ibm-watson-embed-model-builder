"""
Common argparse arguments for all scripts and corresponding handlers
"""

# Standard
from typing import List, Optional, Union
import argparse
import os

# First Party
import alog


def bool_from_env(env_var_name: str, default_val=False) -> bool:
    """Parse a boolean from the environment"""
    return os.environ.get(env_var_name, str(default_val)).lower() == "true"


def number_from_env(
    env_var_name: str,
    number_type=float,
    default_val=None,
) -> Optional[Union[int, float]]:
    """Parse a number from the environment"""
    env_val = os.environ.get(env_var_name)
    if env_val is None:
        return default_val
    return number_type(env_val)


def str_list_from_env(env_var_name: str) -> List[str]:
    """Parse a list from the value of an environment variable"""
    env_val = os.environ.get(env_var_name)
    if not env_val:
        return []
    return [val.strip() for val in env_val.split(",") if val.strip()]


def add_logging_args(parser: argparse.ArgumentParser):
    """Add common argument parsing to the given parser in a parser group"""
    log_args = parser.add_argument_group(
        "logging", description="Global logging configuration"
    )
    log_args.add_argument(
        "--log-level",
        "-l",
        default=os.environ.get("LOG_LEVEL", "info"),
        help="Default level for all logging channels",
    )
    log_args.add_argument(
        "--log-filters",
        "--lf",
        default=os.environ.get("LOG_FILTERS", ""),
        help="Per-channel log channel level filters",
    )
    log_args.add_argument(
        "--log-json",
        "-lj",
        action="store_true",
        default=bool_from_env("LOG_JSON"),
        help="Use the json formatter for log output",
    )
    log_args.add_argument(
        "--log-thread-id",
        "-lt",
        action="store_true",
        default=bool_from_env("LOG_THREAD_ID"),
        help="Include the thread ID with log messages",
    )


def handle_logging_args(args: argparse.Namespace):
    """Perform logging configuration based on parsed args"""
    alog.configure(
        default_level=args.log_level,
        filters=args.log_filters,
        formatter="json" if args.log_json else "pretty",
        thread_id=args.log_thread_id,
    )


def get_command_parser(
    description: str,
    parent_parser: Optional[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Get an ArgumentParser to use within an individual command script"""
    # Initialize the parser with common args
    if parent_parser is None:
        parser = argparse.ArgumentParser(description=description)
        add_logging_args(parser)
    else:
        parser = argparse.ArgumentParser(parents=[parent_parser])
    return parser
