"""
Tests for common arg parsing
"""

# Third Party
import pytest

# First Party
from watson_embed_model_packager.common_args import bool_from_env, number_from_env

# Local
from tests.helpers import env


def test_bool_from_env():
    """Make sure that all permutations of bool_from_env work"""
    with env():
        assert bool_from_env("SOME_ENV") is False
    with env(SOME_ENV="asdfasdfasdf"):
        assert bool_from_env("SOME_ENV") is False
    with env(SOME_ENV="True"):
        assert bool_from_env("SOME_ENV") is True


def test_number_from_env():
    """Make sure that all permutations of number_from_env work"""
    with env():
        assert number_from_env("SOME_ENV") is None
    with env(SOME_ENV="asdfasdfasdf"):
        with pytest.raises(ValueError):
            number_from_env("SOME_ENV")
    with env(SOME_ENV="1"):
        val = number_from_env("SOME_ENV")
        assert val == 1.0
        assert isinstance(val, float)
    with env(SOME_ENV="1.23"):
        assert number_from_env("SOME_ENV") == 1.23
    with env():
        assert number_from_env("SOME_ENV", default_val=3.14) == 3.14
    with env(SOME_ENV="42"):
        val = number_from_env("SOME_ENV", number_type=int)
        assert val == 42
        assert isinstance(val, int)
