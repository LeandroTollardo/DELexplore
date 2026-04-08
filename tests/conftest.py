"""Shared pytest fixtures for DELexplore tests."""

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths to example data (checked-in real DELT-Hit output)
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent / "data" / "examples"
SYNTHETIC_DIR = Path(__file__).parent / "data" / "synthetic"


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Path to the examples directory containing real DELT-Hit output."""
    return EXAMPLES_DIR


@pytest.fixture(scope="session")
def example_counts_path() -> Path:
    """Path to the example counts.txt file."""
    return EXAMPLES_DIR / "counts.txt"


@pytest.fixture(scope="session")
def example_config_path() -> Path:
    """Path to the example config.yaml file."""
    return EXAMPLES_DIR / "config.yaml"


@pytest.fixture(scope="session")
def example_counts_df(example_counts_path):
    """Pre-loaded counts DataFrame from the example counts.txt."""
    from delexplore.io.readers import read_counts

    return read_counts(example_counts_path)


@pytest.fixture(scope="session")
def example_config(example_config_path):
    """Pre-loaded config dict from the example config.yaml."""
    from delexplore.io.readers import read_config

    return read_config(example_config_path)
