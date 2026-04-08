"""Writers for DELexplore analysis outputs."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a Polars DataFrame to a Parquet file.

    Args:
        df: DataFrame to write.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Wrote %d rows to %s", len(df), path)


def write_csv(df: pl.DataFrame, path: Path) -> None:
    """Write a Polars DataFrame to a CSV file.

    Args:
        df: DataFrame to write.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    logger.info("Wrote %d rows to %s", len(df), path)
