"""Multi-level synthon aggregation engine.

Aggregates compound counts from the full trisynthon (or N-synthon) level
down to every mono- and disynthon sub-level, producing independent DataFrames
for each level that feed into enrichment scoring.

Why all levels matter
---------------------
For a 3-cycle library with 1000 BBs per cycle:

  monosynthon diversity  = 1 000                → high statistical power
  disynthon diversity    = 1 000 000            → good power for SAR
  trisynthon diversity   = 1 000 000 000        → Poisson-dominated at typical depths

Enrichment at every level is computed independently with its own diversity
parameter so that z_n values are directly comparable across levels (Faver 2019).

Column convention (matching DELT-Hit output)
--------------------------------------------
Input DataFrame columns: selection, code_1, code_2, ..., code_N, count, id
The number of code_* columns equals n_cycles (auto-detected by io/readers.py).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence

import polars as pl

logger = logging.getLogger(__name__)

# Maps number of columns in a synthon level to its name prefix.
_LEVEL_PREFIX: dict[int, str] = {1: "mono", 2: "di", 3: "tri"}


def _prefix(n: int) -> str:
    return _LEVEL_PREFIX.get(n, f"{n}syn")


def get_all_levels(n_cycles: int) -> list[tuple[str, ...]]:
    """Return every synthon level as a tuple of column names.

    Levels are all k-combinations of code columns for k = 1 … n_cycles,
    in ascending order of k (mono first, full compound last).

    Examples::

        get_all_levels(2)
        # [("code_1",), ("code_2",), ("code_1", "code_2")]

        get_all_levels(3)
        # [("code_1",), ("code_2",), ("code_3",),
        #  ("code_1", "code_2"), ("code_1", "code_3"), ("code_2", "code_3"),
        #  ("code_1", "code_2", "code_3")]

    Args:
        n_cycles: Number of library cycles (= number of code_* columns).

    Returns:
        List of column-name tuples, one per synthon level.
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

    code_cols = [f"code_{i}" for i in range(1, n_cycles + 1)]
    levels: list[tuple[str, ...]] = []
    for k in range(1, n_cycles + 1):
        for combo in itertools.combinations(code_cols, k):
            levels.append(combo)
    return levels


def get_level_name(columns: tuple[str, ...] | Sequence[str]) -> str:
    """Return a human-readable name for a synthon level.

    Examples::

        get_level_name(("code_1",))                   -> "mono_code_1"
        get_level_name(("code_1", "code_2"))          -> "di_code_1_code_2"
        get_level_name(("code_1", "code_2", "code_3")) -> "tri_code_1_code_2_code_3"

    Args:
        columns: Tuple of code column names that define this level.

    Returns:
        Level name string.
    """
    cols = tuple(columns)
    prefix = _prefix(len(cols))
    return f"{prefix}_{'_'.join(cols)}"


def get_diversity(counts_df: pl.DataFrame, level_cols: tuple[str, ...]) -> int:
    """Count distinct observed combinations at a given synthon level.

    Diversity is computed across all selections combined (i.e., the union
    of all code combinations seen in any selection), which gives the best
    estimate of library coverage at that level.

    Args:
        counts_df: Combined counts DataFrame with a ``selection`` column and
            one or more ``code_*`` columns.
        level_cols: Code column names that define this synthon level.

    Returns:
        Number of unique code combinations observed at this level.
    """
    return counts_df.select(list(level_cols)).unique().height


def aggregate_to_level(
    counts_df: pl.DataFrame,
    level_cols: tuple[str, ...],
) -> pl.DataFrame:
    """Aggregate compound counts to a given synthon level.

    Groups by ``(selection, *level_cols)`` and sums ``count``.  The ``id``
    column is dropped because it only makes sense at the full-compound level.

    Args:
        counts_df: Combined counts DataFrame.  Must contain a ``selection``
            column, the columns named in *level_cols*, and a ``count`` column.
        level_cols: Code column names that define this synthon level.

    Returns:
        DataFrame with columns: ``selection``, *level_cols…*, ``count``.
        Sorted by ``selection`` then ``count`` descending.
    """
    group_cols = ["selection", *level_cols]
    agg = (
        counts_df.group_by(group_cols)
        .agg(pl.col("count").sum())
        .sort(["selection", "count"], descending=[False, True])
    )
    # Return columns in a stable order: selection, level_cols..., count
    return agg.select(group_cols + ["count"])


def aggregate_all_levels(
    counts_df: pl.DataFrame,
    n_cycles: int,
) -> dict[str, pl.DataFrame]:
    """Aggregate to every synthon level and return results keyed by level name.

    Args:
        counts_df: Combined counts DataFrame (as returned by
            ``io.readers.load_experiment``).
        n_cycles: Number of library cycles.

    Returns:
        Dict mapping level name → aggregated DataFrame.
        Keys are strings like ``"mono_code_1"``, ``"di_code_1_code_2"``, etc.
    """
    result: dict[str, pl.DataFrame] = {}
    for level_cols in get_all_levels(n_cycles):
        name = get_level_name(level_cols)
        agg = aggregate_to_level(counts_df, level_cols)
        diversity = get_diversity(counts_df, level_cols)
        logger.info(
            "Aggregated %s: %d rows, diversity=%d",
            name,
            len(agg),
            diversity,
        )
        result[name] = agg
    return result
