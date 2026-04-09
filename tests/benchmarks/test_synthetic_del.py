"""Synthetic DEL benchmark tests with known ground-truth planted binders.

Generates synthetic DEL experiments with KNOWN enriched compounds and
verifies the full analysis pipeline recovers them.  Each test exercises
the full stack:

    data generation → multilevel enrichment → consensus ranking → hit recovery

Design notes
------------
* ``generate_synthetic_counts`` creates a long-format DataFrame matching
  ``load_experiment`` output (columns: ``selection``, ``code_1``, ``code_2``,
  ``count``, ``id``).  Zero-count rows are included so every compound appears
  in the enrichment tables.

* True binders receive ``enrichment_fold × base_count`` expected reads in
  target (post-selection) replicates; control replicates are uniform Poisson.

* Monosynthon support scoring (``compute_support_score``) uses
  ``fold_enrichment >= 2.0`` as the enrichment threshold.  For that threshold
  to trigger at the monosynthon level, the library must be small enough that
  one binder compound noticeably raises the whole monosynthon's fold.
  Specifically: ``n_bbs_per_cycle ≤ enrichment_fold − 1 ≈ 46`` for
  enrichment_fold=50.  Tests checking support_score therefore use
  ``n_bbs_per_cycle=30``; the four recovery-quality tests use the default 200.

Run with::

    pytest tests/benchmarks/ -v -m benchmark
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

from delexplore.analyse.multilevel import run_multilevel_enrichment
from delexplore.analyse.rank import compute_composite_rank

# ---------------------------------------------------------------------------
# Constants shared by pipeline helper
# ---------------------------------------------------------------------------

_CODE_COLS = ["code_1", "code_2"]
_POST_SELS = ["target_1", "target_2", "target_3"]
_CTRL_SELS = ["control_1", "control_2", "control_3"]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_counts(
    n_bbs_per_cycle: int = 200,
    n_cycles: int = 2,
    n_replicates: int = 3,
    total_reads_per_replicate: int = 500_000,
    n_true_binders: int = 10,
    enrichment_fold: float = 50.0,
    seed: int = 42,
) -> tuple[pl.DataFrame, list[tuple[int, ...]]]:
    """Generate synthetic DEL counts with planted true binders.

    Args:
        n_bbs_per_cycle: Number of building blocks per library cycle.
            The library size is ``n_bbs_per_cycle ** n_cycles`` compounds.
        n_cycles: Number of library cycles.  Output has ``code_1`` …
            ``code_{n_cycles}`` columns.
        n_replicates: Number of replicates per condition (target and control).
        total_reads_per_replicate: Target reads per replicate.  Actual totals
            differ slightly due to Poisson sampling and enrichment.
        n_true_binders: Number of compounds to plant as enriched hits.
            Chosen uniformly at random (without replacement) from the library.
        enrichment_fold: Expected-count multiplier for planted binders in
            target replicates only.  50.0 = binders are 50 × more likely to
            be sequenced.
        seed: NumPy random seed for full reproducibility.

    Returns:
        Tuple ``(counts_df, planted_binders)`` where:

        - ``counts_df``: Long-format Polars DataFrame with columns
          ``selection``, ``code_1``, …, ``code_N``, ``count``, ``id``.
          Selection names are ``"target_1"`` … ``"target_{n_replicates}"``
          (post-selection) and ``"control_1"`` … ``"control_{n_replicates}"``
          (blank control).  **All** compounds appear in every selection,
          including those with count=0.
        - ``planted_binders``: List of ``n_true_binders`` tuples
          ``(code_1_value, …, code_N_value)`` for the planted hits.
    """
    rng = np.random.default_rng(seed)
    n_compounds = n_bbs_per_cycle**n_cycles

    # All compound combinations: shape (n_compounds, n_cycles), dtype int64
    compounds = np.array(
        list(itertools.product(range(n_bbs_per_cycle), repeat=n_cycles)),
        dtype=np.int64,
    )

    base_count = total_reads_per_replicate / n_compounds
    base_expected = np.full(n_compounds, base_count)

    # Plant the true binders
    binder_indices = rng.choice(n_compounds, size=n_true_binders, replace=False)
    planted_binders: list[tuple[int, ...]] = [
        tuple(int(v) for v in compounds[i]) for i in binder_indices
    ]

    sel_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []
    code_chunks: list[np.ndarray] = []

    for rep_idx in range(n_replicates):
        # Post-selection (target): binders get enrichment_fold boost
        expected_post = base_expected.copy()
        expected_post[binder_indices] *= enrichment_fold
        counts_post = rng.poisson(expected_post).astype(np.int64)

        sel_chunks.append(np.full(n_compounds, f"target_{rep_idx + 1}"))
        count_chunks.append(counts_post)
        code_chunks.append(compounds)

        # Control (blank): uniform Poisson, no enrichment
        counts_ctrl = rng.poisson(base_expected).astype(np.int64)

        sel_chunks.append(np.full(n_compounds, f"control_{rep_idx + 1}"))
        count_chunks.append(counts_ctrl)
        code_chunks.append(compounds)

    all_selections = np.concatenate(sel_chunks)  # (n_rows,)
    all_counts = np.concatenate(count_chunks)     # (n_rows,)
    all_codes = np.vstack(code_chunks)            # (n_rows, n_cycles)

    data: dict[str, object] = {"selection": all_selections.tolist()}
    for ci in range(n_cycles):
        data[f"code_{ci + 1}"] = all_codes[:, ci].tolist()
    data["count"] = all_counts.tolist()

    counts_df = pl.DataFrame(
        data,
        schema={
            "selection": pl.Utf8,
            **{f"code_{ci + 1}": pl.Int64 for ci in range(n_cycles)},
            "count": pl.Int64,
        },
    ).with_columns(
        pl.concat_str(
            [pl.col(f"code_{ci + 1}").cast(pl.Utf8) for ci in range(n_cycles)],
            separator="_",
        ).alias("id")
    )

    return counts_df, planted_binders


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------


def _run_pipeline(counts_df: pl.DataFrame) -> pl.DataFrame:
    """Run the full enrichment + ranking pipeline on a synthetic 2-cycle dataset.

    Returns:
        Ranked DataFrame with columns from the full-compound enrichment level
        plus ``agreement_score``, ``support_score``, ``property_penalty``,
        ``composite_score``, and ``rank``.
    """
    result = run_multilevel_enrichment(
        counts_df,
        n_cycles=2,
        code_cols=_CODE_COLS,
        post_selections=_POST_SELS,
        control_selections=_CTRL_SELS,
    )
    return compute_composite_rank(result, _CODE_COLS)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_full_pipeline_recovers_planted_binders() -> None:
    """Plant 10 binders at 50 × enrichment — all 10 must appear in the top 50.

    With 40 000 compounds (200 × 200) and 50-fold enrichment, planted binders
    accumulate ~50 × more reads than noise.  A functional pipeline must rank
    all 10 within the top 50 positions.
    """
    counts_df, planted = generate_synthetic_counts()
    ranked = _run_pipeline(counts_df)

    top50 = ranked.filter(pl.col("rank") <= 50)
    top50_compounds = {tuple(row) for row in top50.select(_CODE_COLS).rows()}

    missing = [b for b in planted if b not in top50_compounds]
    assert not missing, (
        f"{len(missing)} planted binder(s) not in top 50: {missing}.  "
        f"Their ranks: "
        + str(
            [
                int(
                    ranked.filter(
                        (pl.col("code_1") == b[0]) & (pl.col("code_2") == b[1])
                    )["rank"][0]
                )
                for b in missing
            ]
        )
    )


@pytest.mark.benchmark
def test_precision_at_10() -> None:
    """At least 8 of the top 10 ranked compounds must be planted binders.

    Precision@10 ≥ 0.8 confirms that the ranking correctly puts true hits
    at the very top of the list even before the full top-50 cutoff.
    """
    counts_df, planted = generate_synthetic_counts()
    ranked = _run_pipeline(counts_df)

    top10 = ranked.filter(pl.col("rank") <= 10)
    top10_compounds = {tuple(row) for row in top10.select(_CODE_COLS).rows()}
    planted_set = set(planted)

    precision = len(top10_compounds & planted_set) / 10
    assert precision >= 0.8, (
        f"Precision@10 = {precision:.2f}, expected ≥ 0.8.  "
        f"Top-10 compounds: {sorted(top10_compounds)}, "
        f"planted: {sorted(planted_set)}"
    )


@pytest.mark.benchmark
def test_top_ranked_is_planted() -> None:
    """The rank-1 compound must be one of the planted binders."""
    counts_df, planted = generate_synthetic_counts()
    ranked = _run_pipeline(counts_df)

    rank1 = tuple(ranked.filter(pl.col("rank") == 1).select(_CODE_COLS).row(0))
    planted_set = set(planted)
    assert rank1 in planted_set, (
        f"Rank-1 compound {rank1} is not a planted binder.  "
        f"Planted binders: {sorted(planted_set)}"
    )


@pytest.mark.benchmark
def test_robustness_to_subsampling() -> None:
    """At 10 % sequencing depth, recall@100 must still be ≥ 0.8.

    Reduces total_reads_per_replicate from 500 000 to 50 000.  With
    base_count = 50 000 / 40 000 = 1.25 reads/compound, binders still
    receive ~62.5 expected reads per replicate (50 × boost) vs. ~1.25 for
    noise — a 50-fold signal that should reliably surface all binders.
    """
    counts_df, planted = generate_synthetic_counts(
        total_reads_per_replicate=50_000
    )
    ranked = _run_pipeline(counts_df)

    top100 = ranked.filter(pl.col("rank") <= 100)
    top100_compounds = {tuple(row) for row in top100.select(_CODE_COLS).rows()}
    planted_set = set(planted)

    recall = len(top100_compounds & planted_set) / len(planted)
    assert recall >= 0.8, (
        f"Recall@100 = {recall:.2f}, expected ≥ 0.8.  "
        f"Missed binders: {sorted(planted_set - top100_compounds)}"
    )


@pytest.mark.benchmark
def test_no_enrichment_gives_no_strong_hits() -> None:
    """In a null experiment (enrichment_fold=1.0) no z-score should exceed 3.

    With 40 000 compounds all following Poisson(37.5) after merging 3
    replicates of 500 000 reads, the expected maximum z_n across the library
    is ≈ 0.004 — orders of magnitude below the z > 3 alarm threshold.
    """
    counts_df, _ = generate_synthetic_counts(enrichment_fold=1.0)
    ranked = _run_pipeline(counts_df)

    max_zscore = float(ranked["zscore"].max())
    assert max_zscore < 3.0, (
        f"Max z-score = {max_zscore:.4f} in null experiment, expected < 3.0"
    )


@pytest.mark.benchmark
def test_support_scores_correct() -> None:
    """Planted binders' monosynthons are enriched → support_score > 0.

    Uses a small library (n_bbs_per_cycle=10, 100 compounds) with 3 planted
    binders.  The monosynthon fold-enrichment in this regime is ≈ 2.4 ×,
    comfortably above the 2.0 × threshold used by the support-scoring module.

    Mathematical rationale
    ----------------------
    Monosynthon fold for a monosynthon containing exactly 1 binder::

        fold = (n_bbs - 1 + enrichment_fold) /
               (n_bbs × (1 + n_binders/n_compounds × (enrichment_fold - 1)))

    With n_bbs=10, n_compounds=100, n_binders=3, enrichment_fold=50::

        fold = (9 + 50) / (10 × (1 + 3/100 × 49)) = 59 / 24.7 ≈ 2.39 ✓

    With the default n_bbs_per_cycle=200 the binders contribute so little to
    total reads that monosynthon fold ≈ 1.0 × — below the 2.0 × threshold by
    design.  For large real DEL libraries, monosynthon support is not expected;
    support scoring is most powerful at the disynthon level for 3-cycle data.
    """
    counts_df, planted = generate_synthetic_counts(
        n_bbs_per_cycle=10, n_true_binders=3
    )
    ranked = _run_pipeline(counts_df)

    for binder in planted:
        binder_row = ranked.filter(
            (pl.col("code_1") == binder[0]) & (pl.col("code_2") == binder[1])
        )
        support = float(binder_row["support_score"][0])
        assert support > 0, (
            f"Planted binder {binder} has support_score={support}, expected > 0"
        )
