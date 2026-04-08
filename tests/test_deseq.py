"""Tests for analyse/deseq.py.

PyDESeq2 tests are slow (10-30 s) due to the iterative NB GLM fitting.
They are marked with @pytest.mark.slow so the fast unit-test suite can
skip them: ``pytest -m 'not slow'``

Synthetic fixture enrichment pattern (from conftest.py):
  True binders: (code_1=2, code_2=3) and (code_1=5, code_2=6)
    target_*:  counts 100–500
    blank_*:   counts   1–5
  Bead binders: code_1==7
    all:       counts  40–80  (same in protein and control → log2FC ≈ 0)
  Noise: everything else
    all:       counts   0–20  (approximately equal across conditions)
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from delexplore.analyse.deseq import prepare_deseq_input, run_deseq2

# Synthetic library dimensions
N_BB1, N_BB2 = 10, 8
N_COMPOUNDS = N_BB1 * N_BB2       # 80
N_SELECTIONS = 6                   # 3 blank + 3 target
CODE_COLS = ["code_1", "code_2"]

TRUE_BINDERS = [("2_3"), ("5_6")]  # compound_id strings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metadata(selections: list[str]) -> pl.DataFrame:
    """Build a minimal metadata DataFrame for the given selections."""
    rows = [
        {
            "selection_name": s,
            "condition": "protein" if s.startswith("target") else "no_protein",
        }
        for s in selections
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# prepare_deseq_input
# ---------------------------------------------------------------------------


class TestPrepareDeseqInput:
    def test_returns_two_dataframes(self, synthetic_counts):
        counts_pd, meta_pd = prepare_deseq_input(synthetic_counts, CODE_COLS)
        assert isinstance(counts_pd, pd.DataFrame)
        assert isinstance(meta_pd, pd.DataFrame)

    def test_counts_shape(self, synthetic_counts):
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        # 6 selections × 80 compounds
        assert counts_pd.shape == (N_SELECTIONS, N_COMPOUNDS)

    def test_counts_index_is_selection_names(self, synthetic_counts):
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        assert set(counts_pd.index) == {
            "blank_1", "blank_2", "blank_3",
            "target_1", "target_2", "target_3",
        }

    def test_counts_columns_are_compound_ids(self, synthetic_counts):
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        # Every column should match the pattern "int_int"
        for col in counts_pd.columns:
            parts = col.split("_")
            assert len(parts) == 2
            assert all(p.isdigit() for p in parts)

    def test_true_binder_columns_present(self, synthetic_counts):
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        for cid in TRUE_BINDERS:
            assert cid in counts_pd.columns

    def test_counts_are_integers(self, synthetic_counts):
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        assert counts_pd.dtypes.apply(lambda d: np.issubdtype(d, np.integer)).all()

    def test_no_negative_counts(self, synthetic_counts):
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        assert (counts_pd >= 0).all().all()

    def test_fill_null_with_zero(self, synthetic_counts):
        """After pivot + fill_null, no NaN values should exist."""
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        assert counts_pd.isna().sum().sum() == 0

    def test_metadata_index_matches_counts_index(self, synthetic_counts):
        counts_pd, meta_pd = prepare_deseq_input(synthetic_counts, CODE_COLS)
        assert set(meta_pd.index) == set(counts_pd.index)

    def test_true_binder_high_counts_in_target(self, synthetic_counts):
        """True binder compound_id '2_3' should have high counts for target selections."""
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        target_counts = counts_pd.loc[
            counts_pd.index.str.startswith("target"), "2_3"
        ]
        assert target_counts.min() >= 100

    def test_true_binder_low_counts_in_blank(self, synthetic_counts):
        counts_pd, _ = prepare_deseq_input(synthetic_counts, CODE_COLS)
        blank_counts = counts_pd.loc[
            counts_pd.index.str.startswith("blank"), "2_3"
        ]
        assert blank_counts.max() <= 10

    def test_missing_column_raises(self, synthetic_counts):
        with pytest.raises(ValueError, match="missing required columns"):
            prepare_deseq_input(
                synthetic_counts.drop("code_1"),
                CODE_COLS,
            )


# ---------------------------------------------------------------------------
# run_deseq2  (slow — full NB GLM fitting)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunDeseq2:
    @pytest.fixture(scope="class")
    def deseq_results(self, synthetic_counts):
        """Run DESeq2 once per class; reuse across all tests in the class."""
        selections = [
            "blank_1", "blank_2", "blank_3",
            "target_1", "target_2", "target_3",
        ]
        meta = _metadata(selections)
        return run_deseq2(synthetic_counts, meta, CODE_COLS)

    # ── Output shape and schema ───────────────────────────────────────────

    def test_returns_polars_dataframe(self, deseq_results):
        assert isinstance(deseq_results, pl.DataFrame)

    def test_has_required_columns(self, deseq_results):
        for col in ("compound_id", "log2FoldChange", "pvalue", "padj", "baseMean"):
            assert col in deseq_results.columns

    def test_row_count(self, deseq_results):
        # One row per compound (up to 80); PyDESeq2 independent filtering may
        # remove all-zero features but our synthetic data has no such features.
        assert len(deseq_results) == N_COMPOUNDS

    def test_compound_id_format(self, deseq_results):
        for cid in deseq_results["compound_id"].to_list():
            parts = cid.split("_")
            assert len(parts) == 2 and all(p.isdigit() for p in parts)

    # ── Statistical correctness ───────────────────────────────────────────

    def test_true_binders_positive_log2fc(self, deseq_results):
        """True binders should be more abundant in protein than control → log2FC > 0."""
        for cid in TRUE_BINDERS:
            row = deseq_results.filter(pl.col("compound_id") == cid)
            assert len(row) == 1, f"compound {cid} not in results"
            lfc = row["log2FoldChange"][0]
            assert lfc > 0, f"{cid}: log2FC={lfc:.2f}, expected > 0"

    def test_true_binders_significant(self, deseq_results):
        """True binders should have padj < 0.05."""
        for cid in TRUE_BINDERS:
            row = deseq_results.filter(pl.col("compound_id") == cid)
            padj = row["padj"][0]
            assert padj is not None and padj < 0.05, (
                f"{cid}: padj={padj}, expected < 0.05"
            )

    def test_noise_compounds_mostly_not_significant(self, deseq_results):
        """The majority of noise compounds should NOT be significant (padj >= 0.05).

        Bead binders (code_1==7) are non-specific so they should also be
        non-significant after correcting for multiple testing.
        We allow up to 10 % false positives (generous for a small library).
        """
        binder_ids = set(TRUE_BINDERS)
        non_binders = deseq_results.filter(
            ~pl.col("compound_id").is_in(list(binder_ids))
        )
        # Filter out rows where padj is null (independent filtering by DESeq2)
        testable = non_binders.filter(pl.col("padj").is_not_null())
        n_sig = (testable["padj"] < 0.05).sum()
        frac_sig = n_sig / len(testable) if len(testable) > 0 else 0
        assert frac_sig < 0.10, (
            f"{n_sig}/{len(testable)} non-binder compounds are significant "
            f"(padj<0.05); expected < 10%"
        )

    def test_basemean_positive_for_abundant_compounds(self, deseq_results):
        """True binders have high counts in some conditions → baseMean > 0."""
        for cid in TRUE_BINDERS:
            row = deseq_results.filter(pl.col("compound_id") == cid)
            assert row["baseMean"][0] > 0

    # ── Error handling ────────────────────────────────────────────────────

    def test_single_sample_per_condition_raises(self, synthetic_counts):
        """PyDESeq2 cannot estimate dispersion with < 2 replicates per condition."""
        meta = _metadata(["blank_1", "target_1"])  # 1 sample each
        with pytest.raises(ValueError, match="at least 2 samples"):
            run_deseq2(synthetic_counts, meta, CODE_COLS)

    def test_missing_condition_column_raises(self, synthetic_counts):
        meta = pl.DataFrame({
            "selection_name": ["blank_1", "blank_2", "target_1", "target_2"],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            run_deseq2(synthetic_counts, meta, CODE_COLS)
