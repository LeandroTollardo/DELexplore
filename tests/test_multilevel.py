"""Tests for analyse/multilevel.py.

Synthetic fixture enrichment pattern:
  True binders:  (code_1=2, code_2=3), (code_1=5, code_2=6)
    target_*: 100-500   blank_*: 1-5
  Bead binders:  code_1==7  →  40-80 in all selections (no protein preference)
  Noise:         everything else  →  0-20 in all selections
"""

import polars as pl
import pytest

from delexplore.analyse.multilevel import (
    merge_replicates,
    run_deseq2_enrichment,
    run_multilevel_enrichment,
)

POST_SELS   = ["target_1", "target_2", "target_3"]
CTRL_SELS   = ["blank_1",  "blank_2",  "blank_3"]
CODE_COLS   = ["code_1", "code_2"]
N_CYCLES    = 2
N_BB1, N_BB2 = 10, 8
TRUE_BINDERS = [(2, 3), (5, 6)]   # (code_1, code_2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(selections: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "selection_name": selections,
        "condition": [
            "protein" if s.startswith("target") else "no_protein"
            for s in selections
        ],
    })


# ---------------------------------------------------------------------------
# merge_replicates
# ---------------------------------------------------------------------------


class TestMergeReplicates:
    def test_returns_dataframe_and_int(self, synthetic_counts):
        merged, total = merge_replicates(synthetic_counts, POST_SELS, CODE_COLS)
        assert isinstance(merged, pl.DataFrame)
        assert isinstance(total, int)

    def test_merged_has_code_columns_and_count(self, synthetic_counts):
        merged, _ = merge_replicates(synthetic_counts, POST_SELS, CODE_COLS)
        assert "code_1" in merged.columns
        assert "code_2" in merged.columns
        assert "count" in merged.columns
        assert "selection" not in merged.columns

    def test_one_row_per_compound(self, synthetic_counts):
        merged, _ = merge_replicates(synthetic_counts, POST_SELS, CODE_COLS)
        assert len(merged) == N_BB1 * N_BB2

    def test_counts_are_summed_across_replicates(self, synthetic_counts):
        """For a given compound, merged count == sum across the three replicate selections."""
        c1, c2 = 2, 3  # true binder
        expected_sum = int(
            synthetic_counts
            .filter(
                pl.col("selection").is_in(POST_SELS)
                & (pl.col("code_1") == c1)
                & (pl.col("code_2") == c2)
            )["count"]
            .sum()
        )
        merged, _ = merge_replicates(synthetic_counts, POST_SELS, CODE_COLS)
        actual = int(
            merged
            .filter((pl.col("code_1") == c1) & (pl.col("code_2") == c2))["count"]
            .item()
        )
        assert actual == expected_sum

    def test_total_reads_equals_sum_of_all_counts(self, synthetic_counts):
        _, total = merge_replicates(synthetic_counts, POST_SELS, CODE_COLS)
        expected = int(
            synthetic_counts
            .filter(pl.col("selection").is_in(POST_SELS))["count"]
            .sum()
        )
        assert total == expected

    def test_missing_selection_raises(self, synthetic_counts):
        with pytest.raises(ValueError, match="None of the requested selections"):
            merge_replicates(synthetic_counts, ["nonexistent_sel"], CODE_COLS)

    def test_partial_missing_warns_and_continues(self, synthetic_counts):
        """If some selections are present, partial results are returned with a warning."""
        merged, total = merge_replicates(
            synthetic_counts,
            ["target_1", "nonexistent_sel"],  # one real, one fake
            CODE_COLS,
        )
        assert len(merged) == N_BB1 * N_BB2
        assert total > 0


# ---------------------------------------------------------------------------
# run_multilevel_enrichment
# ---------------------------------------------------------------------------


class TestRunMultilevelEnrichment:
    @pytest.fixture(scope="class")
    def ml_result(self, synthetic_counts):
        return run_multilevel_enrichment(
            synthetic_counts,
            n_cycles=N_CYCLES,
            code_cols=CODE_COLS,
            post_selections=POST_SELS,
            control_selections=CTRL_SELS,
        )

    # ── Structure ────────────────────────────────────────────────────────

    def test_returns_dict(self, ml_result):
        assert isinstance(ml_result, dict)

    def test_two_cycles_gives_three_levels(self, ml_result):
        assert len(ml_result) == 3

    def test_expected_level_keys(self, ml_result):
        assert set(ml_result.keys()) == {
            "mono_code_1",
            "mono_code_2",
            "di_code_1_code_2",
        }

    def test_all_values_are_dataframes(self, ml_result):
        for name, df in ml_result.items():
            assert isinstance(df, pl.DataFrame), f"{name} is not a DataFrame"

    def test_mono_code_1_row_count(self, ml_result):
        assert len(ml_result["mono_code_1"]) == N_BB1

    def test_mono_code_2_row_count(self, ml_result):
        assert len(ml_result["mono_code_2"]) == N_BB2

    def test_di_row_count(self, ml_result):
        assert len(ml_result["di_code_1_code_2"]) == N_BB1 * N_BB2

    # ── Required columns ────────────────────────────────────────────────

    def test_di_has_count_columns(self, ml_result):
        df = ml_result["di_code_1_code_2"]
        for col in ("count_post", "count_control", "total_post", "total_control", "diversity"):
            assert col in df.columns, f"Missing column: {col}"

    def test_di_has_zscore_columns(self, ml_result):
        df = ml_result["di_code_1_code_2"]
        for col in ("zscore", "zscore_ci_lower", "zscore_ci_upper"):
            assert col in df.columns

    def test_di_has_poisson_columns(self, ml_result):
        df = ml_result["di_code_1_code_2"]
        for col in ("poisson_ml_enrichment", "fold_enrichment", "poisson_ci_lower", "poisson_ci_upper"):
            assert col in df.columns

    # ── Diversity values ─────────────────────────────────────────────────

    def test_mono_code_1_diversity(self, ml_result):
        df = ml_result["mono_code_1"]
        assert df["diversity"][0] == N_BB1

    def test_mono_code_2_diversity(self, ml_result):
        df = ml_result["mono_code_2"]
        assert df["diversity"][0] == N_BB2

    def test_di_diversity(self, ml_result):
        df = ml_result["di_code_1_code_2"]
        assert df["diversity"][0] == N_BB1 * N_BB2

    # ── Enrichment correctness: true binders ─────────────────────────────

    def test_true_binders_positive_zscore_at_di_level(self, ml_result):
        """Both true binders should have z_n > 0 at the disynthon level."""
        df = ml_result["di_code_1_code_2"]
        for c1, c2 in TRUE_BINDERS:
            row = df.filter((pl.col("code_1") == c1) & (pl.col("code_2") == c2))
            assert len(row) == 1
            assert row["zscore"][0] > 0, f"True binder ({c1},{c2}) z={row['zscore'][0]:.3f}"

    def test_true_binders_high_zscore_vs_noise(self, ml_result):
        """True binder z-scores must exceed median noise z-score."""
        df = ml_result["di_code_1_code_2"]
        binder_ids = {(c1, c2) for c1, c2 in TRUE_BINDERS}
        noise = df.filter(
            ~(pl.struct(["code_1", "code_2"]).is_in(
                [{"code_1": c1, "code_2": c2} for c1, c2 in binder_ids]
            ))
        )["zscore"].median()
        for c1, c2 in TRUE_BINDERS:
            row = df.filter((pl.col("code_1") == c1) & (pl.col("code_2") == c2))
            assert row["zscore"][0] > noise  # type: ignore[operator]

    def test_true_binders_positive_ml_enrichment(self, ml_result):
        """Both true binders should have ML enrichment > 1 at the disynthon level."""
        df = ml_result["di_code_1_code_2"]
        for c1, c2 in TRUE_BINDERS:
            row = df.filter((pl.col("code_1") == c1) & (pl.col("code_2") == c2))
            ml = row["poisson_ml_enrichment"][0]
            assert ml > 1.0, f"True binder ({c1},{c2}) ML={ml:.3f}"

    # ── Enrichment correctness: mono-level detects enriched building blocks ──

    def test_enriched_bb1_detected_at_mono_level(self, ml_result):
        """code_1=2 (part of true binder (2,3)) should be enriched at monosynthon level."""
        df = ml_result["mono_code_1"]
        row = df.filter(pl.col("code_1") == 2)
        assert row["zscore"][0] > 0

    def test_enriched_bb2_detected_at_mono_level(self, ml_result):
        """code_2=3 (part of true binder (2,3)) should be enriched at monosynthon level."""
        df = ml_result["mono_code_2"]
        row = df.filter(pl.col("code_2") == 3)
        assert row["zscore"][0] > 0

    # ── CI consistency ────────────────────────────────────────────────────

    def test_zscore_ci_brackets_zscore(self, ml_result):
        """ci_lower <= z_score <= ci_upper for all features at every level."""
        for name, df in ml_result.items():
            bad = df.filter(
                (pl.col("zscore_ci_lower") > pl.col("zscore") + 1e-9)
                | (pl.col("zscore") > pl.col("zscore_ci_upper") + 1e-9)
            )
            assert len(bad) == 0, f"{name}: CI does not bracket z-score for {len(bad)} rows"

    def test_poisson_ci_lower_le_upper(self, ml_result):
        for name, df in ml_result.items():
            bad = df.filter(pl.col("poisson_ci_lower") > pl.col("poisson_ci_upper") + 1e-9)
            assert len(bad) == 0, f"{name}: poisson_ci_lower > poisson_ci_upper for {len(bad)} rows"

    def test_no_null_zscores(self, ml_result):
        for name, df in ml_result.items():
            assert df["zscore"].null_count() == 0, f"{name} has null z-scores"

    # ── Method filtering ─────────────────────────────────────────────────

    def test_zscore_only_has_no_poisson_columns(self, synthetic_counts):
        result = run_multilevel_enrichment(
            synthetic_counts, N_CYCLES, CODE_COLS,
            POST_SELS, CTRL_SELS,
            methods=("zscore",),
        )
        for df in result.values():
            assert "zscore" in df.columns
            assert "poisson_ml_enrichment" not in df.columns

    def test_poisson_only_has_no_zscore_columns(self, synthetic_counts):
        result = run_multilevel_enrichment(
            synthetic_counts, N_CYCLES, CODE_COLS,
            POST_SELS, CTRL_SELS,
            methods=("poisson_ml",),
        )
        for df in result.values():
            assert "poisson_ml_enrichment" in df.columns
            assert "zscore" not in df.columns

    def test_unknown_method_raises(self, synthetic_counts):
        with pytest.raises(ValueError, match="Unknown enrichment method"):
            run_multilevel_enrichment(
                synthetic_counts, N_CYCLES, CODE_COLS,
                POST_SELS, CTRL_SELS,
                methods=("zscore", "edgeR"),
            )


# ---------------------------------------------------------------------------
# run_deseq2_enrichment  (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunDeseq2Enrichment:
    @pytest.fixture(scope="class")
    def deseq_ml_result(self, synthetic_counts):
        all_sels = POST_SELS + CTRL_SELS
        meta = _meta(all_sels)
        return run_deseq2_enrichment(
            synthetic_counts, meta, N_CYCLES, CODE_COLS
        )

    def test_returns_dict(self, deseq_ml_result):
        assert isinstance(deseq_ml_result, dict)

    def test_default_levels_excludes_trisynthon(self, deseq_ml_result):
        """Default: for a 2-cycle library, di IS the full compound level (n_cycles=2),
        so only mono levels are run by default."""
        # For n_cycles=2: all levels = [mono_1, mono_2, di]; di has len==n_cycles,
        # so default skips di and runs only mono_code_1 and mono_code_2.
        assert "di_code_1_code_2" not in deseq_ml_result
        assert "mono_code_1" in deseq_ml_result
        assert "mono_code_2" in deseq_ml_result

    def test_results_are_dataframes(self, deseq_ml_result):
        for name, df in deseq_ml_result.items():
            assert isinstance(df, pl.DataFrame), f"{name} not a DataFrame"

    def test_has_expected_deseq_columns(self, deseq_ml_result):
        for name, df in deseq_ml_result.items():
            for col in ("compound_id", "log2FoldChange", "padj"):
                assert col in df.columns, f"{name} missing {col}"

    def test_invalid_level_raises(self, synthetic_counts):
        meta = _meta(POST_SELS + CTRL_SELS)
        with pytest.raises(ValueError, match="Unknown level names"):
            run_deseq2_enrichment(
                synthetic_counts, meta, N_CYCLES, CODE_COLS,
                levels=["tri_code_1_code_2_code_3"],  # invalid for 2-cycle
            )
