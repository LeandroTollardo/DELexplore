"""Tests for analyse/bb_productivity.py."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from delexplore.analyse.bb_productivity import (
    compute_joint_pbind,
    compute_pbind,
    compute_pbind_support_score,
    identify_productive_bbs,
)

# ---------------------------------------------------------------------------
# Synthetic enrichment data
# ---------------------------------------------------------------------------
#
# Library layout: 2 cycles, 10 BBs each → 100 compounds.
#
# Enrichment pattern:
#   code_1 ∈ {0, 1}  AND  code_2 ∈ {0, 1}  → binder (zscore = 3.0)
#   everything else                          → noise   (zscore = 0.0)
#
# This gives exactly 4 binder compounds out of 100.
#
# Expected P(bind) per position:
#   code_1 = 0: 2 binders / 10 compounds = 0.20
#   code_1 = 1: 2 binders / 10 compounds = 0.20
#   code_1 ∈ {2..9}: 0 binders / 10 compounds = 0.00
#
#   code_2 = 0: 0.20, code_2 = 1: 0.20, code_2 ∈ {2..9}: 0.00
#


N_BBS = 10
_BINDER_C1 = {0, 1}
_BINDER_C2 = {0, 1}


def _make_enrichment_df() -> pl.DataFrame:
    """Full-compound enrichment DataFrame for a 10×10 library."""
    rows = []
    for c1 in range(N_BBS):
        for c2 in range(N_BBS):
            is_binder = (c1 in _BINDER_C1) and (c2 in _BINDER_C2)
            rows.append({
                "code_1": c1,
                "code_2": c2,
                "zscore":          3.0 if is_binder else 0.0,
                "fold_enrichment": 50.0 if is_binder else 1.0,
                "count_post":      100 if is_binder else 2,
            })
    return pl.DataFrame(rows)


# Convenience
_DF = _make_enrichment_df()
_COLS = ["code_1", "code_2"]
_PBIND_DEFAULT_THRESHOLD = 1.0   # zscore >= 1.0 → binder


def _pbind() -> dict[str, pl.DataFrame]:
    return compute_pbind(_DF, _COLS, score_col="zscore",
                         binder_threshold=_PBIND_DEFAULT_THRESHOLD)


# ---------------------------------------------------------------------------
# 1. compute_pbind
# ---------------------------------------------------------------------------


class TestComputePbind:
    def test_returns_one_entry_per_position(self) -> None:
        pb = _pbind()
        assert set(pb.keys()) == {"code_1", "code_2"}

    def test_output_has_required_columns(self) -> None:
        pb = _pbind()
        for col, df in pb.items():
            for expected in (col, "n_total", "n_binders", "p_bind",
                             "n_compatible_partners"):
                assert expected in df.columns, f"Missing column '{expected}' in {col}"

    def test_n_total_is_n_bbs_per_position(self) -> None:
        """Every BB appears in exactly N_BBS compounds (10×10 balanced library)."""
        pb = _pbind()
        for df in pb.values():
            assert (df["n_total"] == N_BBS).all()

    def test_enriched_bbs_have_highest_pbind(self) -> None:
        """code_1 ∈ {0,1} and code_2 ∈ {0,1} should have p_bind = 0.20."""
        pb = _pbind()
        for col, binder_set in [("code_1", _BINDER_C1), ("code_2", _BINDER_C2)]:
            df = pb[col]
            for bb in binder_set:
                row = df.filter(pl.col(col) == bb)
                assert len(row) == 1
                p = float(row["p_bind"][0])
                assert abs(p - 0.20) < 1e-9, (
                    f"{col}={bb}: expected p_bind=0.20, got {p}"
                )

    def test_noise_bbs_have_zero_pbind(self) -> None:
        pb = _pbind()
        for col, binder_set in [("code_1", _BINDER_C1), ("code_2", _BINDER_C2)]:
            df = pb[col]
            noise_bbs = [i for i in range(N_BBS) if i not in binder_set]
            for bb in noise_bbs:
                row = df.filter(pl.col(col) == bb)
                p = float(row["p_bind"][0])
                assert p == 0.0, f"{col}={bb}: expected p_bind=0.0, got {p}"

    def test_sorted_by_pbind_descending(self) -> None:
        pb = _pbind()
        for df in pb.values():
            vals = df["p_bind"].to_list()
            assert vals == sorted(vals, reverse=True)

    def test_compatible_partners_higher_for_enriched(self) -> None:
        """Enriched BBs (code_1=0) should have more compatible partners than noise."""
        pb = _pbind()
        df1 = pb["code_1"]
        enriched_partners = float(
            df1.filter(pl.col("code_1") == 0)["n_compatible_partners"][0]
        )
        noise_partners = float(
            df1.filter(pl.col("code_1") == 5)["n_compatible_partners"][0]
        )
        assert enriched_partners > noise_partners

    def test_compatible_partners_equals_n_binder_code2_values(self) -> None:
        """code_1=0 binds with code_2 ∈ {0,1} → 2 compatible partners."""
        pb = _pbind()
        df1 = pb["code_1"]
        partners_0 = int(df1.filter(pl.col("code_1") == 0)["n_compatible_partners"][0])
        assert partners_0 == len(_BINDER_C2)

    def test_noise_bbs_have_zero_compatible_partners(self) -> None:
        pb = _pbind()
        df1 = pb["code_1"]
        # code_1=5 has no binders → no compatible partners
        p = int(df1.filter(pl.col("code_1") == 5)["n_compatible_partners"][0])
        assert p == 0

    def test_n_binders_sums_correctly(self) -> None:
        """Total binders across all code_1 values = 4 (2 binder BBs × 2 binder partners)."""
        pb = _pbind()
        total_binders = int(pb["code_1"]["n_binders"].sum())
        # 4 binder compounds, each appears once per code_1 group
        assert total_binders == len(_BINDER_C1) * len(_BINDER_C2)

    def test_different_threshold_changes_pbind(self) -> None:
        """Higher threshold → no binders if all zscores are exactly 3.0."""
        pb_high = compute_pbind(_DF, _COLS, score_col="zscore",
                                binder_threshold=5.0)
        for df in pb_high.values():
            assert (df["p_bind"] == 0.0).all()

    def test_empty_df_raises(self) -> None:
        empty = pl.DataFrame({"code_1": [], "code_2": [], "zscore": []},
                             schema={"code_1": pl.Int64, "code_2": pl.Int64,
                                     "zscore": pl.Float64})
        with pytest.raises(ValueError, match="empty"):
            compute_pbind(empty, _COLS)

    def test_missing_score_col_raises(self) -> None:
        with pytest.raises(ValueError, match="score_col"):
            compute_pbind(_DF, _COLS, score_col="nonexistent")

    def test_missing_code_col_raises(self) -> None:
        with pytest.raises(ValueError, match="code_cols"):
            compute_pbind(_DF, ["code_1", "code_99"])

    def test_empty_code_cols_raises(self) -> None:
        with pytest.raises(ValueError, match="code_cols"):
            compute_pbind(_DF, [])

    def test_fold_enrichment_score_col(self) -> None:
        """Works with fold_enrichment column and threshold=2.0."""
        pb = compute_pbind(_DF, _COLS, score_col="fold_enrichment",
                           binder_threshold=2.0)
        for col, binder_set in [("code_1", _BINDER_C1), ("code_2", _BINDER_C2)]:
            df = pb[col]
            for bb in binder_set:
                p = float(df.filter(pl.col(col) == bb)["p_bind"][0])
                assert abs(p - 0.20) < 1e-9

    def test_single_position_library(self) -> None:
        """Single code_col → n_compatible_partners = 0 (no partners)."""
        df = pl.DataFrame({"code_1": [0, 1, 2], "zscore": [2.0, 0.0, 0.0]})
        pb = compute_pbind(df, ["code_1"])
        assert (pb["code_1"]["n_compatible_partners"] == 0).all()


# ---------------------------------------------------------------------------
# 2. compute_joint_pbind
# ---------------------------------------------------------------------------


class TestComputeJointPbind:
    def test_returns_one_entry_per_pair(self) -> None:
        jb = compute_joint_pbind(_DF, _COLS)
        assert "code_1__code_2" in jb

    def test_output_has_required_columns(self) -> None:
        jb = compute_joint_pbind(_DF, _COLS)
        df = jb["code_1__code_2"]
        for col in ("bin_i", "bin_j", "bin_i_label", "bin_j_label",
                    "n_compounds", "n_binders",
                    "joint_p_bind", "marginal_product", "interaction"):
            assert col in df.columns, f"Missing column: {col}"

    def test_no_zero_count_rows(self) -> None:
        """Rows with n_compounds=0 should be filtered out."""
        jb = compute_joint_pbind(_DF, _COLS)
        for df in jb.values():
            assert (df["n_compounds"] > 0).all()

    def test_joint_pbind_bounded(self) -> None:
        jb = compute_joint_pbind(_DF, _COLS)
        for df in jb.values():
            assert (df["joint_p_bind"] >= 0.0).all()
            assert (df["joint_p_bind"] <= 1.0).all()

    def test_highest_bin_highest_joint_pbind(self) -> None:
        """The (high, high) bin should have the highest joint P(bind)."""
        jb = compute_joint_pbind(_DF, _COLS, n_bins=4)
        df = jb["code_1__code_2"]
        max_bin_i = int(df["bin_i"].max())
        max_bin_j = int(df["bin_j"].max())
        high_high = df.filter(
            (pl.col("bin_i") == max_bin_i) & (pl.col("bin_j") == max_bin_j)
        )
        if len(high_high) > 0:
            max_jpb = float(df["joint_p_bind"].max())
            assert float(high_high["joint_p_bind"][0]) == pytest.approx(max_jpb, abs=1e-9)

    def test_interaction_can_be_positive(self) -> None:
        """High-high bin should show positive interaction (cooperative binding)."""
        jb = compute_joint_pbind(_DF, _COLS, n_bins=4)
        df = jb["code_1__code_2"]
        assert (df["interaction"] > 0).any()

    def test_sorted_by_bins(self) -> None:
        jb = compute_joint_pbind(_DF, _COLS)
        df = jb["code_1__code_2"]
        # Should be sorted by (bin_i, bin_j)
        assert df["bin_i"].to_list() == sorted(df["bin_i"].to_list())

    def test_three_positions_all_pairs_returned(self) -> None:
        """3 positions → 3 pairs."""
        df3 = _DF.with_columns(pl.col("code_1").alias("code_3"))
        jb = compute_joint_pbind(df3, ["code_1", "code_2", "code_3"])
        assert len(jb) == 3  # C(3,2) = 3

    def test_fewer_than_two_positions_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_joint_pbind(_DF, ["code_1"])

    def test_n_bins_parameter(self) -> None:
        """n_bins controls the bin resolution."""
        jb2 = compute_joint_pbind(_DF, _COLS, n_bins=2)
        jb8 = compute_joint_pbind(_DF, _COLS, n_bins=8)
        # With 2 bins, fewer unique (bin_i, bin_j) combinations
        n2 = len(jb2["code_1__code_2"])
        n8 = len(jb8["code_1__code_2"])
        assert n2 <= n8


# ---------------------------------------------------------------------------
# 3. compute_pbind_support_score
# ---------------------------------------------------------------------------


class TestComputePbindSupportScore:
    def test_output_length_matches_compound_df(self) -> None:
        pb = _pbind()
        scores = compute_pbind_support_score(pb, _DF, _COLS)
        assert len(scores) == len(_DF)

    def test_output_dtype_float64(self) -> None:
        pb = _pbind()
        scores = compute_pbind_support_score(pb, _DF, _COLS)
        assert scores.dtype == np.float64

    def test_binder_compounds_score_higher(self) -> None:
        """Binder compounds (code_1 ∈ {0,1}, code_2 ∈ {0,1}) → higher score."""
        pb = _pbind()
        scores = compute_pbind_support_score(pb, _DF, _COLS)

        binder_mask = (
            _DF["code_1"].is_in(list(_BINDER_C1)) &
            _DF["code_2"].is_in(list(_BINDER_C2))
        ).to_numpy()

        binder_scores = scores[binder_mask]
        noise_scores  = scores[~binder_mask]

        assert binder_scores.mean() > noise_scores.mean(), (
            f"Binder mean {binder_scores.mean():.4f} not > "
            f"noise mean {noise_scores.mean():.4f}"
        )

    def test_scores_bounded(self) -> None:
        """Scores should be in [0, n_positions]."""
        pb = _pbind()
        scores = compute_pbind_support_score(pb, _DF, _COLS)
        assert (scores >= 0.0).all()
        assert (scores <= float(len(_COLS))).all()

    def test_pure_binder_bb_max_score(self) -> None:
        """Compound (0,0): both BBs have p_bind=0.20 → score=0.40."""
        pb = _pbind()
        comp = pl.DataFrame({"code_1": [0], "code_2": [0]})
        scores = compute_pbind_support_score(pb, comp, _COLS)
        assert abs(scores[0] - 0.40) < 1e-9

    def test_pure_noise_score_is_zero(self) -> None:
        """Compound (5,5): both BBs have p_bind=0.0 → score=0.0."""
        pb = _pbind()
        comp = pl.DataFrame({"code_1": [5], "code_2": [5]})
        scores = compute_pbind_support_score(pb, comp, _COLS)
        assert scores[0] == 0.0

    def test_missing_pbind_position_treated_as_zero(self) -> None:
        """P(bind) dict missing one position → contribution is 0."""
        pb = _pbind()
        pb_partial = {"code_1": pb["code_1"]}  # code_2 missing
        scores = compute_pbind_support_score(pb_partial, _DF, _COLS)
        # No contribution from code_2 → max score is 0.20 (from code_1 alone)
        assert float(scores.max()) == pytest.approx(0.20, abs=1e-9)

    def test_empty_code_cols_raises(self) -> None:
        pb = _pbind()
        with pytest.raises(ValueError, match="code_cols"):
            compute_pbind_support_score(pb, _DF, [])

    def test_missing_code_col_in_compound_df_raises(self) -> None:
        pb = _pbind()
        df_missing = _DF.drop("code_2")
        with pytest.raises(ValueError, match="code_cols"):
            compute_pbind_support_score(pb, df_missing, _COLS)

    def test_scores_sum_to_expected_value(self) -> None:
        """For the 4 binder compounds total p_bind per compound = 0.4."""
        pb = _pbind()
        binders = _DF.filter(
            pl.col("code_1").is_in(list(_BINDER_C1)) &
            pl.col("code_2").is_in(list(_BINDER_C2))
        )
        scores = compute_pbind_support_score(pb, binders, _COLS)
        for s in scores:
            assert abs(s - 0.40) < 1e-9


# ---------------------------------------------------------------------------
# 4. identify_productive_bbs
# ---------------------------------------------------------------------------


class TestIdentifyProductiveBbs:
    def test_returns_one_list_per_position(self) -> None:
        pb = _pbind()
        top = identify_productive_bbs(pb)
        assert set(top.keys()) == {"code_1", "code_2"}

    def test_returns_list_of_integers(self) -> None:
        pb = _pbind()
        top = identify_productive_bbs(pb)
        for lst in top.values():
            assert all(isinstance(x, int) for x in lst)

    def test_top_fraction_limits_results(self) -> None:
        """top_fraction=0.10 → ceil(10 * 0.10) = 1 BB per position."""
        pb = _pbind()
        top = identify_productive_bbs(pb, top_fraction=0.10)
        for lst in top.values():
            assert len(lst) == 1

    def test_enriched_bbs_are_in_top_results(self) -> None:
        """The 2 enriched BBs (0 and 1) should be the top 20%."""
        pb = _pbind()
        top = identify_productive_bbs(pb, top_fraction=0.20)
        for col, binder_set in [("code_1", _BINDER_C1), ("code_2", _BINDER_C2)]:
            for bb in binder_set:
                assert bb in top[col], (
                    f"Enriched BB {col}={bb} not in top 20%: {top[col]}"
                )

    def test_noise_bbs_not_in_top_one_percent(self) -> None:
        """Top 1% = 1 BB per position; it should be an enriched one."""
        pb = _pbind()
        top = identify_productive_bbs(pb, top_fraction=0.01)
        for col, binder_set in [("code_1", _BINDER_C1), ("code_2", _BINDER_C2)]:
            assert top[col][0] in binder_set

    def test_at_least_one_bb_always_returned(self) -> None:
        """Even with top_fraction=0.001, at least 1 BB is returned."""
        pb = _pbind()
        top = identify_productive_bbs(pb, top_fraction=0.001)
        for lst in top.values():
            assert len(lst) >= 1

    def test_full_fraction_returns_all_bbs(self) -> None:
        """top_fraction=1.0 → all N_BBS BBs returned."""
        pb = _pbind()
        top = identify_productive_bbs(pb, top_fraction=1.0)
        for lst in top.values():
            assert len(lst) == N_BBS

    def test_invalid_top_fraction_raises(self) -> None:
        pb = _pbind()
        with pytest.raises(ValueError, match="top_fraction"):
            identify_productive_bbs(pb, top_fraction=0.0)
        with pytest.raises(ValueError, match="top_fraction"):
            identify_productive_bbs(pb, top_fraction=1.5)

    def test_empty_pbind_position_returns_empty_list(self) -> None:
        """If a position's DataFrame is empty, returns []."""
        pb = {
            "code_1": _pbind()["code_1"],
            "code_2": pl.DataFrame({"code_2": [], "n_total": [], "n_binders": [],
                                    "p_bind": [], "n_compatible_partners": []},
                                   schema={"code_2": pl.Int64, "n_total": pl.Int64,
                                           "n_binders": pl.Int64, "p_bind": pl.Float64,
                                           "n_compatible_partners": pl.Int64}),
        }
        top = identify_productive_bbs(pb)
        assert top["code_2"] == []


# ---------------------------------------------------------------------------
# 5. Integration: P(bind) support score vs binary support score
# ---------------------------------------------------------------------------


class TestPbindVsBinarySupport:
    def test_pbind_scores_rank_binders_above_noise(self) -> None:
        """P(bind) support scores should consistently rank binders above noise."""
        from delexplore.analyse.rank import compute_support_score

        pb = _pbind()
        pbind_scores = compute_pbind_support_score(pb, _DF, _COLS)

        binder_mask = (
            _DF["code_1"].is_in(list(_BINDER_C1)) &
            _DF["code_2"].is_in(list(_BINDER_C2))
        ).to_numpy()

        # P(bind) scores for binders vs noise
        assert pbind_scores[binder_mask].mean() > pbind_scores[~binder_mask].mean()

    def test_pbind_continuous_discriminates_partial_binders(self) -> None:
        """A compound with one enriched BB should score strictly between binder and noise."""
        pb = _pbind()
        binder   = pl.DataFrame({"code_1": [0], "code_2": [0]})  # both enriched
        partial  = pl.DataFrame({"code_1": [0], "code_2": [5]})  # one enriched
        noise    = pl.DataFrame({"code_1": [5], "code_2": [5]})  # none enriched

        s_binder  = compute_pbind_support_score(pb, binder, _COLS)[0]
        s_partial = compute_pbind_support_score(pb, partial, _COLS)[0]
        s_noise   = compute_pbind_support_score(pb, noise, _COLS)[0]

        assert s_binder > s_partial > s_noise, (
            f"Expected binder({s_binder:.4f}) > partial({s_partial:.4f}) > noise({s_noise:.4f})"
        )
