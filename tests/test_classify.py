"""Tests for analyse/classify.py.

Synthetic enrichment DataFrames are built inline — no need to run the full
multilevel pipeline.  Each DataFrame has columns: code_1, code_2, zscore.

Compound inventory used across tests:
  (0, 0) — true orthosteric binder: enriched target only
  (0, 1) — allosteric binder:       enriched target + inhibitor, not blank
  (0, 2) — cryptic binder:          enriched inhibitor only
  (0, 3) — nonspecific:             enriched blank (+ target)
  (0, 4) — not enriched:            below threshold in all conditions
  (1, 0) — bead artifact:           enriched in one bead type only
  (1, 1) — robust binder:           enriched in all bead types
"""

import polars as pl
import pytest

from delexplore.analyse.classify import classify_bead_artifacts, classify_binders

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

THRESHOLD = 1.0
THRESHOLD_COL = "zscore"

# All 7 test compounds
COMPOUNDS = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1)]


def _df(compounds: list[tuple[int, int]], scores: list[float]) -> pl.DataFrame:
    """Build a minimal enrichment DataFrame with code_1, code_2, zscore."""
    return pl.DataFrame(
        {
            "code_1": [c[0] for c in compounds],
            "code_2": [c[1] for c in compounds],
            THRESHOLD_COL: scores,
        },
        schema={"code_1": pl.Int64, "code_2": pl.Int64, THRESHOLD_COL: pl.Float64},
    )


# ── Condition DataFrames ─────────────────────────────────────────────────────
# Each compound appears with its expected score in that condition.
# Scores >= THRESHOLD (1.0) = enriched.

# target: (0,0), (0,1), (0,3) enriched
_TARGET_DF = _df(
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    [2.5,    2.0,    0.3,    1.8,    0.1],
)

# blank: only (0,3) enriched
_BLANK_DF = _df(
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    [0.2,    0.1,    0.4,    1.5,    0.0],
)

# target + inhibitor: (0,1) and (0,2) enriched
_INHIBITOR_DF = _df(
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    [0.4,    1.2,    1.5,    0.3,    0.0],
)

# ── Bead-type DataFrames ─────────────────────────────────────────────────────
# Bead A: (1,0) and (1,1) enriched
_BEAD_A_DF = _df(
    [(1, 0), (1, 1)],
    [0.3,    2.0],
)

# Bead B: only (1,1) enriched
_BEAD_B_DF = _df(
    [(1, 0), (1, 1)],
    [2.5,    1.5],
)

# Bead C: only (1,1) enriched
_BEAD_C_DF = _df(
    [(1, 0), (1, 1)],
    [0.2,    3.0],
)


# ---------------------------------------------------------------------------
# Helpers to pull a single compound's binder_type from the result
# ---------------------------------------------------------------------------


def _binder_type(result: pl.DataFrame, c1: int, c2: int) -> str:
    row = result.filter((pl.col("code_1") == c1) & (pl.col("code_2") == c2))
    assert len(row) == 1, f"Expected exactly 1 row for ({c1},{c2}), got {len(row)}"
    return row["binder_type"][0]


def _bead_class(result: pl.DataFrame, c1: int, c2: int) -> str:
    row = result.filter((pl.col("code_1") == c1) & (pl.col("code_2") == c2))
    assert len(row) == 1
    return row["bead_classification"][0]


# ---------------------------------------------------------------------------
# classify_binders — structure tests
# ---------------------------------------------------------------------------


class TestClassifyBindersStructure:
    @pytest.fixture(scope="class")
    def result(self):
        return classify_binders(
            {"target": _TARGET_DF, "blank": _BLANK_DF, "target_inhibitor": _INHIBITOR_DF},
            threshold_col=THRESHOLD_COL,
            threshold_value=THRESHOLD,
        )

    def test_returns_dataframe(self, result):
        assert isinstance(result, pl.DataFrame)

    def test_has_binder_type_column(self, result):
        assert "binder_type" in result.columns

    def test_has_code_columns(self, result):
        assert "code_1" in result.columns
        assert "code_2" in result.columns

    def test_no_extra_columns(self, result):
        assert set(result.columns) == {"code_1", "code_2", "binder_type"}

    def test_one_row_per_compound(self, result):
        n_unique = result.select(["code_1", "code_2"]).unique().height
        assert len(result) == n_unique

    def test_all_types_are_valid(self, result):
        valid = {"orthosteric", "allosteric", "cryptic", "nonspecific", "not_enriched"}
        for t in result["binder_type"].to_list():
            assert t in valid, f"Unexpected binder_type: {t}"


# ---------------------------------------------------------------------------
# classify_binders — classification correctness
# ---------------------------------------------------------------------------


class TestClassifyBindersCorrectness:
    @pytest.fixture(scope="class")
    def result(self):
        return classify_binders(
            {"target": _TARGET_DF, "blank": _BLANK_DF, "target_inhibitor": _INHIBITOR_DF},
            threshold_col=THRESHOLD_COL,
            threshold_value=THRESHOLD,
        )

    def test_orthosteric_compound(self, result):
        """(0,0): enriched target, not blank, not inhibitor → orthosteric."""
        assert _binder_type(result, 0, 0) == "orthosteric"

    def test_allosteric_compound(self, result):
        """(0,1): enriched target AND inhibitor, not blank → allosteric."""
        assert _binder_type(result, 0, 1) == "allosteric"

    def test_cryptic_compound(self, result):
        """(0,2): enriched inhibitor only (not target, not blank) → cryptic."""
        assert _binder_type(result, 0, 2) == "cryptic"

    def test_nonspecific_compound(self, result):
        """(0,3): enriched blank → nonspecific (overrides target enrichment)."""
        assert _binder_type(result, 0, 3) == "nonspecific"

    def test_not_enriched_compound(self, result):
        """(0,4): below threshold in all conditions → not_enriched."""
        assert _binder_type(result, 0, 4) == "not_enriched"

    def test_blank_wins_over_target(self, result):
        """Nonspecific takes priority even when the compound is also target-enriched."""
        # (0,3) is enriched in BOTH target and blank → must be "nonspecific"
        assert _binder_type(result, 0, 3) == "nonspecific"


# ---------------------------------------------------------------------------
# classify_binders — no inhibitor condition
# ---------------------------------------------------------------------------


class TestClassifyBindersNoInhibitor:
    @pytest.fixture(scope="class")
    def result(self):
        return classify_binders(
            {"target": _TARGET_DF, "blank": _BLANK_DF},
            threshold_col=THRESHOLD_COL,
            threshold_value=THRESHOLD,
        )

    def test_target_enriched_not_blank_is_orthosteric(self, result):
        """Without an inhibitor condition, enriched target→ orthosteric."""
        assert _binder_type(result, 0, 0) == "orthosteric"

    def test_blank_enriched_is_nonspecific(self, result):
        assert _binder_type(result, 0, 3) == "nonspecific"

    def test_not_enriched_stays_not_enriched(self, result):
        assert _binder_type(result, 0, 4) == "not_enriched"


# ---------------------------------------------------------------------------
# classify_binders — no blank condition
# ---------------------------------------------------------------------------


class TestClassifyBindersNoBlank:
    @pytest.fixture(scope="class")
    def result(self):
        return classify_binders(
            {"target": _TARGET_DF, "target_inhibitor": _INHIBITOR_DF},
        )

    def test_target_only_is_orthosteric(self, result):
        assert _binder_type(result, 0, 0) == "orthosteric"

    def test_target_and_inhibitor_is_allosteric(self, result):
        assert _binder_type(result, 0, 1) == "allosteric"

    def test_inhibitor_only_is_cryptic(self, result):
        assert _binder_type(result, 0, 2) == "cryptic"


# ---------------------------------------------------------------------------
# classify_binders — alternative condition key naming
# ---------------------------------------------------------------------------


class TestClassifyBindersKeyDetection:
    def test_no_protein_key_detected_as_blank(self):
        result = classify_binders(
            {"my_target": _TARGET_DF, "no_protein_run1": _BLANK_DF},
        )
        # (0,3) enriched in blank → nonspecific
        assert _binder_type(result, 0, 3) == "nonspecific"

    def test_control_key_detected_as_blank(self):
        result = classify_binders(
            {"target": _TARGET_DF, "control_beads": _BLANK_DF},
        )
        assert _binder_type(result, 0, 3) == "nonspecific"

    def test_inh_suffix_detected_as_inhibitor(self):
        result = classify_binders(
            {"target": _TARGET_DF, "target_inh": _INHIBITOR_DF},
        )
        assert _binder_type(result, 0, 1) == "allosteric"


# ---------------------------------------------------------------------------
# classify_binders — error handling
# ---------------------------------------------------------------------------


class TestClassifyBindersErrors:
    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            classify_binders({})

    def test_no_target_condition_raises(self):
        with pytest.raises(ValueError, match="No target condition"):
            classify_binders({"blank_1": _BLANK_DF})

    def test_missing_threshold_col_raises(self):
        bad_df = _TARGET_DF.drop(THRESHOLD_COL)
        with pytest.raises(ValueError, match="not found in condition"):
            classify_binders({"target": bad_df, "blank": _BLANK_DF})

    def test_no_code_cols_raises(self):
        no_code = pl.DataFrame({"zscore": [1.0, 2.0]})
        with pytest.raises(ValueError, match="No code_"):
            classify_binders({"target": no_code})


# ---------------------------------------------------------------------------
# classify_bead_artifacts — structure tests
# ---------------------------------------------------------------------------


class TestClassifyBeadArtifactsStructure:
    @pytest.fixture(scope="class")
    def result(self):
        return classify_bead_artifacts(
            {"bead_A": _BEAD_A_DF, "bead_B": _BEAD_B_DF, "bead_C": _BEAD_C_DF},
            threshold_col=THRESHOLD_COL,
            threshold_value=THRESHOLD,
        )

    def test_returns_dataframe(self, result):
        assert isinstance(result, pl.DataFrame)

    def test_has_required_columns(self, result):
        for col in ("code_1", "code_2", "n_enriched", "fraction_enriched", "bead_classification"):
            assert col in result.columns, f"Missing column: {col}"

    def test_n_enriched_is_integer(self, result):
        assert result["n_enriched"].dtype in (pl.Int32, pl.Int64, pl.UInt32)

    def test_fraction_enriched_in_unit_interval(self, result):
        fracs = result["fraction_enriched"].to_list()
        assert all(0.0 <= f <= 1.0 for f in fracs)

    def test_bead_classification_values_valid(self, result):
        valid = {"robust", "bead_artifact_suspect", "not_enriched"}
        for cls in result["bead_classification"].to_list():
            assert cls in valid, f"Unexpected classification: {cls}"


# ---------------------------------------------------------------------------
# classify_bead_artifacts — correctness
# ---------------------------------------------------------------------------


class TestClassifyBeadArtifactsCorrectness:
    @pytest.fixture(scope="class")
    def result(self):
        # Bead A: (1,0)=0.3 (not enr), (1,1)=2.0 (enr)
        # Bead B: (1,0)=2.5 (enr),     (1,1)=1.5 (enr)
        # Bead C: (1,0)=0.2 (not enr), (1,1)=3.0 (enr)
        # (1,0): enriched in 1/3 = 0.33 → bead_artifact_suspect (< 0.5)
        # (1,1): enriched in 3/3 = 1.0  → robust
        return classify_bead_artifacts(
            {"bead_A": _BEAD_A_DF, "bead_B": _BEAD_B_DF, "bead_C": _BEAD_C_DF},
            threshold_col=THRESHOLD_COL,
            threshold_value=THRESHOLD,
            agreement_threshold=0.5,
        )

    def test_robust_compound(self, result):
        """(1,1): enriched in all 3 bead types → robust."""
        assert _bead_class(result, 1, 1) == "robust"

    def test_bead_artifact_compound(self, result):
        """(1,0): enriched in only 1/3 bead types → bead_artifact_suspect."""
        assert _bead_class(result, 1, 0) == "bead_artifact_suspect"

    def test_n_enriched_robust(self, result):
        row = result.filter((pl.col("code_1") == 1) & (pl.col("code_2") == 1))
        assert row["n_enriched"][0] == 3

    def test_n_enriched_artifact(self, result):
        row = result.filter((pl.col("code_1") == 1) & (pl.col("code_2") == 0))
        assert row["n_enriched"][0] == 1

    def test_fraction_enriched_robust(self, result):
        row = result.filter((pl.col("code_1") == 1) & (pl.col("code_2") == 1))
        assert abs(row["fraction_enriched"][0] - 1.0) < 1e-9

    def test_fraction_enriched_artifact(self, result):
        row = result.filter((pl.col("code_1") == 1) & (pl.col("code_2") == 0))
        assert abs(row["fraction_enriched"][0] - 1 / 3) < 1e-6


class TestClassifyBeadArtifactsNotEnriched:
    def test_not_enriched_compound(self):
        bead_a = _df([(2, 0)], [0.1])
        bead_b = _df([(2, 0)], [0.2])
        result = classify_bead_artifacts(
            {"A": bead_a, "B": bead_b},
            threshold_col=THRESHOLD_COL,
            threshold_value=THRESHOLD,
        )
        assert result["bead_classification"][0] == "not_enriched"
        assert result["n_enriched"][0] == 0

    def test_agreement_threshold_effect(self):
        """Higher agreement_threshold makes fewer compounds 'robust'."""
        bead_a = _df([(2, 0)], [2.0])  # enriched
        bead_b = _df([(2, 0)], [0.1])  # not enriched

        result_low = classify_bead_artifacts(
            {"A": bead_a, "B": bead_b},
            threshold_col=THRESHOLD_COL,
            threshold_value=THRESHOLD,
            agreement_threshold=0.5,  # 1/2 = 0.5 >= 0.5 → robust
        )
        result_high = classify_bead_artifacts(
            {"A": bead_a, "B": bead_b},
            threshold_col=THRESHOLD_COL,
            threshold_value=THRESHOLD,
            agreement_threshold=0.9,  # 1/2 = 0.5 < 0.9 → bead_artifact_suspect
        )

        assert result_low["bead_classification"][0] == "robust"
        assert result_high["bead_classification"][0] == "bead_artifact_suspect"


# ---------------------------------------------------------------------------
# classify_bead_artifacts — error handling
# ---------------------------------------------------------------------------


class TestClassifyBeadArtifactsErrors:
    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            classify_bead_artifacts({})

    def test_missing_threshold_col_raises(self):
        bad_df = _BEAD_A_DF.drop(THRESHOLD_COL)
        with pytest.raises(ValueError, match="not found for bead type"):
            classify_bead_artifacts({"bead_A": bad_df, "bead_B": _BEAD_B_DF})

    def test_no_code_cols_raises(self):
        no_code = pl.DataFrame({"zscore": [1.0]})
        with pytest.raises(ValueError, match="No code_"):
            classify_bead_artifacts({"A": no_code})


# ---------------------------------------------------------------------------
# classify_binders — integration: using poisson_ml_enrichment as threshold_col
# ---------------------------------------------------------------------------


class TestClassifyBindersAlternativeThresholdCol:
    def test_poisson_ml_col(self):
        target_df = pl.DataFrame(
            {"code_1": [0, 1], "code_2": [0, 0], "poisson_ml_enrichment": [5.0, 0.8]}
        )
        blank_df = pl.DataFrame(
            {"code_1": [0, 1], "code_2": [0, 0], "poisson_ml_enrichment": [0.9, 0.7]}
        )
        result = classify_binders(
            {"target": target_df, "blank": blank_df},
            threshold_col="poisson_ml_enrichment",
            threshold_value=1.0,
        )
        assert _binder_type(result, 0, 0) == "orthosteric"
        assert _binder_type(result, 1, 0) == "not_enriched"
