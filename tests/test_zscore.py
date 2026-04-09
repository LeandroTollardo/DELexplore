"""Tests for analyse/zscore.py.

All numerical expectations come from the verified values in
docs/references/corrected_formulas.md and docs/references/faver_2019_zscore_method.md.

Verified anchors
----------------
  count=100, total=1000, diversity=100  →  z_n = 0.9045
  count=100, total=1000               →  Agresti-Coull CI = [0.0828, 0.1202]
"""

import numpy as np
import pytest

from delexplore.analyse.zscore import (
    calculate_agresti_coull_ci,
    calculate_zscore,
    zscore_enrichment,
)

# Absolute tolerance for numerical comparisons
ATOL = 1e-4


# ---------------------------------------------------------------------------
# calculate_zscore
# ---------------------------------------------------------------------------


class TestCalculateZscore:
    # ── Verified anchor (corrected_formulas.md) ───────────────────────────

    def test_verified_value(self):
        """count=100, total=1000, diversity=100 → z_n ≈ 0.9045"""
        z = calculate_zscore(np.array([100.0]), total_reads=1000, diversity=100)
        assert abs(z[0] - 0.9045) < ATOL

    # ── Directional / conceptual ──────────────────────────────────────────

    def test_expected_count_gives_zero(self):
        """When observed == expected (count/total == 1/diversity), z_n == 0."""
        # p_i = 1/50 = 0.02; expected count for total=1000 → 20
        z = calculate_zscore(np.array([20.0]), total_reads=1000, diversity=50)
        assert abs(z[0]) < ATOL

    def test_count_above_expected_is_positive(self):
        z = calculate_zscore(np.array([100.0]), total_reads=1000, diversity=100)
        assert z[0] > 0

    def test_count_below_expected_is_negative(self):
        # expected = 1000/100 = 10; use count=1
        z = calculate_zscore(np.array([1.0]), total_reads=1000, diversity=100)
        assert z[0] < 0

    def test_zero_count_does_not_crash(self):
        z = calculate_zscore(np.array([0.0]), total_reads=1000, diversity=100)
        assert np.isfinite(z[0])
        assert z[0] < 0  # zero count is below the expected fraction

    def test_diversity_one_returns_zero(self):
        """Denominator is zero when diversity==1; function must return 0."""
        z = calculate_zscore(np.array([500.0]), total_reads=1000, diversity=1)
        assert z[0] == 0.0

    # ── Vectorised ────────────────────────────────────────────────────────

    def test_vectorised_output_shape(self):
        counts = np.arange(101, dtype=float)  # 0 … 100
        z = calculate_zscore(counts, total_reads=1000, diversity=100)
        assert z.shape == counts.shape

    def test_vectorised_all_finite(self):
        counts = np.arange(101, dtype=float)
        z = calculate_zscore(counts, total_reads=1000, diversity=100)
        assert np.all(np.isfinite(z))

    def test_vectorised_monotone_increasing(self):
        """Higher count at fixed total and diversity → higher z_n."""
        counts = np.arange(101, dtype=float)
        z = calculate_zscore(counts, total_reads=10_000, diversity=100)
        assert np.all(np.diff(z) > 0)

    # ── Input validation ──────────────────────────────────────────────────

    def test_zero_total_reads_raises(self):
        with pytest.raises(ValueError, match="total_reads"):
            calculate_zscore(np.array([10.0]), total_reads=0, diversity=100)

    def test_negative_total_reads_raises(self):
        with pytest.raises(ValueError, match="total_reads"):
            calculate_zscore(np.array([10.0]), total_reads=-1, diversity=100)

    def test_zero_diversity_raises(self):
        with pytest.raises(ValueError, match="diversity"):
            calculate_zscore(np.array([10.0]), total_reads=1000, diversity=0)


# ---------------------------------------------------------------------------
# calculate_agresti_coull_ci
# ---------------------------------------------------------------------------


class TestAgrestiCoullCI:
    # ── Verified anchor (corrected_formulas.md) ───────────────────────────

    def test_verified_lower_bound(self):
        """count=100, total=1000 → ci_lower ≈ 0.0828"""
        lo, _ = calculate_agresti_coull_ci(np.array([100.0]), total_reads=1000)
        assert abs(lo[0] - 0.0828) < ATOL

    def test_verified_upper_bound(self):
        """count=100, total=1000 → ci_upper ≈ 0.1202"""
        _, hi = calculate_agresti_coull_ci(np.array([100.0]), total_reads=1000)
        assert abs(hi[0] - 0.1202) < ATOL

    # ── Properties ────────────────────────────────────────────────────────

    def test_lower_always_le_upper(self):
        counts = np.arange(101, dtype=float)
        lo, hi = calculate_agresti_coull_ci(counts, total_reads=1000)
        assert np.all(lo <= hi)

    def test_bounds_in_unit_interval(self):
        counts = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])
        lo, hi = calculate_agresti_coull_ci(counts, total_reads=1000)
        assert np.all(lo >= 0.0)
        assert np.all(hi <= 1.0)

    def test_zero_count_lower_bound_is_zero(self):
        """For count=0 the Agresti-Coull lower bound clips to 0."""
        lo, _ = calculate_agresti_coull_ci(np.array([0.0]), total_reads=1000)
        assert lo[0] == 0.0

    def test_wider_interval_for_larger_alpha(self):
        """Smaller alpha (99 % CI) gives wider interval than 95 %."""
        counts = np.array([50.0])
        lo_95, hi_95 = calculate_agresti_coull_ci(counts, 1000, alpha=0.05)
        lo_99, hi_99 = calculate_agresti_coull_ci(counts, 1000, alpha=0.01)
        assert (hi_99[0] - lo_99[0]) > (hi_95[0] - lo_95[0])

    def test_narrower_interval_for_more_reads(self):
        """More reads → tighter CI at the same observed proportion."""
        counts_low  = np.array([10.0])
        counts_high = np.array([100.0])
        lo_l, hi_l = calculate_agresti_coull_ci(counts_low,  total_reads=100)
        lo_h, hi_h = calculate_agresti_coull_ci(counts_high, total_reads=1000)
        assert (hi_h[0] - lo_h[0]) < (hi_l[0] - lo_l[0])

    def test_vectorised_shape(self):
        counts = np.arange(101, dtype=float)
        lo, hi = calculate_agresti_coull_ci(counts, total_reads=1000)
        assert lo.shape == counts.shape
        assert hi.shape == counts.shape

    # ── Input validation ──────────────────────────────────────────────────

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            calculate_agresti_coull_ci(np.array([10.0]), total_reads=1000, alpha=1.5)

    def test_zero_total_reads_raises(self):
        with pytest.raises(ValueError, match="total_reads"):
            calculate_agresti_coull_ci(np.array([10.0]), total_reads=0)


# ---------------------------------------------------------------------------
# zscore_enrichment
# ---------------------------------------------------------------------------


class TestZscoreEnrichment:
    def test_returns_three_arrays(self):
        z, lo, hi = zscore_enrichment(
            np.array([100.0]), total_reads=1000, diversity=100
        )
        assert z.shape == lo.shape == hi.shape

    def test_verified_z_value(self):
        z, _, _ = zscore_enrichment(
            np.array([100.0]), total_reads=1000, diversity=100
        )
        assert abs(z[0] - 0.9045) < ATOL

    def test_ci_brackets_zscore(self):
        """For every count in 0..100: ci_lower <= z_score <= ci_upper."""
        counts = np.arange(101, dtype=float)
        z, lo, hi = zscore_enrichment(counts, total_reads=1000, diversity=100)
        assert np.all(lo <= z + 1e-10)  # allow floating-point equality
        assert np.all(z <= hi + 1e-10)

    def test_ci_brackets_zscore_enriched_compound(self):
        """Specifically for an enriched compound (count >> expected)."""
        z, lo, hi = zscore_enrichment(
            np.array([300.0]), total_reads=1000, diversity=100
        )
        assert lo[0] < z[0] < hi[0]

    def test_ci_brackets_zscore_depleted_compound(self):
        """For a depleted compound (count < expected)."""
        z, lo, hi = zscore_enrichment(
            np.array([1.0]), total_reads=1000, diversity=100
        )
        assert lo[0] <= z[0] <= hi[0]

    def test_zero_count_does_not_crash(self):
        z, lo, hi = zscore_enrichment(
            np.array([0.0]), total_reads=1000, diversity=100
        )
        assert np.isfinite(z[0])
        assert np.isfinite(lo[0])
        assert np.isfinite(hi[0])

    def test_diversity_one_all_zeros(self):
        z, lo, hi = zscore_enrichment(
            np.array([500.0]), total_reads=1000, diversity=1
        )
        assert z[0] == 0.0
        assert lo[0] == 0.0
        assert hi[0] == 0.0

    def test_vectorised_100_counts(self):
        """Pass 100-element array; all return values must be finite."""
        counts = np.arange(1, 101, dtype=float)
        z, lo, hi = zscore_enrichment(counts, total_reads=10_000, diversity=200)
        assert np.all(np.isfinite(z))
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))
        assert z.shape == (100,)

    def test_ci_width_decreases_with_more_reads(self):
        """At the same enrichment ratio, more reads → narrower z CI."""
        count_lo, total_lo = np.array([10.0]),  1_000
        count_hi, total_hi = np.array([100.0]), 10_000
        _, lo_l, hi_l = zscore_enrichment(count_lo, total_lo, diversity=100)
        _, lo_h, hi_h = zscore_enrichment(count_hi, total_hi, diversity=100)
        width_lo = hi_l[0] - lo_l[0]
        width_hi = hi_h[0] - lo_h[0]
        assert width_hi < width_lo
