"""Tests for analyse/poisson.py.

All numerical expectations come directly from the verified table in
docs/references/corrected_formulas.md.

Verified Poisson CI table (exact chi-squared method, alpha=0.05)
-----------------------------------------------------------------
  k=0   → [0.00,  3.69]
  k=1   → [0.03,  5.57]
  k=5   → [1.62, 11.67]
  k=10  → [4.80, 18.39]
  k=50  → [37.11, 65.92]
  k=100 → [81.36, 121.63]

Verified ML enrichment (post_total=control_total=1000)
------------------------------------------------------
  post=100, control=100 → 1.0
  post=100, control=0   → 267.67  (finite, not inf)
  post=0,   control=100 → ~0.00374
"""

import numpy as np
import pytest

from delexplore.analyse.poisson import (
    enrichment_with_ci,
    poisson_ci,
    poisson_ml_enrichment,
)

# Tolerances
ATOL_CI = 0.01    # for Poisson CI bounds (table values rounded to 2 dp)
ATOL_ML = 0.01    # for ML enrichment


# ---------------------------------------------------------------------------
# poisson_ci — verified values from corrected_formulas.md
# ---------------------------------------------------------------------------


class TestPoissonCI:
    # ── Full verified table ───────────────────────────────────────────────

    @pytest.mark.parametrize("k, exp_lo, exp_hi", [
        (0,   0.00,  3.69),
        (1,   0.03,  5.57),
        (5,   1.62, 11.67),
        (10,  4.80, 18.39),
        (50, 37.11, 65.92),
        (100, 81.36, 121.63),
    ])
    def test_verified_table(self, k, exp_lo, exp_hi):
        lo, hi = poisson_ci(np.array([float(k)]))
        assert abs(lo[0] - exp_lo) < ATOL_CI, (
            f"k={k}: lower={lo[0]:.4f}, expected {exp_lo}"
        )
        assert abs(hi[0] - exp_hi) < ATOL_CI, (
            f"k={k}: upper={hi[0]:.4f}, expected {exp_hi}"
        )

    # ── k=0 edge case ─────────────────────────────────────────────────────

    def test_k0_lower_is_exactly_zero(self):
        lo, _ = poisson_ci(np.array([0.0]))
        assert lo[0] == 0.0

    def test_k0_upper_is_finite(self):
        _, hi = poisson_ci(np.array([0.0]))
        assert np.isfinite(hi[0])

    # ── Properties ────────────────────────────────────────────────────────

    def test_lower_always_le_upper(self):
        k = np.arange(101, dtype=float)
        lo, hi = poisson_ci(k)
        assert np.all(lo <= hi)

    def test_lower_nonnegative(self):
        k = np.arange(101, dtype=float)
        lo, _ = poisson_ci(k)
        assert np.all(lo >= 0.0)

    def test_k_inside_its_own_ci(self):
        """The observed count k should lie inside its own 95 % CI."""
        k = np.arange(1, 101, dtype=float)
        lo, hi = poisson_ci(k)
        assert np.all(lo <= k)
        assert np.all(k <= hi)

    def test_ci_widens_with_k(self):
        """Wider CI for larger k (absolute width grows even though relative shrinks)."""
        k = np.array([1.0, 10.0, 100.0])
        lo, hi = poisson_ci(k)
        widths = hi - lo
        assert np.all(np.diff(widths) > 0)

    def test_tighter_ci_for_smaller_alpha(self):
        """99 % CI (alpha=0.01) is wider than 95 % CI (alpha=0.05)."""
        k = np.array([20.0])
        lo_95, hi_95 = poisson_ci(k, alpha=0.05)
        lo_99, hi_99 = poisson_ci(k, alpha=0.01)
        assert (hi_99[0] - lo_99[0]) > (hi_95[0] - lo_95[0])

    # ── Vectorised ────────────────────────────────────────────────────────

    def test_vectorised_shape(self):
        k = np.arange(1000, dtype=float)
        lo, hi = poisson_ci(k)
        assert lo.shape == (1000,)
        assert hi.shape == (1000,)

    def test_vectorised_all_finite(self):
        k = np.arange(1000, dtype=float)
        lo, hi = poisson_ci(k)
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))

    # ── Input validation ──────────────────────────────────────────────────

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            poisson_ci(np.array([5.0]), alpha=1.5)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            poisson_ci(np.array([5.0]), alpha=0.0)


# ---------------------------------------------------------------------------
# poisson_ml_enrichment — verified values from corrected_formulas.md
# ---------------------------------------------------------------------------


class TestPoissonMLEnrichment:
    # ── Verified anchors ──────────────────────────────────────────────────

    def test_equal_counts_gives_one(self):
        """post=100, control=100 (same totals) → enrichment ≈ 1.0"""
        ml = poisson_ml_enrichment(
            np.array([100.0]), np.array([100.0]),
            post_total=1000, control_total=1000,
        )
        assert abs(ml[0] - 1.0) < ATOL_ML

    def test_zero_control_gives_finite_value(self):
        """post=100, control=0 → 267.67 (finite — continuity correction prevents ∞)"""
        ml = poisson_ml_enrichment(
            np.array([100.0]), np.array([0.0]),
            post_total=1000, control_total=1000,
        )
        assert np.isfinite(ml[0])
        assert abs(ml[0] - 267.67) < 0.1

    def test_zero_post_gives_small_finite_value(self):
        """post=0, control=100 → small but finite (not zero)"""
        ml = poisson_ml_enrichment(
            np.array([0.0]), np.array([100.0]),
            post_total=1000, control_total=1000,
        )
        assert np.isfinite(ml[0])
        assert ml[0] > 0.0
        assert ml[0] < 0.01   # approximately 0.00374

    # ── Properties ────────────────────────────────────────────────────────

    def test_enrichment_gt_one_when_post_gt_control(self):
        ml = poisson_ml_enrichment(
            np.array([200.0]), np.array([10.0]),
            post_total=1000, control_total=1000,
        )
        assert ml[0] > 1.0

    def test_enrichment_lt_one_when_post_lt_control(self):
        ml = poisson_ml_enrichment(
            np.array([5.0]), np.array([100.0]),
            post_total=1000, control_total=1000,
        )
        assert ml[0] < 1.0

    def test_total_ratio_scales_result(self):
        """Doubling control_total doubles the enrichment."""
        ml_1x = poisson_ml_enrichment(
            np.array([50.0]), np.array([25.0]),
            post_total=1000, control_total=1000,
        )
        ml_2x = poisson_ml_enrichment(
            np.array([50.0]), np.array([25.0]),
            post_total=1000, control_total=2000,
        )
        assert abs(ml_2x[0] - 2 * ml_1x[0]) < ATOL_ML

    def test_correction_zero_fails_on_zero_control(self):
        """With correction=0 and control=0, result should be inf (division by zero).

        This confirms that the default correction=0.375 is what keeps the result
        finite — tested explicitly in test_zero_control_gives_finite_value.
        """
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            ml = poisson_ml_enrichment(
                np.array([10.0]), np.array([0.0]),
                post_total=1000, control_total=1000,
                correction=0.0,
            )
        assert not np.isfinite(ml[0])

    # ── Vectorised ────────────────────────────────────────────────────────

    def test_vectorised_1000_elements(self):
        post = np.random.default_rng(42).integers(0, 500, size=1000).astype(float)
        ctrl = np.random.default_rng(7).integers(0, 500, size=1000).astype(float)
        ml = poisson_ml_enrichment(post, ctrl, post_total=50_000, control_total=50_000)
        assert ml.shape == (1000,)
        assert np.all(np.isfinite(ml))
        assert np.all(ml > 0)

    # ── Input validation ──────────────────────────────────────────────────

    def test_zero_post_total_raises(self):
        with pytest.raises(ValueError, match="post_total"):
            poisson_ml_enrichment(
                np.array([10.0]), np.array([5.0]),
                post_total=0, control_total=1000,
            )

    def test_zero_control_total_raises(self):
        with pytest.raises(ValueError, match="control_total"):
            poisson_ml_enrichment(
                np.array([10.0]), np.array([5.0]),
                post_total=1000, control_total=0,
            )

    def test_negative_correction_raises(self):
        with pytest.raises(ValueError, match="correction"):
            poisson_ml_enrichment(
                np.array([10.0]), np.array([5.0]),
                post_total=1000, control_total=1000,
                correction=-0.1,
            )


# ---------------------------------------------------------------------------
# enrichment_with_ci
# ---------------------------------------------------------------------------


class TestEnrichmentWithCI:
    def test_returns_three_arrays(self):
        e, lo, hi = enrichment_with_ci(
            np.array([100.0]), total_reads=1000, diversity=100
        )
        assert e.shape == lo.shape == hi.shape == (1,)

    def test_enrichment_at_expected_is_one(self):
        """count = total/diversity → enrichment = 1.0"""
        # diversity=100, total=1000 → expected count = 10
        e, _, _ = enrichment_with_ci(
            np.array([10.0]), total_reads=1000, diversity=100
        )
        assert abs(e[0] - 1.0) < 1e-9

    def test_ci_brackets_enrichment(self):
        """lower_ci <= enrichment <= upper_ci for all positive counts."""
        counts = np.arange(1, 101, dtype=float)
        e, lo, hi = enrichment_with_ci(counts, total_reads=1000, diversity=100)
        assert np.all(lo <= e + 1e-12)
        assert np.all(e <= hi + 1e-12)

    def test_zero_count_lower_ci_is_zero(self):
        _, lo, _ = enrichment_with_ci(
            np.array([0.0]), total_reads=1000, diversity=100
        )
        assert lo[0] == 0.0

    def test_zero_count_enrichment_is_zero(self):
        e, _, _ = enrichment_with_ci(
            np.array([0.0]), total_reads=1000, diversity=100
        )
        assert e[0] == 0.0

    def test_enrichment_scales_with_diversity(self):
        """Same count but higher diversity → higher enrichment fold."""
        e_lo, _, _ = enrichment_with_ci(
            np.array([10.0]), total_reads=1000, diversity=100
        )
        e_hi, _, _ = enrichment_with_ci(
            np.array([10.0]), total_reads=1000, diversity=1000
        )
        assert e_hi[0] > e_lo[0]

    def test_vectorised(self):
        counts = np.arange(101, dtype=float)
        e, lo, hi = enrichment_with_ci(counts, total_reads=5000, diversity=500)
        assert e.shape == (101,)
        assert np.all(np.isfinite(e))
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))

    # ── Input validation ──────────────────────────────────────────────────

    def test_zero_total_reads_raises(self):
        with pytest.raises(ValueError, match="total_reads"):
            enrichment_with_ci(np.array([10.0]), total_reads=0, diversity=100)

    def test_zero_diversity_raises(self):
        with pytest.raises(ValueError, match="diversity"):
            enrichment_with_ci(np.array([10.0]), total_reads=1000, diversity=0)
