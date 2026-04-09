"""Poisson confidence interval and ML enrichment (Kuai 2018, Hou 2023).

All formulas are taken verbatim from docs/references/corrected_formulas.md.
The approximate CI formula (documented there as WRONG) is intentionally
not implemented; only the exact chi-squared method is used.

Key references
--------------
  Kuai, O'Keeffe & Arico-Muendel (2018). SLAS Discovery 23(5), 405-416.
  Hou, Xie, Gui, Li & Li (2023). ACS Omega 8, 19057-19071.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def poisson_ci(
    k: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Poisson confidence interval using chi-squared quantiles.

    This is the CORRECT implementation. The approximation
    ``(k+1)*(1 ± 1/sqrt(k+1))^2`` is NOT used — see corrected_formulas.md
    for the error analysis.

    Formula (exact)::

        lower = 0                              when k == 0
              = chi2.ppf(alpha/2,  2*k) / 2   otherwise
        upper = chi2.ppf(1-alpha/2, 2*(k+1)) / 2

    Verified values (corrected_formulas.md)::

        k=0   → [0.00,  3.69]
        k=1   → [0.03,  5.57]
        k=5   → [1.62, 11.67]
        k=10  → [4.80, 18.39]
        k=50  → [37.11, 65.92]
        k=100 → [81.36, 121.63]

    Args:
        k: Observed counts (non-negative integers as float or int array).
        alpha: Significance level.  0.05 gives a 95 % CI.

    Returns:
        ``(lower, upper)`` — two arrays of Poisson rate bounds, same shape
        as *k*.

    Raises:
        ValueError: If alpha is not in (0, 1).
    """
    k = np.asarray(k, dtype=float)

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    lower = np.where(k == 0, 0.0, chi2.ppf(alpha / 2, 2 * k) / 2)
    upper = chi2.ppf(1.0 - alpha / 2, 2 * (k + 1)) / 2
    return lower, upper


def poisson_ml_enrichment(
    post_counts: np.ndarray,
    control_counts: np.ndarray,
    post_total: int,
    control_total: int,
    correction: float = 0.375,
) -> np.ndarray:
    """Maximum-likelihood enrichment fold from the Poisson ratio test.

    Prevents division by zero when control count is 0 via the 3/8
    continuity correction (Hou et al. 2023).

    Formula::

        ML = (control_total / post_total) * ((post + 3/8) / (control + 3/8))

    Verified values (corrected_formulas.md, post_total=control_total=1000)::

        post=100, control=100 → 1.0
        post=100, control=0   → 267.67   (finite — no ZeroDivisionError)
        post=0,   control=100 → 0.00374  (small but finite)

    Args:
        post_counts: Observed counts in the selection condition (k1).
        control_counts: Observed counts in the blank/control (k2).
        post_total: Total decoded reads in the selection condition (n1).
        control_total: Total decoded reads in the blank/control (n2).
        correction: Continuity correction term (default 3/8 = 0.375).

    Returns:
        Array of ML enrichment fold values, same shape as *post_counts*.

    Raises:
        ValueError: If post_total or control_total is <= 0, or correction < 0.
    """
    post_counts = np.asarray(post_counts, dtype=float)
    control_counts = np.asarray(control_counts, dtype=float)

    if post_total <= 0:
        raise ValueError(f"post_total must be > 0, got {post_total}")
    if control_total <= 0:
        raise ValueError(f"control_total must be > 0, got {control_total}")
    if correction < 0:
        raise ValueError(f"correction must be >= 0, got {correction}")

    ratio = control_total / post_total
    ml = ratio * (post_counts + correction) / (control_counts + correction)
    return ml


def enrichment_with_ci(
    counts: np.ndarray,
    total_reads: int,
    diversity: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kuai et al. Method B enrichment fold with exact Poisson CI.

    Formula (corrected_formulas.md)::

        enrichment = counts * (diversity / total_reads)
        lower_ci   = poisson_lower(counts) * (diversity / total_reads)
        upper_ci   = poisson_upper(counts) * (diversity / total_reads)

    The enrichment fold is 1.0 for a compound observed at its expected
    frequency; > 1.0 for enriched compounds.

    Args:
        counts: Observed counts per feature.
        total_reads: Total decoded reads in this condition.
        diversity: Number of distinct features at this synthon level.
        alpha: Significance level for the Poisson CI.

    Returns:
        ``(enrichment, lower_ci, upper_ci)`` — three arrays of the same
        shape as *counts*.

    Raises:
        ValueError: If total_reads <= 0, diversity < 1, or alpha invalid.
    """
    counts = np.asarray(counts, dtype=float)

    if total_reads <= 0:
        raise ValueError(f"total_reads must be > 0, got {total_reads}")
    if diversity < 1:
        raise ValueError(f"diversity must be >= 1, got {diversity}")

    scale = diversity / total_reads
    enrichment = counts * scale

    lower_raw, upper_raw = poisson_ci(counts, alpha=alpha)
    lower_ci = lower_raw * scale
    upper_ci = upper_raw * scale

    return enrichment, lower_ci, upper_ci
