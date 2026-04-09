"""Normalized z-score enrichment metric (Faver et al. 2019).

All formulas are taken verbatim from docs/references/corrected_formulas.md
and docs/references/faver_2019_zscore_method.md, which carry verified
numerical test values.

Key property: z_n is sampling-independent, so monosynthon and trisynthon
features land on the same scale and can be plotted together.

  z_n >= 1  ≈ 30-fold enrichment at monosynthon level
  z_n >= 1  ≈ 1 000-fold enrichment at disynthon level
  z_n >= 1  ≈ 30 000-fold enrichment at trisynthon level
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def calculate_zscore(
    counts: np.ndarray,
    total_reads: int,
    diversity: int,
) -> np.ndarray:
    """Compute the sampling-independent normalized z-score (Faver 2019, Eq. 2).

    Formula::

        p_i  = 1 / diversity
        p_o  = counts / total_reads
        z_n  = (p_o - p_i) / sqrt(p_i * (1 - p_i))

    Args:
        counts: Observed counts per feature (non-negative integer array).
        total_reads: Total decoded reads in this selection/condition.
        diversity: Number of distinct features at this synthon level
            (e.g. number of monosynthons for a position, or full library size
            for trisynthons).

    Returns:
        Array of z_n values, same shape as *counts*.
        Returns an array of zeros when diversity == 1 (denominator is zero).

    Raises:
        ValueError: If total_reads <= 0 or diversity < 1.
    """
    counts = np.asarray(counts, dtype=float)

    if total_reads <= 0:
        raise ValueError(f"total_reads must be > 0, got {total_reads}")
    if diversity < 1:
        raise ValueError(f"diversity must be >= 1, got {diversity}")

    # Edge case: only one possible feature — z-score is undefined (denominator 0)
    if diversity == 1:
        return np.zeros_like(counts)

    p_i = 1.0 / diversity
    p_o = counts / total_reads
    z_n = (p_o - p_i) / np.sqrt(p_i * (1.0 - p_i))
    return z_n


def calculate_agresti_coull_ci(
    counts: np.ndarray,
    total_reads: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Agresti-Coull confidence interval on the observed proportion.

    Preferred over Wald interval at extreme values of p (near 0 or 1),
    which is the typical regime for DEL data (Faver 2019, Eq. 3).

    Formula::

        z_alpha  = norm.ppf(1 - alpha/2)          # 1.96 for 95 % CI
        n_prime  = total_reads + z_alpha**2
        p_adj    = (counts + z_alpha**2 / 2) / n_prime
        margin   = z_alpha * sqrt(p_adj * (1 - p_adj) / n_prime)
        ci_lower = max(0, p_adj - margin)
        ci_upper = min(1, p_adj + margin)

    Verified: counts=100, total=1000 → CI = [0.0828, 0.1202] (corrected_formulas.md).

    Args:
        counts: Observed counts per feature.
        total_reads: Total reads in this condition.
        alpha: Significance level.  0.05 gives a 95 % CI.

    Returns:
        ``(ci_lower, ci_upper)`` — arrays of proportions, same shape as *counts*.

    Raises:
        ValueError: If total_reads <= 0 or alpha not in (0, 1).
    """
    counts = np.asarray(counts, dtype=float)

    if total_reads <= 0:
        raise ValueError(f"total_reads must be > 0, got {total_reads}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    z_alpha = norm.ppf(1.0 - alpha / 2.0)        # 1.959964 for alpha=0.05
    n_prime = total_reads + z_alpha**2
    p_adj = (counts + z_alpha**2 / 2.0) / n_prime
    margin = z_alpha * np.sqrt(p_adj * (1.0 - p_adj) / n_prime)

    ci_lower = np.maximum(0.0, p_adj - margin)
    ci_upper = np.minimum(1.0, p_adj + margin)
    return ci_lower, ci_upper


def calculate_mad_zscore(
    counts: np.ndarray,
    total_reads: int,
    diversity: int,
) -> np.ndarray:
    """Compute the MAD-normalized z-score (Wichert et al. 2024, DELi Eq. 3).

    Replaces the theoretical standard deviation with a robust estimate based
    on the Median Absolute Deviation (MAD), making the score resistant to
    outlier compounds that inflate the variance::

        p_i     = counts / total_reads          (observed fractions)
        median  = median(p_i)
        MAD     = 1.4286 × median(|p_i − median|)
        z_mad   = (p_i − median) / MAD

    The factor 1.4286 ≈ 1 / 0.6745 scales the MAD to be a consistent
    estimator of the standard deviation under Gaussian assumptions, matching
    the convention in Wellnitz et al. (DELi) Eq. 3.

    When MAD == 0 (all observed fractions are identical, or there is only one
    feature), the function returns an array of zeros rather than dividing by
    zero.

    Args:
        counts: Observed counts per feature (non-negative integer array).
        total_reads: Total decoded reads in this selection/condition.
        diversity: Number of distinct features at this synthon level.
            Used only for input validation; not needed in the formula itself.

    Returns:
        Array of MAD z-score values, same shape as *counts*.

    Raises:
        ValueError: If total_reads <= 0 or diversity < 1.
    """
    counts = np.asarray(counts, dtype=float)

    if total_reads <= 0:
        raise ValueError(f"total_reads must be > 0, got {total_reads}")
    if diversity < 1:
        raise ValueError(f"diversity must be >= 1, got {diversity}")

    p_i = counts / total_reads
    med = np.median(p_i)
    mad = 1.4286 * np.median(np.abs(p_i - med))

    if mad == 0.0:
        return np.zeros_like(counts)

    return (p_i - med) / mad


def zscore_enrichment(
    counts: np.ndarray,
    total_reads: int,
    diversity: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute z_n and its Agresti-Coull confidence interval.

    Computes Agresti-Coull CI on the observed proportion *p_o*, then maps
    both CI bounds through the z_n formula (which is a monotone linear
    transformation of p_o), yielding a CI on z_n.

    Conversion::

        z_n(p) = (p - p_i) / sqrt(p_i * (1 - p_i))

        ci_lower_z = z_n(ci_lower_p)
        ci_upper_z = z_n(ci_upper_p)

    Args:
        counts: Observed counts per feature.
        total_reads: Total reads in this condition.
        diversity: Number of distinct features at this synthon level.
        alpha: Significance level for the CI (0.05 → 95 % CI).

    Returns:
        ``(z_score, ci_lower, ci_upper)`` — three arrays of the same shape
        as *counts*.  *ci_lower* and *ci_upper* are z_n values, not
        proportions.
    """
    counts = np.asarray(counts, dtype=float)
    z_score = calculate_zscore(counts, total_reads, diversity)

    ci_lower_p, ci_upper_p = calculate_agresti_coull_ci(counts, total_reads, alpha)

    if diversity == 1:
        # z_n is identically zero; CI is also zero
        return z_score, np.zeros_like(counts), np.zeros_like(counts)

    p_i = 1.0 / diversity
    denominator = np.sqrt(p_i * (1.0 - p_i))
    ci_lower_z = (ci_lower_p - p_i) / denominator
    ci_upper_z = (ci_upper_p - p_i) / denominator

    return z_score, ci_lower_z, ci_upper_z
