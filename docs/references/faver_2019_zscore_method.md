# Reference: Normalized Z-Score Enrichment Metric

Source: Faver, Riehle, Lancia et al. (2019). "Quantitative Comparison of Enrichment
from DNA-Encoded Chemical Library Selections."
ACS Combinatorial Science 21, 75-82.

## Key Finding for Implementation

The normalized z-score is the most interpretable enrichment metric for DEL data.
It scales linearly with fold enrichment, has computable uncertainties, and works
across mono-, di-, and trisynthon features on the same scale.

## Critical Formulas

### Standard Z-Score (Eq. 1 — baseline, but sampling-dependent)

```
z = (C_observed - C_expected) / sqrt(n * p_i * (1 - p_i))
  = (C_observed - n * p_i) / sqrt(n * p_i * (1 - p_i))
```

Where:
- C_observed = observed count for this feature
- n = total decoded reads
- p_i = expected population fraction = 1 / library_diversity_for_this_feature

### Normalized Z-Score (Eq. 2 — RECOMMENDED, sampling-independent)

```
z_n = (p_observed - p_i) / sqrt(p_i * (1 - p_i))
    = sqrt(p_i / (1 - p_i)) * (p_observed / p_i - 1)
```

Where:
- p_observed = C_observed / n
- p_i = expected fraction (1/diversity for that feature level)

This removes sampling dependence. z_n >= 1 indicates significant enrichment.

### Interpretation of z_n

For a 3-cycle library with 1000 building blocks per cycle:
- z_n = 1 for a monosynthon ≈ 30-fold enrichment
- z_n = 1 for a disynthon ≈ 1000-fold enrichment
- z_n = 1 for a trisynthon ≈ 30,000-fold enrichment

This scaling means monosynthon and trisynthon features CAN be plotted together.

### Agresti-Coull Confidence Interval (Eq. 3 — for uncertainty quantification)

```python
# Implementation pseudocode
z_alpha = 1.96  # for 95% CI
n_prime = n + z_alpha**2
p_observed_adj = (C_observed + z_alpha**2 / 2) / n_prime

CI = p_observed_adj ± z_alpha * sqrt(p_observed_adj * (1 - p_observed_adj) / n_prime)
```

Then convert the CI on p_observed to CI on z_n using the z_n formula above.

## Implementation Notes

- The Agresti-Coull interval is preferred over the Wald interval because it gives
  conservative estimates even at extreme values of p (near 0 or 1)
- Uncertainty decreases with: more reads (larger n) AND higher expected population
  (larger p_i, i.e., lower diversity features)
- For trisynthons (high diversity, low p_i), uncertainties are always large —
  this is inherent, not a bug
- z_n = 1 is a reasonable default threshold for "significant enrichment" but may
  need adjustment per library/protocol

## Function Signature Suggestion

```python
def calculate_zscore_enrichment(
    counts: np.ndarray,          # observed counts per compound
    total_reads: int,            # total decoded reads in this condition
    diversity: int,              # library diversity for this feature level
    alpha: float = 0.05,         # significance level for CI
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (z_normalized, ci_lower, ci_upper) arrays."""
```
