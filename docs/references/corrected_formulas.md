# Reference: CORRECTED Enrichment Calculation Methods

Source: Kuai, O'Keeffe & Arico-Muendel (2018). SLAS Discovery 23(5), 405-416.

## FORMULA CORRECTION: Poisson Confidence Interval

The approximation previously documented was INACCURATE at low counts (k < 10),
which is exactly where DEL data needs accuracy. Use the EXACT chi-squared method.

### WRONG (approximate — do NOT use for implementation):
```
lower = (k + 1) * (1 - 1/sqrt(k+1))^2
upper = (k + 1) * (1 + 1/sqrt(k+1))^2
```
Error at k=5: lower off by 0.48, upper off by 0.23 vs exact.

### CORRECT (exact chi-squared — USE THIS):
```python
from scipy.stats import chi2

def poisson_ci(k: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Exact Poisson confidence interval using chi-squared quantiles.
    
    Args:
        k: observed counts (numpy array of non-negative integers)
        alpha: significance level (0.05 for 95% CI)
    
    Returns:
        (lower, upper) numpy arrays
    """
    lower = np.where(k == 0, 0.0, chi2.ppf(alpha / 2, 2 * k) / 2)
    upper = chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return lower, upper
```

### Verified test values (exact method):
| k | lower | upper |
|---|---|---|
| 0 | 0.00 | 3.69 |
| 1 | 0.03 | 5.57 |
| 5 | 1.62 | 11.67 |
| 10 | 4.80 | 18.39 |
| 50 | 37.11 | 65.92 |
| 100 | 81.36 | 121.63 |

## Enrichment with Poisson CI (Method B — CORRECT)

```
enrichment = count * (diversity / total_reads)
enrichment_lower = poisson_ci_lower(count) * (diversity / total_reads)
enrichment_upper = poisson_ci_upper(count) * (diversity / total_reads)
```

## Other Formulas (verified correct)

### Normalized Z-Score (Faver et al. 2019, Eq. 2) — CORRECT
```
p_i = 1 / diversity
p_o = count / total_reads
z_n = (p_o - p_i) / sqrt(p_i * (1 - p_i))
```
Verified: count=100, total=1000, diversity=100 → z_n = 0.9045 ✓

### Poisson ML Enrichment (Hou et al. 2023) — CORRECT
```
ML_enrichment = (control_total / post_total) * ((post_count + 3/8) / (control_count + 3/8))
```
Verified: zero control case returns finite value (267.67) ✓

## SUPPORT SCORE THRESHOLD WARNING

Do NOT use z_n >= 1.0 as a binary enrichment threshold for the multi-level
support score. The z-score at monosynthon level is inherently small because
each BB's count is diluted by contributions from many non-binding compound
partners. With 612 BBs and 5M reads, the strongest enriched BB reaches
z_n ≈ 0.31 (8.6-fold enriched). z_n >= 1.0 would require 30-fold enrichment.

Use fold_enrichment >= 2.0 instead (or top-10% percentile) for support scoring.
The z-score remains the correct metric for RANKING within a single level.

Implementation in rank.py:
- Default: threshold_col="fold_enrichment", threshold_value=2.0
- Adaptive ("auto"): use fold_enrichment >= 2.0 if available; otherwise
  top-90th-percentile of best available score column
- NEVER use threshold_col="zscore", threshold_value=1.0 for support scoring

---

### Agresti-Coull CI (Faver et al. 2019, Eq. 3) — CORRECT
```python
z_alpha = scipy.stats.norm.ppf(1 - alpha/2)  # 1.96 for 95% CI
n_prime = total_reads + z_alpha**2
p_adj = (count + z_alpha**2 / 2) / n_prime
margin = z_alpha * sqrt(p_adj * (1 - p_adj) / n_prime)
ci_lower = max(0, p_adj - margin)
ci_upper = min(1, p_adj + margin)
```
Verified: count=100, total=1000 → CI = [0.0828, 0.1202], brackets true p=0.1 ✓
