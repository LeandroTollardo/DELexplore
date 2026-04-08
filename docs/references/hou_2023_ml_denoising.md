# Reference: ML-Based Denoising and Poisson Ratio Test for Cell-Based DEL

Source: Hou, Xie, Gui, Li & Li (2023). "Machine-Learning-Based Data Analysis
Method for Cell-Based Selection of DNA-Encoded Libraries."
ACS Omega 8, 19057-19071.

## Key Finding for Implementation

Cell-based DEL selections are far noisier than purified-protein selections.
Traditional enrichment metrics produce many false positives. A DNN with MAP
(Maximum A Posteriori) loss function can effectively denoise this data.

## Critical Formulas

### Maximum-Likelihood Enrichment Fold (prevents zero-division)

Standard enrichment = k1/n1 / (k2/n2) fails when k2 = 0.

Maximum-likelihood enrichment from the Poisson ratio test:

```
ML_enrichment = (n2/n1) * ((k1 + 3/8) / (k2 + 3/8))
```

Where:
- k1 = post-selection count for this compound
- k2 = blank/control count for this compound
- n1 = total reads in post-selection
- n2 = total reads in blank/control

The 3/8 continuity correction (from the Poisson ratio test) prevents division
by zero and reduces bias at low counts.

### Normalized Fold-Change (Gerry et al. method, also used)

```
F_n = lambda_lower(post_selection) / lambda_upper(blank)
```

Where lambda_lower and lambda_upper are the lower and upper bounds of the
95% Poisson CI. This gives a conservative enrichment estimate.

### MAP Loss Function for DNN Training

The MAP enrichment with L1 regularization:

```
MAP_enrichment(alpha) = argmax over R of [log P(k1,k2 | R) - alpha * |R|]
```

Where alpha controls the regularization strength:
- Large alpha → conservative (fewer false positives, more false negatives)
- Small alpha → permissive (more false positives, fewer false negatives)

## Implementation Notes

- For cell-based selections: use MAP enrichment with alpha tuning
- For purified protein selections: standard Poisson ratio test is sufficient
- Always merge replicate counts before computing enrichment (sum counts, not
  average enrichments) — merged data follows Poisson distribution
- The MAP DNN uses Extended Connectivity Fingerprints (ECFPs) as input
- Hyperparameter optimization: Bayesian optimization (pyGPGO package)
- Early stopping to prevent overfitting
- Dataset split: 80% train, 10% validation, 10% test

## Function Signature Suggestion

```python
def calculate_poisson_ml_enrichment(
    post_counts: np.ndarray,     # counts in selection condition
    control_counts: np.ndarray,  # counts in blank/control
    post_total: int,             # total reads in selection
    control_total: int,          # total reads in control
    correction: float = 0.375,   # continuity correction (3/8)
) -> np.ndarray:
    """Returns maximum-likelihood enrichment fold per compound."""
```

## Key Table: Noise Characteristics by Selection Type

| Selection Type | Avg Enrichment | Max Enrichment | Noise Level |
|---|---|---|---|
| Purified protein | ~1.0 (control) | 2000-3500 | Low |
| Cell surface | ~1.7-1.8 | 150-340 | High |
| Blank beads | ~1.0 | ~150 | Baseline |

Cell-based selections show 10-20x lower dynamic range than purified protein.
