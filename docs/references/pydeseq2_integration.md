# Reference: PyDESeq2 Integration for DEL Count Data

Source: Muzellec et al. (2023). "PyDESeq2: a python package for bulk RNA-seq
differential expression analysis." Bioinformatics 39(9), btad547.
Source: Love et al. (2014). "Moderated estimation of fold change and dispersion
for RNA-seq data with DESeq2." Genome Biology 15, 550.
Repository: https://github.com/scverse/PyDESeq2 (maintained by scverse since Dec 2025)

---

## Why PyDESeq2 Instead of edgeR

1. **Pure Python** — no R, no Bioconductor, no rpy2 bridge, pip-installable
2. **Same statistical family** — negative binomial GLM with dispersion shrinkage
   (DESeq2 and edgeR both model overdispersed count data with NB distributions)
3. **Part of scverse ecosystem** — same community as scanpy, anndata, squidpy
4. **MIT licensed**, validated against R DESeq2 on 8 TCGA datasets with near-identical results
5. **Eliminates entire dependency class** — no R installation, no BiocManager,
   no conda-only R packages, no Rscript generation

## What PyDESeq2 Does (Statistical Model)

PyDESeq2 implements the DESeq2 methodology:

1. **Size factor estimation** — median-of-ratios normalization
   - Corrects for differences in total sequencing depth between selections
   - Assumption: most features (compounds) are NOT differentially enriched
   - This assumption HOLDS for DEL data (most compounds don't bind)

2. **Dispersion estimation** — per-feature dispersion with shrinkage
   - Fits NB GLM independently per feature to get raw dispersion
   - Fits a trend curve: dispersion as a function of mean count
   - Shrinks individual dispersions toward the trend (empirical Bayes)
   - This stabilizes variance estimates for features with few replicates

3. **Log-fold-change estimation** — with optional shrinkage
   - Fits the specified design formula to get log2 fold changes
   - LFC shrinkage (apeglm method) reduces noise for low-count features
   - Prevents large estimated fold changes driven by noise alone

4. **Wald test** — for statistical significance
   - Tests whether the LFC is significantly different from zero
   - Returns p-values per feature
   - Benjamini-Hochberg correction for multiple testing → padj (FDR)

5. **Outlier detection** — Cook's distance
   - Flags features with individual observations that have extreme influence
   - Optionally replaces outlier counts with trimmed mean

## Mapping DEL Concepts to PyDESeq2

| DEL concept | PyDESeq2 / RNA-seq concept |
|---|---|
| Compound (or synthon feature) | Gene |
| Selection condition instance | Sample |
| Sequence count per compound per selection | Read count per gene per sample |
| "protein" vs "no_protein" group | Condition (treated vs control) |
| Selection replicate | Biological replicate |
| Bead type (HisPURE vs Dynabeads) | Batch variable |
| Target protein identity | Additional factor |

## Input Data Preparation

PyDESeq2 requires two pandas DataFrames:

### Counts matrix
- **Rows:** selection instances (e.g., JPAG_2025_1, JPAG_2025_2, ...)
- **Columns:** compound IDs (e.g., "47_543", "278_453", ...)
- **Values:** integer counts
- **Missing values:** fill with 0 (compound not observed = 0 count)

### Metadata
- **Index:** selection names (must match counts row names exactly)
- **Columns:** condition variables (at minimum: "condition" with values like "protein" / "no_protein")
- **Optional columns:** bead_type, target, protocol (for multi-factor designs)

### Data Reshaping from DELT-Hit Format

DELT-Hit produces counts in LONG format: one row per (selection, compound, count).
PyDESeq2 needs WIDE format: one row per selection, one column per compound.

```python
import polars as pl
import pandas as pd

def prepare_deseq_input(
    counts_df: pl.DataFrame,
    code_cols: list[str],
    metadata_df: pl.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape DEL counts for PyDESeq2.
    
    Args:
        counts_df: Long-format counts with columns:
            selection, code_1, code_2, ..., count
        code_cols: List of code column names (e.g., ["code_1", "code_2"])
        metadata_df: Selection metadata with columns:
            selection_name, condition, (optional: bead_type, target)
    
    Returns:
        (counts_wide_pd, metadata_pd) ready for DeseqDataSet
    """
    # Create compound_id from code columns
    compound_id_expr = pl.concat_str(code_cols, separator="_")
    
    df = counts_df.with_columns(
        compound_id_expr.alias("compound_id")
    )
    
    # Pivot: selections as rows, compounds as columns
    wide = df.pivot(
        on="compound_id",
        index="selection",
        values="count",
    ).fill_null(0)
    
    # Convert to pandas (PyDESeq2 requirement)
    counts_pd = wide.to_pandas().set_index("selection")
    
    # Ensure integer dtype (PyDESeq2 requires int counts)
    counts_pd = counts_pd.astype(int)
    
    # Prepare metadata
    meta_pd = metadata_df.to_pandas().set_index("selection_name")
    
    # Align: metadata rows must match counts rows
    meta_pd = meta_pd.loc[counts_pd.index]
    
    return counts_pd, meta_pd
```

## Running PyDESeq2

### Single-Factor Analysis (Target vs Control)

```python
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

def run_deseq2_single_factor(
    counts_pd: pd.DataFrame,
    metadata_pd: pd.DataFrame,
    contrast: tuple[str, str, str] = ("condition", "protein", "no_protein"),
) -> pd.DataFrame:
    """Run PyDESeq2 with single-factor design.
    
    Args:
        counts_pd: Wide counts matrix (selections × compounds), integer
        metadata_pd: Metadata with "condition" column
        contrast: (factor_name, test_level, reference_level)
    
    Returns:
        DataFrame with columns: baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
        Index: compound_ids
    """
    dds = DeseqDataSet(
        counts=counts_pd,
        metadata=metadata_pd,
        design="~condition",
    )
    
    dds.deseq2()
    
    stats = DeseqStats(dds, contrast=contrast)
    stats.summary()
    
    return stats.results_df
```

### Multi-Factor Analysis (Correcting for Bead Type)

When selections use different bead types, bead-specific binding can confound
target-specific enrichment. A multi-factor design corrects for this:

```python
def run_deseq2_multi_factor(
    counts_pd: pd.DataFrame,
    metadata_pd: pd.DataFrame,
    # metadata must have both "condition" AND "bead_type" columns
    contrast: tuple[str, str, str] = ("condition", "protein", "no_protein"),
) -> pd.DataFrame:
    """Run PyDESeq2 with multi-factor design correcting for bead type.
    
    The design ~bead_type + condition estimates the condition effect
    while holding bead_type constant. This removes bead-specific
    enrichment artifacts from the target-specific signal.
    """
    dds = DeseqDataSet(
        counts=counts_pd,
        metadata=metadata_pd,
        design="~bead_type + condition",
    )
    
    dds.deseq2()
    
    stats = DeseqStats(dds, contrast=contrast)
    stats.summary()
    
    return stats.results_df
```

### Multi-Target Analysis

When the same library is screened against multiple targets, run separate
contrasts for each target:

```python
# For each target protein, compare its selections to blank
for target_name in ["L1CAM_Ig1to4", "L1CAM_Ig1to6", "hCAIX"]:
    # Filter to selections for this target + blanks
    relevant_selections = [s for s in metadata if 
                          metadata[s]["target"] in [target_name, "No Protein"]]
    # Run PyDESeq2 with contrast ("condition", target_name, "No Protein")
    results[target_name] = run_deseq2_single_factor(
        counts_pd.loc[relevant_selections],
        metadata_pd.loc[relevant_selections],
        contrast=("condition", "protein", "no_protein"),
    )
```

## Performance Considerations

### Feature Count Scaling

| Synthon level | Typical diversity | PyDESeq2 runtime | Recommended? |
|---|---|---|---|
| Monosynthon | 100–1,000 | < 5 seconds | ✅ Always |
| Disynthon | 10K–1M | 10 sec – 5 min | ✅ Yes |
| Trisynthon (2-cycle) | 100K–1M | 1–10 min | ✅ Usually OK |
| Trisynthon (3-cycle) | 1M–1B | 10 min – hours | ⚠️ Only if small |

**Strategy:** Run PyDESeq2 at monosynthon and disynthon levels by default.
At trisynthon level, use z-score and Poisson methods (instant computation,
no iterative fitting needed). Offer PyDESeq2 at trisynthon as an opt-in
for small libraries (< 500K compounds).

### Memory Scaling

The pivoted counts matrix has dimensions (n_selections × n_compounds).
For 21 selections × 317K compounds (as in the real data): ~6.7M cells, ~50 MB.
This is fine. For 21 selections × 10M compounds: ~210M cells, ~1.6 GB.
Still feasible but requires attention.

### Minimum Replicate Requirements

PyDESeq2 needs at least **2 samples per condition** for dispersion estimation.
With only 1 sample, the dispersion cannot be estimated → the model fails.

From the example config.yaml:
- no_protein group: 6 selections (JPAG_2025_1 through JPAG_2025_6) ✅
- protein group: 15 selections across multiple targets ✅

If a specific target has < 2 selections, fall back to z-score/Poisson methods
for that comparison and warn the user.

## What PyDESeq2 Provides That z-score/Poisson Don't

1. **Dispersion estimation** — models overdispersion (variance > mean) using
   information borrowed across features. This is the key statistical advantage.
2. **Empirical Bayes shrinkage** — stabilizes fold-change estimates for
   low-count features, reducing false positives from noise.
3. **Multi-factor design** — can correct for bead type, batch, protocol
   differences in a single model. z-score/Poisson cannot do this.
4. **FDR control** — Benjamini-Hochberg adjusted p-values. The z-score has
   no built-in multiple testing correction (you'd need to add BH separately).
5. **Cook's distance** — automatic detection of count outliers (e.g., a single
   selection with an aberrantly high count for one compound).
6. **Size factor normalization** — handles library-size differences between
   selections automatically via the median-of-ratios method.

## What z-score/Poisson Provide That PyDESeq2 Doesn't

1. **Cross-level comparability** — z-score is designed for comparing enrichment
   across monosynthon, disynthon, and trisynthon on the same scale.
2. **Speed** — instant computation, no iterative fitting. Critical for
   trisynthon-level analysis on billion-member libraries.
3. **Single-replicate support** — z-score and Poisson work without replicates.
   PyDESeq2 requires ≥ 2 replicates per condition.
4. **Interpretability** — z_n ≥ 1 means "significantly enriched" regardless
   of level. log2FoldChange requires context to interpret.
5. **Simplicity** — easier to explain to experimental scientists who may not
   understand GLM frameworks.

## Recommended Method Strategy for DELexplore

| Situation | Recommended method |
|---|---|
| ≥ 2 replicates per condition | PyDESeq2 (primary) + z-score (secondary) |
| Single replicate per condition | z-score + Poisson ML only |
| Multi-factor (bead correction) | PyDESeq2 multi-factor only |
| Trisynthon level, large library | z-score + Poisson (fast) |
| Mono/disynthon level | All three methods for consensus |
| Cross-level SAR visualization | z-score (designed for this purpose) |

## Output Format

PyDESeq2 returns per feature:
- `baseMean`: mean normalized count across all selections
- `log2FoldChange`: log2(target/control) enrichment
- `lfcSE`: standard error of the log2FC
- `stat`: Wald test statistic (log2FC / lfcSE)
- `pvalue`: raw p-value from Wald test
- `padj`: Benjamini-Hochberg adjusted p-value (FDR)

For DELexplore's consensus ranking, the relevant columns are:
- `log2FoldChange` → used for ranking (higher = more enriched)
- `padj` → used for significance filtering (padj < 0.05 = significant)

## Edge Cases and Error Handling

### Zero-dominated data
DEL data has many compounds with count=0 across all conditions.
PyDESeq2's independent filtering removes features with very low mean
counts automatically. This is desirable for DEL — it reduces the multiple
testing burden by excluding noise features.

### Convergence failures
For some features, the NB GLM may not converge. PyDESeq2 handles this
by returning NaN for those features. DELexplore should:
1. Log how many features failed to converge
2. Report them as "inconclusive" (not enriched, not depleted)
3. If > 10% fail, warn the user (suggests data quality issues)

### All-zero features
Features with zero counts across ALL selections are automatically removed
by PyDESeq2. This is correct behavior for DEL analysis.

### Negative log2FoldChange
A negative log2FC means the compound is LESS abundant in the target
selection than in the control. This could mean:
- Competitive binding with another enriched compound (rare)
- Systematic loss during the selection process
- Usually just noise. Do not interpret negative log2FC as meaningful
  unless it's consistent across replicates.
