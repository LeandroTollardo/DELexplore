# Reference: Analysis Pipeline Architecture & Library Quality Intelligence

## Part 1: Multi-Level Synthon Analysis

### Why It's Essential

A DEL compound from a 3-cycle library is a combination (BB_A, BB_B, BB_C).
The binding activity might come from any single BB, any pair, or only the full
triple. Analyzing at all levels simultaneously is the scientific standard because:

1. **Statistical power increases at lower synthon levels.** For a library of 10M
   trisynthons with 1000 BBs per cycle: monosynthon diversity = 1000,
   disynthon diversity = 1M, trisynthon diversity = 1B. At typical sequencing
   depths, trisynthon counts are Poisson-dominated noise. Monosynthon counts
   are orders of magnitude higher per feature, giving much better signal.

2. **SAR emerges at the disynthon level.** Faver et al. (2019) showed that in
   sEH selections, no individual trisynthon showed significant enrichment, but
   several disynthons (cycle 1 + cycle 3 pairs) were clearly enriched. The
   binding pharmacophore was defined by the disynthon.

3. **Truncated products can be identified.** Monosynthon enrichment that doesn't
   propagate to higher synthon levels suggests truncated synthesis products
   (incomplete reactions) rather than true binders.

### Implementation: Aggregation Engine

```python
# Pseudocode for synthon aggregation
def aggregate_to_synthon_level(
    counts_matrix: DataFrame,  # columns: selection, B0_id, B1_id, B2_id, count
    level: str,                # "mono_B0", "mono_B1", "di_B0B1", "di_B0B2", etc.
) -> DataFrame:
    """Aggregate compound counts to specified synthon level.
    
    For monosynthon B0: group by (selection, B0_id), sum counts.
    For disynthon B0B1: group by (selection, B0_id, B1_id), sum counts.
    
    The diversity changes at each level:
    - mono_B0: diversity = len(B0_set)
    - mono_B1: diversity = len(B1_set)
    - di_B0B1: diversity = len(B0_set) * len(B1_set)
    - tri_B0B1B2: diversity = full library size
    
    This diversity parameter feeds into the enrichment calculations.
    """
```

For a 3-cycle library, generate ALL seven possible synthon levels:
- Monosynthons: B0, B1, B2 (3 levels)
- Disynthons: B0B1, B0B2, B1B2 (3 levels)
- Trisynthon: B0B1B2 (1 level, = full compound)

Each level gets independent enrichment analysis with its own diversity parameter.

### Enrichment Methods per Level

All three enrichment methods (edgeR, z-score, Poisson ML) should run at
every synthon level. The normalized z-score (Faver et al.) is specifically
designed for cross-level comparison because its scaling factor accounts for
the diversity difference.

Interpretation guide:
- z_n >= 1 for monosynthon ≈ 30-fold enrichment
- z_n >= 1 for disynthon ≈ 1000-fold enrichment  
- z_n >= 1 for trisynthon ≈ 30,000-fold enrichment

This means z_n values ARE directly comparable across levels on the same plot.

---

## Part 2: QC vs. Analysis — Clean Separation

### QC Module: "Is My Data Trustworthy?"

Run BEFORE analysis. Produces `data_quality.json` consumed by analysis module.

| QC Metric | Threshold (Green) | Threshold (Yellow) | Threshold (Red) |
|---|---|---|---|
| Read retention | > 90% | 70-90% | < 70% |
| Sampling ratio (reads/diversity) | > 10 | 1-10 | < 1 |
| Replicate Pearson R | > 0.7 | 0.5-0.7 | < 0.5 |
| BB coverage (% BBs with > 0 reads) | > 95% | 80-95% | < 80% |
| Max single-BB fraction | < 5% | 5-15% | > 15% |

The `data_quality.json` output:
```json
{
  "overall_quality": "yellow",
  "sampling_ratio": {"trisynthon": 0.5, "disynthon": 50, "monosynthon": 5000},
  "replicate_correlation": {"protein_replicates": 0.82, "control_replicates": 0.91},
  "bb_coverage": {"B0": 0.98, "B1": 0.95, "B2": 0.87},
  "warnings": ["Trisynthon analysis underpowered (sampling_ratio < 1)",
                "B2 position has 13% missing building blocks"],
  "recommended_analysis_levels": ["mono", "disynthon"]
}
```

### Analysis Module: "What Are the Hits?"

Reads `data_quality.json` and adapts behavior:
- If trisynthon sampling_ratio < 1: flag trisynthon results as "low confidence"
- If replicate correlation < 0.5: exclude that condition or warn prominently
- If BB coverage < 80% for a position: note potential synthesis issues

### Connection Point

```
QC → data_quality.json → Analysis reads it → adjusts confidence levels
                        → Library Design Assessment reads it → flags synthesis problems
```

---

## Part 3: Consensus Hit Ranking

### The Problem

Three enrichment methods × seven synthon levels = 21 independent scores per
compound. How to combine them into a single actionable ranked list?

### Composite Scoring Architecture

#### Step 1: Multi-method agreement score (per synthon level)

For each compound at trisynthon level, compute:
- edgeR rank (lower = more enriched)
- z-score rank
- Poisson ML rank

Agreement score = geometric mean of ranks (rank-product statistic).
Low score = consistently enriched across all methods.

#### Step 2: Multi-level support score

For each compound (A_i, B_j, C_k):
- Does monosynthon A_i show enrichment? (+1 if z_n > threshold)
- Does monosynthon B_j show enrichment? (+1)
- Does monosynthon C_k show enrichment? (+1)
- Does disynthon A_i,B_j show enrichment? (+2, higher weight)
- Does disynthon A_i,C_k show enrichment? (+2)
- Does disynthon B_j,C_k show enrichment? (+2)

Support score = sum. Max = 9. Higher = more corroborating evidence.

Theory: A compound that is enriched AND whose constituent building blocks
and pairs are independently enriched is far more likely to be a true binder
than one where only the trisynthon score is high (which could be noise or
a synthesis artifact).

#### Step 3: Final composite rank

```
final_score = agreement_score * (1 / (1 + support_score)) * property_penalty
```

Where property_penalty penalizes compounds failing Lipinski/drug-likeness
filters (value > 1 for bad properties, = 1 for drug-like).

#### Step 4: Output

Ranked hit list with columns:
```
rank, smiles, B0_id, B1_id, B2_id, 
edgeR_logFC, edgeR_FDR, zscore, zscore_CI_lower, zscore_CI_upper,
poisson_ml_enrichment, agreement_rank, support_score, 
MW, logP, QED, lipinski_pass, composite_score
```

---

## Part 4: Library Quality Assessment & Design Intelligence

### What Makes a Good Library?

This is the critical question your tool can help answer. A good DEL library:

1. **High synthesis fidelity** — most theoretical compounds actually exist
2. **Uniform representation** — BBs are present in roughly equal amounts
3. **Drug-like property space** — compounds have MW, logP, TPSA in drug-like ranges
4. **Chemical diversity** — BBs cover diverse chemical space, not redundant
5. **Appropriate size for sequencing budget** — library not too large for available
   sequencing depth

### Using Naive Library Data

The naive (unselected) library sequencing is the single most valuable QC dataset.
It tells you what your library ACTUALLY looks like before any selection bias.

#### What naive data reveals:

**1. Synthesis yield per building block**

If all BBs were incorporated equally, naive counts would be uniform (after
Poisson noise). Deviations reveal synthesis bias.

```python
def assess_synthesis_uniformity(naive_counts, level="mono"):
    """Compare observed BB frequencies to expected uniform distribution.
    
    Expected: each BB in a position has count ≈ total_reads / n_BBs_in_position
    Observed: actual count per BB
    
    Metrics:
    - Coefficient of variation (CV) of BB counts: < 0.5 is good, > 1.0 is bad
    - Gini coefficient: 0 = perfect equality, 1 = one BB dominates
    - Chi-squared test: p-value for uniformity hypothesis
    - Identify outlier BBs: those > 3 SD from mean (over-represented)
      or with 0 counts (failed synthesis)
    """
```

**2. Reaction success rates**

Compare disynthon counts across different BB combinations that use the
same reaction. If one reaction step systematically produces lower counts:

```python
def assess_reaction_quality(naive_counts, library_definition):
    """Group compounds by the reaction used at each step.
    
    For each reaction:
    - Mean and variance of product counts
    - Fraction of expected products with zero counts
    - Correlation between BB molecular weight and synthesis success
      (heavy BBs sometimes couple less efficiently)
    """
```

**3. Truncated product detection**

A monosynthon that is highly over-represented in the naive library relative
to its expected frequency suggests truncated products — the first BB was
incorporated but subsequent reaction steps failed for many combinations.

```python
def detect_truncation(naive_mono_counts, naive_di_counts, naive_tri_counts):
    """Flag BBs where monosynthon count >> expected, but disynthon counts
    for combinations involving that BB are low.
    
    Truncation signal: high mono count + low di count = reaction failure
    after this BB position.
    
    This is critical because truncated products WILL appear enriched
    in selections (they bind nonspecifically) and are false positives.
    """
```

### Using DELpure / Self-Purified DEL Data

The Keller et al. (2024, Science) paper from your own lab describes the PureDEL
technology. In self-purified DELs, only fully elaborated products are released
from the solid support, while truncates remain attached.

For PureDEL analysis, the naive library data has a special property: the
count distribution reflects ACTUAL synthesis and encoding yields (not
contaminated by truncated products). This means:

1. **Naive PureDEL counts directly estimate per-compound synthesis yield**
   (unlike standard DELs where truncates confound the signal)

2. **You can normalize selection counts by naive PureDEL counts** to correct
   for synthesis yield bias. This is the approach described by Rama-Garda et al.
   (2021): enrichment = (selection_count / naive_count) normalized by totals.
   
3. **Comparison: standard DEL naive vs. PureDEL naive** reveals the
   truncation fraction. If standard naive shows BB_X at 5x expected frequency
   but PureDEL naive shows BB_X at 1x expected, then ~80% of BB_X's
   apparent abundance in the standard library comes from truncated products.

### Implementation: Library Assessment Module

```
delt-hit library assess --naive_counts=naive.parquet --config_path=config.yaml
```

Produces:
1. **Synthesis report** — per-BB yield estimates, per-reaction success rates,
   overall library uniformity metrics (CV, Gini)
2. **Truncation report** — flagged BBs with suspected truncation, with evidence
   (mono vs di vs tri count ratios)
3. **Property coverage report** — MW, logP, TPSA distributions of the ACTUAL
   library (weighted by synthesis yields, not theoretical enumeration)
4. **Diversity report** — UMAP visualization of actual library coverage vs.
   theoretical, with BB-level chemical space analysis
5. **Design recommendations** — BBs to remove (poor synthesis), BBs to add
   (gaps in chemical space), reactions to improve

### Factors That Affect Library Quality (Checklist)

| Factor | How to Measure | Data Source |
|---|---|---|
| BB synthesis yield | CV of naive monosynthon counts | Naive library |
| Reaction efficiency | Mean/variance of counts per reaction | Naive library |
| Truncated products | Mono/di/tri count ratio per BB | Naive library |
| DNA integrity | Read quality scores, adapter contamination | FASTQ QC |
| Encoding efficiency | Ligation success per step | Naive library |
| PCR bias | Duplicate rate, UMI analysis | Sequencing data |
| Chemical diversity | UMAP coverage, Tanimoto matrices | Enumeration |
| Drug-likeness | Lipinski/QED distribution | Property module |
| Sequencing depth adequacy | Sampling ratio per synthon level | QC module |
| Selection stringency | Enrichment dynamic range | Selection data |

### How Naive Data Feeds Into the Analysis Pipeline

```
naive.fastq.gz ──→ demultiplex ──→ naive_counts.parquet
                                         │
                                         ├──→ library assess (synthesis QC)
                                         │     ├── synthesis_report.html
                                         │     ├── truncation_flags.json
                                         │     └── bb_yield_weights.parquet
                                         │
selection.fastq.gz ─→ demultiplex ─→ selection_counts.parquet
                                         │
                                         ├──→ enrichment analysis
                                         │     (uses bb_yield_weights for normalization)
                                         │
                                         └──→ hit ranking
                                               (penalizes compounds with flagged BBs)
```

Key insight: the naive data is NOT just QC — it feeds forward into the
analysis as a normalization weight. Compounds made from poorly-synthesized
BBs will have artificially low counts that could be mistaken for non-binding.
Normalizing by naive yields corrects this.

---

## Part 5: Chemical Space Exploration Module

### UMAP Implementation

```python
# Core workflow
from umap import UMAP
from hdbscan import HDBSCAN
from rdkit.Chem import AllChem
import numpy as np

def compute_chemical_space(smiles_list, enrichment_scores):
    """UMAP projection of Morgan fingerprints, colored by enrichment."""
    
    # 1. Generate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
           for mol in mols]
    fp_array = np.array(fps)
    
    # 2. UMAP projection
    # Parameters from literature (Hou et al. 2023, Zhang et al. 2023):
    reducer = UMAP(
        metric="jaccard",      # Jaccard distance for binary fingerprints
        n_neighbors=15,        # local neighborhood size
        min_dist=0.1,          # minimum distance between points
        n_components=2,        # 2D projection
    )
    embedding = reducer.fit_transform(fp_array)
    
    # 3. Clustering
    clusterer = HDBSCAN(min_cluster_size=10, min_samples=5)
    cluster_labels = clusterer.fit_predict(embedding)
    
    return embedding, cluster_labels
```

### Per-Position BB Analysis (Zhang et al. 2023 approach)

For each cycle position independently:
1. Compute Tanimoto similarity matrix between all BBs (on truncated structures)
2. UMAP projection per position
3. Scale point size by P(active) = fraction of compounds containing that BB
   that are enriched
4. HDBSCAN clustering on UMAP coordinates
5. Compare within-cluster vs between-cluster P(active) variance

This reveals **activity cliffs** — BBs that are chemically similar but have
dramatically different enrichment, suggesting subtle SAR features.

### Scaffold Analysis

Use RDKit's Murcko scaffold decomposition:
```python
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)
```

Group enriched compounds by scaffold → identify scaffold families that
concentrate binding activity. This is directly actionable for medicinal
chemistry follow-up.

---

## Part 6: Updated Roadmap Integration

### Revised Phase Structure

**Phase 1 (Weeks 1-4): Infrastructure**
- uv-based pyproject.toml, Docker, CI/CD, Snakemake
- No analytical changes

**Phase 2A (Weeks 5-8): Core Analysis Engine**
- Multi-level synthon aggregation engine (all 7 levels for 3-cycle library)
- Normalized z-score + Poisson ML enrichment methods
- QC module with data_quality.json output
- Consensus ranking with multi-level support scoring

**Phase 2B (Weeks 9-12): Library Quality Intelligence** ← NEW
- Naive library analysis: synthesis yield, reaction quality, truncation detection
- PureDEL-aware normalization (naive-count correction)
- Yield-weighted property distributions (actual vs theoretical library)
- Design recommendations engine (BB scoring, gap analysis)

**Phase 3 (Weeks 13-16): Chemical Space & Exploration**
- UMAP visualization module (compound-level and per-position BB-level)
- HDBSCAN clustering with enrichment statistics per cluster
- Scaffold analysis and SAR identification
- Per-position building block activity analysis
- Interactive HTML dashboard for exploration

**Phase 4 (Weeks 17-22): ML Integration**
- Balanced dataset preparation (using multi-level enrichment + naive normalization)
- RF, MLP, ChemProp training with cross-validation
- Virtual screening prediction module
- Applicability domain analysis (UMAP overlap between training and screening sets)

**Phase 5 (Weeks 23-26): Documentation & Publication**
- Tutorial notebooks, benchmarking, application note

### Where Library Design Assessment Fits

Library design assessment is both retrospective AND prospective:

**Retrospective** (after a campaign): "How good was this library? Which BBs
worked? Which reactions need improvement? How much of my library was actually
present?" This uses naive + selection data.

**Prospective** (before synthesis): "If I pick these BBs and reactions, what will
the library look like? How does it compare to known bioactive chemical space?
Where are the gaps?" This uses the enumeration + property modules.

DELT-Hit already has the enumeration and property calculation for prospective
analysis. The retrospective analysis using naive data is the new capability.

### Pipeline Data Flow (Complete)

```
                    ┌──────────────────┐
                    │   Excel Config   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   delt-hit init  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼────┐ ┌───────▼──────┐
     │  library       │ │ naive  │ │  selection   │
     │  enumerate     │ │ FASTQ  │ │  FASTQs      │
     └────────┬───────┘ └───┬────┘ └───────┬──────┘
              │             │              │
     ┌────────▼───────┐ ┌───▼────┐ ┌───────▼──────┐
     │  library       │ │ demux  │ │  demultiplex │
     │  properties    │ │ naive  │ │  selections  │
     └────────┬───────┘ └───┬────┘ └───────┬──────┘
              │             │              │
              │     ┌───────▼──────┐       │
              │     │  library     │       │
              │     │  assess      │       │
              │     │ (synthesis   │       │
              │     │  QC, yield,  │       │
              │     │  truncation) │       │
              │     └───────┬──────┘       │
              │             │              │
              │     bb_yield_weights        │
              │     truncation_flags        │
              │             │              │
              │     ┌───────▼──────────────▼───┐
              │     │        QC module          │
              │     │  (data_quality.json)      │
              │     └───────┬──────────────────┘
              │             │
              │     ┌───────▼──────────────────┐
              │     │  analyse enrichment       │
              │     │  (multi-level, multi-     │
              │     │   method, naive-corrected)│
              │     └───────┬──────────────────┘
              │             │
              │     ┌───────▼──────────────────┐
              │     │  analyse rank             │
              │     │  (consensus scoring)      │
              │     └───────┬──────────────────┘
              │             │
     ┌────────▼─────────────▼──────────────────┐
     │         explore (UMAP, clustering,       │
     │         scaffold, per-position SAR)      │
     └─────────────────┬───────────────────────┘
                       │
              ┌────────▼───────────────────────┐
              │    ml train / ml predict        │
              │    (uses all above as features) │
              └────────────────────────────────┘
```
