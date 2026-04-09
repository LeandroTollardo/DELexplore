# DELexplore — Project Context for Claude Code

## What This Is
DELexplore is a standalone Python tool for statistical analysis, hit ranking,
and chemical space exploration of DNA-encoded library (DEL) screening data.
It processes output from DELT-Hit (counts + config). Pure Python — no R dependency.

## Input Formats (from DELT-Hit)
- **counts.txt**: TSV files, one per selection, in folders like JPAG_2025_1/counts.txt
  Columns: code_1, code_2, count, id (tab-separated, variable number of code columns)
- **config.yaml**: Experiment metadata with selections, targets, groups, bead types,
  building block definitions with SMILES
- **library.parquet**: Optional enumerated library with SMILES per compound
- **deldenoiser output**: Optional TSV with BB IDs + fitness scores (Kómár 2020)
See @docs/references/delthit_output_formats.md for exact format spec.

## Setup & Dev Commands
```
uv sync --all-extras
source .venv/bin/activate
pytest tests/ -v
pytest tests/benchmarks/ -v -m benchmark   # slower ground-truth validation
ruff check src/
```

## Key Dependencies
Python 3.10+, PyDESeq2 (replaces edgeR — NO R needed), polars, numpy, scipy,
click, pyyaml, jinja2, RDKit, umap-learn, hdbscan, plotly, scikit-learn

## Project Structure
```
src/delexplore/
├── cli.py                     # Click CLI (root + qc, analyse, explore groups)
├── io/readers.py              # Read DELT-Hit TSV counts + YAML config + deldenoiser
├── io/writers.py              # Write analysis outputs
├── qc/assess.py               # Data quality assessment + HTML report
├── qc/naive.py                # Naive library synthesis QC
├── analyse/aggregate.py       # Multi-level synthon aggregation (N-cycle)
├── analyse/zscore.py          # Normalized z-score (Faver 2019) + MAD z-score (Wichert 2024)
├── analyse/poisson.py         # Poisson CI + ML enrichment (Kuai/Hou)
├── analyse/deseq.py           # PyDESeq2 negative binomial GLM
├── analyse/multilevel.py      # Multi-level enrichment orchestration
├── analyse/classify.py        # Binder classification + frequent hitter detection
├── analyse/rank.py            # Consensus ranking (rank-product + support + penalty)
├── analyse/bb_productivity.py # P(bind) BB productivity analysis (Zhang 2023)
├── explore/properties.py      # Drug-likeness (Lipinski, bRo5, Veber, Pfizer 3/75, QED)
├── explore/macrocycle.py      # Macrocycle detection + 3D descriptors (PMI, conformers)
├── explore/structures.py      # RDKit structure rendering (grid images, inline SVG)
├── explore/umap_viz.py        # UMAP projections (Morgan FP + Jaccard distance)
├── explore/cluster.py         # HDBSCAN clustering on UMAP space
├── explore/scaffold.py        # Murcko scaffold analysis
├── explore/dashboard.py       # Interactive HTML hit explorer (self-contained)
└── utils/chemistry.py         # Shared RDKit utilities
```

## Coding Conventions
- Type hints on ALL functions
- Google-style docstrings
- pathlib.Path, never os.path
- logging module, never print()
- Polars for DataFrames (pandas only at PyDESeq2 boundary)
- Column names: code_1, code_2, ..., code_N (matching DELT-Hit output)
- RDKit imports gated with try/except for optional dependency messaging

## Architecture Rules
- Input via io/readers.py only (single point of format handling)
- Auto-detect n_cycles from number of code_* columns
- QC produces data_quality.json consumed by analysis
- All enrichment runs at ALL synthon levels automatically
- Poisson CI uses EXACT chi-squared method (see corrected_formulas.md)
- PyDESeq2 for NB GLM (at mono/disynthon level for performance)
- property_penalty from explore/properties.py feeds into analyse/rank.py
- P(bind) from analyse/bb_productivity.py enhances support_score in rank.py
- deldenoiser fitness scores are optional additional input to ranking
- Parquet for intermediate files, CSV export for users
- Dashboard HTML is self-contained (no external JS/CSS dependencies)
- Benchmark tests use @pytest.mark.benchmark, run separately

## Enrichment Methods Available
| Method | Module | Reference |
|--------|--------|-----------|
| Normalized z-score | zscore.py | Faver et al. 2019 |
| MAD z-score | zscore.py | Wichert et al. 2024 |
| Poisson ML enrichment | poisson.py | Hou et al. 2023 |
| Poisson exact CI | poisson.py | Kuai et al. 2018 |
| PyDESeq2 NB GLM | deseq.py | Muzellec et al. 2023 |
| P(bind) BB productivity | bb_productivity.py | Zhang et al. 2023 |
| deldenoiser fitness | io/readers.py (import) | Kómár & Kalinić 2020 |

## CLI Command Groups
```
delexplore qc assess              # data quality assessment + HTML report
delexplore qc naive               # naive library synthesis QC
delexplore analyse zscore          # z-score enrichment at all levels
delexplore analyse poisson         # Poisson ML enrichment at all levels
delexplore analyse deseq           # PyDESeq2 NB GLM enrichment
delexplore analyse rank            # consensus ranking (--library-parquet, --denoised-scores)
delexplore explore properties      # drug-likeness from SMILES
delexplore explore render-hits     # structure grid image
delexplore explore umap            # UMAP chemical space projection
delexplore explore dashboard       # interactive HTML hit explorer
delexplore library-assess          # BB yield + truncation analysis
```

## Reference Documents
@./docs/references/corrected_formulas.md
@./docs/references/delthit_output_formats.md
@./docs/references/analysis_pipeline_architecture.md
@./docs/references/pydeseq2_integration.md
@./docs/references/kuai_2018_poisson_enrichment.md
@./docs/references/faver_2019_zscore_method.md
@./docs/references/hou_2023_ml_denoising.md
@./docs/references/iqbal_2025_del_ml_pipeline.md
@./docs/references/zhang_2023_bb_productivity.md
@./docs/references/wichert_2024_del_data_review.md
@./docs/references/komar_2020_deldenoiser.md
@./docs/references/poongavanam_2026_del_ml_generalizability.md
@./docs/references/competitor_analysis_and_decisions.md
@./docs/references/dev_environment_setup.md
