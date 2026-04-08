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
See @docs/references/delthit_output_formats.md for exact format spec.

## Setup & Dev Commands
```
uv sync --all-extras
source .venv/bin/activate
pytest tests/ -v
ruff check src/
```

## Key Dependencies
Python 3.10+, PyDESeq2 (replaces edgeR — NO R needed), polars, numpy, scipy,
click, pyyaml, jinja2, RDKit, umap-learn, hdbscan, plotly, scikit-learn

## Project Structure
```
src/delexplore/
├── cli.py                # Click CLI
├── io/readers.py         # Read DELT-Hit TSV counts + YAML config
├── io/writers.py         # Write analysis outputs
├── qc/assess.py          # Data quality assessment + HTML report
├── qc/naive.py           # Naive library synthesis QC
├── analyse/aggregate.py  # Multi-level synthon aggregation (N-cycle)
├── analyse/zscore.py     # Normalized z-score (Faver et al. 2019)
├── analyse/poisson.py    # Poisson CI + ML enrichment (Kuai/Hou)
├── analyse/deseq.py      # PyDESeq2 negative binomial GLM
├── analyse/classify.py   # Binder classification
├── analyse/rank.py       # Consensus hit ranking
├── explore/umap_viz.py   # UMAP projections
├── explore/cluster.py    # HDBSCAN clustering
└── explore/scaffold.py   # Murcko scaffold analysis
```

## Coding Conventions
- Type hints on ALL functions
- Google-style docstrings
- pathlib.Path, never os.path
- logging module, never print()
- Polars for DataFrames (pandas only at PyDESeq2 boundary)
- Column names: code_1, code_2, ..., code_N (matching DELT-Hit output)

## Architecture Rules
- Input via io/readers.py only (single point of format handling)
- Auto-detect n_cycles from number of code_* columns
- QC produces data_quality.json consumed by analysis
- All enrichment runs at ALL synthon levels automatically
- Poisson CI uses EXACT chi-squared method (see corrected_formulas.md)
- PyDESeq2 for NB GLM (at mono/disynthon level for performance)
- Parquet for intermediate files, CSV export for users

## Reference Documents
@./docs/references/corrected_formulas.md
@./docs/references/delthit_output_formats.md
@./docs/references/analysis_pipeline_architecture.md
@./docs/references/pydeseq2_integration.md
@./docs/references/kuai_2018_poisson_enrichment.md
@./docs/references/faver_2019_zscore_method.md
@./docs/references/hou_2023_ml_denoising.md
@./docs/references/iqbal_2025_del_ml_pipeline.md
@./docs/references/competitor_analysis_and_decisions.md
@./docs/references/dev_environment_setup.md
