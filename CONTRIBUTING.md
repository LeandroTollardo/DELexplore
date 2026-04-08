# Contributing to DELexplore

## Development setup

```bash
git clone https://github.com/your-org/delexplore.git
cd delexplore
uv sync --all-extras
source .venv/bin/activate
pre-commit install
```

## Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes, add tests
3. `ruff check src/ && ruff format src/ tests/`
4. `pytest tests/ -v`
5. Open a PR against `main`

Pre-commit hooks run ruff lint + format automatically on every commit.

## Project structure

```
src/delexplore/
├── io/readers.py       # All input parsing lives here — single entry point
├── qc/                 # Data quality assessment (runs before analysis)
├── analyse/            # Enrichment methods and consensus ranking
├── explore/            # UMAP, clustering, scaffold analysis
└── utils/chemistry.py  # RDKit helpers (fingerprints, properties)

tests/
├── conftest.py         # Shared fixtures (real example data + synthetic data)
├── data/examples/      # Checked-in real DELT-Hit output (do not modify)
└── data/synthetic/     # Generated at test runtime by conftest.py fixtures
```

## Coding conventions

- **Type hints** on all functions — no bare `Any` unless unavoidable
- **Google-style docstrings** — Args / Returns / Raises sections
- **`pathlib.Path`** everywhere — never `os.path`
- **`logging`** everywhere — never `print()`
- **Polars** for DataFrames — pandas only at the PyDESeq2 boundary
- Column names follow DELT-Hit: `code_1`, `code_2`, ..., `code_N`

## Implementation roadmap

### Phase 2A — Core analysis engine (current)

| Module | Status | Description |
|---|---|---|
| `io/readers.py` | Done | TSV + YAML readers, cycle auto-detection |
| `analyse/aggregate.py` | Pending | Multi-level synthon aggregation |
| `analyse/zscore.py` | Pending | Normalised z-score (Faver 2019) |
| `analyse/poisson.py` | Pending | Exact Poisson CI + ML enrichment (Kuai/Hou) |
| `analyse/deseq.py` | Pending | PyDESeq2 negative binomial GLM |
| `analyse/classify.py` | Pending | Binder classification |
| `analyse/rank.py` | Pending | Consensus hit ranking |
| `qc/assess.py` | Pending | Data quality + `data_quality.json` |

### Phase 2B — Library quality intelligence

- `qc/naive.py` — synthesis yield, reaction efficiency, truncation detection
- Naive-count normalisation of selection enrichment scores

### Phase 3 — Chemical space exploration

- `explore/umap_viz.py` — Morgan fingerprint UMAP (Jaccard distance)
- `explore/cluster.py` — HDBSCAN clustering with per-cluster enrichment stats
- `explore/scaffold.py` — Murcko scaffold grouping

### Phase 4 — ML integration

- Balanced dataset preparation (hard negative sampling)
- RF / MLP / ChemProp training and cross-validation
- Virtual screening prediction module

## Key formulas

See `docs/references/` for the authoritative references. Quick summary:

**Normalised z-score** (Faver et al. 2019):
```
z_n = (p_observed - p_i) / sqrt(p_i * (1 - p_i))
p_i = 1 / diversity_at_this_synthon_level
```

**Exact Poisson CI** (chi-squared method — use this, not the approximation):
```python
lower = chi2.ppf(alpha/2,  2*k)     / 2   # 0 when k==0
upper = chi2.ppf(1-alpha/2, 2*(k+1)) / 2
```

**Poisson ML enrichment** (Hou et al. 2023):
```
ML = (n_control / n_post) * ((k_post + 3/8) / (k_control + 3/8))
```

## Running tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/delexplore --cov-report=term-missing

# One file
pytest tests/test_readers.py -v
```

The synthetic fixtures in `conftest.py` generate 80-compound 2-cycle data
(6 selections) with a known enrichment pattern (true binders, bead binders,
noise) — ideal for unit testing analysis modules without real sequencing data.

## Adding a new analysis module

1. Implement in `src/delexplore/analyse/<module>.py`
2. Add a CLI command in `src/delexplore/cli.py` under the `analyse` group
3. Write tests in `tests/test_<module>.py` using `synthetic_counts` fixture
4. Update the roadmap table above

## Questions / issues

Open a GitHub issue. For scientific questions about the enrichment formulas,
see the reference documents in `docs/references/`.
