# DELexplore

Statistical analysis, hit ranking, and chemical space exploration for DNA-encoded library (DEL) screening data. Pure Python — no R.

> **Prerequisite:** DELexplore processes output from [DELT-Hit](https://github.com/your-org/delt-hit). Run DELT-Hit first for FASTQ demultiplexing, barcode decoding, and library enumeration. DELexplore picks up from the decoded counts files.

## What it does

| Stage | Command | Output |
|---|---|---|
| Quality control | `delexplore qc assess` | `data_quality.json`, HTML report |
| Naive library QC | `delexplore qc naive` | Synthesis yield report |
| Z-score enrichment | `delexplore analyse zscore` | Enrichment parquet |
| Poisson ML enrichment | `delexplore analyse poisson` | Enrichment parquet |
| DESeq2 NB-GLM | `delexplore analyse deseq` | DESeq2 results parquet |
| Consensus ranking | `delexplore analyse rank` | Ranked hit list CSV |
| Chemical space | `delexplore explore umap` | UMAP HTML plot |
| Clustering | `delexplore explore cluster` | Cluster assignments |
| Library assessment | `delexplore library-assess` | Synthesis + truncation reports |

Analysis runs at **all synthon levels automatically** (mono-, di-, trisynthon) — no manual configuration required.

## Installation

```bash
# Core + chemistry + statistics (recommended)
pip install "delexplore[chem,ml]"

# Everything including UMAP/clustering/plots
pip install "delexplore[all]"
```

Requires Python ≥ 3.10.

## Quick start

```bash
# 1. Check data quality before running analysis
delexplore qc assess \
    --config path/to/config.yaml \
    --output results/qc/

# 2. Compute enrichment (runs at all synthon levels)
delexplore analyse zscore \
    --config path/to/config.yaml \
    --output results/enrichment/

delexplore analyse poisson \
    --config path/to/config.yaml \
    --output results/enrichment/

# 3. Rank hits by consensus score
delexplore analyse rank \
    --config path/to/config.yaml \
    --input  results/enrichment/ \
    --output results/hits/

# 4. Explore chemical space
delexplore explore umap \
    --hits   results/hits/ranked_hits.parquet \
    --output results/umap/
```

### Docker

```bash
docker run --rm \
    -v /path/to/experiment:/data \
    -v /path/to/output:/results \
    ghcr.io/your-org/delexplore:latest \
    analyse zscore --config /data/config.yaml --output /results/enrichment/
```

## Input format (from DELT-Hit)

DELexplore reads the standard DELT-Hit output layout:

```
experiment/
├── config.yaml                      # experiment metadata + library definition
└── selections/
    ├── AG24_13/counts.txt           # one TSV per selection
    ├── AG24_14/counts.txt
    └── ...
```

**counts.txt** — tab-separated, sorted descending by count:

```
code_1	code_2	count	id
220	315	234	220_315
296	315	227	296_315
28	315	225	28_315
```

The number of `code_*` columns is detected automatically (2-cycle, 3-cycle, etc.).

**config.yaml** — experiment metadata:

```yaml
selections:
  AG24_13:
    group: protein
    target: ProteinA
    beads: HisPURE Beads
    date: '2024-09-26'
library:
  building_blocks: [B0, B1]
  B0:
    - index: 0
      smiles: "NC(CC1=CC=CC=C1)C(O)=O"
      ...
```

## Development setup

```bash
# Clone and install in editable mode with all dev tools
git clone https://github.com/your-org/delexplore.git
cd delexplore
uv sync --all-extras
source .venv/bin/activate

# Verify installation
delexplore --help

# Run tests
pytest tests/ -v

# Lint
ruff check src/
ruff format src/ tests/

# Install pre-commit hooks (run automatically on git commit)
pre-commit install
```

### Optional dependency groups

| Group | Contents | When you need it |
|---|---|---|
| `chem` | RDKit | Property calculation, scaffold analysis, fingerprints |
| `explore` | UMAP, HDBSCAN, Plotly | Chemical space visualisation |
| `ml` | PyDESeq2, scikit-learn | DESeq2 NB-GLM, ML enrichment |
| `dev` | pytest, ruff, mypy, pre-commit | Development |
| `all` | Everything above | Full install |

## Project status

DELexplore is under active development. The I/O layer (`io/readers.py`) and project infrastructure are complete. Analysis modules are being implemented in phases — see [CONTRIBUTING.md](CONTRIBUTING.md) for the roadmap.

## License

MIT — see [LICENSE](LICENSE).
