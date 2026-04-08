# Reference: Development Environment Setup

## Package Manager: uv (with conda fallback for RDKit)

### Why uv over conda

- uv resolves and installs 10-100x faster than conda
- uv uses standard pyproject.toml (PEP 621), conda uses environment.yml
- uv creates reproducible lockfiles (uv.lock), conda has weaker locking
- uv is pip-compatible — users install DELT-Hit with pip, not conda
- RDKit now publishes PyPI wheels (pip install rdkit), reducing need for conda

### When conda is still needed

- If rdkit wheel fails on a specific platform (some ARM/Linux combos)
- If user needs CUDA-accelerated PyTorch (conda handles CUDA libs better)
- On HPC clusters where conda is the institutional standard

### Recommended pyproject.toml Structure

```toml
[project]
name = "delt-hit"
version = "0.2.0"
description = "DNA-Encoded Library Technology Hit Identification"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "click>=8.0",
    "polars>=0.20",
    "rdkit>=2023.9",
    "pyarrow>=14.0",
    "scipy>=1.11",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "jinja2>=3.1",
]

[project.optional-dependencies]
ml = [
    "scikit-learn>=1.3",
    "torch>=2.0",
    "umap-learn>=0.5",
    "hdbscan>=0.8",
]
viz = [
    "plotly>=5.18",
    "matplotlib>=3.8",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.3",
    "mypy>=1.8",
]
all = ["delt-hit[ml,viz,dev]"]

[project.scripts]
delt-hit = "delt_hit.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

### VS Code Settings (.vscode/settings.json)

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "*.egg-info": true
    }
}
```

### Required VS Code Extensions

- charliermarsh.ruff (linting + formatting)
- ms-python.python (Python language support)
- ms-python.vscode-pylance (type checking)
- anthropic.claude-code (AI coding assistant)
- ms-toolsai.jupyter (for tutorial notebooks)
