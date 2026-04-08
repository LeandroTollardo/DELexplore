# ── Stage 1: build the wheel ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install all production extras (chem, explore, ml) but not dev tooling.
# --no-editable keeps the install self-contained inside /app/.venv.
RUN uv sync \
        --extra chem \
        --extra explore \
        --extra ml \
        --frozen \
        --no-dev \
        --no-editable

# Copy source after dependencies to maximise cache reuse
COPY src/ ./src/
COPY README.md LICENSE ./

# Build and install the package itself
RUN uv pip install --no-deps -e .


# ── Stage 2: slim runtime image ───────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security
RUN useradd --create-home --shell /bin/bash delexplore

WORKDIR /app

# Copy the entire venv and installed package from the builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Make the venv the active Python environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Volume for input data and results (mount at runtime)
VOLUME ["/data", "/results"]

USER delexplore

ENTRYPOINT ["delexplore"]
CMD ["--help"]
