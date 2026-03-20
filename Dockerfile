FROM python:3.12-slim AS base

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and data
COPY src/ src/
COPY data/ data/

# Install the project itself
RUN uv sync --frozen --no-dev

# Ingest documents into ChromaDB at build time
# Requires OPENAI_API_KEY as a build arg for embedding generation
ARG OPENAI_API_KEY
RUN uv run python -m kbee.ingest --dir ./data/sporty_stage_ng --clear

EXPOSE 8787

ENV BASE_URL=http://0.0.0.0:8787

CMD ["uv", "run", "python", "-m", "kbee.realtime_server"]
