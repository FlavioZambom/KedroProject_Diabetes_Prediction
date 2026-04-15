FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY conf/ conf/

COPY data/01_raw/ data/01_raw/

RUN mkdir -p \
    data/02_intermediate \
    data/03_primary \
    data/04_feature \
    data/05_model_input \
    data/06_models \
    data/07_model_output \
    data/08_reporting

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "diabetes-prediction-api"]
