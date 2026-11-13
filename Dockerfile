FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,dashboard]"

COPY configs ./configs
COPY app ./app

# data/, artifacts/ and reports/ are mounted at runtime (see docker-compose.yml).
RUN useradd --create-home appuser && mkdir -p data artifacts reports && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "salesforecast.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
