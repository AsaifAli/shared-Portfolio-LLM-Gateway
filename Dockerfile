
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser


# Test stage
FROM base AS test

USER root

RUN pip install --no-cache-dir pytest httpx

COPY tests ./tests

RUN chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "pytest", "-q"]


# Production/runtime stage
FROM base AS runtime

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
