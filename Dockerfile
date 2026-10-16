# Two stage build: the React interface, then the Python service that also serves it.
FROM node:22-slim AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ALLELION_ROOT=/app
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY allelion/ allelion/
RUN pip install --no-cache-dir .
COPY knowledge/ knowledge/
COPY Makefile ./
COPY --from=web /app/web/dist web/dist
# data/ is mounted at runtime (see compose.yml) or rebuilt with `make all`
EXPOSE 8000
CMD ["uvicorn", "allelion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
