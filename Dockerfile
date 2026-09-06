# --- build the frontend -----------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- runtime ----------------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

# weasyprint needs these for PDF output.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# claude for the API client, qwen for local embeddings, render for PDF.
# Installing only [render] left the container unable to make a single call.
COPY pyproject.toml README.md ./
COPY studysynth ./studysynth
RUN pip install --no-cache-dir ".[claude,qwen,render]"

COPY --from=web /web/dist ./web/dist

EXPOSE 8000
CMD ["python", "-m", "studysynth", "serve"]
