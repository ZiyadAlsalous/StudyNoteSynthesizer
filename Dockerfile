FROM python:3.12-slim
WORKDIR /app

# weasyprint needs these for PDF output.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY studysynth ./studysynth
RUN pip install --no-cache-dir ".[claude,qwen,render]"

EXPOSE 8501
CMD ["python", "-m", "studysynth", "ui"]
