# ── Dockerfile — Hugging Face Spaces (FastAPI) ────────────────
# Target: huggingface/transformers-pytorch-cpu or python:3.11-slim
# HF Spaces free tier: 2 vCPU, 16 GB RAM, port 7860

FROM python:3.11-slim

WORKDIR /app

# System deps for ta / numpy / pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached)
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy project (models, src, data)
COPY . .

# HF Spaces expects the app to listen on 0.0.0.0:7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
