FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
  curl git gdal-bin libgdal-dev gcc \
  && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY inference.py .

# Environment defaults (override at runtime)
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV MY_ENV_V4_TASK="brahmaputra-flood-detection"
ENV MY_ENV_V4_BENCHMARK="chronostasis"

COPY server.py .
COPY tasks.py .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
