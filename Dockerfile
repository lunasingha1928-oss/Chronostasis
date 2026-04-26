FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
  curl gdal-bin libgdal-dev gcc \
  && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
COPY tasks.py .
COPY renderer.py .
COPY inference.py .
COPY gee_codegen.py .
COPY gee_client.py .
COPY server/ ./server/
COPY static/ ./static/
ENV GEE_PROJECT="your-gee-project-id"
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
  CMD curl -f http://localhost:7860/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]