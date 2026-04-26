FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
  curl gdal-bin libgdal-dev gcc \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Core files
COPY server.py .
COPY tasks.py .
COPY renderer.py .
COPY inference.py .

# GEE helpers (safe — server.py handles ImportError if absent)
COPY gee_client.py .
COPY gee_codegen.py .

# Static frontend
COPY static/ ./static/

# Server subfolder (if used)
COPY server/ ./server/

ENV GEE_PROJECT="chronostasis-gee"
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]