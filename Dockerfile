# --------------------------------------------------
# Equipment ML Model API - Docker image
# --------------------------------------------------
FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model artifact
COPY app.py .
COPY model.pkl .

# (optional) copy training script too if you want to retrain inside the container
COPY train_model.py .

EXPOSE 8000

# Basic healthcheck hitting the root endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]