# ============================================================
# Jetson Orin Nano 8GB — VLM Assistant
# ============================================================

FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /app

# ── 1. System deps (cached unless this block changes) ─────────
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    espeak-ng \
    espeak-ng-data \
    libgl1-mesa-dri \
    libglib2.0-0 \
    python3-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Python deps BEFORE copying src (key cache trick) ───────
# Only re-runs if requirements.txt changes, NOT when src/ changes
COPY requirements.txt .
RUN pip3 install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu128 \
    -r requirements.txt

# ── 3. Source code last (changes most often) ──────────────────
COPY src/ ./src/

RUN mkdir -p /app/models /app/data /app/logs

CMD ["python3", "src/main.py"]