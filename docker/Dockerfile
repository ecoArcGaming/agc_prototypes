# ============================================================
# Jetson Orin Nano 8GB — VLM Assistant
# Base: JetPack 6.x / L4T r36.2
# Includes: PyTorch 2.1 + CUDA + TensorRT (GPU-optimised for Jetson)
# ============================================================

FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace model cache — mount a volume here to persist downloads
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Reduces memory fragmentation on Jetson's unified CPU/GPU memory
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y \
    # Audio (sounddevice, librosa, piper-tts)
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    espeak-ng \
    espeak-ng-data \
    # OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    # GPIO
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────
# ⚠️  torch/torchvision/torchaudio are NOT in requirements.txt on purpose.
#     They ship pre-installed in the base image with CUDA support for Jetson.
#     Installing them via pip would replace the GPU build with a CPU-only one.

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY main.py .

RUN mkdir -p /app/models /app/data /app/logs

CMD ["python3", "main.py"]