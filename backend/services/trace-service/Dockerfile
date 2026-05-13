# Trace Service Dockerfile — Neural Video Reconstruction on A100 80GB
# GPU models moved to metadata-service.
# This service: Real-ESRGAN + ProPainter + RIFE + NVENC

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# System dependencies — FFmpeg with NVENC + OpenCV build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/services/trace-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/services/shared ./backend/services/shared
COPY backend/services/trace-service/app ./app
COPY backend/config ./backend/config

ENV PYTHONPATH=/app
ENV TRACE_CACHE_ROOT=/workspace/storage/cache
# Prefer distro ffmpeg (`/usr/bin/ffmpeg`, with libx264) over the Conda build
# (`/opt/conda/bin/ffmpeg`, libopenh264-only and known to fail with "Incorrect
# library version loaded"). Python code resolves ffmpeg via clip_render
# .resolve_ffmpeg(); this PATH override is a defense-in-depth for any other
# subprocess that might still call "ffmpeg" bare.
ENV PATH=/usr/bin:${PATH}

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8004/health', timeout=5).read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
