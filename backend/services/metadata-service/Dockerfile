# Metadata Service Dockerfile — GPU (LightningAI A100 80GB)
# SOTA 2026 AI: Grounding DINO 1.6, EVA-02 ViT-L/14, SigLIP 2, VideoMAE V2

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.4+ (base image has 2.3.1, upgrade needed for Grounding DINO 1.6 compatibility)
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first for better Docker caching
COPY backend/services/metadata-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "transformers>=4.49.0,<5.0.0" accelerate timm sentencepiece safetensors huggingface-hub uvicorn[standard]==0.32.1

# Copy application code
COPY backend/services/shared ./shared
COPY backend/services/metadata-service/app ./app
COPY backend/config ./backend/config

# Expose port (LightningAI cloudspace exposes 8002)
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=20).read()"

# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
