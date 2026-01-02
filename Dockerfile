# ============================================================================
# LLM-Lab Standard Image (Without mamba-ssm)
# ============================================================================
# This Dockerfile builds an image with CUDA 12.8 and PyTorch 2.7.0 support.
# Uses PyTorch-based Mamba2 implementation (no mamba-ssm optimized kernels).
#
# Supports: RTX 5090/5080 (Blackwell), RTX 40xx, RTX 30xx, RTX 20xx, A100, H100, V100
#
# Build command:
#   docker build -f Dockerfile -t blazux/llm-lab:latest .
#
# Or use the build script:
#   ./build-docker.sh
# ============================================================================

# Multi-stage build for efficiency
FROM node:20-slim AS frontend-builder

# Set working directory for frontend
WORKDIR /app/gui/frontend

# Copy frontend package files
COPY gui/frontend/package*.json ./

# Install frontend dependencies
RUN npm ci

# Copy frontend source
COPY gui/frontend/ ./

# Build frontend for production
RUN npm run build

# Main stage - CUDA-enabled base image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy Python requirements
COPY requirements.txt ./
COPY gui/backend/requirements.txt ./gui/backend/

# Install PyTorch 2.7.0 stable with CUDA 12.8 support (RTX 50xx series compatible)
RUN pip install --no-cache-dir torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r gui/backend/requirements.txt

# Copy application code
COPY src/ ./src/
COPY gui/backend/ ./gui/backend/
COPY llm-lab.sh ./

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/gui/frontend/dist ./gui/frontend/dist

# Create necessary directories
RUN mkdir -p data/checkpoints \
    data/sft_checkpoints \
    data/rlhf_checkpoints \
    data/cache

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/gui/backend

# Set working directory to backend
WORKDIR /app/gui/backend

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
