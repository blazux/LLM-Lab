# ============================================================================
# LLM-Lab Image
# ============================================================================
# This Dockerfile builds an image with CUDA 13.0 and PyTorch 2.13.0 support.
#
# Validated on RTX 5090. Other GPUs require their own compatibility checks.
#
# Build command:
#   docker build -f Dockerfile -t blazux/llm-lab:latest .
#
# Or use the build script:
#   ./build-docker.sh
# ============================================================================

# Multi-stage build for efficiency
ARG RUNTIME_IMAGE=python:3.11-slim
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
FROM ${RUNTIME_IMAGE}

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# PyTorch CUDA wheels provide the runtime libraries; --gpus mounts the host driver.
# Keeping Python in the base removes the Ubuntu-only PPA bootstrap dependency.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy Python requirements
COPY requirements.txt constraints.txt ./
COPY gui/backend/requirements.txt ./gui/backend/

# Use the same baseline as bootstrap.sh. CPU builds can override these args.
ARG TORCH_CHANNEL=cu130
RUN python -m pip install --no-cache-dir "torch==2.13.0+${TORCH_CHANNEL}" --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"

# Install Python dependencies
RUN python -m pip install --no-cache-dir -c constraints.txt -r requirements.txt -r gui/backend/requirements.txt

# Copy application code
COPY src/ ./src/
COPY gui/backend/ ./gui/backend/
COPY llm-lab.sh LICENSE VERSION README.md CHANGELOG.md ./
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY recipes/ ./recipes/
COPY docs/ ./docs/
COPY tests/ ./tests/

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
