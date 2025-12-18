# ============================================================================
# LLM-Lab Standard Image (Universal Compatibility)
# ============================================================================
# This Dockerfile builds a lightweight image that works on any system.
# Uses PyTorch-based Mamba2 implementation (no mamba-ssm).
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

# Main stage - Lightweight image with universal compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt ./
COPY gui/backend/requirements.txt ./gui/backend/

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
RUN mkdir -p cache outputs/pretraining outputs/sft outputs/rlhf gui/backend/cache

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/gui/backend

# Set working directory to backend
WORKDIR /app/gui/backend

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
