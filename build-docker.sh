#!/bin/bash
# Build script for LLM-Lab Docker images

set -e

echo "Building LLM-Lab Docker images..."
echo

# Build standard image
echo "================================"
echo "Building standard image (latest)"
echo "================================"
docker build -f Dockerfile -t blazux/llm-lab:latest .
echo "✓ Standard image built successfully"
echo

# Build CUDA image
echo "================================"
echo "Building CUDA image (cuda)"
echo "================================"
docker build -f Dockerfile.cuda -t blazux/llm-lab:cuda .
echo "✓ CUDA image built successfully"
echo

echo "================================"
echo "Build complete!"
echo "================================"
echo
echo "Available images:"
echo "  - blazux/llm-lab:latest (Universal compatibility)"
echo "  - blazux/llm-lab:cuda   (High performance with mamba-ssm)"
echo
echo "To push to Docker Hub:"
echo "  docker push blazux/llm-lab:latest"
echo "  docker push blazux/llm-lab:cuda"
