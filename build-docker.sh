#!/bin/bash
# Build script for the LLM-Lab Docker image

set -e

echo "Building LLM-Lab Docker image..."
echo

echo "================================"
echo "Building image (latest)"
echo "================================"
docker build -f Dockerfile -t blazux/llm-lab:latest .
echo "✓ Image built successfully"
echo

echo "================================"
echo "Build complete!"
echo "================================"
echo
echo "To push to Docker Hub:"
echo "  docker push blazux/llm-lab:latest"
