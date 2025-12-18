# LLM-Lab

A flexible framework for training and fine-tuning Large Language Models from scratch, with support for both Transformer and Mamba2 architectures.

## Features

- **Multiple Architectures**: Transformer (MHA, MQA, GQA) and Mamba2 State Space Models
- **Complete Training Pipeline**: Base training → SFT → RLHF
- **RLHF Algorithms**: PPO, DPO, and GRPO
- **LoRA Support**: Parameter-efficient fine-tuning for both architectures
- **Modern Optimizers**: AdamW, Adafactor, Lion, Sophia, Muon
- **Flexible Configuration**: Interactive CLI or JSON configs

## Quick Start

### Option 1: Docker (Recommended)

We provide two Docker images to suit different needs:

#### Standard Image (Universal Compatibility)
Lightweight image that works on any system with GPU or CPU. Uses PyTorch-based Mamba2 implementation.

**Best for:**
- Quick testing and development
- Systems without NVIDIA GPUs
- Maximum compatibility across hardware
- Smaller Mamba2 models (reduce batch size/sequence length for large models)

```bash
docker run -d -p 8000:8000 \
  --gpus all \
  -v $(pwd)/checkpoints:/app/gui/backend/checkpoints \
  -v $(pwd)/cache:/app/cache \
  --name llm-lab \
  blazux/llm-lab:latest
```

#### CUDA Image (High Performance)
Optimized image with CUDA 12.8 and mamba-ssm kernels compiled for multiple GPU architectures.

**Best for:**
- Production Mamba2 training (100x more memory efficient)
- Large models and long sequences
- NVIDIA GPUs: V100, RTX 20xx/30xx/40xx/50xx, A100, H100

**Requirements:**
- NVIDIA GPU with CUDA support (compute capability 7.0+)
- Docker with NVIDIA Container Toolkit (nvidia-docker)

```bash
docker run -d -p 8000:8000 \
  --gpus all \
  -v $(pwd)/checkpoints:/app/gui/backend/checkpoints \
  -v $(pwd)/cache:/app/cache \
  --name llm-lab \
  blazux/llm-lab:cuda
```

**Note:** The CUDA image is significantly larger (~8GB vs ~2GB) due to compiled kernels for multiple GPU architectures.

---

Access the web interface at http://localhost:8000

Launch the CLI:
```bash
docker exec -it llm-lab ../../llm-lab.sh
```

**The web interface is a work in progress and is far from offering all the features of the CLI version.**

#### Web Interface

The web interface provides a visual workflow for configuring and training your models:

**Model Configuration**
![Model Configuration](model_config.png)

**Training Configuration**
![Training Configuration](training_config.png)

**Training Monitor**
![Training Monitor](training_monitor.png)

### Option 2: Local Installation

```bash
pip install -r requirements.txt
```

**For Mamba2 support (requires CUDA):**
```bash
pip install -r requirements-mamba.txt
```

**Note**: This installs the optimized `mamba-ssm` CUDA kernels which are required for training Mamba2 models efficiently. Requires CUDA toolkit (nvcc) to be installed.

### Training Workflow

```bash
# Launch interactive CLI
./llm-lab.sh
```

**Typical pipeline:**

1. **Configure your model** (Option 1)
   - Choose architecture: Transformer or Mamba2
   - Set model size, attention type, activation, etc.

2. **Train from scratch** (Option 2)
   - Configure datasets, optimizer, learning rate, etc.
   - Run base training on text corpora

3. **Supervised Fine-Tuning** (Option 3)
   - Fine-tune on instruction datasets
   - Optional: Use LoRA for memory efficiency

4. **RLHF Training** (Option 4)
   - Choose algorithm: PPO, DPO, or GRPO
   - Align model with human preferences

5. **Test your model** (Option 6)
   - Run inference to test outputs

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA
- GPU with bfloat16 support (recommended)
- See `requirements.txt` for full dependencies

## Documentation

📚 **[Complete Documentation](docs/index.md)** - Single entry point for all documentation

**Main guides:**
- [Complete Training Guide](docs/full-guide.md) - Step-by-step guide for all training stages
- [Mamba2 Guide](docs/mamba2.md) - State space model documentation


## Contributing

Feel free to open issues or submit pull requests for bugs, features, or improvements.

## License

Under MIT License.
