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

We provide two Docker images to suit different needs. **Both images use CUDA 12.8 and PyTorch 2.7.0 stable** with support for modern GPUs including the RTX 5090/5080 (Blackwell), RTX 40xx, RTX 30xx, RTX 20xx, A100, H100, and V100.

**Requirements (both images):**
- NVIDIA GPU with CUDA support (compute capability 7.0+)
- Docker with NVIDIA Container Toolkit (nvidia-docker)

#### Standard Image (PyTorch Mamba2)
Uses pure PyTorch implementation for Mamba2 (no mamba-ssm optimized kernels).

**Best for:**
- Quick testing and development
- Transformer architectures (same performance as CUDA image)
- Smaller Mamba2 models with reduced batch size/sequence length
- Faster builds (no kernel compilation)

```bash
docker run -d -p 8000:8000 \
  --gpus all \
  -v $(pwd)/data:/app/data \
  --name llm-lab \
  blazux/llm-lab:latest
```

#### CUDA Image (mamba-ssm Optimized)
Includes mamba-ssm optimized CUDA kernels for high-performance Mamba2 training.

**Best for:**
- Production Mamba2 training (100x more memory efficient than PyTorch fallback)
- Large Mamba2 models with long sequences
- Maximum Mamba2 performance

```bash
docker run -d -p 8000:8000 \
  --gpus all \
  -v $(pwd)/data:/app/data \
  --name llm-lab \
  blazux/llm-lab:cuda
```

**Note:** The CUDA image includes pre-compiled mamba-ssm kernels for multiple GPU architectures (compute capability 7.0 through 12.0). **The only difference between the images is the presence of mamba-ssm.**

**Data Directory Structure:**
The single `data/` volume mount contains all persistent data:
- `data/checkpoints/` - Base training checkpoints
- `data/sft_checkpoints/` - Supervised fine-tuning checkpoints
- `data/rlhf_checkpoints/` - RLHF training checkpoints
- `data/cache/` - Model and dataset cache

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

**Install PyTorch 2.7.0+ with CUDA 12.8:**
```bash
pip install torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Install base requirements:**
```bash
pip install -r requirements.txt
```

**For Mamba2 optimized support (optional):**
```bash
pip install -r requirements-mamba.txt
```

**Note**: Installing `mamba-ssm` provides optimized CUDA kernels for efficient Mamba2 training (100x more memory efficient). Requires CUDA toolkit (nvcc) to be installed. Without mamba-ssm, the framework uses a pure PyTorch fallback implementation.

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

- Python 3.11+
- PyTorch 2.7.0+ with CUDA 12.8
- NVIDIA GPU with CUDA support (compute capability 7.0+)
  - Supported: RTX 5090/5080, RTX 40xx, RTX 30xx, RTX 20xx, A100, H100, V100
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
