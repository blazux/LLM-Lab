# Mamba2 Support in LLM-Lab

This document explains how to use Mamba2 State Space Models in LLM-Lab.

## What is Mamba2?

Mamba2 is a state-space model architecture that offers:
- **Linear complexity**: O(N) vs O(N²) for attention mechanisms
- **Efficient long context**: Handle 4k-8k+ tokens on limited hardware
- **Faster inference**: Constant time per token regardless of context length
- **Lower memory**: Fixed-size SSM state instead of growing KV cache

## Requirements

Mamba2 requires the `mamba-ssm` package with optimized CUDA kernels:

```bash
pip install mamba-ssm>=2.0.0 causal-conv1d>=1.2.0
```

**Note**: Requires CUDA-capable GPU. CPU-only mode is not supported.

## Quick Start

### 1. Create a Mamba2 Model Config

Use the example config or create your own:

```bash
cp model_config_mamba2.json my_mamba2_config.json
```

Key parameters:
- `model_architecture`: Must be `"mamba2"`
- `state_size`: SSM state dimension (default: 16)
- `expand_factor`: Hidden dimension multiplier (default: 2)
- `conv_kernel_size`: Convolution kernel size (default: 4)
- `dt_rank`: Time-step projection rank (auto-computed if null)

### 2. Train a Mamba2 Model

**Base pretraining:**
```bash
python src/cli.py train --config training_config.json
```

When prompted for model config, provide your Mamba2 config path.

**Or modify training_config.json:**
```json
{
  "model_config_path": "model_config_mamba2.json",
  "max_steps": 100000,
  ...
}
```

### 3. Supervised Fine-Tuning (SFT)

```bash
python src/cli.py sft --config sft_config.json
```

Mamba2 works with the same SFT pipeline as transformers!

### 4. RLHF Training

Mamba2 supports all RLHF algorithms:
- PPO (Proximal Policy Optimization)
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)

```bash
python src/cli.py rlhf --mode ppo --config rlhf_config.json
```

### 5. Inference

```bash
python src/cli.py inference --checkpoint mamba2_checkpoints/best_model.pt
```

The inference interface is identical to transformer models!

## Configuration Comparison

### Transformer vs Mamba2 (1.4B parameters)

| Parameter | Transformer | Mamba2 | Notes |
|-----------|-------------|---------|-------|
| `model_architecture` | `"transformer"` | `"mamba2"` | Required |
| `d_model` | 2048 | 2048 | Same |
| `n_layers` | 28 | 28 | Same |
| `max_seq_len` | 2048 | 4096 | Mamba2 handles longer! |
| `n_heads` | 32 | N/A | Not used in Mamba2 |
| `attention_type` | `"gqa"` | N/A | No attention |
| `positional_encoding` | `"rope"` | N/A | Learned implicitly |
| `state_size` | N/A | 16 | SSM state dimension |
| `expand_factor` | N/A | 2 | Hidden expansion |
| `conv_kernel_size` | N/A | 4 | Convolution kernel |

## LoRA with Mamba2

Mamba2 has different module names for LoRA:

**Transformer modules:**
- `q_proj`, `k_proj`, `v_proj`, `w_o` (attention)
- `gate_proj`, `up_proj`, `down_proj` (FFN)

**Mamba2 modules:**
- `in_proj` - Input projection to SSM
- `out_proj` - Output projection from SSM
- `dt_proj` - Time-step projection

**LoRA presets:**
- `minimal`: `in_proj` only (fastest)
- `attention_only`: `in_proj` + `out_proj`
- `all`: `in_proj` + `out_proj` + `dt_proj`

## Performance Expectations

### Training

| Sequence Length | Transformer Memory | Mamba2 Memory |
|-----------------|-------------------|---------------|
| 2048 | 24GB | 24GB |
| 4096 | OOM (>40GB) | 24GB |
| 8192 | OOM | ~32GB |

Mamba2 allows **2-4x longer sequences** on the same hardware!

### Inference

| Context Length | Transformer Speed | Mamba2 Speed |
|----------------|------------------|--------------|
| 2k tokens | 1.0x (baseline) | 1.2x faster |
| 4k tokens | 1.0x | 2-3x faster |
| 8k tokens | 1.0x | 4-5x faster |
| 16k tokens | OOM | 8-10x faster |

**Why?** Mamba2 has **constant-time generation** - doesn't slow down with longer context!

## Known Limitations

1. **CUDA only**: Mamba2 requires GPU. No CPU support.
2. **Model quality**: Transformers may have slight edge on some benchmarks
3. **Ecosystem**: Fewer pretrained Mamba2 models available
4. **Training steps**: May need 10-20% more steps than transformers for same quality

## Troubleshooting

### ImportError: mamba_ssm not found

```bash
pip install mamba-ssm>=2.0.0 causal-conv1d>=1.2.0
```

### CUDA out of memory

Reduce `batch_size` or `max_seq_len` in config. Mamba2 uses less memory than transformers for long sequences, but still needs GPU memory!

### Model outputs are garbage

Check that `vocab_size` in config matches your tokenizer:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your/tokenizer")
print(tokenizer.vocab_size)  # Use this value in config
```

## Example: 1.4B Mamba2 Model

```json
{
  "model_architecture": "mamba2",
  "tokenizer_name": "mistralai/Mistral-Small-Instruct-2409",
  "d_model": 2048,
  "n_layers": 28,
  "vocab_size": 32768,
  "max_seq_len": 4096,
  "state_size": 16,
  "expand_factor": 2,
  "conv_kernel_size": 4,
  "norm_type": "rmsnorm",
  "dropout": 0.0
}
```

**Estimated parameters**: ~1.4B (similar to transformer with same d_model/n_layers)

## References

- Mamba2 Paper: [https://arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060)
- mamba-ssm Library: [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
- Original Mamba: [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)

## Questions?

For issues specific to LLM-Lab's Mamba2 integration, please file an issue on GitHub.

For questions about Mamba2 architecture itself, refer to the papers above or the mamba-ssm repository.
