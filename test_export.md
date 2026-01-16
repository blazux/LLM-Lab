# HuggingFace Export Testing Guide

This document describes how to test the HuggingFace export functionality for all supported model configurations.

## Prerequisites

Before testing, ensure you have:

1. **Dependencies installed:**
   ```bash
   pip install safetensors huggingface_hub transformers
   ```

2. **HuggingFace token (for Hub tests):**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

3. **Trained models:** You need checkpoint files from training. If you don't have any, train a small model first using the CLI or GUI.

---

## Test Matrix

The export module supports different architectures with different export formats:

| Architecture | Attention | MoE | Export Format | vLLM Compatible |
|--------------|-----------|-----|---------------|-----------------|
| Transformer | GQA | No | `llama` | Yes |
| Transformer | MHA | No | `llama` | Yes |
| Transformer | MQA | No | `llama` | Yes |
| Transformer | GQA | Yes | `mixtral` | Yes |
| Transformer | MLA | No | `custom` | No |
| Mamba2 | N/A | N/A | `custom` | No |

---

## Test Cases

### Test 1: Standard Transformer (Llama-compatible)

**Configuration:**
- Architecture: `transformer`
- Attention: `gqa`
- Positional encoding: `rope`
- Activation: `swiglu`
- MoE: `false`

**Expected:** Export as `llama` format, vLLM compatible.

**CLI Test:**
```bash
cd /path/to/LLM-Lab
python -m src.cli

# Choose option 1 to configure a model with:
#   - architecture: transformer
#   - attention_type: gqa
#   - positional_encoding: rope
#   - activation: swiglu

# Choose option 2 to train (or use existing checkpoint)

# Choose option 8 to export:
#   - Select checkpoint
#   - Verify "Export format: llama" is detected
#   - Export to local directory
```

**Verification:**
```bash
# Check exported files
ls -la /app/data/hf_export/

# Expected files:
# - config.json (model_type: "llama")
# - model.safetensors
# - README.md
# - tokenizer.json, tokenizer_config.json, etc.

# Verify config
cat /app/data/hf_export/config.json | jq '.model_type'
# Should output: "llama"

# Test loading with transformers
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('/app/data/hf_export')
tokenizer = AutoTokenizer.from_pretrained('/app/data/hf_export')
print('Model loaded successfully!')
print(f'Parameters: {sum(p.numel() for p in model.parameters())}')
"
```

---

### Test 2: MoE Transformer (Mixtral-compatible)

**Configuration:**
- Architecture: `transformer`
- Attention: `gqa`
- MoE: `true`
- Num experts: `8`
- Experts per token: `2`

**Expected:** Export as `mixtral` format, vLLM compatible.

**CLI Test:**
```bash
# Configure model with MoE enabled, then train and export

# Verify detection
# Should show: "Export format: mixtral"
```

**Verification:**
```bash
cat /app/data/hf_export/config.json | jq '.model_type'
# Should output: "mixtral"

cat /app/data/hf_export/config.json | jq '.num_local_experts'
# Should output: 8
```

---

### Test 3: MLA Attention (Custom format)

**Configuration:**
- Architecture: `transformer`
- Attention: `mla`
- Positional encoding: `rope`

**Expected:** Export as `custom` format, NOT vLLM compatible.

**CLI Test:**
```bash
# Configure model with attention_type: mla

# Export should show:
# "Export format: custom"
# "vLLM compatible: False"
```

**Verification:**
```bash
# Check for custom model files
ls -la /app/data/hf_export/

# Expected additional files:
# - configuration_llmlab.py
# - modeling_llmlab.py

# Check config has auto_map
cat /app/data/hf_export/config.json | jq '.auto_map'

# Test loading with trust_remote_code
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    '/app/data/hf_export',
    trust_remote_code=True
)
print('Custom model loaded successfully!')
"
```

---

### Test 4: Mamba2 (Custom format)

**Configuration:**
- Architecture: `mamba2`

**Expected:** Export as `custom` format, NOT vLLM compatible.

**CLI Test:**
```bash
# Configure model with architecture: mamba2

# Export should show:
# "Export format: custom"
# "Reason: Mamba2 architecture requires custom model code"
```

**Verification:**
```bash
cat /app/data/hf_export/config.json | jq '.model_type'
# Should output: "llm-lab-mamba2"

# Test loading
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    '/app/data/hf_export',
    trust_remote_code=True
)
print('Mamba2 model loaded successfully!')
"
```

---

### Test 5: Non-standard Positional Encoding

**Configuration:**
- Architecture: `transformer`
- Attention: `gqa`
- Positional encoding: `alibi` or `sinusoidal`

**Expected:** Export as `custom` format due to non-RoPE encoding.

**Verification:**
```bash
# Should detect as custom
# Check config.json has custom model type
```

---

### Test 6: Hub Upload

**Prerequisites:** Valid HuggingFace token with write access.

**Test:**
```bash
# Export to Hub
python -m src.cli
# Choose option 8
# Choose "Export and push to HuggingFace Hub"
# Enter repo ID: your-username/test-model
# Enter token or use environment variable
```

**Verification:**
```bash
# Check the Hub
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/models/your-username/test-model

# Verify files exist on Hub
# Test loading from Hub
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('your-username/test-model')
print('Model loaded from Hub!')
"
```

---

### Test 7: API Endpoint Tests

**Test format detection:**
```bash
curl "http://localhost:8000/api/export/detect-format?checkpoint_path=/app/data/best_model.pt"
```

**Test local export:**
```bash
curl -X POST "http://localhost:8000/api/export/local" \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "/app/data/best_model.pt",
    "output_dir": "/app/data/hf_export",
    "model_name": "test-model"
  }'
```

**Test checkpoint listing:**
```bash
curl "http://localhost:8000/api/export/checkpoints"
```

**Test validation:**
```bash
curl -X POST "http://localhost:8000/api/export/validate?output_dir=/app/data/hf_export"
```

---

## Validation Checklist

For each exported model, verify:

- [ ] `config.json` exists and has correct `model_type`
- [ ] `model.safetensors` or `pytorch_model.bin` exists
- [ ] `README.md` exists with proper YAML frontmatter
- [ ] Tokenizer files exist (if `include_tokenizer=True`)
- [ ] For custom formats: `configuration_llmlab.py` and `modeling_llmlab.py` exist
- [ ] Model loads with `transformers.AutoModelForCausalLM`
- [ ] Model can generate text:
  ```python
  inputs = tokenizer("Hello", return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=10)
  print(tokenizer.decode(outputs[0]))
  ```

---

## Common Issues

### 1. "Export module not available"
**Solution:** Ensure the export module is in the Python path:
```bash
cd /path/to/LLM-Lab
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### 2. "safetensors not installed"
**Solution:**
```bash
pip install safetensors
```

### 3. "huggingface_hub not installed"
**Solution:**
```bash
pip install huggingface_hub
```

### 4. "No model_config found in checkpoint"
**Cause:** Old checkpoint format or corrupted file.
**Solution:** Retrain the model or check checkpoint contents:
```python
import torch
ckpt = torch.load("checkpoint.pt", map_location="cpu")
print(ckpt.keys())
```

### 5. Custom model fails to load
**Cause:** Missing custom model files in export.
**Solution:** Ensure `configuration_llmlab.py` and `modeling_llmlab.py` are in the export directory.

### 6. vLLM doesn't recognize the model
**Cause:** Model exported as `custom` format instead of `llama`/`mixtral`.
**Solution:** Check model configuration - only GQA/MHA + RoPE + SwiGLU is Llama-compatible.

---

## Quick Test Script

Save this as `test_export.py` and run it:

```python
#!/usr/bin/env python3
"""Quick test script for export functionality"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from export import (
    detect_export_format,
    convert_checkpoint,
    export_to_local,
    ExportFormat,
)
from config import ModelConfig
import tempfile

def test_format_detection():
    """Test format detection for different configs"""
    print("Testing format detection...")

    # Test 1: Standard GQA + RoPE + SwiGLU -> Llama
    config = ModelConfig(
        model_architecture="transformer",
        attention_type="gqa",
        positional_encoding="rope",
        activation="swiglu",
        use_moe=False,
    )
    info = detect_export_format(config)
    assert info.format == ExportFormat.LLAMA, f"Expected LLAMA, got {info.format}"
    assert info.vllm_compatible == True
    print("  ✓ GQA + RoPE + SwiGLU -> llama")

    # Test 2: MoE -> Mixtral
    config = ModelConfig(
        model_architecture="transformer",
        attention_type="gqa",
        use_moe=True,
        num_experts=8,
        num_experts_per_token=2,
    )
    info = detect_export_format(config)
    assert info.format == ExportFormat.MIXTRAL, f"Expected MIXTRAL, got {info.format}"
    assert info.vllm_compatible == True
    print("  ✓ MoE -> mixtral")

    # Test 3: MLA -> Custom
    config = ModelConfig(
        model_architecture="transformer",
        attention_type="mla",
    )
    info = detect_export_format(config)
    assert info.format == ExportFormat.CUSTOM, f"Expected CUSTOM, got {info.format}"
    assert info.vllm_compatible == False
    print("  ✓ MLA -> custom")

    # Test 4: Mamba2 -> Custom
    config = ModelConfig(
        model_architecture="mamba2",
    )
    info = detect_export_format(config)
    assert info.format == ExportFormat.CUSTOM, f"Expected CUSTOM, got {info.format}"
    assert info.vllm_compatible == False
    print("  ✓ Mamba2 -> custom")

    print("All format detection tests passed!")

if __name__ == "__main__":
    test_format_detection()
```

Run with:
```bash
python test_export.py
```

---

## Performance Testing

For large models, monitor:

1. **Memory usage during export:**
   ```bash
   watch -n 1 nvidia-smi  # For GPU
   watch -n 1 free -h     # For CPU RAM
   ```

2. **Export time:**
   The export should complete within reasonable time:
   - Small models (<1B): < 1 minute
   - Medium models (1-7B): 1-5 minutes
   - Large models (>7B): 5-30 minutes

3. **Output file sizes:**
   - `model.safetensors` should be ~2x model parameters (bf16)
   - Example: 1B params ≈ 2GB safetensors file
