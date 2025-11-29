from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Import from existing LLM-Lab source
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / 'src'))

from config.config import ModelConfig

router = APIRouter()

# Component definitions for the frontend
COMPONENT_LIBRARY = {
    "architectures": [
        {"id": "transformer", "name": "Transformer", "color": "#8b5cf6"},
        {"id": "mamba2", "name": "Mamba2", "color": "#06b6d4"}
    ],
    "positional_encodings": [
        {"id": "rope", "name": "RoPE", "description": "Rotary Position Embeddings"},
        {"id": "sinusoidal", "name": "Sinusoidal", "description": "Original Transformer"},
        {"id": "alibi", "name": "ALiBi", "description": "Attention with Linear Biases"},
        {"id": "yarn", "name": "YARN", "description": "Yet Another RoPE Extension"}
    ],
    "attention_types": [
        {"id": "mha", "name": "MHA", "description": "Multi-Head Attention"},
        {"id": "mqa", "name": "MQA", "description": "Multi-Query Attention"},
        {"id": "gqa", "name": "GQA", "description": "Grouped-Query Attention"},
        {"id": "mla", "name": "MLA", "description": "Multi-Head Latent Attention"}
    ],
    "activations": [
        {"id": "relu", "name": "ReLU"},
        {"id": "gelu", "name": "GELU"},
        {"id": "silu", "name": "SiLU"},
        {"id": "leaky_relu", "name": "Leaky ReLU"},
        {"id": "swiglu", "name": "SwiGLU"}
    ],
    "normalizations": [
        {"id": "layernorm", "name": "LayerNorm"},
        {"id": "rmsnorm", "name": "RMSNorm"}
    ],
    "optimizers": [
        {"id": "adamw", "name": "AdamW", "color": "#ef4444"},
        {"id": "muon", "name": "Muon", "color": "#f59e0b"},
        {"id": "lion", "name": "Lion", "color": "#10b981"},
        {"id": "sophia", "name": "Sophia", "color": "#3b82f6"}
    ],
    "schedulers": [
        {"id": "none", "name": "Constant"},
        {"id": "cosine", "name": "Cosine"},
        {"id": "linear", "name": "Linear"},
        {"id": "polynomial", "name": "Polynomial"}
    ]
}

class ModelConfigRequest(BaseModel):
    model_architecture: str
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    d_model: int = 896
    n_layers: int = 24
    max_seq_len: int = 1024
    positional_encoding: Optional[str] = "rope"
    attention_type: Optional[str] = "gqa"
    activation: str = "swiglu"
    n_heads: int = 14
    n_kv_heads: Optional[int] = 2
    d_ff: Optional[int] = None
    norm_type: str = "rmsnorm"
    dropout: float = 0.0

class ConfigValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    parameter_count: Optional[int] = None
    memory_estimate_gb: Optional[float] = None

@router.get("/components")
async def get_components():
    """Get all available component types and options"""
    return COMPONENT_LIBRARY

@router.post("/validate-config")
async def validate_config(config_data: Dict[str, Any]):
    """Validate a model configuration"""
    try:
        # Try to create ModelConfig from the data
        model_config = ModelConfig(**config_data)

        # Calculate parameter count (approximate)
        d_model = config_data.get("d_model", 896)
        n_layers = config_data.get("n_layers", 24)
        vocab_size = config_data.get("vocab_size", 151936)
        d_ff = config_data.get("d_ff", d_model * 4)

        # Rough parameter count calculation
        embedding_params = vocab_size * d_model
        attention_params = n_layers * (4 * d_model * d_model)  # Q, K, V, O projections
        ffn_params = n_layers * (2 * d_model * d_ff)  # Up and down projections
        total_params = embedding_params + attention_params + ffn_params

        # Memory estimate (assuming fp32)
        memory_gb = (total_params * 4) / (1024**3)

        warnings = []
        if total_params > 1_000_000_000:
            warnings.append(f"Large model: {total_params/1e9:.2f}B parameters")
        if memory_gb > 16:
            warnings.append(f"High memory requirement: {memory_gb:.2f} GB")

        return ConfigValidationResponse(
            valid=True,
            errors=[],
            warnings=warnings,
            parameter_count=total_params,
            memory_estimate_gb=round(memory_gb, 2)
        )
    except Exception as e:
        return ConfigValidationResponse(
            valid=False,
            errors=[str(e)],
            warnings=[]
        )

@router.post("/generate-config")
async def generate_config(config_data: Dict[str, Any]):
    """Generate a complete ModelConfig from visual layout"""
    try:
        model_config = ModelConfig(**config_data)
        # Convert to dict for JSON serialization
        config_dict = {
            k: v for k, v in model_config.__dict__.items()
        }
        return {
            "success": True,
            "config": config_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/presets")
async def get_presets():
    """Get pre-configured model presets"""
    return {
        "presets": [
            {
                "id": "tiny",
                "name": "Tiny Model (Testing)",
                "description": "Small model for quick testing",
                "config": {
                    "model_architecture": "transformer",
                    "d_model": 256,
                    "n_layers": 6,
                    "n_heads": 4,
                    "n_kv_heads": 2,
                    "max_seq_len": 512,
                    "positional_encoding": "rope",
                    "attention_type": "gqa",
                    "activation": "swiglu"
                }
            },
            {
                "id": "small",
                "name": "Small Model (125M)",
                "description": "GPT-2 small equivalent",
                "config": {
                    "model_architecture": "transformer",
                    "d_model": 768,
                    "n_layers": 12,
                    "n_heads": 12,
                    "n_kv_heads": 4,
                    "max_seq_len": 1024,
                    "positional_encoding": "rope",
                    "attention_type": "gqa",
                    "activation": "swiglu"
                }
            },
            {
                "id": "medium",
                "name": "Medium Model (350M)",
                "description": "GPT-2 medium equivalent",
                "config": {
                    "model_architecture": "transformer",
                    "d_model": 1024,
                    "n_layers": 24,
                    "n_heads": 16,
                    "n_kv_heads": 4,
                    "max_seq_len": 2048,
                    "positional_encoding": "rope",
                    "attention_type": "gqa",
                    "activation": "swiglu"
                }
            }
        ]
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LLM Lab GUI API"}
