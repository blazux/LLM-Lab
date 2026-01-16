"""
LLM-Lab Export Module

Provides functionality to export trained models to HuggingFace-compatible formats
and publish them to HuggingFace Hub.

Supported export formats:
- llama: For GQA/MHA + RoPE + SwiGLU models (vLLM compatible)
- mixtral: For MoE models (vLLM compatible)
- custom: For MLA, Mamba2, or other custom architectures (requires trust_remote_code)
"""

from .converter import (
    convert_checkpoint,
    detect_export_format,
    remap_weights,
    ExportFormat,
)
from .config_mapper import (
    generate_hf_config,
    get_architecture_info,
)
from .model_card import generate_model_card
from .hub import (
    export_to_hub,
    export_to_local,
    prepare_export_directory,
)

__all__ = [
    # Converter
    "convert_checkpoint",
    "detect_export_format",
    "remap_weights",
    "ExportFormat",
    # Config
    "generate_hf_config",
    "get_architecture_info",
    # Model card
    "generate_model_card",
    # Hub
    "export_to_hub",
    "export_to_local",
    "prepare_export_directory",
]
