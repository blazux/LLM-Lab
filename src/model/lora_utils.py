"""
LoRA configuration utilities for parameter-efficient fine-tuning.
Provides smart preset handling based on model architecture.
"""

from typing import List, Dict
from config import ModelConfig


def get_lora_target_modules(model_config: ModelConfig, preset: str) -> List[str]:
    """
    Get LoRA target modules based on model configuration and preset.

    Args:
        model_config: Model configuration containing architecture details
        preset: One of "minimal", "attention_only", "ffn_only", "all", or "custom"

    Returns:
        List of module names to apply LoRA to
    """
    if model_config.model_architecture == "mamba2":
        # Mamba2-specific modules
        # The mamba-ssm Mamba2 layer has:
        # - in_proj: projects input to 2*d_inner for x and z paths
        # - out_proj: projects d_inner back to d_model
        # - conv1d: optional convolution layer
        # - dt_proj: time-step projection

        mamba_modules = ["in_proj", "out_proj"]  # Main projections
        mamba_extended = mamba_modules + ["dt_proj"]  # Include time-step projection

        preset_mapping = {
            "minimal": ["in_proj"],  # Input projection only
            "attention_only": mamba_modules,  # In Mamba2, equivalent to "main SSM layers"
            "ffn_only": mamba_modules,  # Same as attention_only for Mamba2
            "all": mamba_extended,  # All trainable projections
        }

    else:  # transformer
        # Define attention modules (common to all configurations)
        attention_modules = ["q_proj", "k_proj", "v_proj", "w_o"]

        # Define FFN modules based on activation type
        if model_config.activation == "swiglu":
            ffn_modules = ["gate_proj", "up_proj", "down_proj"]
        else:
            ffn_modules = ["linear1", "linear2"]

        # Map presets to target modules
        preset_mapping = {
            "minimal": ["q_proj", "v_proj"],
            "attention_only": attention_modules,
            "ffn_only": ffn_modules,
            "all": attention_modules + ffn_modules,
        }

    if preset == "custom":
        # For custom, caller will provide target_modules directly
        return []

    if preset not in preset_mapping:
        raise ValueError(
            f"Invalid preset '{preset}'. Choose from: {list(preset_mapping.keys())} or 'custom'"
        )

    return preset_mapping[preset]


def get_available_presets(model_config: ModelConfig) -> Dict[str, str]:
    """
    Get available LoRA presets with descriptions based on model configuration.

    Args:
        model_config: Model configuration containing architecture details

    Returns:
        Dictionary mapping preset names to descriptions
    """
    if model_config.model_architecture == "mamba2":
        presets = {
            "minimal": "Input projection only (fastest, lowest VRAM)",
            "attention_only": "Main SSM layers: in_proj, out_proj",
            "ffn_only": "Main SSM layers: in_proj, out_proj (same as attention_only)",
            "all": "All projections including time-step (maximum adaptation)",
            "custom": "Specify target modules manually"
        }
    else:  # transformer
        # Determine FFN type for descriptions
        if model_config.activation == "swiglu":
            ffn_desc = "gate_proj, up_proj, down_proj"
        else:
            ffn_desc = "linear1, linear2"

        presets = {
            "minimal": "Q+V projections only (fastest, lowest VRAM)",
            "attention_only": "All attention layers: q_proj, k_proj, v_proj, w_o",
            "ffn_only": f"All FFN layers: {ffn_desc}",
            "all": f"Attention + FFN (maximum adaptation capability)",
            "custom": "Specify target modules manually"
        }

    return presets


def apply_lora_to_model(model, model_config: ModelConfig, lora_config_dict: dict):
    """
    Apply LoRA to a model using PEFT library.

    Args:
        model: PyTorch model to apply LoRA to
        model_config: Model configuration for determining target modules
        lora_config_dict: Dictionary containing LoRA configuration:
            - use_lora: bool (if False, returns model unchanged)
            - lora_preset: str (one of the presets or "custom")
            - lora_target_modules: List[str] (used if preset is "custom")
            - lora_r: int (LoRA rank)
            - lora_alpha: int (LoRA alpha scaling)
            - lora_dropout: float (LoRA dropout rate)

    Returns:
        Model with LoRA applied (or original model if use_lora=False)
    """
    if not lora_config_dict.get("use_lora", False):
        return model

    try:
        from peft import get_peft_model, LoraConfig
    except ImportError:
        raise ImportError(
            "PEFT library not found. Install it with: pip install peft>=0.4.0"
        )

    # Determine target modules
    preset = lora_config_dict.get("lora_preset", "minimal")
    if preset == "custom":
        target_modules = lora_config_dict.get("lora_target_modules", ["q_proj", "v_proj"])
    else:
        target_modules = get_lora_target_modules(model_config, preset)

    # Create PEFT LoRA config
    peft_config = LoraConfig(
        r=lora_config_dict.get("lora_r", 8),
        lora_alpha=lora_config_dict.get("lora_alpha", 16),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)

    # Print trainable parameters info
    model.print_trainable_parameters()

    return model
