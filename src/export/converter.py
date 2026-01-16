"""
Weight Converter for HuggingFace Export

Handles converting LLM-Lab checkpoint weights to HuggingFace-compatible formats.
"""

import re
import torch
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


class ExportFormat(Enum):
    """Supported export formats"""
    LLAMA = "llama"          # Llama-compatible (GQA/MHA + RoPE + SwiGLU)
    MIXTRAL = "mixtral"      # Mixtral-compatible (MoE)
    CUSTOM = "custom"        # Custom format (requires trust_remote_code)


@dataclass
class FormatInfo:
    """Information about detected export format"""
    format: ExportFormat
    reason: str
    vllm_compatible: bool
    model_type: str
    architectures: list


# Weight mapping from LLM-Lab to Llama format
LLAMA_WEIGHT_MAP = {
    # Embeddings
    "token_embedding.weight": "model.embed_tokens.weight",

    # Attention (per layer - {i} is layer index)
    "transformer_blocks.{i}.attention.q_proj.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "transformer_blocks.{i}.attention.q_proj.bias": "model.layers.{i}.self_attn.q_proj.bias",
    "transformer_blocks.{i}.attention.k_proj.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "transformer_blocks.{i}.attention.k_proj.bias": "model.layers.{i}.self_attn.k_proj.bias",
    "transformer_blocks.{i}.attention.v_proj.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "transformer_blocks.{i}.attention.v_proj.bias": "model.layers.{i}.self_attn.v_proj.bias",
    "transformer_blocks.{i}.attention.w_o.weight": "model.layers.{i}.self_attn.o_proj.weight",

    # QK norms (Llama 3.x style - may need adjustment for older Llama)
    "transformer_blocks.{i}.attention.q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
    "transformer_blocks.{i}.attention.k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",

    # Feed-forward (SwiGLU)
    "transformer_blocks.{i}.feed_forward.gate_proj.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "transformer_blocks.{i}.feed_forward.up_proj.weight": "model.layers.{i}.mlp.up_proj.weight",
    "transformer_blocks.{i}.feed_forward.down_proj.weight": "model.layers.{i}.mlp.down_proj.weight",

    # Standard FFN (for non-gated activations)
    "transformer_blocks.{i}.feed_forward.linear1.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "transformer_blocks.{i}.feed_forward.linear1.bias": "model.layers.{i}.mlp.gate_proj.bias",
    "transformer_blocks.{i}.feed_forward.linear2.weight": "model.layers.{i}.mlp.down_proj.weight",
    "transformer_blocks.{i}.feed_forward.linear2.bias": "model.layers.{i}.mlp.down_proj.bias",

    # Layer norms
    "transformer_blocks.{i}.norm1.weight": "model.layers.{i}.input_layernorm.weight",
    "transformer_blocks.{i}.norm2.weight": "model.layers.{i}.post_attention_layernorm.weight",

    # Final norm and head
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Weight mapping from LLM-Lab to Mixtral format (MoE)
MIXTRAL_WEIGHT_MAP = {
    # Embeddings
    "token_embedding.weight": "model.embed_tokens.weight",

    # Attention (same as Llama)
    "transformer_blocks.{i}.attention.q_proj.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "transformer_blocks.{i}.attention.q_proj.bias": "model.layers.{i}.self_attn.q_proj.bias",
    "transformer_blocks.{i}.attention.k_proj.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "transformer_blocks.{i}.attention.k_proj.bias": "model.layers.{i}.self_attn.k_proj.bias",
    "transformer_blocks.{i}.attention.v_proj.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "transformer_blocks.{i}.attention.v_proj.bias": "model.layers.{i}.self_attn.v_proj.bias",
    "transformer_blocks.{i}.attention.w_o.weight": "model.layers.{i}.self_attn.o_proj.weight",

    # QK norms
    "transformer_blocks.{i}.attention.q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
    "transformer_blocks.{i}.attention.k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",

    # MoE router
    "transformer_blocks.{i}.feed_forward.router.weight": "model.layers.{i}.block_sparse_moe.gate.weight",

    # MoE experts (per expert - {j} is expert index)
    "transformer_blocks.{i}.feed_forward.experts.{j}.gate_proj.weight": "model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight",
    "transformer_blocks.{i}.feed_forward.experts.{j}.up_proj.weight": "model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight",
    "transformer_blocks.{i}.feed_forward.experts.{j}.down_proj.weight": "model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight",

    # Layer norms
    "transformer_blocks.{i}.norm1.weight": "model.layers.{i}.input_layernorm.weight",
    "transformer_blocks.{i}.norm2.weight": "model.layers.{i}.post_attention_layernorm.weight",

    # Final norm and head
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def detect_export_format(model_config) -> FormatInfo:
    """
    Detect the best export format based on model configuration.

    Args:
        model_config: ModelConfig object or dict with model configuration

    Returns:
        FormatInfo with detected format and metadata
    """
    # Handle both dict and ModelConfig
    if hasattr(model_config, '__dict__'):
        config = model_config
        get = lambda k, d=None: getattr(config, k, d)
    else:
        config = model_config
        get = lambda k, d=None: config.get(k, d)

    architecture = get('model_architecture', 'transformer')
    attention_type = get('attention_type', 'gqa')
    positional_encoding = get('positional_encoding', 'rope')
    activation = get('activation', 'swiglu')
    use_moe = get('use_moe', False)

    # Mamba2 architecture -> custom
    if architecture == "mamba2":
        return FormatInfo(
            format=ExportFormat.CUSTOM,
            reason="Mamba2 architecture requires custom model code",
            vllm_compatible=False,
            model_type="llm-lab-mamba2",
            architectures=["Mamba2LLM"]
        )

    # MLA attention -> custom
    if attention_type == "mla":
        return FormatInfo(
            format=ExportFormat.CUSTOM,
            reason="Multi-Head Latent Attention (MLA) requires custom model code",
            vllm_compatible=False,
            model_type="llm-lab-transformer",
            architectures=["LLMLabForCausalLM"]
        )

    # MoE -> Mixtral format
    if use_moe:
        return FormatInfo(
            format=ExportFormat.MIXTRAL,
            reason="MoE architecture maps to Mixtral format",
            vllm_compatible=True,
            model_type="mixtral",
            architectures=["MixtralForCausalLM"]
        )

    # Standard transformer with compatible options -> Llama format
    compatible_attention = attention_type in ["gqa", "mha", "mqa"]
    compatible_pos_enc = positional_encoding in ["rope", "yarn"]
    compatible_activation = activation in ["swiglu", "silu", "gelu"]

    if compatible_attention and compatible_pos_enc and compatible_activation:
        return FormatInfo(
            format=ExportFormat.LLAMA,
            reason="Architecture compatible with Llama format",
            vllm_compatible=True,
            model_type="llama",
            architectures=["LlamaForCausalLM"]
        )

    # Non-standard positional encoding
    if positional_encoding in ["sinusoidal", "learned", "alibi"]:
        return FormatInfo(
            format=ExportFormat.CUSTOM,
            reason=f"Positional encoding '{positional_encoding}' requires custom model code",
            vllm_compatible=False,
            model_type="llm-lab-transformer",
            architectures=["LLMLabForCausalLM"]
        )

    # Non-standard activation
    if activation in ["geglu", "reglu", "relu", "leaky_relu"]:
        return FormatInfo(
            format=ExportFormat.CUSTOM,
            reason=f"Activation '{activation}' requires custom model code",
            vllm_compatible=False,
            model_type="llm-lab-transformer",
            architectures=["LLMLabForCausalLM"]
        )

    # Fallback to custom
    return FormatInfo(
        format=ExportFormat.CUSTOM,
        reason="Non-standard configuration requires custom model code",
        vllm_compatible=False,
        model_type="llm-lab-transformer",
        architectures=["LLMLabForCausalLM"]
    )


def _apply_weight_mapping(
    state_dict: Dict[str, torch.Tensor],
    weight_map: Dict[str, str],
    num_layers: int,
    num_experts: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Apply weight name mapping to state dict.

    Args:
        state_dict: Original state dict with LLM-Lab weight names
        weight_map: Mapping from LLM-Lab names to target format names
        num_layers: Number of transformer layers
        num_experts: Number of MoE experts (0 if not MoE)

    Returns:
        New state dict with remapped weight names
    """
    new_state_dict = {}

    # Build full mapping with layer indices expanded
    full_map = {}
    for src_pattern, dst_pattern in weight_map.items():
        if "{i}" in src_pattern:
            for i in range(num_layers):
                if "{j}" in src_pattern:
                    # MoE expert pattern
                    for j in range(num_experts):
                        src = src_pattern.replace("{i}", str(i)).replace("{j}", str(j))
                        dst = dst_pattern.replace("{i}", str(i)).replace("{j}", str(j))
                        full_map[src] = dst
                else:
                    src = src_pattern.replace("{i}", str(i))
                    dst = dst_pattern.replace("{i}", str(i))
                    full_map[src] = dst
        else:
            full_map[src_pattern] = dst_pattern

    # Apply mapping
    unmapped_keys = []
    for key, value in state_dict.items():
        if key in full_map:
            new_state_dict[full_map[key]] = value
        else:
            # Keep unmapped keys as-is (might be custom additions)
            unmapped_keys.append(key)
            new_state_dict[key] = value

    if unmapped_keys:
        print(f"Warning: {len(unmapped_keys)} weights not in mapping, kept as-is:")
        for key in unmapped_keys[:5]:
            print(f"  - {key}")
        if len(unmapped_keys) > 5:
            print(f"  ... and {len(unmapped_keys) - 5} more")

    return new_state_dict


def remap_weights(
    state_dict: Dict[str, torch.Tensor],
    model_config,
    target_format: ExportFormat
) -> Dict[str, torch.Tensor]:
    """
    Remap weight names from LLM-Lab format to target HuggingFace format.

    Args:
        state_dict: Original state dict
        model_config: Model configuration
        target_format: Target export format

    Returns:
        State dict with remapped weight names
    """
    # Handle both dict and ModelConfig
    if hasattr(model_config, '__dict__'):
        num_layers = model_config.n_layers
        num_experts = getattr(model_config, 'num_experts', 0) or 0
        use_moe = getattr(model_config, 'use_moe', False)
    else:
        num_layers = model_config.get('n_layers', model_config.get('num_hidden_layers', 24))
        num_experts = model_config.get('num_experts', 0) or 0
        use_moe = model_config.get('use_moe', False)

    if target_format == ExportFormat.LLAMA:
        return _apply_weight_mapping(state_dict, LLAMA_WEIGHT_MAP, num_layers)
    elif target_format == ExportFormat.MIXTRAL:
        return _apply_weight_mapping(state_dict, MIXTRAL_WEIGHT_MAP, num_layers, num_experts)
    else:
        # Custom format - keep original names
        return state_dict


def convert_checkpoint(
    checkpoint_path: str,
    model_config=None,
    target_format: Optional[ExportFormat] = None,
    output_path: Optional[str] = None,
    use_safetensors: bool = True
) -> Tuple[Dict[str, torch.Tensor], FormatInfo]:
    """
    Convert an LLM-Lab checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        model_config: Model configuration (loaded from checkpoint if not provided)
        target_format: Target format (auto-detected if not provided)
        output_path: Path to save converted weights (optional)
        use_safetensors: Use safetensors format (recommended)

    Returns:
        Tuple of (converted state_dict, format_info)
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint

    # Get model config from checkpoint if not provided
    if model_config is None:
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            raise ValueError("model_config not provided and not found in checkpoint")

    # Detect format if not specified
    if target_format is None:
        format_info = detect_export_format(model_config)
        target_format = format_info.format
    else:
        format_info = detect_export_format(model_config)
        format_info.format = target_format

    print(f"Export format: {target_format.value}")
    print(f"Reason: {format_info.reason}")
    print(f"vLLM compatible: {format_info.vllm_compatible}")

    # Remap weights
    print("Remapping weights...")
    converted_state_dict = remap_weights(state_dict, model_config, target_format)

    # Save if output path provided
    if output_path:
        print(f"Saving to {output_path}...")
        if use_safetensors:
            try:
                from safetensors.torch import save_file
                # Ensure all tensors are contiguous
                converted_state_dict = {
                    k: v.contiguous() for k, v in converted_state_dict.items()
                }
                save_file(converted_state_dict, output_path)
            except ImportError:
                print("Warning: safetensors not installed, falling back to torch.save")
                torch.save(converted_state_dict, output_path)
        else:
            torch.save(converted_state_dict, output_path)

    return converted_state_dict, format_info
