"""
Config Mapper for HuggingFace Export

Generates HuggingFace-compatible config.json files based on model configuration
and target export format.
"""

import json
from typing import Dict, Any, Optional
from .converter import ExportFormat, FormatInfo


def get_architecture_info(model_config) -> Dict[str, Any]:
    """
    Extract architecture information from model config.

    Args:
        model_config: ModelConfig object or dict

    Returns:
        Dict with normalized architecture information
    """
    # Handle both dict and ModelConfig
    if hasattr(model_config, '__dict__'):
        get = lambda k, d=None: getattr(model_config, k, d)
    else:
        get = lambda k, d=None: model_config.get(k, d)

    return {
        "model_architecture": get('model_architecture', 'transformer'),
        "d_model": get('d_model', 896),
        "n_layers": get('n_layers', 24),
        "n_heads": get('n_heads', 14),
        "n_kv_heads": get('n_kv_heads'),
        "d_ff": get('d_ff', 4864),
        "vocab_size": get('vocab_size', 151936),
        "max_seq_len": get('max_seq_len', 1024),
        "dropout": get('dropout', 0.0),
        "norm_type": get('norm_type', 'rmsnorm'),
        "norm_eps": get('norm_eps', 1e-6),
        "positional_encoding": get('positional_encoding', 'rope'),
        "attention_type": get('attention_type', 'gqa'),
        "activation": get('activation', 'swiglu'),
        "attention_bias": get('attention_bias', False),
        "tie_word_embeddings": get('tie_word_embeddings', True),
        "tokenizer_name": get('tokenizer_name', 'Qwen/Qwen2.5-0.5B'),
        # MoE
        "use_moe": get('use_moe', False),
        "num_experts": get('num_experts'),
        "num_experts_per_token": get('num_experts_per_token'),
        # MLA
        "d_latent": get('d_latent'),
        "d_rope_latent": get('d_rope_latent'),
        # Mamba2
        "state_size": get('state_size', 64),
        "expand_factor": get('expand_factor', 2),
        "headdim": get('headdim', 64),
        "ngroups": get('ngroups', 1),
        "chunk_size": get('chunk_size', 256),
        "conv_kernel_size": get('conv_kernel_size', 4),
    }


def _generate_llama_config(arch_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Llama-compatible config.json"""

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "bfloat16",

        # Core architecture
        "hidden_size": arch_info["d_model"],
        "intermediate_size": arch_info["d_ff"],
        "num_hidden_layers": arch_info["n_layers"],
        "num_attention_heads": arch_info["n_heads"],
        "vocab_size": arch_info["vocab_size"],
        "max_position_embeddings": arch_info["max_seq_len"],

        # Attention
        "attention_bias": arch_info["attention_bias"],
        "attention_dropout": arch_info["dropout"],

        # GQA support
        "num_key_value_heads": arch_info["n_kv_heads"] or arch_info["n_heads"],

        # Normalization (Llama uses RMSNorm)
        "rms_norm_eps": arch_info["norm_eps"],

        # Activation
        "hidden_act": "silu",  # Llama uses SiLU in SwiGLU

        # Positional encoding
        "rope_theta": 10000.0,
        "rope_scaling": None,

        # Other
        "tie_word_embeddings": arch_info["tie_word_embeddings"],
        "use_cache": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }

    return config


def _generate_mixtral_config(arch_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Mixtral-compatible config.json for MoE models"""

    config = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "torch_dtype": "bfloat16",

        # Core architecture
        "hidden_size": arch_info["d_model"],
        "intermediate_size": arch_info["d_ff"],
        "num_hidden_layers": arch_info["n_layers"],
        "num_attention_heads": arch_info["n_heads"],
        "vocab_size": arch_info["vocab_size"],
        "max_position_embeddings": arch_info["max_seq_len"],

        # Attention
        "attention_bias": arch_info["attention_bias"],
        "attention_dropout": arch_info["dropout"],

        # GQA support
        "num_key_value_heads": arch_info["n_kv_heads"] or arch_info["n_heads"],

        # Normalization
        "rms_norm_eps": arch_info["norm_eps"],

        # Activation
        "hidden_act": "silu",

        # MoE specific
        "num_local_experts": arch_info["num_experts"],
        "num_experts_per_tok": arch_info["num_experts_per_token"],
        "router_aux_loss_coef": 0.01,  # Mixtral default

        # Positional encoding
        "rope_theta": 10000.0,

        # Other
        "tie_word_embeddings": arch_info["tie_word_embeddings"],
        "use_cache": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }

    return config


def _generate_custom_transformer_config(arch_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate custom LLM-Lab transformer config.json"""

    config = {
        "architectures": ["LLMLabForCausalLM"],
        "model_type": "llm-lab-transformer",
        "torch_dtype": "bfloat16",

        # Auto-mapping for transformers library
        "auto_map": {
            "AutoConfig": "configuration_llmlab.LLMLabConfig",
            "AutoModelForCausalLM": "modeling_llmlab.LLMLabForCausalLM"
        },

        # Core architecture
        "hidden_size": arch_info["d_model"],
        "intermediate_size": arch_info["d_ff"],
        "num_hidden_layers": arch_info["n_layers"],
        "num_attention_heads": arch_info["n_heads"],
        "vocab_size": arch_info["vocab_size"],
        "max_position_embeddings": arch_info["max_seq_len"],

        # Custom fields (LLM-Lab specific)
        "d_model": arch_info["d_model"],
        "n_layers": arch_info["n_layers"],
        "n_heads": arch_info["n_heads"],
        "d_ff": arch_info["d_ff"],

        # Attention configuration
        "attention_type": arch_info["attention_type"],
        "attention_bias": arch_info["attention_bias"],
        "num_key_value_heads": arch_info["n_kv_heads"],

        # Normalization
        "norm_type": arch_info["norm_type"],
        "norm_eps": arch_info["norm_eps"],
        "rms_norm_eps": arch_info["norm_eps"] if arch_info["norm_type"] == "rmsnorm" else None,
        "layer_norm_eps": arch_info["norm_eps"] if arch_info["norm_type"] == "layernorm" else None,

        # Activation
        "hidden_act": arch_info["activation"],
        "activation": arch_info["activation"],

        # Positional encoding
        "positional_encoding": arch_info["positional_encoding"],
        "position_embedding_type": arch_info["positional_encoding"],

        # Dropout
        "dropout": arch_info["dropout"],
        "attention_dropout": arch_info["dropout"],

        # Other
        "tie_word_embeddings": arch_info["tie_word_embeddings"],
        "use_cache": True,
        "is_encoder_decoder": False,
    }

    # Add MLA-specific fields
    if arch_info["attention_type"] == "mla":
        config["d_latent"] = arch_info["d_latent"]
        config["d_rope_latent"] = arch_info["d_rope_latent"]

    # Add MoE fields if applicable
    if arch_info["use_moe"]:
        config["use_moe"] = True
        config["num_experts"] = arch_info["num_experts"]
        config["num_experts_per_token"] = arch_info["num_experts_per_token"]

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}

    return config


def _generate_mamba2_config(arch_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate custom Mamba2 config.json"""

    config = {
        "architectures": ["Mamba2LLM"],
        "model_type": "llm-lab-mamba2",
        "torch_dtype": "bfloat16",

        # Auto-mapping for transformers library
        "auto_map": {
            "AutoConfig": "configuration_llmlab.LLMLabConfig",
            "AutoModelForCausalLM": "modeling_llmlab.Mamba2LLM"
        },

        # Core architecture
        "hidden_size": arch_info["d_model"],
        "num_hidden_layers": arch_info["n_layers"],
        "vocab_size": arch_info["vocab_size"],
        "max_position_embeddings": arch_info["max_seq_len"],

        # Mamba2-specific
        "d_model": arch_info["d_model"],
        "n_layers": arch_info["n_layers"],
        "state_size": arch_info["state_size"],
        "expand_factor": arch_info["expand_factor"],
        "headdim": arch_info["headdim"],
        "ngroups": arch_info["ngroups"],
        "chunk_size": arch_info["chunk_size"],
        "conv_kernel_size": arch_info["conv_kernel_size"],

        # Normalization
        "norm_type": arch_info["norm_type"],
        "norm_eps": arch_info["norm_eps"],
        "rms_norm_eps": arch_info["norm_eps"] if arch_info["norm_type"] == "rmsnorm" else None,

        # Other
        "tie_word_embeddings": arch_info["tie_word_embeddings"],
        "is_encoder_decoder": False,
    }

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}

    return config


def generate_hf_config(
    model_config,
    format_info: FormatInfo,
    training_config=None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate HuggingFace-compatible config.json content.

    Args:
        model_config: Model configuration
        format_info: Export format information
        training_config: Optional training configuration for metadata
        extra_metadata: Additional metadata to include

    Returns:
        Dict representing config.json content
    """
    arch_info = get_architecture_info(model_config)

    # Generate config based on format
    if format_info.format == ExportFormat.LLAMA:
        config = _generate_llama_config(arch_info)
    elif format_info.format == ExportFormat.MIXTRAL:
        config = _generate_mixtral_config(arch_info)
    elif arch_info["model_architecture"] == "mamba2":
        config = _generate_mamba2_config(arch_info)
    else:
        config = _generate_custom_transformer_config(arch_info)

    # Add training metadata if provided
    if training_config:
        if hasattr(training_config, '__dict__'):
            config["_llmlab_training"] = {
                "optimizer": getattr(training_config, 'optimizer', None),
                "lr": getattr(training_config, 'lr', None),
                "max_steps": getattr(training_config, 'max_steps', None),
            }
        elif isinstance(training_config, dict):
            config["_llmlab_training"] = {
                "optimizer": training_config.get('optimizer'),
                "lr": training_config.get('lr'),
                "max_steps": training_config.get('max_steps'),
            }

    # Add tokenizer info
    config["tokenizer_class"] = "AutoTokenizer"
    config["_llmlab_tokenizer"] = arch_info["tokenizer_name"]

    # Add extra metadata
    if extra_metadata:
        config.update(extra_metadata)

    return config


def save_hf_config(config: Dict[str, Any], output_path: str):
    """Save config to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {output_path}")
