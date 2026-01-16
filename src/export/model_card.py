"""
Model Card Generator for HuggingFace Export

Generates README.md files with model metadata, usage instructions,
and training information.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from .converter import ExportFormat, FormatInfo
from .config_mapper import get_architecture_info


def _format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes"""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _estimate_parameters(arch_info: Dict[str, Any]) -> int:
    """Estimate total parameters from architecture info"""
    d_model = arch_info["d_model"]
    n_layers = arch_info["n_layers"]
    d_ff = arch_info["d_ff"]
    vocab_size = arch_info["vocab_size"]
    n_heads = arch_info["n_heads"]
    n_kv_heads = arch_info.get("n_kv_heads") or n_heads

    if arch_info["model_architecture"] == "mamba2":
        # Mamba2 parameter estimation
        expand = arch_info.get("expand_factor", 2)
        d_inner = d_model * expand
        state_size = arch_info.get("state_size", 64)

        embed_params = vocab_size * d_model
        layer_params = (
            d_model * 2 * d_inner +  # in_proj
            d_inner * 4 +  # conv
            d_inner * state_size * 3 +  # A, B, C
            d_inner +  # D
            d_inner * d_model  # out_proj
        )
        return embed_params + n_layers * layer_params
    else:
        # Transformer parameter estimation
        embed_params = vocab_size * d_model
        d_k = d_model // n_heads

        # Attention params
        attn_params = (
            d_model * d_model +  # Q
            d_model * n_kv_heads * d_k +  # K
            d_model * n_kv_heads * d_k +  # V
            d_model * d_model  # O
        )

        # FFN params (SwiGLU has 3 projections)
        if arch_info.get("activation") in ["swiglu", "geglu", "reglu"]:
            ffn_params = 3 * d_model * d_ff
        else:
            ffn_params = 2 * d_model * d_ff

        # MoE multiplier
        if arch_info.get("use_moe"):
            num_experts = arch_info.get("num_experts", 8)
            ffn_params = ffn_params * num_experts + d_model * num_experts  # + router

        layer_params = attn_params + ffn_params + 2 * d_model  # + norms
        return embed_params + n_layers * layer_params


def generate_model_card(
    model_config,
    format_info: FormatInfo,
    training_config=None,
    model_name: Optional[str] = None,
    model_description: Optional[str] = None,
    license_type: str = "apache-2.0",
    tags: Optional[List[str]] = None,
    datasets_used: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    author: Optional[str] = None,
) -> str:
    """
    Generate a model card (README.md) for HuggingFace Hub.

    Args:
        model_config: Model configuration
        format_info: Export format information
        training_config: Optional training configuration
        model_name: Model name for the card
        model_description: Short description
        license_type: License identifier
        tags: Additional tags
        datasets_used: List of datasets used for training
        metrics: Evaluation metrics
        author: Model author

    Returns:
        String containing the README.md content
    """
    arch_info = get_architecture_info(model_config)
    param_count = _estimate_parameters(arch_info)

    # Build tags list
    all_tags = ["text-generation", "llm-lab"]
    if arch_info["model_architecture"] == "mamba2":
        all_tags.append("mamba")
        all_tags.append("state-space-model")
    else:
        all_tags.append("transformer")
        if arch_info.get("use_moe"):
            all_tags.append("moe")
            all_tags.append("mixture-of-experts")

    if format_info.vllm_compatible:
        all_tags.append("vllm")

    if tags:
        all_tags.extend(tags)

    # YAML frontmatter
    yaml_parts = [
        "---",
        f"license: {license_type}",
        "library_name: transformers",
        f"tags:",
    ]
    for tag in all_tags:
        yaml_parts.append(f"  - {tag}")

    if datasets_used:
        yaml_parts.append("datasets:")
        for ds in datasets_used:
            yaml_parts.append(f"  - {ds}")

    yaml_parts.append("pipeline_tag: text-generation")
    yaml_parts.append("---")

    yaml_section = "\n".join(yaml_parts)

    # Model name
    display_name = model_name or "LLM-Lab Model"

    # Architecture description
    if arch_info["model_architecture"] == "mamba2":
        arch_desc = "Mamba2 State Space Model"
    elif arch_info.get("use_moe"):
        arch_desc = f"Mixture of Experts Transformer ({arch_info['num_experts']} experts)"
    else:
        attn_type = arch_info.get("attention_type", "gqa").upper()
        arch_desc = f"Transformer with {attn_type} attention"

    # Build the README content
    content_parts = [
        yaml_section,
        "",
        f"# {display_name}",
        "",
    ]

    if model_description:
        content_parts.extend([model_description, ""])

    # Model summary table
    content_parts.extend([
        "## Model Details",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Architecture | {arch_desc} |",
        f"| Parameters | {_format_number(param_count)} |",
        f"| Hidden Size | {arch_info['d_model']} |",
        f"| Layers | {arch_info['n_layers']} |",
        f"| Context Length | {arch_info['max_seq_len']} |",
        f"| Vocabulary Size | {_format_number(arch_info['vocab_size'])} |",
    ])

    if arch_info["model_architecture"] == "transformer":
        content_parts.append(f"| Attention Heads | {arch_info['n_heads']} |")
        if arch_info.get("n_kv_heads") and arch_info["n_kv_heads"] != arch_info["n_heads"]:
            content_parts.append(f"| KV Heads | {arch_info['n_kv_heads']} |")
        content_parts.append(f"| Intermediate Size | {arch_info['d_ff']} |")
        content_parts.append(f"| Activation | {arch_info.get('activation', 'swiglu')} |")
        content_parts.append(f"| Position Encoding | {arch_info.get('positional_encoding', 'rope')} |")

    content_parts.append("")

    # vLLM compatibility notice
    if format_info.vllm_compatible:
        content_parts.extend([
            "## Inference",
            "",
            "This model is compatible with vLLM for fast inference:",
            "",
            "```bash",
            f"vllm serve {model_name or 'your-username/model-name'}",
            "```",
            "",
        ])
    else:
        content_parts.extend([
            "## Inference",
            "",
            "This model requires `trust_remote_code=True` due to custom architecture:",
            "",
        ])

    # Usage code
    content_parts.extend([
        "### Using Transformers",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f"model_name = \"{model_name or 'your-username/model-name'}\"",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(model_name)",
    ])

    if not format_info.vllm_compatible:
        content_parts.append("model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)")
    else:
        content_parts.append("model = AutoModelForCausalLM.from_pretrained(model_name)")

    content_parts.extend([
        "",
        "# Generate text",
        "inputs = tokenizer(\"Hello, how are you?\", return_tensors=\"pt\")",
        "outputs = model.generate(**inputs, max_new_tokens=100)",
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        "```",
        "",
    ])

    # Training info
    if training_config or datasets_used or metrics:
        content_parts.extend(["## Training", ""])

        if datasets_used:
            content_parts.append("### Datasets")
            content_parts.append("")
            for ds in datasets_used:
                content_parts.append(f"- {ds}")
            content_parts.append("")

        if training_config:
            if hasattr(training_config, '__dict__'):
                tc = training_config
                get_tc = lambda k, d=None: getattr(tc, k, d)
            else:
                get_tc = lambda k, d=None: training_config.get(k, d)

            content_parts.extend([
                "### Hyperparameters",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
            ])

            if get_tc('optimizer'):
                content_parts.append(f"| Optimizer | {get_tc('optimizer')} |")
            if get_tc('lr'):
                content_parts.append(f"| Learning Rate | {get_tc('lr')} |")
            if get_tc('max_steps'):
                content_parts.append(f"| Max Steps | {get_tc('max_steps')} |")
            if get_tc('batch_size'):
                content_parts.append(f"| Batch Size | {get_tc('batch_size')} |")
            if get_tc('gradient_accumulation_steps'):
                content_parts.append(f"| Gradient Accumulation | {get_tc('gradient_accumulation_steps')} |")

            content_parts.append("")

        if metrics:
            content_parts.extend([
                "### Evaluation Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            for name, value in metrics.items():
                if isinstance(value, float):
                    content_parts.append(f"| {name} | {value:.4f} |")
                else:
                    content_parts.append(f"| {name} | {value} |")
            content_parts.append("")

    # Footer
    content_parts.extend([
        "## About",
        "",
        f"This model was trained using [LLM-Lab](https://github.com/your-repo/llm-lab), "
        "an open-source tool for training language models.",
        "",
        f"Export format: `{format_info.format.value}`",
        "",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d')}",
    ])

    if author:
        content_parts.append(f"\nAuthor: {author}")

    return "\n".join(content_parts)


def save_model_card(content: str, output_path: str):
    """Save model card to file"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Saved model card to {output_path}")
