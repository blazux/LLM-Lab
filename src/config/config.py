import json
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration (supports both Transformer and Mamba2)"""

    # Architecture selection
    model_architecture: str = "transformer"  # "transformer" or "mamba2"

    # Common parameters
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    d_model: int = 896
    n_layers: int = 24
    vocab_size: int = 151936
    max_seq_len: int = 1024
    dropout: float = 0.0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6

    # Transformer-specific parameters (optional for Mamba2)
    positional_encoding: Optional[str] = "rope"
    attention_type: Optional[str] = "gqa"
    activation: Optional[str] = "swiglu"
    n_heads: Optional[int] = 14
    n_kv_heads: Optional[int] = 2
    d_ff: Optional[int] = 4864
    sliding_window: Optional[int] = None
    attention_bias: bool = False

    # MLA-specific parameters (Multi-Head Latent Attention)
    d_latent: Optional[int] = None  # Latent dimension for KV compression (e.g., d_model // 4)
    d_rope_latent: Optional[int] = None  # Separate latent dim for RoPE (optional)

    # MoE parameters (Mixture of Experts - optional for Transformer)
    use_moe: bool = False  # Enable MoE in transformer feed-forward layers
    num_experts: Optional[int] = 8  # Number of expert FFNs per layer
    num_experts_per_token: Optional[int] = 2  # Top-K routing (how many experts process each token)
    load_balancing_loss_weight: float = 0.01  # Weight for load balancing auxiliary loss
    router_z_loss_weight: float = 0.001  # Weight for router z-loss (prevents overconfident routing)
    moe_layers: Optional[List[int]] = None  # Which layers use MoE (None = all layers, or list like [0, 2, 4, 6])

    # Mamba2-specific parameters (optional for Transformer)
    state_size: int = 64  # SSM state dimension (d_state) - 16=minimal, 64=balanced, 128=optimal
    expand_factor: int = 2  # Expansion ratio for Mamba2
    dt_rank: Optional[int] = None  # Rank for Δ (time step) - auto if None
    conv_kernel_size: int = 4  # Convolution kernel size
    use_bias: bool = True  # Whether to use bias in projections
    headdim: int = 64  # Head dimension for Mamba2 (64 or 128)
    ngroups: int = 1  # Number of groups for Mamba2 (1=no grouping, 8=efficient)
    chunk_size: int = 256  # Chunk size for Mamba2 processing

    # PEFT compatibility attributes
    tie_word_embeddings: bool = True  # We do tie embeddings in model.py
    is_encoder_decoder: bool = False  # Decoder-only model
    model_type: str = "custom_transformer"  # Custom model type (PEFT compat)

    def __post_init__(self):
        """Validate configuration based on architecture type"""
        if self.model_architecture == "transformer":
            # Transformer-specific validation
            assert self.n_heads is not None, "n_heads required for transformer"
            assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
            if self.attention_type == "gqa":
                assert self.n_kv_heads is not None, "n_kv_heads required for GQA"
                assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

            self.d_k = self.d_model // self.n_heads
            if self.attention_type == "gqa":
                self.n_kv_groups = self.n_heads // self.n_kv_heads

            # MLA-specific validation and defaults
            if self.attention_type == "mla":
                # Set default latent dimension to d_model // 4 if not specified
                if self.d_latent is None:
                    self.d_latent = max(self.d_model // 4, self.d_k)
                # Set default RoPE latent dimension (typically same as d_k)
                if self.d_rope_latent is None:
                    self.d_rope_latent = self.d_k

            # MoE-specific validation
            if self.use_moe:
                assert self.num_experts is not None and self.num_experts > 1, "num_experts must be > 1 for MoE"
                assert self.num_experts_per_token is not None and self.num_experts_per_token > 0, "num_experts_per_token must be > 0"
                assert self.num_experts_per_token <= self.num_experts, "num_experts_per_token must be <= num_experts"
                assert self.d_ff is not None, "d_ff required for MoE"
                # Validate moe_layers if specified
                if self.moe_layers is not None:
                    assert all(0 <= layer < self.n_layers for layer in self.moe_layers), \
                        f"moe_layers must be in range [0, {self.n_layers-1}]"

        elif self.model_architecture == "mamba2":
            # Mamba2-specific validation
            assert self.state_size > 0, "state_size must be positive"
            assert self.expand_factor > 0, "expand_factor must be positive"

            # Auto-compute dt_rank if not specified (following Mamba2 paper)
            if self.dt_rank is None:
                self.dt_rank = (self.d_model + 15) // 16

            # Set d_k for compatibility (not used in Mamba2 but may be checked)
            self.d_k = self.d_model

        else:
            raise ValueError(f"Unknown model_architecture: {self.model_architecture}")

    def get(self, key: str, default=None):
        """Dict-like get method for PEFT compatibility"""
        return getattr(self, key, default)

    def __contains__(self, key: str):
        """Dict-like 'in' operator for PEFT compatibility"""
        return hasattr(self, key)

    def count_params(self) -> int:
        """Estimate total number of parameters"""
        # Embeddings (common to both architectures)
        embed_params = self.vocab_size * self.d_model

        if self.model_architecture == "transformer":
            # Per-layer parameters for Transformer
            # Attention: Q, K, V projections + output projection
            if self.attention_type == "mha":
                attn_params = 4 * self.d_model * self.d_model
            elif self.attention_type == "mqa":
                attn_params = self.d_model * self.d_model + 2 * self.d_model * self.d_k + self.d_model * self.d_model
            elif self.attention_type == "mla":
                # MLA: Q projection + KV down-projection + KV up-projections + output projection
                q_params = self.d_model * self.d_model
                kv_down_params = self.d_model * self.d_latent
                k_up_params = self.d_latent * self.d_model
                v_up_params = self.d_latent * self.d_model
                out_params = self.d_model * self.d_model
                attn_params = q_params + kv_down_params + k_up_params + v_up_params + out_params
            else:  # gqa
                attn_params = self.d_model * self.d_model + 2 * self.n_kv_heads * self.d_k * self.d_model + self.d_model * self.d_model

            # Feed-forward (with MoE support)
            if self.activation in ["swiglu", "geglu", "reglu"]:
                single_ffn_params = 3 * self.d_model * self.d_ff
            else:
                single_ffn_params = 2 * self.d_model * self.d_ff

            # MoE: multiply by num_experts and add router
            if self.use_moe:
                router_params = self.d_model * self.num_experts
                ff_params = single_ffn_params * self.num_experts + router_params
            else:
                ff_params = single_ffn_params

            # Normalization (2 per layer)
            norm_params = 4 * self.d_model

            layer_params = attn_params + ff_params + norm_params

        elif self.model_architecture == "mamba2":
            # Per-layer parameters for Mamba2
            d_inner = self.d_model * self.expand_factor

            # Auto-compute dt_rank if not set (same as __post_init__)
            dt_rank = self.dt_rank if self.dt_rank is not None else (self.d_model + 15) // 16

            # Input projection (d_model -> 2 * d_inner for x and z)
            input_proj_params = self.d_model * (2 * d_inner)

            # SSM parameters (A, B, C, D, dt)
            # A: (d_inner, state_size)
            # B: (d_inner, state_size)
            # C: (d_inner, state_size)
            # D: (d_inner,)
            # dt: (d_inner, dt_rank) + (dt_rank,)
            ssm_params = (
                d_inner * self.state_size +  # A
                d_inner * self.state_size +  # B
                d_inner * self.state_size +  # C
                d_inner +  # D
                d_inner * dt_rank + dt_rank  # dt projection
            )

            # Convolution kernel
            conv_params = d_inner * self.conv_kernel_size

            # Output projection (d_inner -> d_model)
            output_proj_params = d_inner * self.d_model

            # Normalization (1 per layer for pre-norm)
            norm_params = 2 * self.d_model

            layer_params = input_proj_params + ssm_params + conv_params + output_proj_params + norm_params

        else:
            raise ValueError(f"Unknown model_architecture: {self.model_architecture}")

        total_params = embed_params + self.n_layers * layer_params + self.d_model  # +d_model for final norm

        return total_params

    def to_hf_config(self) -> dict:
        """Convert to HuggingFace-compatible config format"""

        if self.model_architecture == "transformer":
            # Base HuggingFace config structure
            hf_config = {
                # Model identification
                "model_type": "llm-lab-transformer",
                "architectures": ["TransformerLLM"],

                # Core architecture (using HF standard names)
                "hidden_size": self.d_model,
                "num_hidden_layers": self.n_layers,
                "num_attention_heads": self.n_heads,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": self.max_seq_len,

                # Feed-forward
                "intermediate_size": self.d_ff,
                "hidden_act": self.activation,

                # Regularization
                "dropout": self.dropout,

                # Normalization
                "layer_norm_eps": self.norm_eps if self.norm_type == "layernorm" else None,
                "rms_norm_eps": self.norm_eps if self.norm_type == "rmsnorm" else None,

                # Attention configuration
                "attention_bias": self.attention_bias,
                "attention_type": self.attention_type,

                # Positional encoding
                "position_embedding_type": self.positional_encoding,

                # Model settings
                "tie_word_embeddings": self.tie_word_embeddings,
                "is_encoder_decoder": False,
                "use_cache": True,

                # Tokenizer
                "tokenizer_name": self.tokenizer_name,
            }

            # Add GQA-specific config
            if self.attention_type == "gqa" and self.n_kv_heads:
                hf_config["num_key_value_heads"] = self.n_kv_heads

            # Add MLA-specific config
            if self.attention_type == "mla":
                hf_config["d_latent"] = self.d_latent
                hf_config["d_rope_latent"] = self.d_rope_latent

            # Add MoE-specific config
            if self.use_moe:
                hf_config["use_moe"] = self.use_moe
                hf_config["num_experts"] = self.num_experts
                hf_config["num_experts_per_token"] = self.num_experts_per_token
                hf_config["load_balancing_loss_weight"] = self.load_balancing_loss_weight
                hf_config["router_z_loss_weight"] = self.router_z_loss_weight
                if self.moe_layers:
                    hf_config["moe_layers"] = self.moe_layers

            # Add sliding window if used
            if self.sliding_window:
                hf_config["sliding_window"] = self.sliding_window

            # Remove None values for cleaner config
            hf_config = {k: v for k, v in hf_config.items() if v is not None}

        elif self.model_architecture == "mamba2":
            # Mamba2 config (less standardized in HF)
            hf_config = {
                "model_type": "mamba2",
                "architectures": ["Mamba2LLM"],

                # Core
                "hidden_size": self.d_model,
                "num_hidden_layers": self.n_layers,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": self.max_seq_len,

                # Mamba2-specific
                "state_size": self.state_size,
                "expand_factor": self.expand_factor,
                "conv_kernel_size": self.conv_kernel_size,
                "headdim": self.headdim,
                "ngroups": self.ngroups,
                "chunk_size": self.chunk_size,

                # Normalization
                "layer_norm_eps": self.norm_eps if self.norm_type == "layernorm" else None,
                "rms_norm_eps": self.norm_eps if self.norm_type == "rmsnorm" else None,

                # Model settings
                "tie_word_embeddings": self.tie_word_embeddings,
                "is_encoder_decoder": False,

                # Tokenizer
                "tokenizer_name": self.tokenizer_name,
            }

            hf_config = {k: v for k, v in hf_config.items() if v is not None}

        return hf_config

    def save_hf_config(self, path: str):
        """Save config in HuggingFace format (as config.json)"""
        hf_config = self.to_hf_config()
        with open(path, 'w') as f:
            json.dump(hf_config, f, indent=2)

    def save(self, path: str, also_save_hf: bool = True):
        """Save config to JSON file

        Args:
            path: Path to save internal config (e.g., model_config.json)
            also_save_hf: If True, also saves HuggingFace format as config.json
        """
        import os

        # Save internal format
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        # Also save HuggingFace format
        if also_save_hf:
            # Save in same directory as config.json
            dir_path = os.path.dirname(path) or "."
            hf_path = os.path.join(dir_path, "config.json")
            self.save_hf_config(hf_path)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model
    model_config_path: str = "/app/data/model_config.json"

    # Training steps
    max_steps: int = 10000

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1

    # Optimizer-specific parameters
    # AdamW
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    # Muon
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    # Lion
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    # Sophia
    sophia_beta1: float = 0.965
    sophia_beta2: float = 0.99
    sophia_rho: float = 0.04

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Batch and accumulation
    batch_size: int = 1
    gradient_accumulation_steps: int = 64
    grad_clip: float = 1.0

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100
    save_best_only: bool = True

    # Datasets
    datasets: list = None

    def __post_init__(self):
        if self.datasets is None:
            # Default pretraining datasets - industry standards
            #
            # Top choices for base pretraining:
            # 1. FineWeb-Edu (1.3T tokens) - High-quality educational content (used by Llama-3, SmolLM)
            # 2. FineWeb (15T tokens) - Full web corpus, quality-filtered
            # 3. RedPajama-v2 (30T tokens) - Massive multilingual dataset
            # 4. The Pile (825GB) - Academic standard, diverse sources
            # 5. Dolma (3T tokens) - Very high quality (AI2/OLMo)
            #
            # For multilingual: Add FineWeb-2 subsets (supports 95+ languages)

            self.datasets = [{"name": "HuggingFaceFW/fineweb-edu", "split": "train"}]

            # Example: Multilingual (English + French)
            # self.datasets = [
            #     {"name": "HuggingFaceFW/fineweb-edu", "weight": 2.0},
            #     {"name": "HuggingFaceFW/fineweb-2", "subset": "fra_Latn", "weight": 1.0}
            # ]

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))
