import json
from dataclasses import dataclass, asdict, fields
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    # Architecture selection
    model_architecture: str = "transformer"
    assembly_template: Optional[str] = None  # Starting architecture, not a claim of checkpoint compatibility
    assembly: Optional[dict] = None  # Executable graph; legacy fields remain for older checkpoints

    # Common parameters
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    d_model: int = 896
    n_layers: int = 24
    vocab_size: int = 151936
    max_seq_len: int = 1024
    dropout: float = 0.0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6

    # Attention / feed-forward parameters
    positional_encoding: Optional[str] = "rope"
    attention_type: Optional[str] = "gqa"
    activation: Optional[str] = "swiglu"
    n_heads: Optional[int] = 14
    n_kv_heads: Optional[int] = 2
    d_ff: Optional[int] = 4864
    sliding_window: Optional[int] = None
    attention_bias: bool = False

    # Modern attention options
    use_attention_gate: bool = False  # Qwen3-Next-style sigmoid output gate on attention
    nope_every: Optional[int] = None  # Skip positional encoding every Nth layer (Llama 4 iRoPE: 4)
    rope_fraction: float = 0.5  # Fraction of RoPE pairs that rotate (0.5 = historical default)
    use_sparse_attention: bool = False  # DSA-style top-k sparse attention (GQA only)
    sparse_top_k: int = 512  # Tokens each query may attend to
    sparse_indexer_heads: int = 4  # Lightning indexer heads
    sparse_indexer_dim: int = 32  # Lightning indexer head dim
    hybrid_pattern: Optional[str] = None  # "L:A" linear:attention interleaving, e.g. "3:1" (Qwen3-Next).
                                          # L Gated-DeltaNet layers then A attention layers, repeating.

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
    moe_router_type: str = "softmax"  # "softmax" (classic, aux losses) or "sigmoid" (DeepSeek-V3)
    moe_aux_free_bias: bool = False  # DeepSeek-V3 aux-loss-free balancing via per-expert routing bias
    moe_bias_update_rate: float = 0.001  # Bias adjustment speed (gamma) for aux-free balancing
    moe_shared_expert: bool = False  # One always-active shared expert (DeepSeek-MoE style)

    # PEFT compatibility attributes
    tie_word_embeddings: bool = True  # We do tie embeddings in model.py
    is_encoder_decoder: bool = False  # Decoder-only model
    model_type: str = "custom_transformer"  # Custom model type (PEFT compat)

    def __post_init__(self):
        """Validate configuration"""
        if self.assembly is not None:
            from model.assembly_runtime import validate_assembly
            if (type(self.d_model) is not int or not 1 <= self.d_model <= 8192
                    or type(self.max_seq_len) is not int or not 2 <= self.max_seq_len <= 32768
                    or type(self.vocab_size) is not int or not 2 <= self.vocab_size <= 1_000_000):
                raise ValueError('Invalid assembly width or context length')
            if self.dropout != 0:
                raise ValueError('Visual assemblies have no global dropout: set dropout=0')
            validate_assembly(self.assembly, self.d_model, self.vocab_size)
            self.model_type = 'llm-lab-assembly'
            return
        if self.model_architecture != "transformer":
            raise ValueError(f"Unknown model_architecture: {self.model_architecture}")

        assert self.n_heads is not None, "n_heads required for transformer"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.attention_type == "gqa":
            assert self.n_kv_heads is not None, "n_kv_heads required for GQA"
            assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        # Auto d_ff: ~8/3 x d_model (SwiGLU convention), rounded to a multiple of 64
        if self.d_ff is None:
            self.d_ff = max(64, round(self.d_model * 8 / 3 / 64) * 64)

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

        if self.use_sparse_attention:
            assert self.attention_type == "gqa", "sparse attention currently requires attention_type='gqa'"

        # Hybrid (Gated DeltaNet) validation
        if self.hybrid_pattern:
            parts = self.hybrid_pattern.split(':')
            assert len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit(), \
                f"hybrid_pattern must look like '3:1', got {self.hybrid_pattern!r}"
            self._hybrid_linear, self._hybrid_attn = int(parts[0]), int(parts[1])
            assert self._hybrid_linear > 0 and self._hybrid_attn > 0, \
                "hybrid_pattern needs at least one layer of each kind"
            assert self.positional_encoding in ("rope", "yarn", "alibi"), \
                "hybrid_pattern requires an attention-level positional encoding (rope/yarn/alibi)"

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

    def is_linear_layer(self, layer_idx: int) -> bool:
        """True if this layer uses Gated DeltaNet instead of attention"""
        if not self.hybrid_pattern:
            return False
        cycle = self._hybrid_linear + self._hybrid_attn
        return (layer_idx % cycle) < self._hybrid_linear

    def get(self, key: str, default=None):
        """Dict-like get method for PEFT compatibility"""
        return getattr(self, key, default)

    def __contains__(self, key: str):
        """Dict-like 'in' operator for PEFT compatibility"""
        return hasattr(self, key)

    def count_params(self) -> int:
        """Estimate total number of parameters"""
        if self.assembly is not None:
            import torch
            from model.assembly_model import AssemblyLLM
            with torch.device('meta'):
                return AssemblyLLM(self).count_parameters()
        embed_params = self.vocab_size * self.d_model

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

        if self.use_attention_gate:
            attn_params += self.d_model * self.d_model
        if getattr(self, 'use_sparse_attention', False):
            ih, idim = self.sparse_indexer_heads, self.sparse_indexer_dim
            attn_params += self.d_model * ih * idim + self.d_model * idim + self.d_model * ih

        # Feed-forward (with MoE support)
        if self.activation in ["swiglu", "geglu", "reglu"]:
            single_ffn_params = 3 * self.d_model * self.d_ff
        else:
            single_ffn_params = 2 * self.d_model * self.d_ff

        # MoE: multiply by num_experts and add router (+ optional shared expert)
        if self.use_moe:
            router_params = self.d_model * self.num_experts
            ff_params = single_ffn_params * self.num_experts + router_params
            if getattr(self, 'moe_shared_expert', False):
                ff_params += single_ffn_params
        else:
            ff_params = single_ffn_params

        # Normalization (2 per layer)
        norm_params = 4 * self.d_model

        # Gated DeltaNet layers (hybrid): q,k,v,gate,out projections + per-head params
        inner = self.d_model
        gdn_params = 5 * self.d_model * inner + 2 * self.d_model * self.n_heads \
            + 2 * self.n_heads + self.d_k

        attn_layer = attn_params + ff_params + norm_params
        gdn_layer = gdn_params + ff_params + norm_params

        if self.hybrid_pattern:
            n_linear = sum(1 for i in range(self.n_layers) if self.is_linear_layer(i))
            total_layers = n_linear * gdn_layer + (self.n_layers - n_linear) * attn_layer
        else:
            total_layers = self.n_layers * attn_layer

        total_params = embed_params + total_layers + self.d_model  # +d_model for final norm

        return total_params

    def to_hf_config(self) -> dict:
        """Convert the actual architecture to Hugging Face configuration."""
        if self.assembly is not None:
            return {'model_type': 'llm-lab-assembly', 'assembly': self.assembly, 'assembly_template': self.assembly_template,
                    'd_model': self.d_model, 'vocab_size': self.vocab_size,
                    'max_seq_len': self.max_seq_len, 'tie_word_embeddings': self.tie_word_embeddings}

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
        return {k: v for k, v in hf_config.items() if v is not None}

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
        """Load config from JSON file, ignoring unknown keys (e.g. from older versions)"""
        with open(path, 'r') as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model
    model_config_path: str = "/app/data/model_config.json"

    # Pretraining optimizer updates (one update includes all accumulated batches)
    max_steps: int = 10000

    # Optimizer
    optimizer: str = "adamw"
    use_8bit_optimizer: bool = False  # Explicit choice; installing bitsandbytes must not change a recipe
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
    scheduler: str = "cosine"  # "cosine", "wsd", "linear", "polynomial", "constant", or "adaptive"
    warmup_steps: int = 1000
    wsd_decay_fraction: float = 0.15  # WSD only: fraction of steps spent in final linear decay

    # Adaptive scheduler parameters (only used when scheduler="adaptive")
    adaptive_window: int = 10  # Number of eval steps to consider for trend
    adaptive_increase_factor: float = 1.05  # Multiply LR by this when loss decreasing well
    adaptive_decrease_factor: float = 0.9  # Multiply LR by this when loss plateauing/increasing
    adaptive_patience: int = 3  # Wait this many evals before adjusting LR again
    adaptive_min_lr: float = 1e-6  # Minimum learning rate
    adaptive_threshold: float = 0.01  # Min relative improvement to be considered "decreasing"

    # Batch and accumulation
    batch_size: int = 1
    gradient_accumulation_steps: int = 64
    grad_clip: float = 1.0

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100
    save_best_only: bool = True

    # Loss function
    loss_fn: str = "cross_entropy"  # "cross_entropy" or "maxis"
    z_loss_weight: float = 0.0  # LM-head z-loss (PaLM/OLMo2 style, 1e-4 typical); cross_entropy only

    # Performance
    use_compile: bool = False  # torch.compile the model (faster after a warmup compile)
    gradient_checkpointing: bool = True

    # Data pipeline
    use_doc_masks: bool = False  # Block attention across packed document boundaries (Llama 3 style)
    # MAXIS loss hyperparameters (only used when loss_fn="maxis")
    maxis_low_rank_dim: int = 64     # Dimension for scout pass and aux loss
    maxis_n_candidates: int = 2048   # Negative samples per chunk
    maxis_chunk_size: int = 32       # Tokens per inner loop iteration
    maxis_aux_weight: float = 0.2    # Weight of the Matryoshka auxiliary loss

    # Datasets
    datasets: list = None
    validation_datasets: Optional[list] = None  # Separate source; no sequence-level holdout when supplied

    def __post_init__(self):
        for name in ('max_steps', 'batch_size', 'gradient_accumulation_steps', 'eval_every', 'eval_steps'):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0 < self.wsd_decay_fraction <= 1:
            raise ValueError("wsd_decay_fraction must be in (0, 1]")
        if self.datasets is None:
            # The GUI offers beginner presets with explicit held-out documents.
            self.datasets = [{"name": "HuggingFaceFW/fineweb-edu", "split": "train"}]

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file, ignoring unknown keys (e.g. from older versions)"""
        with open(path, 'r') as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
