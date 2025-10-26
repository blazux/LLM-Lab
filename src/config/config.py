import json
from dataclasses import dataclass, asdict
from typing import Optional


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

    # Mamba2-specific parameters (optional for Transformer)
    state_size: int = 16  # SSM state dimension (d_state)
    expand_factor: int = 2  # Expansion ratio for Mamba2
    dt_rank: Optional[int] = None  # Rank for Δ (time step) - auto if None
    conv_kernel_size: int = 4  # Convolution kernel size
    use_bias: bool = True  # Whether to use bias in projections

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
            else:  # gqa
                attn_params = self.d_model * self.d_model + 2 * self.n_kv_heads * self.d_k * self.d_model + self.d_model * self.d_model

            # Feed-forward
            if self.activation == "swiglu":
                ff_params = 3 * self.d_model * self.d_ff
            else:
                ff_params = 2 * self.d_model * self.d_ff

            # Normalization (2 per layer)
            norm_params = 4 * self.d_model

            layer_params = attn_params + ff_params + norm_params

        elif self.model_architecture == "mamba2":
            # Per-layer parameters for Mamba2
            d_inner = self.d_model * self.expand_factor

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
                d_inner * self.dt_rank + self.dt_rank  # dt projection
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

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model
    model_config_path: str = "model_config.json"

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
