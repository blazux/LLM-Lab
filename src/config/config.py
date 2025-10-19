import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    # Architecture choices
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    positional_encoding: str = "rope"
    attention_type: str = "gqa"
    norm_type: str = "rmsnorm"
    activation: str = "swiglu"

    # Model parameters
    d_model: int = 896
    n_heads: int = 14
    n_kv_heads: int = 2
    d_ff: int = 4864
    n_layers: int = 24
    vocab_size: int = 151936
    max_seq_len: int = 1024
    dropout: float = 0.0

    # Additional parameters
    sliding_window: Optional[int] = None
    attention_bias: bool = False
    norm_eps: float = 1e-6

    # PEFT compatibility attributes
    tie_word_embeddings: bool = True  # We do tie embeddings in model.py
    is_encoder_decoder: bool = False  # Decoder-only model
    model_type: str = "custom_transformer"  # Custom model type

    def __post_init__(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.attention_type == "gqa":
            assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_k = self.d_model // self.n_heads
        if self.attention_type == "gqa":
            self.n_kv_groups = self.n_heads // self.n_kv_heads

    def get(self, key: str, default=None):
        """Dict-like get method for PEFT compatibility"""
        return getattr(self, key, default)

    def __contains__(self, key: str):
        """Dict-like 'in' operator for PEFT compatibility"""
        return hasattr(self, key)

    def count_params(self) -> int:
        """Estimate total number of parameters"""
        # Embeddings
        embed_params = self.vocab_size * self.d_model

        # Per-layer parameters
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
            self.datasets = [{"name": "HuggingFaceFW/fineweb-edu", "split": "train"}]

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))
