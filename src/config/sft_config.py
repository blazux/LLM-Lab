import json
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration"""

    # Policy Model
    policy_checkpoint: str = "checkpoints/best_model.pt"

    # Dataset Configuration
    datasets: List[dict] = None
    validation_splits: Optional[List[str]] = None  # Optional: specify validation split names to try (e.g., ["validation", "val"])

    # Training Hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    max_steps: int = 5000
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimizer
    optimizer: str = "adamw"

    # Optimizer-specific parameters
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 100

    # Logging & Checkpointing
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 500
    eval_steps: int = 50
    save_best_only: bool = True

    # Output
    output_dir: str = "sft_checkpoints"

    # LoRA Configuration (Parameter-Efficient Fine-Tuning)
    use_lora: bool = False
    lora_preset: str = "minimal"  # Options: "minimal", "attention_only", "ffn_only", "all", "custom"
    lora_target_modules: Optional[List[str]] = None  # Used only if lora_preset="custom"
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05  # LoRA dropout rate

    # Sequence length (will be inferred from model if not set)
    max_seq_len: Optional[int] = None

    def __post_init__(self):
        """Validate and set defaults"""
        # Ensure batch_size is compatible with gradient_accumulation_steps
        if self.datasets is None:
            # Default to SmolTalk2 SFT config with everyday conversations
            # User should customize this based on their needs
            self.datasets = [{
                "name": "HuggingFaceTB/smoltalk2",
                "subset": "SFT",
                "split": "smoltalk_smollm3_everyday_conversations_no_think"
            }]

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))
