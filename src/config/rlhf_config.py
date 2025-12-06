import json
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class RLHFConfig:
    """RLHF training configuration (PPO or DPO)"""

    # Algorithm Selection
    algorithm: str = "ppo"  # Options: "ppo", "dpo", or "grpo"

    # Policy Model
    policy_checkpoint: str = "outputs/sft/best_model.pt"

    # Reward Model (only used for PPO)
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"

    # Reference Model (only used for DPO)
    reference_checkpoint: Optional[str] = None  # If None, will use policy_checkpoint as reference

    # Dataset
    datasets: List[dict] = None

    # PPO Hyperparameters
    batch_size: int = 128
    mini_batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # PPO-specific
    ppo_epochs: int = 4
    learning_rate: float = 1.4e-5
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clip range
    vf_coef: float = 0.1  # Value function coefficient

    # GAE parameters
    gamma: float = 1.0  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda

    # Training steps
    max_steps: int = 10000

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    # GRPO-specific parameters
    group_size: int = 4  # Number of responses to generate per prompt (GRPO only)
    grpo_temperature: float = 1.0  # Generation temperature for GRPO groups

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

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

    # Logging & Checkpointing
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 500
    eval_steps: int = 50

    # Output
    output_dir: str = "outputs/rlhf"

    # LoRA Configuration (Parameter-Efficient Fine-Tuning)
    use_lora: bool = False
    lora_preset: str = "minimal"  # Options: "minimal", "attention_only", "ffn_only", "all", "custom"
    lora_target_modules: Optional[List[str]] = None  # Used only if lora_preset="custom"
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05  # LoRA dropout rate

    # Resume
    resume_from: Optional[str] = None

    def __post_init__(self):
        """Validate and set defaults"""
        # Validate algorithm
        assert self.algorithm in ["ppo", "dpo", "grpo"], \
            f"algorithm must be 'ppo', 'dpo', or 'grpo', got '{self.algorithm}'"

        if self.datasets is None:
            # Default RLHF datasets - industry standards for preference alignment
            #
            # For DPO (Direct Preference Optimization):
            # 1. Anthropic/hh-rlhf (169k pairs) - Gold standard for helpfulness/harmlessness
            # 2. Argilla/ultrafeedback-binarized-preferences-cleaned (64k) - Used by Zephyr
            # 3. Intel/orca_dpo_pairs (13k) - High-quality reasoning pairs
            #
            # For PPO/GRPO (Policy Optimization with Reward Model):
            # - Any prompt dataset works (model generates responses, reward model scores them)
            # - Anthropic/hh-rlhf works great for general helpfulness
            #
            # Reward Models (for PPO/GRPO):
            # 1. OpenAssistant/reward-model-deberta-v3-large-v2 (current default)
            # 2. weqweasdas/RM-Mistral-7B (newer, better performance)

            if self.algorithm == "dpo":
                # DPO needs preference pairs (chosen vs rejected)
                self.datasets = [{
                    "name": "Anthropic/hh-rlhf",
                    "subset": None,
                    "split": "train"
                }]
                # Alternative DPO datasets:
                # self.datasets = [{"name": "Argilla/ultrafeedback-binarized-preferences-cleaned", "split": "train"}]
            else:
                # PPO/GRPO use prompts + reward model
                self.datasets = [{
                    "name": "Anthropic/hh-rlhf",
                    "subset": None,
                    "split": "train"
                }]

        # Ensure mini_batch_size divides batch_size
        assert self.batch_size % self.mini_batch_size == 0, \
            "batch_size must be divisible by mini_batch_size"

        # Clip range for value function defaults to same as policy
        if self.clip_range_vf is None:
            self.clip_range_vf = self.clip_range

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))
