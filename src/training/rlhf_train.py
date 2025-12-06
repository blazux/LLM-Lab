"""
RLHF Training Dispatcher

This module routes RLHF training to the appropriate algorithm:
- PPO (Proximal Policy Optimization) - see ppo_train.py
- DPO (Direct Preference Optimization) - see dpo_train.py
- GRPO (Group Relative Policy Optimization) - see grpo_train.py
"""

import os
from config import RLHFConfig
from .ppo_train import train_ppo
from .dpo_train import train_dpo
from .grpo_train import train_grpo


def train_rlhf(config: RLHFConfig, callback=None):
    """Main RLHF training dispatcher - routes to PPO, DPO, or GRPO

    Args:
        config: RLHF configuration
        callback: Optional callback object with methods on_step, on_eval, on_log
    """
    # Create output directory and save config
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(f"{config.output_dir}/rlhf_config.json")

    if config.algorithm == "dpo":
        return train_dpo(config, callback=callback)
    elif config.algorithm == "ppo":
        return train_ppo(config, callback=callback)
    elif config.algorithm == "grpo":
        return train_grpo(config, callback=callback)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}. Must be 'ppo', 'dpo', or 'grpo'.")
