"""
RLHF Training Dispatcher

This module routes RLHF training to the appropriate algorithm:
- PPO (Proximal Policy Optimization) - see ppo_train.py
- DPO (Direct Preference Optimization) - see dpo_train.py
"""

from rlhf_config import RLHFConfig
from ppo_train import train_ppo
from dpo_train import train_dpo


def train_rlhf(config: RLHFConfig):
    """Main RLHF training dispatcher - routes to PPO or DPO"""
    if config.algorithm == "dpo":
        return train_dpo(config)
    elif config.algorithm == "ppo":
        return train_ppo(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}. Must be 'ppo' or 'dpo'.")
