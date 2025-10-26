"""Mamba2 State Space Model implementation"""

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

from config import ModelConfig
from .bricks import Mamba2Block, NORM_TYPES


class Mamba2LLM(nn.Module):
    """
    Mamba2-based Language Model

    Architecture:
    - Token embedding with scaling
    - Stack of Mamba2 blocks (no positional encoding needed)
    - Final normalization
    - Language modeling head (tied with embedding)

    Key differences from Transformer:
    - No attention mechanism (replaced with SSM)
    - No explicit positional encoding (learned through convolution)
    - Linear complexity O(N) instead of O(N²)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        assert config.model_architecture == "mamba2", "This model requires model_architecture='mamba2'"

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Mamba2 doesn't use explicit positional encodings
        # (position info is learned through the conv1d and SSM state)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Stack of Mamba2 blocks
        self.mamba_blocks = nn.ModuleList([
            Mamba2Block(config) for _ in range(config.n_layers)
        ])

        # Final normalization
        NormClass = NORM_TYPES[config.norm_type]
        self.norm = NormClass(config.d_model, eps=config.norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)

        # Output projection (weight tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x=None, input_ids=None, use_checkpoint: bool = False, **kwargs):
        """
        Forward pass

        Args:
            x: input token ids (batch, seq_len) - for backward compatibility
            input_ids: input token ids (batch, seq_len) - for PEFT compatibility
            use_checkpoint: whether to use gradient checkpointing

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Handle both calling conventions (x for legacy, input_ids for PEFT)
        if input_ids is not None:
            x = input_ids
        elif x is None:
            raise ValueError("Either 'x' or 'input_ids' must be provided")

        # Embedding with scaling (same as transformer)
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.embedding_dropout(x)

        # Mamba2 blocks (no positional encoding needed)
        for block in self.mamba_blocks:
            if use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Final normalization and output
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ========================================================================
    # PEFT COMPATIBILITY METHODS
    # ========================================================================

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation (PEFT compatibility)"""
        return {"input_ids": input_ids}

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search (PEFT compatibility)"""
        # Mamba2 doesn't use KV cache, but PEFT may check for this method
        return past_key_values

    def get_input_embeddings(self):
        """Get input embedding layer (PEFT compatibility)"""
        return self.token_embedding

    def set_input_embeddings(self, new_embeddings):
        """Set input embedding layer (PEFT compatibility)"""
        self.token_embedding = new_embeddings

    def get_output_embeddings(self):
        """Get output embedding layer (PEFT compatibility)"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embedding layer (PEFT compatibility)"""
        self.lm_head = new_embeddings
