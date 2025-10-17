import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

from config import ModelConfig
from .bricks import (
    POSITIONAL_ENCODINGS,
    ATTENTION_TYPES,
    NORM_TYPES,
    ACTIVATION_TYPES,
    SwiGLU,
    StandardFFN
)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Get norm class
        NormClass = NORM_TYPES[config.norm_type]

        # Attention
        AttentionClass = ATTENTION_TYPES[config.attention_type]
        self.attention = AttentionClass(config)

        # Feed-forward
        if config.activation == 'swiglu':
            self.feed_forward = SwiGLU(config.d_model, config.d_ff, config.dropout)
        else:
            self.feed_forward = StandardFFN(config.d_model, config.d_ff, config.activation, config.dropout)

        # Normalization
        self.norm1 = NormClass(config.d_model, eps=config.norm_eps)
        self.norm2 = NormClass(config.d_model, eps=config.norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Pre-norm architecture
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class TransformerLLM(nn.Module):
    """Complete transformer language model assembled from bricks"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding
        if config.positional_encoding in ['rope', 'yarn']:
            # RoPE and YARN are applied in attention, not here
            self.pos_encoding = None
            self.position_dropout = nn.Dropout(config.dropout)
        elif config.positional_encoding == 'alibi':
            # ALiBi is also applied in attention
            self.pos_encoding = None
            self.position_dropout = nn.Dropout(config.dropout)
        elif config.positional_encoding == 'sinusoidal':
            PosClass = POSITIONAL_ENCODINGS['sinusoidal']
            self.pos_encoding = PosClass(config.d_model, config.max_seq_len)
            self.position_dropout = nn.Dropout(config.dropout)
        else:
            self.pos_encoding = None
            self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Set positional encoding for attention layers
        if config.positional_encoding in ['rope', 'yarn', 'alibi']:
            self._setup_attention_pos_encoding()

        # Final normalization
        NormClass = NORM_TYPES[config.norm_type]
        self.norm = NormClass(config.d_model, eps=config.norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)

        # Output projection (weight tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _setup_attention_pos_encoding(self):
        """Setup positional encoding for attention layers"""
        if self.config.positional_encoding == 'rope':
            PosClass = POSITIONAL_ENCODINGS['rope']
            pos_enc = PosClass(self.config.d_k, self.config.max_seq_len)
        elif self.config.positional_encoding == 'yarn':
            PosClass = POSITIONAL_ENCODINGS['yarn']
            pos_enc = PosClass(self.config.d_k, self.config.max_seq_len)
        elif self.config.positional_encoding == 'alibi':
            PosClass = POSITIONAL_ENCODINGS['alibi']
            pos_enc = PosClass(self.config.n_heads, self.config.max_seq_len)
        else:
            return

        # Assign to all attention layers
        for block in self.transformer_blocks:
            block.attention.pos_encoding = pos_enc

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
        """
        # Handle both calling conventions (x for legacy, input_ids for PEFT)
        if input_ids is not None:
            x = input_ids
        elif x is None:
            raise ValueError("Either 'x' or 'input_ids' must be provided")

        # Embedding with scaling
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)

        # Positional encoding (only for sinusoidal)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self.position_dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
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

    # PEFT compatibility methods
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation (PEFT compatibility)"""
        return {"input_ids": input_ids}

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search (PEFT compatibility)"""
        # Not used in training, but PEFT may check for it
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
