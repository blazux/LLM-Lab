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

    def forward(self, x, use_checkpoint: bool = False):
        """
        Forward pass
        Args:
            x: input token ids (batch, seq_len)
            use_checkpoint: whether to use gradient checkpointing
        """
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
