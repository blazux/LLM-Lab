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
    GeGLU,
    ReGLU,
    StandardFFN,
    MoEFFN
)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization"""
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Get norm class
        NormClass = NORM_TYPES[config.norm_type]

        # Attention
        AttentionClass = ATTENTION_TYPES[config.attention_type]
        self.attention = AttentionClass(config)

        # Feed-forward: Check if this layer should use MoE
        use_moe_this_layer = config.use_moe and (
            config.moe_layers is None or layer_idx in config.moe_layers
        )

        if use_moe_this_layer:
            # Use Mixture of Experts
            self.feed_forward = MoEFFN(config)
            self.is_moe = True
        else:
            # Use standard FFN
            if config.activation == 'swiglu':
                self.feed_forward = SwiGLU(config.d_model, config.d_ff, config.dropout)
            elif config.activation == 'geglu':
                self.feed_forward = GeGLU(config.d_model, config.d_ff, config.dropout)
            elif config.activation == 'reglu':
                self.feed_forward = ReGLU(config.d_model, config.d_ff, config.dropout)
            else:
                self.feed_forward = StandardFFN(config.d_model, config.d_ff, config.activation, config.dropout)
            self.is_moe = False

        # Normalization
        self.norm1 = NormClass(config.d_model, eps=config.norm_eps)
        self.norm2 = NormClass(config.d_model, eps=config.norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Pre-norm architecture
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward (handle MoE aux loss)
        if self.is_moe:
            ff_out, aux_loss = self.feed_forward(self.norm2(x))
            x = x + self.dropout(ff_out)
            return x, aux_loss
        else:
            ff_out = self.feed_forward(self.norm2(x))
            x = x + self.dropout(ff_out)
            return x, None


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
        elif config.positional_encoding in ['sinusoidal', 'learned']:
            PosClass = POSITIONAL_ENCODINGS[config.positional_encoding]
            self.pos_encoding = PosClass(config.d_model, config.max_seq_len)
            self.position_dropout = nn.Dropout(config.dropout)
        else:
            self.pos_encoding = None
            self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks (pass layer_idx for MoE layer selection)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
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

        Returns:
            logits: (batch, seq_len, vocab_size)
            aux_loss: MoE auxiliary loss (or None if no MoE layers)
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

        # Transformer blocks (accumulate MoE aux loss)
        total_aux_loss = None
        for block in self.transformer_blocks:
            if use_checkpoint and self.training:
                # Note: gradient checkpointing with MoE returns both x and aux_loss
                x, aux_loss = checkpoint(block, x, use_reentrant=False)
            else:
                x, aux_loss = block(x)

            # Accumulate aux loss from MoE layers
            if aux_loss is not None:
                if total_aux_loss is None:
                    total_aux_loss = aux_loss
                else:
                    total_aux_loss = total_aux_loss + aux_loss

        # Final normalization and output
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits, total_aux_loss

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
