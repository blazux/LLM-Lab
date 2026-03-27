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

    def forward(self, x, past_key_values=None, start_pos=0):
        # Pre-norm architecture
        attn_out, new_kv = self.attention(self.norm1(x), past_key_values=past_key_values, start_pos=start_pos)
        x = x + self.dropout(attn_out)

        # Feed-forward (handle MoE aux loss)
        if self.is_moe:
            ff_out, aux_loss = self.feed_forward(self.norm2(x))
            x = x + self.dropout(ff_out)
            return x, aux_loss, new_kv
        else:
            ff_out = self.feed_forward(self.norm2(x))
            x = x + self.dropout(ff_out)
            return x, None, new_kv


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

        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
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

    def forward(self, x=None, input_ids=None, use_checkpoint: bool = False,
                past_key_values=None, use_cache: bool = False,
                return_hidden: bool = False, **kwargs):
        """
        Forward pass
        Args:
            x: input token ids (batch, seq_len) - for backward compatibility
            input_ids: input token ids (batch, seq_len) - for PEFT compatibility
            use_checkpoint: whether to use gradient checkpointing
            past_key_values: list of (k, v) tuples per layer for KV cache
            use_cache: whether to return updated KV cache

        Returns:
            logits: (batch, seq_len, vocab_size)
            aux_loss: MoE auxiliary loss (or None if no MoE layers)
            past_key_values (optional): updated KV cache (only if use_cache=True)
        """
        # Handle both calling conventions (x for legacy, input_ids for PEFT)
        if input_ids is not None:
            x = input_ids
        elif x is None:
            raise ValueError("Either 'x' or 'input_ids' must be provided")

        # Embedding with scaling
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)

        # Compute start_pos for positional encoding (needed during KV cache decode)
        pos_start = 0
        if past_key_values is not None and len(past_key_values) > 0:
            pos_start = past_key_values[0][0].size(2)

        # Positional encoding (only for sinusoidal/learned — RoPE/YARN/ALiBi applied in attention)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x, pos_start)
        x = self.position_dropout(x)

        # Transformer blocks (accumulate MoE aux loss, thread KV cache)
        total_aux_loss = None
        moe_layer_count = 0
        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.transformer_blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            start_pos = layer_past[0].size(2) if layer_past is not None else 0

            if use_checkpoint and self.training:
                # Gradient checkpointing is training-only; never combined with use_cache
                x, aux_loss, _ = checkpoint(block, x, use_reentrant=False)
            else:
                x, aux_loss, new_kv = block(x, past_key_values=layer_past, start_pos=start_pos)

            if use_cache:
                new_past_key_values.append(new_kv)

            # Accumulate aux loss from MoE layers
            if aux_loss is not None:
                moe_layer_count += 1
                if total_aux_loss is None:
                    total_aux_loss = aux_loss
                else:
                    total_aux_loss = total_aux_loss + aux_loss

        # Normalize aux loss by number of MoE layers to keep scale independent of depth
        if total_aux_loss is not None and moe_layer_count > 1:
            total_aux_loss = total_aux_loss / moe_layer_count

        # Final normalization and output
        x = self.norm(x)
        x = self.output_dropout(x)

        if return_hidden:
            # Return pre-lm_head hidden states (used by MAXIS loss)
            if use_cache:
                return x, total_aux_loss, new_past_key_values
            return x, total_aux_loss

        logits = self.lm_head(x)

        if use_cache:
            return logits, total_aux_loss, new_past_key_values
        return logits, total_aux_loss

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # PEFT compatibility methods
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}

    def _reorder_cache(self, past_key_values, beam_idx):
        return [(k.index_select(0, beam_idx), v.index_select(0, beam_idx))
                for k, v in past_key_values]

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
