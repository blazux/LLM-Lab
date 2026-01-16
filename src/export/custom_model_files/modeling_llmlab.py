"""
LLM-Lab Model Implementation for HuggingFace Transformers

This file is bundled with custom model exports to enable loading via
AutoModelForCausalLM.from_pretrained() with trust_remote_code=True.

Contains self-contained implementations of:
- TransformerLLM with various attention types (GQA, MHA, MQA, MLA)
- Mamba2LLM state space model
- Various activations (SwiGLU, GeGLU, etc.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_llmlab import LLMLabConfig


# =============================================================================
# POSITIONAL ENCODINGS
# =============================================================================

class RoPE(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, d_k: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=d_k // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(d_k // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        """x: (batch, n_heads, seq_len, d_k)"""
        assert self.cos.size(0) >= x_BTHD.size(-2)
        cos = self.cos[None, None, :x_BTHD.size(-2), :]
        sin = self.sin[None, None, :x_BTHD.size(-2), :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_BTHD)


# =============================================================================
# ATTENTION MECHANISMS
# =============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for Grouped Query Attention"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention"""
    def __init__(self, config: LLMLabConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # QK normalization
        if config.norm_type == "rmsnorm":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps)

        self.pos_encoding = None

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.pos_encoding is not None:
            q = self.pos_encoding(q)
            k = self.pos_encoding(k)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.w_o(attn_output)


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) - DeepSeek style"""
    def __init__(self, config: LLMLabConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.d_latent = config.d_latent or (config.hidden_size // 4)
        self.d_rope_latent = config.d_rope_latent or self.head_dim
        self.dropout = config.attention_dropout
        self.positional_encoding = config.positional_encoding

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.kv_down = nn.Linear(self.hidden_size, self.d_latent, bias=False)

        if self.positional_encoding in ["rope", "yarn"]:
            self.k_rope_proj = nn.Linear(self.d_latent, self.num_heads * self.d_rope_latent, bias=False)
            self.k_nope_proj = nn.Linear(self.d_latent, self.num_heads * (self.head_dim - self.d_rope_latent), bias=False)
        else:
            self.k_proj = nn.Linear(self.d_latent, self.num_heads * self.head_dim, bias=False)

        self.v_proj = nn.Linear(self.d_latent, self.num_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if config.norm_type == "rmsnorm":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps)

        self.pos_encoding = None

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        latent = self.kv_down(x)
        v = self.v_proj(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)

        if self.positional_encoding in ["rope", "yarn"]:
            k_rope = self.k_rope_proj(latent).view(batch_size, seq_len, self.num_heads, self.d_rope_latent)
            k_nope = self.k_nope_proj(latent).view(batch_size, seq_len, self.num_heads, self.head_dim - self.d_rope_latent)

            q = self.q_norm(q)
            q = q.transpose(1, 2)
            k_rope = k_rope.transpose(1, 2)
            k_nope = k_nope.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.pos_encoding is not None:
                q_rope = q[..., :self.d_rope_latent]
                q_nope = q[..., self.d_rope_latent:]
                q_rope = self.pos_encoding(q_rope)
                k_rope = self.pos_encoding(k_rope)
                q = torch.cat([q_rope, q_nope], dim=-1)

            k = torch.cat([k_rope, k_nope], dim=-1)
            k = k.transpose(1, 2)
            k = self.k_norm(k)
            k = k.transpose(1, 2)
        else:
            k = self.k_proj(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.w_o(attn_output)


# =============================================================================
# FEED-FORWARD NETWORKS
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation"""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GeGLU(nn.Module):
    """GeGLU activation"""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.gelu(self.gate_proj(x)) * self.up_proj(x)))


class StandardFFN(nn.Module):
    """Standard FFN with configurable activation"""
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.silu if activation == "silu" else F.relu

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, config: LLMLabConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention
        if config.attention_type == "mla":
            self.attention = MultiHeadLatentAttention(config)
        else:
            self.attention = GroupedQueryAttention(config)

        # Feed-forward
        if config.hidden_act in ["swiglu", "silu"]:
            self.feed_forward = SwiGLU(config.hidden_size, config.intermediate_size, config.dropout)
        elif config.hidden_act == "geglu":
            self.feed_forward = GeGLU(config.hidden_size, config.intermediate_size, config.dropout)
        else:
            self.feed_forward = StandardFFN(config.hidden_size, config.intermediate_size, config.hidden_act, config.dropout)

        # Normalization
        if config.norm_type == "rmsnorm":
            self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
            self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


# =============================================================================
# MAIN MODEL
# =============================================================================

class LLMLabPreTrainedModel(PreTrainedModel):
    """Base class for LLM-Lab models"""
    config_class = LLMLabConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class LLMLabForCausalLM(LLMLabPreTrainedModel):
    """LLM-Lab Transformer for Causal Language Modeling"""

    def __init__(self, config: LLMLabConfig):
        super().__init__(config)
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional encoding
        if config.positional_encoding in ["rope", "yarn"]:
            self.pos_encoding = RoPE(config.head_dim, config.max_position_embeddings)
        else:
            self.pos_encoding = None

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])

        # Setup positional encoding for attention layers
        if self.pos_encoding is not None:
            for block in self.transformer_blocks:
                block.attention.pos_encoding = self.pos_encoding

        # Final normalization
        if config.norm_type == "rmsnorm":
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            x = self.token_embedding(input_ids) * math.sqrt(self.config.hidden_size)
        else:
            x = inputs_embeds * math.sqrt(self.config.hidden_size)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final norm and output
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


# =============================================================================
# MAMBA2 MODEL
# =============================================================================

class Mamba2Block(nn.Module):
    """Mamba2 State Space Model block"""
    def __init__(self, config: LLMLabConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_state = config.state_size
        self.d_conv = config.conv_kernel_size
        self.expand = config.expand_factor
        self.d_inner = self.d_model * self.expand

        # Normalization
        if config.norm_type == "rmsnorm":
            self.norm = nn.RMSNorm(self.d_model, eps=config.norm_eps)
        else:
            self.norm = nn.LayerNorm(self.d_model, eps=config.norm_eps)

        # Projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, self.d_conv,
            groups=self.d_inner, padding=self.d_conv - 1, bias=True
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.randn(self.d_inner, self.d_state)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        dt_rank = (self.d_model + 15) // 16
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, dt_rank, bias=False)
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_input, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_input.transpose(1, 2))[..., :x.size(1)].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(self.x_proj(x_conv)))
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        A = -torch.exp(self.A_log)

        # Simplified selective scan
        y = self._selective_scan(x_conv, dt, A, B, C)
        y = y + self.D * x_conv
        y = y * F.silu(z)

        return residual + self.out_proj(y)

    def _selective_scan(self, x, dt, A, B, C):
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        y = []

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(1)
            x_t = x[:, t, :].unsqueeze(-1)
            C_t = C[:, t, :].unsqueeze(1)

            h = h + dt_t * (A * h + B_t * x_t)
            y_t = (C_t * h).sum(dim=-1)
            y.append(y_t)

        return torch.stack(y, dim=1)


class Mamba2LLM(LLMLabPreTrainedModel):
    """Mamba2 State Space Model for Language Modeling"""

    def __init__(self, config: LLMLabConfig):
        super().__init__(config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Mamba2Block(config) for _ in range(config.num_hidden_layers)
        ])

        if config.norm_type == "rmsnorm":
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
