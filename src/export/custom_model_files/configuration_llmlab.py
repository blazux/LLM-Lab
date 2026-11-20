"""
LLM-Lab Configuration for HuggingFace Transformers

This file is bundled with custom model exports to enable loading via
AutoConfig.from_pretrained() with trust_remote_code=True.
"""

from transformers import PretrainedConfig


class LLMLabConfig(PretrainedConfig):
    """
    Configuration class for LLM-Lab models.

    Supports transformer architectures with various attention
    mechanisms (GQA, MHA, MQA, MLA) and activations.
    """

    model_type = "llm-lab-transformer"

    def __init__(
        self,
        # Core architecture
        model_architecture: str = "transformer",
        vocab_size: int = 151936,
        hidden_size: int = 896,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 1024,

        # Transformer-specific
        num_attention_heads: int = 14,
        num_key_value_heads: int = None,
        intermediate_size: int = 4864,
        attention_type: str = "gqa",
        positional_encoding: str = "rope",
        hidden_act: str = "swiglu",
        attention_bias: bool = False,
        attention_dropout: float = 0.0,

        # Normalization
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        rms_norm_eps: float = None,
        layer_norm_eps: float = None,

        # Modern attention options
        use_attention_gate: bool = False,
        nope_every: int = None,
        rope_fraction: float = 0.5,
        hybrid_pattern: str = None,
        use_sparse_attention: bool = False,
        sparse_top_k: int = 512,
        sparse_indexer_heads: int = 4,
        sparse_indexer_dim: int = 32,

        # MLA-specific
        d_latent: int = None,
        d_rope_latent: int = None,

        # MoE-specific
        use_moe: bool = False,
        num_experts: int = None,
        num_experts_per_token: int = None,
        moe_router_type: str = "softmax",
        moe_aux_free_bias: bool = False,
        moe_shared_expert: bool = False,

        # Other
        tie_word_embeddings: bool = True,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,

        # Legacy field mappings (for LLM-Lab internal config)
        d_model: int = None,
        n_layers: int = None,
        n_heads: int = None,
        n_kv_heads: int = None,
        d_ff: int = None,
        activation: str = None,

        **kwargs
    ):
        # Handle legacy LLM-Lab field names
        if d_model is not None:
            hidden_size = d_model
        if n_layers is not None:
            num_hidden_layers = n_layers
        if n_heads is not None:
            num_attention_heads = n_heads
        if n_kv_heads is not None:
            num_key_value_heads = n_kv_heads
        if d_ff is not None:
            intermediate_size = d_ff
        if activation is not None:
            hidden_act = activation

        # Set norm eps based on type
        if rms_norm_eps is None and norm_type == "rmsnorm":
            rms_norm_eps = norm_eps
        if layer_norm_eps is None and norm_type == "layernorm":
            layer_norm_eps = norm_eps

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        # Core architecture
        self.model_architecture = model_architecture
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings

        # Transformer-specific
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_type = attention_type
        self.positional_encoding = positional_encoding
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Normalization
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_eps = layer_norm_eps

        # Modern attention options
        self.use_attention_gate = use_attention_gate
        self.nope_every = nope_every
        self.rope_fraction = rope_fraction
        self.hybrid_pattern = hybrid_pattern
        self.use_sparse_attention = use_sparse_attention
        self.sparse_top_k = sparse_top_k
        self.sparse_indexer_heads = sparse_indexer_heads
        self.sparse_indexer_dim = sparse_indexer_dim

        # MLA-specific
        self.d_latent = d_latent
        self.d_rope_latent = d_rope_latent

        # MoE-specific
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_router_type = moe_router_type
        self.moe_aux_free_bias = moe_aux_free_bias
        self.moe_shared_expert = moe_shared_expert

        # Other
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Computed values
        self.head_dim = hidden_size // num_attention_heads

    @property
    def d_model(self):
        """Alias for hidden_size (LLM-Lab naming)"""
        return self.hidden_size

    @property
    def n_layers(self):
        """Alias for num_hidden_layers (LLM-Lab naming)"""
        return self.num_hidden_layers

    @property
    def n_heads(self):
        """Alias for num_attention_heads (LLM-Lab naming)"""
        return self.num_attention_heads

    @property
    def n_kv_heads(self):
        """Alias for num_key_value_heads (LLM-Lab naming)"""
        return self.num_key_value_heads

    @property
    def d_ff(self):
        """Alias for intermediate_size (LLM-Lab naming)"""
        return self.intermediate_size

    @property
    def d_k(self):
        """Per-head dimension"""
        return self.head_dim
