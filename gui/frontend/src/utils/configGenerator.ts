import { Node, Edge } from 'reactflow';

interface ModelConfig {
  model_architecture: string;
  tokenizer_name: string;
  d_model: number;
  n_layers: number;
  vocab_size: number;
  max_seq_len: number;
  dropout: number;
  norm_type: string;
  norm_eps: number;
  // Transformer-specific
  positional_encoding?: string;
  attention_type?: string;
  activation?: string;
  n_heads?: number;
  n_kv_heads?: number;
  d_ff?: number;
  sliding_window?: number | null;
  // MLA-specific
  d_latent?: number | null;
  d_rope_latent?: number | null;
  // Mamba2-specific
  state_size?: number;
  expand_factor?: number;
  conv_kernel_size?: number;
  headdim?: number;
  ngroups?: number;
  chunk_size?: number;
  // Common
  tie_word_embeddings: boolean;
  model_type: string;
}

export const generateConfigFromNodes = (
  nodes: Node[],
  edges: Edge[]
): Partial<ModelConfig> => {
  // Detect architecture from presence of Mamba2 or Transformer-specific nodes
  const hasMamba2Nodes = nodes.some(node =>
    node.type === 'ssmcore' ||
    node.type === 'temporalconv' ||
    node.type === 'gating' ||
    node.type === 'headprojection'
  );
  const hasTransformerNodes = nodes.some(node =>
    node.type === 'rope' ||
    node.type === 'alibi' ||
    node.type === 'yarn' ||
    node.type === 'sinusoidal' ||
    node.type === 'mha' ||
    node.type === 'gqa' ||
    node.type === 'mqa' ||
    node.type === 'mla'
  );

  // Determine architecture: Mamba2 if it has Mamba2 nodes, otherwise Transformer
  const isMamba2 = hasMamba2Nodes && !hasTransformerNodes;

  const config: Partial<ModelConfig> = {
    model_architecture: isMamba2 ? 'mamba2' : 'transformer',
    model_type: isMamba2 ? 'custom_mamba2' : 'custom_transformer',
    max_seq_len: 1024, // Default, can be overridden by embedding node
    dropout: 0.0,
    norm_eps: 1e-6,
  };

  // Find the loop edge to determine n_layers
  const loopEdge = edges.find((edge) => edge.data?.isLoop);
  if (loopEdge) {
    config.n_layers = loopEdge.data.repeatCount || 24;
  }

  // Extract config from each node type
  nodes.forEach((node) => {
    switch (node.type) {
      case 'tokenizer':
        config.tokenizer_name = node.data.tokenizer_name || 'Qwen/Qwen2.5-0.5B';
        break;

      case 'embedding':
        config.d_model = node.data.d_model || 896;
        config.vocab_size = node.data.vocab_size || 151936;
        // Allow configuring max_seq_len from embedding node
        if (node.data.max_seq_len) {
          config.max_seq_len = node.data.max_seq_len;
        }
        break;

      // Positional encoding nodes (Transformer only)
      case 'rope':
        config.positional_encoding = 'rope';
        // max_seq_len now configured from embedding node, not here
        break;
      case 'alibi':
        config.positional_encoding = 'alibi';
        break;
      case 'yarn':
        config.positional_encoding = 'yarn';
        break;
      case 'sinusoidal':
        config.positional_encoding = 'sinusoidal';
        break;
      case 'learned':
        config.positional_encoding = 'learned';
        break;

      // Attention nodes (Transformer only)
      case 'mha':
        config.attention_type = 'mha';
        config.n_heads = node.data.n_heads || 14;
        config.n_kv_heads = node.data.n_heads || 14; // MHA has same number of KV heads as Q heads
        config.sliding_window = node.data.sliding_window || null;
        break;
      case 'gqa':
        config.attention_type = 'gqa';
        config.n_heads = node.data.n_heads || 14;
        config.n_kv_heads = node.data.n_kv_heads || 2;
        config.sliding_window = node.data.sliding_window || null;
        break;
      case 'mqa':
        config.attention_type = 'mqa';
        config.n_heads = node.data.n_heads || 14;
        config.n_kv_heads = 1; // MQA always uses 1 KV head
        config.sliding_window = node.data.sliding_window || null;
        break;
      case 'mla':
        config.attention_type = 'mla';
        config.n_heads = node.data.n_heads || 14;
        config.sliding_window = node.data.sliding_window || null;
        config.d_latent = node.data.d_latent || null;
        config.d_rope_latent = node.data.d_rope_latent || null;
        break;

      // Normalization nodes (both architectures)
      case 'rmsnorm':
        config.norm_type = 'rmsnorm';
        break;
      case 'layernorm':
        config.norm_type = 'layernorm';
        break;

      // FFN nodes (Transformer only)
      case 'swiglu':
        config.activation = 'swiglu';
        config.d_ff = node.data.d_ff || 3584;
        break;
      case 'geglu':
        config.activation = 'geglu';
        config.d_ff = node.data.d_ff || 3584;
        break;
      case 'reglu':
        config.activation = 'reglu';
        config.d_ff = node.data.d_ff || 3584;
        break;
      case 'gelu':
        config.activation = 'gelu';
        config.d_ff = node.data.d_ff || 3584;
        break;
      case 'relu':
        config.activation = 'relu';
        config.d_ff = node.data.d_ff || 3584;
        break;

      // Mamba2 nodes
      case 'ssmcore':
        config.state_size = node.data.state_size || 64;
        break;
      case 'temporalconv':
        config.conv_kernel_size = node.data.conv_kernel_size || 4;
        break;
      case 'gating':
        config.expand_factor = node.data.expand_factor || 2;
        break;
      case 'headprojection':
        config.headdim = node.data.headdim || 64;
        config.ngroups = node.data.ngroups || 1;
        break;

      case 'lmhead':
        config.tie_word_embeddings = node.data.tie_weights ?? true;
        break;
    }
  });

  // Set default Mamba2 parameters if architecture is mamba2 but nodes are missing
  if (isMamba2) {
    if (!config.state_size) config.state_size = 64;
    if (!config.expand_factor) config.expand_factor = 2;
    if (!config.conv_kernel_size) config.conv_kernel_size = 4;
    if (!config.headdim) config.headdim = 64;
    if (!config.ngroups) config.ngroups = 1;
    config.chunk_size = 256; // Always set chunk_size for Mamba2
  }

  return config;
};

export const downloadConfig = (config: Partial<ModelConfig>, filename: string = 'model_config.json') => {
  const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
