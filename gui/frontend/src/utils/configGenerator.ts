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
  positional_encoding: string;
  attention_type: string;
  activation: string;
  n_heads: number;
  n_kv_heads: number;
  d_ff: number;
  tie_word_embeddings: boolean;
  model_type: string;
}

export const generateConfigFromNodes = (
  nodes: Node[],
  edges: Edge[]
): Partial<ModelConfig> => {
  const config: Partial<ModelConfig> = {
    model_architecture: 'transformer',
    model_type: 'custom_transformer',
    max_seq_len: 1024,
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
        break;

      // Positional encoding nodes
      case 'rope':
        config.positional_encoding = 'rope';
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

      // Attention nodes
      case 'mha':
        config.attention_type = 'mha';
        config.n_heads = node.data.n_heads || 14;
        config.n_kv_heads = node.data.n_heads || 14; // MHA has same number of KV heads as Q heads
        break;
      case 'gqa':
        config.attention_type = 'gqa';
        config.n_heads = node.data.n_heads || 14;
        config.n_kv_heads = node.data.n_kv_heads || 2;
        break;
      case 'mqa':
        config.attention_type = 'mqa';
        config.n_heads = node.data.n_heads || 14;
        config.n_kv_heads = 1; // MQA always uses 1 KV head
        break;
      case 'mla':
        config.attention_type = 'mla';
        config.n_heads = node.data.n_heads || 14;
        break;

      // Normalization nodes
      case 'rmsnorm':
        config.norm_type = 'rmsnorm';
        break;
      case 'layernorm':
        config.norm_type = 'layernorm';
        break;

      // FFN nodes
      case 'swiglu':
        config.activation = 'swiglu';
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

      case 'lmhead':
        config.tie_word_embeddings = node.data.tie_weights ?? true;
        break;
    }
  });

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
