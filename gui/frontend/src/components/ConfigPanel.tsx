import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { Node } from 'reactflow';
import { getCheckpoints, Checkpoint } from '../services/trainingApi';

interface ConfigPanelProps {
  node: Node | null;
  onClose: () => void;
  onUpdate: (nodeId: string, data: any) => void;
}

const ConfigPanel = ({ node, onClose, onUpdate }: ConfigPanelProps) => {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(false);
  const [useCustomPath, setUseCustomPath] = useState(false);

  // Fetch checkpoints when needed
  useEffect(() => {
    const shouldFetchCheckpoints =
      (node?.type === 'model' && node?.data?.resume_training) ||
      node?.type === 'basemodel';

    if (shouldFetchCheckpoints) {
      setLoadingCheckpoints(true);
      getCheckpoints()
        .then((ckpts) => {
          setCheckpoints(ckpts);
          setLoadingCheckpoints(false);
        })
        .catch((err) => {
          console.error('Failed to fetch checkpoints:', err);
          setLoadingCheckpoints(false);
        });
    }
  }, [node?.type, node?.data?.resume_training]);

  if (!node) return null;

  const handleChange = (key: string, value: any) => {
    onUpdate(node.id, {
      ...node.data,
      [key]: value,
    });
  };

  const renderConfigFields = () => {
    const type = node.type;

    switch (type) {
      case 'tokenizer':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Tokenizer Name
              </label>
              <input
                type="text"
                value={node.data.tokenizer_name || 'Qwen/Qwen2.5-0.5B'}
                onChange={(e) => handleChange('tokenizer_name', e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                HuggingFace tokenizer identifier
              </p>
            </div>
          </div>
        );

      case 'embedding':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Model Dimension (d_model)
              </label>
              <input
                type="number"
                value={node.data.d_model || 896}
                onChange={(e) => handleChange('d_model', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Hidden size of the model
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Vocabulary Size
              </label>
              <input
                type="number"
                value={node.data.vocab_size || 151936}
                onChange={(e) => handleChange('vocab_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Number of tokens in the vocabulary
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Dropout
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={node.data.dropout ?? 0.0}
                onChange={(e) => handleChange('dropout', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Dropout probability for regularization (0.0 - 1.0)
              </p>
            </div>
          </div>
        );

      // Positional Encoding nodes
      case 'rope':
        return (
          <div className="space-y-4">
            <div className="text-slate-300 text-sm">
              <p className="mb-2">RoPE (Rotary Position Embedding) is a modern positional encoding method that applies rotations to query and key vectors.</p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Max Sequence Length
              </label>
              <input
                type="number"
                value={node.data.max_seq_len || 1024}
                onChange={(e) => handleChange('max_seq_len', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Maximum sequence length for positional encoding
              </p>
            </div>
          </div>
        );

      case 'alibi':
        return (
          <div className="space-y-4">
            <div className="text-slate-300 text-sm">
              <p className="mb-2">ALiBi (Attention with Linear Biases) adds position-dependent biases to attention scores.</p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Max Sequence Length
              </label>
              <input
                type="number"
                value={node.data.max_seq_len || 1024}
                onChange={(e) => handleChange('max_seq_len', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Maximum sequence length for positional encoding
              </p>
            </div>
          </div>
        );

      case 'yarn':
        return (
          <div className="space-y-4">
            <div className="text-slate-300 text-sm">
              <p className="mb-2">YARN (Yet Another RoPE extensioN) extends RoPE for longer context lengths.</p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Max Sequence Length
              </label>
              <input
                type="number"
                value={node.data.max_seq_len || 1024}
                onChange={(e) => handleChange('max_seq_len', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Maximum sequence length for positional encoding
              </p>
            </div>
          </div>
        );

      case 'sinusoidal':
        return (
          <div className="space-y-4">
            <div className="text-slate-300 text-sm">
              <p className="mb-2">Sinusoidal Positional Encoding uses sine and cosine functions at different frequencies (original Transformer).</p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Max Sequence Length
              </label>
              <input
                type="number"
                value={node.data.max_seq_len || 1024}
                onChange={(e) => handleChange('max_seq_len', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Maximum sequence length for positional encoding
              </p>
            </div>
          </div>
        );

      case 'learned':
        return (
          <div className="space-y-4">
            <div className="text-slate-300 text-sm">
              <p className="mb-2">Learned Positional Embeddings are trainable position vectors (like BERT/GPT-1). Classic baseline for comparing against modern methods like RoPE.</p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Max Sequence Length
              </label>
              <input
                type="number"
                value={node.data.max_seq_len || 1024}
                onChange={(e) => handleChange('max_seq_len', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Maximum sequence length for positional encoding
              </p>
            </div>
          </div>
        );

      // Attention nodes
      case 'mha':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Number of Heads
              </label>
              <input
                type="number"
                value={node.data.n_heads || 14}
                onChange={(e) => handleChange('n_heads', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Multi-Head Attention splits attention into multiple heads for richer representations
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Sliding Window Size
              </label>
              <input
                type="number"
                value={node.data.sliding_window || ''}
                onChange={(e) => handleChange('sliding_window', e.target.value ? parseInt(e.target.value) : null)}
                placeholder="None (full attention)"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Limit attention to nearby tokens (e.g., 512, 1024, 2048). Leave empty for full attention.
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Attention Bias
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={node.data.attention_bias ?? false}
                  onChange={(e) => handleChange('attention_bias', e.target.checked)}
                  className="w-4 h-4 bg-slate-700 border-slate-600 rounded text-green-500 focus:ring-2 focus:ring-green-500"
                />
                <span className="text-white text-sm">
                  Use bias in attention projections
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Typically disabled for modern LLMs (Llama, Qwen, Mistral)
              </p>
            </div>
          </div>
        );

      case 'gqa':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium text-slate-300 mb-2 block">
                  Heads
                </label>
                <input
                  type="number"
                  value={node.data.n_heads || 14}
                  onChange={(e) => handleChange('n_heads', parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-300 mb-2 block">
                  KV Heads
                </label>
                <input
                  type="number"
                  value={node.data.n_kv_heads || 2}
                  onChange={(e) => handleChange('n_kv_heads', parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
              </div>
            </div>
            <p className="text-xs text-slate-400">
              Grouped-Query Attention groups multiple query heads per key/value head for efficiency
            </p>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Sliding Window Size
              </label>
              <input
                type="number"
                value={node.data.sliding_window || ''}
                onChange={(e) => handleChange('sliding_window', e.target.value ? parseInt(e.target.value) : null)}
                placeholder="None (full attention)"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Limit attention to nearby tokens (e.g., 512, 1024, 2048). Leave empty for full attention.
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Attention Bias
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={node.data.attention_bias ?? false}
                  onChange={(e) => handleChange('attention_bias', e.target.checked)}
                  className="w-4 h-4 bg-slate-700 border-slate-600 rounded text-emerald-500 focus:ring-2 focus:ring-emerald-500"
                />
                <span className="text-white text-sm">
                  Use bias in attention projections
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Typically disabled for modern LLMs (Llama, Qwen, Mistral)
              </p>
            </div>
          </div>
        );

      case 'mqa':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Number of Heads
              </label>
              <input
                type="number"
                value={node.data.n_heads || 14}
                onChange={(e) => handleChange('n_heads', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Multi-Query Attention uses a single key/value head shared across all query heads
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Sliding Window Size
              </label>
              <input
                type="number"
                value={node.data.sliding_window || ''}
                onChange={(e) => handleChange('sliding_window', e.target.value ? parseInt(e.target.value) : null)}
                placeholder="None (full attention)"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Limit attention to nearby tokens (e.g., 512, 1024, 2048). Leave empty for full attention.
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Attention Bias
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={node.data.attention_bias ?? false}
                  onChange={(e) => handleChange('attention_bias', e.target.checked)}
                  className="w-4 h-4 bg-slate-700 border-slate-600 rounded text-teal-500 focus:ring-2 focus:ring-teal-500"
                />
                <span className="text-white text-sm">
                  Use bias in attention projections
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Typically disabled for modern LLMs (Llama, Qwen, Mistral)
              </p>
            </div>
          </div>
        );

      case 'mla':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Number of Heads
              </label>
              <input
                type="number"
                value={node.data.n_heads || 14}
                onChange={(e) => handleChange('n_heads', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Multi-head Latent Attention uses compressed latent representations for efficient attention
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Latent Dimension (d_latent)
              </label>
              <input
                type="number"
                value={node.data.d_latent || ''}
                onChange={(e) => handleChange('d_latent', e.target.value ? parseInt(e.target.value) : null)}
                placeholder="Auto (d_model / 4)"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Compressed dimension for KV cache (default: d_model / 4)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                RoPE Latent Dimension (d_rope_latent)
              </label>
              <input
                type="number"
                value={node.data.d_rope_latent || ''}
                onChange={(e) => handleChange('d_rope_latent', e.target.value ? parseInt(e.target.value) : null)}
                placeholder="Auto (d_model / n_heads)"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Separate latent dimension for RoPE (default: d_model / n_heads)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Sliding Window Size
              </label>
              <input
                type="number"
                value={node.data.sliding_window || ''}
                onChange={(e) => handleChange('sliding_window', e.target.value ? parseInt(e.target.value) : null)}
                placeholder="None (full attention)"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Limit attention to nearby tokens (e.g., 512, 1024, 2048). Leave empty for full attention.
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Attention Bias
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={node.data.attention_bias ?? false}
                  onChange={(e) => handleChange('attention_bias', e.target.checked)}
                  className="w-4 h-4 bg-slate-700 border-slate-600 rounded text-cyan-500 focus:ring-2 focus:ring-cyan-500"
                />
                <span className="text-white text-sm">
                  Use bias in attention projections
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Typically disabled for modern LLMs (Llama, Qwen, Mistral)
              </p>
            </div>
          </div>
        );

      // Normalization nodes
      case 'rmsnorm':
        return (
          <div className="space-y-4">
            <div className="text-slate-300 text-sm">
              <p className="mb-2">RMSNorm (Root Mean Square Normalization) is a faster alternative to LayerNorm.</p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Epsilon
              </label>
              <input
                type="number"
                step="0.000001"
                value={node.data.norm_eps ?? 1e-6}
                onChange={(e) => handleChange('norm_eps', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Small constant for numerical stability (default: 1e-6)
              </p>
            </div>
          </div>
        );

      case 'layernorm':
        return (
          <div className="space-y-4">
            <div className="text-slate-300 text-sm">
              <p className="mb-2">LayerNorm is the classic normalization method from the original Transformer.</p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Epsilon
              </label>
              <input
                type="number"
                step="0.000001"
                value={node.data.norm_eps ?? 1e-5}
                onChange={(e) => handleChange('norm_eps', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Small constant for numerical stability (default: 1e-5)
              </p>
            </div>
          </div>
        );

      // FFN nodes
      case 'swiglu':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Hidden Dimension (d_ff)
              </label>
              <input
                type="number"
                value={node.data.d_ff || 3584}
                onChange={(e) => handleChange('d_ff', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                SwiGLU is a modern activation function used in state-of-the-art models
              </p>
            </div>
          </div>
        );

      case 'geglu':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Hidden Dimension (d_ff)
              </label>
              <input
                type="number"
                value={node.data.d_ff || 3584}
                onChange={(e) => handleChange('d_ff', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                GeGLU (GELU-Gated Linear Unit) is used in T5, PaLM, and many modern LLMs
              </p>
            </div>
          </div>
        );

      case 'reglu':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Hidden Dimension (d_ff)
              </label>
              <input
                type="number"
                value={node.data.d_ff || 3584}
                onChange={(e) => handleChange('d_ff', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-red-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                ReGLU (ReLU-Gated Linear Unit) is simpler and faster than SwiGLU/GeGLU
              </p>
            </div>
          </div>
        );

      case 'gelu':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Hidden Dimension (d_ff)
              </label>
              <input
                type="number"
                value={node.data.d_ff || 3584}
                onChange={(e) => handleChange('d_ff', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-red-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                GELU (Gaussian Error Linear Unit) is a smooth activation function
              </p>
            </div>
          </div>
        );

      case 'relu':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Hidden Dimension (d_ff)
              </label>
              <input
                type="number"
                value={node.data.d_ff || 3584}
                onChange={(e) => handleChange('d_ff', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-pink-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                ReLU is the classic activation function: max(0, x)
              </p>
            </div>
          </div>
        );

      case 'moe_router':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Number of Experts
              </label>
              <input
                type="number"
                min="2"
                max="64"
                value={node.data.num_experts || 8}
                onChange={(e) => handleChange('num_experts', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Total number of expert FFNs (typically 8 for Mixtral, 64 for DeepSeek)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Experts per Token (Top-K)
              </label>
              <input
                type="number"
                min="1"
                max="8"
                value={node.data.num_experts_per_token || 2}
                onChange={(e) => handleChange('num_experts_per_token', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                How many experts process each token (typically 2)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Load Balancing Loss Weight
              </label>
              <input
                type="number"
                step="0.001"
                min="0"
                max="1"
                value={node.data.load_balancing_loss_weight || 0.01}
                onChange={(e) => handleChange('load_balancing_loss_weight', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Auxiliary loss weight to encourage even expert utilization (default: 0.01)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Router Z-Loss Weight
              </label>
              <input
                type="number"
                step="0.0001"
                min="0"
                max="0.1"
                value={node.data.router_z_loss_weight || 0.001}
                onChange={(e) => handleChange('router_z_loss_weight', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Prevents overconfident routing for stability (default: 0.001)
              </p>
            </div>
            <div className="p-3 bg-purple-900/20 border border-purple-500/30 rounded-md">
              <p className="text-xs text-purple-300">
                💡 Connect an FFN node below to define the expert architecture.
                The router will create {node.data.num_experts || 8} copies with learned routing.
              </p>
            </div>
          </div>
        );

      case 'lmhead':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Weight Tying
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={node.data.tie_weights ?? true}
                  onChange={(e) => handleChange('tie_weights', e.target.checked)}
                  className="w-4 h-4 bg-slate-700 border-slate-600 rounded text-pink-500 focus:ring-2 focus:ring-pink-500"
                />
                <span className="text-white text-sm">
                  Tie weights with embedding layer
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                When enabled, shares the embedding matrix for output projection (saves memory). Vocabulary size is configured in the Embedding node.
              </p>
            </div>
          </div>
        );

      case 'dataset':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Dataset Name
              </label>
              <input
                type="text"
                value={node.data.dataset_name || node.data.name || ''}
                onChange={(e) => {
                  // Update dataset_name and clear old 'name' field to avoid conflicts
                  onUpdate(node.id, {
                    ...node.data,
                    dataset_name: e.target.value,
                    name: undefined  // Clear old field
                  });
                }}
                placeholder="HuggingFaceFW/fineweb-edu"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                HuggingFace dataset identifier
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Subset (optional)
              </label>
              <input
                type="text"
                value={node.data.subset || ''}
                onChange={(e) => handleChange('subset', e.target.value || undefined)}
                placeholder="e.g., fra_Latn"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Leave empty if dataset has no subsets
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Split
              </label>
              <input
                type="text"
                value={node.data.split || 'train'}
                onChange={(e) => handleChange('split', e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Weight (optional)
              </label>
              <input
                type="number"
                step="0.1"
                value={node.data.weight || 1.0}
                onChange={(e) => handleChange('weight', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Sampling weight for dataset mixing
              </p>
            </div>
          </div>
        );

      case 'optimizer':
        const optimizerType = node.data.optimizer || 'muon';
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Optimizer Type
              </label>
              <select
                value={optimizerType}
                onChange={(e) => handleChange('optimizer', e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
              >
                <option value="adamw">AdamW</option>
                <option value="muon">Muon</option>
                <option value="lion">Lion</option>
                <option value="sophia">Sophia</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.001"
                value={node.data.lr || 0.05}
                onChange={(e) => handleChange('lr', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Weight Decay
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.weight_decay || 0.1}
                onChange={(e) => handleChange('weight_decay', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
            </div>

            {/* AdamW-specific parameters */}
            {optimizerType === 'adamw' && (
              <>
                <div className="border-t border-slate-600 pt-4">
                  <h4 className="text-sm font-semibold text-orange-400 mb-3">AdamW Parameters</h4>
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Beta1
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={node.data.adamw_beta1 || 0.9}
                    onChange={(e) => handleChange('adamw_beta1', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Beta2
                  </label>
                  <input
                    type="number"
                    step="0.001"
                    value={node.data.adamw_beta2 || 0.999}
                    onChange={(e) => handleChange('adamw_beta2', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Epsilon
                  </label>
                  <input
                    type="number"
                    step="0.00000001"
                    value={node.data.adamw_eps || 1e-8}
                    onChange={(e) => handleChange('adamw_eps', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
              </>
            )}

            {/* Muon-specific parameters */}
            {optimizerType === 'muon' && (
              <>
                <div className="border-t border-slate-600 pt-4">
                  <h4 className="text-sm font-semibold text-orange-400 mb-3">Muon Parameters</h4>
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Momentum
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={node.data.muon_momentum || 0.95}
                    onChange={(e) => handleChange('muon_momentum', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Use Nesterov
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={node.data.muon_nesterov ?? true}
                      onChange={(e) => handleChange('muon_nesterov', e.target.checked)}
                      className="w-4 h-4 bg-slate-700 border-slate-600 rounded text-orange-500 focus:ring-2 focus:ring-orange-500"
                    />
                    <span className="text-white text-sm">
                      Enable Nesterov momentum
                    </span>
                  </div>
                </div>
              </>
            )}

            {/* Lion-specific parameters */}
            {optimizerType === 'lion' && (
              <>
                <div className="border-t border-slate-600 pt-4">
                  <h4 className="text-sm font-semibold text-orange-400 mb-3">Lion Parameters</h4>
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Beta1
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={node.data.lion_beta1 || 0.9}
                    onChange={(e) => handleChange('lion_beta1', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Beta2
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={node.data.lion_beta2 || 0.99}
                    onChange={(e) => handleChange('lion_beta2', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
              </>
            )}

            {/* Sophia-specific parameters */}
            {optimizerType === 'sophia' && (
              <>
                <div className="border-t border-slate-600 pt-4">
                  <h4 className="text-sm font-semibold text-orange-400 mb-3">Sophia Parameters</h4>
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Beta1
                  </label>
                  <input
                    type="number"
                    step="0.001"
                    value={node.data.sophia_beta1 || 0.965}
                    onChange={(e) => handleChange('sophia_beta1', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Beta2
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={node.data.sophia_beta2 || 0.99}
                    onChange={(e) => handleChange('sophia_beta2', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Rho (Clipping)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={node.data.sophia_rho || 0.04}
                    onChange={(e) => handleChange('sophia_rho', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
              </>
            )}
          </div>
        );

      case 'scheduler':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Scheduler Type
              </label>
              <select
                value={node.data.scheduler || 'cosine'}
                onChange={(e) => handleChange('scheduler', e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="none">None (Constant)</option>
                <option value="cosine">Cosine</option>
                <option value="linear">Linear</option>
                <option value="polynomial">Polynomial</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Warmup Steps
              </label>
              <input
                type="number"
                value={node.data.warmup_steps || 1000}
                onChange={(e) => handleChange('warmup_steps', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>
        );

      case 'adamw':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                value={node.data.lr || 0.0001}
                onChange={(e) => handleChange('lr', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-red-500"
              />
            </div>
            <div className="border-t border-slate-600 pt-4">
              <h4 className="text-sm font-semibold text-red-400 mb-3">AdamW Parameters</h4>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Beta1
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.beta1 || 0.9}
                onChange={(e) => handleChange('beta1', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-red-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Beta2
              </label>
              <input
                type="number"
                step="0.001"
                value={node.data.beta2 || 0.999}
                onChange={(e) => handleChange('beta2', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-red-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Epsilon
              </label>
              <input
                type="number"
                step="0.00000001"
                value={node.data.eps || 1e-8}
                onChange={(e) => handleChange('eps', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-red-500"
              />
            </div>
          </div>
        );

      case 'muon':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.001"
                value={node.data.lr || 0.05}
                onChange={(e) => handleChange('lr', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
            </div>
            <div className="border-t border-slate-600 pt-4">
              <h4 className="text-sm font-semibold text-orange-400 mb-3">Muon Parameters</h4>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Momentum
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.momentum || 0.95}
                onChange={(e) => handleChange('momentum', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Use Nesterov
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={node.data.nesterov ?? true}
                  onChange={(e) => handleChange('nesterov', e.target.checked)}
                  className="w-4 h-4 bg-slate-700 border-slate-600 rounded text-orange-500 focus:ring-2 focus:ring-orange-500"
                />
                <span className="text-white text-sm">
                  Enable Nesterov momentum
                </span>
              </div>
            </div>
          </div>
        );

      case 'lion':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                value={node.data.lr || 0.0003}
                onChange={(e) => handleChange('lr', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div className="border-t border-slate-600 pt-4">
              <h4 className="text-sm font-semibold text-amber-400 mb-3">Lion Parameters</h4>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Beta1
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.beta1 || 0.9}
                onChange={(e) => handleChange('beta1', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Beta2
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.beta2 || 0.99}
                onChange={(e) => handleChange('beta2', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
          </div>
        );

      case 'sophia':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                value={node.data.lr || 0.0001}
                onChange={(e) => handleChange('lr', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-rose-500"
              />
            </div>
            <div className="border-t border-slate-600 pt-4">
              <h4 className="text-sm font-semibold text-rose-400 mb-3">Sophia Parameters</h4>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Beta1
              </label>
              <input
                type="number"
                step="0.001"
                value={node.data.beta1 || 0.965}
                onChange={(e) => handleChange('beta1', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-rose-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Beta2
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.beta2 || 0.99}
                onChange={(e) => handleChange('beta2', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-rose-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Rho (Clipping)
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.rho || 0.04}
                onChange={(e) => handleChange('rho', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-rose-500"
              />
            </div>
          </div>
        );

      case 'cosine':
      case 'linear':
      case 'polynomial':
      case 'constant':
        const schedulerColors = {
          cosine: 'purple',
          linear: 'violet',
          polynomial: 'fuchsia',
          constant: 'slate'
        };
        const color = schedulerColors[type as keyof typeof schedulerColors] || 'purple';
        return (
          <div className="space-y-4">
            {type !== 'constant' && (
              <div>
                <label className="text-sm font-medium text-slate-300 mb-2 block">
                  Warmup Steps
                </label>
                <input
                  type="number"
                  value={node.data.warmup_steps || 1000}
                  onChange={(e) => handleChange('warmup_steps', parseInt(e.target.value))}
                  className={`w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-${color}-500`}
                />
                <p className="text-xs text-slate-400 mt-1">
                  Number of steps for learning rate warmup
                </p>
              </div>
            )}
            {type === 'polynomial' && (
              <div>
                <label className="text-sm font-medium text-slate-300 mb-2 block">
                  Power
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={node.data.power || 2.0}
                  onChange={(e) => handleChange('power', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-fuchsia-500"
                />
                <p className="text-xs text-slate-400 mt-1">
                  Polynomial decay power
                </p>
              </div>
            )}
            {type === 'constant' && (
              <div className="text-slate-400 text-sm">
                <p>Constant learning rate - no decay applied.</p>
                <p className="mt-2">The optimizer's base learning rate will remain unchanged throughout training.</p>
              </div>
            )}
          </div>
        );

      case 'hyperparams':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Batch Size
              </label>
              <input
                type="number"
                value={node.data.batch_size || 1}
                onChange={(e) => handleChange('batch_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Gradient Accumulation Steps
              </label>
              <input
                type="number"
                value={node.data.gradient_accumulation_steps || 64}
                onChange={(e) => handleChange('gradient_accumulation_steps', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Max Steps
              </label>
              <input
                type="number"
                value={node.data.max_steps || 10000}
                onChange={(e) => handleChange('max_steps', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Total steps for fresh training, or additional steps when resuming from checkpoint
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Weight Decay
              </label>
              <input
                type="number"
                step="0.001"
                value={node.data.weight_decay ?? 0.0}
                onChange={(e) => handleChange('weight_decay', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                L2 regularization weight (0.0 = no weight decay)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Gradient Clipping
              </label>
              <input
                type="number"
                step="0.1"
                value={node.data.grad_clip || 1.0}
                onChange={(e) => handleChange('grad_clip', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Eval Every (steps)
              </label>
              <input
                type="number"
                value={node.data.eval_every || 500}
                onChange={(e) => handleChange('eval_every', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Eval Steps
              </label>
              <input
                type="number"
                value={node.data.eval_steps || 100}
                onChange={(e) => handleChange('eval_steps', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Number of batches to use for each evaluation
              </p>
            </div>
          </div>
        );

      case 'model':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Model Configuration Source
              </label>
              <select
                value={node.data.config_source || 'current'}
                onChange={(e) => handleChange('config_source', e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="current">Use Current Model Design</option>
                <option value="file">Load from JSON File</option>
              </select>
              <p className="text-xs text-slate-400 mt-1">
                {node.data.config_source === 'current'
                  ? 'Use the model designed in Model Architecture tab'
                  : 'Load model config from a JSON file'}
              </p>
            </div>

            {node.data.config_source === 'file' && (
              <div>
                <label className="text-sm font-medium text-slate-300 mb-2 block">
                  Model Config Path
                </label>
                <input
                  type="text"
                  value={node.data.config_path || ''}
                  onChange={(e) => handleChange('config_path', e.target.value)}
                  placeholder="model_config.json"
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            )}

            <div className="border-t border-slate-600 pt-4">
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Training Mode
              </label>
              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="training_mode"
                    checked={!node.data.resume_training}
                    onChange={() => handleChange('resume_training', false)}
                    className="text-indigo-500 focus:ring-2 focus:ring-indigo-500"
                  />
                  <span className="text-white text-sm">Fresh Training (from scratch)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="training_mode"
                    checked={node.data.resume_training || false}
                    onChange={() => handleChange('resume_training', true)}
                    className="text-indigo-500 focus:ring-2 focus:ring-indigo-500"
                  />
                  <span className="text-white text-sm">Resume from Checkpoint</span>
                </label>
              </div>
            </div>

            {node.data.resume_training && (
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium text-slate-300 mb-2 block">
                    Select Checkpoint
                  </label>

                  {!useCustomPath ? (
                    <>
                      <select
                        value={node.data.checkpoint_path || ''}
                        onChange={(e) => handleChange('checkpoint_path', e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        disabled={loadingCheckpoints}
                      >
                        <option value="">
                          {loadingCheckpoints ? 'Loading checkpoints...' : 'Select a checkpoint...'}
                        </option>
                        {checkpoints.map((ckpt) => (
                          <option key={ckpt.path} value={ckpt.path}>
                            {ckpt.name} - {ckpt.type} (Step: {ckpt.step}, {ckpt.size_mb}MB)
                          </option>
                        ))}
                      </select>
                      <div className="flex items-center justify-between mt-2">
                        <p className="text-xs text-slate-400">
                          Available checkpoints from data/ folder
                        </p>
                        <button
                          onClick={() => setUseCustomPath(true)}
                          className="text-xs text-indigo-400 hover:text-indigo-300"
                        >
                          Enter custom path
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <input
                        type="text"
                        value={node.data.checkpoint_path || ''}
                        onChange={(e) => handleChange('checkpoint_path', e.target.value)}
                        placeholder="/app/data/checkpoints/best_model.pt"
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                      <div className="flex items-center justify-between mt-2">
                        <p className="text-xs text-slate-400">
                          Enter full path to checkpoint file
                        </p>
                        <button
                          onClick={() => setUseCustomPath(false)}
                          className="text-xs text-indigo-400 hover:text-indigo-300"
                        >
                          Select from list
                        </button>
                      </div>
                    </>
                  )}
                </div>

                {/* Show checkpoint info */}
                {node.data.checkpoint_path && (
                  <div className="border-t border-slate-600 pt-3 mt-3">
                    {(() => {
                      const selectedCheckpoint = checkpoints.find(
                        (ckpt) => ckpt.path === node.data.checkpoint_path
                      );
                      if (selectedCheckpoint && typeof selectedCheckpoint.step === 'number') {
                        return (
                          <div className="bg-slate-700 p-3 rounded-lg">
                            <p className="text-xs text-slate-300 mb-1">
                              Checkpoint Status
                            </p>
                            <p className="text-sm text-white font-semibold">
                              {selectedCheckpoint.step.toLocaleString()} steps already completed
                            </p>
                            <p className="text-xs text-slate-400 mt-1">
                              {selectedCheckpoint.type} model • {selectedCheckpoint.size_mb}MB
                            </p>
                            <p className="text-xs text-amber-400 mt-2">
                              💡 Max Steps in hyperparameters will be added to this checkpoint
                            </p>
                          </div>
                        );
                      }
                      return null;
                    })()}
                  </div>
                )}
              </div>
            )}
          </div>
        );

      case 'basemodel':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Checkpoint Path
              </label>

              {!useCustomPath ? (
                <>
                  <select
                    value={node.data.checkpoint_path || ''}
                    onChange={(e) => handleChange('checkpoint_path', e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={loadingCheckpoints}
                  >
                    <option value="">
                      {loadingCheckpoints ? 'Loading checkpoints...' : 'Select a checkpoint...'}
                    </option>
                    {checkpoints.map((ckpt) => (
                      <option key={ckpt.path} value={ckpt.path}>
                        {ckpt.name} - {ckpt.type} (Step: {ckpt.step}, {ckpt.size_mb}MB)
                      </option>
                    ))}
                  </select>
                  <div className="flex items-center justify-between mt-2">
                    <p className="text-xs text-slate-400">
                      Available checkpoints from data/ folder
                    </p>
                    <button
                      onClick={() => setUseCustomPath(true)}
                      className="text-xs text-blue-400 hover:text-blue-300"
                    >
                      Enter custom path
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <input
                    type="text"
                    value={node.data.checkpoint_path || ''}
                    onChange={(e) => handleChange('checkpoint_path', e.target.value)}
                    placeholder="/app/data/checkpoints/best_model.pt"
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <div className="flex items-center justify-between mt-2">
                    <p className="text-xs text-slate-400">
                      Enter full path to checkpoint file
                    </p>
                    <button
                      onClick={() => setUseCustomPath(false)}
                      className="text-xs text-blue-400 hover:text-blue-300"
                    >
                      Select from list
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        );

      case 'lora':
        return (
          <div className="space-y-4">
            <p className="text-xs text-slate-400">
              Parameter-efficient fine-tuning with LoRA. Remove this node to disable LoRA.
            </p>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                LoRA Preset
              </label>
              <select
                value={node.data.preset || 'minimal'}
                onChange={(e) => handleChange('preset', e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="minimal">Minimal (Q, V)</option>
                <option value="attention_only">Attention Only</option>
                <option value="ffn_only">FFN Only</option>
                <option value="all">All Layers</option>
                <option value="custom">Custom</option>
              </select>
              <p className="text-xs text-slate-400 mt-1">
                Selects which layers to apply LoRA to
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                LoRA Rank (r)
              </label>
              <input
                type="number"
                value={node.data.lora_r || 8}
                onChange={(e) => handleChange('lora_r', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Lower rank = fewer parameters (default: 8)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                LoRA Alpha
              </label>
              <input
                type="number"
                value={node.data.lora_alpha || 16}
                onChange={(e) => handleChange('lora_alpha', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Scaling factor (default: 16)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                LoRA Dropout
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.lora_dropout || 0.05}
                onChange={(e) => handleChange('lora_dropout', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Dropout probability (default: 0.05)
              </p>
            </div>
          </div>
        );

      case 'ppo_reward':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Reward Model Name
              </label>
              <input
                type="text"
                value={node.data.model_name || 'OpenAssistant/reward-model-deberta-v3-large-v2'}
                onChange={(e) => handleChange('model_name', e.target.value)}
                placeholder="OpenAssistant/reward-model-deberta-v3-large-v2"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-red-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                HuggingFace reward model identifier for PPO
              </p>
            </div>
          </div>
        );

      case 'dpo_reference':
        return (
          <div className="space-y-4">
            <p className="text-xs text-slate-400 mb-2">
              Reference model for DPO. Leave checkpoint empty to use the policy checkpoint.
            </p>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Reference Checkpoint (Optional)
              </label>
              <input
                type="text"
                value={node.data.checkpoint_path || ''}
                onChange={(e) => handleChange('checkpoint_path', e.target.value)}
                placeholder="(use policy checkpoint)"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Path to reference model checkpoint (optional)
              </p>
            </div>
          </div>
        );

      case 'grpo_reward':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Reward Model Name
              </label>
              <input
                type="text"
                value={node.data.model_name || 'OpenAssistant/reward-model-deberta-v3-large-v2'}
                onChange={(e) => handleChange('model_name', e.target.value)}
                placeholder="OpenAssistant/reward-model-deberta-v3-large-v2"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                HuggingFace reward model identifier for GRPO
              </p>
            </div>
          </div>
        );

      case 'rlhf_hyperparams':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Batch Size
              </label>
              <input
                type="number"
                value={node.data.batch_size || 4}
                onChange={(e) => handleChange('batch_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Mini Batch Size
              </label>
              <input
                type="number"
                value={node.data.mini_batch_size || 1}
                onChange={(e) => handleChange('mini_batch_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Batch size for each optimization step
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.00001"
                value={node.data.learning_rate || 0.00001}
                onChange={(e) => handleChange('learning_rate', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Max Steps
              </label>
              <input
                type="number"
                value={node.data.max_steps || 1000}
                onChange={(e) => handleChange('max_steps', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Max New Tokens
              </label>
              <input
                type="number"
                value={node.data.max_new_tokens || 128}
                onChange={(e) => handleChange('max_new_tokens', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Maximum tokens to generate per response
              </p>
            </div>

            {/* PPO-specific parameters */}
            <div className="border-t border-slate-600 pt-4">
              <h4 className="text-sm font-semibold text-amber-400 mb-3">PPO Parameters</h4>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                PPO Epochs
              </label>
              <input
                type="number"
                value={node.data.ppo_epochs || 4}
                onChange={(e) => handleChange('ppo_epochs', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Clip Range
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.clip_range || 0.2}
                onChange={(e) => handleChange('clip_range', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Gamma (Discount Factor)
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.gamma || 0.99}
                onChange={(e) => handleChange('gamma', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                GAE Lambda
              </label>
              <input
                type="number"
                step="0.01"
                value={node.data.gae_lambda || 0.95}
                onChange={(e) => handleChange('gae_lambda', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Value Function Coefficient
              </label>
              <input
                type="number"
                step="0.1"
                value={node.data.vf_coef || 0.5}
                onChange={(e) => handleChange('vf_coef', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>

            {/* GRPO-specific parameters */}
            <div className="border-t border-slate-600 pt-4">
              <h4 className="text-sm font-semibold text-amber-400 mb-3">GRPO Parameters</h4>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Group Size
              </label>
              <input
                type="number"
                value={node.data.group_size || 4}
                onChange={(e) => handleChange('group_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
              <p className="text-xs text-slate-400 mt-1">
                Number of responses to generate per prompt
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                GRPO Temperature
              </label>
              <input
                type="number"
                step="0.1"
                value={node.data.grpo_temperature || 0.01}
                onChange={(e) => handleChange('grpo_temperature', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
          </div>
        );

      // Mamba2 nodes
      case 'ssmcore':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                State Size (d_state)
              </label>
              <input
                type="number"
                value={node.data.state_size || 64}
                onChange={(e) => handleChange('state_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                min="1"
              />
              <p className="text-xs text-slate-400 mt-1">
                Recurrent state memory dimension - larger = more memory capacity
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Common values: 16 (minimal), 32 (small), 64 (balanced), 128 (optimal)
              </p>
            </div>
            <div className="text-slate-300 text-sm">
              <p className="mb-2">SSM Core is the heart of the state space model, maintaining hidden state across the sequence.</p>
            </div>
          </div>
        );

      case 'temporalconv':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Convolution Kernel Size
              </label>
              <input
                type="number"
                value={node.data.conv_kernel_size || 4}
                onChange={(e) => handleChange('conv_kernel_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
                min="1"
              />
              <p className="text-xs text-slate-400 mt-1">
                Short-range pattern capture window size
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Common values: 3 (fast), 4 (balanced), 5 (better context), 7 (maximum context)
              </p>
            </div>
            <div className="text-slate-300 text-sm">
              <p className="mb-2">Temporal convolution captures local dependencies before the SSM processes long-range patterns.</p>
            </div>
          </div>
        );

      case 'gating':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Expansion Factor
              </label>
              <input
                type="number"
                value={node.data.expand_factor || 2}
                onChange={(e) => handleChange('expand_factor', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                min="1"
                step="0.5"
              />
              <p className="text-xs text-slate-400 mt-1">
                Channel expansion ratio for model expressiveness
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Common values: 1.5 (efficient), 2 (balanced), 2.5 (expressive), 3 (maximum)
              </p>
            </div>
            <div className="text-slate-300 text-sm">
              <p className="mb-2">Gating mechanism controls information flow, similar to GLU activation in transformers.</p>
            </div>
          </div>
        );

      case 'headprojection':
        return (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Head Dimension
              </label>
              <input
                type="number"
                value={node.data.headdim || 64}
                onChange={(e) => handleChange('headdim', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                min="1"
              />
              <p className="text-xs text-slate-400 mt-1">
                Head dimension (like attention head size)
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Common values: 32 (fast), 64 (balanced), 128 (expressive)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Number of Groups
              </label>
              <input
                type="number"
                value={node.data.ngroups || 1}
                onChange={(e) => handleChange('ngroups', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                min="1"
              />
              <p className="text-xs text-slate-400 mt-1">
                Grouping for computational efficiency
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Common values: 1 (no grouping), 4 (grouped), 8 (efficient)
              </p>
            </div>
            <div className="text-slate-300 text-sm">
              <p className="mb-2">Multi-head structure similar to attention, allowing parallel processing of different features.</p>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-slate-400 text-sm">
            No configuration available for this block type.
          </div>
        );
    }
  };

  return (
    <AnimatePresence>
      {node && (
        <motion.div
          initial={{ x: 400, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 400, opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="absolute top-0 right-0 w-96 h-full bg-slate-800 border-l border-slate-700 shadow-2xl overflow-y-auto z-10"
        >
          {/* Header */}
          <div className="sticky top-0 bg-slate-800 border-b border-slate-700 p-4 flex items-center justify-between">
            <div>
              <h2 className="text-white font-bold text-lg">Configure Block</h2>
              <p className="text-slate-400 text-sm">{node.data.label || node.type}</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>

          {/* Configuration Fields */}
          <div className="p-4">
            {renderConfigFields()}
          </div>

          {/* Footer Info */}
          <div className="sticky bottom-0 bg-slate-800 border-t border-slate-700 p-4">
            <div className="text-xs text-slate-400">
              💡 Changes are applied in real-time
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ConfigPanel;
