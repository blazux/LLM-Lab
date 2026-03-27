import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { ModelConfig } from '../../types'

interface Props {
  config: ModelConfig
  onChange: (config: ModelConfig) => void
}

const KNOWN_TOKENIZERS = [
  { value: 'Qwen/Qwen2.5-0.5B', label: 'Qwen2.5 (vocab 151,936) — recommended' },
  { value: 'EleutherAI/gpt-neox-20b', label: 'GPT-NeoX (vocab 50,257)' },
  { value: 'meta-llama/Llama-2-7b-hf', label: 'LLaMA 2 (vocab 32,000)' },
  { value: 'mistralai/Mistral-7B-v0.1', label: 'Mistral (vocab 32,000)' },
  { value: 'custom', label: 'Custom HuggingFace path…' },
]

export default function TokenizerStep({ config, onChange }: Props) {
  const [showInfo, setShowInfo] = useState(false)

  const isCustom = !KNOWN_TOKENIZERS.slice(0, -1).some((t) => t.value === config.tokenizer_name)
  const selectValue = isCustom ? 'custom' : config.tokenizer_name

  function handlePreset(val: string) {
    if (val === 'custom') return
    onChange({ ...config, tokenizer_name: val })
  }

  function handleCustom(val: string) {
    onChange({ ...config, tokenizer_name: val })
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h3 className="text-base font-semibold text-slate-200 mb-1">Tokenizer</h3>
        <p className="text-sm text-slate-500">
          Choose the tokenizer used to encode your training data.
          The <span className="text-slate-300 font-medium">vocab_size</span> in your model config must match the tokenizer's vocabulary.
        </p>
      </div>

      {/* Preset selector */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <label className="text-sm text-slate-400 w-36 flex-none">Tokenizer preset</label>
          <select
            value={selectValue}
            onChange={(e) => handlePreset(e.target.value)}
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm appearance-none"
          >
            {KNOWN_TOKENIZERS.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
        </div>

        {/* Custom path input — shown when preset is "custom" or an unrecognised value */}
        <div className="flex items-center gap-3">
          <label className="text-sm text-slate-400 w-36 flex-none">
            {isCustom ? 'Custom path' : 'tokenizer_name'}
          </label>
          <input
            type="text"
            value={config.tokenizer_name}
            onChange={(e) => handleCustom(e.target.value)}
            placeholder="organisation/model-name"
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm placeholder-slate-600 font-mono"
          />
        </div>
      </div>

      {/* Vocab size match indicator */}
      <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 flex items-center gap-4">
        <div className="w-8 h-8 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center text-accent text-xs font-bold flex-none">T</div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-slate-200">Current vocab_size</div>
          <div className="text-xs text-slate-500 mt-0.5 truncate font-mono">{config.tokenizer_name}</div>
        </div>
        <div className="text-lg font-semibold text-accent tabular-nums">{config.vocab_size.toLocaleString()}</div>
      </div>

      {/* Warning if vocab_size might not match */}
      {config.tokenizer_name === 'EleutherAI/gpt-neox-20b' && config.vocab_size !== 50257 && (
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
          <p className="text-xs text-amber-400">
            GPT-NeoX tokenizer has vocab size 50,257 but your model is set to {config.vocab_size.toLocaleString()}.
            Update <span className="font-medium">vocab_size</span> in the Model step to match.
          </p>
        </div>
      )}
      {(config.tokenizer_name === 'meta-llama/Llama-2-7b-hf' || config.tokenizer_name === 'mistralai/Mistral-7B-v0.1') && config.vocab_size !== 32000 && (
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
          <p className="text-xs text-amber-400">
            This tokenizer has vocab size 32,000 but your model is set to {config.vocab_size.toLocaleString()}.
            Update <span className="font-medium">vocab_size</span> in the Model step to match.
          </p>
        </div>
      )}

      {/* Collapsible info */}
      <div className="border border-slate-800 rounded-xl overflow-hidden">
        <button
          className="w-full flex items-center gap-2 px-4 py-3 text-left text-xs text-slate-500 hover:text-slate-400 hover:bg-slate-800/30 transition-colors"
          onClick={() => setShowInfo(!showInfo)}
        >
          {showInfo ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          How tokenizer loading works
        </button>
        {showInfo && (
          <div className="px-4 pb-4 text-xs text-slate-500 leading-relaxed space-y-2 border-t border-slate-800">
            <p>
              During <strong className="text-slate-400">pretraining</strong>, the tokenizer is downloaded from HuggingFace Hub using the name above and cached locally.
              Only the tokenizer files are used — the model weights are ignored.
            </p>
            <p>
              During <strong className="text-slate-400">SFT</strong>, the tokenizer is loaded from the policy checkpoint directory, so this field is not used.
            </p>
            <p>
              The tokenizer name is stored in the model config and exported alongside the weights so the model can be loaded elsewhere.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
