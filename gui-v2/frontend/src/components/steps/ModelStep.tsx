import { useEffect, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { ModelConfig, TransformerConfig, Mamba2Config } from '../../types'
import { defaultTransformerConfig, defaultMamba2Config } from '../../types'
import ModelDiagram from '../ModelDiagram'

// ─── Shared form primitives ───────────────────────────────────────────────────

function FormSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{title}</span>
        <div className="flex-1 h-px bg-slate-800" />
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  )
}

interface InputFieldProps {
  label: string
  type?: 'text' | 'number' | 'float'
  value: string | number | boolean
  onChange: (val: string | number | boolean) => void
  hint?: string
  min?: number
  max?: number
  step?: number
  placeholder?: string
}

function InputField({ label, type = 'text', value, onChange, hint, min, max, step, placeholder }: InputFieldProps) {
  if (type === 'float') {
    return (
      <div className="flex items-center gap-3">
        <label className="text-sm text-slate-400 w-40 flex-none">{label}</label>
        <div className="flex-1">
          <input
            type="number"
            value={value as number}
            onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
            min={min}
            max={max}
            step={step ?? 0.0001}
            placeholder={placeholder}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm"
          />
          {hint && <p className="text-xs text-slate-600 mt-1">{hint}</p>}
        </div>
      </div>
    )
  }
  if (type === 'number') {
    return (
      <div className="flex items-center gap-3">
        <label className="text-sm text-slate-400 w-40 flex-none">{label}</label>
        <div className="flex-1">
          <input
            type="number"
            value={value as number}
            onChange={(e) => onChange(parseInt(e.target.value) || 0)}
            min={min}
            max={max}
            step={step ?? 1}
            placeholder={placeholder}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm"
          />
          {hint && <p className="text-xs text-slate-600 mt-1">{hint}</p>}
        </div>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400 w-40 flex-none">{label}</label>
      <div className="flex-1">
        <input
          type="text"
          value={value as string}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm placeholder-slate-600"
        />
        {hint && <p className="text-xs text-slate-600 mt-1">{hint}</p>}
      </div>
    </div>
  )
}

interface SelectFieldProps {
  label: string
  value: string
  onChange: (val: string) => void
  options: { value: string; label: string }[]
  hint?: string
}

function SelectField({ label, value, onChange, options, hint }: SelectFieldProps) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400 w-40 flex-none">{label}</label>
      <div className="flex-1">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm appearance-none"
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        {hint && <p className="text-xs text-slate-600 mt-1">{hint}</p>}
      </div>
    </div>
  )
}

interface ToggleFieldProps {
  label: string
  value: boolean
  onChange: (val: boolean) => void
  hint?: string
}

function ToggleField({ label, value, onChange, hint }: ToggleFieldProps) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400 w-40 flex-none">{label}</label>
      <div className="flex-1 flex items-center gap-2">
        <button
          type="button"
          onClick={() => onChange(!value)}
          className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none ${
            value ? 'bg-accent' : 'bg-slate-700'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
              value ? 'translate-x-4' : 'translate-x-0'
            }`}
          />
        </button>
        {hint && <p className="text-xs text-slate-600">{hint}</p>}
      </div>
    </div>
  )
}

interface OptionalNumberProps {
  label: string
  value: number | ''
  onChange: (val: number | '') => void
  hint?: string
  placeholder?: string
  step?: number
}

function AdvancedSection({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border border-slate-800 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-slate-500 hover:text-slate-400 hover:bg-slate-800/30 transition-colors"
      >
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <span className="uppercase tracking-wider font-semibold">Advanced</span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 space-y-3 border-t border-slate-800">
          {children}
        </div>
      )}
    </div>
  )
}

function OptionalNumberField({ label, value, onChange, hint, placeholder, step }: OptionalNumberProps) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400 w-40 flex-none">{label}</label>
      <div className="flex-1">
        <input
          type="number"
          value={value === '' ? '' : value}
          onChange={(e) => {
            const v = e.target.value
            onChange(v === '' ? '' : (parseInt(v) || ''))
          }}
          step={step ?? 1}
          placeholder={placeholder ?? 'auto'}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm placeholder-slate-600"
        />
        {hint && <p className="text-xs text-slate-600 mt-1">{hint}</p>}
      </div>
    </div>
  )
}

// ─── Transformer Form ─────────────────────────────────────────────────────────

function TransformerForm({ cfg, onChange }: { cfg: TransformerConfig; onChange: (c: TransformerConfig) => void }) {
  function set<K extends keyof TransformerConfig>(key: K, val: TransformerConfig[K]) {
    onChange({ ...cfg, [key]: val })
  }

  const showKvHeads = cfg.attention_type === 'gqa' || cfg.attention_type === 'mqa'
  const showMLA = cfg.attention_type === 'mla'

  return (
    <div className="space-y-6">
      <FormSection title="Core dimensions">
        <InputField label="d_model" type="number" value={cfg.d_model} onChange={(v) => set('d_model', v as number)} hint="Model dimension" />
        <InputField label="n_layers" type="number" value={cfg.n_layers} onChange={(v) => set('n_layers', v as number)} />
        <InputField label="vocab_size" type="number" value={cfg.vocab_size} onChange={(v) => set('vocab_size', v as number)} />
        <InputField label="max_seq_len" type="number" value={cfg.max_seq_len} onChange={(v) => set('max_seq_len', v as number)} />
        <InputField label="dropout" type="float" value={cfg.dropout} onChange={(v) => set('dropout', v as number)} step={0.01} min={0} max={1} />
      </FormSection>

      <FormSection title="Normalization">
        <SelectField
          label="norm_type"
          value={cfg.norm_type}
          onChange={(v) => set('norm_type', v as TransformerConfig['norm_type'])}
          options={[{ value: 'rmsnorm', label: 'RMSNorm' }, { value: 'layernorm', label: 'LayerNorm' }]}
        />
        <InputField label="norm_eps" type="float" value={cfg.norm_eps} onChange={(v) => set('norm_eps', v as number)} step={1e-8} />
      </FormSection>

      <FormSection title="Attention">
        <SelectField
          label="attention_type"
          value={cfg.attention_type}
          onChange={(v) => set('attention_type', v as TransformerConfig['attention_type'])}
          options={[
            { value: 'mha', label: 'MHA — Multi-Head Attention' },
            { value: 'mqa', label: 'MQA — Multi-Query Attention' },
            { value: 'gqa', label: 'GQA — Grouped-Query Attention' },
            { value: 'mla', label: 'MLA — Multi-Latent Attention' },
          ]}
        />
        <InputField label="n_heads" type="number" value={cfg.n_heads} onChange={(v) => set('n_heads', v as number)} />
        {showKvHeads && (
          <InputField label="n_kv_heads" type="number" value={cfg.n_kv_heads} onChange={(v) => set('n_kv_heads', v as number)} hint="KV heads (GQA/MQA)" />
        )}
        {showMLA && (
          <>
            <OptionalNumberField label="d_latent" value={cfg.d_latent} onChange={(v) => set('d_latent', v)} hint="Latent dimension for MLA" />
            <OptionalNumberField label="d_rope_latent" value={cfg.d_rope_latent} onChange={(v) => set('d_rope_latent', v)} hint="RoPE latent dim for MLA" />
          </>
        )}
        <ToggleField label="attention_bias" value={cfg.attention_bias} onChange={(v) => set('attention_bias', v)} />
        <OptionalNumberField label="sliding_window" value={cfg.sliding_window} onChange={(v) => set('sliding_window', v)} hint="Optional sliding window size" />
      </FormSection>

      <FormSection title="Positional Encoding">
        <SelectField
          label="positional_encoding"
          value={cfg.positional_encoding}
          onChange={(v) => set('positional_encoding', v as TransformerConfig['positional_encoding'])}
          options={[
            { value: 'rope', label: 'RoPE (implicit, recommended)' },
            { value: 'sinusoidal', label: 'Sinusoidal' },
            { value: 'alibi', label: 'ALiBi (implicit)' },
            { value: 'yarn', label: 'YaRN' },
          ]}
        />
      </FormSection>

      <FormSection title="Feed-Forward Network">
        <SelectField
          label="activation"
          value={cfg.activation}
          onChange={(v) => set('activation', v as TransformerConfig['activation'])}
          options={[
            { value: 'swiglu', label: 'SwiGLU (recommended)' },
            { value: 'geglu', label: 'GeGLU' },
            { value: 'reglu', label: 'ReGLU' },
            { value: 'gelu', label: 'GELU' },
            { value: 'relu', label: 'ReLU' },
            { value: 'silu', label: 'SiLU' },
          ]}
        />
        <OptionalNumberField label="d_ff" value={cfg.d_ff} onChange={(v) => set('d_ff', v)} hint="FFN dimension (blank = auto ~2.67×d_model)" />
      </FormSection>

      <FormSection title="Mixture of Experts">
        <ToggleField label="use_moe" value={cfg.use_moe} onChange={(v) => set('use_moe', v)} />
        {cfg.use_moe && (
          <>
            <InputField label="num_experts" type="number" value={cfg.num_experts} onChange={(v) => set('num_experts', v as number)} />
            <InputField label="experts_per_token" type="number" value={cfg.num_experts_per_token} onChange={(v) => set('num_experts_per_token', v as number)} hint="Top-K routing" />
            <AdvancedSection>
              <InputField label="lb_loss_weight" type="float" value={cfg.load_balancing_loss_weight} onChange={(v) => set('load_balancing_loss_weight', v as number)} step={0.001} hint="Load balancing aux loss weight" />
              <InputField label="router_z_loss_wt" type="float" value={cfg.router_z_loss_weight} onChange={(v) => set('router_z_loss_weight', v as number)} step={0.001} hint="Router z-loss weight" />
            </AdvancedSection>
          </>
        )}
      </FormSection>

      <AdvancedSection>
        <ToggleField label="tie_word_embeddings" value={cfg.tie_word_embeddings} onChange={(v) => set('tie_word_embeddings', v)} hint="Share input/output embedding weights" />
      </AdvancedSection>
    </div>
  )
}

// ─── Mamba2 Form ──────────────────────────────────────────────────────────────

function Mamba2Form({ cfg, onChange }: { cfg: Mamba2Config; onChange: (c: Mamba2Config) => void }) {
  function set<K extends keyof Mamba2Config>(key: K, val: Mamba2Config[K]) {
    onChange({ ...cfg, [key]: val })
  }

  return (
    <div className="space-y-6">
      <FormSection title="Core dimensions">
        <InputField label="d_model" type="number" value={cfg.d_model} onChange={(v) => set('d_model', v as number)} />
        <InputField label="n_layers" type="number" value={cfg.n_layers} onChange={(v) => set('n_layers', v as number)} />
        <InputField label="vocab_size" type="number" value={cfg.vocab_size} onChange={(v) => set('vocab_size', v as number)} />
        <InputField label="max_seq_len" type="number" value={cfg.max_seq_len} onChange={(v) => set('max_seq_len', v as number)} />
        <InputField label="dropout" type="float" value={cfg.dropout} onChange={(v) => set('dropout', v as number)} step={0.01} min={0} max={1} />
      </FormSection>

      <FormSection title="Normalization">
        <SelectField
          label="norm_type"
          value={cfg.norm_type}
          onChange={(v) => set('norm_type', v as Mamba2Config['norm_type'])}
          options={[{ value: 'rmsnorm', label: 'RMSNorm' }, { value: 'layernorm', label: 'LayerNorm' }]}
        />
        <InputField label="norm_eps" type="float" value={cfg.norm_eps} onChange={(v) => set('norm_eps', v as number)} step={1e-8} />
      </FormSection>

      <FormSection title="SSM Parameters">
        <InputField label="state_size" type="number" value={cfg.state_size} onChange={(v) => set('state_size', v as number)} hint="SSM state dimension" />
        <InputField label="expand_factor" type="number" value={cfg.expand_factor} onChange={(v) => set('expand_factor', v as number)} />
        <InputField label="headdim" type="number" value={cfg.headdim} onChange={(v) => set('headdim', v as number)} hint="64 or 128" />
        <AdvancedSection>
          <InputField label="conv_kernel_size" type="number" value={cfg.conv_kernel_size} onChange={(v) => set('conv_kernel_size', v as number)} />
          <InputField label="ngroups" type="number" value={cfg.ngroups} onChange={(v) => set('ngroups', v as number)} hint="1 = no grouping, 8 = efficient" />
          <InputField label="chunk_size" type="number" value={cfg.chunk_size} onChange={(v) => set('chunk_size', v as number)} />
          <ToggleField label="tie_word_embeddings" value={cfg.tie_word_embeddings} onChange={(v) => set('tie_word_embeddings', v)} hint="Share input/output embedding weights" />
        </AdvancedSection>
      </FormSection>
    </div>
  )
}

// ─── ModelStep ────────────────────────────────────────────────────────────────

interface Props {
  config: ModelConfig
  onChange: (config: ModelConfig) => void
  readOnly?: boolean
}

export default function ModelStep({ config, onChange, readOnly }: Props) {
  // Switch architecture
  function handleArchChange(arch: 'transformer' | 'mamba2') {
    if (arch === config.architecture) return
    if (arch === 'transformer') onChange(defaultTransformerConfig())
    else onChange(defaultMamba2Config())
  }

  if (readOnly) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-slate-500">Model architecture is inherited from the pretrained checkpoint.</p>
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 space-y-2">
          <div className="flex gap-3">
            <span className="text-slate-500 text-sm w-32">Architecture</span>
            <span className="text-slate-200 text-sm font-medium capitalize">{config.architecture}</span>
          </div>
          <div className="flex gap-3">
            <span className="text-slate-500 text-sm w-32">d_model</span>
            <span className="text-slate-200 text-sm">{config.d_model}</span>
          </div>
          <div className="flex gap-3">
            <span className="text-slate-500 text-sm w-32">n_layers</span>
            <span className="text-slate-200 text-sm">{config.n_layers}</span>
          </div>
          <div className="flex gap-3">
            <span className="text-slate-500 text-sm w-32">vocab_size</span>
            <span className="text-slate-200 text-sm">{config.vocab_size.toLocaleString()}</span>
          </div>
        </div>
        <div className="h-[400px] rounded-xl overflow-hidden border border-slate-800">
          <ModelDiagram config={config} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-6 h-full min-h-0">
      {/* Left: Form */}
      <div className="w-[420px] flex-none overflow-y-auto pr-2">
        {/* Architecture selector */}
        <div className="mb-6">
          <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-2">Architecture</p>
          <div className="flex gap-2">
            {(['transformer', 'mamba2'] as const).map((a) => (
              <button
                key={a}
                onClick={() => handleArchChange(a)}
                className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-all ${
                  config.architecture === a
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-slate-700 text-slate-400 hover:border-slate-600'
                }`}
              >
                {a === 'transformer' ? 'Transformer' : 'Mamba2'}
              </button>
            ))}
          </div>
        </div>

        {config.architecture === 'transformer' ? (
          <TransformerForm cfg={config} onChange={onChange} />
        ) : (
          <Mamba2Form cfg={config as Mamba2Config} onChange={onChange} />
        )}
      </div>

      {/* Right: Live diagram */}
      <div className="flex-1 min-w-0 rounded-xl overflow-hidden border border-slate-800 bg-slate-900">
        <div className="px-4 py-2.5 border-b border-slate-800 flex items-center gap-2">
          <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">Architecture Preview</span>
          <span className="text-xs text-slate-600">— updates live</span>
        </div>
        <div style={{ height: 'calc(100% - 40px)' }}>
          <ModelDiagram config={config} />
        </div>
      </div>
    </div>
  )
}
