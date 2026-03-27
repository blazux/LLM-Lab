import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { TrainingConfig, SFTConfig, ProjectType } from '../../types'

// ─── Shared primitives ────────────────────────────────────────────────────────

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

function InputField({
  label,
  type = 'text',
  value,
  onChange,
  hint,
  min,
  max,
  step,
  placeholder,
}: {
  label: string
  type?: 'text' | 'number' | 'float'
  value: string | number
  onChange: (v: string | number) => void
  hint?: string
  min?: number
  max?: number
  step?: number
  placeholder?: string
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400 w-44 flex-none">{label}</label>
      <div className="flex-1">
        <input
          type="number"
          value={value}
          onChange={(e) => {
            if (type === 'float') onChange(parseFloat(e.target.value) || 0)
            else if (type === 'number') onChange(parseInt(e.target.value) || 0)
            else onChange(e.target.value)
          }}
          min={min}
          max={max}
          step={step ?? (type === 'float' ? 0.0001 : 1)}
          placeholder={placeholder}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm"
        />
        {hint && <p className="text-xs text-slate-600 mt-1">{hint}</p>}
      </div>
    </div>
  )
}

function SelectField({
  label,
  value,
  onChange,
  options,
  hint,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
  hint?: string
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400 w-44 flex-none">{label}</label>
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

function ToggleField({
  label,
  value,
  onChange,
}: {
  label: string
  value: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400 w-44 flex-none">{label}</label>
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
    </div>
  )
}

function TextField({
  label,
  value,
  onChange,
  placeholder,
  hint,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
  hint?: string
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400 w-44 flex-none">{label}</label>
      <div className="flex-1">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm placeholder-slate-600"
        />
        {hint && <p className="text-xs text-slate-600 mt-1">{hint}</p>}
      </div>
    </div>
  )
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

// ─── TrainingStep ─────────────────────────────────────────────────────────────

interface Props {
  projectType: ProjectType
  config: TrainingConfig
  onChange: (cfg: TrainingConfig) => void
}

export default function TrainingStep({ projectType, config, onChange }: Props) {
  const isSFT = projectType === 'sft' || projectType === 'rlhf'
  const sft = config as SFTConfig

  function set<K extends keyof TrainingConfig>(key: K, val: TrainingConfig[K]) {
    onChange({ ...config, [key]: val })
  }

  function setSFT<K extends keyof SFTConfig>(key: K, val: SFTConfig[K]) {
    onChange({ ...config, [key]: val } as SFTConfig)
  }

  return (
    <div className="space-y-6 max-w-2xl overflow-y-auto pr-2" style={{ maxHeight: 'calc(100vh - 200px)' }}>

      {/* SFT-specific: checkpoint & LoRA */}
      {isSFT && (
        <>
          <FormSection title="Checkpoint">
            <TextField
              label="policy_checkpoint"
              value={sft.policy_checkpoint || ''}
              onChange={(v) => setSFT('policy_checkpoint', v)}
              placeholder="/app/data/checkpoint_step_10000.pt"
              hint="Path to the pretrained model checkpoint"
            />
          </FormSection>

          <FormSection title="LoRA">
            <ToggleField label="use_lora" value={sft.use_lora || false} onChange={(v) => setSFT('use_lora', v)} />
            {sft.use_lora && (
              <>
                <SelectField
                  label="lora_preset"
                  value={sft.lora_preset || 'attention_only'}
                  onChange={(v) => setSFT('lora_preset', v as SFTConfig['lora_preset'])}
                  options={[
                    { value: 'minimal', label: 'Minimal' },
                    { value: 'attention_only', label: 'Attention Only' },
                    { value: 'ffn_only', label: 'FFN Only' },
                    { value: 'all', label: 'All Layers' },
                    { value: 'custom', label: 'Custom' },
                  ]}
                />
                <InputField label="lora_r" type="number" value={sft.lora_r || 8} onChange={(v) => setSFT('lora_r', v as number)} hint="LoRA rank" />
                <InputField label="lora_alpha" type="number" value={sft.lora_alpha || 16} onChange={(v) => setSFT('lora_alpha', v as number)} />
                <InputField label="lora_dropout" type="float" value={sft.lora_dropout || 0.05} onChange={(v) => setSFT('lora_dropout', v as number)} step={0.01} min={0} max={1} />
              </>
            )}
          </FormSection>
        </>
      )}

      {/* Optimizer */}
      <FormSection title="Optimizer">
        <SelectField
          label="optimizer"
          value={config.optimizer}
          onChange={(v) => set('optimizer', v as TrainingConfig['optimizer'])}
          options={[
            { value: 'adamw', label: 'AdamW' },
            { value: 'muon', label: 'Muon' },
            { value: 'lion', label: 'Lion' },
            { value: 'sophia', label: 'Sophia' },
          ]}
        />

        {/* lr vs learning_rate for SFT */}
        {isSFT ? (
          <InputField
            label="learning_rate"
            type="float"
            value={sft.learning_rate ?? 5e-6}
            onChange={(v) => setSFT('learning_rate', v as number)}
            step={1e-7}
            hint="SFT learning rate (typically much smaller)"
          />
        ) : (
          <InputField label="lr" type="float" value={config.lr} onChange={(v) => set('lr', v as number)} step={1e-5} />
        )}
        <InputField label="weight_decay" type="float" value={config.weight_decay} onChange={(v) => set('weight_decay', v as number)} step={0.01} />

        {isSFT ? (
          <InputField label="max_grad_norm" type="float" value={sft.max_grad_norm ?? 1.0} onChange={(v) => setSFT('max_grad_norm', v as number)} step={0.1} />
        ) : (
          <InputField label="grad_clip" type="float" value={config.grad_clip} onChange={(v) => set('grad_clip', v as number)} step={0.1} />
        )}

        {/* Optimizer-specific params */}
        {config.optimizer === 'adamw' && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="adamw_beta1" type="float" value={config.adamw_beta1} onChange={(v) => set('adamw_beta1', v as number)} step={0.001} />
            <InputField label="adamw_beta2" type="float" value={config.adamw_beta2} onChange={(v) => set('adamw_beta2', v as number)} step={0.001} />
            <InputField label="adamw_eps" type="float" value={config.adamw_eps} onChange={(v) => set('adamw_eps', v as number)} step={1e-9} />
          </div>
        )}
        {config.optimizer === 'muon' && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="muon_momentum" type="float" value={config.muon_momentum} onChange={(v) => set('muon_momentum', v as number)} step={0.01} />
            <ToggleField label="muon_nesterov" value={config.muon_nesterov} onChange={(v) => set('muon_nesterov', v)} />
          </div>
        )}
        {config.optimizer === 'lion' && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="lion_beta1" type="float" value={config.lion_beta1} onChange={(v) => set('lion_beta1', v as number)} step={0.01} />
            <InputField label="lion_beta2" type="float" value={config.lion_beta2} onChange={(v) => set('lion_beta2', v as number)} step={0.01} />
          </div>
        )}
        {config.optimizer === 'sophia' && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="sophia_beta1" type="float" value={config.sophia_beta1} onChange={(v) => set('sophia_beta1', v as number)} step={0.001} />
            <InputField label="sophia_beta2" type="float" value={config.sophia_beta2} onChange={(v) => set('sophia_beta2', v as number)} step={0.001} />
            <InputField label="sophia_rho" type="float" value={config.sophia_rho} onChange={(v) => set('sophia_rho', v as number)} step={0.001} />
          </div>
        )}
      </FormSection>

      {/* Scheduler */}
      <FormSection title="Scheduler">
        <SelectField
          label="scheduler"
          value={config.scheduler}
          onChange={(v) => set('scheduler', v as TrainingConfig['scheduler'])}
          options={[
            { value: 'cosine', label: 'Cosine Annealing' },
            { value: 'linear', label: 'Linear Decay' },
            { value: 'polynomial', label: 'Polynomial' },
            { value: 'constant', label: 'Constant' },
            { value: 'adaptive', label: 'Adaptive' },
          ]}
        />
        {config.scheduler === 'adaptive' && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="adaptive_window" type="number" value={config.adaptive_window} onChange={(v) => set('adaptive_window', v as number)} />
            <InputField label="adaptive_patience" type="number" value={config.adaptive_patience} onChange={(v) => set('adaptive_patience', v as number)} />
            <InputField label="increase_factor" type="float" value={config.adaptive_increase_factor} onChange={(v) => set('adaptive_increase_factor', v as number)} step={0.01} />
            <InputField label="decrease_factor" type="float" value={config.adaptive_decrease_factor} onChange={(v) => set('adaptive_decrease_factor', v as number)} step={0.01} />
            <InputField label="min_lr" type="float" value={config.adaptive_min_lr} onChange={(v) => set('adaptive_min_lr', v as number)} step={1e-7} />
            <InputField label="threshold" type="float" value={config.adaptive_threshold} onChange={(v) => set('adaptive_threshold', v as number)} step={0.001} />
          </div>
        )}
      </FormSection>

      {/* Loss */}
      <FormSection title="Loss Function">
        <SelectField
          label="loss_fn"
          value={config.loss_fn}
          onChange={(v) => set('loss_fn', v as TrainingConfig['loss_fn'])}
          options={[
            { value: 'cross_entropy', label: 'Cross Entropy' },
            { value: 'maxis', label: 'MAXIS' },
          ]}
        />
        {config.loss_fn === 'maxis' && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="n_candidates" type="number" value={config.maxis_n_candidates} onChange={(v) => set('maxis_n_candidates', v as number)} />
            <InputField label="low_rank_dim" type="number" value={config.maxis_low_rank_dim} onChange={(v) => set('maxis_low_rank_dim', v as number)} />
            <InputField label="chunk_size" type="number" value={config.maxis_chunk_size} onChange={(v) => set('maxis_chunk_size', v as number)} />
            <InputField label="aux_weight" type="float" value={config.maxis_aux_weight} onChange={(v) => set('maxis_aux_weight', v as number)} step={0.01} />
          </div>
        )}
      </FormSection>

      {/* Training loop */}
      <FormSection title="Training Loop">
        <InputField label="batch_size" type="number" value={config.batch_size} onChange={(v) => set('batch_size', v as number)} min={1} />
        <InputField label="grad_accum_steps" type="number" value={config.gradient_accumulation_steps} onChange={(v) => set('gradient_accumulation_steps', v as number)} hint="Effective batch = batch_size × accum_steps" />
        <InputField label="max_steps" type="number" value={config.max_steps} onChange={(v) => set('max_steps', v as number)} />
        <InputField label="warmup_steps" type="number" value={config.warmup_steps} onChange={(v) => set('warmup_steps', v as number)} />
        <InputField label="eval_every" type="number" value={config.eval_every} onChange={(v) => set('eval_every', v as number)} />
        <InputField label="eval_steps" type="number" value={config.eval_steps} onChange={(v) => set('eval_steps', v as number)} />
        {isSFT && (
          <>
            <InputField label="log_every" type="number" value={sft.log_every ?? 10} onChange={(v) => setSFT('log_every', v as number)} />
            <InputField label="save_every" type="number" value={sft.save_every ?? 500} onChange={(v) => setSFT('save_every', v as number)} />
          </>
        )}
        <AdvancedSection>
          <ToggleField label="save_best_only" value={config.save_best_only ?? true} onChange={(v) => set('save_best_only', v)} />
          {isSFT && (
            <>
              <InputField
                label="dropout_override"
                type="float"
                value={sft.dropout_override ?? ''}
                onChange={(v) => setSFT('dropout_override', v === 0 ? undefined : v as number)}
                step={0.01}
                min={0}
                max={1}
                placeholder="(use model default)"
                hint="Override model dropout during SFT"
              />
              <div>
                <label className="text-sm text-slate-400 w-44 inline-block mb-1">lora_target_modules</label>
                <input
                  type="text"
                  value={(sft.lora_target_modules ?? []).join(', ')}
                  onChange={(e) => {
                    const modules = e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                    setSFT('lora_target_modules', modules.length > 0 ? modules : undefined)
                  }}
                  placeholder="q_proj, v_proj (comma-separated, custom preset only)"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-accent text-sm placeholder-slate-600"
                />
                <p className="text-xs text-slate-600 mt-1">Only used when lora_preset is set to custom</p>
              </div>
            </>
          )}
        </AdvancedSection>
      </FormSection>
    </div>
  )
}
