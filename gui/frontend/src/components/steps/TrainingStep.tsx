import { useId, useState } from 'react'
import { ChevronDown, ChevronRight, MemoryStick } from 'lucide-react'
import type { TrainingConfig, SFTConfig, RLHFConfig, ProjectType, ModelConfig, TransformerConfig } from '../../types'
import { estimateTrainMemory, fmtParams } from '../../utils/estimate'

// ─── Shared primitives ────────────────────────────────────────────────────────

function FormSection({ title, learn, children }: { title: string; learn?: string; children: React.ReactNode }) {
  const [showLearn, setShowLearn] = useState(false)
  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{title}</span>
        <div className="flex-1 h-px bg-slate-800" />
        {learn && (
          <button
            type="button"
            onClick={() => setShowLearn(!showLearn)}
            className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${
              showLearn
                ? 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10'
                : 'text-slate-500 border-slate-700 hover:text-slate-300'
            }`}
          >
            {showLearn ? 'hide' : 'what is this?'}
          </button>
        )}
      </div>
      {learn && showLearn && (
        <p className="text-xs text-slate-400 leading-relaxed bg-slate-800/40 border border-slate-700/60 rounded-lg px-3 py-2 mb-3">
          {learn}
        </p>
      )}
      <div className="space-y-3">{children}</div>
    </div>
  )
}

const SECTION_LEARN = {
  optimizer:
    'The optimizer decides how weights move given their gradients. AdamW keeps a running mean and variance of each gradient and is the safe default. Muon orthogonalizes updates for the 2D matrices — the trick behind the nanoGPT speedrun records, often faster per step at small scale. Lion is a lighter sign-based variant; Sophia (simplified here) scales by a curvature estimate. When in doubt: AdamW, lr 3e-4, weight_decay 0.1.',
  scheduler:
    'The learning rate is the most sensitive hyperparameter, and good runs vary it over time: a warmup ramp (protects the randomly-initialized model from huge early updates), then a decay. Cosine is the classic shape. WSD (warmup-stable-decay) holds the LR constant on a long plateau and only decays in the final stretch — its superpower is that you can extend a run or branch from any plateau checkpoint without replanning the whole schedule.',
  loss:
    'Cross-entropy is THE language-modeling loss: maximize the probability of the true next token; language perplexity is exp(token-averaged cross-entropy), excluding auxiliary objectives and z-loss — “how many tokens is the model hesitating between”. The optional z-loss keeps logit magnitudes bounded on long runs. MAXIS is this lab’s sampled-softmax experiment: it approximates the full softmax with a subset of negative candidates, saving memory on very large vocabularies.',
  loop:
    'The effective batch = batch_size × grad_accum_steps: gradients from several small forward passes are accumulated before one weight update, simulating a large batch within your VRAM budget. For pretraining, max_steps counts weight updates: tokens seen ≈ max_steps × effective batch × (seq_len − 1). doc-boundary masks (advanced) stop attention from leaking between the different documents packed into one sequence.',
  algorithm:
    'Pretraining and SFT teach the model to imitate text; RLHF optimizes it directly for what you actually want. PPO is the classic recipe: the policy generates responses, a separate reward model scores them, and a value network stabilizes the updates — powerful but heavy (3 models in memory). DPO skips reinforcement learning entirely: it trains directly on chosen/rejected preference pairs against a frozen reference — the simplest and cheapest option, and the best starting point. GRPO removes the value network: it generates a group of responses per prompt and uses the group mean reward as the baseline — the algorithm behind DeepSeek-R1. The sequence-level importance option is GSPO (Qwen3); clip-higher and dynamic sampling come from DAPO.',
  generation:
    'PPO and GRPO create their own training data: at every step the policy generates responses to dataset prompts, which are then scored by the reward model. These knobs control that sampling. Temperature/top-k/top-p tune exploration — too greedy and the model never stumbles on better answers to be rewarded for; too random and the reward signal drowns in noise. max_new_tokens bounds rollout length and directly drives step time and VRAM.',
} as const

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
  const fieldId = useId()
  return (
    <div className="flex items-center gap-3">
      <label htmlFor={fieldId} className="text-sm text-slate-400 w-44 flex-none">{label}</label>
      <div className="flex-1">
        <input
          id={fieldId}
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
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-hidden focus:border-accent text-sm"
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
  const fieldId = useId()
  return (
    <div className="flex items-center gap-3">
      <label htmlFor={fieldId} className="text-sm text-slate-400 w-44 flex-none">{label}</label>
      <div className="flex-1">
        <select
          id={fieldId}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-hidden focus:border-accent text-sm appearance-none"
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
  const fieldId = useId()
  return (
    <div className="flex items-center gap-3">
      <label htmlFor={fieldId} className="text-sm text-slate-400 w-44 flex-none">{label}</label>
      <button
          id={fieldId}
        role="switch" aria-checked={value}
        type="button"
        onClick={() => onChange(!value)}
        className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-hidden ${
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
  const fieldId = useId()
  return (
    <div className="flex items-center gap-3">
      <label htmlFor={fieldId} className="text-sm text-slate-400 w-44 flex-none">{label}</label>
      <div className="flex-1">
        <input
          id={fieldId}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-hidden focus:border-accent text-sm placeholder-slate-600"
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

// ─── Memory estimate panel ────────────────────────────────────────────────────

function MemoryPanel({ model, config, projectType }: {
  model: ModelConfig
  config: TrainingConfig
  projectType: ProjectType
}) {
  const isPretrain = projectType === 'pretrain'
  const sft = config as SFTConfig
  const rlhf = config as RLHFConfig
  const cfg = model as TransformerConfig

  const est = estimateTrainMemory(cfg, {
    masterWeights: isPretrain,
    batchSize: config.batch_size,
    // the pretrain loop always checkpoints activations; SFT/RLHF do not
    gradientCheckpointing: isPretrain,
    optimizer: config.optimizer,
    useLora: !isPretrain && (sft.use_lora ?? false),
    loraR: sft.lora_r ?? 8,
  })

  const rows: [string, number][] = [
    [isPretrain ? 'Model weights (fp32)' : 'Model weights (bf16)', est.weightsGB],
    ['Gradients', est.gradsGB],
    ['Optimizer states (fp32)', est.optimizerGB],
    ['Activations', est.activationsGB],
    ['Logits + loss (vocab)', est.logitsGB],
    ['CUDA overhead', est.overheadGB],
  ]
  const maxRow = Math.max(...rows.map(([, v]) => v), 1e-9)

  // Token budget (pretrain only — max_steps counts optimizer updates)
  const tokensPerStep = config.batch_size * config.gradient_accumulation_steps * (cfg.max_seq_len - 1)
  const totalTokens = config.max_steps * tokensPerStep
  const ratio = totalTokens / Math.max(est.params, 1)

  const tips: string[] = []
  if (est.logitsGB / est.totalGB > 0.35) {
    tips.push(
      `The ${fmtParams(cfg.vocab_size)}-entry vocabulary dominates memory (logits tensor). A 32k tokenizer (LLaMA/Mistral) or the MAXIS loss would cut this sharply.`
    )
  }
  if (est.activationsGB / est.totalGB > 0.35 && config.batch_size > 1) {
    tips.push(
      'Activations dominate: lower batch_size and raise grad_accum_steps — same effective batch, less VRAM.'
    )
  }
  if (isPretrain && ratio < 20) {
    tips.push(
      ratio < 5
        ? `Only ~${ratio.toFixed(1)} tokens per parameter — far below the ~20 of compute-optimal training. Raise max_steps or shrink the model.`
        : `~${ratio.toFixed(0)} tokens per parameter — usable, but ~20+ is the compute-optimal reference (Chinchilla).`
    )
  }

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-1">
        <MemoryStick size={14} className="text-accent" />
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Estimated GPU memory</span>
        <span className="ml-auto text-base font-bold text-slate-100">~{est.totalGB.toFixed(1)} GB</span>
      </div>
      <p className="text-[11px] text-slate-600 mb-3">
        {fmtParams(est.params)} params
        {est.trainableParams !== est.params && ` (${fmtParams(est.trainableParams)} trainable with LoRA)`}
        {' '}· batch {config.batch_size} × seq {cfg.max_seq_len} · indicative ±20%
        {!isPretrain && ' · based on the Model step config'}
      </p>
      <div className="space-y-1.5">
        {rows.map(([label, gb]) => (
          <div key={label} className="flex items-center gap-2 text-[11px]">
            <span className="w-40 flex-none text-slate-500">{label}</span>
            <div className="flex-1 h-1.5 bg-slate-900 rounded-full overflow-hidden">
              <div className="h-full bg-accent/60 rounded-full" style={{ width: `${(gb / maxRow) * 100}%` }} />
            </div>
            <span className="w-14 flex-none text-right text-slate-400">{gb >= 0.1 ? gb.toFixed(1) : gb.toFixed(2)} GB</span>
          </div>
        ))}
      </div>

      {isPretrain && (
        <div className="mt-3 pt-3 border-t border-slate-700/60 grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-[10px] text-slate-600 uppercase tracking-wider">Tokens / step</div>
            <div className="text-xs font-semibold text-slate-300">{fmtParams(tokensPerStep)}</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-600 uppercase tracking-wider">Total tokens</div>
            <div className="text-xs font-semibold text-slate-300">{fmtParams(totalTokens)}</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-600 uppercase tracking-wider">Tokens / param</div>
            <div className={`text-xs font-semibold ${ratio >= 20 ? 'text-emerald-400' : ratio >= 5 ? 'text-amber-400' : 'text-red-400'}`}>
              {ratio >= 100 ? Math.round(ratio) : ratio.toFixed(1)}
            </div>
          </div>
        </div>
      )}

      {projectType === 'rlhf' && (
        <p className="mt-3 pt-3 border-t border-slate-700/60 text-[11px] text-slate-500 leading-relaxed">
          {rlhf.algorithm === 'ppo' &&
            'PPO adds a frozen reference copy, a value head and the reward model on top of this — budget roughly 2.5-3× the policy alone.'}
          {rlhf.algorithm === 'dpo' &&
            'DPO adds a frozen bf16 reference copy of the model — budget roughly +' }
          {rlhf.algorithm === 'dpo' && `${est.weightsGB.toFixed(1)} GB on top of this.`}
          {(rlhf.algorithm === 'grpo' || !rlhf.algorithm) &&
            `GRPO generates group_size (${rlhf.group_size ?? 4}) responses per prompt plus reward-model scoring — budget roughly 1.5-2× this estimate.`}
        </p>
      )}

      {tips.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-700/60 space-y-1.5">
          {tips.map((t, i) => (
            <p key={i} className="text-[11px] text-slate-400 leading-relaxed flex gap-1.5">
              <span className="text-amber-400 flex-none">💡</span>
              <span>{t}</span>
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── TrainingStep ─────────────────────────────────────────────────────────────

interface Props {
  projectType: ProjectType
  modelConfig: ModelConfig
  config: TrainingConfig
  onChange: (cfg: TrainingConfig) => void
}

export default function TrainingStep({ projectType, modelConfig, config, onChange }: Props) {
  const isSFT = projectType === 'sft' || projectType === 'rlhf'
  const isRLHF = projectType === 'rlhf'
  const sft = config as SFTConfig
  const rlhf = config as RLHFConfig

  function set<K extends keyof TrainingConfig>(key: K, val: TrainingConfig[K]) {
    onChange({ ...config, [key]: val })
  }

  function setSFT<K extends keyof SFTConfig>(key: K, val: SFTConfig[K]) {
    onChange({ ...config, [key]: val } as SFTConfig)
  }

  function setRLHF<K extends keyof RLHFConfig>(key: K, val: RLHFConfig[K]) {
    onChange({ ...config, [key]: val } as RLHFConfig)
  }

  const algo = rlhf.algorithm ?? 'grpo'

  return (
    <div className="space-y-6 max-w-2xl overflow-y-auto pr-2" style={{ maxHeight: 'calc(100vh - 200px)' }}>

      {/* Live memory estimate — reacts to batch_size / optimizer / LoRA */}
      <MemoryPanel model={modelConfig} config={config} projectType={projectType} />

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

          {isRLHF && (
            <>
              <FormSection title="Algorithm" learn={SECTION_LEARN.algorithm}>
                <SelectField
                  label="algorithm"
                  value={algo}
                  onChange={(v) => setRLHF('algorithm', v as RLHFConfig['algorithm'])}
                  options={[
                    { value: 'grpo', label: 'GRPO — group-relative (DeepSeek-R1)' },
                    { value: 'dpo', label: 'DPO — direct preference (simplest)' },
                    { value: 'ppo', label: 'PPO — classic RLHF' },
                  ]}
                  hint="DPO needs chosen/rejected pairs; PPO and GRPO need prompts + a reward model"
                />
                {(algo === 'ppo' || algo === 'grpo') && (
                  <TextField
                    label="reward_model_name"
                    value={rlhf.reward_model_name ?? ''}
                    onChange={(v) => setRLHF('reward_model_name', v)}
                    placeholder="OpenAssistant/reward-model-deberta-v3-large-v2"
                    hint="HuggingFace model that scores generated responses"
                  />
                )}
                {algo === 'ppo' && (
                  <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
                    <InputField label="ppo_epochs" type="number" value={rlhf.ppo_epochs ?? 4} onChange={(v) => setRLHF('ppo_epochs', v as number)} hint="Optimization passes over each rollout batch" />
                    <InputField label="clip_range" type="float" value={rlhf.clip_range ?? 0.2} onChange={(v) => setRLHF('clip_range', v as number)} step={0.01} hint="PPO ratio clip — keeps the policy close to its rollout version" />
                    <InputField label="vf_coef" type="float" value={rlhf.vf_coef ?? 0.1} onChange={(v) => setRLHF('vf_coef', v as number)} step={0.01} hint="Value-network loss weight" />
                    <InputField label="gamma" type="float" value={rlhf.gamma ?? 1.0} onChange={(v) => setRLHF('gamma', v as number)} step={0.01} hint="Discount factor. 1.0 for text (no time preference)" />
                    <InputField label="gae_lambda" type="float" value={rlhf.gae_lambda ?? 0.95} onChange={(v) => setRLHF('gae_lambda', v as number)} step={0.01} hint="GAE advantage smoothing" />
                  </div>
                )}
                {algo === 'dpo' && (
                  <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
                    <InputField label="beta" type="float" value={rlhf.clip_range ?? 0.2} onChange={(v) => setRLHF('clip_range', v as number)} step={0.01} hint="DPO temperature: how hard preferences push away from the reference. 0.1-0.5 typical" />
                    <TextField
                      label="reference_checkpoint"
                      value={rlhf.reference_checkpoint ?? ''}
                      onChange={(v) => setRLHF('reference_checkpoint', v)}
                      placeholder="(empty = frozen copy of policy_checkpoint)"
                      hint="Frozen model DPO regularizes against"
                    />
                  </div>
                )}
                {algo === 'grpo' && (
                  <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
                    <InputField label="group_size" type="number" value={rlhf.group_size ?? 4} onChange={(v) => setRLHF('group_size', v as number)} min={2} hint="Responses generated per prompt — the group mean reward is the baseline" />
                    <SelectField
                      label="importance_level"
                      value={rlhf.grpo_importance_level ?? 'token'}
                      onChange={(v) => setRLHF('grpo_importance_level', v as RLHFConfig['grpo_importance_level'])}
                      options={[
                        { value: 'token', label: 'token — classic GRPO' },
                        { value: 'sequence', label: 'sequence — GSPO (Qwen3)' },
                      ]}
                    />
                    <InputField
                      label="clip_higher"
                      type="float"
                      value={rlhf.grpo_clip_higher ?? ''}
                      onChange={(v) => setRLHF('grpo_clip_higher', v === 0 ? '' : (v as number))}
                      step={0.01}
                      placeholder="(empty = unclipped legacy GRPO)"
                      hint="DAPO asymmetric upper clip, e.g. 0.28 — lets good tokens grow faster than bad ones shrink"
                    />
                    <ToggleField label="dynamic_sampling" value={rlhf.grpo_dynamic_sampling ?? false} onChange={(v) => setRLHF('grpo_dynamic_sampling', v)} />
                    <InputField label="grpo_temperature" type="float" value={rlhf.grpo_temperature ?? 1.0} onChange={(v) => setRLHF('grpo_temperature', v as number)} step={0.05} hint="Sampling temperature inside each group" />
                  </div>
                )}
              </FormSection>

              {(algo === 'ppo' || algo === 'grpo') && (
                <FormSection title="Generation" learn={SECTION_LEARN.generation}>
                  <InputField label="max_new_tokens" type="number" value={rlhf.max_new_tokens ?? 128} onChange={(v) => setRLHF('max_new_tokens', v as number)} min={1} hint="Rollout length — drives step time and VRAM" />
                  <InputField label="temperature" type="float" value={rlhf.temperature ?? 1.0} onChange={(v) => setRLHF('temperature', v as number)} step={0.05} />
                  <InputField label="top_k" type="number" value={rlhf.top_k ?? 0} onChange={(v) => setRLHF('top_k', v as number)} min={0} hint="0 = disabled" />
                  <InputField label="top_p" type="float" value={rlhf.top_p ?? 1.0} onChange={(v) => setRLHF('top_p', v as number)} step={0.05} min={0} max={1} hint="1.0 = disabled" />
                </FormSection>
              )}
            </>
          )}

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
      <FormSection title="Optimizer" learn={SECTION_LEARN.optimizer}>
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
          <InputField label="lr" type="float" value={config.lr} onChange={(v) => set('lr', v as number)} step={1e-5} hint="Learning rate. 3e-4 is a good pretraining default; use 10-100× smaller for fine-tuning" />
        )}
        <InputField label="weight_decay" type="float" value={config.weight_decay} onChange={(v) => set('weight_decay', v as number)} step={0.01} hint="L2 regularization strength. 0.1 is the modern LLM default" />

        {isSFT ? (
          <InputField label="max_grad_norm" type="float" value={sft.max_grad_norm ?? 1.0} onChange={(v) => setSFT('max_grad_norm', v as number)} step={0.1} />
        ) : (
          <InputField label="grad_clip" type="float" value={config.grad_clip} onChange={(v) => set('grad_clip', v as number)} step={0.1} hint="Caps gradient norm to prevent unstable updates. 1.0 is standard" />
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

      {/* Scheduler — RLHF runs at constant LR (backend has no scheduler) */}
      {!isRLHF && (
      <FormSection title="Scheduler" learn={SECTION_LEARN.scheduler}>
        <SelectField
          label="scheduler"
          value={config.scheduler}
          onChange={(v) => set('scheduler', v as TrainingConfig['scheduler'])}
          options={[
            { value: 'cosine', label: 'Cosine Annealing' },
            ...(!isSFT ? [{ value: 'wsd', label: 'WSD — Warmup / Stable / Decay (modern)' }] : []),
            { value: 'linear', label: 'Linear Decay' },
            { value: 'polynomial', label: 'Polynomial' },
            { value: 'constant', label: 'Constant' },
            { value: 'adaptive', label: 'Adaptive' },
          ]}
          hint="WSD keeps LR constant then decays at the end — lets you extend training from any plateau checkpoint (SmolLM/MiniCPM style)"
        />
        {config.scheduler === 'wsd' && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="decay_fraction" type="float" value={config.wsd_decay_fraction} onChange={(v) => set('wsd_decay_fraction', v as number)} step={0.05} hint="Fraction of steps in the final decay. 0.1-0.2 typical" />
          </div>
        )}
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
      )}

      {/* Loss — RLHF's loss is defined by the algorithm above */}
      {!isRLHF && (
      <FormSection title="Loss Function" learn={SECTION_LEARN.loss}>
        <SelectField
          label="loss_fn"
          value={config.loss_fn}
          onChange={(v) => set('loss_fn', v as TrainingConfig['loss_fn'])}
          options={[
            { value: 'cross_entropy', label: 'Cross Entropy' },
            { value: 'maxis', label: 'MAXIS' },
          ]}
        />
        {config.loss_fn === 'cross_entropy' && !isSFT && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="z_loss_weight" type="float" value={config.z_loss_weight} onChange={(v) => set('z_loss_weight', v as number)} step={1e-5} hint="0 = off. 1e-4 (PaLM/OLMo2) stabilizes long runs by keeping logits bounded" />
          </div>
        )}
        {config.loss_fn === 'maxis' && (
          <div className="ml-4 pl-3 border-l-2 border-slate-700 space-y-3">
            <InputField label="n_candidates" type="number" value={config.maxis_n_candidates} onChange={(v) => set('maxis_n_candidates', v as number)} />
            <InputField label="low_rank_dim" type="number" value={config.maxis_low_rank_dim} onChange={(v) => set('maxis_low_rank_dim', v as number)} />
            <InputField label="chunk_size" type="number" value={config.maxis_chunk_size} onChange={(v) => set('maxis_chunk_size', v as number)} />
            <InputField label="aux_weight" type="float" value={config.maxis_aux_weight} onChange={(v) => set('maxis_aux_weight', v as number)} step={0.01} />
          </div>
        )}
      </FormSection>
      )}

      {/* Training loop */}
      <FormSection title="Training Loop" learn={SECTION_LEARN.loop}>
        <InputField label="batch_size" type="number" value={config.batch_size} onChange={(v) => set('batch_size', v as number)} min={1} hint={isRLHF ? 'Prompts per rollout batch' : 'Sequences per forward pass. Raise until you run out of VRAM, then use grad accumulation'} />
        {isRLHF && (
          <InputField label="mini_batch_size" type="number" value={rlhf.mini_batch_size ?? 4} onChange={(v) => setRLHF('mini_batch_size', v as number)} min={1} hint="Optimization slice of the rollout batch — must divide batch_size" />
        )}
        <InputField label="grad_accum_steps" type="number" value={config.gradient_accumulation_steps} onChange={(v) => set('gradient_accumulation_steps', v as number)} hint="Effective batch = batch_size × accum_steps" />
        <InputField label="max_steps" type="number" value={config.max_steps} onChange={(v) => set('max_steps', v as number)} hint={isRLHF ? 'Total RLHF steps — hundreds usually suffice, alignment overtrains fast' : isSFT ? 'Total training batches. Tokens seen ≈ max_steps × batch_size × seq_len' : 'Total weight updates. Tokens seen ≈ max_steps × batch_size × gradient_accumulation_steps × (seq_len − 1)'} />
        {!isRLHF && (
          <InputField label="warmup_steps" type="number" value={config.warmup_steps} onChange={(v) => set('warmup_steps', v as number)} hint="Steps to ramp LR from 0 — protects early training. ~1-10% of max_steps" />
        )}
        <InputField label="eval_every" type="number" value={config.eval_every} onChange={(v) => set('eval_every', v as number)} />
        {!isRLHF && (
          <InputField label="eval_steps" type="number" value={config.eval_steps} onChange={(v) => set('eval_steps', v as number)} />
        )}
        {isSFT && (
          <>
            <InputField label="log_every" type="number" value={sft.log_every ?? 10} onChange={(v) => setSFT('log_every', v as number)} />
            <InputField label="save_every" type="number" value={sft.save_every ?? 500} onChange={(v) => setSFT('save_every', v as number)} />
          </>
        )}
        <AdvancedSection>
          {!isRLHF && <ToggleField label="save_best_only" value={config.save_best_only ?? true} onChange={(v) => set('save_best_only', v)} />}
          {!isSFT && <ToggleField label="torch.compile (experimental)" value={config.use_compile ?? false} onChange={(v) => set('use_compile', v)} />}
          {!isSFT && <ToggleField label="doc-boundary masks" value={config.use_doc_masks ?? false} onChange={(v) => set('use_doc_masks', v)} />}
          {isSFT && (
            <>
              {!isRLHF && (
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
              )}
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
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-hidden focus:border-accent text-sm placeholder-slate-600"
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
