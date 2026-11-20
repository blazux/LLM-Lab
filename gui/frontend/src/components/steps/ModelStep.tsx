import type { ModelConfig } from '../../types'
import AssemblyWorkshop from '../AssemblyWorkshop'

interface Props {
  config: ModelConfig
  onChange: (config: ModelConfig) => void
  readOnly?: boolean
}

export default function ModelStep({ config, onChange, readOnly }: Props) {
  if (readOnly) {
    return <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 text-sm leading-7 text-slate-400">
      The architecture is inherited from the checkpoint selected in Training.
      Project settings do not replace it. To build a different model,
      create a pretraining project in the workshop.
    </div>
  }

  return (
    <div className="h-full w-full">
      <AssemblyWorkshop config={config} onChange={onChange} />
    </div>
  )
}
