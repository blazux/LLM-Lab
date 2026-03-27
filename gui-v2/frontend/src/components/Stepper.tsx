import { Check } from 'lucide-react'

export interface Step {
  id: number
  label: string
  icon: React.ReactNode
}

interface Props {
  steps: Step[]
  current: number
  onChange: (step: number) => void
}

export default function Stepper({ steps, current, onChange }: Props) {
  return (
    <nav className="flex flex-col gap-1">
      {steps.map((step, idx) => {
        const isCompleted = idx < current
        const isActive = idx === current
        return (
          <div key={step.id} className="relative">
            {/* Connector line */}
            {idx < steps.length - 1 && (
              <div
                className={`absolute left-4 top-10 w-0.5 h-4 transition-colors ${
                  idx < current ? 'bg-accent' : 'bg-slate-700'
                }`}
              />
            )}
            <button
              onClick={() => onChange(idx)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all text-left ${
                isActive
                  ? 'bg-accent/10 text-accent'
                  : isCompleted
                  ? 'text-slate-300 hover:bg-slate-800'
                  : 'text-slate-500 hover:bg-slate-800/50'
              }`}
            >
              {/* Circle */}
              <div
                className={`flex-none w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold border transition-all ${
                  isActive
                    ? 'border-accent bg-accent text-slate-950'
                    : isCompleted
                    ? 'border-accent bg-accent/20 text-accent'
                    : 'border-slate-700 bg-slate-800 text-slate-500'
                }`}
              >
                {isCompleted ? <Check size={14} /> : step.icon}
              </div>
              <span className="font-medium">{step.label}</span>
            </button>
          </div>
        )
      })}
    </nav>
  )
}
