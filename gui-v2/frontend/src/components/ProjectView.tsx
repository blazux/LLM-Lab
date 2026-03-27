import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft,
  Cpu,
  Type,
  Database,
  Settings,
  Play,
  Brain,
  GitBranch,
  FlaskConical,
} from 'lucide-react'
import { useProjects } from '../context/ProjectContext'
import type { ModelConfig, TrainingConfig, ProjectType } from '../types'
import Stepper, { type Step } from './Stepper'
import ModelStep from './steps/ModelStep'
import TokenizerStep from './steps/TokenizerStep'
import DatasetStep from './steps/DatasetStep'
import TrainingStep from './steps/TrainingStep'
import RunStep from './steps/RunStep'

// ─── Step definitions ─────────────────────────────────────────────────────────

function getSteps(type: ProjectType): Step[] {
  if (type === 'pretrain') {
    return [
      { id: 0, label: 'Model', icon: <Cpu size={14} /> },
      { id: 1, label: 'Tokenizer', icon: <Type size={14} /> },
      { id: 2, label: 'Dataset', icon: <Database size={14} /> },
      { id: 3, label: 'Training', icon: <Settings size={14} /> },
      { id: 4, label: 'Run', icon: <Play size={14} /> },
    ]
  }
  if (type === 'sft') {
    return [
      { id: 0, label: 'Model', icon: <Cpu size={14} /> },
      { id: 1, label: 'Tokenizer', icon: <Type size={14} /> },
      { id: 2, label: 'Dataset', icon: <Database size={14} /> },
      { id: 3, label: 'Training', icon: <Settings size={14} /> },
      { id: 4, label: 'Run', icon: <Play size={14} /> },
    ]
  }
  // RLHF
  return [
    { id: 0, label: 'Model', icon: <Brain size={14} /> },
    { id: 1, label: 'Tokenizer', icon: <Type size={14} /> },
    { id: 2, label: 'Dataset', icon: <Database size={14} /> },
    { id: 3, label: 'Training', icon: <Settings size={14} /> },
    { id: 4, label: 'Run', icon: <Play size={14} /> },
  ]
}

const TYPE_COLOR: Record<ProjectType, string> = {
  pretrain: 'text-blue-400',
  sft: 'text-purple-400',
  rlhf: 'text-amber-400',
}

const TYPE_ICON: Record<ProjectType, React.ReactNode> = {
  pretrain: <Cpu size={14} />,
  sft: <Brain size={14} />,
  rlhf: <GitBranch size={14} />,
}

// ─── ProjectView ──────────────────────────────────────────────────────────────

interface Props {
  projectId: string
  onBack: () => void
}

export default function ProjectView({ projectId, onBack }: Props) {
  const { getProject, updateProject } = useProjects()
  const project = getProject(projectId)

  const [currentStep, setCurrentStep] = useState(project?.currentStep ?? 0)

  const handleStepChange = useCallback(
    (step: number) => {
      setCurrentStep(step)
      updateProject(projectId, { currentStep: step })
    },
    [projectId, updateProject]
  )

  const handleModelChange = useCallback(
    (config: ModelConfig) => {
      updateProject(projectId, { modelConfig: config })
    },
    [projectId, updateProject]
  )

  const handleTrainingChange = useCallback(
    (config: TrainingConfig) => {
      updateProject(projectId, { trainingConfig: config })
    },
    [projectId, updateProject]
  )

  if (!project) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <p className="text-slate-500 mb-4">Project not found</p>
          <button onClick={onBack} className="text-accent hover:underline text-sm">
            Back to dashboard
          </button>
        </div>
      </div>
    )
  }

  const steps = getSteps(project.type)
  const isModelReadOnly = project.type === 'sft' || project.type === 'rlhf'

  function renderStepContent() {
    switch (currentStep) {
      case 0:
        return (
          <ModelStep
            config={project!.modelConfig}
            onChange={handleModelChange}
            readOnly={isModelReadOnly && project!.type !== 'sft'}
          />
        )
      case 1:
        return <TokenizerStep config={project!.modelConfig} onChange={handleModelChange} />
      case 2:
        return (
          <DatasetStep
            config={project!.trainingConfig}
            onChange={handleTrainingChange}
          />
        )
      case 3:
        return (
          <TrainingStep
            projectType={project!.type}
            config={project!.trainingConfig}
            onChange={handleTrainingChange}
          />
        )
      case 4:
        return <RunStep project={project!} />
      default:
        return null
    }
  }

  const isRunStep = currentStep === 4

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur flex-none">
        <div className="px-6 py-3 flex items-center gap-4">
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 text-slate-500 hover:text-slate-300 transition-colors text-sm"
          >
            <ArrowLeft size={16} />
            Projects
          </button>
          <div className="w-px h-4 bg-slate-700" />
          <FlaskConical size={16} className="text-accent" />
          <span className="font-semibold text-slate-100">{project.name}</span>
          <span className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-slate-800 ${TYPE_COLOR[project.type]}`}>
            {TYPE_ICON[project.type]}
            {project.type.toUpperCase()}
          </span>
        </div>
      </header>

      {/* Body */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Sidebar stepper */}
        <aside className="w-52 flex-none bg-slate-900/50 border-r border-slate-800 px-4 py-6">
          <Stepper
            steps={steps}
            current={currentStep}
            onChange={handleStepChange}
          />

          {/* Nav buttons */}
          <div className="mt-8 flex flex-col gap-2">
            {currentStep > 0 && (
              <button
                onClick={() => handleStepChange(currentStep - 1)}
                className="w-full py-2 text-xs text-slate-500 hover:text-slate-300 border border-slate-800 rounded-lg transition-colors"
              >
                ← Back
              </button>
            )}
            {currentStep < steps.length - 1 && (
              <button
                onClick={() => handleStepChange(currentStep + 1)}
                className="w-full py-2 text-xs bg-accent/10 text-accent hover:bg-accent/20 border border-accent/20 rounded-lg font-medium transition-colors"
              >
                Next →
              </button>
            )}
          </div>

          {/* Step info */}
          <div className="mt-6 text-xs text-slate-600">
            Step {currentStep + 1} of {steps.length}
          </div>
        </aside>

        {/* Main content */}
        <main
          className={`flex-1 min-w-0 overflow-auto ${isRunStep ? 'p-6' : 'p-8'}`}
          style={isRunStep ? { height: 'calc(100vh - 52px)' } : undefined}
        >
          {/* Step header */}
          <div className="mb-6 flex-none">
            <h2 className="text-xl font-bold text-slate-100">{steps[currentStep]?.label}</h2>
            <p className="text-sm text-slate-500 mt-0.5">
              {currentStep === 0 && 'Configure the model architecture'}
              {currentStep === 1 && 'Tokenizer settings'}
              {currentStep === 2 && 'Select training datasets'}
              {currentStep === 3 && 'Configure training hyperparameters'}
              {currentStep === 4 && 'Launch and monitor training'}
            </p>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -12 }}
              transition={{ duration: 0.15 }}
              className={isRunStep ? 'h-full' : ''}
            >
              {renderStepContent()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}
