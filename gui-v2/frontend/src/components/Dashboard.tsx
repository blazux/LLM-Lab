import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Cpu, Brain, GitBranch, Trash2, ArrowRight, X, FlaskConical } from 'lucide-react'
import { useProjects } from '../context/ProjectContext'
import type { ProjectType } from '../types'

interface Props {
  onOpenProject: (id: string) => void
}

const TYPE_META: Record<ProjectType, { label: string; description: string; icon: React.ReactNode; color: string }> = {
  pretrain: {
    label: 'Pretrain',
    description: 'Train a model from scratch on raw text corpora',
    icon: <Cpu size={20} />,
    color: 'text-blue-400 border-blue-500/30 bg-blue-500/5',
  },
  sft: {
    label: 'SFT',
    description: 'Fine-tune a pretrained model on instruction data',
    icon: <Brain size={20} />,
    color: 'text-purple-400 border-purple-500/30 bg-purple-500/5',
  },
  rlhf: {
    label: 'RLHF',
    description: 'Align a model using reinforcement learning from human feedback',
    icon: <GitBranch size={20} />,
    color: 'text-amber-400 border-amber-500/30 bg-amber-500/5',
  },
}

export default function Dashboard({ onOpenProject }: Props) {
  const { projects, createProject, deleteProject } = useProjects()
  const [showModal, setShowModal] = useState(false)
  const [newName, setNewName] = useState('')
  const [newType, setNewType] = useState<ProjectType>('pretrain')
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  function handleCreate() {
    if (!newName.trim()) return
    const p = createProject(newName.trim(), newType)
    setShowModal(false)
    setNewName('')
    setNewType('pretrain')
    onOpenProject(p.id)
  }

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FlaskConical size={24} className="text-accent" />
            <span className="text-lg font-semibold text-slate-100">LLM Lab</span>
            <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full">v2</span>
          </div>
          <button
            onClick={() => setShowModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent-dim text-slate-950 rounded-lg font-medium text-sm transition-colors"
          >
            <Plus size={16} />
            New Project
          </button>
        </div>
      </header>

      {/* Body */}
      <main className="max-w-6xl mx-auto px-6 py-10">
        {projects.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-32 text-center">
            <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center mb-6">
              <FlaskConical size={32} className="text-slate-600" />
            </div>
            <h2 className="text-xl font-semibold text-slate-300 mb-2">No projects yet</h2>
            <p className="text-slate-500 mb-8 max-w-sm">
              Create your first project to start configuring and training an LLM.
            </p>
            <button
              onClick={() => setShowModal(true)}
              className="flex items-center gap-2 px-5 py-2.5 bg-accent hover:bg-accent-dim text-slate-950 rounded-lg font-medium transition-colors"
            >
              <Plus size={16} />
              Create Project
            </button>
          </div>
        ) : (
          <>
            <div className="mb-8">
              <h1 className="text-2xl font-bold text-slate-100">Projects</h1>
              <p className="text-slate-500 mt-1">{projects.length} project{projects.length !== 1 ? 's' : ''}</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <AnimatePresence>
                {projects.map((project) => {
                  const meta = TYPE_META[project.type]
                  const isDeleting = deleteConfirm === project.id
                  return (
                    <motion.div
                      key={project.id}
                      layout
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="relative bg-slate-900 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-colors group"
                    >
                      {/* Type badge */}
                      <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-medium mb-4 ${meta.color}`}>
                        {meta.icon}
                        {meta.label}
                      </div>

                      <h3 className="text-slate-100 font-semibold text-lg mb-1 pr-8">{project.name}</h3>
                      <p className="text-slate-500 text-sm mb-4">{meta.description}</p>

                      <div className="text-xs text-slate-600">
                        Updated {new Date(project.updatedAt).toLocaleDateString()}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-slate-800">
                        {isDeleting ? (
                          <>
                            <span className="text-xs text-slate-400 flex-1">Delete this project?</span>
                            <button
                              onClick={() => { deleteProject(project.id); setDeleteConfirm(null) }}
                              className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors"
                            >
                              Yes, delete
                            </button>
                            <button
                              onClick={() => setDeleteConfirm(null)}
                              className="text-xs px-2 py-1 bg-slate-800 text-slate-400 rounded hover:bg-slate-700 transition-colors"
                            >
                              Cancel
                            </button>
                          </>
                        ) : (
                          <>
                            <button
                              onClick={() => setDeleteConfirm(project.id)}
                              className="p-1.5 text-slate-600 hover:text-red-400 rounded transition-colors"
                            >
                              <Trash2 size={14} />
                            </button>
                            <button
                              onClick={() => onOpenProject(project.id)}
                              className="flex items-center gap-1.5 ml-auto px-3 py-1.5 bg-accent/10 text-accent hover:bg-accent/20 rounded-lg text-sm font-medium transition-colors"
                            >
                              Open
                              <ArrowRight size={14} />
                            </button>
                          </>
                        )}
                      </div>
                    </motion.div>
                  )
                })}
              </AnimatePresence>
            </div>
          </>
        )}
      </main>

      {/* Create Project Modal */}
      <AnimatePresence>
        {showModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={(e) => { if (e.target === e.currentTarget) setShowModal(false) }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              className="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-full max-w-md shadow-2xl"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-slate-100">New Project</h2>
                <button onClick={() => setShowModal(false)} className="text-slate-500 hover:text-slate-300 transition-colors">
                  <X size={20} />
                </button>
              </div>

              <div className="space-y-5">
                {/* Name */}
                <div>
                  <label className="block text-sm text-slate-400 mb-1.5">Project name</label>
                  <input
                    autoFocus
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleCreate() }}
                    placeholder="My LLM Project"
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-100 placeholder-slate-600 focus:outline-none focus:border-accent text-sm"
                  />
                </div>

                {/* Type */}
                <div>
                  <label className="block text-sm text-slate-400 mb-2">Training type</label>
                  <div className="grid grid-cols-3 gap-2">
                    {(Object.entries(TYPE_META) as [ProjectType, typeof TYPE_META[ProjectType]][]).map(([type, meta]) => (
                      <button
                        key={type}
                        onClick={() => setNewType(type)}
                        className={`flex flex-col items-center gap-2 p-3 rounded-xl border text-sm font-medium transition-all ${
                          newType === type
                            ? 'border-accent bg-accent/10 text-accent'
                            : 'border-slate-700 text-slate-400 hover:border-slate-600'
                        }`}
                      >
                        {meta.icon}
                        {meta.label}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-slate-600 mt-2">{TYPE_META[newType].description}</p>
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <button
                    onClick={() => setShowModal(false)}
                    className="flex-1 py-2.5 rounded-lg border border-slate-700 text-slate-400 hover:text-slate-300 text-sm transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreate}
                    disabled={!newName.trim()}
                    className="flex-1 py-2.5 rounded-lg bg-accent hover:bg-accent-dim text-slate-950 font-medium text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    Create
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
