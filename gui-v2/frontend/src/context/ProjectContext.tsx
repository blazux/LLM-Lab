import React, { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import type { Project, ProjectType, ModelConfig, TrainingConfig } from '../types'
import {
  defaultTransformerConfig,
  defaultTrainingConfig,
  defaultSFTConfig,
} from '../types'

const STORAGE_KEY = 'llmlab_v2_projects'

interface ProjectContextValue {
  projects: Project[]
  createProject: (name: string, type: ProjectType) => Project
  updateProject: (id: string, updates: Partial<Project>) => void
  deleteProject: (id: string) => void
  getProject: (id: string) => Project | undefined
}

const ProjectContext = createContext<ProjectContextValue | null>(null)

function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as Project[]) : []
  } catch {
    return []
  }
}

function saveProjects(projects: Project[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects))
}

function makeDefaultModelConfig(): ModelConfig {
  return defaultTransformerConfig()
}

function makeDefaultTrainingConfig(type: ProjectType): TrainingConfig {
  if (type === 'sft') return defaultSFTConfig()
  return defaultTrainingConfig()
}

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [projects, setProjects] = useState<Project[]>(loadProjects)

  useEffect(() => {
    saveProjects(projects)
  }, [projects])

  const createProject = useCallback((name: string, type: ProjectType): Project => {
    const now = new Date().toISOString()
    const project: Project = {
      id: uuidv4(),
      name,
      type,
      createdAt: now,
      updatedAt: now,
      modelConfig: makeDefaultModelConfig(),
      trainingConfig: makeDefaultTrainingConfig(type),
      currentStep: 0,
    }
    setProjects((prev) => [...prev, project])
    return project
  }, [])

  const updateProject = useCallback((id: string, updates: Partial<Project>) => {
    setProjects((prev) =>
      prev.map((p) =>
        p.id === id
          ? { ...p, ...updates, updatedAt: new Date().toISOString() }
          : p
      )
    )
  }, [])

  const deleteProject = useCallback((id: string) => {
    setProjects((prev) => prev.filter((p) => p.id !== id))
  }, [])

  const getProject = useCallback(
    (id: string) => projects.find((p) => p.id === id),
    [projects]
  )

  return (
    <ProjectContext.Provider value={{ projects, createProject, updateProject, deleteProject, getProject }}>
      {children}
    </ProjectContext.Provider>
  )
}

export function useProjects(): ProjectContextValue {
  const ctx = useContext(ProjectContext)
  if (!ctx) throw new Error('useProjects must be used inside ProjectProvider')
  return ctx
}
