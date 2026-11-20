import { useEffect, useState } from 'react'
import { ProjectProvider } from './context/ProjectContext'
import Dashboard from './components/Dashboard'
import ProjectView from './components/ProjectView'
import ToolsView from './components/ToolsView'
import LearningExperiment from './components/LearningExperiment'
import { isLearningExperimentId, LEARNING_EXPERIMENT_ID, type LearningExperimentId, type LearningPreset } from './services/learning'

type AppView =
  | { view: 'dashboard' }
  | { view: 'project'; projectId: string }
  | { view: 'tools' }
  | { view: 'learning'; experimentId: LearningExperimentId; preset?: LearningPreset }

function linkedView(): AppView {
  const id = new URLSearchParams(window.location.hash.slice(1)).get('experiment')
  return isLearningExperimentId(id) ? { view: 'learning', experimentId: id } : { view: 'dashboard' }
}

export default function App() {
  const [appView, setAppView] = useState<AppView>(linkedView)
  useEffect(() => {
    const navigate = () => setAppView(linkedView())
    window.addEventListener('hashchange', navigate)
    return () => window.removeEventListener('hashchange', navigate)
  }, [])
  function openLearning(experimentId: LearningExperimentId = LEARNING_EXPERIMENT_ID, preset?: LearningPreset) {
    window.history.pushState(null, '', `${window.location.pathname}${window.location.search}#experiment=${experimentId}`)
    setAppView({ view: 'learning', experimentId, preset })
  }
  function backToDashboard() {
    window.history.pushState(null, '', `${window.location.pathname}${window.location.search}`)
    setAppView({ view: 'dashboard' })
  }

  return (
    <ProjectProvider>
      {appView.view === 'dashboard' ? (
        <Dashboard
          onOpenProject={(id) => setAppView({ view: 'project', projectId: id })}
          onOpenTools={() => setAppView({ view: 'tools' })}
          onOpenLearning={openLearning}
        />
      ) : appView.view === 'project' ? (
        <ProjectView
          projectId={appView.projectId}
          onBack={() => setAppView({ view: 'dashboard' })}
        />
      ) : appView.view === 'learning' ? (
        <LearningExperiment key={appView.experimentId} experimentId={appView.experimentId} initialPreset={appView.preset} onOpenExperiment={openLearning} onBack={backToDashboard} />
      ) : (
        <ToolsView onBack={() => setAppView({ view: 'dashboard' })} />
      )}
    </ProjectProvider>
  )
}
