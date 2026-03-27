import { useState } from 'react'
import { ProjectProvider } from './context/ProjectContext'
import Dashboard from './components/Dashboard'
import ProjectView from './components/ProjectView'

type AppView =
  | { view: 'dashboard' }
  | { view: 'project'; projectId: string }

export default function App() {
  const [appView, setAppView] = useState<AppView>({ view: 'dashboard' })

  return (
    <ProjectProvider>
      {appView.view === 'dashboard' ? (
        <Dashboard
          onOpenProject={(id) => setAppView({ view: 'project', projectId: id })}
        />
      ) : (
        <ProjectView
          projectId={appView.projectId}
          onBack={() => setAppView({ view: 'dashboard' })}
        />
      )}
    </ProjectProvider>
  )
}
