import Canvas from './components/Canvas';
import { TrainingProvider } from './context/TrainingContext';

function App() {
  return (
    <TrainingProvider>
      <div className="flex h-screen w-screen bg-slate-900">
        <Canvas />
      </div>
    </TrainingProvider>
  );
}

export default App;
