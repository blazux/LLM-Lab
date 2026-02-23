import Canvas from './components/Canvas';
import { TrainingProvider } from './context/TrainingContext';

function App() {
  return (
    <TrainingProvider>
      <div className="flex h-screen w-screen" style={{ backgroundColor: '#0f1219' }}>
        <Canvas />
      </div>
    </TrainingProvider>
  );
}

export default App;
