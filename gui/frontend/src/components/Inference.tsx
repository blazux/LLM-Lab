import { useState, useEffect } from 'react';
import { Play, Loader2, Trash2 } from 'lucide-react';
import { motion } from 'framer-motion';

const API_BASE_URL = 'http://localhost:8000/api';

interface Checkpoint {
  name: string;
  path: string;
  step: number | string;
  type: string;
  size_mb: number;
  modified: number;
}

interface GenerateResponse {
  generated_text: string;
  tokens_generated: number;
  tokens_per_second: number;
  prompt_tokens: number;
}

const Inference = () => {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('');
  const [generatedText, setGeneratedText] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Generation parameters
  const [maxTokens, setMaxTokens] = useState<number>(100);
  const [temperature, setTemperature] = useState<number>(1.0);
  const [topP, setTopP] = useState<number>(0.9);
  const [topK, setTopK] = useState<number>(50);
  const [repetitionPenalty, setRepetitionPenalty] = useState<number>(1.0);
  const [strategy, setStrategy] = useState<string>('top_p');

  // Stats
  const [tokensGenerated, setTokensGenerated] = useState<number>(0);
  const [tokensPerSecond, setTokensPerSecond] = useState<number>(0);
  const [promptTokens, setPromptTokens] = useState<number>(0);

  // Load checkpoints on mount
  useEffect(() => {
    loadCheckpoints();
  }, []);

  const loadCheckpoints = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/inference/checkpoints`);
      const data = await response.json();
      setCheckpoints(data.checkpoints || []);
      if (data.checkpoints && data.checkpoints.length > 0) {
        setSelectedCheckpoint(data.checkpoints[0].path);
      }
    } catch (error) {
      console.error('Failed to load checkpoints:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!selectedCheckpoint || !prompt.trim()) {
      alert('Please select a checkpoint and enter a prompt');
      return;
    }

    try {
      setIsGenerating(true);
      setGeneratedText('');
      setTokensGenerated(0);
      setTokensPerSecond(0);

      const response = await fetch(`${API_BASE_URL}/inference/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          checkpoint_path: selectedCheckpoint,
          prompt: prompt,
          max_tokens: maxTokens,
          temperature: temperature,
          top_k: topK,
          top_p: topP,
          repetition_penalty: repetitionPenalty,
          strategy: strategy,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Generation failed');
      }

      const data: GenerateResponse = await response.json();
      setGeneratedText(data.generated_text);
      setTokensGenerated(data.tokens_generated);
      setTokensPerSecond(data.tokens_per_second);
      setPromptTokens(data.prompt_tokens);
    } catch (error: any) {
      console.error('Generation failed:', error);
      alert(`Generation failed: ${error.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleUnloadModel = async () => {
    try {
      await fetch(`${API_BASE_URL}/inference/unload`, { method: 'POST' });
      alert('Model unloaded successfully');
    } catch (error) {
      console.error('Failed to unload model:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-8 h-8 text-purple-500 animate-spin" />
      </div>
    );
  }

  if (checkpoints.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400">
        <p className="text-lg mb-2">No checkpoints found</p>
        <p className="text-sm">Train a model first to create checkpoints</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-6 bg-slate-900 overflow-hidden">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-2">💬 Inference</h1>
        <p className="text-slate-400 text-sm">Test your trained models with custom prompts</p>
      </div>

      <div className="flex-1 flex gap-6 overflow-hidden">
        {/* Left Panel - Input */}
        <div className="flex-1 flex flex-col gap-4 overflow-auto">
          {/* Checkpoint Selection */}
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <label className="text-sm font-medium text-slate-300 mb-2 block">
              Select Checkpoint
            </label>
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              {checkpoints.map((ckpt) => (
                <option key={ckpt.path} value={ckpt.path}>
                  {ckpt.name} - Step {ckpt.step} - {ckpt.type} ({ckpt.size_mb}MB)
                </option>
              ))}
            </select>
          </div>

          {/* Prompt Input */}
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 flex-1 flex flex-col">
            <label className="text-sm font-medium text-slate-300 mb-2 block">
              Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt here..."
              className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
            />
          </div>

          {/* Generation Parameters */}
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Generation Parameters</h3>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Max Tokens</label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Temperature</label>
                <input
                  type="number"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Top-P</label>
                <input
                  type="number"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Top-K</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Repetition Penalty</label>
                <input
                  type="number"
                  step="0.1"
                  value={repetitionPenalty}
                  onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                  className="w-full px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Strategy</label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                >
                  <option value="top_p">Top-P</option>
                  <option value="top_k">Top-K</option>
                  <option value="greedy">Greedy</option>
                </select>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !selectedCheckpoint || !prompt.trim()}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition-all"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Generate
                </>
              )}
            </button>
            <button
              onClick={handleUnloadModel}
              className="px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg flex items-center gap-2 transition-all"
              title="Unload model from memory"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Right Panel - Output */}
        <div className="flex-1 flex flex-col gap-4">
          {/* Stats */}
          {(tokensGenerated > 0 || promptTokens > 0) && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-slate-800 rounded-lg p-4 border border-slate-700"
            >
              <h3 className="text-sm font-medium text-slate-300 mb-3">Generation Stats</h3>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-purple-400">{tokensGenerated}</div>
                  <div className="text-xs text-slate-400">Tokens Generated</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-400">{tokensPerSecond.toFixed(1)}</div>
                  <div className="text-xs text-slate-400">Tokens/Second</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-blue-400">{promptTokens}</div>
                  <div className="text-xs text-slate-400">Prompt Tokens</div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Generated Output */}
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 flex-1 flex flex-col">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Generated Text</h3>
            <div className="flex-1 px-3 py-2 bg-slate-900 border border-slate-700 rounded-md text-white overflow-auto whitespace-pre-wrap font-mono text-sm">
              {generatedText || (
                <span className="text-slate-500 italic">Generated text will appear here...</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Inference;
