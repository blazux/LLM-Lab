import { useState, useEffect } from 'react';
import { GitMerge, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { motion } from 'framer-motion';

const API_BASE_URL = 'http://localhost:8000/api';

interface Checkpoint {
  name: string;
  path: string;
  type: string;
  has_lora: boolean;
  size_mb: number;
  modified: number;
}

interface Adapter {
  name: string;
  path: string;
  rank: number | string;
  alpha: number | string;
  target_modules: string[];
  modified: number;
}

interface MergeResult {
  success: boolean;
  message: string;
  output_path?: string;
  details?: {
    base_checkpoint?: string;
    adapter_path?: string;
    checkpoint?: string;
    method: string;
  };
}

const MergeLoRA = () => {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isMerging, setIsMerging] = useState<boolean>(false);

  // Form state
  const [inputType, setInputType] = useState<string>('adapter_folder');
  const [selectedBase, setSelectedBase] = useState<string>('');
  const [selectedAdapter, setSelectedAdapter] = useState<string>('');
  const [selectedFullCheckpoint, setSelectedFullCheckpoint] = useState<string>('');
  const [outputPath, setOutputPath] = useState<string>('');

  // Result state
  const [mergeResult, setMergeResult] = useState<MergeResult | null>(null);

  // Load available files on mount
  useEffect(() => {
    loadAvailable();
  }, []);

  // Auto-generate output path when selections change
  useEffect(() => {
    if (inputType === 'adapter_folder' && selectedBase) {
      const baseName = selectedBase.replace('.pt', '');
      setOutputPath(`${baseName}_merged.pt`);
    } else if (inputType === 'full_checkpoint' && selectedFullCheckpoint) {
      const baseName = selectedFullCheckpoint.replace('.pt', '');
      setOutputPath(`${baseName}_merged.pt`);
    }
  }, [inputType, selectedBase, selectedFullCheckpoint]);

  const loadAvailable = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/merge/available`);
      const data = await response.json();
      setCheckpoints(data.checkpoints || []);
      setAdapters(data.adapters || []);

      // Auto-select first items
      if (data.checkpoints && data.checkpoints.length > 0) {
        setSelectedBase(data.checkpoints[0].path);
        // Find first checkpoint with LoRA for full checkpoint method
        const loraCheckpoint = data.checkpoints.find((c: Checkpoint) => c.has_lora);
        if (loraCheckpoint) {
          setSelectedFullCheckpoint(loraCheckpoint.path);
        }
      }
      if (data.adapters && data.adapters.length > 0) {
        setSelectedAdapter(data.adapters[0].path);
      }
    } catch (error) {
      console.error('Failed to load available files:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleMerge = async () => {
    if (inputType === 'adapter_folder' && (!selectedBase || !selectedAdapter)) {
      alert('Please select both a base checkpoint and an adapter folder');
      return;
    }

    if (inputType === 'full_checkpoint' && !selectedFullCheckpoint) {
      alert('Please select a checkpoint to merge');
      return;
    }

    if (!outputPath.trim()) {
      alert('Please specify an output path');
      return;
    }

    try {
      setIsMerging(true);
      setMergeResult(null);

      const requestBody = {
        input_type: inputType,
        adapter_path: inputType === 'adapter_folder' ? selectedAdapter : '',
        base_checkpoint_path: inputType === 'adapter_folder' ? selectedBase : selectedFullCheckpoint,
        output_path: outputPath,
      };

      const response = await fetch(`${API_BASE_URL}/merge/lora`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const data: MergeResult = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Merge failed');
      }

      setMergeResult(data);

      // Reload available files to show the new merged checkpoint
      setTimeout(() => loadAvailable(), 1000);
    } catch (error: any) {
      console.error('Merge failed:', error);
      setMergeResult({
        success: false,
        message: error.message || 'Merge operation failed',
      });
    } finally {
      setIsMerging(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-8 h-8 text-teal-500 animate-spin" />
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

  const loraCheckpoints = checkpoints.filter((c) => c.has_lora);

  return (
    <div className="h-full flex flex-col p-6 bg-slate-900 overflow-hidden">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-2">🔀 LoRA Merge</h1>
        <p className="text-slate-400 text-sm">
          Merge LoRA adapters into base model to stack SFT and RLHF improvements
        </p>
      </div>

      <div className="flex-1 flex gap-6 overflow-hidden">
        {/* Left Panel - Configuration */}
        <div className="flex-1 flex flex-col gap-4 overflow-auto">
          {/* Merge Method Selection */}
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <label className="text-sm font-medium text-slate-300 mb-2 block">Merge Method</label>
            <select
              value={inputType}
              onChange={(e) => setInputType(e.target.value)}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="adapter_folder">Adapter Folder (Recommended)</option>
              <option value="full_checkpoint">Full Checkpoint</option>
            </select>
            <p className="text-xs text-slate-400 mt-2">
              {inputType === 'adapter_folder'
                ? 'Load adapters from a separate folder (lightweight, recommended for new training)'
                : 'Load adapters from a checkpoint containing both base and LoRA weights'}
            </p>
          </div>

          {/* Adapter Folder Method */}
          {inputType === 'adapter_folder' && (
            <>
              {/* Base Checkpoint Selection */}
              <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <label className="text-sm font-medium text-slate-300 mb-2 block">
                  Base Checkpoint
                </label>
                <select
                  value={selectedBase}
                  onChange={(e) => setSelectedBase(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
                >
                  {checkpoints.map((ckpt) => (
                    <option key={ckpt.path} value={ckpt.path}>
                      {ckpt.name} - {ckpt.type} ({ckpt.size_mb}MB)
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-400 mt-2">
                  The base model checkpoint to merge adapters into
                </p>
              </div>

              {/* Adapter Selection */}
              <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <label className="text-sm font-medium text-slate-300 mb-2 block">
                  LoRA Adapters
                </label>
                {adapters.length > 0 ? (
                  <>
                    <select
                      value={selectedAdapter}
                      onChange={(e) => setSelectedAdapter(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
                    >
                      {adapters.map((adapter) => (
                        <option key={adapter.path} value={adapter.path}>
                          {adapter.name} - r={adapter.rank}, alpha={adapter.alpha}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-slate-400 mt-2">
                      LoRA adapter folder from SFT or RLHF training
                    </p>
                  </>
                ) : (
                  <div className="text-sm text-slate-400 italic">
                    No adapter folders found. Train with LoRA to create adapters.
                  </div>
                )}
              </div>
            </>
          )}

          {/* Full Checkpoint Method */}
          {inputType === 'full_checkpoint' && (
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Checkpoint with LoRA
              </label>
              {loraCheckpoints.length > 0 ? (
                <>
                  <select
                    value={selectedFullCheckpoint}
                    onChange={(e) => setSelectedFullCheckpoint(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
                  >
                    {loraCheckpoints.map((ckpt) => (
                      <option key={ckpt.path} value={ckpt.path}>
                        {ckpt.name} - {ckpt.type} ({ckpt.size_mb}MB)
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-400 mt-2">
                    Checkpoint containing both base model and LoRA weights
                  </p>
                </>
              ) : (
                <div className="text-sm text-slate-400 italic">
                  No checkpoints with LoRA weights found. Use adapter folder method instead.
                </div>
              )}
            </div>
          )}

          {/* Output Path */}
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <label className="text-sm font-medium text-slate-300 mb-2 block">
              Output Path
            </label>
            <input
              type="text"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              placeholder="/app/data/merged_model.pt"
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-teal-500"
            />
            <p className="text-xs text-slate-400 mt-2">Path where the merged checkpoint will be saved</p>
          </div>

          {/* Merge Button */}
          <button
            onClick={handleMerge}
            disabled={
              isMerging ||
              !outputPath.trim() ||
              (inputType === 'adapter_folder' && (!selectedBase || !selectedAdapter)) ||
              (inputType === 'full_checkpoint' && !selectedFullCheckpoint)
            }
            className="px-6 py-3 bg-gradient-to-r from-teal-600 to-teal-700 hover:from-teal-700 hover:to-teal-800 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition-all"
          >
            {isMerging ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Merging...
              </>
            ) : (
              <>
                <GitMerge className="w-5 h-5" />
                Merge LoRA Adapters
              </>
            )}
          </button>
        </div>

        {/* Right Panel - Info & Results */}
        <div className="flex-1 flex flex-col gap-4 overflow-auto">
          {/* Workflow Info */}
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Workflow</h3>
            <div className="space-y-2 text-sm text-slate-400">
              <div className="flex items-start gap-2">
                <div className="bg-blue-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs flex-shrink-0 mt-0.5">
                  1
                </div>
                <p>Pretrain base model</p>
              </div>
              <div className="flex items-start gap-2">
                <div className="bg-blue-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs flex-shrink-0 mt-0.5">
                  2
                </div>
                <p>SFT with LoRA → save adapters</p>
              </div>
              <div className="flex items-start gap-2">
                <div className="bg-teal-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs flex-shrink-0 mt-0.5">
                  3
                </div>
                <p>
                  <strong>Merge SFT LoRA</strong> into base → new checkpoint
                </p>
              </div>
              <div className="flex items-start gap-2">
                <div className="bg-blue-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs flex-shrink-0 mt-0.5">
                  4
                </div>
                <p>RLHF with LoRA on merged checkpoint</p>
              </div>
              <div className="flex items-start gap-2">
                <div className="bg-teal-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs flex-shrink-0 mt-0.5">
                  5
                </div>
                <p>
                  <strong>Merge RLHF LoRA</strong> → final checkpoint
                </p>
              </div>
            </div>
          </div>

          {/* Statistics */}
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Available</h3>
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-blue-400">{checkpoints.length}</div>
                <div className="text-xs text-slate-400">Checkpoints</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-teal-400">{adapters.length}</div>
                <div className="text-xs text-slate-400">Adapter Folders</div>
              </div>
            </div>
          </div>

          {/* Merge Result */}
          {mergeResult && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`rounded-lg p-4 border ${
                mergeResult.success
                  ? 'bg-green-900/20 border-green-700'
                  : 'bg-red-900/20 border-red-700'
              }`}
            >
              <div className="flex items-start gap-3">
                {mergeResult.success ? (
                  <CheckCircle className="w-6 h-6 text-green-400 flex-shrink-0" />
                ) : (
                  <XCircle className="w-6 h-6 text-red-400 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <h3
                    className={`text-sm font-medium mb-2 ${
                      mergeResult.success ? 'text-green-300' : 'text-red-300'
                    }`}
                  >
                    {mergeResult.success ? 'Merge Successful!' : 'Merge Failed'}
                  </h3>
                  <p className="text-sm text-slate-300 mb-2">{mergeResult.message}</p>
                  {mergeResult.output_path && (
                    <div className="bg-slate-900 rounded px-3 py-2 mt-2">
                      <div className="text-xs text-slate-400 mb-1">Output:</div>
                      <div className="text-sm text-white font-mono break-all">
                        {mergeResult.output_path}
                      </div>
                    </div>
                  )}
                  {mergeResult.details && (
                    <div className="text-xs text-slate-400 mt-2">
                      Method: {mergeResult.details.method}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MergeLoRA;
