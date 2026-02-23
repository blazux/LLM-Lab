import { Node, Edge } from 'reactflow';
import { CheckCircle2, AlertCircle, XCircle } from 'lucide-react';

interface ValidationPanelProps {
  nodes: Node[];
  edges: Edge[];
  mode: 'model' | 'training' | 'sft' | 'rlhf';
  architectureFilter?: 'transformer' | 'mamba2';
}

interface ValidationItem {
  label: string;
  status: 'complete' | 'warning' | 'missing';
  message?: string;
}

const ValidationPanel = ({ nodes, edges, mode, architectureFilter }: ValidationPanelProps) => {
  const validateModel = (): ValidationItem[] => {
    const validations: ValidationItem[] = [];

    // Detect architecture from nodes if they exist, otherwise use filter
    const hasMamba2Nodes = nodes.some(n =>
      n.type === 'ssmcore' || n.type === 'temporalconv' ||
      n.type === 'gating' || n.type === 'headprojection'
    );
    const hasTransformerNodes = nodes.some(n =>
      ['mha', 'gqa', 'mqa', 'mla', 'rope', 'alibi', 'yarn', 'sinusoidal'].includes(n.type || '')
    );

    // Use nodes to detect architecture if they exist, otherwise use the filter
    let isMamba2: boolean;
    if (hasMamba2Nodes || hasTransformerNodes) {
      isMamba2 = hasMamba2Nodes && !hasTransformerNodes;
    } else {
      isMamba2 = architectureFilter === 'mamba2';
    }

    // Check for tokenizer
    const hasTokenizer = nodes.some(n => n.type === 'tokenizer');
    validations.push({
      label: 'Tokenizer',
      status: hasTokenizer ? 'complete' : 'missing',
      message: hasTokenizer ? undefined : 'Add a Tokenizer node'
    });

    // Check for embedding
    const hasEmbedding = nodes.some(n => n.type === 'embedding');
    validations.push({
      label: 'Embedding',
      status: hasEmbedding ? 'complete' : 'missing',
      message: hasEmbedding ? undefined : 'Add an Embedding node'
    });

    if (isMamba2) {
      // Mamba2-specific validations
      const hasSSMCore = nodes.some(n => n.type === 'ssmcore');
      validations.push({
        label: 'SSM Core',
        status: hasSSMCore ? 'complete' : 'missing',
        message: hasSSMCore ? undefined : 'Add SSM Core node'
      });

      const hasTemporalConv = nodes.some(n => n.type === 'temporalconv');
      validations.push({
        label: 'Temporal Convolution',
        status: hasTemporalConv ? 'complete' : 'missing',
        message: hasTemporalConv ? undefined : 'Add Temporal Conv node'
      });

      const hasGating = nodes.some(n => n.type === 'gating');
      validations.push({
        label: 'Gating',
        status: hasGating ? 'complete' : 'missing',
        message: hasGating ? undefined : 'Add Gating node'
      });

      const hasHeadProjection = nodes.some(n => n.type === 'headprojection');
      validations.push({
        label: 'Head Projection',
        status: hasHeadProjection ? 'complete' : 'missing',
        message: hasHeadProjection ? undefined : 'Add Head Projection node'
      });
    } else {
      // Transformer-specific validations
      const hasPosEncoding = nodes.some(n => ['rope', 'alibi', 'yarn', 'sinusoidal'].includes(n.type || ''));
      validations.push({
        label: 'Positional Encoding',
        status: hasPosEncoding ? 'complete' : 'warning',
        message: hasPosEncoding ? undefined : 'Add RoPE, ALiBi, YARN, or Sinusoidal'
      });

      const hasAttention = nodes.some(n => ['mha', 'gqa', 'mqa', 'mla'].includes(n.type || ''));
      validations.push({
        label: 'Attention',
        status: hasAttention ? 'complete' : 'missing',
        message: hasAttention ? undefined : 'Add MHA, GQA, MQA, or MLA'
      });

      const hasFFN = nodes.some(n => ['swiglu', 'gelu', 'relu'].includes(n.type || ''));
      validations.push({
        label: 'Feed-Forward',
        status: hasFFN ? 'complete' : 'missing',
        message: hasFFN ? undefined : 'Add SwiGLU, GELU, or ReLU'
      });
    }

    // Common validations (both architectures)
    const hasNorm = nodes.some(n => ['rmsnorm', 'layernorm'].includes(n.type || ''));
    validations.push({
      label: 'Normalization',
      status: hasNorm ? 'complete' : 'missing',
      message: hasNorm ? undefined : 'Add RMSNorm or LayerNorm'
    });

    // Check for loop edge
    const hasLoop = edges.some(e => e.data?.isLoop);
    const loopMessage = isMamba2
      ? 'Connect Norm back to Temporal Conv for n_layers'
      : 'Connect second Norm back to Attention for n_layers';
    validations.push({
      label: 'Layer Loop',
      status: hasLoop ? 'complete' : 'warning',
      message: hasLoop ? `${edges.find(e => e.data?.isLoop)?.data?.repeatCount || 0} layers` : loopMessage
    });

    // Check for LM Head
    const hasLMHead = nodes.some(n => n.type === 'lmhead');
    validations.push({
      label: 'LM Head',
      status: hasLMHead ? 'complete' : 'missing',
      message: hasLMHead ? undefined : 'Add an LM Head node'
    });

    return validations;
  };

  const validateTraining = (): ValidationItem[] => {
    const validations: ValidationItem[] = [];

    // Check for model
    const hasModel = nodes.some(n => n.type === 'model');
    validations.push({
      label: 'Model',
      status: hasModel ? 'complete' : 'missing',
      message: hasModel ? undefined : 'Add a Model node'
    });

    // Check for dataset
    const hasDataset = nodes.some(n => n.type === 'dataset');
    validations.push({
      label: 'Dataset',
      status: hasDataset ? 'complete' : 'missing',
      message: hasDataset ? undefined : 'Add a Dataset node'
    });

    // Check for optimizer
    const hasOptimizer = nodes.some(n => ['adamw', 'muon', 'lion', 'sophia'].includes(n.type || ''));
    validations.push({
      label: 'Optimizer',
      status: hasOptimizer ? 'complete' : 'missing',
      message: hasOptimizer ? undefined : 'Add AdamW, Muon, Lion, or Sophia'
    });

    // Check for scheduler
    const hasScheduler = nodes.some(n => ['cosine', 'linear', 'polynomial', 'constant', 'adaptive'].includes(n.type || ''));
    validations.push({
      label: 'LR Scheduler',
      status: hasScheduler ? 'complete' : 'warning',
      message: hasScheduler ? undefined : 'Add a scheduler (Cosine, Linear, Adaptive, etc.)'
    });

    // Check for hyperparameters
    const hasHyperparams = nodes.some(n => n.type === 'hyperparams');
    validations.push({
      label: 'Hyperparameters',
      status: hasHyperparams ? 'complete' : 'warning',
      message: hasHyperparams ? undefined : 'Add a Hyperparameters node (defaults will be used)'
    });

    return validations;
  };

  const validateSFT = (): ValidationItem[] => {
    const validations: ValidationItem[] = [];

    // Check for base model
    const hasBaseModel = nodes.some(n => n.type === 'basemodel');
    validations.push({
      label: 'Base Model',
      status: hasBaseModel ? 'complete' : 'missing',
      message: hasBaseModel ? undefined : 'Add a Base Model node'
    });

    // Check for dataset
    const hasDataset = nodes.some(n => n.type === 'dataset');
    validations.push({
      label: 'Dataset',
      status: hasDataset ? 'complete' : 'missing',
      message: hasDataset ? undefined : 'Add at least one Dataset node'
    });

    // Check for optimizer
    const hasOptimizer = nodes.some(n => ['adamw', 'muon', 'lion', 'sophia'].includes(n.type || ''));
    validations.push({
      label: 'Optimizer',
      status: hasOptimizer ? 'complete' : 'missing',
      message: hasOptimizer ? undefined : 'Add AdamW, Muon, Lion, or Sophia'
    });

    // Check for scheduler (optional but recommended)
    const hasScheduler = nodes.some(n => ['cosine', 'linear', 'polynomial', 'constant', 'adaptive'].includes(n.type || ''));
    validations.push({
      label: 'LR Scheduler',
      status: hasScheduler ? 'complete' : 'warning',
      message: hasScheduler ? undefined : 'Recommended: Add a scheduler'
    });

    // Check for hyperparameters (optional)
    const hasHyperparams = nodes.some(n => n.type === 'hyperparams');
    validations.push({
      label: 'Hyperparameters',
      status: hasHyperparams ? 'complete' : 'warning',
      message: hasHyperparams ? undefined : 'Optional: Configure training hyperparameters'
    });

    // Check for LoRA (optional)
    const hasLoRA = nodes.some(n => n.type === 'lora');
    validations.push({
      label: 'LoRA',
      status: hasLoRA ? 'complete' : 'warning',
      message: hasLoRA ? 'Using LoRA' : 'Optional: Add LoRA for parameter-efficient fine-tuning'
    });

    return validations;
  };

  const validateRLHF = (): ValidationItem[] => {
    const validations: ValidationItem[] = [];

    // Check for policy model
    const hasPolicyModel = nodes.some(n => n.type === 'basemodel');
    validations.push({
      label: 'Policy Model',
      status: hasPolicyModel ? 'complete' : 'missing',
      message: hasPolicyModel ? undefined : 'Add a Policy Model node'
    });

    // Check for exactly one algorithm node
    const hasPPO = nodes.some(n => n.type === 'ppo_reward');
    const hasDPO = nodes.some(n => n.type === 'dpo_reference');
    const hasGRPO = nodes.some(n => n.type === 'grpo_reward');
    const algorithmCount = [hasPPO, hasDPO, hasGRPO].filter(Boolean).length;

    if (algorithmCount === 1) {
      const algorithm = hasPPO ? 'PPO' : hasDPO ? 'DPO' : 'GRPO';
      validations.push({
        label: 'Algorithm',
        status: 'complete',
        message: `Using ${algorithm}`
      });
    } else if (algorithmCount === 0) {
      validations.push({
        label: 'Algorithm',
        status: 'missing',
        message: 'Add one algorithm node (PPO/DPO/GRPO)'
      });
    } else {
      validations.push({
        label: 'Algorithm',
        status: 'warning',
        message: `Multiple algorithms detected (${algorithmCount}) - use only one`
      });
    }

    // Check for dataset
    const hasDataset = nodes.some(n => n.type === 'dataset');
    validations.push({
      label: 'Dataset',
      status: hasDataset ? 'complete' : 'missing',
      message: hasDataset ? undefined : 'Add at least one Dataset node'
    });

    // Check for optimizer
    const hasOptimizer = nodes.some(n => ['adamw', 'muon', 'lion', 'sophia'].includes(n.type || ''));
    validations.push({
      label: 'Optimizer',
      status: hasOptimizer ? 'complete' : 'missing',
      message: hasOptimizer ? undefined : 'Add AdamW, Muon, Lion, or Sophia'
    });

    // Check for scheduler (optional)
    const hasScheduler = nodes.some(n => ['cosine', 'linear', 'polynomial', 'constant', 'adaptive'].includes(n.type || ''));
    validations.push({
      label: 'LR Scheduler',
      status: hasScheduler ? 'complete' : 'warning',
      message: hasScheduler ? undefined : 'Recommended: Add a scheduler'
    });

    // Check for RLHF hyperparameters
    const hasRLHFHyperparams = nodes.some(n => n.type === 'rlhf_hyperparams');
    validations.push({
      label: 'RLHF Hyperparameters',
      status: hasRLHFHyperparams ? 'complete' : 'warning',
      message: hasRLHFHyperparams ? undefined : 'Optional: Configure RLHF hyperparameters'
    });

    // Check for LoRA (optional)
    const hasLoRA = nodes.some(n => n.type === 'lora');
    validations.push({
      label: 'LoRA',
      status: hasLoRA ? 'complete' : 'warning',
      message: hasLoRA ? 'Using LoRA' : 'Optional: Add LoRA for parameter-efficient fine-tuning'
    });

    return validations;
  };

  const validations = mode === 'model' ? validateModel() : mode === 'sft' ? validateSFT() : mode === 'rlhf' ? validateRLHF() : validateTraining();
  const completeCount = validations.filter(v => v.status === 'complete').length;
  const totalCount = validations.length;
  const percentage = Math.round((completeCount / totalCount) * 100);

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 shadow-xl">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold text-sm">Configuration Status</h3>
        <div className="text-xs font-mono text-slate-400">
          {completeCount}/{totalCount} ({percentage}%)
        </div>
      </div>

      <div className="space-y-2">
        {validations.map((validation, index) => (
          <div key={index} className="flex items-start gap-2">
            {validation.status === 'complete' && (
              <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
            )}
            {validation.status === 'warning' && (
              <AlertCircle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
            )}
            {validation.status === 'missing' && (
              <XCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
            )}
            <div className="flex-1 min-w-0">
              <div className={`text-xs font-medium ${
                validation.status === 'complete' ? 'text-green-400' :
                validation.status === 'warning' ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {validation.label}
              </div>
              {validation.message && (
                <div className="text-xs text-slate-400 mt-0.5">
                  {validation.message}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ValidationPanel;
