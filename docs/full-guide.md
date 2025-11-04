# LLM-Lab Complete Documentation

This document provides a comprehensive guide to training Large Language Models using LLM-Lab, from initial model configuration through base training, supervised fine-tuning, and reinforcement learning from human feedback.

---

## Table of Contents

1. [Training Pipeline Overview](#training-pipeline-overview)
2. [Model Configuration](#model-configuration)
3. [Base Training (Pretraining)](#base-training-pretraining)
4. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
   - [LoRA for SFT](#lora-for-sft)
5. [RLHF Training](#rlhf-training)
   - [PPO (Proximal Policy Optimization)](#ppo-proximal-policy-optimization)
   - [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
   - [GRPO (Group Relative Policy Optimization)](#grpo-group-relative-policy-optimization)
   - [LoRA for RLHF](#lora-for-rlhf)
6. [Model Inference and Testing](#model-inference-and-testing)
7. [LoRA Adapter Merging](#lora-adapter-merging)
8. [Best Practices and Tips](#best-practices-and-tips)
9. [Troubleshooting](#troubleshooting)

---

## Training Pipeline Overview

The typical LLM training pipeline consists of three main stages:

### 1. **Base Training (Pretraining)**
- Trains the model from scratch on large text corpora
- Learns basic language patterns, syntax, and general knowledge
- Most computationally expensive phase
- Results in a "base model" that understands language but doesn't follow instructions

### 2. **Supervised Fine-Tuning (SFT)**
- Fine-tunes the base model on instruction-following datasets
- Teaches the model to respond to prompts and follow instructions
- Much faster than base training
- Results in an "instruction-tuned model" that can chat and follow directions

### 3. **Reinforcement Learning from Human Feedback (RLHF)**
- Further refines the model to align with human preferences
- Uses reward models or preference data to optimize responses
- Focuses on making outputs more helpful, harmless, and honest
- Results in a "RLHF-aligned model" optimized for human preferences

**Typical Flow:**
```
Base Training → SFT → RLHF → Production Model
```

You can skip stages depending on your starting point:
- Starting from scratch? Do all three stages.
- Have a base model? Start with SFT.
- Have an SFT model? Jump to RLHF.

---

## Model Configuration

Before training, you need to configure your model architecture. This defines the structure and capacity of your neural network.

### Running Configuration

From the CLI, select option 1: "Configure new model"

### Architecture Parameters

#### **Tokenizer**
- **Parameter:** `tokenizer_name`
- **Type:** String (HuggingFace tokenizer identifier)
- **Default:** `"Qwen/Qwen2.5-0.5B"`
- **What it does:** Defines how text is converted to tokens (numbers) the model can process
- **How to choose:**
  - Use a tokenizer that matches your target language(s)
  - Popular options: `gpt2`, `meta-llama/Llama-2-7b-hf`, `Qwen/Qwen2.5-0.5B`
  - Vocab size affects model size and performance

#### **Positional Encoding**
- **Parameter:** `positional_encoding`
- **Options:** `sinusoidal`, `rope`, `alibi`, `yarn`
- **Default:** `rope`
- **What it does:** Tells the model about token positions in the sequence
- **Comparison:**
  - **`sinusoidal`**: Classic Transformer approach, additive, learned positions
  - **`rope`** (Rotary Position Embedding): Modern standard, excellent extrapolation to longer sequences
  - **`alibi`** (Attention with Linear Biases): No explicit encoding, biases attention scores directly
  - **`yarn`** (Yet Another RoPE Extension): Enhanced RoPE with better long-context handling
- **Recommendation:** Use `rope` for most cases, `yarn` if you need very long contexts (>4K tokens)

#### **Attention Type**
- **Parameter:** `attention_type`
- **Options:** `mha`, `mqa`, `gqa`
- **Default:** `gqa`
- **What it does:** Determines how the model computes attention
- **Comparison:**
  - **`mha`** (Multi-Head Attention): Standard attention, each head has its own K/V projections
    - Most parameters, highest quality
    - Slowest inference
  - **`mqa`** (Multi-Query Attention): All heads share single K/V projection
    - Fewest parameters, fastest inference
    - Slightly lower quality
  - **`gqa`** (Grouped Query Attention): Middle ground, heads share K/V in groups
    - Balanced parameters and speed
    - Near-MHA quality with better efficiency
- **Recommendation:** Use `gqa` for best balance (2-4 KV heads is typical)

#### **Normalization Type**
- **Parameter:** `norm_type`
- **Options:** `layernorm`, `rmsnorm`
- **Default:** `rmsnorm`
- **What it does:** Normalizes activations for training stability
- **Comparison:**
  - **`layernorm`**: Classic approach, normalizes mean and variance
  - **`rmsnorm`**: Simpler, only normalizes scale (root mean square)
    - Faster, fewer parameters
    - Modern models prefer this
- **Recommendation:** Use `rmsnorm` unless you have a specific reason not to

#### **Feed-Forward Activation**
- **Parameter:** `activation`
- **Options:** `relu`, `gelu`, `silu`, `leaky_relu`, `swiglu`
- **Default:** `swiglu`
- **What it does:** Non-linear activation function in feed-forward layers
- **Comparison:**
  - **`relu`**: Classic, fast but limited expressiveness
  - **`gelu`**: Smooth approximation, better than ReLU
  - **`silu`** (SiLU/Swish): Smooth, self-gated, good performance
  - **`leaky_relu`**: Fixes dying ReLU problem
  - **`swiglu`**: Gated variant of SiLU, state-of-the-art
    - Used in Llama, PaLM, and other top models
    - 1.5x more parameters but worth it
- **Recommendation:** Use `swiglu` for best performance

### Model Size Parameters

#### **Embedding Dimension (`d_model`)**
- **Type:** Integer
- **Default:** `896`
- **What it does:** Size of token embeddings and hidden states throughout the model
- **Constraints:** Must be divisible by `n_heads`
- **Typical values:**
  - Small models: 512-1024
  - Medium models: 1024-2048
  - Large models: 2048-8192
- **Impact:** Higher = more capacity but more computation

#### **Number of Attention Heads (`n_heads`)**
- **Type:** Integer
- **Default:** `14`
- **What it does:** Number of parallel attention computations
- **Constraints:**
  - `d_model` must be divisible by `n_heads`
  - For GQA: `n_heads` must be divisible by `n_kv_heads`
- **Typical values:** 8-64 (often a power of 2)
- **Impact:** More heads = more attention patterns learned

#### **Number of KV Heads (`n_kv_heads`)** *(GQA only)*
- **Type:** Integer
- **Default:** `2`
- **What it does:** Number of key/value heads in Grouped Query Attention
- **Constraints:** `n_heads` must be divisible by this
- **Typical values:** 2-8 (much less than `n_heads`)
- **Impact:** Lower = faster inference, fewer parameters

#### **Feed-Forward Dimension (`d_ff`)**
- **Type:** Integer
- **Default:** `4864`
- **What it does:** Hidden dimension in feed-forward layers
- **Typical values:** Usually 4x `d_model` (for SwiGLU, use ~2.67x due to gating)
- **Impact:** Larger = more expressive but more parameters

#### **Number of Layers (`n_layers`)**
- **Type:** Integer
- **Default:** `24`
- **What it does:** Number of transformer blocks stacked
- **Typical values:**
  - Small: 12-18
  - Medium: 24-32
  - Large: 40-80+
- **Impact:** More layers = deeper reasoning but slower training

#### **Vocabulary Size (`vocab_size`)**
- **Type:** Integer
- **Default:** `151936` (Qwen tokenizer size)
- **What it does:** Number of unique tokens the model can represent
- **Note:** Automatically set based on tokenizer, rarely needs manual configuration

#### **Maximum Sequence Length (`max_seq_len`)**
- **Type:** Integer
- **Default:** `1024`
- **What it does:** Maximum number of tokens the model can process at once
- **Typical values:** 512, 1024, 2048, 4096, 8192
- **Impact:** Longer = more context but quadratic memory growth

#### **Dropout Rate (`dropout`)**
- **Type:** Float (0.0 - 1.0)
- **Default:** `0.0`
- **What it does:** Randomly drops neurons during training for regularization
- **Recommendation:**
  - Use 0.0 for large datasets (dropout not needed)
  - Use 0.1 for smaller datasets to prevent overfitting

### Advanced Parameters

#### **Sliding Window (`sliding_window`)**
- **Type:** Integer or `None`
- **Default:** `None`
- **What it does:** Limits attention to a local window (like Mistral)
- **Use case:** Very long sequences with local attention patterns

#### **Attention Bias (`attention_bias`)**
- **Type:** Boolean
- **Default:** `False`
- **What it does:** Adds learnable bias to attention projections
- **Note:** Most modern models don't use this

#### **Normalization Epsilon (`norm_eps`)**
- **Type:** Float
- **Default:** `1e-6`
- **What it does:** Small constant added to normalization for numerical stability
- **Note:** Rarely needs changing

### Parameter Count Estimation

The tool automatically estimates total parameters based on your configuration:
- Smaller models: 100M - 1B parameters
- Medium models: 1B - 10B parameters
- Large models: 10B+ parameters

**Memory estimation (training):**
- ~4 bytes per parameter (fp32) or ~2 bytes (fp16/bf16)
- Training needs 4-6x model size (optimizer states, gradients, activations)
- Example: 1B parameter model needs ~12-24GB VRAM for training

### Saving Configuration

Configurations are saved as JSON files (default: `model_config.json`) and can be loaded later.

---

## Base Training (Pretraining)

Base training teaches your model language from scratch using large text datasets.

### Running Base Training

From the CLI, select option 2: "Base training"

You can either:
1. Use an existing training config
2. Configure new training

### Training Parameters

#### **Maximum Steps (`max_steps`)**
- **Type:** Integer
- **Default:** `10000`
- **What it does:** Total number of training steps
- **How to choose:**
  - Small models: 10,000 - 50,000 steps
  - Medium models: 50,000 - 500,000 steps
  - Large models: 500,000 - 2,000,000+ steps
- **Note:** One step = one batch update (affected by gradient accumulation)

#### **Optimizer (`optimizer`)**
- **Options:** `adamw`, `adafactor`, `lion`, `sophia`, `muon`
- **Default:** `adamw`
- **What each does:**
  - **`adamw`**: Adam with decoupled weight decay, proven and reliable
    - Best for: Most use cases, well-understood hyperparameters
    - **Parameters:**
      - `adamw_beta1` (default: 0.9) - Exponential decay rate for first moment
      - `adamw_beta2` (default: 0.999) - Exponential decay rate for second moment
      - `adamw_eps` (default: 1e-8) - Small constant for numerical stability
  - **`adafactor`**: Memory-efficient alternative to Adam
    - Best for: Large models with limited VRAM
    - **Parameters:** No additional parameters (uses defaults from transformers library)
  - **`lion`**: Newer optimizer, often faster convergence with less memory
    - Best for: Experimental, when you want faster training
    - **Parameters:**
      - `lion_beta1` (default: 0.9) - Momentum for EMA of gradients
      - `lion_beta2` (default: 0.99) - Momentum for update direction
  - **`sophia`**: Second-order optimizer using Hessian information
    - Best for: When you have compute budget for better optimization
    - **Parameters:**
      - `sophia_beta1` (default: 0.965) - Momentum for first moment
      - `sophia_beta2` (default: 0.99) - Momentum for Hessian diagonal estimate
      - `sophia_rho` (default: 0.04) - Clipping threshold for updates
  - **`muon`**: Momentum-based optimizer with orthogonalization via Newton-Schulz iteration
    - Best for: Alternative to AdamW with different convergence properties
    - **Parameters:**
      - `muon_momentum` (default: 0.95) - Momentum coefficient
      - `muon_nesterov` (default: True) - Use Nesterov momentum
    - **Note:** Muon applies to 2D parameters; 1D parameters use AdamW with 0.1x learning rate
- **Recommendation:** Start with `adamw`, experiment with others if needed

**Important:** The CLI will only prompt for parameters relevant to your selected optimizer. All parameters can also be manually edited in the configuration JSON file.

#### **Learning Rate (`lr`)**
- **Type:** Float
- **Default:** `3e-4`
- **What it does:** Step size for parameter updates
- **Typical ranges:**
  - Small models: 1e-3 to 5e-4
  - Medium models: 3e-4 to 1e-4
  - Large models: 1e-4 to 3e-5
- **Impact:** Too high = unstable training; too low = slow convergence
- **Recommendation:** Start with 3e-4 and adjust based on loss curves

#### **Weight Decay (`weight_decay`)**
- **Type:** Float
- **Default:** `0.1`
- **What it does:** L2 regularization on parameters
- **Typical values:** 0.01 to 0.1
- **Impact:** Prevents overfitting, adds small penalty for large weights
- **Recommendation:** 0.1 is standard for most models

#### **Learning Rate Scheduler (`scheduler`)**
- **Options:** `none`, `cosine`, `linear`, `polynomial`
- **Default:** `cosine`
- **What each does:**
  - **`none`**: Constant learning rate throughout training
  - **`cosine`**: Smoothly decays LR following cosine curve
    - Most popular, smooth convergence
  - **`linear`**: Linear decay from initial to 0
    - Simple, predictable
  - **`polynomial`**: Polynomial decay
    - Flexible middle ground
- **Recommendation:** Use `cosine` for most cases

#### **Warmup Steps (`warmup_steps`)**
- **Type:** Integer
- **Default:** `1000`
- **What it does:** Number of steps to linearly increase LR from 0 to target
- **Why it matters:** Prevents instability at training start
- **Typical values:** 500 - 5000 (about 1-10% of total steps)
- **Recommendation:** Use 1000-2000 for most models

#### **Batch Size (`batch_size`)**
- **Type:** Integer
- **Default:** `1`
- **What it does:** Number of sequences processed simultaneously per device
- **Constraints:** Limited by VRAM
- **Typical values:** 1-16 per GPU
- **Impact:** Larger batches = more stable gradients but more memory
- **Note:** Combined with gradient accumulation for effective larger batches

#### **Gradient Accumulation Steps (`gradient_accumulation_steps`)**
- **Type:** Integer
- **Default:** `64`
- **What it does:** Accumulate gradients over N batches before updating
- **Effective batch size** = `batch_size × gradient_accumulation_steps`
- **Why use it:** Simulates large batch training without the memory cost
- **Typical values:** 8-128
- **Example:** batch_size=1, accumulation=64 → effective batch size = 64

#### **Gradient Clipping (`grad_clip`)**
- **Type:** Float
- **Default:** `1.0`
- **What it does:** Maximum norm for gradients, prevents exploding gradients
- **Typical values:** 0.5 - 2.0
- **Impact:** Essential for training stability
- **Recommendation:** Keep at 1.0 unless you see gradient explosions

#### **Evaluation Frequency (`eval_every`)**
- **Type:** Integer
- **Default:** `500`
- **What it does:** Evaluate validation loss every N steps
- **Typical values:** 100-1000
- **Impact:** More frequent = better monitoring but slower training

#### **Evaluation Steps (`eval_steps`)**
- **Type:** Integer
- **Default:** `100`
- **What it does:** Number of validation batches to evaluate
- **Typical values:** 50-200
- **Impact:** More steps = better estimate but slower evaluation

#### **Save Best Only (`save_best_only`)**
- **Type:** Boolean
- **Default:** `True`
- **What it does:** Only save checkpoint when validation loss improves
- **Recommendation:** Keep True to save disk space

### Dataset Configuration

Datasets are loaded from HuggingFace and can be interleaved with custom weights.

**Format:** `dataset_name | subset (optional) | weight (optional)`

**Examples:**
```
HuggingFaceFW/fineweb-edu
HuggingFaceFW/fineweb-2 | fra_Latn | 1.0
HuggingFaceFW/fineweb-2 | spa_Latn | 2.0
```

**Multiple datasets:** Automatically interleaved based on weights
- Weights are **relative**, not absolute percentages
- Higher weight = sampled more frequently
- Useful for balancing languages or domains

**Weight Examples:**
- `[1.0, 1.0]` = 50/50 split between two datasets
- `[2.0, 1.0]` = 66.7% first dataset, 33.3% second dataset
- `[3.0, 1.0]` = 75% first dataset, 25% second dataset
- If weight is omitted, it defaults to 1.0

**How it works:**
1. All weights are summed (e.g., [2.0, 1.0] → total = 3.0)
2. Each weight is divided by the total to get probability (e.g., 2.0/3.0 = 0.667, 1.0/3.0 = 0.333)
3. Datasets are sampled according to these probabilities during training

**Default dataset:** `HuggingFaceFW/fineweb-edu` (high-quality educational content)

### Checkpointing and Resuming

The tool automatically saves:
- Best model checkpoint based on validation loss
- Model architecture configuration
- Training configuration

**To resume training:**
1. Select resume from checkpoint
2. Provide checkpoint path
3. Optionally extend training beyond original `max_steps`
4. Choose whether to load optimizer state (say 'no' if switching optimizers)

**Important:** Resuming loads the model weights and continues from the saved step count.

---

## Supervised Fine-Tuning (SFT)

SFT teaches your base model to follow instructions and engage in dialogue.

### When to Use SFT

- You have a base model that understands language but doesn't follow instructions
- You want to teach specific task formats (Q&A, chat, coding, etc.)
- You need the model to respond coherently to prompts

### Running SFT

From the CLI, select option 3: "SFT training (Supervised Fine-Tuning)"

### SFT Parameters

#### **Base Model Checkpoint (`policy_checkpoint`)**
- **Type:** String (path)
- **Default:** `"checkpoints/best_model.pt"`
- **What it does:** Path to your pretrained base model
- **Note:** Only model weights are loaded; optimizer state is reset

#### **Batch Size (`batch_size`)**
- **Type:** Integer
- **Default:** `4`
- **What it does:** Sequences per batch
- **Typical values:** 1-8 (SFT uses longer sequences than base training)
- **Note:** Lower than base training due to longer sequences

#### **Gradient Accumulation (`gradient_accumulation_steps`)**
- **Type:** Integer
- **Default:** `16`
- **What it does:** Same as base training
- **Effective batch size:** Often 16-128 for SFT

#### **Maximum Steps (`max_steps`)**
- **Type:** Integer
- **Default:** `5000`
- **What it does:** Total SFT training steps
- **Typical values:** 2,000 - 10,000 steps
- **Note:** Much shorter than base training (you're fine-tuning, not learning from scratch)

#### **Learning Rate (`learning_rate`)**
- **Type:** Float
- **Default:** `5e-6`
- **What it does:** Learning rate for fine-tuning
- **Important:** Usually 10-100x smaller than base training!
- **Typical values:** 1e-6 to 1e-5
- **Why smaller:** Prevents catastrophic forgetting of base knowledge

#### **Weight Decay (`weight_decay`)**
- **Type:** Float
- **Default:** `0.01`
- **What it does:** Same as base training
- **Note:** Often smaller than base training (0.01 vs 0.1)

#### **Max Gradient Norm (`max_grad_norm`)**
- **Type:** Float
- **Default:** `1.0`
- **What it does:** Gradient clipping threshold

#### **Optimizer (`optimizer`)**
- **Options:** Same as base training (`adamw`, `muon`, `lion`, `sophia`, `adafactor`)
- **Default:** `adamw`
- **Parameters:** See [Base Training Optimizer Parameters](#optimizer-optimizer) for details on each optimizer's specific parameters
  - The CLI will prompt for parameters relevant to your chosen optimizer
  - All parameters can be manually edited in `sft_config.json`
- **Recommendation:** AdamW works well for SFT

#### **Scheduler (`scheduler`)**
- **Options:** `none`, `cosine`, `linear`, `polynomial`
- **Default:** `cosine`
- **Recommendation:** Cosine with short warmup

#### **Warmup Steps (`warmup_steps`)**
- **Type:** Integer
- **Default:** `100`
- **What it does:** LR warmup period
- **Note:** Shorter than base training (100-500 steps)

#### **Logging and Checkpointing**
- **`log_every`**: Log training metrics every N steps (default: 10)
- **`save_every`**: Save checkpoint every N steps (default: 500)
- **`eval_every`**: Evaluate every N steps (default: 500)
- **`eval_steps`**: Number of validation steps (default: 50)
- **`save_best_only`**: Only save best checkpoint (default: True)

#### **Output Directory (`output_dir`)**
- **Type:** String
- **Default:** `"sft_checkpoints"`
- **What it does:** Where to save SFT checkpoints

### SFT Dataset Format

SFT datasets must have instruction-response pairs. Common formats:

**Format 1: Messages/Conversations**
```json
{
  "messages": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

**Format 2: Instruction/Response**
```json
{
  "instruction": "Explain Python",
  "response": "Python is a programming language..."
}
```

**Format 3: Prompt/Completion**
```json
{
  "prompt": "What is Python?",
  "completion": "Python is a programming language..."
}
```

### Recommended SFT Datasets

- **`HuggingFaceTB/smoltalk2`**: High-quality conversational data
- **`OpenAssistant/oasst1`**: Community-created instruction data
- **`tatsu-lab/alpaca`**: Instruction-following dataset
- **`HuggingFaceH4/ultrachat_200k`**: Large-scale chat dataset

### SFT Best Practices

1. **Start with a good base model** - Your base model should already understand language
2. **Use low learning rates** - 10-100x lower than base training to preserve base knowledge
3. **Don't overtrain** - 5,000-10,000 steps is often enough
4. **Monitor validation loss** - Stop when it starts increasing (overfitting)
5. **Test your model** - Try inference frequently to ensure quality

---

### LoRA for SFT

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that reduces memory requirements and speeds up training by only training small adapter matrices instead of the full model.

#### **When to Use LoRA**

- You have limited VRAM (can't fit full model + optimizer states)
- You want faster iteration cycles
- You want to create multiple task-specific adapters from the same base model
- You need to save disk space (adapters are much smaller than full checkpoints)

#### **LoRA Parameters**

**Enable LoRA (`use_lora`)**
- **Type:** Boolean
- **Default:** `False`
- **What it does:** Enables LoRA training instead of full fine-tuning
- **Note:** CLI will prompt for this option

**LoRA Preset (`lora_preset`)**
- **Options:** `minimal`, `attention_only`, `ffn_only`, `all`, `custom`
- **Default:** `minimal`
- **What each does:**
  - **`minimal`**: Q and V projections only
    - Lightest option, often sufficient for many tasks
    - ~5-10% of full model parameters
  - **`attention_only`**: All attention projections (Q, K, V, output)
    - More comprehensive attention adaptation
    - ~10-20% of full model parameters
  - **`ffn_only`**: Feed-forward layers only (gate, up, down projections)
    - Good for task-specific knowledge
    - ~15-30% of full model parameters
  - **`all`**: Both attention and feed-forward layers
    - Maximum adaptation capability
    - ~25-40% of full model parameters
  - **`custom`**: Manually specify target modules
    - For advanced users who know exactly what to adapt
    - Specify module names (e.g., `["q_proj", "v_proj", "gate_proj"]`)

**LoRA Rank (`lora_r`)**
- **Type:** Integer
- **Default:** `8`
- **What it does:** Rank of the low-rank decomposition (size of adapter bottleneck)
- **Typical values:** 4-64
- **Impact:**
  - Lower (4-8): Less memory, faster, may limit adaptation capability
  - Medium (8-16): Good balance for most tasks
  - Higher (32-64): More expressive, closer to full fine-tuning quality
- **Recommendation:** Start with 8, increase if quality is insufficient

**LoRA Alpha (`lora_alpha`)**
- **Type:** Integer
- **Default:** `16`
- **What it does:** Scaling factor for LoRA updates
- **Typical values:** 8-32
- **Relationship:** Usually set to `2 × lora_r`
- **Impact:** Higher alpha = stronger LoRA influence
- **Recommendation:** Use `2 × lora_r` as default (e.g., r=8 → alpha=16)

**LoRA Dropout (`lora_dropout`)**
- **Type:** Float (0.0-1.0)
- **Default:** `0.05`
- **What it does:** Dropout applied to LoRA adapters for regularization
- **Typical values:** 0.0-0.1
- **Recommendation:** 0.05 is good default, use 0.0 for larger datasets

**LoRA Target Modules (`lora_target_modules`)** *(custom preset only)*
- **Type:** List of strings
- **Default:** `None`
- **What it does:** Manually specify which modules to apply LoRA to
- **Example:** `["q_proj", "v_proj", "gate_proj", "up_proj"]`
- **Note:** Only used when `lora_preset="custom"`

#### **LoRA Output**

When training with LoRA:
- **Full checkpoints** still contain base model + adapters (for resuming training)
- **Lightweight adapters** saved separately in `{output_dir}/best_lora_adapters/`
  - These are ~10-100MB instead of multiple GB
  - Can be loaded with PEFT/HuggingFace libraries
  - Can be merged back into base model using CLI tool

#### **LoRA Training Tips**

1. **Memory savings:** LoRA uses 30-60% less VRAM than full fine-tuning
2. **Quality trade-off:** Typically 95-99% of full fine-tuning quality
3. **Start minimal:** Try `minimal` preset first, increase if needed
4. **Learning rate:** Can use slightly higher LR than full fine-tuning (e.g., 1e-5 instead of 5e-6)
5. **Multiple adapters:** Train different adapters for different tasks from same base

#### **When NOT to Use LoRA**

- You have plenty of VRAM and want maximum quality
- You're doing extensive domain adaptation (full fine-tuning may be better)
- You're training the base model from scratch (LoRA is for fine-tuning only)

---

## RLHF Training

Reinforcement Learning from Human Feedback aligns your model with human preferences using reward models or preference data.

### When to Use RLHF

- You have an SFT model that follows instructions but could be more helpful/harmless
- You want to optimize for specific qualities (safety, helpfulness, truthfulness)
- You have preference data or access to reward models

### RLHF Algorithm Comparison

| Feature | PPO | DPO | GRPO |
|---------|-----|-----|------|
| **Complexity** | High | Low | Medium |
| **Speed** | Slow | Fast | Medium |
| **Sample Efficiency** | Low | High | High |
| **Stability** | Can be unstable | Very stable | Stable |
| **Needs** | Reward model | Reference model + preference data | Reward model |
| **Best For** | Maximum control | Simplicity & speed | Balance |

### Running RLHF

From the CLI, select option 4: "RLHF training (PPO/DPO/GRPO)"

---

### PPO (Proximal Policy Optimization)

PPO is the classic RLHF approach used by InstructGPT and ChatGPT.

#### **How PPO Works**

1. Generate responses from current policy
2. Score responses with reward model
3. Compute advantages (how good each response is)
4. Update policy to increase probability of high-reward responses
5. Repeat

#### **PPO-Specific Parameters**

**Policy Checkpoint (`policy_checkpoint`)**
- Path to your SFT model
- This is the model being optimized

**Reward Model (`reward_model_name`)**
- HuggingFace identifier for reward model
- **Recommended:** `OpenAssistant/reward-model-deberta-v3-large-v2`
- **What it does:** Scores generated responses for quality/safety/helpfulness
- **Note:** Reward model is frozen (not trained)

**PPO Epochs (`ppo_epochs`)**
- **Type:** Integer
- **Default:** `4`
- **What it does:** Number of optimization epochs per batch
- **Typical values:** 2-8
- **Impact:** More epochs = more optimization per batch but risk of overfitting

**Clip Range (`clip_range`)**
- **Type:** Float
- **Default:** `0.2`
- **What it does:** Limits how much policy can change per update (prevents collapse)
- **Typical values:** 0.1-0.3
- **Impact:** Smaller = more conservative updates

**Value Function Coefficient (`vf_coef`)**
- **Type:** Float
- **Default:** `0.1`
- **What it does:** Weight for value loss in total loss
- **Note:** This implementation uses simplified value function

**Gamma (`gamma`)**
- **Type:** Float
- **Default:** `1.0`
- **What it does:** Discount factor for future rewards
- **Range:** 0.9-1.0
- **Impact:** Lower = focuses on immediate rewards; higher = considers long-term

**GAE Lambda (`gae_lambda`)**
- **Type:** Float
- **Default:** `0.95`
- **What it does:** Controls bias-variance tradeoff in advantage estimation
- **Typical values:** 0.9-0.99
- **Impact:** Higher = smoother but potentially biased advantages

**Batch Size (`batch_size`)**
- **Type:** Integer
- **Default:** `128`
- **What it does:** Number of prompts per training batch
- **Note:** PPO uses larger batches than supervised training

**Mini-Batch Size (`mini_batch_size`)**
- **Type:** Integer
- **Default:** `32`
- **What it does:** Process batches in mini-batches to save VRAM
- **Constraint:** `batch_size` must be divisible by `mini_batch_size`

**Generation Parameters:**
- **`max_new_tokens`**: Maximum response length (default: 128)
- **`temperature`**: Sampling temperature (default: 1.0)
- **`top_k`**: Top-k sampling (default: 0 = disabled)
- **`top_p`**: Nucleus sampling (default: 1.0)

#### **PPO Training Tips**

1. **Start with a good SFT model** - PPO refines, doesn't create from scratch
2. **Watch reward curves** - Should increase steadily
3. **Monitor KL divergence** - Too high = policy drifting too far
4. **Batch size matters** - Larger batches = more stable but slower
5. **Be patient** - PPO is slow but powerful

---

### DPO (Direct Preference Optimization)

DPO is a simpler, more stable alternative to PPO that learns directly from preference pairs.

#### **How DPO Works**

1. Start with preference pairs (chosen vs rejected responses)
2. Compare policy's probability of chosen vs rejected
3. Update policy to increase probability of chosen, decrease rejected
4. No reward model needed - learns from preferences directly

#### **DPO-Specific Parameters**

**Policy Checkpoint (`policy_checkpoint`)**
- Your SFT model to optimize

**Reference Checkpoint (`reference_checkpoint`)**
- **Type:** String (path) or None
- **Default:** `None` (uses policy checkpoint)
- **What it does:** Frozen reference model to prevent drift
- **Recommendation:** Leave as None to use same model as reference

**Beta (`clip_range` reused)**
- **Type:** Float
- **Default:** `0.2`
- **What it does:** Controls strength of preference optimization
- **Typical values:** 0.1-0.5
- **Impact:** Higher = stronger optimization but risk of overfitting

**Batch Size (`batch_size`)**
- **Default:** `128`
- **Note:** Same as PPO

**Mini-Batch Size (`mini_batch_size`)**
- **Default:** `32`
- **Note:** Same as PPO

#### **DPO Dataset Requirements**

DPO requires preference datasets with `chosen` and `rejected` fields:

```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing uses quantum mechanics...",
  "rejected": "Quantum computing is just regular computing but faster..."
}
```

**Recommended datasets:**
- `Anthropic/hh-rlhf`: Human feedback on helpfulness and harmlessness
- `argilla/ultrafeedback-binarized-preferences`: Curated preference data

#### **DPO Training Tips**

1. **Quality over quantity** - Good preference data is crucial
2. **Lower learning rates** - Usually 1e-6 to 1e-5
3. **Monitor win rate** - % of time chosen > rejected
4. **Fast convergence** - Often 1,000-3,000 steps is enough
5. **Very stable** - Much less likely to collapse than PPO

---

### GRPO (Group Relative Policy Optimization)

GRPO is a newer approach that generates multiple responses per prompt and learns from group comparisons.

#### **How GRPO Works**

1. For each prompt, generate N responses (group)
2. Score all responses with reward model
3. Compare each response to group average
4. Update policy to increase probability of above-average responses
5. Repeat

**Key insight:** Learns what makes responses better *relative to alternatives*, not absolute reward values.

#### **GRPO-Specific Parameters**

**Policy Checkpoint (`policy_checkpoint`)**
- Your SFT model to optimize

**Reward Model (`reward_model_name`)**
- Same as PPO
- **Recommended:** `OpenAssistant/reward-model-deberta-v3-large-v2`

**Group Size (`group_size`)**
- **Type:** Integer
- **Default:** `4`
- **What it does:** Number of responses generated per prompt
- **Typical values:** 2-8
- **Impact:**
  - Larger = better exploration, more diverse comparisons
  - Larger = slower (generates more responses)
- **Recommendation:** 4 is a good balance

**GRPO Temperature (`grpo_temperature`)**
- **Type:** Float
- **Default:** `1.0`
- **What it does:** Sampling temperature for generating response groups
- **Typical values:** 0.7-1.2
- **Impact:** Higher = more diverse group (better exploration)
- **Recommendation:**
  - 1.0 for balanced exploration
  - 0.8 for more consistent responses
  - 1.2 for maximum diversity

**Batch Size (`batch_size`)**
- **Default:** `128`
- **Note:** Number of *prompts* (will generate `batch_size × group_size` total responses)

**Mini-Batch Size (`mini_batch_size`)**
- **Default:** `32`
- **Note:** Same mini-batching as PPO/DPO

**Training Epochs (`ppo_epochs`)**
- **Note:** Reused from PPO config, controls optimization iterations
- **Default:** `4`

#### **GRPO Advantages**

1. **No value function needed** - Simpler than PPO
2. **No reference model needed** - Simpler than DPO
3. **Sample efficient** - Learns from multiple responses per prompt
4. **Stable** - Group comparison is more robust than absolute rewards
5. **Fast** - Faster than PPO, similar to DPO

#### **GRPO Training Tips**

1. **Adjust group size** - Start with 4, increase if you have compute
2. **Temperature matters** - Higher diversity can improve learning
3. **Watch group variance** - If all responses too similar, increase temperature
4. **Monitor best-of-N** - Track quality of top response in each group
5. **Efficient exploration** - GRPO naturally explores via multiple samples

---

### LoRA for RLHF

LoRA can be used with any RLHF algorithm (PPO, DPO, or GRPO) to reduce memory requirements during reinforcement learning.

#### **When to Use LoRA in RLHF**

- Limited VRAM for policy optimization
- Want to experiment with different reward signals without full model training
- Creating multiple aligned variants from the same SFT model
- Faster iteration during RLHF experimentation

#### **LoRA Parameters for RLHF**

All LoRA parameters are identical to SFT (see [LoRA for SFT](#lora-for-sft)):
- `use_lora`: Enable LoRA (default: False)
- `lora_preset`: Which modules to adapt (default: "minimal")
- `lora_r`: Rank of adapters (default: 8)
- `lora_alpha`: Scaling factor (default: 16)
- `lora_dropout`: Regularization dropout (default: 0.05)
- `lora_target_modules`: Custom module list (if preset="custom")

#### **LoRA RLHF Workflow**

**Scenario 1: SFT with LoRA → RLHF with LoRA**
1. Train SFT with LoRA adapters
2. Continue RLHF with same LoRA configuration
3. Adapters are updated by policy gradients
4. Base model stays frozen throughout

**Scenario 2: SFT with LoRA → Merge → RLHF without LoRA**
1. Train SFT with LoRA adapters
2. Merge adapters into base model (CLI option 5)
3. Run RLHF on merged model (full fine-tuning)
4. Recommended if you have VRAM for RLHF

**Scenario 3: Full SFT → RLHF with LoRA**
1. Train SFT without LoRA (full model)
2. Run RLHF with LoRA to save memory
3. Only policy updates are through adapters
4. Good for memory-constrained RLHF

#### **Important Considerations**

**Policy Gradients with LoRA:**
- Only adapter parameters receive policy gradient updates
- Base model remains frozen during RLHF
- This can be beneficial (preserves SFT quality) or limiting (less flexibility)

**Reference Model (DPO only):**
- When using LoRA with DPO, the reference model is automatically the frozen base
- The policy model = base + adapters
- This naturally provides the reference vs. policy comparison DPO needs

**Reward Model (PPO/GRPO):**
- Reward model is always separate and frozen
- LoRA on policy doesn't affect reward model

#### **LoRA RLHF Training Tips**

1. **Rank selection:** RLHF may benefit from slightly higher rank (16-32) than SFT
2. **Preset choice:** `all` preset often works better for RLHF than minimal
3. **Learning rate:** Can use higher LR with LoRA since only adapters are updated
4. **Merging:** If training SFT with LoRA for RLHF later, test both merged and unmerged
5. **Quality:** LoRA RLHF typically achieves 90-95% of full RLHF quality

#### **When NOT to Use LoRA in RLHF**

- You have sufficient VRAM for full model RLHF
- You need maximum alignment quality
- You're doing extensive reward optimization (full model may converge better)

---

### Common RLHF Parameters

These apply to all three RLHF algorithms:

**Optimizer (`optimizer`)**
- **Options:** Same as base training and SFT (`adamw`, `muon`, `lion`, `sophia`, `adafactor`)
- **Default:** `adamw`
- **Parameters:** See [Base Training Optimizer Parameters](#optimizer-optimizer) for details on each optimizer's specific parameters
  - The CLI will prompt for parameters relevant to your chosen optimizer
  - All parameters can be manually edited in `rlhf_config.json`
- **Note:** Optimizer selection was added in recent version; older configs may not have this field

**Learning Rate (`learning_rate`)**
- **Default:** `1.4e-5`
- **Typical range:** 1e-6 to 5e-5
- **Note:** Usually lower than SFT

**Weight Decay (`weight_decay`)**
- **Default:** `0.0`
- **Note:** Often disabled for RLHF

**Max Gradient Norm (`max_grad_norm`)**
- **Default:** `1.0`
- **What it does:** Gradient clipping

**Max Steps (`max_steps`)**
- **Default:** `10000`
- **Typical values:** 5,000-20,000
- **Note:** RLHF converges faster than base training

**Logging:**
- **`log_every`**: Default 10
- **`save_every`**: Default 500
- **`eval_every`**: Default 500

**Dataset (`datasets`)**
- For PPO/GRPO: Any prompt dataset
- For DPO: Must have preference pairs
- **Default:** `Anthropic/hh-rlhf`

---

## Model Inference and Testing

Test your model at any stage of training.

### Running Inference

From the CLI, select option 5: "Test model (inference)"

### Parameters

**Checkpoint Path**
- Path to model checkpoint (e.g., `checkpoints/best_model.pt`)
- Can test base, SFT, or RLHF models

**Interactive Mode**
- Type prompts and get responses
- Ctrl+C to exit

### Generation Parameters

The inference system uses:
- **Temperature:** 0.7 (can be adjusted in code)
- **Top-p:** 0.9
- **Max tokens:** 256

### Evaluating Model Quality

**For Base Models:**
- Should produce coherent text
- May not follow instructions well
- Look for: grammar, coherence, factual knowledge

**For SFT Models:**
- Should follow instructions
- Should engage in dialogue
- Look for: instruction-following, helpfulness, coherence

**For RLHF Models:**
- Should be helpful, harmless, and honest
- Should refuse unsafe requests
- Should provide nuanced, high-quality responses
- Look for: safety, quality, alignment with preferences

---

## LoRA Adapter Merging

If you've trained with LoRA, you'll want to merge the adapters back into the base model for deployment or further training without LoRA.

### When to Merge LoRA Adapters

- **Before RLHF:** If you did SFT with LoRA but want full RLHF (not LoRA RLHF)
- **For deployment:** To create a single model file instead of base + adapters
- **For inference:** Merged models can be slightly faster for inference
- **For compatibility:** Some deployment frameworks work better with merged models

### Running Merge Tool

From the CLI, select option 5: "Merge LoRA adapters"

### Input Options

**Option 1: Adapter Folder (Recommended)**
- **Input:** Path to lightweight LoRA adapter folder (e.g., `sft_checkpoints/best_lora_adapters/`)
- **Base model:** Path to original base model checkpoint
- **What happens:**
  1. Loads base model weights
  2. Loads LoRA adapters from folder
  3. Merges adapters into base model weights
  4. Saves merged checkpoint

**Option 2: Full Checkpoint**
- **Input:** Path to full checkpoint containing base + adapters (e.g., `sft_checkpoints/best_model.pt`)
- **What happens:**
  1. Loads full checkpoint
  2. Detects LoRA parameters
  3. Merges adapters into base weights
  4. Saves merged checkpoint

### Merge Process

The merge process:
1. **Loads base model** - Creates the base transformer model
2. **Loads LoRA adapters** - Either from adapter folder or full checkpoint
3. **Merges weights** - Mathematically combines base weights with LoRA updates
   - Formula: `W_merged = W_base + (LoRA_A × LoRA_B) × (alpha/r)`
4. **Saves merged model** - Checkpoint with merged weights (no LoRA parameters)

### Output

**Merged checkpoint contains:**
- Merged model weights (base + adapters combined)
- Model configuration
- Training metadata (step count, metrics, etc.)

**What's removed:**
- LoRA-specific parameters (lora_A, lora_B matrices)
- LoRA configuration
- Adapter-specific metadata

**File size:**
- Same as base model (LoRA adapters are merged, not added)
- Can delete adapter files after merging if desired

### Using Merged Models

**For RLHF:**
```
1. Train SFT with LoRA → sft_checkpoints/best_model.pt
2. Merge adapters → sft_checkpoints/best_model_merged.pt
3. Use merged checkpoint for RLHF training
```

**For deployment:**
```
1. Train SFT/RLHF with LoRA → checkpoints/best_model.pt
2. Merge adapters → checkpoints/best_model_merged.pt
3. Deploy merged checkpoint (no adapter loading needed)
```

**For inference:**
```
1. Merge adapters → model_merged.pt
2. Use CLI inference tool with merged checkpoint
3. Slightly faster than loading base + adapters separately
```

### Important Notes

1. **Irreversible:** Merging cannot be undone - keep original adapters if you might need them
2. **Same quality:** Merged model has identical quality to base + adapters
3. **No training needed:** Merge is a mathematical operation, not training
4. **Checkpoint size:** Merged checkpoint is same size as base model
5. **Multiple adapters:** Can't merge multiple adapters into one model (they'd interfere)

### Troubleshooting Merge

**"Checkpoint doesn't have LoRA parameters"**
- You're trying to merge a model that wasn't trained with LoRA
- Or the checkpoint is already merged
- Solution: Check if you actually used LoRA during training

**"Could not load LoRA config"**
- Checkpoint missing SFTConfig or RLHFConfig metadata
- Solution: Use Option 1 (adapter folder) instead

**"Adapter folder not found"**
- Path is incorrect or adapters weren't saved
- Solution: Check that LoRA training completed and saved adapters

---

## Best Practices and Tips

### General Training

1. **Start small** - Test your pipeline with a small model before scaling
2. **Monitor actively** - Watch loss curves, they tell you everything
3. **Save configs** - Always save configurations for reproducibility
4. **Checkpoint frequently** - Disk is cheap, lost training is expensive
5. **Use mixed precision** - Tool uses bfloat16 automatically for efficiency

### Learning Rate Selection

**Rule of thumb:**
- Base training: 3e-4 for small models, 1e-4 for large
- SFT: 10-100x smaller than base training
- RLHF: Similar to SFT or slightly higher

**Symptoms of wrong LR:**
- Too high: Loss explodes, NaN values
- Too low: Loss decreases very slowly
- Just right: Steady, smooth decrease

### Batch Size and Memory

**Effective batch size** = `batch_size × gradient_accumulation_steps`

**Recommendations:**
- Base training: Effective batch size 64-512
- SFT: Effective batch size 16-128
- RLHF: Effective batch size 64-256

**Memory management:**
- Increase `gradient_accumulation_steps` instead of `batch_size` to save VRAM
- Reduce `max_seq_len` if OOM
- Use `mini_batch_size` in RLHF to process large batches

### Dataset Selection

**Base training:**
- Use diverse, high-quality data
- `fineweb-edu` is excellent for education-focused models
- Mix domains for general models
- Dataset weights are relative: `[2.0, 1.0]` = 66.7%/33.3% split

**SFT:**
- Quality > quantity
- Match dataset to your use case (chat, code, Q&A, etc.)
- Mix datasets for versatility

**RLHF:**
- For PPO/GRPO: Simple prompts work fine
- For DPO: Quality of preferences is critical

### LoRA Usage

**When to use LoRA:**
- Limited VRAM (saves 30-60% memory)
- Fast iteration (smaller checkpoints)
- Multiple task variants from same base

**LoRA best practices:**
1. **Start minimal:** Use `minimal` preset first, increase if quality insufficient
2. **Rank selection:** 8 for SFT, 16-32 for RLHF
3. **Alpha = 2×rank:** Standard scaling (e.g., r=8 → alpha=16)
4. **Learning rate:** Can be slightly higher than full fine-tuning
5. **Merge for RLHF:** If doing SFT with LoRA → RLHF without LoRA, merge first

**LoRA quality expectations:**
- SFT with LoRA: 95-99% of full fine-tuning quality
- RLHF with LoRA: 90-95% of full RLHF quality
- Higher rank → closer to full quality

### Optimizer Selection

**By use case:**
- **General purpose:** AdamW (reliable, well-tested)
- **Memory constrained:** Adafactor or Lion
- **Experimental:** Muon or Sophia
- **Fastest convergence:** Lion (often)

**Parameters to tune:**
- AdamW: beta2 (try 0.95-0.999), eps (usually keep default)
- Lion: beta1 and beta2 (similar to AdamW betas)
- Muon: momentum (0.90-0.95), try both with/without Nesterov
- Sophia: rho for clipping (0.03-0.05)

### Training Duration

**Don't overtrain:**
- Base: Stop when loss plateaus
- SFT: 5-10K steps usually enough
- RLHF: 5-20K steps, watch for overfitting

**Validation is key:**
- Always monitor validation loss
- Stop when validation loss increases (overfitting)
- Test frequently with inference

---

## Troubleshooting

### Training Issues

**Loss is NaN**
- Cause: Learning rate too high, numerical instability
- Fix: Lower learning rate, check for bad data

**Loss not decreasing**
- Cause: Learning rate too low, bad initialization, bad data
- Fix: Increase LR, check dataset, verify model loaded correctly

**OOM (Out of Memory)**
- Cause: Batch too large, sequence too long, model too big
- Fix: Reduce `batch_size`, reduce `max_seq_len`, increase `gradient_accumulation_steps`

**Training very slow**
- Cause: Inefficient batching, large model, long sequences
- Fix: Increase `batch_size` (if memory allows), reduce evaluation frequency

### RLHF Issues

**PPO rewards not increasing**
- Cause: Wrong reward model, LR too low, policy too far from initialization
- Fix: Try different reward model, increase LR, restart from better SFT checkpoint

**DPO loss decreasing but quality not improving**
- Cause: Bad preference data, overfitting
- Fix: Check dataset quality, reduce training steps, lower beta

**GRPO responses all similar**
- Cause: Temperature too low, model too confident
- Fix: Increase `grpo_temperature`, check if model is overfitted

### Model Quality Issues

**Model repeating text**
- Cause: Insufficient training, repetition in dataset
- Fix: Train longer, filter dataset, adjust generation parameters

**Model refusing to respond**
- Cause: Over-alignment during RLHF, too much safety training
- Fix: Reduce RLHF training, adjust reward model

**Model hallucinating**
- Cause: Normal for LLMs, worse with insufficient training
- Fix: More training data, RLHF for factuality, prompt engineering

### LoRA Issues

**LoRA training not improving**
- Cause: Rank too low, wrong modules targeted, LR too low
- Fix: Increase `lora_r` (try 16 or 32), use `all` preset, increase learning rate

**LoRA quality worse than expected**
- Cause: Rank too low, insufficient training, wrong preset
- Fix: Increase rank to 16-32, train longer, try `all` preset instead of `minimal`

**Can't load LoRA adapters**
- Cause: Adapter files missing, wrong path, PEFT version mismatch
- Fix: Check adapter folder exists, verify path, update PEFT library

**Merge failed**
- Cause: Checkpoint doesn't have LoRA parameters, corrupted checkpoint
- Fix: Verify checkpoint was trained with LoRA, try Option 1 (adapter folder) instead

**LoRA using too much memory**
- Cause: Rank too high, too many modules targeted
- Fix: Reduce rank to 4-8, use `minimal` or `attention_only` preset

### Performance Tips

**Speed up training:**
1. Increase `batch_size` if memory allows
2. Reduce `eval_every` and `eval_steps`
3. Use fewer logging steps
4. Use faster optimizer (Lion vs AdamW)

**Improve quality:**
1. More training data (base training)
2. Better quality data (SFT/RLHF)
3. Longer training (up to a point)
4. Better hyperparameters (learning rate, batch size)

---

## Quick Reference

### Typical Hyperparameters

**Small Model (~100M params)**
```
Base Training:
- LR: 5e-4
- Batch size: 8, Accumulation: 16 (effective: 128)
- Steps: 10,000-50,000

SFT:
- LR: 1e-5
- Batch size: 4, Accumulation: 16 (effective: 64)
- Steps: 3,000-5,000

RLHF:
- LR: 2e-5
- Batch size: 64, Mini-batch: 16
- Steps: 5,000-10,000
```

**Medium Model (~1B params)**
```
Base Training:
- LR: 3e-4
- Batch size: 4, Accumulation: 32 (effective: 128)
- Steps: 100,000-500,000

SFT:
- LR: 5e-6
- Batch size: 2, Accumulation: 32 (effective: 64)
- Steps: 5,000-10,000

RLHF:
- LR: 1e-5
- Batch size: 128, Mini-batch: 32
- Steps: 10,000-20,000
```

**Large Model (~7B params)**
```
Base Training:
- LR: 1e-4
- Batch size: 1, Accumulation: 128 (effective: 128)
- Steps: 500,000+

SFT:
- LR: 1e-6
- Batch size: 1, Accumulation: 64 (effective: 64)
- Steps: 5,000-10,000

RLHF:
- LR: 5e-6
- Batch size: 64, Mini-batch: 16
- Steps: 10,000-20,000
```

---

Happy training! 🚀
