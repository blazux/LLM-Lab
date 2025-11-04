# LLM-Lab Documentation

Welcome to the complete documentation for LLM-Lab! This is your single entry point for all information about training and fine-tuning Large Language Models.

## 📖 Documentation Structure

### Main Documentation

- **[📘 Complete Training Guide](full-guide.md)** - Comprehensive guide covering all training stages
  - Model Configuration
  - Base Training (Pretraining)
  - Supervised Fine-Tuning (SFT)
  - RLHF (PPO, DPO, GRPO)
  - LoRA for parameter-efficient fine-tuning
  - Best practices and troubleshooting

### Specialized Guides

- **[Mamba2 State Space Models](mamba2.md)** - Alternative to Transformers with linear complexity
  - What is Mamba2?
  - Quick start guide
  - Performance benefits
  - LoRA support for Mamba2
  - Configuration examples

---

## Training Pipeline Overview

The typical LLM training workflow:

```
1. Configure Model
   ↓
2. Base Training (Pretraining)
   ↓
3. Supervised Fine-Tuning (SFT)
   ↓
4. RLHF Alignment
   ↓
5. Production Model
```

### Stage 1: Model Configuration

Define your architecture (Transformer or Mamba2), size, and components.

→ **[See Model Configuration section](full-guide.md#model-configuration)**

### Stage 2: Base Training

Train from scratch on large text corpora to learn language patterns.

→ **[See Base Training section](full-guide.md#base-training-pretraining)**

### Stage 3: Supervised Fine-Tuning (SFT)

Fine-tune on instruction datasets to follow directions.

→ **[See SFT section](full-guide.md#supervised-fine-tuning-sft)**

### Stage 4: RLHF

Align with human preferences using reinforcement learning.

→ **[See RLHF section](full-guide.md#rlhf-training)**

---

## Quick Navigation by Task

### "I want to..."

**...train a model from scratch**
1. [Configure your model](full-guide.md#model-configuration)
2. [Set up base training](full-guide.md#base-training-pretraining)
3. Start training!

**...fine-tune an existing checkpoint**
1. [Set up SFT](full-guide.md#supervised-fine-tuning-sft)
2. Optional: [Use LoRA for efficiency](full-guide.md#lora-for-sft)

**...align a model with RLHF**
1. [Choose an algorithm](full-guide.md#rlhf-training) (PPO, DPO, or GRPO)
2. [Configure preference data](full-guide.md#rlhf-training)

**...use Mamba2 instead of Transformers**
1. [Read about Mamba2](mamba2.md)
2. [Configure Mamba2 model](mamba2.md#quick-start)
3. Follow normal training pipeline

**...reduce memory usage**
1. [Enable LoRA](full-guide.md#lora-for-sft)
2. Use gradient checkpointing
3. Reduce batch size or sequence length

**...fix an error**
→ **[See Troubleshooting section](full-guide.md#troubleshooting)**

---

## Architecture Support

### Transformers
- Multi-Head Attention (MHA)
- Multi-Query Attention (MQA)
- Grouped Query Attention (GQA)
- Sliding window attention
- RoPE, ALiBi, YARN positional encodings

### Mamba2 State Space Models
- Linear O(N) complexity
- No attention mechanism
- Efficient long-context processing
- Same training pipeline as Transformers

→ **[See Mamba2 documentation](mamba2.md)**

---

## Need Help?

1. **Check [Troubleshooting](troubleshooting.md)** for common issues
2. **Read the relevant guide** from the links above
3. **Open an issue** on GitHub if you find a bug

---

## Contributing

Found an error in the docs? Want to improve something? Contributions are welcome!

Please open an issue or pull request on GitHub.
