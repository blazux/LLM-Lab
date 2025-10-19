# 🧠 LLM-Lab

*A minimalistic tool to build, train, and possibly break your own LLM — locally, gloriously, and without safety nets.*

---

## 🧩 Requirements

This tool assumes you have:

- A working **CUDA environment** with `bfloat16` support (mandatory).  
- A **basic understanding of what you’re doing** (optional but recommended).  

If something goes wrong, you’ll know.  
If you don’t know, you’ll learn.  
That’s how science works.

Any mistake will result in a **beautiful, educational CUDA or Torch error**.  
Take it as feedback from the universe.

---

## 🧱 Model Configuration

Define what your model is made of — think of it as Lego for nerds.

### 🧬 Choose your components:
- **Tokenizer** → any on HuggingFace (you break it, you fix it)  
- **Positional Encoding** → `sinusoidal`, `rope`, `alibi`, `yarn`  
- **Normalization** → `layernorm`, `rmsnorm`  
- **Attention** → `Multi-Headed`, `Multi-Query`, `Grouped Query`  
- **Feed Forward Activation** → `ReLU`, `GeLU`, `Leaky ReLU`, `SiLU`, `SwiGLU`  

### ⚙️ Define model size:
- `d_model` (embedding size) must be divisible by the number of heads — because maths.  
- For Grouped Query Attention: number of attention heads must be divisible by `n_kv_heads`.  

Once your architecture is defined, the config is saved in **Hugging Face format**.

---

## 🔥 Base Training (Not calling it Pretraining to avoid killing motivation)

Welcome to the world of gradient descent and existential doubt.

### Configure your training:
- **Steps** → the more, the merrier (and slower).
- **Optimizer** → `AdamW`, `Adafactor`, `Lion`, `Sophia`, `Muon`.
  - Each optimizer has its own tunable parameters (betas, momentum, etc.) — the CLI will ask.
- **Scheduler** → `None`, `Cosine`, `Linear decay`, `Polynomial decay`.
- **Hyperparams** → learning rate, batch size, gradient accumulation, clipping, etc.
- **Datasets** → any on HuggingFace. Multiple datasets are automatically interleaved for you (you're welcome).
  - **Weights** are relative (e.g., `[1.0, 1.0]` = 50/50 split, `[2.0, 1.0]` = 66.7%/33.3%).

Then, press *train* and watch numbers go down.  
Or not.

---

## 🎓 SFT (Supervised Fine-Tuning)

Now that your model knows “language,” let’s teach it to **follow instructions** instead of screaming random tokens.

SFT fine-tunes your pretrained model on **instruction / dialogue datasets** — the kind where humans pretend to be helpful.

### Setup:
- **Base model** → checkpoint from pretraining (weights only; optimizer state is reset).  
- **Optimizer / Scheduler / Hyperparams** → same as pretraining (smaller learning rate is wise).  
- **Training steps** → ~5k–10k, depending on how stubborn your model is.  
- **Datasets** → things like `HuggingFaceTB/smoltalk2`, `OpenAssistant/oasst1`, or any instruction dataset.

The token counter resets. The suffering does not.

### 🔧 LoRA (Low-Rank Adaptation)

Don't have enough VRAM? **LoRA** lets you fine-tune efficiently by training small adapter matrices instead of the full model.

- **Enable LoRA** → CLI will ask if you want it (default: no).
- **Presets** → pick what to adapt:
  - `minimal` → Q and V projections only (lightweight, often enough)
  - `attention_only` → all attention projections (Q, K, V, O)
  - `ffn_only` → feed-forward layers only (gate, up, down)
  - `all` → everything that can be adapted
  - `custom` → you choose which modules (for the bold)
- **LoRA rank (r)** → bottleneck size (default: 8). Higher = more capacity, more VRAM.
- **LoRA alpha** → scaling factor (default: 16). Usually set to `2 × r`.
- **LoRA dropout** → regularization (default: 0.05).

After training, you get **lightweight adapter files** instead of full checkpoints.
Use the **"Merge LoRA adapters"** tool (CLI option 5) to bake them back into the base model for RLHF.

You save VRAM. You might lose a tiny bit of performance. That's the trade-off.

---

## 🧠 RLHF (Reinforcement Learning from Human Feedback)

Because supervised fine-tuning teaches "what to say,"
but **RLHF teaches "how to say it to please humans."**

Currently supports **PPO**, **DPO**, and **GRPO** — three paths to preference alignment, each with its own philosophy.

---

### 🥊 PPO (Proximal Policy Optimization)

The “gym bro” of training — all about steps, rewards, and regret minimization.

- **Reward model** → pick one from HuggingFace (`OpenAssistant/reward-model-deberta-v3-large-v2` is a good start).  
- **Batch / Mini-batch / Epochs / LR** → you know the drill.  
- **Gamma** → how much the model cares about the future (discount factor).  
- **Lambda** → how smooth the advantage estimation is (0.9 = impulsive, 1.0 = zen).  
- **Dataset** → as usual, any text dataset will do — just make sure it’s relevant.

Expect slow improvement, occasional reward spikes, and philosophical questions like *“is this actually working?”*

---

### 🧘 DPO (Direct Preference Optimization)

PPO's calmer, mathier cousin.
No sampling loops, no KL penalties — just clean, supervised gradients based on preferences.

- **Reference model** → usually your SFT model (or leave blank to use the same).
- **Batch / Mini-batch / LR / etc.** → same logic as before.
- **Beta** → controls how strongly the model obeys the reward (too high = robotic, too low = chaotic).
- **Dataset** → must include human preference pairs (good/bad responses).

DPO is simple, elegant, and 80% less likely to explode.
But that last 20% still exists.

---

### 🎯 GRPO (Group Relative Policy Optimization)

The efficient middle ground — simpler than PPO, more sample-efficient than both.
GRPO generates **multiple responses per prompt**, then learns from group comparisons.

- **Reward model** → same as PPO (e.g., `OpenAssistant/reward-model-deberta-v3-large-v2`).
- **Group size** → how many responses to generate per prompt (default: 4). More = better exploration, slower training.
- **GRPO temperature** → sampling temperature for generating response groups (default: 1.0). Higher = more diverse outputs.
- **Batch / Mini-batch / LR / etc.** → familiar territory.
- **Dataset** → any prompt dataset works (same as PPO).

The magic: instead of comparing to a value function (PPO) or a reference model (DPO),
GRPO compares each response to the **group average** — reinforcing what's relatively better.

Faster than PPO, no reference model needed like DPO, and surprisingly stable.
Your mileage may vary, but at least it won't take forever.

---

### 🔧 LoRA for RLHF

LoRA works for RLHF too — same concept, same trade-offs.

- **Available for PPO, DPO, and GRPO** → same presets and settings as SFT.
- **When to use it** → when your GPU is crying, or when you want faster iteration.
- **Merging adapters** → if you trained SFT with LoRA, merge first before RLHF (unless you're doing RLHF with LoRA too).

Training LoRA adapters during RLHF means only the adapter weights get updated by policy gradients.
The base model stays frozen, which can be good or bad depending on your philosophical stance.

---

## 🧪 Model Testing

Once your model has been pretrained, fine-tuned, reinforced, or not — at any point you want in fact... 
you can finally **talk to it**.

- Load your best checkpoint  
- Type a prompt  
- Watch your digital creation respond 

If it hallucinates, that’s part of the charm.

---

## 💭 Philosophy

- **No magic, no automation, no safety nets.**  
- You configure everything, you run everything, you break everything.  
- Every crash is a feature — a step toward enlightenment.  

> “You don’t need 10,000 GPUs to train a model.  
> You just need curiosity, patience, and a high pain tolerance.”

---

## ⚖️ License

MIT — because chaos should be open-source.