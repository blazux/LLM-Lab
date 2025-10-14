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
- **Scheduler** → `None`, `Cosine`, `Linear decay`, `Polynomial decay`.  
- **Hyperparams** → learning rate, batch size, gradient accumulation, clipping, etc.  
- **Datasets** → any on HuggingFace. Multiple datasets are automatically interleaved for you (you’re welcome).

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

---

## 🧠 RLHF (Reinforcement Learning from Human Feedback)

Because supervised fine-tuning teaches “what to say,”  
but **RLHF teaches “how to say it to please humans.”**

Currently supports **PPO** and **DPO**, the two main schools of reinforcement enlightenment.

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

PPO’s calmer, mathier cousin.  
No sampling loops, no KL penalties — just clean, supervised gradients based on preferences.

- **Reference model** → usually your SFT model (or leave blank to use the same).  
- **Batch / Mini-batch / LR / etc.** → same logic as before.  
- **Beta** → controls how strongly the model obeys the reward (too high = robotic, too low = chaotic).  
- **Dataset** → must include human preference pairs (good/bad responses).

DPO is simple, elegant, and 80% less likely to explode.  
But that last 20% still exists.

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