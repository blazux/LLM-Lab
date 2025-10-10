## 🧠 LLM-lab

A minimalistic tool to setup and train LLM.


### This tool will assume you have :

 - a proper CUDA configuration with `bfloat16` support (mandatory)
 - a basic understanding of what you're doing (optional)

Any mistake will end in CUDA or Torch error, but that's part of the fun.

### Model configuration :

Configure the component you want for your model :
 - Tokenizer : pick any on HuggingFace
 - Positional Encoding : `sinusoidal`, `rope`, `alibi`, `yarn`
 - Normalization : `layernorm`, `rmsnorm`
 - Attention : Multi-Headed, Multi-Query, Grouped Query
 - Feed forward : `ReLU`, `GeLU`, `Leaky ReLU`, `SiLU`, `SwiGLU` 

Then configure the size of the model :
 - Embedding (`d_model`) must be divisible by the number of attention heads
 - If you use Grouped Query Attention : number of attention heads must be divisible by KV heads `n_kv_heads` 

### Base training :

Configure the training :
 - Training steps : the higher the better
 - Optimizer : `AdamW`, `Adafactor`, `Lion`, `Sophia`, `Muon`
 - Scheduler : `None`, `Cosine`, `Linear decay with warmup`, `Polynomial decay` 
 - Learning rate / Batch size / Gradiant Accumulation / Gradiant clipping etc...
 - Datasets : pick any on HuggingFace, if you use more than one they'll be interleaved automatically

Start the training and wait until it's finished.

### RLHF Training :

RLHF support only PPO and DPO.

#### PPO :

RLHF will use Proximal Policy Optimization : 

 - Reward model : pick any on HuggingFace
 - Batch / Mini batch / epoch / etc ...
 - Gamma : Discount Factor
 - Lambda : Smoothing factor
 - Datasets : pick any on HuggingFace

#### DPO :

RLHF will use Direct Preference Optimization : 

 - Reference Model : use the base model as much as possible
 - Batch / Mini batch / learning rate / etc ...
 - Beta parameter : contrast temperature

### Model testing :

Talk to your trained model and enjoy !


