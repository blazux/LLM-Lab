import torch
import torch.nn.functional as F
from torch.amp import autocast

from model import TransformerLLM
from data import load_tokenizer


def load_model_for_inference(checkpoint_path: str):
    """Load model from checkpoint for inference (supports both base and RLHF models)"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Detect model type
    is_rlhf = 'rlhf_config' in checkpoint
    is_sft = 'sft_config' in checkpoint

    if is_rlhf:
        model_type = "RLHF-trained"
    elif is_sft:
        model_type = "SFT (instruction-tuned)"
    else:
        model_type = "Base (pretrained)"

    model_config = checkpoint['model_config']

    # Load tokenizer and validate vocab_size
    tokenizer = load_tokenizer(model_config.tokenizer_name)

    if model_config.vocab_size != tokenizer.vocab_size:
        print(f"   ⚠️  WARNING: Checkpoint vocab_size ({model_config.vocab_size}) != tokenizer vocab_size ({tokenizer.vocab_size})")
        print(f"   This indicates a tokenizer mismatch. Generation quality may be poor.")
        print(f"   Check that tokenizer_name in config matches the one used during training.")

    model = TransformerLLM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Print model information
    print(f"✅ Loaded {model_type} model")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Training step: {checkpoint.get('step', 'N/A')}")
    if is_rlhf:
        rlhf_config = checkpoint['rlhf_config']
        print(f"   Policy checkpoint: {rlhf_config.policy_checkpoint}")
        print(f"   Reward model: {rlhf_config.reward_model_name}")
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Device: {device}")

    return model, tokenizer, device


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    strategy: str = "top_p"
):
    """
    Generate text from a prompt

    Args:
        model: the language model
        tokenizer: the tokenizer
        device: torch device
        prompt: input text
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_k: top-k sampling parameter
        top_p: nucleus sampling parameter
        repetition_penalty: penalty for repeating tokens
        strategy: sampling strategy ("greedy", "top_k", "top_p", "beam")
    """
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_tokens = input_ids[0].tolist()

    for _ in range(max_tokens):
        # Get logits for next token
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)

        # Get logits for last position
        next_token_logits = logits[0, -1, :] / temperature

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                next_token_logits[token_id] /= repetition_penalty

        # Sampling strategies
        if strategy == "greedy":
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        elif strategy == "top_k":
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_idx]

        elif strategy == "top_p":
            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        else:  # default to top_p
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append to sequence
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Truncate if sequence gets too long
        max_len = model.config.max_seq_len
        if input_ids.size(1) > max_len:
            input_ids = input_ids[:, -max_len:]

    # Decode
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


def interactive_inference(checkpoint_path: str):
    """Interactive inference mode (supports base and RLHF-trained models)"""
    print("\n🤖 Loading model for inference...")
    model, tokenizer, device = load_model_for_inference(checkpoint_path)

    # Detect if this is an SFT/RLHF model (needs chat template)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_chat_model = 'sft_config' in checkpoint or 'rlhf_config' in checkpoint
    print()

    print("=" * 60)
    print("Interactive Inference Mode")
    if is_chat_model:
        print("Chat model detected - using chat template")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        prompt = input("Prompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        if not prompt.strip():
            continue

        # Get generation parameters
        try:
            max_tokens = int(input("Max tokens (default 100): ") or "100")
            temperature = float(input("Temperature (default 1.0): ") or "1.0")
            strategy = input("Strategy [greedy/top_k/top_p] (default top_p): ") or "top_p"

            if strategy == "top_k":
                top_k = int(input("Top-k (default 50): ") or "50")
                top_p = 1.0
            elif strategy == "top_p":
                top_k = 50
                top_p = float(input("Top-p (default 0.9): ") or "0.9")
            else:
                top_k = 50
                top_p = 0.9

            repetition_penalty = float(input("Repetition penalty (default 1.0): ") or "1.0")

        except ValueError:
            print("Invalid input, using defaults")
            max_tokens = 100
            temperature = 1.0
            strategy = "top_p"
            top_k = 50
            top_p = 0.9
            repetition_penalty = 1.0

        print("\n" + "=" * 60)
        print("Generating...")
        print("=" * 60 + "\n")

        # Apply chat template if this is a chat model
        if is_chat_model and hasattr(tokenizer, 'apply_chat_template'):
            # Format as chat message
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        generated_text = generate_text(
            model, tokenizer, device, formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            strategy=strategy
        )

        print(generated_text)
        print("\n" + "=" * 60 + "\n")

    print("\n👋 Exiting inference mode...")
