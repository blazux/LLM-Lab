import torch
import torch.nn.functional as F
from torch.amp import autocast

from model.factory import build_model
from data import load_tokenizer


def load_model_for_inference(checkpoint_path: str):
    """Load model from checkpoint for inference (supports both base and RLHF models)"""
    # Clean up any existing GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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

    # Load tokenizer
    tokenizer = load_tokenizer(model_config.tokenizer_name)

    # Note: SFT models now use text markers ("User:", "Assistant:") instead of special tokens
    # No tokenizer modifications needed - the model already knows these tokens

    model = build_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Convert to bfloat16 to match training and avoid dtype mismatch with autocast
    model = model.to(device=device, dtype=torch.bfloat16)
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
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.size(1)
    generated_tokens = input_ids[0].tolist()

    # Prefill: run full prompt through model to populate KV cache
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _, past_key_values = model(input_ids, use_cache=True)
    # logits: (1, prompt_len, vocab_size) — use last position for first token
    next_token_logits = logits[0, -1, :]

    for step in range(max_tokens):
        effective_temp = max(temperature, 1e-7) if strategy != "greedy" else 1.0
        scaled_logits = next_token_logits / effective_temp

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                if scaled_logits[token_id] < 0:
                    scaled_logits[token_id] *= repetition_penalty
                else:
                    scaled_logits[token_id] /= repetition_penalty

        # Sampling
        if strategy == "greedy":
            next_token = torch.argmax(scaled_logits, dim=-1, keepdim=True)
        elif strategy == "top_k":
            top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_idx]
        else:  # top_p (default)
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            scaled_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token.item())

        # Incremental decode: feed only the new token
        next_input = next_token.unsqueeze(0)  # (1, 1)
        max_len = model.config.max_seq_len
        if len(generated_tokens) >= max_len:
            break

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _, past_key_values = model(next_input, past_key_values=past_key_values, use_cache=True)
        next_token_logits = logits[0, 0, :]

    new_tokens = generated_tokens[prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


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
        if is_chat_model:
            # Use text markers matching SFT training format
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
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
