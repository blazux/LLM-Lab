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
    prompt_length = input_ids.size(1)

    # Debug: Show input tokens
    print(f"DEBUG: Input to model ({prompt_length} tokens):")
    print(f"   Token IDs: {input_ids[0].tolist()[:20]}{'...' if prompt_length > 20 else ''}")
    print(f"   Last 5 tokens: {[tokenizer.decode([t]) for t in input_ids[0, -5:].tolist()]}")

    generated_tokens = input_ids[0].tolist()

    for step in range(max_tokens):
        # Get logits for next token
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)  # Model returns (logits, aux_loss)

        # Get logits for last position
        # Clamp temperature to avoid division by zero/near-zero
        effective_temp = max(temperature, 1e-7) if strategy != "greedy" else 1.0
        next_token_logits = logits[0, -1, :] / effective_temp

        # Debug: Show top 5 predictions for first token
        if step == 0:
            top5_logits, top5_indices = torch.topk(next_token_logits * effective_temp, 5)  # Undo temperature for display
            top5_probs = torch.softmax(top5_logits, dim=-1)
            print(f"DEBUG: Top 5 predictions for first generated token:")
            for i, (idx, prob) in enumerate(zip(top5_indices.tolist(), top5_probs.tolist())):
                token_str = tokenizer.decode([idx])
                print(f"   {i+1}. '{token_str}' (id={idx}, prob={prob:.3f})")

        # Apply repetition penalty
        # For positive logits: divide by penalty (makes smaller = less likely)
        # For negative logits: multiply by penalty (makes more negative = less likely)
        if repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
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

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = generated_tokens[prompt_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
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
        if is_chat_model:
            # Use text markers matching SFT training format
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
            # Debug: show tokenization
            debug_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
            print(f"DEBUG: Formatted prompt: {repr(formatted_prompt)}")
            print(f"DEBUG: Token IDs: {debug_tokens[:20]}{'...' if len(debug_tokens) > 20 else ''}")
            print(f"DEBUG: Decoded tokens: {[tokenizer.decode([t]) for t in debug_tokens[:10]]}")
            print()
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
