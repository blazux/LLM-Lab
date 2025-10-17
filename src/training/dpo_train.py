import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
import time
import os
from tqdm import tqdm
from typing import Optional

from model import TransformerLLM
from config import RLHFConfig
from data import load_tokenizer


def load_policy_model(checkpoint_path: str, device: torch.device, rlhf_config: Optional[RLHFConfig] = None):
    """Load policy model from checkpoint"""
    print(f"Loading policy model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config = checkpoint['model_config']
    model = TransformerLLM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Apply LoRA if configured
    if rlhf_config and rlhf_config.use_lora:
        print("\n🔧 Applying LoRA for parameter-efficient fine-tuning...")
        from model.lora_utils import apply_lora_to_model

        lora_config_dict = {
            'use_lora': rlhf_config.use_lora,
            'lora_preset': rlhf_config.lora_preset,
            'lora_target_modules': rlhf_config.lora_target_modules,
            'lora_r': rlhf_config.lora_r,
            'lora_alpha': rlhf_config.lora_alpha,
            'lora_dropout': rlhf_config.lora_dropout
        }

        model = apply_lora_to_model(model, model_config, lora_config_dict)
        print(f"   ✓ LoRA applied with preset: {rlhf_config.lora_preset}")

    model = model.to(device)

    tokenizer = load_tokenizer(model_config.tokenizer_name)

    print(f"✅ Policy model loaded ({model.count_parameters():,} parameters)")
    return model, tokenizer, model_config


def load_reference_model(checkpoint_path: str, device: torch.device):
    """Load reference model from checkpoint (for DPO)"""
    print(f"Loading reference model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config = checkpoint['model_config']
    model = TransformerLLM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Reference model is frozen

    print(f"✅ Reference model loaded ({model.count_parameters():,} parameters)")
    return model


def prepare_dataset(config: RLHFConfig, tokenizer):
    """Prepare streaming dataset for DPO"""
    print("Loading datasets...")

    datasets_list = []
    weights = []

    for ds_config in config.datasets:
        ds_name = ds_config['name']
        ds_subset = ds_config.get('subset', None)
        ds_split = ds_config.get('split', 'train')
        ds_weight = ds_config.get('weight', 1.0)

        print(f"  Loading: {ds_name}" + (f" ({ds_subset})" if ds_subset else ""))

        if ds_subset:
            ds = load_dataset(ds_name, ds_subset, split=ds_split, streaming=True)
        else:
            ds = load_dataset(ds_name, split=ds_split, streaming=True)

        datasets_list.append(ds)
        weights.append(ds_weight)

    # Interleave datasets if multiple
    if len(datasets_list) > 1:
        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        dataset = interleave_datasets(datasets_list, probabilities=probabilities)
    else:
        dataset = datasets_list[0]

    print(f"✅ Dataset prepared")
    return dataset


def compute_log_probs(model, tokenizer, prompts, responses, device, requires_grad=False):
    """Compute log probabilities for responses given prompts

    Args:
        requires_grad: If True, keeps gradients (for policy model). If False, no gradients (for reference model).
    """
    log_probs = []

    if not requires_grad:
        model.eval()

    # Get model's max sequence length
    max_seq_len = model.config.max_seq_len

    for prompt, response in zip(prompts, responses):
        # Tokenize response first to know its length
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        response_length = len(response_tokens)

        if response_length == 0:
            log_probs.append(torch.tensor(0.0, device=device, requires_grad=requires_grad))
            continue

        # If response itself is too long, truncate it
        if response_length >= max_seq_len - 2:  # Leave room for at least 2 prompt tokens
            response_tokens = response_tokens[-(max_seq_len - 2):]
            response_length = len(response_tokens)

        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

        # Truncate prompt if combined sequence is too long
        # Reserve space for response + 1 (for computing logits at response positions)
        max_prompt_length = max_seq_len - response_length - 1
        if len(prompt_tokens) > max_prompt_length and max_prompt_length > 0:
            # Keep the end of the prompt (more relevant context)
            prompt_tokens = prompt_tokens[-max_prompt_length:]
        elif max_prompt_length <= 0:
            # Edge case: response is too long, use minimal prompt
            prompt_tokens = prompt_tokens[:1] if len(prompt_tokens) > 0 else []

        # Combine tokens: prompt + response
        combined_tokens = prompt_tokens + response_tokens
        input_ids = torch.tensor([combined_tokens], dtype=torch.long, device=device)

        # Conditionally disable gradients for reference model
        if requires_grad:
            # Policy model - keep gradients
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
        else:
            # Reference model - no gradients
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)

        # Get logits for response tokens
        response_logits = logits[0, -response_length-1:-1, :]
        response_token_ids = torch.tensor(response_tokens, device=device)

        # Compute log probabilities
        log_probs_tensor = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs_tensor.gather(1, response_token_ids.unsqueeze(1)).squeeze()

        # Average log prob over tokens
        mean_log_prob = token_log_probs.mean()

        # CRITICAL: Detach for reference model to free memory
        if not requires_grad:
            mean_log_prob = mean_log_prob.detach()

        log_probs.append(mean_log_prob)

        # Free intermediate tensors
        del input_ids, logits, response_logits, log_probs_tensor, token_log_probs
        if not requires_grad:
            torch.cuda.empty_cache()

    return torch.stack(log_probs)


def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """
    Compute DPO loss

    DPO optimizes: E[log sigmoid(beta * (log(pi/pi_ref)(y_w) - log(pi/pi_ref)(y_l)))]
    where y_w is chosen/preferred response and y_l is rejected response
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    logits = beta * (policy_logratios - reference_logratios)

    loss = -F.logsigmoid(logits).mean()

    # Compute accuracy (how often policy prefers chosen over rejected)
    accuracy = (logits > 0).float().mean()

    return loss, accuracy


def dpo_update(
    policy_model,
    reference_model,
    optimizer,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    config: RLHFConfig,
    device
):
    """Perform DPO update with mini-batching to save VRAM"""
    policy_model.train()

    # Process in mini-batches to avoid OOM
    mini_batch_size = config.mini_batch_size
    batch_size = len(prompts)
    num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size

    total_loss = 0
    total_accuracy = 0
    optimizer.zero_grad()

    for i in range(num_mini_batches):
        start_idx = i * mini_batch_size
        end_idx = min(start_idx + mini_batch_size, batch_size)

        mini_prompts = prompts[start_idx:end_idx]
        mini_chosen = chosen_responses[start_idx:end_idx]
        mini_rejected = rejected_responses[start_idx:end_idx]

        # Compute reference log probs first (no gradients, can free immediately)
        reference_chosen_logps = compute_log_probs(
            reference_model, tokenizer, mini_prompts, mini_chosen, device, requires_grad=False
        ).detach()  # Detach immediately to free computation graph

        reference_rejected_logps = compute_log_probs(
            reference_model, tokenizer, mini_prompts, mini_rejected, device, requires_grad=False
        ).detach()  # Detach immediately

        # Clear CUDA cache after reference model computations
        torch.cuda.empty_cache()

        # Compute policy log probs (with gradients)
        policy_chosen_logps = compute_log_probs(
            policy_model, tokenizer, mini_prompts, mini_chosen, device, requires_grad=True
        )

        policy_rejected_logps = compute_log_probs(
            policy_model, tokenizer, mini_prompts, mini_rejected, device, requires_grad=True
        )

        # Compute DPO loss for this mini-batch
        loss, accuracy = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=config.clip_range
        )

        # Scale loss by number of mini-batches for gradient accumulation
        loss = loss / num_mini_batches

        # Backward pass (accumulate gradients)
        loss.backward()

        total_loss += loss.item()
        total_accuracy += accuracy.item()

        # Free memory
        del policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        torch.cuda.empty_cache()

    # Update weights after accumulating gradients from all mini-batches
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
    optimizer.step()

    return total_loss, total_accuracy / num_mini_batches


def train_dpo(config: RLHFConfig):
    """Main DPO training loop"""
    print("\n" + "=" * 60)
    print("RLHF Training with DPO")
    print("=" * 60 + "\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load policy model
    policy_model, tokenizer, model_config = load_policy_model(config.policy_checkpoint, device, config)

    # Load reference model (if not specified, use same checkpoint as policy)
    # Note: Reference model is NOT modified with LoRA (it stays frozen)
    reference_checkpoint = config.reference_checkpoint or config.policy_checkpoint
    reference_model = load_reference_model(reference_checkpoint, device)

    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    step = 0
    start_time = time.time()

    # Iterate through dataset
    dataset_iter = iter(dataset)

    try:
        while step < config.max_steps:
            step += 1

            # Collect batch of preference pairs
            prompts = []
            chosen_responses = []
            rejected_responses = []

            for _ in range(config.batch_size):
                try:
                    sample = next(dataset_iter)

                    # Extract prompt and responses from sample
                    # Support different dataset formats
                    prompt = None
                    chosen = None
                    rejected = None

                    if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
                        # Standard preference format
                        prompt = sample['prompt']
                        chosen = sample['chosen']
                        rejected = sample['rejected']
                    elif 'chosen' in sample and 'rejected' in sample:
                        # HH-RLHF format - chosen/rejected contain full conversations
                        # Extract prompt from chosen (everything before the last response)
                        chosen_parts = sample['chosen'].split('\n\n')
                        if len(chosen_parts) >= 2:
                            prompt = '\n\n'.join(chosen_parts[:-1])
                            chosen = chosen_parts[-1]
                        else:
                            prompt = ""
                            chosen = sample['chosen']

                        rejected_parts = sample['rejected'].split('\n\n')
                        if len(rejected_parts) >= 2:
                            rejected = rejected_parts[-1]
                        else:
                            rejected = sample['rejected']
                    else:
                        # Fallback: try to construct from available fields
                        prompt = sample.get('prompt', sample.get('question', sample.get('instruction', 'Hello')))
                        chosen = sample.get('chosen', sample.get('response', ''))
                        rejected = sample.get('rejected', sample.get('response', ''))

                    # Ensure all are strings and non-empty
                    if not isinstance(prompt, str) or not prompt.strip():
                        prompt = "Hello"
                    if not isinstance(chosen, str) or not chosen.strip():
                        chosen = "Yes"
                    if not isinstance(rejected, str) or not rejected.strip():
                        rejected = "No"

                    prompts.append(prompt.strip())
                    chosen_responses.append(chosen.strip())
                    rejected_responses.append(rejected.strip())

                except StopIteration:
                    # Reset iterator if we reach the end
                    dataset_iter = iter(dataset)
                    sample = next(dataset_iter)

                    # Same extraction logic
                    prompt = sample.get('prompt', 'Hello')
                    chosen = sample.get('chosen', 'Yes')
                    rejected = sample.get('rejected', 'No')

                    prompts.append(prompt.strip() if isinstance(prompt, str) else 'Hello')
                    chosen_responses.append(chosen.strip() if isinstance(chosen, str) else 'Yes')
                    rejected_responses.append(rejected.strip() if isinstance(rejected, str) else 'No')

            # Debug: print first sample on first step
            if step == 1:
                print(f"\n📝 Sample preference pair:")
                print(f"   Prompt: '{prompts[0][:100]}...'")
                print(f"   Chosen: '{chosen_responses[0][:100]}...'")
                print(f"   Rejected: '{rejected_responses[0][:100]}...'")

            # Perform DPO update
            loss, accuracy = dpo_update(
                policy_model,
                reference_model,
                optimizer,
                tokenizer,
                prompts,
                chosen_responses,
                rejected_responses,
                config,
                device
            )

            # Clear CUDA cache periodically to prevent memory fragmentation
            if step % 10 == 0:
                torch.cuda.empty_cache()

            # Logging
            if step % config.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                remaining_steps = config.max_steps - step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                print(f"Step {step}/{config.max_steps} | "
                      f"Loss: {loss:.4f} | "
                      f"Accuracy: {accuracy:.4f} | "
                      f"ETA: {eta_minutes:.1f}m")

            # Save checkpoint
            if step % config.save_every == 0:
                if config.use_lora:
                    # For LoRA training: only save adapters
                    adapter_dir = os.path.join(config.output_dir, f"lora_adapters_step_{step}")
                    policy_model.save_pretrained(adapter_dir)
                    print(f"💾 Saved LoRA adapters checkpoint (step {step})")

                    # Save metadata
                    import json
                    metadata = {
                        'step': step,
                        'base_checkpoint': config.policy_checkpoint,
                        'algorithm': 'dpo'
                    }
                    with open(os.path.join(adapter_dir, "training_metadata.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)
                else:
                    # For full fine-tuning: save complete checkpoint
                    checkpoint_path = os.path.join(config.output_dir, f"dpo_step_{step}.pt")
                    torch.save({
                        'step': step,
                        'model_state_dict': policy_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_config': model_config,
                        'rlhf_config': config,
                    }, checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")

            # Evaluation
            if step % config.eval_every == 0:
                print("\n" + "-" * 60)
                print("Sample Preference Pair:")
                print("-" * 60)
                print(f"Prompt: {prompts[0]}")
                print(f"Chosen: {chosen_responses[0]}")
                print(f"Rejected: {rejected_responses[0]}")
                print("-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    # Save final checkpoint
    if config.use_lora:
        # For LoRA training: only save adapters
        adapter_dir = os.path.join(config.output_dir, "final_lora_adapters")
        policy_model.save_pretrained(adapter_dir)
        print(f"💾 Saved final LoRA adapters")

        # Save metadata
        import json
        metadata = {
            'step': step,
            'base_checkpoint': config.policy_checkpoint,
            'algorithm': 'dpo'
        }
        with open(os.path.join(adapter_dir, "training_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        # For full fine-tuning: save complete checkpoint
        final_path = os.path.join(config.output_dir, "final_model.pt")
        torch.save({
            'step': step,
            'model_state_dict': policy_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': model_config,
            'rlhf_config': config,
        }, final_path)
        print(f"💾 Saved final model checkpoint: {final_path}")

    print(f"\n✅ Training complete!")
    print(f"Total steps: {step}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
