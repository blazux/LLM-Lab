import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, interleave_datasets
import time
import os
from tqdm import tqdm
from typing import Optional

from model.factory import build_model
from config import RLHFConfig
from data import load_tokenizer
from training.report import TrainingReport


def get_trend_indicator(current, previous):
    """Get trend indicator arrow for metrics"""
    if previous is None:
        return ""
    threshold = abs(previous * 0.001)
    diff = current - previous
    if abs(diff) < threshold:
        return "→"
    elif diff > 0:
        return "↑"
    else:
        return "↓"


def load_policy_model(checkpoint_path: str, device: torch.device, rlhf_config: Optional[RLHFConfig] = None):
    """Load policy model from checkpoint"""
    print(f"Loading policy model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config = checkpoint['model_config']
    model = build_model(model_config)
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


def load_reward_model(model_name: str, device: torch.device):
    """Load reward model from HuggingFace (PPO-specific)"""
    print(f"Loading reward model: {model_name}...")

    reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        dtype=torch.bfloat16
    ).to(device)
    reward_model.eval()

    print(f"✅ Reward model loaded")
    return reward_model, reward_tokenizer


def prepare_dataset(config: RLHFConfig, tokenizer):
    """Prepare streaming dataset for PPO"""
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


@torch.no_grad()
def compute_rewards(
    reward_model,
    reward_tokenizer,
    prompts,
    responses,
    device
):
    """Compute rewards for generated responses (PPO-specific)"""
    rewards = []

    for prompt, response in zip(prompts, responses):
        # Combine prompt and response
        full_text = prompt + response

        # Tokenize for reward model
        inputs = reward_tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        # Get reward score
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = reward_model(**inputs)
            reward = outputs.logits[0].item()

        rewards.append(reward)

        # Free memory
        del inputs, outputs

    return torch.tensor(rewards, device=device)


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts,
    config: RLHFConfig,
    device
):
    """Generate responses from policy model (PPO-specific)"""
    responses = []
    all_log_probs = []

    model.eval()

    for idx, prompt in enumerate(prompts):
        # Ensure prompt is valid
        if not prompt or not prompt.strip():
            prompt = "Hello, how are you?"

        # Tokenize prompt - ensure Long dtype
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device).long()

        # Skip if empty prompt (shouldn't happen now, but safety check)
        if input_ids.size(1) == 0:
            print(f"⚠️  Warning: Empty token sequence for prompt {idx}: '{prompt}'")
            # Try with a fallback prompt
            input_ids = tokenizer.encode("Hello", return_tensors="pt").to(device).long()
            if input_ids.size(1) == 0:
                # If even "Hello" doesn't work, skip
                responses.append("")
                all_log_probs.append([])
                continue

        generated_tokens = []
        log_probs = []

        current_input = input_ids

        for _ in range(config.max_new_tokens):
            # Ensure current_input is Long dtype before passing to model
            current_input = current_input.long()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(current_input)

            # Get logits for last position
            next_token_logits = logits[0, -1, :] / config.temperature

            # Apply top-k filtering if needed
            if config.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, config.top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p filtering if needed
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).long()

            # Compute log probability
            log_prob = torch.log(probs[next_token])
            log_probs.append(log_prob.item())

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token.item())
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1).long()

            # Truncate if too long
            if current_input.size(1) > model.config.max_seq_len:
                current_input = current_input[:, -model.config.max_seq_len:].long()

        # Decode response
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response)
        all_log_probs.append(log_probs)

    return responses, all_log_probs


def compute_advantages(rewards, values, gamma, gae_lambda):
    """Compute GAE advantages (PPO-specific)"""
    advantages = []
    returns = []

    gae = 0
    next_value = 0

    # Work backwards
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * gae_lambda * gae

        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])

        next_value = values[i]

    advantages = torch.tensor(advantages)
    returns = torch.tensor(returns)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def ppo_update(
    model,
    optimizer,
    tokenizer,
    prompts,
    responses,
    old_log_probs,
    rewards,
    advantages,
    config: RLHFConfig,
    device
):
    """Perform PPO update with mini-batching to save VRAM"""
    model.train()

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    n_updates = 0

    # Get mini-batch size
    mini_batch_size = config.mini_batch_size
    batch_size = len(prompts)

    for _ in range(config.ppo_epochs):
        # Process in mini-batches
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size

        for i in range(num_mini_batches):
            start_idx = i * mini_batch_size
            end_idx = min(start_idx + mini_batch_size, batch_size)

            mini_prompts = prompts[start_idx:end_idx]
            mini_responses = responses[start_idx:end_idx]
            mini_old_log_probs = old_log_probs[start_idx:end_idx]
            mini_rewards = rewards[start_idx:end_idx]
            mini_advantages = advantages[start_idx:end_idx]

            # Recompute log probs and values for current policy
            batch_log_probs = []
            batch_entropies = []

            for prompt, response in zip(mini_prompts, mini_responses):
                # Tokenize response first to know its length
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                response_length = len(response_tokens)

                if response_length == 0:
                    continue

                # Get model's max sequence length
                max_seq_len = model.config.max_seq_len

                # If response itself is too long, truncate it
                if response_length >= max_seq_len - 2:
                    response_tokens = response_tokens[-(max_seq_len - 2):]
                    response_length = len(response_tokens)

                # Tokenize prompt
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

                # Truncate prompt if combined sequence is too long
                max_prompt_length = max_seq_len - response_length - 1
                if len(prompt_tokens) > max_prompt_length and max_prompt_length > 0:
                    prompt_tokens = prompt_tokens[-max_prompt_length:]
                elif max_prompt_length <= 0:
                    prompt_tokens = prompt_tokens[:1] if len(prompt_tokens) > 0 else []

                # Combine tokens
                combined_tokens = prompt_tokens + response_tokens
                input_ids = torch.tensor([combined_tokens], dtype=torch.long, device=device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)

                # Get logits for response tokens
                response_logits = logits[0, -response_length-1:-1, :]
                response_token_ids = torch.tensor(response_tokens, device=device)

                # Compute log probabilities
                log_probs = F.log_softmax(response_logits, dim=-1)
                token_log_probs = log_probs.gather(1, response_token_ids.unsqueeze(1)).squeeze()
                mean_log_prob = token_log_probs.mean()

                # Compute entropy
                probs = F.softmax(response_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                batch_log_probs.append(mean_log_prob)
                batch_entropies.append(entropy)

                # Free intermediate tensors
                del input_ids, logits, response_logits, log_probs, token_log_probs

            if len(batch_log_probs) == 0:
                continue

            # Convert to tensors
            new_log_probs = torch.stack(batch_log_probs)
            entropies = torch.stack(batch_entropies)

            # Compute ratio
            old_log_probs_tensor = torch.tensor([sum(lp) / len(lp) if len(lp) > 0 else 0.0 for lp in mini_old_log_probs], device=device)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # Clipped surrogate loss
            surr1 = ratio * mini_advantages.to(device)
            surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * mini_advantages.to(device)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (simplified - using reward as value target)
            value_loss = 0

            # Entropy bonus
            entropy_loss = -entropies.mean()

            # Total loss
            loss = policy_loss + config.vf_coef * value_loss + 0.01 * entropy_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss if isinstance(value_loss, (float, int)) else value_loss.item()
            total_entropy += entropies.mean().item()
            n_updates += 1

            # Free memory after each mini-batch
            del new_log_probs, entropies, ratio, policy_loss, loss
            torch.cuda.empty_cache()

    avg_policy_loss = total_policy_loss / max(n_updates, 1)
    avg_value_loss = total_value_loss / max(n_updates, 1)
    avg_entropy = total_entropy / max(n_updates, 1)

    return avg_policy_loss, avg_value_loss, avg_entropy


def train_ppo(config: RLHFConfig, callback=None):
    """Main PPO training loop

    Args:
        config: RLHF configuration
        callback: Optional callback object with methods on_step, on_eval, on_log
    """
    print("\n" + "=" * 60)
    print("RLHF Training with PPO")
    print("=" * 60 + "\n")

    if callback and hasattr(callback, 'on_log'):
        callback.on_log("Starting PPO training...", "info")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    policy_model, tokenizer, model_config = load_policy_model(config.policy_checkpoint, device, config)
    reward_model, reward_tokenizer = load_reward_model(config.reward_model_name, device)

    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)

    # Setup optimizer
    from optimizers import setup_optimizer
    optimizers = setup_optimizer(policy_model, config)
    optimizer = optimizers[0]  # For RLHF, we always use single optimizer

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize training report
    report = TrainingReport(
        training_type="rlhf_ppo",
        model_config=model_config,
        training_config=config,
        output_dir=config.output_dir,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    step = 0
    start_time = time.time()

    # Track previous metrics for trend indicators
    prev_reward = None
    prev_policy_loss = None
    prev_entropy = None

    # Iterate through dataset
    dataset_iter = iter(dataset)

    try:
        while step < config.max_steps:
            step += 1

            # Collect batch of prompts
            prompts = []
            for _ in range(config.batch_size):
                try:
                    sample = next(dataset_iter)
                    # Extract prompt from sample (adjust based on dataset format)
                    prompt = None

                    if 'prompt' in sample:
                        prompt = sample['prompt']
                    elif 'text' in sample:
                        prompt = sample['text'][:200]  # Use first 200 chars as prompt
                    elif 'chosen' in sample:
                        # For HH-RLHF style datasets
                        prompt = sample['chosen'].split('\n\n')[0] if sample['chosen'] else ""
                    elif 'question' in sample:
                        prompt = sample['question']
                    elif 'instruction' in sample:
                        prompt = sample['instruction']
                    else:
                        # Default: use first key
                        first_value = list(sample.values())[0] if sample.values() else ""
                        prompt = str(first_value)[:200]

                    # Ensure prompt is a non-empty string
                    if not prompt or not isinstance(prompt, str):
                        prompt = "Hello"  # Fallback to simple prompt

                    # Strip whitespace
                    prompt = prompt.strip()
                    if not prompt:
                        prompt = "Hello"  # Fallback if empty after strip

                    prompts.append(prompt)

                except StopIteration:
                    # Reset iterator if we reach the end
                    dataset_iter = iter(dataset)
                    sample = next(dataset_iter)

                    # Same extraction logic
                    prompt = sample.get('prompt') or sample.get('text', '')[:200] or sample.get('question', '') or "Hello"
                    prompts.append(prompt.strip() if prompt.strip() else "Hello")

            # Debug: print first prompt on first step
            if step == 1:
                print(f"\n📝 Sample prompt: '{prompts[0][:100]}...'")

            # Generate responses
            responses, old_log_probs = generate_responses(
                policy_model, tokenizer, prompts, config, device
            )

            # Compute rewards
            rewards = compute_rewards(
                reward_model, reward_tokenizer, prompts, responses, device
            )

            # Compute advantages (simplified without value function)
            advantages = rewards - rewards.mean()
            advantages = advantages / (rewards.std() + 1e-8)

            # Perform PPO update
            policy_loss, value_loss, entropy = ppo_update(
                policy_model,
                optimizer,
                tokenizer,
                prompts,
                responses,
                old_log_probs,
                rewards,
                advantages,
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

                # Get trend indicators
                mean_reward = rewards.mean().item()
                reward_trend = get_trend_indicator(mean_reward, prev_reward)
                policy_loss_trend = get_trend_indicator(policy_loss, prev_policy_loss)
                entropy_trend = get_trend_indicator(entropy, prev_entropy)

                print(f"Step {step}/{config.max_steps} | "
                      f"Reward {mean_reward:.4f} {reward_trend} ± {rewards.std().item():.4f} | "
                      f"Policy Loss {policy_loss:.4f} {policy_loss_trend} | "
                      f"Entropy {entropy:.4f} {entropy_trend} | "
                      f"ETA: {eta_minutes:.1f}m")

                # Log to training report
                current_lr = optimizer.param_groups[0]['lr']
                report.log_step(
                    step=step,
                    loss=policy_loss,
                    learning_rate=current_lr,
                    reward=mean_reward,
                )

                # Callback for metrics
                if callback and hasattr(callback, 'on_step'):
                    callback.on_step(step, policy_loss, current_lr, None)

                # Update previous values
                prev_reward = mean_reward
                prev_policy_loss = policy_loss
                prev_entropy = entropy

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
                        'algorithm': 'ppo'
                    }
                    with open(os.path.join(adapter_dir, "training_metadata.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)
                else:
                    # For full fine-tuning: save complete checkpoint
                    checkpoint_path = os.path.join(config.output_dir, f"ppo_step_{step}.pt")
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
                print("Sample Generation:")
                print("-" * 60)
                eval_prompt = prompts[0]
                eval_response = responses[0]
                print(f"Prompt: {eval_prompt}")
                print(f"Response: {eval_response}")
                print(f"Reward: {rewards[0].item():.4f}")
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
            'algorithm': 'ppo'
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

    training_time = time.time() - start_time

    # Generate training report PDF
    checkpoint_path = os.path.join(config.output_dir, "final_lora_adapters") if config.use_lora else os.path.join(config.output_dir, "final_model.pt")
    report.finalize(
        final_metrics={
            'final_reward': prev_reward if prev_reward is not None else 0.0,
            'final_policy_loss': prev_policy_loss if prev_policy_loss is not None else 0.0,
            'training_time_minutes': training_time / 60,
            'total_steps': step,
            'algorithm': 'PPO',
        },
        checkpoint_path=checkpoint_path,
    )
    pdf_path = report.generate_pdf()
    print(f"📄 Training report saved: {pdf_path}")

    print(f"\n✅ Training complete!")
    print(f"Total steps: {step}")
    print(f"Total time: {training_time / 60:.1f} minutes")
