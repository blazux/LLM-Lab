import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import math
import time
import os
from tqdm import tqdm

from config import ModelConfig, SFTConfig
from model.factory import build_model
from data import load_tokenizer, create_sft_dataset, sft_collate_fn
from optimizers import setup_optimizer
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


def evaluate_sft_model(model: nn.Module, val_loader: DataLoader, max_eval_steps: int):
    """Evaluate model on SFT validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_eval_steps:
                break

            x, y = x.to(device), y.to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux_loss = model(x)
                # Ignore padding tokens (-100) in loss
                loss = F.cross_entropy(
                    logits.view(-1, model.config.vocab_size),
                    y.view(-1),
                    ignore_index=-100
                )
                # Add MoE auxiliary loss if present
                if aux_loss is not None:
                    loss = loss + aux_loss

            # Count only non-padding tokens
            valid_tokens = (y != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

            predictions = logits.argmax(dim=-1)
            total_correct += ((predictions == y) & (y != -100)).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity
    }


def setup_sft_scheduler(optimizer, config: SFTConfig):
    """Setup learning rate scheduler for SFT"""
    if config.scheduler == "cosine":
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            else:
                progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.scheduler == "linear":
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            else:
                return max(0.0, (config.max_steps - step) / (config.max_steps - config.warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.scheduler == "polynomial":
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            else:
                progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
                return (1 - progress) ** 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:  # none
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    return scheduler


def log_message(callback, message: str, level: str = "info"):
    """Helper to log both to console and callback"""
    print(message, flush=True)
    if callback and hasattr(callback, 'on_log'):
        callback.on_log(message, level)


def train_sft(config: SFTConfig, callback=None):
    """
    Supervised Fine-Tuning training function

    Args:
        config: SFT configuration
        callback: Optional callback object with methods on_step, on_eval, on_log
    """
    # Set memory-efficient CUDA allocation strategy
    if torch.cuda.is_available():
        # Help avoid memory fragmentation
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        torch.cuda.empty_cache()

    os.makedirs(config.output_dir, exist_ok=True)

    # Save configuration file for reproducibility
    config.save(f"{config.output_dir}/sft_config.json")

    # Load checkpoint to get model config and weights
    log_message(callback, f"🔄 Loading policy model from {config.policy_checkpoint}...")
    if not os.path.exists(config.policy_checkpoint):
        raise FileNotFoundError(f"Policy checkpoint not found: {config.policy_checkpoint}")

    checkpoint = torch.load(config.policy_checkpoint, map_location="cpu", weights_only=False)

    # Extract model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        raise ValueError("Checkpoint does not contain model_config")

    # Use model's max_seq_len if not specified in config
    if config.max_seq_len is None:
        config.max_seq_len = model_config.max_seq_len

    # Override dropout if specified in SFT config
    if config.dropout is not None:
        original_dropout = model_config.dropout
        model_config.dropout = config.dropout
        print(f"   📝 Dropout override: {original_dropout} → {config.dropout}", flush=True)

    # Load tokenizer
    log_message(callback, "📚 Loading tokenizer and datasets...")
    tokenizer = load_tokenizer(model_config.tokenizer_name)

    # Add chat special tokens for SFT
    chat_special_tokens = ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    special_tokens_dict = {"additional_special_tokens": chat_special_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    if num_added > 0:
        print(f"   ✓ Added {num_added} chat special tokens: {chat_special_tokens}")

    # Check for vocab size mismatch
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = model_config.vocab_size
    print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
    print(f"   Model vocab size: {model_vocab_size}")
    if tokenizer_vocab_size != model_vocab_size:
        print(f"   ⚠️  WARNING: Vocab size mismatch detected!")
        print(f"   This will cause embedding errors if tokenizer produces IDs >= {model_vocab_size}")

    # Display dataset configuration
    print("\n📊 Dataset Configuration:")
    for i, ds_config in enumerate(config.datasets, 1):
        ds_name = ds_config.get('name', 'unknown')
        ds_subset = ds_config.get('subset', None)
        ds_split = ds_config.get('split', 'train')
        print(f"   {i}. {ds_name}")
        if ds_subset:
            print(f"      Subset: {ds_subset}")
        print(f"      Split: {ds_split}")

    # Create SFT datasets (note: start_offset is NOT used - we start fresh with SFT data)
    print("⏳ Creating training dataset...", flush=True)
    ds_start = time.time()
    train_dataset = create_sft_dataset(
        config.datasets,
        tokenizer,
        config.max_seq_len,
        split="train"
    )
    print(f"   ✓ Training dataset created in {time.time() - ds_start:.1f}s", flush=True)

    # For validation, try to use a validation split if available
    # Try common validation split names: validation, val, test
    from data import list_dataset_splits

    # Use validation_splits from config if provided, otherwise use defaults
    validation_split_names = config.validation_splits if config.validation_splits else ['validation', 'val', 'test']

    val_dataset = None
    for val_split_name in validation_split_names:
        # Check if this validation split exists for all datasets
        all_splits_exist = True
        val_datasets_config = []

        for ds_config in config.datasets:
            val_config = ds_config.copy()
            original_split = val_config.get('split', 'train')
            ds_name = ds_config['name']
            ds_subset = ds_config.get('subset', None)

            # Try replacing 'train' with validation split name if it exists in the split
            if 'train' in original_split.lower():
                candidate_split = original_split.lower().replace('train', val_split_name)
            else:
                # For custom split names, just try the validation split name directly
                candidate_split = val_split_name

            # Check if this split exists
            available_splits = list_dataset_splits(ds_name, ds_subset)
            if available_splits and candidate_split not in available_splits:
                all_splits_exist = False
                break

            val_config['split'] = candidate_split
            val_datasets_config.append(val_config)

        # If all splits exist, try loading
        if all_splits_exist and val_datasets_config:
            try:
                val_dataset = create_sft_dataset(
                    val_datasets_config,
                    tokenizer,
                    config.max_seq_len,
                    split=val_split_name
                )
                print(f"✓ Using validation split: {val_split_name}")
                break  # Successfully loaded, exit loop
            except Exception as e:
                # If loading still fails despite split existing, continue to next option
                continue

    # If no validation split found, use train split
    if val_dataset is None:
        print("⚠️  No validation split found, using training split for validation", flush=True)
        print("⏳ Creating validation dataset from training data...", flush=True)
        val_ds_start = time.time()
        val_dataset = create_sft_dataset(
            config.datasets,
            tokenizer,
            config.max_seq_len,
            split="train"
        )
        print(f"   ✓ Validation dataset created in {time.time() - val_ds_start:.1f}s", flush=True)

    # Create data loaders
    log_message(callback, "📊 Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=sft_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=sft_collate_fn
    )

    # Pre-fetch first batch to initialize HuggingFace streaming connections
    # This is where the actual network connections are established
    print("⏳ Warming up data pipeline (connecting to HuggingFace)...", flush=True)
    warmup_start = time.time()
    train_iter = iter(train_loader)
    first_batch = next(train_iter)
    print(f"   ✓ Data pipeline ready in {time.time() - warmup_start:.1f}s", flush=True)

    # Clear memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize model
    log_message(callback, f"🔧 Building {model_config.model_architecture} model...")
    model = build_model(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Free checkpoint memory immediately after loading weights
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Resize embeddings if vocab size mismatch
    # Use max of len(tokenizer) and max_special_id+1 to ensure all token IDs fit
    # (HuggingFace can assign special token IDs >= len(tokenizer))
    max_special_id = max(tokenizer.all_special_ids) if tokenizer.all_special_ids else 0
    required_vocab_size = max(tokenizer_vocab_size, max_special_id + 1)
    if required_vocab_size != tokenizer_vocab_size:
        tokenizer_vocab_size = required_vocab_size

    if tokenizer_vocab_size != model_vocab_size:
        print(f"\n🔧 Resizing embeddings from {model_vocab_size} to {tokenizer_vocab_size}...")

        # Get old embeddings and their device
        old_embeddings = model.token_embedding.weight.data
        old_vocab_size, embedding_dim = old_embeddings.shape
        embed_device = old_embeddings.device

        # Create new embedding layer with larger vocab (on same device)
        new_embedding = nn.Embedding(tokenizer_vocab_size, embedding_dim, device=embed_device)

        # Copy old embeddings
        new_embedding.weight.data[:old_vocab_size] = old_embeddings

        # Initialize new embeddings (for the extra tokens)
        # Use same initialization as original (normal distribution with std=0.02)
        nn.init.normal_(new_embedding.weight.data[old_vocab_size:], mean=0.0, std=0.02)

        # Replace token embedding
        model.token_embedding = new_embedding

        # Resize lm_head (output layer) as well since it's tied with embeddings
        old_lm_head_weight = model.lm_head.weight.data
        new_lm_head = nn.Linear(embedding_dim, tokenizer_vocab_size, bias=False, device=embed_device)
        new_lm_head.weight.data[:old_vocab_size] = old_lm_head_weight
        nn.init.normal_(new_lm_head.weight.data[old_vocab_size:], mean=0.0, std=0.02)
        model.lm_head = new_lm_head

        # Re-tie weights between embedding and lm_head
        model.lm_head.weight = model.token_embedding.weight

        # Update model config
        model.config.vocab_size = tokenizer_vocab_size

        print(f"   ✓ Added {tokenizer_vocab_size - old_vocab_size} new token embeddings")

    # Apply LoRA if configured
    if config.use_lora:
        print("\n🔧 Applying LoRA for parameter-efficient fine-tuning...")
        from model.lora_utils import apply_lora_to_model

        lora_config_dict = {
            'use_lora': config.use_lora,
            'lora_preset': config.lora_preset,
            'lora_target_modules': config.lora_target_modules,
            'lora_r': config.lora_r,
            'lora_alpha': config.lora_alpha,
            'lora_dropout': config.lora_dropout
        }

        model = apply_lora_to_model(model, model_config, lora_config_dict)
        print(f"   ✓ LoRA applied with preset: {config.lora_preset}")

    # Cast model to bfloat16 for memory efficiency (reduces memory usage by 50%)
    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"   Model dtype: {next(model.parameters()).dtype}")

    total_params = model.count_parameters()
    print(f"   Total parameters: {total_params:,}")

    # Note: We intentionally do NOT restore optimizer/scheduler state from checkpoint
    # because SFT is a fresh training phase with different data and potentially different hyperparameters
    print(f"\n   ℹ️  Starting fresh optimizer state (not loading from checkpoint)")
    print(f"   ℹ️  Token count reset - starting from beginning of SFT dataset")

    # Setup optimizer
    print(f"\n🔧 Setting up optimizer ({config.optimizer})...")

    # Create a temporary training config with SFT parameters for setup_optimizer
    from config import TrainingConfig
    temp_config = TrainingConfig()
    temp_config.optimizer = config.optimizer
    temp_config.lr = config.learning_rate
    temp_config.weight_decay = config.weight_decay
    temp_config.muon_momentum = config.muon_momentum
    temp_config.muon_nesterov = config.muon_nesterov
    temp_config.lion_beta1 = config.lion_beta1
    temp_config.lion_beta2 = config.lion_beta2
    temp_config.max_steps = config.max_steps
    temp_config.warmup_steps = config.warmup_steps
    temp_config.scheduler = config.scheduler

    optimizers = setup_optimizer(model, temp_config)

    # Setup schedulers for all optimizers
    schedulers = []
    for optimizer in optimizers:
        scheduler = setup_sft_scheduler(optimizer, config)
        schedulers.append(scheduler)

    # Note: GradScaler is NOT used with bfloat16 (only needed for float16)
    # BFloat16 has same exponent range as float32, so no gradient scaling needed

    # Training state
    start_step = 0
    best_val_loss = float('inf')

    # Track previous metrics for trend indicators
    prev_val_loss = None
    prev_val_acc = None

    # Initialize training report
    report = TrainingReport(
        training_type="sft",
        model_config=model_config,
        training_config=config,
        output_dir=config.output_dir,
    )

    # Training loop
    log_message(callback, "🚀 Starting SFT training...", "success")
    model.train()
    step = start_step
    start_time = time.time()

    pbar = tqdm(total=config.max_steps, desc="SFT Training", initial=step)

    # Use the pre-warmed iterator and first batch from earlier
    # (train_iter and first_batch were created during data pipeline warmup)
    use_prefetched = True

    while step < config.max_steps:
        if use_prefetched:
            # Use the pre-fetched first batch
            x, y = first_batch
            del first_batch  # Free memory - no longer needed
            use_prefetched = False
        else:
            try:
                x, y = next(train_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                train_iter = iter(train_loader)
                x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Forward pass with bfloat16
        # Gradient checkpointing: only for transformers, not for Mamba2
        # Mamba2's sequential scan + checkpointing recomputation uses more memory
        use_checkpoint = (model_config.model_architecture == "transformer")
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, aux_loss = model(x, use_checkpoint=use_checkpoint)
            # Ignore padding tokens (-100) in loss
            loss = F.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                y.view(-1),
                ignore_index=-100
            )
            # Add MoE auxiliary loss if present
            if aux_loss is not None:
                loss = loss + aux_loss
            loss = loss / config.gradient_accumulation_steps

        # No gradient scaling needed for bfloat16 (unlike float16)
        loss.backward()

        # Optimizer step after accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm
            )

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

        # Step scheduler every iteration (not just after gradient accumulation)
        for scheduler in schedulers:
            scheduler.step()

        # Logging
        if step % config.log_every == 0:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                valid_mask = (y != -100)
                accuracy = ((predictions == y) & valid_mask).float().sum() / valid_mask.float().sum()
                current_loss = loss.item() * config.gradient_accumulation_steps
                perplexity = math.exp(min(current_loss, 20))

            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{accuracy:.3f}',
                'ppl': f'{perplexity:.1f}',
                'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
            })

            # Log to training report
            current_lr = optimizers[0].param_groups[0]['lr']
            report.log_step(
                step=step,
                loss=current_loss,
                learning_rate=current_lr,
                perplexity=perplexity,
                accuracy=accuracy.item() if hasattr(accuracy, 'item') else accuracy,
            )

            # Callback for metrics
            if callback and hasattr(callback, 'on_step'):
                callback.on_step(step, current_loss, current_lr, perplexity)

        # Evaluation
        if step % config.eval_every == 0 and step > 0:
            eval_metrics = evaluate_sft_model(model, val_loader, config.eval_steps)

            # Get trend indicators
            val_loss_trend = get_trend_indicator(eval_metrics['val_loss'], prev_val_loss)
            val_acc_trend = get_trend_indicator(eval_metrics['val_accuracy'], prev_val_acc)

            # Mark if this is the best val loss
            is_best = eval_metrics['val_loss'] < best_val_loss
            best_marker = "★" if is_best else ""

            print(f"\n   Step {step}/{config.max_steps} | "
                  f"Val Loss {eval_metrics['val_loss']:.4f} {val_loss_trend}{best_marker} | "
                  f"Val Acc {eval_metrics['val_accuracy']:.1%} {val_acc_trend} | "
                  f"Val PPL {eval_metrics['val_perplexity']:.1f}")

            # Log eval to training report
            report.log_eval(
                step=step,
                loss=eval_metrics['val_loss'],
                perplexity=eval_metrics['val_perplexity'],
                accuracy=eval_metrics['val_accuracy'],
            )

            # Callback for eval metrics (for GUI monitor)
            if callback and hasattr(callback, 'on_eval'):
                callback.on_eval(step, eval_metrics['val_loss'], eval_metrics['val_perplexity'])

            # Update previous values
            prev_val_loss = eval_metrics['val_loss']
            prev_val_acc = eval_metrics['val_accuracy']

            # Save checkpoint if best
            if eval_metrics['val_loss'] < best_val_loss:
                best_val_loss = eval_metrics['val_loss']

                if config.use_lora:
                    # For LoRA training: only save adapters (base model is frozen)
                    adapter_dir = f"{config.output_dir}/best_lora_adapters"
                    model.save_pretrained(adapter_dir)

                    # Save minimal metadata separately
                    import json
                    metadata = {
                        'step': step,
                        'best_val_loss': best_val_loss,
                        'eval_metrics': eval_metrics,
                        'base_checkpoint': config.policy_checkpoint,  # Reference to base model
                    }
                    with open(f"{config.output_dir}/best_lora_adapters/training_metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)

                    print(f"\n   ✨ New best validation loss: {best_val_loss:.4f}")
                    print(f"   💾 Best LoRA adapters saved: {config.output_dir}/best_lora_adapters")
                else:
                    # For full fine-tuning: save complete checkpoint
                    # Move state dicts to CPU to avoid doubling GPU memory usage
                    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    checkpoint_data = {
                        'model_state_dict': model_state,
                        'optimizer_states': [opt.state_dict() for opt in optimizers],
                        'scheduler_states': [sch.state_dict() for sch in schedulers],
                        'model_config': model_config,
                        'sft_config': config,
                        'step': step,
                        'best_val_loss': best_val_loss,
                        'eval_metrics': eval_metrics
                    }

                    torch.save(checkpoint_data, f"{config.output_dir}/sft_best_model.pt")
                    del model_state, checkpoint_data  # Free CPU memory

                    print(f"\n   ✨ New best validation loss: {best_val_loss:.4f}")
                    print(f"   💾 Best model saved: {config.output_dir}/sft_best_model.pt")

            # Save checkpoint periodically
            if not config.save_best_only and step % config.save_every == 0:
                if config.use_lora:
                    # For LoRA training: only save adapters
                    adapter_dir = f"{config.output_dir}/lora_adapters_step_{step}"
                    model.save_pretrained(adapter_dir)
                    print(f"   💾 Saved LoRA adapters checkpoint (step {step})")

                    # Save metadata
                    import json
                    metadata = {
                        'step': step,
                        'eval_metrics': eval_metrics,
                        'base_checkpoint': config.policy_checkpoint,
                    }
                    with open(f"{adapter_dir}/training_metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                else:
                    # For full fine-tuning: save complete checkpoint
                    # Move state dicts to CPU to avoid doubling GPU memory usage
                    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    checkpoint_data = {
                        'model_state_dict': model_state,
                        'optimizer_states': [opt.state_dict() for opt in optimizers],
                        'scheduler_states': [sch.state_dict() for sch in schedulers],
                        'model_config': model_config,
                        'sft_config': config,
                        'step': step,
                        'eval_metrics': eval_metrics
                    }
                    torch.save(checkpoint_data, f"{config.output_dir}/checkpoint_step_{step}.pt")
                    del model_state, checkpoint_data  # Free CPU memory

        step += 1
        pbar.update(1)

    pbar.close()

    # Final evaluation
    print("\n📊 Running final evaluation...")
    final_eval = evaluate_sft_model(model, val_loader, config.eval_steps)
    print(f"   Final Loss: {final_eval['val_loss']:.4f}")
    print(f"   Final Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Final Perplexity: {final_eval['val_perplexity']:.2f}")

    # Save final model
    if config.use_lora:
        # For LoRA training: only save adapters
        adapter_dir = f"{config.output_dir}/final_lora_adapters"
        model.save_pretrained(adapter_dir)
        print(f"💾 Saved final LoRA adapters")

        # Save metadata
        import json
        metadata = {
            'step': step,
            'final_metrics': final_eval,
            'base_checkpoint': config.policy_checkpoint,
        }
        with open(f"{adapter_dir}/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    else:
        # For full fine-tuning: save complete checkpoint
        # Move state dicts to CPU to avoid doubling GPU memory usage
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            'model_state_dict': model_state,
            'optimizer_states': [opt.state_dict() for opt in optimizers],
            'scheduler_states': [sch.state_dict() for sch in schedulers],
            'model_config': model_config,
            'sft_config': config,
            'step': step,
            'final_metrics': final_eval
        }, f"{config.output_dir}/sft_final_model.pt")
        del model_state  # Free CPU memory
        print(f"💾 Saved final model checkpoint")

    training_time = time.time() - start_time
    log_message(callback, f"✅ SFT training completed in {training_time / 60:.1f} minutes", "success")

    # Generate training report PDF
    checkpoint_path = f"{config.output_dir}/final_lora_adapters" if config.use_lora else f"{config.output_dir}/sft_final_model.pt"
    report.finalize(
        final_metrics={
            'final_loss': final_eval['val_loss'],
            'final_perplexity': final_eval['val_perplexity'],
            'final_accuracy': final_eval['val_accuracy'],
            'best_val_loss': best_val_loss,
            'training_time_minutes': training_time / 60,
            'total_steps': step,
        },
        checkpoint_path=checkpoint_path,
    )
    pdf_path = report.generate_pdf()
    print(f"📄 Training report saved: {pdf_path}")

    # Clean up GPU memory
    print("\n🧹 Cleaning up GPU memory...")
    del model
    del optimizers
    del schedulers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("   ✓ GPU memory freed")
