import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import math
import time
import os
from functools import partial
from tqdm import tqdm

from config import ModelConfig, SFTConfig
from model.factory import build_model
from data import load_tokenizer, create_sft_dataset, sft_collate_fn
from optimizers import setup_optimizer
from training.report import TrainingReport
from training.train import AdaptiveLRScheduler


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

    elif config.scheduler == "adaptive":
        # Get base learning rate from optimizer
        base_lr = optimizer.param_groups[0]['lr']
        scheduler = AdaptiveLRScheduler(
            optimizer=optimizer,
            warmup_steps=config.warmup_steps,
            base_lr=base_lr,
            window_size=config.adaptive_window,
            increase_factor=config.adaptive_increase_factor,
            decrease_factor=config.adaptive_decrease_factor,
            patience=config.adaptive_patience,
            min_lr=config.adaptive_min_lr,
            threshold=config.adaptive_threshold
        )

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

    # Use text markers for chat roles instead of special tokens
    # This avoids needing to resize embeddings and the associated training instability
    # Format: "User: ...\n\nAssistant: ...\n\nSystem: ...\n\n"
    #
    # IMPORTANT: BPE tokenizers produce different tokens based on context!
    # "Assistant:" alone tokenizes differently than "\n\nAssistant:" in context.
    # We must tokenize in the same context as the actual data format.
    #
    # Extract marker tokens by tokenizing in context and finding the difference
    def get_marker_tokens_in_context(marker_text, prefix="\n\n"):
        """Get tokens for a marker as it appears after a prefix (e.g., newlines)"""
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        full_tokens = tokenizer.encode(prefix + marker_text, add_special_tokens=False)
        # The marker tokens are everything after the prefix tokens
        return full_tokens[len(prefix_tokens):]

    # Get tokens for markers in different contexts
    # BPE tokenizers produce different tokens based on preceding context
    # So we need both standalone and after-newline versions for each marker
    user_marker_standalone = tokenizer.encode("User:", add_special_tokens=False)
    user_marker_newline = get_marker_tokens_in_context("User:")
    assistant_marker_standalone = tokenizer.encode("Assistant:", add_special_tokens=False)
    assistant_marker_newline = get_marker_tokens_in_context("Assistant:")
    system_marker_standalone = tokenizer.encode("System:", add_special_tokens=False)
    system_marker_newline = get_marker_tokens_in_context("System:")

    print(f"   ✓ Using text markers for chat roles (no special tokens needed)")
    print(f"   Marker tokens (context-aware):")
    print(f"      User: standalone={user_marker_standalone}, after \\n\\n={user_marker_newline}")
    print(f"      Assistant: standalone={assistant_marker_standalone}, after \\n\\n={assistant_marker_newline}")
    print(f"      System: standalone={system_marker_standalone}, after \\n\\n={system_marker_newline}")

    # Combine markers (both contexts) for the collate function
    # The collate function will search for all variants
    user_marker_tokens = (user_marker_standalone, user_marker_newline)
    assistant_marker_tokens = (assistant_marker_standalone, assistant_marker_newline)
    system_marker_tokens = (system_marker_standalone, system_marker_newline)

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

    # Create data loaders with loss masking for assistant responses only
    log_message(callback, "📊 Creating data loaders...")
    collate_with_masking = partial(
        sft_collate_fn,
        tokenizer=tokenizer,
        assistant_marker_tokens=assistant_marker_tokens,
        user_marker_tokens=user_marker_tokens,
        system_marker_tokens=system_marker_tokens
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_with_masking
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_with_masking
    )

    # Pre-fetch first batch to initialize HuggingFace streaming connections
    # This is where the actual network connections are established
    print("⏳ Warming up data pipeline (connecting to HuggingFace)...", flush=True)
    warmup_start = time.time()
    train_iter = iter(train_loader)
    first_batch = next(train_iter)
    print(f"   ✓ Data pipeline ready in {time.time() - warmup_start:.1f}s", flush=True)

    # Thorough data sanity check - verify multiple batches before training
    # This catches tokenization/marker detection issues early
    print(f"\n   📊 Data sanity check (validating marker detection)...")

    x_debug, y_debug = first_batch
    print(f"      Batch shape: x={x_debug.shape}, y={y_debug.shape}")

    # Check multiple batches for marker detection issues
    num_validation_batches = 10
    batches_with_no_valid_tokens = 0
    total_valid_tokens = 0
    total_tokens_checked = 0

    # Check the first batch
    valid_in_batch = (y_debug != -100).sum().item()
    total_in_batch = y_debug.numel()
    total_valid_tokens += valid_in_batch
    total_tokens_checked += total_in_batch
    if valid_in_batch == 0:
        batches_with_no_valid_tokens += 1

    # Check additional batches
    for _ in range(num_validation_batches - 1):
        try:
            x_check, y_check = next(train_iter)
            valid_in_batch = (y_check != -100).sum().item()
            total_in_batch = y_check.numel()
            total_valid_tokens += valid_in_batch
            total_tokens_checked += total_in_batch
            if valid_in_batch == 0:
                batches_with_no_valid_tokens += 1
        except StopIteration:
            break

    # Report results
    valid_ratio = total_valid_tokens / total_tokens_checked * 100 if total_tokens_checked > 0 else 0
    print(f"      Checked {num_validation_batches} batches: {valid_ratio:.1f}% tokens are trainable")

    if batches_with_no_valid_tokens > 0:
        print(f"      ⚠️  WARNING: {batches_with_no_valid_tokens}/{num_validation_batches} batches had 0 valid tokens!")
        print(f"         This indicates marker detection issues. Check tokenization.")

    if valid_ratio < 10:
        print(f"      ❌ CRITICAL: Only {valid_ratio:.1f}% trainable tokens - marker detection is broken!")
        print(f"         Training will likely fail. Please check:")
        print(f"         1. Data format matches 'User: ...\\n\\nAssistant: ...'")
        print(f"         2. Marker tokenization is correct")
        raise RuntimeError("Data sanity check failed: insufficient trainable tokens")
    elif valid_ratio < 30:
        print(f"      ⚠️  WARNING: Low trainable token ratio ({valid_ratio:.1f}%)")
    else:
        print(f"      ✓ Data looks healthy")

    # Show sample for debugging
    print(f"\n      Sample batch details:")
    print(f"      First 20 x tokens: {x_debug[0, :20].tolist()}")
    print(f"      Decoded x[:20]: {[tokenizer.decode([t]) if t > 0 else '<PAD>' for t in x_debug[0, :20].tolist()]}")

    # Find and show where Assistant: marker was detected
    masked_tokens = (y_debug[0] == -100).sum().item()
    active_tokens = y_debug[0].numel() - masked_tokens
    print(f"      First sequence: {masked_tokens} masked, {active_tokens} active tokens")

    # Reset iterator for training (we consumed some batches for validation)
    train_iter = iter(train_loader)
    first_batch = next(train_iter)

    # Clear memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize model
    log_message(callback, f"🔧 Building {model_config.model_architecture} model...")
    model = build_model(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move empty model to device first so weights load directly into VRAM.
    # load_state_dict will then copy from CPU checkpoint → GPU model (no double RAM usage).
    model = model.to(device=device, dtype=torch.bfloat16)

    # Load model weights directly into VRAM
    model.load_state_dict(checkpoint['model_state_dict'])

    # Free checkpoint memory immediately after loading weights
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        # LoRA adapter weights are initialized on CPU - move them to device
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

        # Safety check: skip batches with no valid tokens to train on
        # This can happen if marker detection fails for some data items
        valid_tokens = (y != -100).sum().item()
        if valid_tokens == 0:
            print(f"\n⚠️  Step {step}: Skipping batch with 0 valid tokens (marker detection failed)")
            step += 1
            pbar.update(1)
            continue

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

        # NaN detection - stop early if loss explodes
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf loss detected at step {step}! Diagnosing...")

            # Diagnose the source of NaN
            logits_nan = torch.isnan(logits).any().item()
            logits_inf = torch.isinf(logits).any().item()
            y_issues = (y[y != -100] < 0).any().item() if (y != -100).any() else False
            valid_tokens = (y != -100).sum().item()

            print(f"   📊 Diagnostics:")
            print(f"      Logits contain NaN: {logits_nan}")
            print(f"      Logits contain Inf: {logits_inf}")
            print(f"      Logits min/max: {logits.min().item():.2f} / {logits.max().item():.2f}")
            print(f"      Valid tokens (not -100): {valid_tokens}")
            print(f"      Invalid targets (negative, not -100): {y_issues}")
            print(f"      aux_loss: {aux_loss}")

            if logits_nan or logits_inf:
                # Check model weights for NaN
                nan_params = sum(1 for p in model.parameters() if torch.isnan(p).any())
                inf_params = sum(1 for p in model.parameters() if torch.isinf(p).any())
                print(f"      Model params with NaN: {nan_params}")
                print(f"      Model params with Inf: {inf_params}")

            if valid_tokens == 0:
                print(f"      ⚠️  No valid tokens to compute loss on!")
                print(f"      First 50 y values: {y[0, :50].tolist()}")

            pbar.close()
            raise RuntimeError(f"Training stopped: NaN/Inf loss at step {step}")

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
            if callback:
                if hasattr(callback, 'on_log'):
                    callback.on_log(f"Step {step}: Val Loss={eval_metrics['val_loss']:.4f}, Val PPL={eval_metrics['val_perplexity']:.1f}", "info")
                if hasattr(callback, 'on_eval'):
                    callback.on_eval(step, eval_metrics['val_loss'], eval_metrics['val_perplexity'])

            # Notify adaptive schedulers of evaluation (for LR adjustment)
            for scheduler in schedulers:
                if hasattr(scheduler, 'on_eval'):
                    adj_info = scheduler.on_eval(eval_metrics['val_loss'])
                    if adj_info.get('adjusted'):
                        direction_emoji = "📈" if adj_info['direction'] == 'increase' else "📉"
                        print(f"   {direction_emoji} Adaptive LR: {adj_info['direction']} → {adj_info['new_lr']:.2e} (trend: {adj_info['trend']:.4f})")
                        if callback and hasattr(callback, 'on_log'):
                            callback.on_log(f"Adaptive LR {adj_info['direction']}: {adj_info['new_lr']:.2e}", "info")

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
