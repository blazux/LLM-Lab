import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import math
import time
import os
from tqdm import tqdm

from config import ModelConfig, TrainingConfig
from model import TransformerLLM
from optimizers import setup_optimizer
from data import StreamingTokenDataset, lm_collate_fn, load_tokenizer, create_token_stream


def get_trend_indicator(current, previous):
    """Get trend indicator arrow for metrics

    Args:
        current: Current value
        previous: Previous value

    Returns:
        Arrow character (↑/↓/→)
    """
    if previous is None:
        return ""

    threshold = abs(previous * 0.001)  # 0.1% threshold for "no change"
    diff = current - previous

    if abs(diff) < threshold:
        return "→"
    elif diff > 0:
        return "↑"
    else:
        return "↓"


def evaluate_model(model: nn.Module, val_loader: DataLoader, config: TrainingConfig, vocab_size: int):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break

            x, y = x.to(device), y.to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity
    }


def setup_schedulers(optimizers: list, config: TrainingConfig, start_step: int = 0):
    """Setup learning rate schedulers

    Args:
        optimizers: List of optimizers
        config: Training configuration
        start_step: Starting step (for resuming training). When resuming, the scheduler
                   will treat start_step as step 0 and apply warmup/decay relative to it.
    """
    schedulers = []

    for optimizer in optimizers:
        if config.scheduler == "cosine":
            def lr_lambda(step):
                # Adjust step to be relative to start_step
                relative_step = step - start_step

                if relative_step < config.warmup_steps:
                    return relative_step / config.warmup_steps
                else:
                    # Total steps from start_step to max_steps
                    total_steps = config.max_steps - start_step
                    progress = (relative_step - config.warmup_steps) / (total_steps - config.warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

        elif config.scheduler == "linear":
            def lr_lambda(step):
                relative_step = step - start_step
                total_steps = config.max_steps - start_step

                if relative_step < config.warmup_steps:
                    return relative_step / config.warmup_steps
                else:
                    return max(0.0, (total_steps - relative_step) / (total_steps - config.warmup_steps))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

        elif config.scheduler == "polynomial":
            def lr_lambda(step):
                relative_step = step - start_step
                total_steps = config.max_steps - start_step

                if relative_step < config.warmup_steps:
                    return relative_step / config.warmup_steps
                else:
                    progress = (relative_step - config.warmup_steps) / (total_steps - config.warmup_steps)
                    return (1 - progress) ** 2

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

        else:  # none
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0, last_epoch=start_step - 1)

        schedulers.append(scheduler)

    return schedulers


def train_model(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    checkpoint_path: str = None,
    output_dir: str = "checkpoints",
    additional_steps: int = 0,
    load_optimizer_state: bool = True
):
    """Main training function

    Args:
        model_config: Model configuration
        train_config: Training configuration
        checkpoint_path: Path to checkpoint to resume from
        output_dir: Directory to save checkpoints
        additional_steps: Additional steps to train beyond checkpoint (0 = use config max_steps)
        load_optimizer_state: Whether to load optimizer/scheduler state from checkpoint (set False when switching optimizers)
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and create data streams
    print("\n📚 Loading tokenizer and datasets...")
    tokenizer = load_tokenizer(model_config.tokenizer_name)

    # Validate vocab_size matches tokenizer
    if model_config.vocab_size != tokenizer.vocab_size:
        print(f"   ⚠️  WARNING: model_config.vocab_size ({model_config.vocab_size}) != tokenizer.vocab_size ({tokenizer.vocab_size})")
        print(f"   Automatically updating to {tokenizer.vocab_size} to match tokenizer")

    model_config.vocab_size = tokenizer.vocab_size

    # Create token stream
    token_stream_fn = create_token_stream(train_config.datasets, tokenizer)

    # Initialize model
    print("\n🔧 Building model...")
    model = TransformerLLM(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = model.count_parameters()
    print(f"   Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_optimizer(model, train_config)

    # Setup gradient scaler for bfloat16
    scaler = GradScaler()

    # Load checkpoint if provided
    start_step = 0
    total_tokens_seen = 0
    best_val_loss = float('inf')
    target_steps = train_config.max_steps

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n🔄 Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        start_step = checkpoint.get('step', 0)
        total_tokens_seen = checkpoint.get('total_tokens_seen', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"   Resumed from step {start_step}, val_loss={best_val_loss:.4f}")

        if load_optimizer_state:
            if 'optimizer_states' in checkpoint:
                for opt, opt_state in zip(optimizers, checkpoint['optimizer_states']):
                    opt.load_state_dict(opt_state)
            print("   Loaded optimizer state")
        else:
            print("   Skipped loading optimizer state (starting fresh)")

        # Handle extended training
        if additional_steps > 0:
            target_steps = start_step + additional_steps
            print(f"   Training {additional_steps} additional steps (to step {target_steps})")
        elif start_step >= train_config.max_steps:
            print(f"   ⚠️  Checkpoint is at step {start_step}, but max_steps is {train_config.max_steps}")
            print(f"   No training will occur. Use additional_steps to extend training.")
            target_steps = start_step  # Will exit immediately
        else:
            target_steps = train_config.max_steps
            remaining = target_steps - start_step
            print(f"   Training {remaining} remaining steps (to step {target_steps})")

    # Setup schedulers AFTER determining start_step
    # Two scenarios:
    # 1. Resume interrupted training (load_optimizer_state=True): Schedule over ENTIRE range (0→max_steps)
    # 2. Extend completed training (load_optimizer_state=False): Schedule over NEW range (current→target)

    if load_optimizer_state and checkpoint_path:
        # Scenario 1: Resume interrupted training
        # - Create scheduler for the FULL range (step 0 → max_steps)
        # - Manually set last_epoch to resume point (NOT load state - causes LR mismatch)
        print(f"\n🔧 Setting up learning rate schedulers (continuing original schedule 0→{train_config.max_steps})...")

        # Set initial_lr before creating scheduler
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', train_config.lr)

        # Create fresh scheduler with compatible lambda functions
        schedulers = setup_schedulers(optimizers, train_config, start_step=0)

        # Manually set last_epoch to resume point and update LR
        for scheduler in schedulers:
            scheduler.last_epoch = start_step
            # Recompute LR based on current step
            for param_group, lr_lambda in zip(scheduler.optimizer.param_groups, scheduler.lr_lambdas):
                param_group['lr'] = param_group['initial_lr'] * lr_lambda(scheduler.last_epoch)

        print(f"   Scheduler resuming at step {start_step} (LR={optimizers[0].param_groups[0]['lr']:.2e})")
    else:
        # Scenario 2: Extend completed training OR fresh training
        # - Create scheduler for the NEW range (start_step → target_steps)
        # - Fresh warmup+decay over additional steps
        print(f"\n🔧 Setting up learning rate schedulers (new schedule {start_step}→{target_steps})...")

        # Set initial_lr manually when last_epoch > -1
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        schedulers = setup_schedulers(optimizers, train_config, start_step=start_step)

    # Create datasets
    print("\n📊 Creating data loaders...")
    train_dataset = StreamingTokenDataset(
        token_stream_fn,
        model_config.max_seq_len,
        split="train",
        start_offset=total_tokens_seen
    )
    val_dataset = StreamingTokenDataset(
        token_stream_fn,
        model_config.max_seq_len,
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        collate_fn=lm_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        collate_fn=lm_collate_fn
    )

    # Training loop
    print(f"\n🚀 Starting training from step {start_step}...")
    model.train()
    step = start_step
    start_time = time.time()

    # Track previous metrics for trend indicators
    prev_loss = None
    prev_acc = None
    prev_val_loss = None
    prev_val_acc = None

    pbar = tqdm(total=target_steps, desc="Training", initial=step)

    while step < target_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= target_steps:
                break

            x, y = x.to(device), y.to(device)
            tokens_in_batch = x.numel()
            total_tokens_seen += tokens_in_batch

            # Forward pass with bfloat16
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x, use_checkpoint=True)
                loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))
                loss = loss / train_config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            # Optimizer step after accumulation
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.grad_clip
                )

                for optimizer in optimizers:
                    scaler.step(optimizer)
                    optimizer.zero_grad()

                for scheduler in schedulers:
                    scheduler.step()

                scaler.update()

            # Logging
            if step % 10 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * train_config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % train_config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, train_config, model_config.vocab_size)

                # Get trend indicators
                val_loss_trend = get_trend_indicator(eval_metrics['val_loss'], prev_val_loss)
                val_acc_trend = get_trend_indicator(eval_metrics['val_accuracy'], prev_val_acc)

                # Mark if this is the best val loss
                is_best = eval_metrics['val_loss'] < best_val_loss
                best_marker = "★" if is_best else ""

                print(f"\n   Step {step}/{target_steps} | "
                      f"Val Loss {eval_metrics['val_loss']:.4f} {val_loss_trend}{best_marker} | "
                      f"Val Acc {eval_metrics['val_accuracy']:.1%} {val_acc_trend} | "
                      f"Val PPL {eval_metrics['val_perplexity']:.1f}")

                # Update previous values
                prev_val_loss = eval_metrics['val_loss']
                prev_val_acc = eval_metrics['val_accuracy']

                # Save checkpoint if best
                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

                    checkpoint_data = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_states': [opt.state_dict() for opt in optimizers],
                        'scheduler_states': [sch.state_dict() for sch in schedulers],
                        'model_config': model_config,
                        'train_config': train_config,
                        'step': step,
                        'total_tokens_seen': total_tokens_seen,
                        'best_val_loss': best_val_loss,
                        'eval_metrics': eval_metrics
                    }

                    torch.save(checkpoint_data, f"{output_dir}/best_model.pt")

                    print(f"\n   ✨ New best validation loss: {best_val_loss:.4f}")
                    print(f"   💾 Best model saved: {output_dir}/best_model.pt")

                # Always save latest checkpoint
                if not train_config.save_best_only:
                    torch.save(checkpoint_data, f"{output_dir}/latest_checkpoint.pt")

            step += 1
            pbar.update(1)

    pbar.close()

    # Final evaluation
    print("\n📊 Running final evaluation...")
    final_eval = evaluate_model(model, val_loader, train_config, model_config.vocab_size)
    print(f"   Final Loss: {final_eval['val_loss']:.4f}")
    print(f"   Final Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Final Perplexity: {final_eval['val_perplexity']:.2f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler_states': [sch.state_dict() for sch in schedulers],
        'model_config': model_config,
        'train_config': train_config,
        'step': step,
        'total_tokens_seen': total_tokens_seen,
        'final_metrics': final_eval
    }, f"{output_dir}/final_model.pt")

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time / 60:.1f} minutes")

    return model, final_eval
