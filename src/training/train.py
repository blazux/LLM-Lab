import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import math
import time
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

from config import ModelConfig, TrainingConfig
from model.factory import build_model
from optimizers import setup_optimizer
from data import StreamingTokenDataset, lm_collate_fn, load_tokenizer, create_token_stream
from training.report import TrainingReport


class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler that adjusts LR based on loss trends.

    Unlike fixed schedules (cosine, linear), this scheduler observes the actual
    training dynamics and adjusts accordingly:

    - If loss is decreasing well → slightly increase LR (we can go faster)
    - If loss is plateauing → decrease LR (need finer updates)
    - If loss is increasing → decrease LR more (we're overshooting)

    Uses a sliding window of eval losses to compute trends and includes
    patience/cooldown to prevent oscillations.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        warmup_steps: int,
        window_size: int = 10,
        increase_factor: float = 1.05,
        decrease_factor: float = 0.9,
        patience: int = 3,
        min_lr: float = 1e-6,
        threshold: float = 0.01,
    ):
        """
        Args:
            optimizer: The optimizer to adjust
            base_lr: Starting learning rate (after warmup)
            warmup_steps: Number of steps for linear warmup
            window_size: Number of eval losses to consider for trend
            increase_factor: Multiply LR by this when loss is decreasing well
            decrease_factor: Multiply LR by this when loss is plateauing/increasing
            patience: Minimum evals between LR adjustments
            min_lr: Minimum learning rate (floor)
            threshold: Minimum relative improvement to count as "decreasing"
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.window_size = window_size
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold

        # State
        self.current_lr = base_lr
        self.loss_history = []
        self.steps_since_adjustment = 0
        self.last_epoch = -1
        self.in_warmup = True

        # Set initial LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0  # Start at 0 for warmup
            param_group['initial_lr'] = base_lr

    def step(self, step: int = None):
        """Update LR based on current step (for warmup)"""
        if step is not None:
            self.last_epoch = step
        else:
            self.last_epoch += 1

        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step / self.warmup_steps)
            self.in_warmup = True
        else:
            # After warmup, use adaptive LR
            lr = self.current_lr
            self.in_warmup = False

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def on_eval(self, eval_loss: float) -> dict:
        """
        Called after each evaluation with the validation loss.
        Returns a dict with adjustment info.

        Args:
            eval_loss: The validation loss from this eval

        Returns:
            dict with keys: 'adjusted', 'direction', 'new_lr', 'trend'
        """
        if self.in_warmup:
            return {'adjusted': False, 'direction': 'warmup', 'new_lr': None, 'trend': None}

        self.loss_history.append(eval_loss)
        self.steps_since_adjustment += 1

        # Keep only window_size losses
        if len(self.loss_history) > self.window_size:
            self.loss_history = self.loss_history[-self.window_size:]

        # Need at least 3 points to compute trend
        if len(self.loss_history) < 3:
            return {'adjusted': False, 'direction': 'collecting', 'new_lr': None, 'trend': None}

        # Check patience
        if self.steps_since_adjustment < self.patience:
            return {'adjusted': False, 'direction': 'patience', 'new_lr': None, 'trend': None}

        # Compute trend using simple linear regression
        trend = self._compute_trend()
        result = {'adjusted': False, 'direction': None, 'new_lr': None, 'trend': trend}

        # Compute relative improvement
        first_loss = self.loss_history[0]
        last_loss = self.loss_history[-1]
        relative_change = (first_loss - last_loss) / (first_loss + 1e-8)

        if trend < -self.threshold and relative_change > self.threshold:
            # Loss is decreasing well → try increasing LR
            old_lr = self.current_lr
            new_lr = min(self.current_lr * self.increase_factor, self.base_lr * 2)  # Cap at 2x base
            if new_lr > self.current_lr:
                self.current_lr = new_lr
                self._apply_lr()
                self.steps_since_adjustment = 0
                logger.info(f"[AdaptiveLR] Increasing LR: {old_lr:.2e} -> {new_lr:.2e} (trend: {trend:.4f}, rel_change: {relative_change:.4f})")
                result = {'adjusted': True, 'direction': 'increase', 'new_lr': new_lr, 'trend': trend}

        elif trend > self.threshold or relative_change < -self.threshold:
            # Loss is increasing or getting worse → decrease LR
            old_lr = self.current_lr
            new_lr = max(self.current_lr * self.decrease_factor, self.min_lr)
            if new_lr < self.current_lr:
                self.current_lr = new_lr
                self._apply_lr()
                self.steps_since_adjustment = 0
                logger.info(f"[AdaptiveLR] Decreasing LR (loss increasing): {old_lr:.2e} -> {new_lr:.2e} (trend: {trend:.4f}, rel_change: {relative_change:.4f})")
                result = {'adjusted': True, 'direction': 'decrease', 'new_lr': new_lr, 'trend': trend}

        elif abs(trend) < self.threshold and abs(relative_change) < self.threshold:
            # Loss is plateauing → decrease LR to fine-tune
            old_lr = self.current_lr
            new_lr = max(self.current_lr * self.decrease_factor, self.min_lr)
            if new_lr < self.current_lr:
                self.current_lr = new_lr
                self._apply_lr()
                self.steps_since_adjustment = 0
                logger.info(f"[AdaptiveLR] Decreasing LR (plateau): {old_lr:.2e} -> {new_lr:.2e} (trend: {trend:.4f}, rel_change: {relative_change:.4f})")
                result = {'adjusted': True, 'direction': 'plateau', 'new_lr': new_lr, 'trend': trend}

        return result

    def _compute_trend(self) -> float:
        """Compute trend (slope) of loss history using linear regression"""
        n = len(self.loss_history)
        if n < 2:
            return 0.0

        # Normalize losses for numerical stability
        mean_loss = sum(self.loss_history) / n
        normalized = [l / (mean_loss + 1e-8) for l in self.loss_history]

        # Simple linear regression: slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        x_mean = (n - 1) / 2
        numerator = sum((i - x_mean) * (y - 1.0) for i, y in enumerate(normalized))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator < 1e-8:
            return 0.0

        return numerator / denominator

    def _apply_lr(self):
        """Apply current_lr to optimizer"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'current_lr': self.current_lr,
            'loss_history': self.loss_history,
            'steps_since_adjustment': self.steps_since_adjustment,
            'last_epoch': self.last_epoch,
            'in_warmup': self.in_warmup,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.current_lr = state_dict['current_lr']
        self.loss_history = state_dict['loss_history']
        self.steps_since_adjustment = state_dict['steps_since_adjustment']
        self.last_epoch = state_dict['last_epoch']
        self.in_warmup = state_dict['in_warmup']
        self._apply_lr()

# Global stop flag for graceful training termination
_stop_requested = False

def request_stop():
    """Request graceful stop of training"""
    global _stop_requested
    _stop_requested = True

def reset_stop_flag():
    """Reset the stop flag (called at start of training)"""
    global _stop_requested
    _stop_requested = False

def is_stop_requested():
    """Check if stop has been requested"""
    return _stop_requested


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
                logits, aux_loss = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                # Add MoE auxiliary loss if present
                if aux_loss is not None:
                    loss = loss + aux_loss

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
        if config.scheduler == "adaptive":
            # Adaptive scheduler that responds to loss trends
            scheduler = AdaptiveLRScheduler(
                optimizer=optimizer,
                base_lr=config.lr,
                warmup_steps=config.warmup_steps,
                window_size=config.adaptive_window,
                increase_factor=config.adaptive_increase_factor,
                decrease_factor=config.adaptive_decrease_factor,
                patience=config.adaptive_patience,
                min_lr=config.adaptive_min_lr,
                threshold=config.adaptive_threshold,
            )
            # Set to start_step for resuming
            if start_step > 0:
                scheduler.last_epoch = start_step - 1

        elif config.scheduler == "cosine":
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

        else:  # constant
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0, last_epoch=start_step - 1)

        schedulers.append(scheduler)

    return schedulers


def train_model(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    checkpoint_path: str = None,
    output_dir: str = "/app/data",
    additional_steps: int = 0,
    load_optimizer_state: bool = True,
    callback: callable = None
):
    """Main training function

    Args:
        model_config: Model configuration
        train_config: Training configuration
        checkpoint_path: Path to checkpoint to resume from
        output_dir: Directory to save checkpoints
        additional_steps: Additional steps to train beyond checkpoint (0 = use config max_steps)
        load_optimizer_state: Whether to load optimizer/scheduler state from checkpoint (set False when switching optimizers)
        callback: Optional callback object with on_step(step, loss, lr, ppl) and on_log(message, level) methods
    """

    os.makedirs(output_dir, exist_ok=True)

    # Save configuration files for reproducibility
    model_config.save(f"{output_dir}/model_config.json")
    train_config.save(f"{output_dir}/training_config.json")

    # Load tokenizer and create data streams
    print("\n📚 Loading tokenizer and datasets...")
    if callback and hasattr(callback, 'on_log'):
        callback.on_log("Loading tokenizer and datasets...", "info")
    tokenizer = load_tokenizer(model_config.tokenizer_name)

    # Validate vocab_size matches tokenizer
    if model_config.vocab_size != tokenizer.vocab_size:
        print(f"   ⚠️  WARNING: model_config.vocab_size ({model_config.vocab_size}) != tokenizer.vocab_size ({tokenizer.vocab_size})")
        print(f"   Automatically updating to {tokenizer.vocab_size} to match tokenizer")

    model_config.vocab_size = tokenizer.vocab_size

    # Create token stream
    token_stream_fn = create_token_stream(train_config.datasets, tokenizer)

    # Initialize model
    print(f"\n🔧 Building {model_config.model_architecture} model...")
    if callback and hasattr(callback, 'on_log'):
        callback.on_log(f"Building {model_config.model_architecture} model...", "info")
    model = build_model(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cast model to bfloat16 for memory efficiency (reduces memory usage by 50%)
    model = model.to(device=device, dtype=torch.bfloat16)

    total_params = model.count_parameters()
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model dtype: {next(model.parameters()).dtype}")
    if callback and hasattr(callback, 'on_log'):
        callback.on_log(f"Model initialized: {total_params:,} parameters", "info")

    # Setup optimizers
    optimizers = setup_optimizer(model, train_config)

    # Note: GradScaler is NOT used with bfloat16 (only needed for float16)
    # BFloat16 has same exponent range as float32, so no gradient scaling needed

    # Load checkpoint if provided
    start_step = 0
    total_tokens_seen = 0
    best_val_loss = float('inf')
    target_steps = train_config.max_steps

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n🔄 Loading checkpoint from {checkpoint_path}...")
        # Load directly to device so weights go straight into VRAM (no CPU RAM staging)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

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

        del checkpoint

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
        # Set to start_step so that the first scheduler.step() call will compute LR for start_step + 1
        for scheduler in schedulers:
            scheduler.last_epoch = start_step
            # Recompute LR based on current step (skip for AdaptiveLRScheduler which handles its own LR)
            if hasattr(scheduler, 'lr_lambdas'):
                for param_group, lr_lambda in zip(scheduler.optimizer.param_groups, scheduler.lr_lambdas):
                    param_group['lr'] = param_group['initial_lr'] * lr_lambda(start_step)
            elif isinstance(scheduler, AdaptiveLRScheduler):
                scheduler.step(start_step)

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

        # Initialize scheduler LR for the first iteration
        # The scheduler was created with last_epoch = start_step - 1
        # We need to set it to start_step and compute the initial LR
        for scheduler in schedulers:
            scheduler.last_epoch = start_step
            # Manually set initial LR for the first iteration (skip for AdaptiveLRScheduler which handles its own LR)
            if hasattr(scheduler, 'lr_lambdas'):
                for param_group, lr_lambda in zip(scheduler.optimizer.param_groups, scheduler.lr_lambdas):
                    param_group['lr'] = param_group['initial_lr'] * lr_lambda(start_step)
            elif isinstance(scheduler, AdaptiveLRScheduler):
                scheduler.step(start_step)

        print(f"   Scheduler starting at step {start_step} (LR={optimizers[0].param_groups[0]['lr']:.2e})")

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

    # Initialize training report
    report = TrainingReport(
        training_type="base",
        model_config=model_config,
        training_config=train_config,
        output_dir=output_dir,
    )

    # Training loop
    print(f"\n🚀 Starting training from step {start_step}...")
    if callback and hasattr(callback, 'on_log'):
        callback.on_log(f"Starting training from step {start_step}...", "success")

    # Reset stop flag at start of training
    reset_stop_flag()

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

            # Check for stop request
            if is_stop_requested():
                print(f"\n⏹️  Stop requested at step {step}. Saving checkpoint...")
                if callback and hasattr(callback, 'on_log'):
                    callback.on_log(f"Training stopped by user at step {step}", "warning")
                break

            x, y = x.to(device), y.to(device)
            tokens_in_batch = x.numel()
            total_tokens_seen += tokens_in_batch

            # Forward pass with bfloat16
            # Gradient checkpointing: only for transformers, not for Mamba2
            # Mamba2's sequential scan + checkpointing recomputation uses more memory
            use_checkpoint = (model_config.model_architecture == "transformer")
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux_loss = model(x, use_checkpoint=use_checkpoint)
                loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))
                # Add MoE auxiliary loss if present
                if aux_loss is not None:
                    loss = loss + aux_loss
                loss = loss / train_config.gradient_accumulation_steps

            # No gradient scaling needed for bfloat16 (unlike float16)
            loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.grad_clip
                )

                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

            # Step scheduler every iteration (not just after gradient accumulation)
            for scheduler in schedulers:
                scheduler.step()

            # Logging
            if step % 10 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * train_config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))
                    current_lr = optimizers[0].param_groups[0]['lr']

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{current_lr:.2e}'
                })

                # Log to training report
                report.log_step(
                    step=step,
                    loss=current_loss,
                    learning_rate=current_lr,
                    perplexity=perplexity,
                    accuracy=accuracy,
                    tokens_seen=total_tokens_seen,
                )

                # Callback for metrics
                if callback and hasattr(callback, 'on_step'):
                    callback.on_step(step, current_loss, current_lr, perplexity)

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

                # Log eval to training report
                report.log_eval(
                    step=step,
                    loss=eval_metrics['val_loss'],
                    perplexity=eval_metrics['val_perplexity'],
                    accuracy=eval_metrics['val_accuracy'],
                )

                # Callback for evaluation
                if callback and hasattr(callback, 'on_log'):
                    callback.on_log(f"Step {step}: Val Loss={eval_metrics['val_loss']:.4f}, Val PPL={eval_metrics['val_perplexity']:.1f}", "info")

                # Callback for eval metrics
                if callback and hasattr(callback, 'on_eval'):
                    callback.on_eval(step, eval_metrics['val_loss'], eval_metrics['val_perplexity'])

                # Update previous values
                prev_val_loss = eval_metrics['val_loss']
                prev_val_acc = eval_metrics['val_accuracy']

                # Adaptive scheduler: update based on eval loss
                if train_config.scheduler == "adaptive":
                    for scheduler in schedulers:
                        if isinstance(scheduler, AdaptiveLRScheduler):
                            adj_info = scheduler.on_eval(eval_metrics['val_loss'])
                            if adj_info['adjusted']:
                                direction_emoji = "📈" if adj_info['direction'] == 'increase' else "📉"
                                print(f"   {direction_emoji} Adaptive LR: {adj_info['direction']} → {adj_info['new_lr']:.2e} (trend: {adj_info['trend']:.4f})")
                                if callback and hasattr(callback, 'on_log'):
                                    callback.on_log(f"Adaptive LR {adj_info['direction']}: {adj_info['new_lr']:.2e}", "info")

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

                    # Callback for checkpoint save
                    if callback and hasattr(callback, 'on_log'):
                        callback.on_log(f"New best model saved: val_loss={best_val_loss:.4f}", "success")

                # Always save latest checkpoint
                if not train_config.save_best_only:
                    torch.save(checkpoint_data, f"{output_dir}/latest_checkpoint.pt")

            step += 1
            pbar.update(1)

        # Check if we broke out due to stop request
        if is_stop_requested():
            break

    pbar.close()

    # Final evaluation
    print("\n📊 Running final evaluation...")
    if callback and hasattr(callback, 'on_log'):
        callback.on_log("Running final evaluation...", "info")
    final_eval = evaluate_model(model, val_loader, train_config, model_config.vocab_size)
    print(f"   Final Loss: {final_eval['val_loss']:.4f}")
    print(f"   Final Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Final Perplexity: {final_eval['val_perplexity']:.2f}")
    if callback and hasattr(callback, 'on_log'):
        callback.on_log(f"Training completed! Final Loss={final_eval['val_loss']:.4f}, PPL={final_eval['val_perplexity']:.2f}", "success")

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

    # Generate training report PDF
    report.finalize(
        final_metrics=final_eval,
        checkpoint_path=f"{output_dir}/final_model.pt",
    )
    report.generate_pdf()

    # Clean up GPU memory
    print("\n🧹 Cleaning up GPU memory...")
    del model
    del optimizers
    del schedulers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("   ✓ GPU memory freed")
