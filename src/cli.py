#!/usr/bin/env python3
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, TrainingConfig, RLHFConfig, SFTConfig
from training import train_model, train_rlhf, train_sft
from inference import interactive_inference
from model import POSITIONAL_ENCODINGS, ATTENTION_TYPES, ACTIVATION_TYPES
from optimizers import OPTIMIZER_NAMES
from data import list_dataset_splits

# ANSI color codes for terminal styling
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    # Additional colors
    PURPLE = '\033[35m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'


def print_header():
    """Print CLI header with enhanced styling"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{'🧠  L L M - L A B O R A T O R Y':^88}{Colors.RESET}")
    print(f"{Colors.DIM}{'Train, fine-tune, and deploy custom language models':^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 80}{Colors.RESET}\n")


def print_menu():
    """Print main menu with enhanced styling and descriptions"""
    print(f"\n{Colors.BOLD}{Colors.WHITE}┌─ Main Menu {'─' * 64}┐{Colors.RESET}\n")

    # Model Setup Section
    print(f"{Colors.BOLD}{Colors.YELLOW}  ⚙️  MODEL SETUP{Colors.RESET}")
    print(f"{Colors.GRAY}  ├{'─' * 76}{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET} {Colors.CYAN}1.{Colors.RESET} {Colors.BOLD}Configure new model{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}    {Colors.DIM}Define architecture, choose attention mechanisms, set model size{Colors.RESET}\n")

    # Training Pipeline Section
    print(f"{Colors.BOLD}{Colors.GREEN}  🚀 TRAINING PIPELINE{Colors.RESET}")
    print(f"{Colors.GRAY}  ├{'─' * 76}{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET} {Colors.CYAN}2.{Colors.RESET} {Colors.BOLD}Base training{Colors.RESET} {Colors.DIM}(from scratch){Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}    {Colors.DIM}Pre-train a model on raw text data from the ground up{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET} {Colors.CYAN}3.{Colors.RESET} {Colors.BOLD}SFT training{Colors.RESET} {Colors.DIM}(instruction tuning){Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}    {Colors.DIM}Fine-tune on instruction datasets to follow user commands{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET} {Colors.CYAN}4.{Colors.RESET} {Colors.BOLD}RLHF training{Colors.RESET} {Colors.DIM}(alignment){Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}    {Colors.DIM}Align model with human preferences using PPO, DPO, or GRPO{Colors.RESET}\n")

    # Tools Section
    print(f"{Colors.BOLD}{Colors.BLUE}  🔧 TOOLS & UTILITIES{Colors.RESET}")
    print(f"{Colors.GRAY}  ├{'─' * 76}{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET} {Colors.CYAN}5.{Colors.RESET} {Colors.BOLD}Merge LoRA adapters{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}    {Colors.DIM}Merge parameter-efficient LoRA weights back into base model{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET} {Colors.CYAN}6.{Colors.RESET} {Colors.BOLD}Test model{Colors.RESET} {Colors.DIM}(interactive inference){Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}    {Colors.DIM}Chat with your trained model in real-time{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET} {Colors.CYAN}7.{Colors.RESET} {Colors.BOLD}Exit{Colors.RESET}")
    print(f"  {Colors.BOLD}│{Colors.RESET}    {Colors.DIM}Close the application{Colors.RESET}")

    print(f"\n{Colors.BOLD}{Colors.WHITE}└{'─' * 78}┘{Colors.RESET}")


def print_section_header(title: str, icon: str = ""):
    """Print a styled section header"""
    full_title = f"{icon} {title}" if icon else title
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'─' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}{full_title:^88}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'─' * 80}{Colors.RESET}\n")


def print_subsection(title: str, icon: str = ""):
    """Print a styled subsection"""
    full_title = f"{icon} {title}" if icon else title
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{full_title}{Colors.RESET}")
    print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")


def print_success(message: str):
    """Print a success message"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print an error message"""
    print(f"\n{Colors.BOLD}{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print a warning message"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message: str):
    """Print an info message"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}ℹ {message}{Colors.RESET}")


def get_input(prompt: str, default=None, type_fn=str):
    """Get user input with optional default value"""
    if default is not None:
        user_input = input(f"{Colors.BOLD}➤{Colors.RESET} {prompt} {Colors.DIM}(default: {default}){Colors.RESET}: ").strip()
        if not user_input:
            return default
    else:
        user_input = input(f"{Colors.BOLD}➤{Colors.RESET} {prompt}: ").strip()

    try:
        return type_fn(user_input)
    except ValueError:
        print(f"{Colors.YELLOW}Invalid input, using default: {default}{Colors.RESET}")
        return default


def configure_model():
    """Interactive model configuration"""
    print_section_header("Model Configuration", "⚙️")

    config = ModelConfig()

    # Architecture selection
    print_subsection("Architecture Selection", "🏛️")
    print(f"\n  {Colors.CYAN}Available architectures:{Colors.RESET}")
    print(f"    {Colors.BOLD}transformer{Colors.RESET} - Traditional attention-based model (O(N²) complexity)")
    print(f"    {Colors.BOLD}mamba2{Colors.RESET}     - State-space model (O(N) complexity, longer context)")
    print()
    config.model_architecture = get_input(
        "Model architecture",
        default=config.model_architecture
    )

    if config.model_architecture not in ["transformer", "mamba2"]:
        print_error(f"Invalid architecture: {config.model_architecture}")
        print(f"  {Colors.DIM}Supported: transformer, mamba2{Colors.RESET}")
        return

    # Tokenizer
    print_subsection("Tokenizer Configuration", "📖")
    config.tokenizer_name = get_input(
        "Tokenizer name (HuggingFace)",
        default=config.tokenizer_name
    )

    # Common parameters
    print_subsection("Model Parameters", "🔢")
    config.d_model = get_input("Embedding size (d_model)", default=config.d_model, type_fn=int)
    config.n_layers = get_input("Number of layers", default=config.n_layers, type_fn=int)
    config.max_seq_len = get_input("Maximum sequence length", default=config.max_seq_len, type_fn=int)
    config.dropout = get_input("Dropout rate", default=config.dropout, type_fn=float)

    print(f"\n  {Colors.DIM}Normalization options: {Colors.BOLD}layernorm, rmsnorm{Colors.RESET}")
    config.norm_type = get_input(
        "Normalization type",
        default=config.norm_type
    )

    if config.model_architecture == "transformer":
        # Transformer-specific configuration
        print_subsection("Transformer Architecture", "🔀")

        print(f"\n  {Colors.DIM}Positional Encoding options: {Colors.BOLD}{', '.join(POSITIONAL_ENCODINGS.keys())}{Colors.RESET}")
        config.positional_encoding = get_input(
            "Positional encoding",
            default=config.positional_encoding
        )

        print(f"\n  {Colors.DIM}Attention type options: {Colors.BOLD}{', '.join(ATTENTION_TYPES.keys())}{Colors.RESET}")
        config.attention_type = get_input(
            "Attention type",
            default=config.attention_type
        )

        config.n_heads = get_input("Number of attention heads", default=config.n_heads, type_fn=int)

        if config.attention_type == "gqa":
            config.n_kv_heads = get_input("Number of KV heads (for GQA)", default=config.n_kv_heads, type_fn=int)

        print(f"\n  {Colors.DIM}Activation options: {Colors.BOLD}{', '.join(ACTIVATION_TYPES.keys())}{Colors.RESET}")
        config.activation = get_input(
            "Feed-forward activation",
            default=config.activation
        )

        config.d_ff = get_input("Feed-forward hidden size (d_ff)", default=config.d_ff, type_fn=int)

    elif config.model_architecture == "mamba2":
        # Mamba2-specific configuration
        print_subsection("Mamba2 State-Space Architecture", "🌊")

        print(f"\n  {Colors.CYAN}ℹ{Colors.RESET}  Mamba2 replaces attention with state-space models")
        print(f"     {Colors.DIM}Offers O(N) complexity and efficient long-context processing{Colors.RESET}")
        print()

        config.state_size = get_input(
            "SSM state dimension (d_state)",
            default=config.state_size,
            type_fn=int
        )

        config.expand_factor = get_input(
            "Hidden dimension expansion factor",
            default=config.expand_factor,
            type_fn=int
        )

        config.conv_kernel_size = get_input(
            "Convolution kernel size",
            default=config.conv_kernel_size,
            type_fn=int
        )

        print(f"\n  {Colors.DIM}Note: dt_rank is auto-computed if not specified{Colors.RESET}")
        dt_rank_input = get_input(
            "Time-step projection rank (or 'auto')",
            default="auto"
        )
        if dt_rank_input.lower() == "auto":
            config.dt_rank = None  # Will be auto-computed
        else:
            config.dt_rank = int(dt_rank_input)

    # Save config
    print_subsection("Save Configuration", "💾")

    # Suggest appropriate default filename based on architecture
    if config.model_architecture == "mamba2":
        default_save_path = "model_config_mamba2.json"
    else:
        default_save_path = "model_config.json"

    save_path = get_input("Save config to", default=default_save_path)

    # Sync vocab_size with tokenizer before saving
    from data import load_tokenizer
    tokenizer = load_tokenizer(config.tokenizer_name)
    if config.vocab_size != tokenizer.vocab_size:
        print(f"\n  {Colors.YELLOW}ℹ{Colors.RESET}  Detected vocab_size mismatch:")
        print(f"     Config: {config.vocab_size} → Tokenizer: {tokenizer.vocab_size}")
        print(f"     Updating config to match tokenizer...")
        config.vocab_size = tokenizer.vocab_size

    config.save(save_path)

    # Show parameter count
    param_count = config.count_params()
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}✓ {config.model_architecture.capitalize()} model configured successfully!{Colors.RESET}")
    print(f"  {Colors.WHITE}Architecture:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}{config.model_architecture}{Colors.RESET}")
    print(f"  {Colors.WHITE}Estimated parameters:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}{param_count:,}{Colors.RESET}")
    print(f"  {Colors.WHITE}Config saved to:{Colors.RESET} {Colors.CYAN}{save_path}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'─' * 60}{Colors.RESET}")

    if config.model_architecture == "mamba2":
        print(f"\n  {Colors.YELLOW}⚠{Colors.RESET}  {Colors.BOLD}Mamba2 requires:{Colors.RESET}")
        print(f"     pip install mamba-ssm>=2.0.0 causal-conv1d>=1.2.0")
        print(f"     See MAMBA2.md for details")


def configure_training():
    """Interactive base training configuration"""
    print_section_header("Base Training Configuration", "🎯")

    # Load model config
    print_subsection("Model Configuration", "📖")
    model_config_path = get_input("Model config path", default="model_config.json")

    if not os.path.exists(model_config_path):
        print_error(f"Model config not found: {model_config_path}")
        print(f"  {Colors.DIM}Please configure a model first (option 1 in main menu){Colors.RESET}")
        return None, None

    model_config = ModelConfig.load(model_config_path)
    train_config = TrainingConfig()
    train_config.model_config_path = model_config_path

    # Training steps
    print_subsection("Training Steps", "🏃")
    train_config.max_steps = get_input("Maximum training steps", default=train_config.max_steps, type_fn=int)

    # Optimizer
    print_subsection("Optimizer Configuration", "🔧")
    print(f"  {Colors.DIM}Available optimizers: {Colors.BOLD}{', '.join(OPTIMIZER_NAMES)}{Colors.RESET}\n")
    train_config.optimizer = get_input("Optimizer", default=train_config.optimizer)
    train_config.lr = get_input("Learning rate", default=train_config.lr, type_fn=float)
    train_config.weight_decay = get_input("Weight decay", default=train_config.weight_decay, type_fn=float)

    # Optimizer-specific parameters
    if train_config.optimizer.lower() == "adamw":
        print(f"\n  {Colors.BOLD}{Colors.CYAN}AdamW-specific parameters{Colors.RESET}")
        train_config.adamw_beta1 = get_input("Beta1", default=train_config.adamw_beta1, type_fn=float)
        train_config.adamw_beta2 = get_input("Beta2", default=train_config.adamw_beta2, type_fn=float)
        train_config.adamw_eps = get_input("Epsilon", default=train_config.adamw_eps, type_fn=float)
    elif train_config.optimizer.lower() == "muon":
        print(f"\n  {Colors.BOLD}{Colors.CYAN}Muon-specific parameters{Colors.RESET}")
        train_config.muon_momentum = get_input("Momentum", default=train_config.muon_momentum, type_fn=float)
        nesterov_input = get_input("Use Nesterov? [y/n]", default="y" if train_config.muon_nesterov else "n")
        train_config.muon_nesterov = nesterov_input.lower() in ['y', 'yes']
    elif train_config.optimizer.lower() == "lion":
        print(f"\n  {Colors.BOLD}{Colors.CYAN}Lion-specific parameters{Colors.RESET}")
        train_config.lion_beta1 = get_input("Beta1", default=train_config.lion_beta1, type_fn=float)
        train_config.lion_beta2 = get_input("Beta2", default=train_config.lion_beta2, type_fn=float)
    elif train_config.optimizer.lower() == "sophia":
        print(f"\n  {Colors.BOLD}{Colors.CYAN}Sophia-specific parameters{Colors.RESET}")
        train_config.sophia_beta1 = get_input("Beta1", default=train_config.sophia_beta1, type_fn=float)
        train_config.sophia_beta2 = get_input("Beta2", default=train_config.sophia_beta2, type_fn=float)
        train_config.sophia_rho = get_input("Rho (clipping)", default=train_config.sophia_rho, type_fn=float)

    # Scheduler
    print_subsection("Learning Rate Scheduler", "📈")
    print(f"  {Colors.DIM}Available schedulers: {Colors.BOLD}none, cosine, linear, polynomial{Colors.RESET}\n")
    train_config.scheduler = get_input("Scheduler", default=train_config.scheduler)
    train_config.warmup_steps = get_input("Warmup steps", default=train_config.warmup_steps, type_fn=int)

    # Batch and accumulation
    print_subsection("Batch Configuration", "📦")
    train_config.batch_size = get_input("Batch size", default=train_config.batch_size, type_fn=int)
    train_config.gradient_accumulation_steps = get_input(
        "Gradient accumulation steps",
        default=train_config.gradient_accumulation_steps,
        type_fn=int
    )
    train_config.grad_clip = get_input("Gradient clipping", default=train_config.grad_clip, type_fn=float)

    # Evaluation
    print_subsection("Evaluation", "📊")
    train_config.eval_every = get_input("Eval every N steps", default=train_config.eval_every, type_fn=int)
    train_config.eval_steps = get_input("Steps per evaluation", default=train_config.eval_steps, type_fn=int)

    save_best = get_input("Save best model only? [y/n]", default="y")
    train_config.save_best_only = save_best.lower() in ['y', 'yes']

    # Dataset configuration
    print_subsection("Dataset Configuration", "📚")
    print(f"{Colors.DIM}Enter datasets (one per line, empty line to finish)")
    print(f"Format: dataset_name | subset (optional) | weight (optional){Colors.RESET}\n")
    print(f"  {Colors.CYAN}Examples:{Colors.RESET}")
    print(f"    {Colors.DIM}HuggingFaceFW/fineweb-edu")
    print(f"    HuggingFaceFW/fineweb-2 | fra_Latn | 1.0")
    print(f"    HuggingFaceFW/fineweb-2 | spa_Latn | 2.0{Colors.RESET}")

    datasets = []
    while True:
        dataset_input = input(f"\n{Colors.BOLD}➤{Colors.RESET} Dataset {len(datasets) + 1} (or press Enter to finish): ").strip()
        if not dataset_input:
            break

        parts = [p.strip() for p in dataset_input.split('|')]
        ds_config = {"name": parts[0]}

        if len(parts) > 1 and parts[1]:
            ds_config["subset"] = parts[1]
        if len(parts) > 2 and parts[2]:
            ds_config["weight"] = float(parts[2])

        datasets.append(ds_config)
        subset_info = f" ({ds_config.get('subset', 'no subset')})" if 'subset' in ds_config or len(parts) > 1 else ""
        print(f"  {Colors.GREEN}✓{Colors.RESET} Added: {Colors.CYAN}{ds_config['name']}{Colors.RESET}{subset_info}")

    if datasets:
        train_config.datasets = datasets
    else:
        print(f"  {Colors.DIM}Using default dataset: HuggingFaceFW/fineweb-edu{Colors.RESET}")

    # Save config
    print_subsection("Save Configuration", "💾")
    save_path = get_input("Save training config to", default="training_config.json")
    train_config.save(save_path)

    print_success(f"Training configuration saved to: {save_path}")

    return model_config, train_config


def start_training():
    """Start base training with configuration"""
    print_section_header("Start Base Training", "🎯")

    # Option to use existing config or create new
    print(f"\n  {Colors.CYAN}1.{Colors.RESET} Use existing training config")
    print(f"  {Colors.CYAN}2.{Colors.RESET} Configure new training\n")

    choice = get_input("Choice", default="1")

    if choice == "2":
        model_config, train_config = configure_training()
        if model_config is None:
            return
    else:
        train_config_path = get_input("Training config path", default="training_config.json")
        if not os.path.exists(train_config_path):
            print_error(f"Training config not found: {train_config_path}")
            return

        train_config = TrainingConfig.load(train_config_path)

        if not os.path.exists(train_config.model_config_path):
            print_error(f"Model config not found: {train_config.model_config_path}")
            return

        model_config = ModelConfig.load(train_config.model_config_path)

    # Check for checkpoint
    checkpoint_path = None
    additional_steps = 0
    load_optimizer_state = True
    resume = get_input("\nResume from checkpoint? [y/n]", default="n")
    if resume.lower() in ['y', 'yes']:
        checkpoint_path = get_input("Checkpoint path", default="checkpoints/best_model.pt")
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            checkpoint_path = None
        else:
            # Load checkpoint to check current step
            import torch
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                current_step = ckpt.get('step', 0)
                print(f"\n📍 Checkpoint is at step {current_step}")
                print(f"   Config max_steps: {train_config.max_steps}")

                if current_step >= train_config.max_steps:
                    print(f"   ⚠️  Checkpoint has already reached max_steps!")
                    extend = get_input("   Train additional steps? [y/n]", default="y")
                    if extend.lower() in ['y', 'yes']:
                        additional_steps = get_input("   How many additional steps?", default=10000, type_fn=int)
                else:
                    remaining = train_config.max_steps - current_step
                    print(f"   Will train {remaining} remaining steps to reach {train_config.max_steps}")
                    extend = get_input("   Train even more steps beyond config? [y/n]", default="n")
                    if extend.lower() in ['y', 'yes']:
                        additional_steps = get_input("   How many additional steps beyond max_steps?", default=0, type_fn=int)

                # Ask about optimizer state loading
                print(f"\n🔧 Current config optimizer: {train_config.optimizer}")
                load_opt = get_input("   Load optimizer state from checkpoint? [y/n] (say 'n' if switching optimizers)", default="n")
                load_optimizer_state = load_opt.lower() in ['y', 'yes']
            except Exception as e:
                print(f"⚠️  Could not read checkpoint: {e}")
                checkpoint_path = None

    # Output directory
    output_dir = get_input("\nOutput directory", default="checkpoints")

    # Confirm and start
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}Ready to start base training!{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'─' * 60}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Model:{Colors.RESET} {Colors.CYAN}{model_config.d_model}d, {model_config.n_layers}L, {model_config.n_heads}H{Colors.RESET}")

    if checkpoint_path and additional_steps > 0:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        current_step = ckpt.get('step', 0)
        print(f"  {Colors.YELLOW}Resuming from:{Colors.RESET} {Colors.CYAN}step {current_step}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Target steps:{Colors.RESET} {Colors.CYAN}{current_step + additional_steps}{Colors.RESET}")
    elif checkpoint_path:
        print(f"  {Colors.YELLOW}Steps:{Colors.RESET} {Colors.CYAN}{train_config.max_steps}{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}Steps:{Colors.RESET} {Colors.CYAN}{train_config.max_steps}{Colors.RESET}")

    print(f"  {Colors.YELLOW}Optimizer:{Colors.RESET} {Colors.CYAN}{train_config.optimizer}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Learning rate:{Colors.RESET} {Colors.CYAN}{train_config.lr}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'─' * 60}{Colors.RESET}\n")

    confirm = get_input("Start base training? [y/n]", default="y")
    if confirm.lower() not in ['y', 'yes']:
        print_warning("Base training cancelled.")
        return

    # Update training_config.json with new max_steps if additional steps were added
    if additional_steps > 0 and choice == "1":
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        current_step = ckpt.get('step', 0)
        new_max_steps = current_step + additional_steps
        train_config.max_steps = new_max_steps
        train_config.save(train_config_path)
        print(f"✓ Updated {train_config_path} with new max_steps: {new_max_steps}")

    # Start base training
    try:
        train_model(model_config, train_config, checkpoint_path, output_dir, additional_steps, load_optimizer_state)
    except KeyboardInterrupt:
        print_warning("\nBase training interrupted by user")
    except Exception as e:
        print_error(f"\nBase training failed: {e}")
        raise


def configure_sft():
    """Interactive SFT configuration"""
    print("\n" + "-" * 60)
    print("SFT Configuration")
    print("-" * 60)

    config = SFTConfig()

    # Policy checkpoint
    print("\n🤖 Base Model")
    config.policy_checkpoint = get_input(
        "Base model checkpoint path",
        default=config.policy_checkpoint
    )

    if not os.path.exists(config.policy_checkpoint):
        print(f"❌ Base model checkpoint not found: {config.policy_checkpoint}")
        return None

    # Training parameters
    print("\n⚙️  Training Hyperparameters")
    config.batch_size = get_input("Batch size", default=config.batch_size, type_fn=int)
    config.gradient_accumulation_steps = get_input(
        "Gradient accumulation steps",
        default=config.gradient_accumulation_steps,
        type_fn=int
    )
    config.max_steps = get_input("Maximum training steps", default=config.max_steps, type_fn=int)

    # Optimizer
    print(f"\n🔧 Optimizer")
    print(f"Options: {', '.join(OPTIMIZER_NAMES)}")
    config.optimizer = get_input("Optimizer", default=config.optimizer)
    config.learning_rate = get_input("Learning rate", default=config.learning_rate, type_fn=float)
    config.weight_decay = get_input("Weight decay", default=config.weight_decay, type_fn=float)

    # Optimizer-specific parameters
    if config.optimizer.lower() == "adamw":
        print(f"\n   AdamW-specific parameters")
        config.adamw_beta1 = get_input("   Beta1", default=config.adamw_beta1, type_fn=float)
        config.adamw_beta2 = get_input("   Beta2", default=config.adamw_beta2, type_fn=float)
        config.adamw_eps = get_input("   Epsilon", default=config.adamw_eps, type_fn=float)
    elif config.optimizer.lower() == "muon":
        print(f"\n   Muon-specific parameters")
        config.muon_momentum = get_input("   Momentum", default=config.muon_momentum, type_fn=float)
        nesterov_input = get_input("   Use Nesterov? [y/n]", default="y" if config.muon_nesterov else "n")
        config.muon_nesterov = nesterov_input.lower() in ['y', 'yes']
    elif config.optimizer.lower() == "lion":
        print(f"\n   Lion-specific parameters")
        config.lion_beta1 = get_input("   Beta1", default=config.lion_beta1, type_fn=float)
        config.lion_beta2 = get_input("   Beta2", default=config.lion_beta2, type_fn=float)
    elif config.optimizer.lower() == "sophia":
        print(f"\n   Sophia-specific parameters")
        config.sophia_beta1 = get_input("   Beta1", default=config.sophia_beta1, type_fn=float)
        config.sophia_beta2 = get_input("   Beta2", default=config.sophia_beta2, type_fn=float)
        config.sophia_rho = get_input("   Rho (clipping)", default=config.sophia_rho, type_fn=float)

    # Scheduler
    print(f"\n📈 Learning Rate Scheduler")
    print("Options: none, cosine, linear, polynomial")
    config.scheduler = get_input("Scheduler", default=config.scheduler)
    config.warmup_steps = get_input("Warmup steps", default=config.warmup_steps, type_fn=int)

    # Evaluation
    print("\n📊 Evaluation")
    config.eval_every = get_input("Eval every N steps", default=config.eval_every, type_fn=int)
    config.eval_steps = get_input("Steps per evaluation", default=config.eval_steps, type_fn=int)

    save_best = get_input("Save best model only? [y/n]", default="y")
    config.save_best_only = save_best.lower() in ['y', 'yes']

    # LoRA configuration
    print("\n🔧 LoRA (Parameter-Efficient Fine-Tuning)")
    use_lora = get_input("Use LoRA? [y/n]", default="n")
    config.use_lora = use_lora.lower() in ['y', 'yes']

    if config.use_lora:
        # Load model config to determine available presets
        import torch
        try:
            checkpoint = torch.load(config.policy_checkpoint, map_location="cpu", weights_only=False)
            model_config = checkpoint.get('model_config')

            if model_config:
                from model.lora_utils import get_available_presets
                presets = get_available_presets(model_config)

                print("\nAvailable LoRA presets:")
                for preset_name, description in presets.items():
                    print(f"  {preset_name}: {description}")

                config.lora_preset = get_input("LoRA preset", default=config.lora_preset)

                if config.lora_preset == "custom":
                    print("\nEnter target modules (comma-separated)")
                    print("Examples: q_proj,v_proj or gate_proj,up_proj,down_proj")
                    modules_str = input("Target modules: ").strip()
                    if modules_str:
                        config.lora_target_modules = [m.strip() for m in modules_str.split(',')]
            else:
                print("⚠️  Could not load model config from checkpoint")
                print("   Using default preset")
                config.lora_preset = get_input("LoRA preset", default=config.lora_preset)
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Using default preset")
            config.lora_preset = get_input("LoRA preset", default=config.lora_preset)

        config.lora_r = get_input("LoRA rank (r)", default=config.lora_r, type_fn=int)
        config.lora_alpha = get_input("LoRA alpha", default=config.lora_alpha, type_fn=int)
        config.lora_dropout = get_input("LoRA dropout", default=config.lora_dropout, type_fn=float)

    # Dataset configuration
    print("\n📚 Dataset Configuration")
    print("Enter datasets (one per line, empty line to finish)")
    print("Format: dataset_name | subset (optional) | split (optional) | weight (optional)")
    print("\nExamples:")
    print("  HuggingFaceTB/smoltalk2 | SFT | smoltalk_smollm3_everyday_conversations_no_think")
    print("  OpenAssistant/oasst1 | | train | 1.5")
    print("  HuggingFaceH4/ultrachat_200k | | train_sft | 2.0")

    datasets = []
    while True:
        dataset_input = input(f"\nDataset {len(datasets) + 1} (or press Enter to finish): ").strip()
        if not dataset_input:
            break

        parts = [p.strip() for p in dataset_input.split('|')]
        ds_name = parts[0]
        ds_subset = parts[1] if len(parts) > 1 and parts[1] else None
        ds_split = parts[2] if len(parts) > 2 and parts[2] else None
        ds_weight = float(parts[3]) if len(parts) > 3 and parts[3] else None

        # If no split provided, try to show available splits
        if not ds_split:
            print(f"\n  Fetching available splits for {ds_name}...")
            available_splits = list_dataset_splits(ds_name, ds_subset)
            if available_splits:
                print(f"  Available splits ({len(available_splits)} total):")
                # Show splits in a nice format
                for i, split in enumerate(available_splits[:15], 1):
                    print(f"    {i}. {split}")
                if len(available_splits) > 15:
                    print(f"    ... and {len(available_splits) - 15} more")

                # Let user choose
                split_choice = input(f"\n  Enter split name or number (1-{len(available_splits)}): ").strip()
                if split_choice.isdigit():
                    idx = int(split_choice) - 1
                    if 0 <= idx < len(available_splits):
                        ds_split = available_splits[idx]
                else:
                    ds_split = split_choice

            if not ds_split:
                print("  ⚠️  No split specified, skipping this dataset")
                continue

        ds_config = {"name": ds_name, "split": ds_split}
        if ds_subset:
            ds_config["subset"] = ds_subset
        if ds_weight:
            ds_config["weight"] = ds_weight

        datasets.append(ds_config)
        weight_str = f" (weight: {ds_weight})" if ds_weight else ""
        print(f"  Added: {ds_name}" + (f" ({ds_subset})" if ds_subset else "") + f" [split: {ds_split}]" + weight_str)

    if datasets:
        config.datasets = datasets
    else:
        print("Using default dataset: HuggingFaceTB/smoltalk2 (SFT, everyday conversations)")

    # Save config
    save_path = get_input("\n💾 Save SFT config to", default="sft_config.json")
    config.save(save_path)

    print(f"\n✅ SFT configuration saved to: {save_path}")
    return config


def start_sft_training():
    """Start SFT training"""
    print("\n" + "-" * 60)
    print("Start SFT Training")
    print("-" * 60)

    # Option to use existing config or create new
    choice = get_input(
        "\n1. Use existing SFT config\n2. Configure new SFT training\nChoice",
        default="1"
    )

    if choice == "2":
        config = configure_sft()
        if config is None:
            return
    else:
        config_path = get_input("SFT config path", default="sft_config.json")
        if not os.path.exists(config_path):
            print(f"❌ SFT config not found: {config_path}")
            return

        config = SFTConfig.load(config_path)

    # Output directory
    config.output_dir = get_input("\nOutput directory", default=config.output_dir)

    # Confirm and start
    print("\n" + "=" * 60)
    print(f"Ready to start SFT training!")
    print(f"  Base model: {config.policy_checkpoint}")
    print(f"\n  Datasets:")
    for ds in config.datasets:
        ds_name = ds.get('name', 'unknown')
        ds_subset = ds.get('subset', None)
        ds_split = ds.get('split', 'train')
        print(f"    - {ds_name}")
        if ds_subset:
            print(f"      Subset: {ds_subset}")
        print(f"      Split: {ds_split}")
    print(f"\n  Steps: {config.max_steps}")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print("=" * 60)

    confirm = get_input("\nStart SFT training? [y/n]", default="y")
    if confirm.lower() not in ['y', 'yes']:
        print("SFT training cancelled.")
        return

    # Start training
    try:
        train_sft(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  SFT training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ SFT training failed: {e}")
        raise


def merge_lora_adapters():
    """Merge LoRA adapters into base model"""
    print_section_header("Merge LoRA Adapters", "🔀")

    print(f"{Colors.CYAN}ℹ{Colors.RESET}  This tool merges LoRA adapter weights back into the base model.")
    print(f"  {Colors.DIM}Use this after LoRA training to create a standard checkpoint for RLHF.{Colors.RESET}\n")

    # Ask for input type
    print("Choose input type:")
    print("  1. Full checkpoint (.pt file) - contains base model + LoRA")
    print("  2. Adapter folder - lightweight adapters saved by PEFT")
    input_type = get_input("Input type", default="2")

    import torch

    try:
        if input_type == "2":
            # Load from adapter folder (new method)
            adapter_path = get_input("LoRA adapter folder path", default="sft_checkpoints/best_lora_adapters")

            if not os.path.exists(adapter_path):
                print(f"❌ Adapter folder not found: {adapter_path}")
                return

            # Check for adapter files
            adapter_config_file = os.path.join(adapter_path, "adapter_config.json")
            if not os.path.exists(adapter_config_file):
                print(f"❌ Not a valid adapter folder: missing adapter_config.json")
                return

            # Get base model checkpoint
            base_checkpoint_path = get_input("Base model checkpoint path", default="checkpoints/best_model.pt")

            if not os.path.exists(base_checkpoint_path):
                print(f"❌ Base checkpoint not found: {base_checkpoint_path}")
                return

            print(f"\n🔄 Loading base model from {base_checkpoint_path}...")
            checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)

            if 'model_config' not in checkpoint:
                print("❌ Checkpoint does not contain model_config")
                return

            model_config = checkpoint['model_config']

            # Create base model
            from model import TransformerLLM
            print("🔧 Creating base model...")
            base_model = TransformerLLM(model_config)
            base_model.load_state_dict(checkpoint['model_state_dict'])

            # Load PEFT model with adapters
            from peft import PeftModel
            print(f"📥 Loading LoRA adapters from {adapter_path}...")
            peft_model = PeftModel.from_pretrained(base_model, adapter_path)

            # Merge adapters
            print("🔀 Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()

            print("✓ Successfully merged LoRA adapters!")

            # Save merged model
            output_path = get_input("\n💾 Save merged model to",
                                   default=base_checkpoint_path.replace('.pt', '_merged.pt'))

            print(f"\n💾 Saving merged model to {output_path}...")

            # Prepare checkpoint data
            merged_checkpoint = {
                'model_state_dict': merged_model.state_dict(),
                'model_config': model_config,
            }

            # Copy over other metadata if available
            for key in ['step', 'best_val_loss', 'eval_metrics', 'final_metrics']:
                if key in checkpoint:
                    merged_checkpoint[key] = checkpoint[key]

            torch.save(merged_checkpoint, output_path)

            print("\n✅ Successfully merged and saved!")
            print(f"   Merged checkpoint: {output_path}")
            print(f"   Base model: {base_checkpoint_path}")
            print(f"   Adapters: {adapter_path}")
            print("\n💡 You can now use the merged checkpoint for RLHF training without LoRA.")
            return

        # Original method: Load from full checkpoint
        checkpoint_path = get_input("LoRA checkpoint path", default="sft_checkpoints/best_model.pt")

        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return

        # Load checkpoint
        print(f"\n🔄 Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Get model config
        if 'model_config' not in checkpoint:
            print("❌ Checkpoint does not contain model_config")
            return

        model_config = checkpoint['model_config']

        # Check if checkpoint has LoRA parameters
        state_dict = checkpoint['model_state_dict']
        has_lora = any('lora' in key.lower() for key in state_dict.keys())

        if not has_lora:
            print("⚠️  This checkpoint doesn't appear to have LoRA parameters")
            print("   It may already be a merged model.")
            proceed = get_input("Continue anyway? [y/n]", default="n")
            if proceed.lower() not in ['y', 'yes']:
                return
        else:
            print(f"✓ Detected LoRA parameters in checkpoint")
            lora_params = [key for key in state_dict.keys() if 'lora' in key.lower()]
            print(f"   Found {len(lora_params)} LoRA parameter tensors")

        # Create base model
        from model import TransformerLLM
        print("\n🔧 Creating base model...")
        base_model = TransformerLLM(model_config)

        # Load state dict (with LoRA parameters)
        print("📥 Loading state dict...")
        try:
            # Try to load directly first
            base_model.load_state_dict(state_dict, strict=False)

            # Now apply PEFT to load LoRA properly
            from peft import PeftModel

            # We need to reconstruct the PEFT model to merge properly
            # This is a bit tricky - we need to know the LoRA config
            # Let's try to infer it or ask the user

            print("\n🔧 Detecting LoRA configuration...")

            # Try to load LoRA config from checkpoint if available
            lora_config_dict = None
            if 'sft_config' in checkpoint and hasattr(checkpoint['sft_config'], 'use_lora'):
                sft_config = checkpoint['sft_config']
                if sft_config.use_lora:
                    lora_config_dict = {
                        'use_lora': True,
                        'lora_preset': sft_config.lora_preset,
                        'lora_target_modules': sft_config.lora_target_modules,
                        'lora_r': sft_config.lora_r,
                        'lora_alpha': sft_config.lora_alpha,
                        'lora_dropout': sft_config.lora_dropout
                    }
                    print(f"✓ Found LoRA config: preset={sft_config.lora_preset}, r={sft_config.lora_r}")
            elif 'rlhf_config' in checkpoint and hasattr(checkpoint['rlhf_config'], 'use_lora'):
                rlhf_config = checkpoint['rlhf_config']
                if rlhf_config.use_lora:
                    lora_config_dict = {
                        'use_lora': True,
                        'lora_preset': rlhf_config.lora_preset,
                        'lora_target_modules': rlhf_config.lora_target_modules,
                        'lora_r': rlhf_config.lora_r,
                        'lora_alpha': rlhf_config.lora_alpha,
                        'lora_dropout': rlhf_config.lora_dropout
                    }
                    print(f"✓ Found LoRA config: preset={rlhf_config.lora_preset}, r={rlhf_config.lora_r}")

            if lora_config_dict is None:
                print("⚠️  Could not find LoRA config in checkpoint")
                print("   Cannot automatically merge adapters")
                print("\n💡 Tip: If this is a LoRA checkpoint, it should contain SFTConfig or RLHFConfig")
                return

            # Apply LoRA to create PEFT model
            from model.lora_utils import apply_lora_to_model
            print("\n🔧 Applying LoRA configuration...")
            peft_model = apply_lora_to_model(base_model, model_config, lora_config_dict)

            # Load the state dict into PEFT model
            print("📥 Loading LoRA weights...")
            peft_model.load_state_dict(state_dict)

            # Merge adapters
            print("\n🔀 Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()

            print("✓ Successfully merged LoRA adapters!")

        except Exception as e:
            print(f"❌ Failed to merge LoRA adapters: {e}")
            print("\nThis might happen if:")
            print("  1. The checkpoint doesn't have proper LoRA parameters")
            print("  2. The LoRA configuration is incompatible")
            print("  3. PEFT library version mismatch")
            return

        # Save merged model
        output_path = get_input("\n💾 Save merged model to",
                               default=checkpoint_path.replace('.pt', '_merged.pt'))

        print(f"\n💾 Saving merged model to {output_path}...")

        # Prepare checkpoint data (without LoRA-specific info)
        merged_checkpoint = {
            'model_state_dict': merged_model.state_dict(),
            'model_config': model_config,
        }

        # Optionally include other metadata (but remove LoRA configs)
        if 'step' in checkpoint:
            merged_checkpoint['step'] = checkpoint['step']
        if 'best_val_loss' in checkpoint:
            merged_checkpoint['best_val_loss'] = checkpoint['best_val_loss']
        if 'eval_metrics' in checkpoint:
            merged_checkpoint['eval_metrics'] = checkpoint['eval_metrics']
        if 'final_metrics' in checkpoint:
            merged_checkpoint['final_metrics'] = checkpoint['final_metrics']

        torch.save(merged_checkpoint, output_path)

        print("\n✅ Successfully merged and saved!")
        print(f"   Merged checkpoint: {output_path}")
        print(f"   Original checkpoint: {checkpoint_path}")
        print("\n💡 You can now use the merged checkpoint for RLHF training without LoRA.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


def start_inference():
    """Start inference mode"""
    print_section_header("Interactive Inference Mode", "💬")

    checkpoint_path = get_input("Checkpoint path", default="checkpoints/best_model.pt")

    if not os.path.exists(checkpoint_path):
        print_error(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"\n{Colors.BOLD}{Colors.CYAN}Loading model...{Colors.RESET}")
    try:
        interactive_inference(checkpoint_path)
    except KeyboardInterrupt:
        print_warning("\nInference interrupted by user")
    except Exception as e:
        print_error(f"\nInference failed: {e}")
        raise


def configure_rlhf():
    """Interactive RLHF configuration"""
    print("\n" + "-" * 60)
    print("RLHF Configuration")
    print("-" * 60)

    config = RLHFConfig()

    # Algorithm selection
    print("\n🧠 Algorithm Selection")
    print("Options: ppo, dpo, grpo")
    config.algorithm = get_input(
        "Algorithm",
        default=config.algorithm
    ).lower()

    if config.algorithm not in ["ppo", "dpo", "grpo"]:
        print(f"❌ Invalid algorithm: {config.algorithm}")
        print("   Must be 'ppo', 'dpo', or 'grpo'")
        return None

    # Policy checkpoint
    print("\n🤖 Policy Model")
    config.policy_checkpoint = get_input(
        "Policy checkpoint path",
        default=config.policy_checkpoint
    )

    if not os.path.exists(config.policy_checkpoint):
        print(f"❌ Policy checkpoint not found: {config.policy_checkpoint}")
        return None

    # Algorithm-specific configuration
    if config.algorithm == "ppo":
        # Reward model (only for PPO)
        print("\n🎁 Reward Model")
        config.reward_model_name = get_input(
            "Reward model (HuggingFace)",
            default=config.reward_model_name
        )
    elif config.algorithm == "grpo":
        # Reward model (for GRPO)
        print("\n🎁 Reward Model")
        config.reward_model_name = get_input(
            "Reward model (HuggingFace)",
            default=config.reward_model_name
        )
        # GRPO-specific parameters
        print("\n🔢 GRPO Parameters")
        config.group_size = get_input(
            "Group size (number of responses per prompt)",
            default=config.group_size,
            type_fn=int
        )
        config.grpo_temperature = get_input(
            "Generation temperature for GRPO",
            default=config.grpo_temperature,
            type_fn=float
        )
    elif config.algorithm == "dpo":
        # Reference model (only for DPO)
        print("\n📚 Reference Model")
        print("Leave empty to use the same checkpoint as policy model")
        reference_checkpoint = get_input(
            "Reference checkpoint path (optional)",
            default=""
        )
        if reference_checkpoint and reference_checkpoint.strip():
            config.reference_checkpoint = reference_checkpoint.strip()
            if not os.path.exists(config.reference_checkpoint):
                print(f"⚠️  Reference checkpoint not found: {config.reference_checkpoint}")
                print("   Will use policy checkpoint as reference")
                config.reference_checkpoint = None

    # Training parameters
    print("\n⚙️  Training Hyperparameters")
    config.batch_size = get_input("Batch size", default=config.batch_size, type_fn=int)
    config.mini_batch_size = get_input("Mini-batch size", default=config.mini_batch_size, type_fn=int)

    if config.algorithm == "ppo":
        config.ppo_epochs = get_input("PPO epochs per batch", default=config.ppo_epochs, type_fn=int)

    # Optimizer
    print(f"\n🔧 Optimizer")
    print(f"Options: {', '.join(OPTIMIZER_NAMES)}")
    config.optimizer = get_input("Optimizer", default=config.optimizer)
    config.learning_rate = get_input("Learning rate", default=config.learning_rate, type_fn=float)
    config.weight_decay = get_input("Weight decay", default=config.weight_decay, type_fn=float)

    # Optimizer-specific parameters
    if config.optimizer.lower() == "adamw":
        print(f"\n   AdamW-specific parameters")
        config.adamw_beta1 = get_input("   Beta1", default=config.adamw_beta1, type_fn=float)
        config.adamw_beta2 = get_input("   Beta2", default=config.adamw_beta2, type_fn=float)
        config.adamw_eps = get_input("   Epsilon", default=config.adamw_eps, type_fn=float)
    elif config.optimizer.lower() == "muon":
        print(f"\n   Muon-specific parameters")
        config.muon_momentum = get_input("   Momentum", default=config.muon_momentum, type_fn=float)
        nesterov_input = get_input("   Use Nesterov? [y/n]", default="y" if config.muon_nesterov else "n")
        config.muon_nesterov = nesterov_input.lower() in ['y', 'yes']
    elif config.optimizer.lower() == "lion":
        print(f"\n   Lion-specific parameters")
        config.lion_beta1 = get_input("   Beta1", default=config.lion_beta1, type_fn=float)
        config.lion_beta2 = get_input("   Beta2", default=config.lion_beta2, type_fn=float)
    elif config.optimizer.lower() == "sophia":
        print(f"\n   Sophia-specific parameters")
        config.sophia_beta1 = get_input("   Beta1", default=config.sophia_beta1, type_fn=float)
        config.sophia_beta2 = get_input("   Beta2", default=config.sophia_beta2, type_fn=float)
        config.sophia_rho = get_input("   Rho (clipping)", default=config.sophia_rho, type_fn=float)

    if config.algorithm == "ppo":
        config.clip_range = get_input("Clip range (epsilon)", default=config.clip_range, type_fn=float)
    elif config.algorithm == "dpo":
        config.clip_range = get_input("Beta parameter (controls strength)", default=config.clip_range, type_fn=float)

    # GAE parameters (only for PPO)
    if config.algorithm == "ppo":
        print("\n📊 GAE Parameters")
        config.gamma = get_input("Gamma (discount)", default=config.gamma, type_fn=float)
        config.gae_lambda = get_input("Lambda (GAE)", default=config.gae_lambda, type_fn=float)

    # Training steps
    print("\n🏃 Training")
    config.max_steps = get_input("Maximum training steps", default=config.max_steps, type_fn=int)

    # Generation parameters
    print("\n📝 Generation Parameters")
    config.max_new_tokens = get_input("Max new tokens", default=config.max_new_tokens, type_fn=int)
    config.temperature = get_input("Temperature", default=config.temperature, type_fn=float)

    # LoRA configuration
    print("\n🔧 LoRA (Parameter-Efficient Fine-Tuning)")
    use_lora = get_input("Use LoRA? [y/n]", default="n")
    config.use_lora = use_lora.lower() in ['y', 'yes']

    if config.use_lora:
        # Load model config to determine available presets
        import torch
        try:
            checkpoint = torch.load(config.policy_checkpoint, map_location="cpu", weights_only=False)
            model_config = checkpoint.get('model_config')

            if model_config:
                from model.lora_utils import get_available_presets
                presets = get_available_presets(model_config)

                print("\nAvailable LoRA presets:")
                for preset_name, description in presets.items():
                    print(f"  {preset_name}: {description}")

                config.lora_preset = get_input("LoRA preset", default=config.lora_preset)

                if config.lora_preset == "custom":
                    print("\nEnter target modules (comma-separated)")
                    print("Examples: q_proj,v_proj or gate_proj,up_proj,down_proj")
                    modules_str = input("Target modules: ").strip()
                    if modules_str:
                        config.lora_target_modules = [m.strip() for m in modules_str.split(',')]
            else:
                print("⚠️  Could not load model config from checkpoint")
                print("   Using default preset")
                config.lora_preset = get_input("LoRA preset", default=config.lora_preset)
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Using default preset")
            config.lora_preset = get_input("LoRA preset", default=config.lora_preset)

        config.lora_r = get_input("LoRA rank (r)", default=config.lora_r, type_fn=int)
        config.lora_alpha = get_input("LoRA alpha", default=config.lora_alpha, type_fn=int)
        config.lora_dropout = get_input("LoRA dropout", default=config.lora_dropout, type_fn=float)

    # Dataset configuration
    print("\n📚 Dataset Configuration")
    print("Enter datasets (one per line, empty line to finish)")
    print("Format: dataset_name | subset (optional) | split (optional)")
    print("\nExamples:")
    print("  Anthropic/hh-rlhf")
    print("  nvidia/HelpSteer3")

    datasets = []
    while True:
        dataset_input = input(f"\nDataset {len(datasets) + 1} (or press Enter to finish): ").strip()
        if not dataset_input:
            break

        parts = [p.strip() for p in dataset_input.split('|')]
        ds_config = {"name": parts[0]}

        if len(parts) > 1 and parts[1]:
            ds_config["subset"] = parts[1]
        if len(parts) > 2 and parts[2]:
            ds_config["split"] = parts[2]

        datasets.append(ds_config)
        print(f"  Added: {ds_config['name']}")

    if datasets:
        config.datasets = datasets
    else:
        print("Using default dataset: Anthropic/hh-rlhf")

    # Save config
    save_path = get_input("\n💾 Save RLHF config to", default="rlhf_config.json")
    config.save(save_path)

    print(f"\n✅ RLHF configuration saved to: {save_path}")
    return config


def start_rlhf_training():
    """Start RLHF training"""
    print("\n" + "-" * 60)
    print("Start RLHF Training")
    print("-" * 60)

    # Option to use existing config or create new
    choice = get_input(
        "\n1. Use existing RLHF config\n2. Configure new RLHF training\nChoice",
        default="1"
    )

    if choice == "2":
        config = configure_rlhf()
        if config is None:
            return
    else:
        config_path = get_input("RLHF config path", default="rlhf_config.json")
        if not os.path.exists(config_path):
            print(f"❌ RLHF config not found: {config_path}")
            return

        config = RLHFConfig.load(config_path)

    # Output directory
    config.output_dir = get_input("\nOutput directory", default=config.output_dir)

    # Confirm and start
    print("\n" + "=" * 60)
    print(f"Ready to start RLHF training with {config.algorithm.upper()}!")
    print(f"  Algorithm: {config.algorithm.upper()}")
    print(f"  Policy: {config.policy_checkpoint}")

    if config.algorithm == "ppo":
        print(f"  Reward model: {config.reward_model_name}")
    elif config.algorithm == "grpo":
        print(f"  Reward model: {config.reward_model_name}")
        print(f"  Group size: {config.group_size}")
        print(f"  GRPO temperature: {config.grpo_temperature}")
    elif config.algorithm == "dpo":
        ref_model = config.reference_checkpoint or config.policy_checkpoint
        print(f"  Reference model: {ref_model}")

    print(f"  Steps: {config.max_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print("=" * 60)

    confirm = get_input("\nStart RLHF training? [y/n]", default="y")
    if confirm.lower() not in ['y', 'yes']:
        print("RLHF training cancelled.")
        return

    # Start training
    try:
        train_rlhf(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  RLHF training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ RLHF training failed: {e}")
        raise


def main():
    """Main CLI loop"""
    print_header()

    while True:
        print_menu()
        choice = input(f"\n{Colors.BOLD}➤{Colors.RESET} {Colors.WHITE}Enter your choice {Colors.DIM}(1-7){Colors.RESET}: ").strip()

        if choice == "1":
            configure_model()
        elif choice == "2":
            start_training()
        elif choice == "3":
            start_sft_training()
        elif choice == "4":
            start_rlhf_training()
        elif choice == "5":
            merge_lora_adapters()
        elif choice == "6":
            start_inference()
        elif choice == "7":
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'─' * 60}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.PURPLE}{'Thank you for using LLM-Laboratory!':^60}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'─' * 60}{Colors.RESET}\n")
            sys.exit(0)
        else:
            print_error(f"Invalid choice: '{choice}'. Please enter a number between 1-7.")


if __name__ == "__main__":
    main()
