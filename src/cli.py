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


def print_header():
    """Print CLI header"""
    print("\n" + "=" * 60)
    print(" " * 15 + "🧠 LLM-Laboratory")
    print("=" * 60)


def print_menu():
    """Print main menu"""
    print("\nMain Menu:")
    print("  1. Configure new model")
    print("  2. Base training")
    print("  3. SFT training (Supervised Fine-Tuning)")
    print("  4. RLHF training (PPO/DPO/GRPO)")
    print("  5. Merge LoRA adapters")
    print("  6. Test model (inference)")
    print("  7. Exit")


def get_input(prompt: str, default=None, type_fn=str):
    """Get user input with optional default value"""
    if default is not None:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        if not user_input:
            return default
    else:
        user_input = input(f"{prompt}: ").strip()

    try:
        return type_fn(user_input)
    except ValueError:
        print(f"Invalid input, using default: {default}")
        return default


def configure_model():
    """Interactive model configuration"""
    print("\n" + "-" * 60)
    print("Model Configuration")
    print("-" * 60)

    config = ModelConfig()

    # Tokenizer
    print("\n📖 Tokenizer Configuration")
    config.tokenizer_name = get_input(
        "Tokenizer name (HuggingFace)",
        default=config.tokenizer_name
    )

    # Architecture choices
    print("\n🏗️  Architecture Choices")

    print(f"\nPositional Encoding options: {', '.join(POSITIONAL_ENCODINGS.keys())}")
    config.positional_encoding = get_input(
        "Positional encoding",
        default=config.positional_encoding
    )

    print(f"\nAttention type options: {', '.join(ATTENTION_TYPES.keys())}")
    config.attention_type = get_input(
        "Attention type",
        default=config.attention_type
    )

    print(f"\nNormalization options: layernorm, rmsnorm")
    config.norm_type = get_input(
        "Normalization type",
        default=config.norm_type
    )

    print(f"\nActivation options: {', '.join(ACTIVATION_TYPES.keys())}")
    config.activation = get_input(
        "Feed-forward activation",
        default=config.activation
    )

    # Model parameters
    print("\n⚙️  Model Parameters")
    config.d_model = get_input("Embedding size (d_model)", default=config.d_model, type_fn=int)
    config.n_heads = get_input("Number of attention heads", default=config.n_heads, type_fn=int)

    if config.attention_type == "gqa":
        config.n_kv_heads = get_input("Number of KV heads (for GQA)", default=config.n_kv_heads, type_fn=int)

    config.d_ff = get_input("Feed-forward hidden size (d_ff)", default=config.d_ff, type_fn=int)
    config.n_layers = get_input("Number of layers", default=config.n_layers, type_fn=int)
    config.max_seq_len = get_input("Maximum sequence length", default=config.max_seq_len, type_fn=int)
    config.dropout = get_input("Dropout rate", default=config.dropout, type_fn=float)

    # Save config
    save_path = get_input("\n💾 Save config to", default="model_config.json")
    config.save(save_path)

    # Show parameter count
    param_count = config.count_params()
    print(f"\n✅ Model configured!")
    print(f"   Estimated parameters: {param_count:,}")
    print(f"   Config saved to: {save_path}")


def configure_training():
    """Interactive base training configuration"""
    print("\n" + "-" * 60)
    print("Base Training Configuration")
    print("-" * 60)

    # Load model config
    model_config_path = get_input("\n📖 Model config path", default="model_config.json")

    if not os.path.exists(model_config_path):
        print(f"❌ Model config not found: {model_config_path}")
        print("   Please configure a model first.")
        return None, None

    model_config = ModelConfig.load(model_config_path)
    train_config = TrainingConfig()
    train_config.model_config_path = model_config_path

    # Training steps
    print("\n🏃 Training Steps")
    train_config.max_steps = get_input("Maximum training steps", default=train_config.max_steps, type_fn=int)

    # Optimizer
    print(f"\n🔧 Optimizer")
    print(f"Options: {', '.join(OPTIMIZER_NAMES)}")
    train_config.optimizer = get_input("Optimizer", default=train_config.optimizer)
    train_config.lr = get_input("Learning rate", default=train_config.lr, type_fn=float)
    train_config.weight_decay = get_input("Weight decay", default=train_config.weight_decay, type_fn=float)

    # Scheduler
    print(f"\n📈 Learning Rate Scheduler")
    print("Options: none, cosine, linear, polynomial")
    train_config.scheduler = get_input("Scheduler", default=train_config.scheduler)
    train_config.warmup_steps = get_input("Warmup steps", default=train_config.warmup_steps, type_fn=int)

    # Batch and accumulation
    print("\n📦 Batch Configuration")
    train_config.batch_size = get_input("Batch size", default=train_config.batch_size, type_fn=int)
    train_config.gradient_accumulation_steps = get_input(
        "Gradient accumulation steps",
        default=train_config.gradient_accumulation_steps,
        type_fn=int
    )
    train_config.grad_clip = get_input("Gradient clipping", default=train_config.grad_clip, type_fn=float)

    # Evaluation
    print("\n📊 Evaluation")
    train_config.eval_every = get_input("Eval every N steps", default=train_config.eval_every, type_fn=int)
    train_config.eval_steps = get_input("Steps per evaluation", default=train_config.eval_steps, type_fn=int)

    save_best = get_input("Save best model only? [y/n]", default="y")
    train_config.save_best_only = save_best.lower() in ['y', 'yes']

    # Dataset configuration
    print("\n📚 Dataset Configuration")
    print("Enter datasets (one per line, empty line to finish)")
    print("Format: dataset_name | subset (optional) | weight (optional)")
    print("\nExamples:")
    print("  HuggingFaceFW/fineweb-edu")
    print("  HuggingFaceFW/fineweb-2 | fra_Latn | 1.0")
    print("  HuggingFaceFW/fineweb-2 | eng_Latn | 2.0")

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
            ds_config["weight"] = float(parts[2])

        datasets.append(ds_config)
        print(f"  Added: {ds_config['name']}" + (f" ({ds_config.get('subset', 'no subset')})" if 'subset' in ds_config or len(parts) > 1 else ""))

    if datasets:
        train_config.datasets = datasets
    else:
        print("Using default dataset: HuggingFaceFW/fineweb-edu")

    # Save config
    save_path = get_input("\n💾 Save training config to", default="training_config.json")
    train_config.save(save_path)

    print(f"\n✅ Training configuration saved to: {save_path}")

    return model_config, train_config


def start_training():
    """Start base training with configuration"""
    print("\n" + "-" * 60)
    print("Start Base Training")
    print("-" * 60)

    # Option to use existing config or create new
    choice = get_input(
        "\n1. Use existing training config\n2. Configure new training\nChoice",
        default="1"
    )

    if choice == "2":
        model_config, train_config = configure_training()
        if model_config is None:
            return
    else:
        train_config_path = get_input("Training config path", default="training_config.json")
        if not os.path.exists(train_config_path):
            print(f"❌ Training config not found: {train_config_path}")
            return

        train_config = TrainingConfig.load(train_config_path)

        if not os.path.exists(train_config.model_config_path):
            print(f"❌ Model config not found: {train_config.model_config_path}")
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
                load_opt = get_input("   Load optimizer state from checkpoint? [y/n] (say 'n' if switching optimizers)", default="y")
                load_optimizer_state = load_opt.lower() in ['y', 'yes']
            except Exception as e:
                print(f"⚠️  Could not read checkpoint: {e}")
                checkpoint_path = None

    # Output directory
    output_dir = get_input("\nOutput directory", default="checkpoints")

    # Confirm and start
    print("\n" + "=" * 60)
    print("Ready to start base training!")
    print(f"  Model: {model_config.d_model}d, {model_config.n_layers}L, {model_config.n_heads}H")
    if checkpoint_path and additional_steps > 0:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        current_step = ckpt.get('step', 0)
        print(f"  Resuming from: step {current_step}")
        print(f"  Target steps: {current_step + additional_steps}")
    elif checkpoint_path:
        print(f"  Steps: {train_config.max_steps}")
    else:
        print(f"  Steps: {train_config.max_steps}")
    print(f"  Optimizer: {train_config.optimizer}")
    print(f"  LR: {train_config.lr}")
    print("=" * 60)

    confirm = get_input("\nStart base training? [y/n]", default="y")
    if confirm.lower() not in ['y', 'yes']:
        print("Base training cancelled.")
        return

    # Start base training
    try:
        train_model(model_config, train_config, checkpoint_path, output_dir, additional_steps, load_optimizer_state)
    except KeyboardInterrupt:
        print("\n\n⚠️  Base training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Base training failed: {e}")
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
    print("  HuggingFaceFW/fineweb-edu-sft | | train | 2.0")

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
    print("\n" + "-" * 60)
    print("Merge LoRA Adapters")
    print("-" * 60)

    print("\nThis tool merges LoRA adapter weights back into the base model.")
    print("Use this after LoRA training to create a standard checkpoint for RLHF.\n")

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
    print("\n" + "-" * 60)
    print("Inference Mode")
    print("-" * 60)

    checkpoint_path = get_input("\nCheckpoint path", default="checkpoints/best_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    try:
        interactive_inference(checkpoint_path)
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Inference failed: {e}")
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

    config.learning_rate = get_input("Learning rate", default=config.learning_rate, type_fn=float)

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
        choice = input("\nChoice: ").strip()

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
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
