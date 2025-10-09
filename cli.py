#!/usr/bin/env python3
import os
import sys

from config import ModelConfig, TrainingConfig
from train import train_model
from inference import interactive_inference
from bricks import POSITIONAL_ENCODINGS, ATTENTION_TYPES, ACTIVATION_TYPES
from optimizers import OPTIMIZER_NAMES


def print_header():
    """Print CLI header"""
    print("\n" + "=" * 60)
    print(" " * 15 + "🧠 Simple LLM Builder")
    print("=" * 60)


def print_menu():
    """Print main menu"""
    print("\nMain Menu:")
    print("  1. Configure new model")
    print("  2. Train model")
    print("  3. Test model (inference)")
    print("  4. Exit")


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
    """Interactive training configuration"""
    print("\n" + "-" * 60)
    print("Training Configuration")
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
    """Start training with configuration"""
    print("\n" + "-" * 60)
    print("Start Training")
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
            except Exception as e:
                print(f"⚠️  Could not read checkpoint: {e}")
                checkpoint_path = None

    # Output directory
    output_dir = get_input("\nOutput directory", default="checkpoints")

    # Confirm and start
    print("\n" + "=" * 60)
    print("Ready to start training!")
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

    confirm = get_input("\nStart training? [y/n]", default="y")
    if confirm.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return

    # Start training
    try:
        train_model(model_config, train_config, checkpoint_path, output_dir, additional_steps)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
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
            start_inference()
        elif choice == "4":
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
