# Docker Volume Configuration

## Centralized Outputs Directory

All training artifacts (configs, checkpoints, models) are now saved to a single `outputs/` directory with the following structure:

```
outputs/
├── pretraining/
│   ├── model_config.json        # Model architecture configuration
│   ├── training_config.json     # Training hyperparameters
│   ├── best_model.pt           # Best checkpoint based on validation loss
│   └── latest_checkpoint.pt    # Latest checkpoint (if save_best_only=False)
│
├── sft/
│   ├── sft_config.json         # SFT configuration
│   ├── best_model.pt           # Best SFT checkpoint
│   └── latest_checkpoint.pt    # Latest SFT checkpoint
│
└── rlhf/
    ├── rlhf_config.json        # RLHF configuration (PPO/DPO/GRPO)
    ├── best_model.pt           # Best RLHF checkpoint
    └── latest_checkpoint.pt    # Latest RLHF checkpoint
```

## Running with Docker

### Mount the outputs directory to persist your training results:

```bash
docker run -v $(pwd)/outputs:/app/outputs -p 8000:8000 llm-lab
```

### With GPU support:

```bash
docker run --gpus all -v $(pwd)/outputs:/app/outputs -p 8000:8000 llm-lab
```

### Also mount cache to avoid re-downloading datasets:

```bash
docker run --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/cache:/app/cache \
  -p 8000:8000 \
  llm-lab
```

## Training Pipeline

The default checkpoint paths follow this flow:

1. **Pretraining** → saves to `outputs/pretraining/best_model.pt`
2. **SFT** → loads from `outputs/pretraining/best_model.pt`, saves to `outputs/sft/best_model.pt`
3. **RLHF** → loads from `outputs/sft/best_model.pt`, saves to `outputs/rlhf/best_model.pt`

## Accessing Your Models

After training in Docker, all your models and configs will be available in the `outputs/` directory on your host machine. You can:

- Copy checkpoints to another machine
- Version control your configs (JSON files)
- Resume training by mounting the same `outputs/` directory
- Load models for inference using the saved checkpoints

## Configuration Files

Each training stage automatically saves its configuration JSON file, making it easy to:
- Reproduce experiments
- Track hyperparameters
- Share configurations with others
- Resume or modify training runs
