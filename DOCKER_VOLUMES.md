# Docker Volume Configuration

## Checkpoints Directory

All training artifacts (configs, checkpoints, models) are saved to the `checkpoints/` directory with the following structure:

```
checkpoints/
├── best_model.pt              # Best checkpoint based on validation loss
└── latest_checkpoint.pt       # Latest checkpoint (if save_best_only=False)
```

## Running with Docker

### Mount the checkpoints directory to persist your training results:

```bash
docker run -v $(pwd)/checkpoints:/app/gui/backend/checkpoints -p 8000:8000 llm-lab
```

### With GPU support:

```bash
docker run --gpus all -v $(pwd)/checkpoints:/app/gui/backend/checkpoints -p 8000:8000 llm-lab
```

### Also mount cache to avoid re-downloading datasets:

```bash
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/gui/backend/checkpoints \
  -v $(pwd)/cache:/app/cache \
  -p 8000:8000 \
  llm-lab
```

## Training Pipeline

All training stages save their checkpoints to `checkpoints/best_model.pt`:

1. **Pretraining** → saves to `checkpoints/best_model.pt`
2. **SFT** → loads from `checkpoints/best_model.pt`, saves updated model to `checkpoints/best_model.pt`
3. **RLHF** → loads from `checkpoints/best_model.pt`, saves updated model to `checkpoints/best_model.pt`

## Accessing Your Models

After training in Docker, all your models will be available in the `checkpoints/` directory on your host machine. You can:

- Copy checkpoints to another machine
- Resume training by mounting the same `checkpoints/` directory
- Load models for inference using the saved checkpoints

## Cache Directory

The cache directory stores downloaded datasets and tokenizer files. Mounting it avoids re-downloading:

```bash
-v $(pwd)/cache:/app/cache
```
