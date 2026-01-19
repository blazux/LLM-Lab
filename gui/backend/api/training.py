import asyncio
import sys
import threading
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from sse_starlette.sse import EventSourceResponse
import json
import queue
import time

# Import from existing LLM-Lab source
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / 'src'))

from config.config import ModelConfig, TrainingConfig
from config.sft_config import SFTConfig
from config.rlhf_config import RLHFConfig
from training.train import train_model, request_stop
from training.sft_train import train_sft
from training.rlhf_train import train_rlhf

router = APIRouter()

# Global training state
training_state = {
    "is_training": False,
    "current_step": 0,
    "max_steps": 0,
    "current_loss": None,
    "current_ppl": None,
    "current_lr": None,
    "status": "idle",
    "error": None,
    "model_config": None,  # Store full model configuration
    "training_config": None,  # Store full training configuration
    "training_type": None  # "pretraining", "sft", or "rlhf"
}

# Queue for metrics updates
metrics_queue = queue.Queue()

# Training thread
training_thread = None


class TrainingRequest(BaseModel):
    model_cfg: Dict[str, Any]
    training_cfg: Dict[str, Any]
    checkpoint_path: Optional[str] = None
    output_dir: str = "/app/data"
    additional_steps: int = 0


class RLHFRequest(BaseModel):
    algorithm: str  # "ppo", "dpo", "grpo"
    policy_checkpoint: str
    datasets: List[Dict[str, Any]]
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    mini_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    max_grad_norm: float
    log_every: int
    save_every: int
    eval_every: int
    output_dir: str
    # Generation parameters
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    # LoRA configuration
    use_lora: bool
    lora_preset: Optional[str] = "minimal"
    lora_target_modules: Optional[List[str]] = None
    lora_r: Optional[int] = 8
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.05
    # Optimizer-specific params
    adamw_beta1: Optional[float] = 0.9
    adamw_beta2: Optional[float] = 0.999
    adamw_eps: Optional[float] = 1e-8
    muon_momentum: Optional[float] = 0.95
    muon_nesterov: Optional[bool] = True
    lion_beta1: Optional[float] = 0.9
    lion_beta2: Optional[float] = 0.99
    sophia_beta1: Optional[float] = 0.965
    sophia_beta2: Optional[float] = 0.99
    sophia_rho: Optional[float] = 0.04
    # Algorithm-specific params
    reward_model_name: Optional[str] = None
    reference_checkpoint: Optional[str] = None
    ppo_epochs: Optional[int] = 4
    clip_range: Optional[float] = 0.2
    gamma: Optional[float] = 1.0
    gae_lambda: Optional[float] = 0.95
    vf_coef: Optional[float] = 0.1
    group_size: Optional[int] = 4
    grpo_temperature: Optional[float] = 1.0


class SFTRequest(BaseModel):
    policy_checkpoint: str
    datasets: List[Dict[str, Any]]
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    scheduler: str
    max_grad_norm: float
    log_every: int
    save_every: int
    eval_every: int
    eval_steps: int
    save_best_only: bool
    output_dir: str
    # Model override
    dropout: Optional[float] = None
    # LoRA configuration
    use_lora: bool
    lora_preset: Optional[str] = "minimal"
    lora_target_modules: Optional[List[str]] = None
    lora_r: Optional[int] = 8
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.05
    # Optimizer-specific params
    adamw_beta1: Optional[float] = 0.9
    adamw_beta2: Optional[float] = 0.999
    adamw_eps: Optional[float] = 1e-8
    muon_momentum: Optional[float] = 0.95
    muon_nesterov: Optional[bool] = True
    lion_beta1: Optional[float] = 0.9
    lion_beta2: Optional[float] = 0.99
    sophia_beta1: Optional[float] = 0.965
    sophia_beta2: Optional[float] = 0.99
    sophia_rho: Optional[float] = 0.04


class MetricsCallback:
    """Callback to capture training metrics"""

    def __init__(self, metrics_queue: queue.Queue):
        self.metrics_queue = metrics_queue
        self.step = 0

    def on_step(self, step: int, loss: float, lr: float, ppl: float = None):
        """Called after each training step"""
        self.step = step

        # Update global state
        training_state["current_step"] = step
        training_state["current_loss"] = loss
        training_state["current_lr"] = lr
        if ppl is not None:
            training_state["current_ppl"] = ppl

        # Send to queue for SSE
        metrics = {
            "type": "metrics",
            "step": step,
            "loss": float(loss),
            "lr": float(lr),
            "perplexity": float(ppl) if ppl is not None else None,
            "timestamp": time.time()
        }
        self.metrics_queue.put(metrics)

    def on_eval(self, step: int, eval_loss: float, eval_ppl: float):
        """Called after evaluation"""
        # Send eval metrics to queue for SSE
        eval_metrics = {
            "type": "eval_metrics",
            "step": step,
            "eval_loss": float(eval_loss),
            "eval_perplexity": float(eval_ppl),
            "timestamp": time.time()
        }
        self.metrics_queue.put(eval_metrics)

    def on_log(self, message: str, level: str = "info"):
        """Called when there's a log message"""
        log_data = {
            "type": "log",
            "level": level,
            "message": message,
            "timestamp": time.time()
        }
        self.metrics_queue.put(log_data)


def run_training(model_config_dict: Dict, train_config_dict: Dict,
                 checkpoint_path: Optional[str] = None, output_dir: str = "/app/data",
                 additional_steps: int = 0):
    """Run training in a separate thread"""
    try:
        # Clear CUDA cache before starting to free memory from previous runs
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Update training state
        training_state["is_training"] = True
        training_state["status"] = "starting"
        training_state["error"] = None
        training_state["model_config"] = model_config_dict
        training_state["training_config"] = train_config_dict
        training_state["training_type"] = "pretraining"

        # Handle checkpoint-based config loading or fresh training
        start_step = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Load checkpoint to extract config and step count
            try:
                metrics_queue.put({
                    "type": "log",
                    "level": "info",
                    "message": f"Loading checkpoint: {checkpoint_path}",
                    "timestamp": time.time()
                })

                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                start_step = checkpoint.get('step', 0)

                # Extract model config from checkpoint
                if 'model_config' not in checkpoint:
                    raise ValueError(f"Checkpoint does not contain model_config. Cannot resume training.")

                model_config = checkpoint['model_config']

                # Save checkpoint's config to disk (overwrite any existing config)
                model_config.save(f"{output_dir}/model_config.json")

                # Log that we're using checkpoint config, not UI config
                metrics_queue.put({
                    "type": "log",
                    "level": "info",
                    "message": f"✓ Using model config from checkpoint (ignoring UI form)",
                    "timestamp": time.time()
                })
                metrics_queue.put({
                    "type": "log",
                    "level": "info",
                    "message": f"Config from checkpoint - Tokenizer: {model_config.tokenizer_name}",
                    "timestamp": time.time()
                })
                metrics_queue.put({
                    "type": "log",
                    "level": "info",
                    "message": f"Config from checkpoint - Model: architecture={model_config.model_architecture}, d_model={model_config.d_model}, n_layers={model_config.n_layers}, attention={model_config.attention_type}, use_moe={model_config.use_moe}",
                    "timestamp": time.time()
                })

                # Update training state with checkpoint config
                training_state["model_config"] = model_config.__dict__

            except Exception as e:
                error_msg = f"Failed to load checkpoint config: {e}"
                metrics_queue.put({
                    "type": "log",
                    "level": "error",
                    "message": error_msg,
                    "timestamp": time.time()
                })
                raise ValueError(error_msg)
        else:
            # Fresh training - use config from UI
            metrics_queue.put({
                "type": "log",
                "level": "info",
                "message": f"Config received - Tokenizer: {model_config_dict.get('tokenizer_name', 'NOT SET')}",
                "timestamp": time.time()
            })
            metrics_queue.put({
                "type": "log",
                "level": "info",
                "message": f"Config received - Model: architecture={model_config_dict.get('model_architecture')}, d_model={model_config_dict.get('d_model')}, n_layers={model_config_dict.get('n_layers')}, attention={model_config_dict.get('attention_type')}",
                "timestamp": time.time()
            })

            model_config = ModelConfig(**model_config_dict)
            model_config.save(f"{output_dir}/model_config.json")

        # Create training config object
        train_config = TrainingConfig(**train_config_dict)

        # Update max_steps in state (this is the target step, not the count)
        if additional_steps > 0:
            # When using additional_steps, target is start_step + additional_steps
            training_state["max_steps"] = start_step + additional_steps
            metrics_queue.put({
                "type": "log",
                "level": "info",
                "message": f"Resuming from step {start_step}, training {additional_steps} more steps to reach step {start_step + additional_steps}",
                "timestamp": time.time()
            })
        else:
            training_state["max_steps"] = train_config.max_steps

        # Log start
        metrics_queue.put({
            "type": "log",
            "level": "success",
            "message": f"Starting training: {model_config.model_architecture} with {model_config.n_layers} layers",
            "timestamp": time.time()
        })

        training_state["status"] = "training"

        # Create callback for metrics
        callback = MetricsCallback(metrics_queue)

        # Run actual training with callback
        train_model(
            model_config=model_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            additional_steps=additional_steps,
            callback=callback
        )

        # Training completed
        metrics_queue.put({
            "type": "log",
            "level": "success",
            "message": "Training completed successfully!",
            "timestamp": time.time()
        })

        training_state["status"] = "completed"
        training_state["is_training"] = False

        # Clear CUDA cache after training completes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        # Training failed
        error_msg = str(e)
        training_state["error"] = error_msg
        training_state["status"] = "error"
        training_state["is_training"] = False

        metrics_queue.put({
            "type": "log",
            "level": "error",
            "message": f"Training failed: {error_msg}",
            "timestamp": time.time()
        })

        # Clear CUDA cache after training fails to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_sft(sft_config_dict: Dict):
    """Run SFT training in a separate thread"""
    try:
        # Clear CUDA cache before starting to free memory from previous runs
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Update training state
        training_state["is_training"] = True
        training_state["status"] = "starting"
        training_state["error"] = None
        training_state["model_config"] = None  # SFT loads from checkpoint
        training_state["training_config"] = sft_config_dict
        training_state["training_type"] = "sft"

        # Log received config
        metrics_queue.put({
            "type": "log",
            "level": "info",
            "message": f"SFT Config - Base model: {sft_config_dict.get('policy_checkpoint')}",
            "timestamp": time.time()
        })
        metrics_queue.put({
            "type": "log",
            "level": "info",
            "message": f"SFT Config - Datasets: {len(sft_config_dict.get('datasets', []))} dataset(s), LoRA: {sft_config_dict.get('use_lora', False)}",
            "timestamp": time.time()
        })

        # Create config object
        sft_config = SFTConfig(**sft_config_dict)

        # Update max_steps in state
        training_state["max_steps"] = sft_config.max_steps

        # Log start
        metrics_queue.put({
            "type": "log",
            "level": "success",
            "message": f"Starting SFT training from {sft_config.policy_checkpoint}" + (" with LoRA" if sft_config.use_lora else ""),
            "timestamp": time.time()
        })

        training_state["status"] = "training"

        # Create callback for metrics
        callback = MetricsCallback(metrics_queue)

        # Run actual SFT training with callback
        train_sft(
            config=sft_config,
            callback=callback
        )

        # Training completed
        metrics_queue.put({
            "type": "log",
            "level": "success",
            "message": "SFT training completed successfully!",
            "timestamp": time.time()
        })

        training_state["status"] = "completed"
        training_state["is_training"] = False

        # Clear CUDA cache after training completes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        # Training failed
        error_msg = str(e)
        training_state["error"] = error_msg
        training_state["status"] = "error"
        training_state["is_training"] = False

        metrics_queue.put({
            "type": "log",
            "level": "error",
            "message": f"SFT training failed: {error_msg}",
            "timestamp": time.time()
        })

        # Clear CUDA cache after training fails to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@router.post("/start")
async def start_training(request: TrainingRequest):
    """Start training in the background"""
    global training_thread

    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")

    # Calculate correct max_steps for progress tracking
    target_max_steps = request.training_cfg.get("max_steps", 10000)

    # If resuming from checkpoint with additional steps, calculate target
    if request.checkpoint_path and request.additional_steps > 0:
        try:
            import torch
            checkpoint = torch.load(request.checkpoint_path, map_location="cpu", weights_only=False)
            start_step = checkpoint.get('step', 0)
            target_max_steps = start_step + request.additional_steps
        except Exception as e:
            print(f"Warning: Could not load checkpoint for step calculation: {e}")

    # Reset state
    training_state.update({
        "is_training": True,
        "current_step": 0,
        "max_steps": target_max_steps,
        "current_loss": None,
        "current_ppl": None,
        "current_lr": None,
        "status": "starting",
        "error": None
    })

    # Clear old metrics
    while not metrics_queue.empty():
        metrics_queue.get()

    # Start training in background thread
    training_thread = threading.Thread(
        target=run_training,
        args=(
            request.model_cfg,
            request.training_cfg,
            request.checkpoint_path,
            request.output_dir,
            request.additional_steps
        ),
        daemon=True
    )
    training_thread.start()

    return {"success": True, "message": "Training started"}


@router.post("/sft/start")
async def start_sft_training(request: SFTRequest):
    """Start SFT training in the background"""
    global training_thread

    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")

    # Reset state
    training_state.update({
        "is_training": True,
        "current_step": 0,
        "max_steps": request.max_steps,
        "current_loss": None,
        "current_ppl": None,
        "current_lr": None,
        "status": "starting",
        "error": None
    })

    # Clear old metrics
    while not metrics_queue.empty():
        metrics_queue.get()

    # Convert request to dict for config
    sft_config_dict = request.dict()

    # Start SFT training in background thread
    training_thread = threading.Thread(
        target=run_sft,
        args=(sft_config_dict,),
        daemon=True
    )
    training_thread.start()

    return {"success": True, "message": "SFT training started"}


def run_rlhf(rlhf_config_dict: Dict):
    """Run RLHF training in a separate thread"""
    try:
        # Clear CUDA cache before starting to free memory from previous runs
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Update training state
        training_state["is_training"] = True
        training_state["status"] = "starting"
        training_state["error"] = None
        training_state["model_config"] = None  # RLHF loads from checkpoint
        training_state["training_config"] = rlhf_config_dict
        training_state["training_type"] = "rlhf"

        # Log received config
        metrics_queue.put({
            "type": "log",
            "level": "info",
            "message": f"RLHF Config - Algorithm: {rlhf_config_dict.get('algorithm').upper()}, Policy: {rlhf_config_dict.get('policy_checkpoint')}",
            "timestamp": time.time()
        })

        if rlhf_config_dict.get('algorithm') == 'ppo':
            metrics_queue.put({
                "type": "log",
                "level": "info",
                "message": f"PPO - Reward Model: {rlhf_config_dict.get('reward_model_name')}, Epochs: {rlhf_config_dict.get('ppo_epochs')}",
                "timestamp": time.time()
            })
        elif rlhf_config_dict.get('algorithm') == 'dpo':
            ref_model = rlhf_config_dict.get('reference_checkpoint') or rlhf_config_dict.get('policy_checkpoint')
            metrics_queue.put({
                "type": "log",
                "level": "info",
                "message": f"DPO - Reference Model: {ref_model}, Beta: {rlhf_config_dict.get('clip_range')}",
                "timestamp": time.time()
            })
        elif rlhf_config_dict.get('algorithm') == 'grpo':
            metrics_queue.put({
                "type": "log",
                "level": "info",
                "message": f"GRPO - Reward Model: {rlhf_config_dict.get('reward_model_name')}, Group Size: {rlhf_config_dict.get('group_size')}",
                "timestamp": time.time()
            })

        # Create config object
        rlhf_config = RLHFConfig(**rlhf_config_dict)

        # Update max_steps in state
        training_state["max_steps"] = rlhf_config.max_steps

        # Log start
        lora_status = " with LoRA" if rlhf_config.use_lora else ""
        metrics_queue.put({
            "type": "log",
            "level": "success",
            "message": f"Starting {rlhf_config.algorithm.upper()} training from {rlhf_config.policy_checkpoint}{lora_status}",
            "timestamp": time.time()
        })

        training_state["status"] = "training"

        # Create callback for metrics
        callback = MetricsCallback(metrics_queue)

        # Run actual RLHF training with callback
        train_rlhf(
            config=rlhf_config,
            callback=callback
        )

        # Training completed
        metrics_queue.put({
            "type": "log",
            "level": "success",
            "message": f"{rlhf_config.algorithm.upper()} training completed successfully!",
            "timestamp": time.time()
        })

        training_state["status"] = "completed"
        training_state["is_training"] = False

        # Clear CUDA cache after training completes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        # Training failed
        error_msg = str(e)
        training_state["error"] = error_msg
        training_state["status"] = "error"
        training_state["is_training"] = False

        metrics_queue.put({
            "type": "log",
            "level": "error",
            "message": f"RLHF training failed: {error_msg}",
            "timestamp": time.time()
        })

        # Clear CUDA cache after training fails to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@router.post("/rlhf/start")
async def start_rlhf_training(request: RLHFRequest):
    """Start RLHF training in the background"""
    global training_thread

    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")

    # Reset state
    training_state.update({
        "is_training": True,
        "current_step": 0,
        "max_steps": request.max_steps,
        "current_loss": None,
        "current_ppl": None,
        "current_lr": None,
        "status": "starting",
        "error": None
    })

    # Clear old metrics
    while not metrics_queue.empty():
        metrics_queue.get()

    # Convert request to dict for config
    rlhf_config_dict = request.dict()

    # Start RLHF training in background thread
    training_thread = threading.Thread(
        target=run_rlhf,
        args=(rlhf_config_dict,),
        daemon=True
    )
    training_thread.start()

    return {"success": True, "message": f"{request.algorithm.upper()} training started"}


@router.post("/stop")
async def stop_training():
    """Stop training gracefully"""
    if not training_state["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")

    # Request graceful stop via the global flag
    request_stop()
    training_state["status"] = "stopping"

    return {"success": True, "message": "Stop requested - training will stop after current step"}


@router.get("/status")
async def get_status():
    """Get current training status"""
    return training_state


@router.get("/metrics/stream")
async def stream_metrics():
    """Stream training metrics via Server-Sent Events"""

    async def event_generator():
        """Generate SSE events from metrics queue"""
        try:
            # On new connection, send current state snapshot
            snapshot = {
                "type": "snapshot",
                "is_training": training_state["is_training"],
                "current_step": training_state["current_step"],
                "max_steps": training_state["max_steps"],
                "current_loss": training_state["current_loss"],
                "current_ppl": training_state["current_ppl"],
                "current_lr": training_state["current_lr"],
                "status": training_state["status"],
                "error": training_state["error"],
                "model_config": training_state["model_config"],
                "training_config": training_state["training_config"],
                "training_type": training_state["training_type"],
                "timestamp": time.time()
            }
            yield {
                "event": "snapshot",
                "data": json.dumps(snapshot)
            }

            # Clear only very old backlogged metrics from queue (older than 30 seconds)
            # This keeps recent training logs while preventing flooding with stale data
            current_time = time.time()
            max_age = 30  # seconds
            cleared_count = 0
            kept_metrics = []

            while not metrics_queue.empty():
                try:
                    metric = metrics_queue.get_nowait()
                    # Keep metrics from last 30 seconds
                    if metric.get('timestamp', 0) > current_time - max_age:
                        kept_metrics.append(metric)
                    else:
                        cleared_count += 1
                except queue.Empty:
                    break

            # Put kept metrics back in queue
            for metric in kept_metrics:
                metrics_queue.put(metric)

            # Log if we cleared metrics (useful for debugging)
            if cleared_count > 0:
                print(f"SSE reconnect: Cleared {cleared_count} old metrics (>{max_age}s), kept {len(kept_metrics)} recent ones")

            # Now stream only new real-time metrics
            while True:
                # Check if there are metrics in queue
                try:
                    # Non-blocking get with timeout
                    metric = metrics_queue.get(timeout=0.1)
                    yield {
                        "event": "message",
                        "data": json.dumps(metric)
                    }
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    if not training_state["is_training"] and training_state["status"] in ["completed", "error"]:
                        # Training ended, send final status and close
                        yield {
                            "event": "status",
                            "data": json.dumps({
                                "type": "status",
                                "status": training_state["status"],
                                "error": training_state.get("error")
                            })
                        }
                        break

                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            # Client disconnected
            pass

    return EventSourceResponse(event_generator())
