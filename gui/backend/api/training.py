import asyncio
import sys
import threading
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Header, Query
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

from run_store import get_run_store, LANGUAGE_METRICS, LEGACY_METRICS, clean

router = APIRouter()


def make_config(cls, data: Dict[str, Any]):
    """Instantiate a config dataclass, ignoring keys it doesn't know about."""
    from dataclasses import fields
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def idle_state():
    return dict(is_training=False, current_step=0, max_steps=0, current_loss=None,
                current_ppl=None, current_lr=None, current_language_loss=None,
                status='idle', error=None, model_config=None, training_config=None,
                training_type=None, run_id=None, output_dir=None, metric_definition=None)


class PersistentState(dict):
    def update(self, *args, **kwargs):
        changes = clean(dict(*args, **kwargs))
        super().update(changes)
        if self.get('run_id'):
            get_run_store().update(self['run_id'], changes)

    def __setitem__(self, key, value):
        self.update({key: value})


training_state = PersistentState(idle_state())


def restore_state():
    if not training_state.get('run_id'):
        records = get_run_store().list()
        if records:
            dict.update(training_state, records[0])


class EventSink:
    """Compatibility with callbacks that previously wrote to a consuming queue."""
    def put(self, event):
        if training_state.get('run_id'):
            get_run_store().append(training_state['run_id'], {
                'metric_definition': training_state.get('metric_definition'), **event})

    def empty(self):
        return True


metrics_queue = EventSink()


def begin_run(request, kind, model, config):
    requested = request.output_dir
    output = None if requested == '/app/data' else str(Path(requested).resolve())
    if output:
        if any(run['output_dir'] == output for run in get_run_store().list()):
            raise HTTPException(409, 'This output folder already belongs to a recorded run. Choose a new folder.')
        if any((folder / name).exists() for folder in (Path(output), Path(output) / 'checkpoints')
               for name in ('final_model.pt', 'best_model.pt', 'latest_checkpoint.pt')):
            raise HTTPException(409, 'This folder already contains a trained model. Choose a new output folder; registered checkpoint resume creates one automatically.')
    definition = LANGUAGE_METRICS if kind == 'pretraining' else LEGACY_METRICS
    record = get_run_store().create({**idle_state(), 'is_training': True, 'status': 'starting',
        'output_dir': output, 'model_config': model, 'training_config': config,
        'training_type': kind, 'metric_definition': definition,
        'checkpoint_path': getattr(request, 'checkpoint_path', None)}, request.project_id)
    request.output_dir = record['output_dir']
    if kind != 'pretraining':
        config['output_dir'] = request.output_dir
        get_run_store().update(record['run_id'], {'training_config': config})
    dict.clear(training_state)
    dict.update(training_state, get_run_store().get(record['run_id']))
    return record['run_id']


# Training thread
training_thread = None


class TrainingRequest(BaseModel):
    project_id: Optional[str] = None
    model_cfg: Dict[str, Any]
    training_cfg: Dict[str, Any]
    checkpoint_path: Optional[str] = None
    output_dir: str = "/app/data"
    additional_steps: int = 0


class RLHFRequest(BaseModel):
    project_id: Optional[str] = None
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
    grpo_clip_higher: Optional[float] = None
    grpo_dynamic_sampling: Optional[bool] = False
    grpo_importance_level: Optional[str] = "token"


class SFTRequest(BaseModel):
    project_id: Optional[str] = None
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
    # Adaptive scheduler params
    adaptive_window: Optional[int] = 10
    adaptive_increase_factor: Optional[float] = 1.05
    adaptive_decrease_factor: Optional[float] = 0.9
    adaptive_patience: Optional[int] = 3
    adaptive_min_lr: Optional[float] = 1e-6
    adaptive_threshold: Optional[float] = 0.01
    max_grad_norm: float
    log_every: int
    save_every: int
    eval_every: int
    eval_steps: int
    save_best_only: bool
    output_dir: str
    # Loss
    loss_fn: Optional[str] = "cross_entropy"
    maxis_n_candidates: Optional[int] = 2048
    maxis_low_rank_dim: Optional[int] = 64
    maxis_chunk_size: Optional[int] = 32
    maxis_aux_weight: Optional[float] = 0.2
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

    def on_metrics(self, metrics):
        training_state.update(current_step=metrics['step'], current_loss=metrics['loss'],
            current_lr=metrics['lr'], current_ppl=metrics.get('perplexity'),
            current_language_loss=metrics.get('language_loss'))
        self.step = metrics['step']
        self.metrics_queue.put({'type': 'metrics', 'timestamp': time.time(), **metrics})

    def on_evaluation_metrics(self, metrics):
        self.metrics_queue.put({'type': 'eval_metrics', 'timestamp': time.time(), **metrics})

    def on_update(self, step: int):
        """Track every optimizer update, even between metric logging intervals."""
        self.step = step
        training_state['current_step'] = step

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
        os.makedirs(output_dir, exist_ok=True)
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
        load_optimizer_state = True  # Will be set to False if optimizer/scheduler changed
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

                # Check if optimizer/scheduler changed - if so, don't load optimizer state
                checkpoint_train_config = checkpoint.get('train_config')
                if checkpoint_train_config:
                    old_optimizer = getattr(checkpoint_train_config, 'optimizer', None)
                    old_scheduler = getattr(checkpoint_train_config, 'scheduler', None)
                    new_optimizer = train_config_dict.get('optimizer')
                    new_scheduler = train_config_dict.get('scheduler')

                    if old_optimizer != new_optimizer:
                        load_optimizer_state = False
                        metrics_queue.put({
                            "type": "log",
                            "level": "info",
                            "message": f"Optimizer changed ({old_optimizer} → {new_optimizer}), skipping optimizer state load",
                            "timestamp": time.time()
                        })
                    if old_scheduler != new_scheduler:
                        load_optimizer_state = False
                        metrics_queue.put({
                            "type": "log",
                            "level": "info",
                            "message": f"Scheduler changed ({old_scheduler} → {new_scheduler}), skipping scheduler state load",
                            "timestamp": time.time()
                        })

                del checkpoint  # Free memory

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

            model_config = make_config(ModelConfig, model_config_dict)
            model_config.save(f"{output_dir}/model_config.json")

        # Create training config object
        train_config = make_config(TrainingConfig, train_config_dict)

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

        training_state.update(model_config=model_config.__dict__, training_config={**train_config.__dict__, "max_steps": training_state["max_steps"]})

        # Create callback for metrics
        callback = MetricsCallback(metrics_queue)

        # Run actual training with callback
        train_model(
            model_config=model_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            additional_steps=additional_steps,
            load_optimizer_state=load_optimizer_state,
            callback=callback
        )

        # Training completed
        metrics_queue.put({
            "type": "log",
            "level": "success",
            "message": "Training stopped; checkpoint saved." if training_state.get("status") == "stopping" else "Training completed successfully!",
            "timestamp": time.time()
        })

        training_state["status"] = "stopped" if training_state.get("status") == "stopping" else "completed"
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
        sft_config = make_config(SFTConfig, sft_config_dict)

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

        training_state["status"] = "stopped" if training_state.get("status") == "stopping" else "completed"
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

    if request.model_cfg.get('assembly') is not None:
        from .assembly import config_from_request
        try:
            config_from_request(request.model_cfg)
        except (ValueError, TypeError, AssertionError) as error:
            raise HTTPException(422, detail={'message': str(error), 'node_id': getattr(error, 'node_id', None),
                                             'hint': getattr(error, 'hint', 'Fix the graph before starting training.')})

    if request.checkpoint_path:
        import torch
        from .assembly import config_from_request
        from model.assembly_runtime import assembly_signature
        try:
            saved = torch.load(request.checkpoint_path, map_location='cpu', weights_only=False)['model_config']
            incoming = request.model_cfg.get('assembly')
            if incoming is not None or getattr(saved, 'assembly', None) is not None:
                if (incoming is None or getattr(saved, 'assembly', None) is None
                        or assembly_signature(config_from_request(request.model_cfg)) != assembly_signature(saved)):
                    raise HTTPException(409, 'This graph differs from the checkpoint. Restore its graph or start a new training run.')
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(422, f'Could not resume this checkpoint: {error}')

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

    run_id = begin_run(request, 'pretraining', request.model_cfg, request.training_cfg)

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

    return {"success": True, "message": "Training started", "run_id": run_id}


@router.post("/sft/start")
async def start_sft_training(request: SFTRequest):
    """Start SFT training in the background"""
    global training_thread

    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")

    run_id = begin_run(request, 'sft', None, request.model_dump())

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

    return {"success": True, "message": "SFT training started", "run_id": run_id}


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
        rlhf_config = make_config(RLHFConfig, rlhf_config_dict)

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

        training_state["status"] = "stopped" if training_state.get("status") == "stopping" else "completed"
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

    run_id = begin_run(request, 'rlhf', None, request.model_dump())

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

    return {"success": True, "message": f"{request.algorithm.upper()} training started", "run_id": run_id}


@router.post("/stop")
async def stop_training(run_id: Optional[str] = None):
    if run_id and run_id != training_state.get("run_id"):
        raise HTTPException(409, "This is not the active run.")
    """Stop training gracefully"""
    if not training_state["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")

    if training_state.get('training_type') == 'learning':
        from .learning import stop_run
        return await stop_run(training_state['run_id'])

    # Request graceful stop via the global flag
    request_stop()
    training_state["status"] = "stopping"

    return {"success": True, "message": "Stop requested - training will stop after current step"}


@router.get('/status')
async def get_status():
    restore_state()
    return training_state


@router.get('/runs')
async def list_runs(project_id: Optional[str] = None):
    return {'runs': get_run_store().list(project_id)}


def require_run(run_id):
    try:
        return get_run_store().get(run_id)
    except KeyError:
        raise HTTPException(404, 'Run not found')


class ImportRunRequest(BaseModel):
    output_dir: str
    project_id: Optional[str] = None


class ResumeRunRequest(BaseModel):
    additional_steps: int = 0


def run_checkpoints(run):
    root = Path(run['output_dir'])
    files = [folder / name for folder in (root, root / 'checkpoints')
             for name in ('final_model.pt', 'latest_checkpoint.pt', 'best_model.pt')]
    return sorted((path for path in files if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)


@router.post('/runs/import')
async def import_run(request: ImportRunRequest):
    from dataclasses import asdict, is_dataclass
    import torch
    root = Path(request.output_dir).resolve()
    for existing in get_run_store().list():
        if existing['output_dir'] == str(root):
            if existing.get('project_id') not in (None, request.project_id):
                raise HTTPException(409, 'This folder already belongs to another project.')
            if request.project_id:
                get_run_store().update(existing['run_id'], {'project_id': request.project_id})
            return {'run_id': existing['run_id']}
    files = run_checkpoints({'output_dir': str(root)})
    if not files:
        raise HTTPException(422, 'No saved training checkpoint found in this folder or its checkpoints subfolder.')
    try:
        saved = torch.load(files[0], map_location='cpu', weights_only=False)
        model, config = saved['model_config'], saved['train_config']
        model = asdict(model) if is_dataclass(model) else dict(model)
        config = asdict(config) if is_dataclass(config) else dict(config)
        step = saved.get('step', 0)
        definition = saved.get('metric_definition', LEGACY_METRICS)
        record = get_run_store().create({**idle_state(), 'output_dir': str(root),
            'status': 'completed' if step >= config.get('max_steps', step) else 'stopped',
            'current_step': step, 'max_steps': config.get('max_steps', step),
            'model_config': model, 'training_config': config, 'training_type': 'pretraining',
            'metric_definition': definition, 'checkpoint_path': str(files[0])}, request.project_id)
        metrics = saved.get('final_metrics', saved.get('eval_metrics', {}))
        if 'val_loss' in metrics:
            get_run_store().append(record['run_id'], {'type': 'eval_metrics', 'step': step,
                'eval_loss': metrics['val_loss'], 'eval_perplexity': metrics.get('val_perplexity'),
                'eval_language_loss': metrics.get('val_language_loss'), 'metric_definition': definition})
        get_run_store().append(record['run_id'], {'type': 'log', 'level': 'info',
            'message': 'Imported checkpoint and its saved final evaluation. Earlier live metric history is unavailable.'})
        return {'run_id': record['run_id']}
    except (KeyError, ValueError, RuntimeError, TypeError, OSError) as error:
        raise HTTPException(422, f'Could not import this pretraining checkpoint: {error}')


@router.post('/runs/{run_id}/resume')
async def resume_run(run_id: str, request: ResumeRunRequest):
    from dataclasses import asdict
    import torch
    run = require_run(run_id)
    if run['training_type'] != 'pretraining':
        raise HTTPException(422, 'Registered checkpoint resume currently supports pretraining runs.')
    if request.additional_steps < 0:
        raise HTTPException(422, 'additional_steps must be non-negative')
    files = run_checkpoints(run)
    if not files:
        raise HTTPException(422, 'No resumable checkpoint has been saved yet.')
    try:
        saved = torch.load(files[0], map_location='cpu', weights_only=False)
        model, config = asdict(saved['model_config']), asdict(saved['train_config'])
        if request.additional_steps == 0 and saved['step'] >= config['max_steps']:
            raise HTTPException(409, 'This checkpoint reached its budget. Set additional optimizer steps to extend it.')
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
        raise HTTPException(422, f'Could not read the checkpoint configuration: {error}')
    return await start_training(TrainingRequest(model_cfg=model, training_cfg=config,
        checkpoint_path=str(files[0]), additional_steps=request.additional_steps,
        output_dir='/app/data', project_id=run.get('project_id')))


@router.get('/runs/{run_id}')
async def get_run(run_id: str):
    return require_run(run_id)


@router.get('/runs/{run_id}/events')
async def get_events(run_id: str, after: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=5000)):
    require_run(run_id)
    return {'events': get_run_store().events(run_id, after, limit)}


@router.get('/metrics/stream')
async def stream_metrics(run_id: Optional[str] = None, after: int = Query(0, ge=0),
                         last_event_id: Optional[str] = Header(None)):
    restore_state()
    selected = run_id or training_state.get('run_id')
    if selected is None:
        raise HTTPException(404, 'No run has been recorded yet')
    require_run(selected)
    try:
        cursor = max(after, int(last_event_id or 0))
    except ValueError:
        raise HTTPException(400, 'Invalid Last-Event-ID')

    async def event_generator():
        nonlocal cursor
        yield {'event': 'snapshot', 'data': json.dumps({'type': 'snapshot', **require_run(selected)})}
        previous_status = None
        while True:
            events = get_run_store().events(selected, cursor)
            for event in events:
                cursor = event['event_id']
                yield {'id': str(cursor), 'event': 'message', 'data': json.dumps(event)}
            state = require_run(selected)
            signature = (state['status'], state['is_training'])
            if signature != previous_status:
                previous_status = signature
                yield {'event': 'status', 'data': json.dumps({'type': 'status', **state})}
            await asyncio.sleep(0.25)

    return EventSourceResponse(event_generator())
