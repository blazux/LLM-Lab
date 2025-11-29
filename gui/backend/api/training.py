import asyncio
import sys
import threading
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
from training.train import train_model, request_stop

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
    "error": None
}

# Queue for metrics updates
metrics_queue = queue.Queue()

# Training thread
training_thread = None


class TrainingRequest(BaseModel):
    model_cfg: Dict[str, Any]
    training_cfg: Dict[str, Any]
    checkpoint_path: Optional[str] = None
    output_dir: str = "checkpoints"


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
                 checkpoint_path: Optional[str] = None, output_dir: str = "checkpoints"):
    """Run training in a separate thread"""
    try:
        # Update training state
        training_state["is_training"] = True
        training_state["status"] = "starting"
        training_state["error"] = None

        # Log received config for verification
        metrics_queue.put({
            "type": "log",
            "level": "info",
            "message": f"Config received - Tokenizer: {model_config_dict.get('tokenizer_name', 'NOT SET')}",
            "timestamp": time.time()
        })
        metrics_queue.put({
            "type": "log",
            "level": "info",
            "message": f"Config received - Model: d_model={model_config_dict.get('d_model')}, n_layers={model_config_dict.get('n_layers')}, attention={model_config_dict.get('attention_type')}",
            "timestamp": time.time()
        })

        # Create config objects
        model_config = ModelConfig(**model_config_dict)
        train_config = TrainingConfig(**train_config_dict)

        # Update max_steps in state
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


@router.post("/start")
async def start_training(request: TrainingRequest):
    """Start training in the background"""
    global training_thread

    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")

    # Reset state
    training_state.update({
        "is_training": True,
        "current_step": 0,
        "max_steps": request.training_cfg.get("max_steps", 10000),
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
            request.output_dir
        ),
        daemon=True
    )
    training_thread.start()

    return {"success": True, "message": "Training started"}


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
