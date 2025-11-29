import sys
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
import os

# Import from existing LLM-Lab source
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / 'src'))

from inference.inference import load_model_for_inference, generate_text

router = APIRouter()

# Global model cache (avoid reloading for consecutive requests)
_model_cache = {
    "checkpoint_path": None,
    "model": None,
    "tokenizer": None,
    "device": None
}


class GenerateRequest(BaseModel):
    checkpoint_path: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    strategy: str = "top_p"


class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    tokens_per_second: float
    prompt_tokens: int


@router.get("/checkpoints")
async def list_checkpoints():
    """List available checkpoints"""
    checkpoints_dir = Path(project_root) / "checkpoints"

    if not checkpoints_dir.exists():
        return {"checkpoints": []}

    checkpoints = []
    for ckpt_file in checkpoints_dir.glob("*.pt"):
        # Get file stats
        stat = ckpt_file.stat()

        # Try to load checkpoint metadata
        try:
            checkpoint = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
            step = checkpoint.get('step', 'N/A')
            is_rlhf = 'rlhf_config' in checkpoint
            is_sft = 'sft_config' in checkpoint

            if is_rlhf:
                model_type = "RLHF"
            elif is_sft:
                model_type = "SFT"
            else:
                model_type = "Base"

            checkpoints.append({
                "name": ckpt_file.name,
                "path": str(ckpt_file),
                "step": step,
                "type": model_type,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            })
        except Exception as e:
            # If we can't load the checkpoint, still list it but with minimal info
            checkpoints.append({
                "name": ckpt_file.name,
                "path": str(ckpt_file),
                "step": "Unknown",
                "type": "Unknown",
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            })

    # Sort by modified time (newest first)
    checkpoints.sort(key=lambda x: x['modified'], reverse=True)

    return {"checkpoints": checkpoints}


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a checkpoint"""
    global _model_cache

    # Validate checkpoint exists
    if not os.path.exists(request.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.checkpoint_path}")

    try:
        # Load model (use cache if same checkpoint)
        if _model_cache["checkpoint_path"] != request.checkpoint_path:
            print(f"Loading model from {request.checkpoint_path}...")
            model, tokenizer, device = load_model_for_inference(request.checkpoint_path)
            _model_cache["checkpoint_path"] = request.checkpoint_path
            _model_cache["model"] = model
            _model_cache["tokenizer"] = tokenizer
            _model_cache["device"] = device
        else:
            print("Using cached model...")
            model = _model_cache["model"]
            tokenizer = _model_cache["tokenizer"]
            device = _model_cache["device"]

        # Count prompt tokens
        prompt_tokens = len(tokenizer.encode(request.prompt))

        # Generate text
        start_time = time.time()
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            strategy=request.strategy
        )
        end_time = time.time()

        # Calculate stats
        total_tokens = len(tokenizer.encode(generated_text))
        tokens_generated = total_tokens - prompt_tokens
        generation_time = end_time - start_time
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

        return GenerateResponse(
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2),
            prompt_tokens=prompt_tokens
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/unload")
async def unload_model():
    """Unload the currently loaded model to free memory"""
    global _model_cache

    if _model_cache["model"] is not None:
        del _model_cache["model"]
        del _model_cache["tokenizer"]
        _model_cache["checkpoint_path"] = None
        _model_cache["model"] = None
        _model_cache["tokenizer"] = None
        _model_cache["device"] = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"success": True, "message": "Model unloaded"}

    return {"success": False, "message": "No model loaded"}
