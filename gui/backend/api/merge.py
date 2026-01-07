import sys
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch

# Import from existing LLM-Lab source
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / 'src'))

from model.model import TransformerLLM
from model.mamba2_model import Mamba2LLM
from model.lora_utils import apply_lora_to_model

router = APIRouter()


class MergeRequest(BaseModel):
    adapter_path: str
    base_checkpoint_path: str
    output_path: str
    input_type: str = "adapter_folder"  # "adapter_folder" or "full_checkpoint"


class MergeResponse(BaseModel):
    success: bool
    message: str
    output_path: Optional[str] = None
    details: Optional[Dict] = None


@router.get("/available")
async def list_available_for_merge():
    """List available checkpoints and adapter folders for merging"""
    data_dir = Path("/app/data")

    checkpoints = []
    adapters = []

    if data_dir.exists():
        # Find all .pt checkpoint files
        for ckpt_file in data_dir.glob("*.pt"):
            try:
                stat = ckpt_file.stat()
                checkpoint = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)

                # Determine checkpoint type
                is_rlhf = 'rlhf_config' in checkpoint
                is_sft = 'sft_config' in checkpoint
                has_lora = any('lora' in key.lower() for key in checkpoint.get('model_state_dict', {}).keys())

                if is_rlhf:
                    model_type = "RLHF"
                elif is_sft:
                    model_type = "SFT"
                else:
                    model_type = "Base"

                checkpoints.append({
                    "name": ckpt_file.name,
                    "path": str(ckpt_file),
                    "type": model_type,
                    "has_lora": has_lora,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime
                })
            except Exception as e:
                # Still list the file even if we can't fully parse it
                checkpoints.append({
                    "name": ckpt_file.name,
                    "path": str(ckpt_file),
                    "type": "Unknown",
                    "has_lora": False,
                    "size_mb": round(ckpt_file.stat().st_size / (1024 * 1024), 2),
                    "modified": ckpt_file.stat().st_mtime
                })

        # Find adapter folders (directories with adapter_config.json)
        for item in data_dir.iterdir():
            if item.is_dir():
                adapter_config = item / "adapter_config.json"
                if adapter_config.exists():
                    try:
                        import json
                        with open(adapter_config, 'r') as f:
                            config = json.load(f)

                        adapters.append({
                            "name": item.name,
                            "path": str(item),
                            "rank": config.get("r", "Unknown"),
                            "alpha": config.get("lora_alpha", "Unknown"),
                            "target_modules": config.get("target_modules", []),
                            "modified": item.stat().st_mtime
                        })
                    except Exception as e:
                        # Still list it even if we can't parse the config
                        adapters.append({
                            "name": item.name,
                            "path": str(item),
                            "rank": "Unknown",
                            "alpha": "Unknown",
                            "target_modules": [],
                            "modified": item.stat().st_mtime
                        })

    # Sort by modified time (newest first)
    checkpoints.sort(key=lambda x: x['modified'], reverse=True)
    adapters.sort(key=lambda x: x['modified'], reverse=True)

    return {
        "checkpoints": checkpoints,
        "adapters": adapters
    }


@router.post("/lora", response_model=MergeResponse)
async def merge_lora(request: MergeRequest):
    """Merge LoRA adapters into base model"""
    try:
        from peft import PeftModel

        if request.input_type == "adapter_folder":
            # Method 1: Load from adapter folder (recommended)
            if not os.path.exists(request.adapter_path):
                raise HTTPException(status_code=404, detail=f"Adapter folder not found: {request.adapter_path}")

            # Check for adapter files
            adapter_config_file = os.path.join(request.adapter_path, "adapter_config.json")
            if not os.path.exists(adapter_config_file):
                raise HTTPException(status_code=400, detail="Not a valid adapter folder: missing adapter_config.json")

            # Load base model checkpoint
            if not os.path.exists(request.base_checkpoint_path):
                raise HTTPException(status_code=404, detail=f"Base checkpoint not found: {request.base_checkpoint_path}")

            print(f"Loading base model from {request.base_checkpoint_path}...")
            checkpoint = torch.load(request.base_checkpoint_path, map_location="cpu", weights_only=False)

            if 'model_config' not in checkpoint:
                raise HTTPException(status_code=400, detail="Checkpoint does not contain model_config")

            model_config = checkpoint['model_config']

            # Create base model (support both architectures)
            print("Creating base model...")
            if model_config.model_architecture == "mamba2":
                base_model = Mamba2LLM(model_config)
            else:
                base_model = TransformerLLM(model_config)

            base_model.load_state_dict(checkpoint['model_state_dict'])

            # Load PEFT model with adapters
            print(f"Loading LoRA adapters from {request.adapter_path}...")
            peft_model = PeftModel.from_pretrained(base_model, request.adapter_path)

            # Merge adapters
            print("Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()

            # Save merged model
            print(f"Saving merged model to {request.output_path}...")

            merged_checkpoint = {
                'model_state_dict': merged_model.state_dict(),
                'model_config': model_config,
            }

            # Copy over metadata
            for key in ['step', 'best_val_loss', 'eval_metrics', 'final_metrics', 'total_tokens_seen']:
                if key in checkpoint:
                    merged_checkpoint[key] = checkpoint[key]

            torch.save(merged_checkpoint, request.output_path)

            return MergeResponse(
                success=True,
                message="Successfully merged LoRA adapters",
                output_path=request.output_path,
                details={
                    "base_checkpoint": request.base_checkpoint_path,
                    "adapter_path": request.adapter_path,
                    "method": "adapter_folder"
                }
            )

        else:  # full_checkpoint
            # Method 2: Load from full checkpoint containing LoRA weights
            checkpoint_path = request.base_checkpoint_path  # In this mode, base_checkpoint is actually the full checkpoint

            if not os.path.exists(checkpoint_path):
                raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            if 'model_config' not in checkpoint:
                raise HTTPException(status_code=400, detail="Checkpoint does not contain model_config")

            model_config = checkpoint['model_config']
            state_dict = checkpoint['model_state_dict']

            # Check if checkpoint has LoRA parameters
            has_lora = any('lora' in key.lower() for key in state_dict.keys())

            if not has_lora:
                raise HTTPException(
                    status_code=400,
                    detail="This checkpoint doesn't appear to have LoRA parameters. It may already be a merged model."
                )

            # Create base model
            print("Creating base model...")
            if model_config.model_architecture == "mamba2":
                base_model = Mamba2LLM(model_config)
            else:
                base_model = TransformerLLM(model_config)

            # Try to find LoRA config in checkpoint
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

            if lora_config_dict is None:
                raise HTTPException(
                    status_code=400,
                    detail="Could not find LoRA config in checkpoint. Cannot automatically merge adapters."
                )

            # Apply LoRA to create PEFT model
            print("Applying LoRA configuration...")
            peft_model = apply_lora_to_model(base_model, model_config, lora_config_dict)

            # Load the state dict into PEFT model
            print("Loading LoRA weights...")
            peft_model.load_state_dict(state_dict)

            # Merge adapters
            print("Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()

            # Save merged model
            print(f"Saving merged model to {request.output_path}...")

            merged_checkpoint = {
                'model_state_dict': merged_model.state_dict(),
                'model_config': model_config,
            }

            # Copy over metadata (but remove LoRA configs)
            for key in ['step', 'best_val_loss', 'eval_metrics', 'final_metrics', 'total_tokens_seen']:
                if key in checkpoint:
                    merged_checkpoint[key] = checkpoint[key]

            torch.save(merged_checkpoint, request.output_path)

            return MergeResponse(
                success=True,
                message="Successfully merged LoRA adapters from checkpoint",
                output_path=request.output_path,
                details={
                    "checkpoint": checkpoint_path,
                    "method": "full_checkpoint"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")
