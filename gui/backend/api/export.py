"""
Export API endpoints for HuggingFace Hub integration
"""

import sys
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch

# Import from existing LLM-Lab source
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / 'src'))

router = APIRouter()


class ExportRequest(BaseModel):
    checkpoint_path: str
    output_dir: Optional[str] = None
    repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    private: bool = False
    model_name: Optional[str] = None
    model_description: Optional[str] = None
    license_type: str = "apache-2.0"
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    use_safetensors: bool = True
    include_tokenizer: bool = True


class ExportResponse(BaseModel):
    success: bool
    message: str
    format: Optional[str] = None
    vllm_compatible: Optional[bool] = None
    output_path: Optional[str] = None
    hub_url: Optional[str] = None
    files: Optional[List[str]] = None
    details: Optional[Dict[str, Any]] = None


class FormatDetectionResponse(BaseModel):
    format: str
    reason: str
    vllm_compatible: bool
    model_type: str
    architectures: List[str]
    checkpoint_info: Dict[str, Any]


@router.get("/detect-format")
async def detect_export_format(checkpoint_path: str) -> FormatDetectionResponse:
    """
    Detect the best export format for a checkpoint.

    Returns format info including whether the model is vLLM compatible.
    """
    try:
        from export import detect_export_format
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Export module not available: {e}")

    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_config = checkpoint.get('model_config')

        if model_config is None:
            raise HTTPException(status_code=400, detail="No model_config found in checkpoint")

        format_info = detect_export_format(model_config)

        # Extract checkpoint info
        checkpoint_info = {
            "has_model_config": 'model_config' in checkpoint,
            "has_train_config": 'train_config' in checkpoint,
            "has_sft_config": 'sft_config' in checkpoint,
            "has_rlhf_config": 'rlhf_config' in checkpoint,
            "has_eval_metrics": 'eval_metrics' in checkpoint or 'final_metrics' in checkpoint,
            "step": checkpoint.get('step'),
        }

        # Get model architecture info
        if hasattr(model_config, '__dict__'):
            checkpoint_info["architecture"] = getattr(model_config, 'model_architecture', 'transformer')
            checkpoint_info["attention_type"] = getattr(model_config, 'attention_type', 'gqa')
            checkpoint_info["use_moe"] = getattr(model_config, 'use_moe', False)
            checkpoint_info["d_model"] = getattr(model_config, 'd_model', 0)
            checkpoint_info["n_layers"] = getattr(model_config, 'n_layers', 0)
        else:
            checkpoint_info["architecture"] = model_config.get('model_architecture', 'transformer')
            checkpoint_info["attention_type"] = model_config.get('attention_type', 'gqa')
            checkpoint_info["use_moe"] = model_config.get('use_moe', False)
            checkpoint_info["d_model"] = model_config.get('d_model', 0)
            checkpoint_info["n_layers"] = model_config.get('n_layers', 0)

        return FormatDetectionResponse(
            format=format_info.format.value,
            reason=format_info.reason,
            vllm_compatible=format_info.vllm_compatible,
            model_type=format_info.model_type,
            architectures=format_info.architectures,
            checkpoint_info=checkpoint_info,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/local")
async def export_to_local(request: ExportRequest) -> ExportResponse:
    """
    Export model to local directory in HuggingFace format.
    """
    try:
        from export import export_to_local as do_export, detect_export_format
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Export module not available: {e}")

    if not os.path.exists(request.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.checkpoint_path}")

    output_dir = request.output_dir or "/app/data/hf_export"

    try:
        # Get format info first
        checkpoint = torch.load(request.checkpoint_path, map_location='cpu', weights_only=False)
        model_config = checkpoint.get('model_config')
        format_info = detect_export_format(model_config)

        # Do the export
        do_export(
            checkpoint_path=request.checkpoint_path,
            output_dir=output_dir,
            model_name=request.model_name,
            model_description=request.model_description,
            license_type=request.license_type,
            author=request.author,
            tags=request.tags,
            use_safetensors=request.use_safetensors,
            include_tokenizer=request.include_tokenizer,
        )

        # List exported files
        files = []
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)

        return ExportResponse(
            success=True,
            message="Model exported successfully",
            format=format_info.format.value,
            vllm_compatible=format_info.vllm_compatible,
            output_path=output_dir,
            files=files,
        )

    except Exception as e:
        return ExportResponse(
            success=False,
            message=str(e),
        )


@router.post("/hub")
async def export_to_hub(request: ExportRequest) -> ExportResponse:
    """
    Export model and push to HuggingFace Hub.
    """
    try:
        from export import export_to_hub as do_export, detect_export_format
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Export module not available: {e}")

    if not os.path.exists(request.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.checkpoint_path}")

    if not request.repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required for Hub export")

    # Get token from request or environment
    token = request.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        # Get format info first
        checkpoint = torch.load(request.checkpoint_path, map_location='cpu', weights_only=False)
        model_config = checkpoint.get('model_config')
        format_info = detect_export_format(model_config)

        # Do the export
        url = do_export(
            checkpoint_path=request.checkpoint_path,
            repo_id=request.repo_id,
            token=token,
            private=request.private,
            model_name=request.model_name or request.repo_id,
            model_description=request.model_description,
            license_type=request.license_type,
            author=request.author,
            tags=request.tags,
            use_safetensors=request.use_safetensors,
            include_tokenizer=request.include_tokenizer,
        )

        return ExportResponse(
            success=True,
            message="Model uploaded successfully",
            format=format_info.format.value,
            vllm_compatible=format_info.vllm_compatible,
            hub_url=url,
            details={
                "repo_id": request.repo_id,
                "private": request.private,
                "vllm_command": f"vllm serve {request.repo_id}" if format_info.vllm_compatible else None,
            }
        )

    except Exception as e:
        return ExportResponse(
            success=False,
            message=str(e),
        )


@router.get("/checkpoints")
async def list_checkpoints():
    """
    List available checkpoints for export.
    """
    data_dir = Path("/app/data")
    checkpoints = []

    if data_dir.exists():
        for ckpt_file in data_dir.glob("*.pt"):
            try:
                stat = ckpt_file.stat()
                checkpoint = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)

                # Determine checkpoint type
                model_config = checkpoint.get('model_config')

                info = {
                    "name": ckpt_file.name,
                    "path": str(ckpt_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "has_model_config": model_config is not None,
                }

                if model_config:
                    if hasattr(model_config, '__dict__'):
                        info["architecture"] = getattr(model_config, 'model_architecture', 'transformer')
                        info["attention_type"] = getattr(model_config, 'attention_type', 'unknown')
                    else:
                        info["architecture"] = model_config.get('model_architecture', 'transformer')
                        info["attention_type"] = model_config.get('attention_type', 'unknown')

                checkpoints.append(info)

            except Exception as e:
                checkpoints.append({
                    "name": ckpt_file.name,
                    "path": str(ckpt_file),
                    "size_mb": round(ckpt_file.stat().st_size / (1024 * 1024), 2),
                    "modified": ckpt_file.stat().st_mtime,
                    "error": str(e),
                })

    return {"checkpoints": checkpoints}


@router.post("/validate")
async def validate_export(output_dir: str):
    """
    Validate an export directory has all required files.
    """
    try:
        from export.hub import validate_export as do_validate
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Export module not available: {e}")

    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail=f"Directory not found: {output_dir}")

    results = do_validate(output_dir)
    return results
