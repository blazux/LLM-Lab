"""
HuggingFace Hub Export Module

Provides functionality to export models locally or push to HuggingFace Hub.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

from .converter import (
    convert_checkpoint,
    detect_export_format,
    ExportFormat,
    FormatInfo,
)
from .config_mapper import generate_hf_config, save_hf_config
from .model_card import generate_model_card, save_model_card


# Path to bundled custom model files (relative to this module)
CUSTOM_MODEL_FILES_DIR = Path(__file__).parent / "custom_model_files"


def _copy_custom_model_files(output_dir: str, format_info: FormatInfo):
    """
    Copy custom model implementation files for trust_remote_code support.

    Only needed for custom export formats that aren't natively supported
    by transformers library.
    """
    if format_info.format in [ExportFormat.LLAMA, ExportFormat.MIXTRAL]:
        # Standard formats don't need custom files
        return

    src_dir = CUSTOM_MODEL_FILES_DIR
    if not src_dir.exists():
        print(f"Warning: Custom model files directory not found at {src_dir}")
        print("Custom format export will require manual model file creation")
        return

    files_to_copy = [
        "configuration_llmlab.py",
        "modeling_llmlab.py",
    ]

    for filename in files_to_copy:
        src_path = src_dir / filename
        if src_path.exists():
            dst_path = Path(output_dir) / filename
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename} to output directory")
        else:
            print(f"Warning: {filename} not found in custom model files")


def _save_tokenizer(output_dir: str, tokenizer_name: str):
    """
    Download and save tokenizer files to output directory.

    Args:
        output_dir: Directory to save tokenizer files
        tokenizer_name: HuggingFace tokenizer name/path
    """
    try:
        from transformers import AutoTokenizer

        print(f"Downloading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved tokenizer to {output_dir}")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
        print("You may need to manually add tokenizer files")


def prepare_export_directory(
    checkpoint_path: str,
    output_dir: str,
    model_config=None,
    training_config=None,
    target_format: Optional[ExportFormat] = None,
    model_name: Optional[str] = None,
    model_description: Optional[str] = None,
    license_type: str = "apache-2.0",
    author: Optional[str] = None,
    tags: Optional[List[str]] = None,
    datasets_used: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    use_safetensors: bool = True,
    include_tokenizer: bool = True,
) -> FormatInfo:
    """
    Prepare a complete HuggingFace-compatible export directory.

    This function:
    1. Converts checkpoint weights to target format
    2. Generates config.json
    3. Generates README.md (model card)
    4. Copies custom model files if needed
    5. Saves tokenizer files

    Args:
        checkpoint_path: Path to LLM-Lab .pt checkpoint
        output_dir: Directory to save exported files
        model_config: Model config (loaded from checkpoint if not provided)
        training_config: Training config for metadata
        target_format: Target format (auto-detected if not provided)
        model_name: Model name for model card
        model_description: Short description for model card
        license_type: License identifier
        author: Model author
        tags: Additional tags for model card
        datasets_used: Datasets used for training
        metrics: Evaluation metrics
        use_safetensors: Use safetensors format (recommended)
        include_tokenizer: Include tokenizer files

    Returns:
        FormatInfo with export details
    """
    import torch

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint to get config if not provided
    if model_config is None or training_config is None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if model_config is None and 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        if training_config is None:
            training_config = checkpoint.get('train_config') or checkpoint.get('sft_config')

        # Also extract metrics from checkpoint if available
        if metrics is None:
            metrics = checkpoint.get('eval_metrics') or checkpoint.get('final_metrics')

    if model_config is None:
        raise ValueError("model_config not provided and not found in checkpoint")

    # Detect export format
    if target_format is None:
        format_info = detect_export_format(model_config)
    else:
        format_info = detect_export_format(model_config)
        format_info.format = target_format

    print(f"\n{'='*50}")
    print(f"Export Format: {format_info.format.value}")
    print(f"Reason: {format_info.reason}")
    print(f"vLLM Compatible: {format_info.vllm_compatible}")
    print(f"{'='*50}\n")

    # 1. Convert and save weights
    weights_filename = "model.safetensors" if use_safetensors else "pytorch_model.bin"
    weights_path = os.path.join(output_dir, weights_filename)

    converted_weights, _ = convert_checkpoint(
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        target_format=format_info.format,
        output_path=weights_path,
        use_safetensors=use_safetensors,
    )

    # 2. Generate and save config.json
    hf_config = generate_hf_config(
        model_config=model_config,
        format_info=format_info,
        training_config=training_config,
    )
    config_path = os.path.join(output_dir, "config.json")
    save_hf_config(hf_config, config_path)

    # 3. Generate and save README.md
    # Extract datasets from training config if not provided
    if datasets_used is None and training_config is not None:
        if hasattr(training_config, 'datasets'):
            ds_list = training_config.datasets
        elif isinstance(training_config, dict):
            ds_list = training_config.get('datasets', [])
        else:
            ds_list = []

        if ds_list:
            datasets_used = []
            for ds in ds_list:
                if isinstance(ds, dict):
                    datasets_used.append(ds.get('name', str(ds)))
                else:
                    datasets_used.append(str(ds))

    model_card = generate_model_card(
        model_config=model_config,
        format_info=format_info,
        training_config=training_config,
        model_name=model_name,
        model_description=model_description,
        license_type=license_type,
        author=author,
        tags=tags,
        datasets_used=datasets_used,
        metrics=metrics,
    )
    readme_path = os.path.join(output_dir, "README.md")
    save_model_card(model_card, readme_path)

    # 4. Copy custom model files if needed
    _copy_custom_model_files(output_dir, format_info)

    # 5. Save tokenizer
    if include_tokenizer:
        if hasattr(model_config, 'tokenizer_name'):
            tokenizer_name = model_config.tokenizer_name
        elif isinstance(model_config, dict):
            tokenizer_name = model_config.get('tokenizer_name', 'Qwen/Qwen2.5-0.5B')
        else:
            tokenizer_name = 'Qwen/Qwen2.5-0.5B'

        _save_tokenizer(output_dir, tokenizer_name)

    print(f"\n{'='*50}")
    print(f"Export complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")

    # List exported files
    print("\nExported files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  - {f} ({size_str})")

    return format_info


def export_to_local(
    checkpoint_path: str,
    output_dir: str,
    **kwargs
) -> str:
    """
    Export model to local directory.

    Convenience wrapper around prepare_export_directory.

    Args:
        checkpoint_path: Path to checkpoint
        output_dir: Output directory
        **kwargs: Additional arguments for prepare_export_directory

    Returns:
        Path to output directory
    """
    prepare_export_directory(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        **kwargs
    )
    return output_dir


def export_to_hub(
    checkpoint_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    local_dir: Optional[str] = None,
    commit_message: str = "Upload model",
    **kwargs
) -> str:
    """
    Export model and push to HuggingFace Hub.

    Args:
        checkpoint_path: Path to checkpoint
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        token: HuggingFace API token (uses cached token if not provided)
        private: Create private repository
        local_dir: Local directory for export (temp dir if not provided)
        commit_message: Commit message for upload
        **kwargs: Additional arguments for prepare_export_directory

    Returns:
        URL of the uploaded model
    """
    import tempfile

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for pushing to Hub. "
            "Install with: pip install huggingface_hub"
        )

    # Use provided local_dir or create temp directory
    if local_dir:
        output_dir = local_dir
        cleanup = False
    else:
        output_dir = tempfile.mkdtemp(prefix="llmlab_export_")
        cleanup = True

    try:
        # Set model_name from repo_id if not provided
        if 'model_name' not in kwargs:
            kwargs['model_name'] = repo_id

        # Prepare export directory
        format_info = prepare_export_directory(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            **kwargs
        )

        # Create repo if it doesn't exist
        print(f"\nCreating repository: {repo_id}")
        api = HfApi(token=token)

        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type="model",
                exist_ok=True,
            )
        except Exception as e:
            print(f"Note: {e}")

        # Upload folder
        print(f"Uploading to {repo_id}...")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
        )

        url = f"https://huggingface.co/{repo_id}"
        print(f"\nModel uploaded successfully!")
        print(f"URL: {url}")

        if format_info.vllm_compatible:
            print(f"\nvLLM command:")
            print(f"  vllm serve {repo_id}")
        else:
            print(f"\nNote: This model requires trust_remote_code=True")

        return url

    finally:
        # Cleanup temp directory
        if cleanup and os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def validate_export(export_dir: str) -> Dict[str, Any]:
    """
    Validate an export directory has all required files.

    Args:
        export_dir: Path to export directory

    Returns:
        Dict with validation results
    """
    required_files = ["config.json", "README.md"]
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]

    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "files": [],
    }

    # Check required files
    for f in required_files:
        path = os.path.join(export_dir, f)
        if os.path.exists(path):
            results["files"].append(f)
        else:
            results["errors"].append(f"Missing required file: {f}")
            results["valid"] = False

    # Check weight files (at least one required)
    has_weights = False
    for f in weight_files:
        path = os.path.join(export_dir, f)
        if os.path.exists(path):
            results["files"].append(f)
            has_weights = True

    if not has_weights:
        results["errors"].append("Missing weight file (model.safetensors or pytorch_model.bin)")
        results["valid"] = False

    # Check tokenizer files (warning if missing)
    has_tokenizer = False
    for f in tokenizer_files:
        path = os.path.join(export_dir, f)
        if os.path.exists(path):
            results["files"].append(f)
            has_tokenizer = True

    if not has_tokenizer:
        results["warnings"].append("Missing tokenizer files - model may not load correctly")

    # Check config.json content
    config_path = os.path.join(export_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

        if "model_type" not in config:
            results["warnings"].append("config.json missing 'model_type' field")

        if "architectures" not in config:
            results["warnings"].append("config.json missing 'architectures' field")

        # Check if custom files needed
        if config.get("auto_map"):
            custom_files = ["configuration_llmlab.py", "modeling_llmlab.py"]
            for f in custom_files:
                path = os.path.join(export_dir, f)
                if not os.path.exists(path):
                    results["errors"].append(
                        f"Custom format requires {f} but file not found"
                    )
                    results["valid"] = False
                else:
                    results["files"].append(f)

    return results
