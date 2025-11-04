"""Model factory for building different architectures"""

from config import ModelConfig


def build_model(config: ModelConfig):
    """
    Factory function to build a model based on config.model_architecture

    Args:
        config: ModelConfig with model_architecture field

    Returns:
        Model instance (TransformerLLM or Mamba2LLM)

    Raises:
        ValueError: If model_architecture is not recognized
    """
    if config.model_architecture == "transformer":
        from model import TransformerLLM
        return TransformerLLM(config)

    elif config.model_architecture == "mamba2":
        from model.mamba2_model import Mamba2LLM
        return Mamba2LLM(config)

    else:
        raise ValueError(
            f"Unknown model_architecture: {config.model_architecture}. "
            f"Supported architectures: 'transformer', 'mamba2'"
        )


def get_supported_architectures():
    """Return list of supported model architectures"""
    return ["transformer", "mamba2"]
