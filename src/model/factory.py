"""Model factory for building the transformer architecture"""

from config import ModelConfig


def build_model(config: ModelConfig):
    """
    Factory function to build a model from a ModelConfig.

    Args:
        config: ModelConfig describing the architecture

    Returns:
        TransformerLLM instance

    Raises:
        ValueError: If model_architecture is not recognized
    """
    if getattr(config, 'assembly', None) is not None:
        from .assembly_model import AssemblyLLM
        return AssemblyLLM(config)
    if config.model_architecture == "transformer":
        from model import TransformerLLM
        return TransformerLLM(config)

    raise ValueError(
        f"Unknown model_architecture: {config.model_architecture}. "
        f"Supported architectures: 'transformer'"
    )


def get_supported_architectures():
    """Return list of supported model architectures"""
    return ["transformer"]
