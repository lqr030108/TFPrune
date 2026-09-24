"""TFPrune: future-functional visual-token compression."""

__version__ = "0.1.0"


def tfprune(model, target_vision_tokens=128):
    """Attach fixed-budget compression to a Vicuna-based LLaVA model."""
    from .llava.llava_inject import tfprune as inject

    return inject(model, target_vision_tokens=target_vision_tokens)
