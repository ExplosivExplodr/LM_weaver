"""lm-weaver: a small transformer language model, trained from scratch.

The network, its configuration and the tokenizer live in `model`. `trainer`
builds a checkpoint and `loader` chats with one; both share `model` so a model
trained by one always loads in the other.
"""
from .model import (
    AppConfig,
    CharTokenizer,
    ModelConfig,
    ScratchLanguageModel,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "AppConfig",
    "CharTokenizer",
    "ModelConfig",
    "ScratchLanguageModel",
    "load_checkpoint",
    "save_checkpoint",
]

__version__ = "1.0.0"
