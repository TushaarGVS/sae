from typing import Callable, TypeVar

import jaxtyping as jt
import torch

F = TypeVar("F", bound=Callable)


def typed(function: F) -> F:
    return function


"""
Notation:
- b = batch size
- l = sequence length
- d = model dimension (same as RG-LRU dimension)
- e = expanded autoencoder dimension
"""

Tokens = jt.Integer[torch.Tensor, "*b l"]
Activations = jt.Float[torch.Tensor, "*b l d"]
JumpReluThreshold = jt.Float[torch.Tensor, "e"]
SaeActivations = jt.Float[torch.Tensor, "*b l e"]
