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
- d = model dimension
- f = number of feature directions
- e = expanded autoencoder dimension
"""

# Toy models of superposition.
TmsFeatures = jt.Float[torch.Tensor, "*b f"]
TmsWeights = jt.Float[torch.Tensor, "d f"]
TmsActivations = jt.Float[torch.Tensor, "*b d"]

# RecurrentGemma (RG-LRU expansion = model dimension).
Tokens = jt.Integer[torch.Tensor, "*b l"]
Activations = jt.Float[torch.Tensor, "*b l d"]

# Sparse autoencoder.
JumpReluThreshold = jt.Float[torch.Tensor, "e"]
SaeActivations = jt.Float[torch.Tensor, "*b l e"]
