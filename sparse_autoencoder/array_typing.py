from typing import Callable, TypeVar

import jaxtyping as jt
import torch

F = TypeVar("F", bound=Callable)


def typed(function: F) -> F:
    return function


"""
Notation:
- nnz = number of non-zero elements
- b = batch size
- l = sequence length
- d = model dimension
- f = number of feature directions
- e = expanded autoencoder dimension
"""

# General (e.g., coo sparse tensors, etc.).
CooIndices = jt.Float[torch.Tensor, "2 nnz"]
CooValues = jt.Float[torch.Tensor, "nnz"]

# Toy models of superposition.
TmsFeatures = jt.Float[torch.Tensor, "*b f"]
TmsWeights = jt.Float[torch.Tensor, "d f"]
TmsWeightsTr = jt.Float[torch.Tensor, "f d"]
TmsBias = jt.Float[torch.Tensor, "f"]
TmsActivations = jt.Float[torch.Tensor, "*b d"]

# RecurrentGemma (RG-LRU expansion = model dimension).
Tokens = jt.Integer[torch.Tensor, "*b l"]
Activations = jt.Float[torch.Tensor, "*b l d"]

# Sparse autoencoder.
JumpReluThreshold = jt.Float[torch.Tensor, "e"]
SaeActivations = jt.Float[torch.Tensor, "*b l e"]
