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
"""

Fl = lambda size: jt.Float[torch.Tensor, size]
Fl32 = lambda size: jt.Float32[torch.Tensor, size]

# General (e.g., coo sparse tensors, etc.).
CooIndices = jt.Float[torch.Tensor, "2 nnz"]
CooValues = jt.Float[torch.Tensor, "nnz"]

# Toy models of superposition.
TmsFeatures = jt.Float[torch.Tensor, "*b f"]
TmsWeights = jt.Float[torch.Tensor, "d f"]
TmsWeightsTr = jt.Float[torch.Tensor, "f d"]
TmsBias = jt.Float[torch.Tensor, "f"]
TmsActivations = jt.Float[torch.Tensor, "*b d"]

TmsSaePreBias = jt.Float[torch.Tensor, "d"]
TmsSaeLatentBias = jt.Float[torch.Tensor, "f"]
TmsSaeEncoderWeights = jt.Float[torch.Tensor, "d f"]
TmsSaeDecoderWeights = jt.Float[torch.Tensor, "f d"]

# RecurrentGemma (RG-LRU expansion = model dimension).
Tokens = jt.Integer[torch.Tensor, "*b l"]
RGemmaActivations = jt.Float[torch.Tensor, "*b l d"]

# Sparse autoencoder.
Features = jt.Float[torch.Tensor, "*b l f"]

SaePreBias = jt.Float[torch.Tensor, "d"]
SaeLatentBias = jt.Float[torch.Tensor, "f"]
SaeEncoderWeights = jt.Float[torch.Tensor, "d f"]
SaeDecoderWeights = jt.Float[torch.Tensor, "f d"]
