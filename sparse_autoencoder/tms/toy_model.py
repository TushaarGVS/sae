import torch
from einops import einsum
from torch import nn

import sparse_autoencoder.array_typing as at
from sparse_autoencoder.modules.activations import relu


class ToyModel(nn.Module):
    """Computes relu(W.T(Wx) + b)."""

    def __init__(
        self,
        d_model: int,
        n_features: int,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        self.device = device
        self.W: at.TmsWeights = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(d_model, n_features, device=device))
        )
        self.b = nn.Parameter(torch.zeros(n_features, device=device))

    def save_pretrained(self, model_filepath: str) -> None:
        torch.save(self.state_dict(), model_filepath)

    def from_pretrained(self, model_filepath: str):
        state_dict = torch.load(model_filepath, map_location=self.device)
        self.load_state_dict(state_dict)

    def forward(self, x: at.TmsFeatures) -> at.TmsActivations:
        activations = einsum(self.W, x, "d f, b f -> b d")
        recons = einsum(self.W.T, activations, "f d, b d -> b f")
        return relu(recons + self.b)
