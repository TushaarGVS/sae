import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn

import sparse_autoencoder.array_typing as at
from sparse_autoencoder.modules.activations import relu
from sparse_autoencoder.modules.sparse_matmul import coo_sparse_dense_matmul


class ToyModel(nn.Module):
    """Computes relu(W.T @ W @ x + bias)."""

    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.W_tr: at.TmsWeightsTr = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(n_features, d_model))
        )
        self.bias: at.TmsBias = nn.Parameter(torch.zeros(n_features))

    def save_pretrained(self, model_filepath: str) -> None:
        torch.save(self.state_dict(), model_filepath)

    def from_pretrained(self, model_filepath: str):
        state_dict = torch.load(model_filepath, map_location=torch.device("cpu"))
        self.load_state_dict(state_dict)

    def forward(self, x: at.TmsFeatures) -> at.TmsFeatures:
        activations = einsum(x, self.W_tr, "b f, f d -> b d")
        recons = einsum(activations, self.W_tr, "b d, f d -> b f")
        return relu(recons + self.bias)


@torch.compile(fullgraph=True, backend="eager")
def _toy_model_fwd_kernel(
    x: at.TmsFeatures, W_tr: at.TmsWeightsTr, bias: at.TmsBias
) -> at.TmsFeatures:
    return relu(F.linear(coo_sparse_dense_matmul(x, W_tr), W_tr, bias))


class FastToyModel(nn.Module):
    """Computes relu(W.T @ W @ x + bias)."""

    def __init__(self, d_model: int, n_features: int):
        super().__init__()

        self.d_model = d_model
        self.n_features = n_features
        self.W_tr: at.TmsWeightsTr = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(n_features, d_model))
        )
        self.bias: at.TmsBias = nn.Parameter(torch.zeros(n_features))

    def save_pretrained(self, model_filepath: str) -> None:
        torch.save(self.state_dict(), model_filepath)

    def from_pretrained(self, model_filepath: str):
        state_dict = torch.load(model_filepath, map_location=torch.device("cpu"))
        self.load_state_dict(state_dict)

    def forward(self, x: at.TmsFeatures) -> at.TmsFeatures:
        return _toy_model_fwd_kernel(x, self.W_tr, self.bias)
