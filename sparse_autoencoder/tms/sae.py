# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.
from typing import Any

import torch
from torch import autograd
from torch import nn
from torch.amp import custom_fwd, custom_bwd

import sparse_autoencoder.array_typing as at
from sparse_autoencoder.modules.activations import topk
from sparse_autoencoder.modules.utils import contiguous


class FastAutoencoderWrapper(autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: at.TmsActivations,
        pre_bias: at.TmsSaePreBias,
        W_enc: at.TmsSaeEncoderWeights,
        latent_bias: at.TmsSaeLatentBias,
        k: int,
    ) -> Any:
        latents_pre_act = (x - pre_bias) @ W_enc
        latents = topk(latents_pre_act, k, latent_bias)

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any) -> Any:
        pass


class TmsFastAutoencoder(nn.Module):
    """
    TopK sparse autoencoder: https://arxiv.org/pdf/2406.04093.
    - latents = topk(((x: [B D] - pre_bias: [D]) @ W_enc: [D F]) + latent_bias: [F])
    - recons: [B D] = (latents: [B F] @ W_dec: [F D]) + pre_bias: [D]

    Note: `topk` is implemented as `relu(topk(x + bias, k))`.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        k: int,
        dead_steps_threshold: int,
        auxk: int | None = None,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.k = k
        self.dead_steps_threshold = dead_steps_threshold
        self.auxk = auxk

        self.pre_bias: at.TmsSaePreBias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias: at.TmsSaeLatentBias = nn.Parameter(torch.zeros(d_model))
        self.W_enc: at.TmsSaeEncoderWeights = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(d_model, n_features))
        )
        self.W_dec: at.TmsSaeDecoderWeights = nn.Parameter(self.W_enc.data.clone().T)
        self._unit_norm_decoder()

        self.register_buffer(
            "stats_last_nonzero", torch.zeros(n_features, dtype=torch.long)
        )

    def _auxk_mask_(self, x: torch.Tensor):
        dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
        x *= dead_mask  # inplace to save memory
        return x

    def _unit_norm_decoder(self) -> None:
        """Normalize latent directions of decoder to unit norm."""
        self.W_dec.data /= self.W_dec.data.norm(dim=1)

    def forward(self, x: at.TmsActivations):
        return FastAutoencoderWrapper.apply(x)
