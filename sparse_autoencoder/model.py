# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.
from typing import Dict, Any

import torch
from torch import nn

from sparse_autoencoder.modules.activations import ACTIVATION_CLASSES


class AutoEncoder(nn.Module):
    """
    latents = activation(encoder(x - pre_bias) + latent_bias))
    recons = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        n_latents: int,
        d_model: int,
        activation: str | None = None,
        act_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__()

        self.n_latents = n_latents
        self.d_model = d_model
        self.activation = ACTIVATION_CLASSES.get(activation, ACTIVATION_CLASSES["topk"])

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))

        self.encoder = nn.Linear(d_model, n_latents, bias=False)
        self.decoder = nn.Linear(n_latents, d_model, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        unit_norm_decoder_(self)

    def forward(self):
        pass


def unit_norm_decoder_(autoencoder: AutoEncoder) -> None:
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)
