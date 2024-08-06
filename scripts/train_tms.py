from typing import Callable, Tuple

import numpy as np
import torch
from einops import einsum
from torch import nn
from torch.optim import AdamW
from tqdm import trange

import sparse_autoencoder.array_typing as at
from sparse_autoencoder.modules.activations import relu


class TmsModel(nn.Module):
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


def generate_feature_batch(
    batch_size: int,
    n_features: int,
    feature_probs: torch.Tensor | float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
) -> at.TmsFeatures:
    if feature_probs is None:
        feature_probs = torch.ones(batch_size, n_features, device=device)
    elif isinstance(feature_probs, float):
        feature_probs = torch.tensor(feature_probs, device=device).broadcast_to(
            batch_size, n_features
        )

    feature_vals = torch.rand(batch_size, n_features, dtype=dtype, device=device)
    feature_seeds = torch.rand(batch_size, n_features, dtype=dtype, device=device)
    batch = torch.where(feature_seeds <= feature_probs, feature_vals, 0.0)
    return batch


def mse_loss(feature_batch: at.TmsFeatures, recons: at.TmsFeatures) -> torch.Tensor:
    return ((feature_batch - recons) ** 2).mean()


def train_tms(
    d_model: int,
    n_features: int,
    feature_probs: torch.Tensor | float | None = None,
    batch_size: int = 1024,
    steps: int = 10_000,
    lr: float = 1e-3,
    lr_scale: Callable[[Tuple[int, int]], float] | None = None,
    device: torch.device = torch.device("cuda"),
    log_freq: int = 100,
    model_save_path: str | None = None,
):
    if lr_scale is None:
        # Default: cosine lr scale.
        lr_scale = lambda step_steps: np.cos(
            0.5 * np.pi * step_steps[0] / (step_steps[1] - 1)
        )

    model = TmsModel(n_features=n_features, d_model=d_model, device=device)
    optim = AdamW(model.parameters(), lr=lr)

    pbar = trange(steps, desc="tms training")
    for step in pbar:
        step_lr = lr * lr_scale((step, steps))
        for group in optim.param_groups:
            group["lr"] = step_lr

        optim.zero_grad(set_to_none=True)
        feature_batch = generate_feature_batch(
            batch_size=batch_size,
            n_features=n_features,
            feature_probs=feature_probs,
            device=device,
        )
        recons = model(feature_batch)
        loss = mse_loss(feature_batch=feature_batch, recons=recons)
        loss.backward()
        loss = loss.item()  # drop buffers
        optim.step()

        if (step + 1) % log_freq == 0 or (step + 1 == steps):
            pbar.set_postfix(loss=loss, lr=step_lr)

    if model_save_path is not None:
        model.save_pretrained(model_save_path)


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class ToyTmsModelConfig:
        d_model = 2
        n_features = 6
        feature_proba = 0.1

    config = ToyTmsModelConfig()
    # 10,000 steps: loss=0.0074, lr=6.12e-20.
    train_tms(
        d_model=config.d_model,
        n_features=config.n_features,
        feature_probs=config.feature_proba,
        batch_size=1024,
        steps=10_000,
        lr=1e-3,
        lr_scale=None,
        device=torch.device("cuda"),
        model_save_path=f"artefacts/toy_model-spars={1 - config.feature_proba}.pt",
    )
