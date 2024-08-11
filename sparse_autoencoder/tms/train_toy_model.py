from typing import Callable, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import trange

import sparse_autoencoder.array_typing as at
from sparse_autoencoder.modules.loss import mse_loss
from sparse_autoencoder.tms.toy_model import FastToyModel, ToyModel


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


def torch_mse_loss(
    feature_batch: at.TmsFeatures, recons: at.TmsFeatures
) -> torch.Tensor:
    return ((recons - feature_batch) ** 2).mean()


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
    use_fast_model: bool = False,
) -> FastToyModel | ToyModel:
    if lr_scale is None:
        # Default: cosine lr scale.
        lr_scale = lambda step_steps: np.cos(
            0.5 * np.pi * step_steps[0] / (step_steps[1] - 1)
        )

    if use_fast_model:
        model = FastToyModel(n_features=n_features, d_model=d_model)
    else:
        model = ToyModel(n_features=n_features, d_model=d_model)
    model.cuda()
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
        loss = mse_loss(recons, feature_batch)
        loss.backward()
        loss = loss.item()  # drop buffers
        optim.step()

        if (step + 1) % log_freq == 0 or (step + 1 == steps):
            pbar.set_postfix(loss=loss, lr=step_lr)

    if model_save_path is not None:
        model.save_pretrained(model_save_path)
    return model
