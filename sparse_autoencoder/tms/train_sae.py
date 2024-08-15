# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.

from typing import Callable, Tuple

import numpy as np
import torch
from jaxtyping import Float
from tqdm import trange

from sparse_autoencoder.modules.loss import mse_loss
from sparse_autoencoder.modules.sparse_matmul import coo_sparse_dense_matmul
from sparse_autoencoder.tms.sae import TmsFastAutoencoder, TmsAutoencoder
from sparse_autoencoder.tms.toy_model import FastToyModel
from sparse_autoencoder.tms.train_toy_model import generate_feature_batch
from sparse_autoencoder.utils.logger import Logger


def _get_grad_norm(sae: TmsFastAutoencoder | TmsAutoencoder) -> torch.Tensor:
    total_sq_norm = torch.zeros((), device="cuda", dtype=torch.float32)
    for param in sae.parameters():
        if param.grad is not None:
            total_sq_norm += (param.grad.float() ** 2).sum()
    return total_sq_norm.sqrt()


def auxk_loss(
    auxk_recons: torch.Tensor, model_recon_err: Float[torch.Tensor, "*b d"]
) -> torch.Tensor:
    recon_err_mu = model_recon_err.mean(dim=0)
    loss = mse_loss(auxk_recons, model_recon_err) / mse_loss(
        recon_err_mu[None, :].broadcast_to(model_recon_err.shape), model_recon_err
    )
    return loss


def train_tms_sae(
    toy_model_filepath: str,
    d_model: int,
    n_features: int,
    k: int,
    dead_steps_threshold: int,
    auxk: int | None,
    auxk_coeff: float = 1 / 32,
    batch_size: int = 4096,
    steps: int = 20_000,
    feature_proba: float = 0.01,
    lr: float = 1e-3,
    lr_scale: Callable[[Tuple[int, int]], float] | None = None,
    clip_grad: float | None = None,
    model_save_path: str | None = None,
    wandb_entity: str | None = None,
    run_id: str = "tms_sae",
    log_freq: int = 100,
) -> TmsFastAutoencoder | TmsAutoencoder:
    if lr_scale is None:
        lr_scale = lambda step_steps: np.cos(
            0.5 * np.pi * step_steps[0] / (step_steps[1] - 1)
        )

    if auxk is None:
        auxk = d_model / 2
    config = dict(
        n_features=n_features,
        d_model=d_model,
        k=k,
        dead_steps_threshold=dead_steps_threshold,
        auxk=auxk,
        auxk_coeff=auxk_coeff,
        toy_model_filepath=toy_model_filepath,
        batch_size=batch_size,
        steps=steps,
        feature_proba=feature_proba,
        lr=lr,
        lr_scale=lr_scale.__name__ if lr_scale is not None else "cosine",
        clip_grad=clip_grad,
        model_save_path=model_save_path,
        run_id=run_id,
    )
    logger = Logger(config=config, wandb_entity=wandb_entity, run_id=run_id, RANK=0)

    sae = TmsFastAutoencoder(
        n_features=n_features,
        d_model=d_model,
        k=k,
        dead_steps_threshold=dead_steps_threshold,
        auxk=auxk,
    )
    sae.cuda()
    toy_model = FastToyModel(d_model=d_model, n_features=n_features)
    toy_model.from_pretrained(toy_model_filepath)
    toy_model.eval()
    tms_W_tr = toy_model.W_tr.cuda()

    scaler = torch.amp.GradScaler()
    optim = torch.optim.AdamW(sae.parameters(), lr=lr, eps=6.25e-10, fused=True)

    pbar = trange(steps, desc="sae training")
    for step in pbar:
        step_lr = lr * lr_scale((step, steps))
        for group in optim.param_groups:
            group["lr"] = step_lr

        optim.zero_grad(set_to_none=True)
        with torch.inference_mode():
            feature_batch: Float[torch.Tensor, "*b f"] = generate_feature_batch(
                batch_size=batch_size,
                n_features=n_features,
                feature_probs=feature_proba,
                device=torch.device("cuda"),
            )
            activations = coo_sparse_dense_matmul(feature_batch, tms_W_tr)

        with torch.amp.autocast(device_type="cuda"):
            recons, auxk_info = sae(activations)
            _mse_loss = mse_loss(recons, activations)
            _auxk_loss = auxk_coeff * auxk_loss(
                sae.decode(auxk_info["auxk_idxs"], auxk_info["auxk_vals"])
                - sae.pre_bias.detach(),
                activations - recons.detach(),
            ).nan_to_num(0)
            loss = _mse_loss + _auxk_loss
            logger.lazy_log_kv("train_recons", _mse_loss)
            logger.lazy_log_kv("train_auxk_recons", _auxk_loss)

        _unscaled_loss = loss.item()
        logger.lazy_log_kv("loss_scale", scaler.get_scale())
        logger.eager_log_kv("train_loss", _unscaled_loss)

        loss = scaler.scale(loss)
        loss.backward()
        loss = loss.item()  # drop buffers

        sae.unit_norm_decoder_()
        sae.unit_norm_decoder_grad_adjustment_()

        scaler.unscale_(optim)  # for gradient clipping
        grad_norm = logger.lazy_log_kv("grad_norm", _get_grad_norm(sae))
        if clip_grad is not None:
            grads = [param.grad for param in sae.parameters() if param.grad is not None]
            torch._foreach_mul_(
                grads, clip_grad / torch.clamp(grad_norm, min=clip_grad)
            )

        scaler.step(optim)
        scaler.update()

        logger.dump_lazy_logged_kvs()
        if (step + 1) % log_freq == 0 or (step + 1 == steps):
            pbar.set_postfix(loss=_unscaled_loss, lr=step_lr)

    if model_save_path is not None:
        sae.save_pretrained(model_save_path)
    return sae


if __name__ == "__main__":
    _ = train_tms_sae(
        toy_model_filepath="artefacts/toy_model-n_feat=1024-d_model=32-spars=0.99.pt",
        d_model=32,
        n_features=1024,
        k=4,
        dead_steps_threshold=256,
        auxk=16,
        auxk_coeff=1 / 32,
        batch_size=4096,
        steps=20_000,
        feature_proba=0.01,
        lr=1e-3,
        lr_scale=None,
        clip_grad=None,
        model_save_path="artefacts/sae-n_feat=1024-d_model=32-spars=0.99.pt",
        wandb_entity="xyznlp",
        run_id="tms_sae",
        log_freq=100,
    )
