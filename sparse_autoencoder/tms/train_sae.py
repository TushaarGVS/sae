# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.

import os
from contextlib import nullcontext
from typing import Callable, Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch import profiler
from tqdm import trange

from sparse_autoencoder.modules.loss import mse_auxk_loss, mse_loss
from sparse_autoencoder.modules.sparse_matmul import coo_sparse_dense_matmul
from sparse_autoencoder.tms.sae import TmsFastAutoencoder, TmsAutoencoder
from sparse_autoencoder.tms.toy_model import FastToyModel
from sparse_autoencoder.tms.train_toy_model import generate_feature_batch
from sparse_autoencoder.utils.logger import Logger


def _trace_handler(trace_save_path: str):
    os.makedirs(trace_save_path, exist_ok=True)

    def handler(proton: profiler):
        proton.export_chrome_trace(
            os.path.join(trace_save_path, f"step_{proton.step_num}.json")
        )

    return handler


def _get_grad_norm(sae: TmsFastAutoencoder | TmsAutoencoder) -> torch.Tensor:
    total_sq_norm = torch.zeros((), device="cuda", dtype=torch.float32)
    for param in sae.parameters():
        if param.grad is not None:
            total_sq_norm += (param.grad.float() ** 2).sum()
    return total_sq_norm.sqrt()


def auxk_recons_loss(
    auxk_recons: torch.Tensor, model_recon_err: Float[torch.Tensor, "*b d"]
) -> torch.Tensor:
    """
    auxk_recons = sae.decode(auxk_idxs, auxk_vals) - sae.pre_bias.detach()
    model_recon_err = activations - recons.detach()
    """
    recon_err_mu = model_recon_err.mean(dim=0)
    loss = mse_loss(auxk_recons, model_recon_err) / mse_loss(
        recon_err_mu[None].broadcast_to(model_recon_err.shape), model_recon_err
    )
    return loss


def train_tms_sae(
    toy_model_filepath: str,
    d_model: int,
    n_features: int,
    k: int,
    dead_steps_threshold: int,
    mse_scale: float = 1,
    auxk: int | None = None,
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
    eval_freq: int = 100,
    log_freq: int = 100,
    trace_save_path: str | None = None,
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
    with (
        profiler.profile(
            schedule=profiler.schedule(
                skip_first=10, wait=5, warmup=2, active=6, repeat=4
            ),
            on_trace_ready=_trace_handler(trace_save_path),
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if trace_save_path is not None
        else nullcontext()
    ) as proton:
        if trace_save_path is not None:
            proton.step()

        for step in pbar:
            step_lr = lr * lr_scale((step, steps))
            for group in optim.param_groups:
                group["lr"] = step_lr
            logger.lazy_log_kv("step", step)
            logger.lazy_log_kv("step_lr", step_lr)

            optim.zero_grad(set_to_none=True)
            with torch.no_grad():
                with profiler.record_function("generate_train_activations"):
                    feature_batch: Float[torch.Tensor, "*b f"] = generate_feature_batch(
                        batch_size=batch_size,
                        n_features=n_features,
                        feature_probs=feature_proba,
                        device=torch.device("cuda"),
                    )
                    activations = coo_sparse_dense_matmul(feature_batch, tms_W_tr)

            sae.train()
            with torch.amp.autocast(device_type="cuda"):
                with profiler.record_function("train_sae_fwd"):
                    recons, auxk_info = sae(activations)
                    auxk_recons = (
                        sae.decode(
                            topk_idxs=auxk_info["auxk_idxs"],
                            topk_vals=auxk_info["auxk_vals"],
                        )
                        - sae.pre_bias.detach()
                    )
                with profiler.record_function("train_loss"):
                    _mse_loss, _auxk_loss = mse_auxk_loss(
                        recons, auxk_recons, activations
                    )
                    loss = mse_scale * _mse_loss + auxk_coeff * _auxk_loss
                    logger.lazy_log_kv("train_recons", _mse_loss)
                    logger.lazy_log_kv("train_auxk_recons", _auxk_loss)
                    frac_neurons_active = (
                        torch.sum(sae.stats_last_nonzero <= dead_steps_threshold).item()
                        / n_features
                    )
                    logger.lazy_log_kv("train_frac_neurons_active", frac_neurons_active)

            _unscaled_loss = loss.item()
            logger.lazy_log_kv("loss_scale", scaler.get_scale())
            logger.lazy_log_kv("train_loss", _unscaled_loss)

            with profiler.record_function("train_sae_bwd"):
                loss = scaler.scale(loss)
                loss.backward()
                loss = loss.item()  # drop buffers

            with profiler.record_function("decoder_unit_norm"):
                sae.unit_norm_decoder_()
                sae.unit_norm_decoder_grad_adjustment_()

            with profiler.record_function("gradient_clipping"):
                scaler.unscale_(optim)  # for gradient clipping
                grad_norm = logger.lazy_log_kv("grad_norm", _get_grad_norm(sae))
                if clip_grad is not None:
                    grads = [
                        param.grad
                        for param in sae.parameters()
                        if param.grad is not None
                    ]
                    torch._foreach_mul_(
                        grads, clip_grad / torch.clamp(grad_norm, min=clip_grad)
                    )

            scaler.step(optim)
            scaler.update()

            if (step + 1) % eval_freq == 0 or (step + 1 == steps):
                sae.eval()
                with torch.inference_mode():
                    with profiler.record_function("generate_val_activations"):
                        val_feature_batch: Float[torch.Tensor, "*b f"] = (
                            generate_feature_batch(
                                batch_size=batch_size,
                                n_features=n_features,
                                feature_probs=feature_proba,
                                device=torch.device("cuda"),
                            )
                        )
                        val_activations = coo_sparse_dense_matmul(
                            val_feature_batch, tms_W_tr
                        )
                    with torch.amp.autocast(device_type="cuda"):
                        with profiler.record_function("val_sae_fwd"):
                            val_recons, _ = sae(val_activations)
                        with profiler.record_function("val_loss"):
                            loss = mse_loss(val_recons, val_activations).item()
                            frac_neurons_active = (
                                torch.sum(
                                    sae.stats_last_nonzero <= dead_steps_threshold
                                ).item()
                                / n_features
                            )
                            logger.lazy_log_kv("val_recons", loss)
                            logger.lazy_log_kv(
                                "val_frac_neurons_active", frac_neurons_active
                            )

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
        dead_steps_threshold=512,
        mse_scale=1,
        auxk=16,
        auxk_coeff=1 / 32,
        batch_size=4096,
        steps=20_000,
        feature_proba=0.01,
        lr=1e-4,
        lr_scale=None,
        clip_grad=None,
        model_save_path="artefacts/sae-n_feat=1024-d_model=32-spars=0.99.pt",
        wandb_entity="xyznlp",
        run_id="tms_sae-n_feat=1024-d_model=32-spars=0.99",
        eval_freq=100,
        log_freq=100,
        trace_save_path="artefacts/sae-n_feat=1024-d_model=32-spars=0.99-trace",
    )
