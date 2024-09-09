import glob
import gzip
import pickle
import random
from contextlib import nullcontext
from typing import Callable, Tuple, List, Literal

import numpy as np
import torch
from torch import profiler
from tqdm import tqdm

import sparse_autoencoder.array_typing as at
from recurrentgemma.array_typing import Variant
from sparse_autoencoder.modules.loss import mse_auxk_loss, mse_loss
from sparse_autoencoder.modules.utils import next_power_of_2
from sparse_autoencoder.sae import FastAutoencoder
from sparse_autoencoder.utils.logger import Logger


def _get_grad_norm(sae: FastAutoencoder) -> torch.Tensor:
    total_sq_norm = torch.zeros((), device="cuda", dtype=torch.float32)
    for param in sae.parameters():
        if param.grad is not None:
            total_sq_norm += (param.grad.float() ** 2).sum()
    return total_sq_norm.sqrt()


def batch_activations(
    act_filenames: List[str],
    batch_size: int,
    layer_num: int = 30,
    act_type: Literal["mlp_activation", "rg_lru_states"] = "rg_lru_states",
    stream: torch.cuda.Stream = None,
    drop_last: bool = False,
):
    """
    Returns flat activations of (batch, d_model) in streaming mode.

    RecurrentGemma 9b-it (input_ids, rg_lru_states, mlp_activations):
    - rnn layers: 0, 30
    - attention layers: 2, 29 (= rg_lru is none)
    """
    tensors: List[at.Fl("bl d")] = []
    running_batch_len = 0
    for filename in act_filenames:
        try:
            act_dict = pickle.load(gzip.open(filename, "rb"))
        except:
            continue
        batch_act_list: List[at.Fl("l d")] = act_dict[f"blocks.{layer_num}"][act_type]
        tensors.extend(batch_act_list)
        running_batch_len += sum(batch_act.shape[0] for batch_act in batch_act_list)
        if running_batch_len < batch_size:
            continue

        while running_batch_len >= batch_size:
            if len(tensors) == 1:
                concat = tensors[0]
            else:
                with torch.cuda.stream(stream):
                    concat = torch.cat(tensors, dim=0)

            offset = 0
            while offset + batch_size <= concat.shape[0]:
                yield concat[offset : offset + batch_size]
                running_batch_len -= batch_size
                offset += batch_size
            tensors = [concat[offset:]] if offset < concat.shape[0] else []

    if len(tensors) > 0 and not drop_last:
        yield torch.cat(tensors, dim=0)


def train_sae(
    activations_dir: str,
    layer_num: int,
    activation_type: Literal["mlp", "rg_lru"],
    d_model: int,
    n_features: int,
    k: int,
    dead_tokens_threshold: int,
    mse_scale: float = 1,
    auxk: int | None = None,
    auxk_coeff: float = 1 / 32,
    eval_split: float = 0.01,
    batch_size: int = 8192,
    train_steps: int = 50_000,
    lr: float = 1e-3,
    lr_scale: Callable[[Tuple[int, int]], float] | None = None,
    clip_grad: float | None = None,
    model_save_path: str | None = None,
    wandb_entity: str | None = None,
    run_id: str = "sae",
    eval_freq: int = 100,
    log_freq: int = 100,
    save_freq: int = 100,
    trace_save_path: str | None = None,
    seed: int = 4740,
) -> FastAutoencoder:
    _ckpt_save_path = model_save_path.rsplit(".", 1)[0]
    _lr_scale_id = lr_scale.__name__ if lr_scale is not None else "cosine"
    if lr_scale is None:
        lr_scale = lambda step_steps: np.cos(
            0.5 * np.pi * step_steps[0] / (step_steps[1] - 1)
        )

    activation_type = (
        "rg_lru_states" if activation_type == "rg_lru" else "mlp_activations"
    )
    if auxk is None:
        auxk = next_power_of_2(int(d_model / 2))
    if auxk_coeff is None:
        auxk_coeff = 1 / 32
    dead_steps_threshold = int(dead_tokens_threshold / batch_size)
    config = dict(
        activations_dir=activations_dir,
        layer_num=layer_num,
        activation_type=activation_type,
        n_features=n_features,
        d_model=d_model,
        k=k,
        dead_tokens_threshold=dead_tokens_threshold,
        auxk=auxk,
        auxk_coeff=auxk_coeff,
        eval_split=eval_split,
        batch_size=batch_size,
        lr=lr,
        lr_scale=_lr_scale_id,
        clip_grad=clip_grad,
        model_save_path=model_save_path,
        run_id=run_id,
        seed=seed,
    )
    logger = Logger(config=config, wandb_entity=wandb_entity, run_id=run_id, RANK=0)

    sae = FastAutoencoder(
        n_features=n_features,
        d_model=d_model,
        k=k,
        dead_steps_threshold=dead_steps_threshold,
        auxk=auxk,
    )
    sae.cuda()

    scaler = torch.amp.GradScaler()
    optim = torch.optim.AdamW(
        sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10, fused=True
    )

    activation_filenames = glob.glob(f"{glob.escape(activations_dir)}/*.pkl.gz")
    random.Random(seed).shuffle(activation_filenames)
    num_val_files = int(len(activation_filenames) * eval_split)
    train_activations_iter = batch_activations(
        act_filenames=activation_filenames[num_val_files:],
        batch_size=batch_size,
        layer_num=layer_num,
        act_type=activation_type,
        stream=None,
        drop_last=False,
    )
    train_pbar = tqdm(train_activations_iter, desc="train", leave=True)
    with (
        profiler.profile(
            schedule=profiler.schedule(
                skip_first=10, wait=5, warmup=2, active=6, repeat=4
            ),
            on_trace_ready=profiler.tensorboard_trace_handler(trace_save_path),
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if trace_save_path is not None
        else nullcontext()
    ) as proton:
        for step, train_activations in enumerate(train_pbar):
            if step >= train_steps:
                # Assume steps <= len(dataset).
                break
            if trace_save_path is not None:
                proton.step()
            train_activations = train_activations.cuda()

            step_lr = lr * lr_scale((step, train_steps))
            for group in optim.param_groups:
                group["lr"] = step_lr
            logger.lazy_log_kv("step", step)
            logger.lazy_log_kv("tokens", batch_size * step)
            logger.lazy_log_kv("step_lr", step_lr)

            optim.zero_grad(set_to_none=True)
            sae.train()
            with torch.amp.autocast(device_type="cuda"):
                with profiler.record_function("train_sae_fwd"):
                    recons, auxk_info = sae(train_activations)
                    auxk_recons = (
                        sae.decode(
                            topk_idxs=auxk_info["auxk_idxs"],
                            topk_vals=auxk_info["auxk_vals"],
                        )
                        - sae.pre_bias.detach()
                    )
                with profiler.record_function("train_loss"):
                    _mse_loss, _auxk_loss = mse_auxk_loss(
                        recons, auxk_recons, train_activations
                    )
                    loss = mse_scale * _mse_loss + auxk_coeff * _auxk_loss
                    logger.lazy_log_kv("train_recons", _mse_loss)
                    logger.lazy_log_kv("train_auxk_recons", _auxk_loss)
                    frac_neurons_dead = (
                        torch.sum(sae.stats_last_nonzero > dead_steps_threshold).item()
                        / n_features
                    )
                    logger.lazy_log_kv("train_frac_neurons_dead", frac_neurons_dead)

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

            if (step + 1) % eval_freq == 0:
                sae.eval()
                with torch.inference_mode():
                    val_activations_iter = batch_activations(
                        act_filenames=activation_filenames[:num_val_files],
                        batch_size=batch_size,
                        layer_num=layer_num,
                        act_type=activation_type,
                        stream=None,
                        drop_last=False,
                    )
                    val_pbar = tqdm(val_activations_iter, desc="eval", leave=False)
                    for val_activations in val_pbar:
                        val_activations = val_activations.cuda()
                        with torch.amp.autocast(device_type="cuda"):
                            with profiler.record_function("val_sae_fwd"):
                                val_recons, _ = sae(val_activations)
                            with profiler.record_function("val_loss"):
                                loss = mse_loss(val_recons, val_activations).item()
                                frac_neurons_dead = (
                                    torch.sum(
                                        sae.stats_last_nonzero > dead_steps_threshold
                                    ).item()
                                    / n_features
                                )
                                logger.lazy_log_kv("val_recons", loss)
                                logger.lazy_log_kv(
                                    "val_frac_neurons_dead", frac_neurons_dead
                                )

            logger.dump_lazy_logged_kvs()
            if (step + 1) % log_freq == 0 or (step + 1 == train_steps):
                train_pbar.set_postfix(loss=_unscaled_loss, lr=step_lr)
            if (step + 1) % save_freq == 0 and model_save_path is not None:
                sae.save_pretrained(f"{_ckpt_save_path}.ckpt{step}.pt")

    # Run one final evaluation to record model performance.
    sae.eval()
    with torch.inference_mode():
        val_activations_iter = batch_activations(
            act_filenames=activation_filenames[:num_val_files],
            batch_size=batch_size,
            layer_num=layer_num,
            act_type=activation_type,
            stream=None,
            drop_last=False,
        )
        val_pbar = tqdm(val_activations_iter, desc="eval", leave=False)
        for val_activations in val_pbar:
            val_activations = val_activations.cuda()
            with torch.amp.autocast(device_type="cuda"):
                with profiler.record_function("val_sae_fwd"):
                    val_recons, _ = sae(val_activations)
                with profiler.record_function("val_loss"):
                    loss = mse_loss(val_recons, val_activations).item()
                    frac_neurons_dead = (
                        torch.sum(sae.stats_last_nonzero > dead_steps_threshold).item()
                        / n_features
                    )
                    logger.lazy_log_kv("val_recons", loss)
                    logger.lazy_log_kv("val_frac_neurons_dead", frac_neurons_dead)
    if model_save_path is not None:
        sae.save_pretrained(model_save_path)
    return sae


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class RecurrentGemmaSaeCfg:
        # RecurrentGemma config.
        variant: Variant = "9b"
        data: str = "minipile"
        d_model: int = 4_096

        # Autoencoder config.
        layer_num: int = 30
        activation_type: Literal["mlp", "rg_lru"] = "rg_lru"
        n_features_scale: int = 32
        k: int = 32
        dead_tokens_threshold: int = 10_000_000
        auxk_coeff: float = 1 / 32

        # Training config.
        batch_size: int = 16_384
        train_steps: int = 100_000
        lr: float = 1e-4
        clip_grad: float | None = 1.0

    cfg = RecurrentGemmaSaeCfg()
    run_id = (
        f"sae-"
        f"model=rgemma_{cfg.variant}-"
        f"act={cfg.activation_type}-"
        f"data={cfg.data}-"
        f"n_feat={cfg.n_features_scale * cfg.d_model}"
    )

    _ = train_sae(
        activations_dir=f"/share/rush/tg352/sae/minipile/{cfg.variant}/artefacts",
        layer_num=cfg.layer_num,
        activation_type=cfg.activation_type,
        d_model=cfg.d_model,
        n_features=(cfg.n_features_scale * cfg.d_model),
        k=cfg.k,
        dead_tokens_threshold=cfg.dead_tokens_threshold,
        auxk=next_power_of_2(int(cfg.d_model / 2)),
        auxk_coeff=cfg.auxk_coeff,
        eval_split=0.01,
        batch_size=cfg.batch_size,
        train_steps=cfg.train_steps,
        lr=cfg.lr,
        clip_grad=cfg.clip_grad,
        model_save_path=f"artefacts/{run_id}.pt",
        wandb_entity="xyznlp",
        run_id=run_id,
        eval_freq=100,
        log_freq=10,
        save_freq=500,
        trace_save_path=None,
    )
