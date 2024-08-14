# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.

from typing import Any, Tuple, Dict

import torch
import torch.nn.functional as F
import triton
from jaxtyping import Float
from torch import autograd
from torch import nn
from torch.amp import custom_fwd, custom_bwd

import sparse_autoencoder.array_typing as at
from sparse_autoencoder.modules.activations import _topk_fwd_kernel
from sparse_autoencoder.modules.sparse_matmul import (
    _dense_transpose_sparse_matmul_fwd_kernel,
    sparse_dense_matmul,
)
from sparse_autoencoder.modules.utils import contiguous


def auxk_mask_(
    x: at.TmsFeatures,
    stats_last_nonzero: Float[torch.Tensor, "f"],
    dead_steps_threshold: int,
):
    dead_mask = stats_last_nonzero > dead_steps_threshold
    x *= dead_mask  # inplace to save memory
    return x


class TmsAutoencoder(nn.Module):
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
        self.latent_bias: at.TmsSaeLatentBias = nn.Parameter(torch.zeros(n_features))
        self.W_enc: at.TmsSaeEncoderWeights = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(d_model, n_features))
        )
        self.W_dec: at.TmsSaeDecoderWeights = nn.Parameter(self.W_enc.data.clone().T)
        self._unit_norm_decoder()

        self.register_buffer(
            "stats_last_nonzero", torch.zeros(n_features, dtype=torch.long)
        )

    def _unit_norm_decoder(self: "TmsFastAutoencoder") -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

    def forward(
        self, x: at.TmsActivations
    ) -> Tuple[at.TmsActivations, Dict[str, Float[torch.Tensor, "*b auxk"] | None]]:
        latents_pre_act = ((x - self.pre_bias) @ self.W_enc) + self.latent_bias
        latents_topk = latents_pre_act.topk(dim=-1, k=self.k)
        topk_idxs, topk_vals = latents_topk.indices, latents_topk.values
        topk_vals = F.relu(topk_vals)

        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(
            dim=0,
            index=topk_idxs.reshape(-1),
            src=(topk_idxs > 1e-3).to(tmp.dtype).reshape(-1),
        )
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1
        if self.auxk is not None:
            latents_pre_act = auxk_mask_(
                latents_pre_act, self.stats_last_nonzero, self.dead_steps_threshold
            )
            auxk_topk = latents_pre_act.topk(dim=-1, k=self.auxk)
            auxk_idxs, auxk_vals = auxk_topk.indices, auxk_topk.values
            auxk_vals = F.relu(auxk_vals)
        else:
            auxk_idxs, auxk_vals = None, None

        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
        recons = (latents @ self.W_dec) + self.pre_bias
        return recons, dict(auxk_idxs=auxk_idxs, auxk_vals=auxk_vals)


class FastEncoderAutograd(autograd.Function):
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
        auxk: int | None,
        stats_last_nonzero: Float[torch.Tensor, "f"],
        dead_steps_threshold: int,
    ) -> Tuple[
        Float[torch.Tensor, "*b k"],
        Float[torch.Tensor, "*b k"],
        Float[torch.Tensor, "*b auxk"],
        Float[torch.Tensor, "*b auxk"],
        Float[torch.Tensor, "f"],
    ]:
        x_pre_bias_diff = x - pre_bias
        latents_pre_act: at.TmsFeatures = x_pre_bias_diff @ W_enc
        topk_idxs, topk_vals = _topk_fwd_kernel(latents_pre_act, k, latent_bias)

        # Compute the number of times a feature was picked in topk and was nonzero.
        num_times_feat_nonzero = torch.zeros_like(stats_last_nonzero)
        num_times_feat_nonzero.scatter_add_(
            dim=0,
            index=topk_idxs.view(-1),
            src=(topk_vals > 1e-3).to(stats_last_nonzero.dtype).view(-1),
        )
        stats_last_nonzero *= 1 - num_times_feat_nonzero.clamp(max=1)  # 0 for active
        stats_last_nonzero += 1

        # For auxk loss: reconstruction error using top-auxk dead latents.
        auxk_idxs, auxk_vals = None, None
        if auxk is not None:
            auxk_idxs, auxk_vals = _topk_fwd_kernel(
                auxk_mask_(latents_pre_act, stats_last_nonzero, dead_steps_threshold),
                auxk,
            )

        ctx.save_for_backward(x_pre_bias_diff, W_enc, topk_idxs, auxk_idxs)
        return topk_idxs, topk_vals, auxk_idxs, auxk_vals, stats_last_nonzero

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, dtopk_idxs, dtopk_vals, dauxk_idxs, dauxk_vals, dstats_last_nonzero
    ) -> Tuple[
        None,
        Float[torch.Tensor, "d"],
        Float[torch.Tensor, "d f"],
        Float[torch.Tensor, "f"],
        None,
        None,
        None,
        None,
    ]:
        """
        dpre_bias = -(dy @ W_enc.T).sum(0) = -dy.sum(0) @ W_enc.T
        dW_enc = (x - pre_bias).T: [D B] @ dy: [B F] (= dense.T @ sparse)
        dlatent_bias = dy.sum(0)
        """
        x_pre_bias_diff, W_enc, topk_idxs, auxk_idxs = ctx.saved_tensors
        if auxk_idxs is not None:
            all_idxs: Float[torch.Tensor, "b k_auxk"] = torch.cat(
                [topk_idxs, auxk_idxs], dim=-1
            )
            all_dvals: Float[torch.Tensor, "b k_auxk"] = torch.cat(
                [dtopk_vals, dauxk_vals], dim=-1
            )
        else:
            all_idxs: Float[torch.Tensor, "b k"] = topk_idxs
            all_dvals: Float[torch.Tensor, "b k"] = dtopk_vals

        n_features = W_enc.shape[-1]
        dy_sum0 = torch.zeros(n_features, dtype=torch.float32, device=all_dvals.device)
        dy_sum0.scatter_add_(
            dim=0, index=all_idxs.view(-1), src=all_dvals.view(-1).to(torch.float32)
        )

        dim_b, dim_k_auxk = all_idxs.shape
        dim_d = x_pre_bias_diff.shape[1]
        dW_enc = torch.zeros(
            dim_d, n_features, device=all_dvals.device, dtype=all_dvals.dtype
        )
        BLOCK_B = triton.next_power_of_2(dim_b)
        BLOCK_D = triton.next_power_of_2(dim_d)
        _dense_transpose_sparse_matmul_fwd_kernel[(dim_k_auxk,)](
            dense_ptr=x_pre_bias_diff,
            sparse_idxs_ptr=all_idxs,
            sparse_vals_ptr=all_dvals,
            out_ptr=dW_enc,
            stride_dense_n=x_pre_bias_diff.stride(0),
            stride_dense_a=x_pre_bias_diff.stride(1),
            stride_sparse_idxs_n=all_idxs.stride(0),
            stride_sparse_idxs_k=all_idxs.stride(1),
            stride_sparse_vals_n=all_dvals.stride(0),
            stride_sparse_vals_k=all_dvals.stride(1),
            stride_out_a=dW_enc.stride(0),
            stride_out_b=dW_enc.stride(1),
            dim_n=dim_b,
            dim_a=dim_d,
            BLOCK_N=BLOCK_B,
            BLOCK_A=BLOCK_D,
        )
        return None, -dy_sum0 @ W_enc.T, dW_enc, dy_sum0, None, None, None, None


fast_encoder_autograd = FastEncoderAutograd.apply


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
        self.latent_bias: at.TmsSaeLatentBias = nn.Parameter(torch.zeros(n_features))
        self.W_enc: at.TmsSaeEncoderWeights = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(d_model, n_features))
        )
        self.W_dec: at.TmsSaeDecoderWeights = nn.Parameter(self.W_enc.data.clone().T)
        self._unit_norm_decoder()

        # Last step where a neuron was noted to be nonzero.
        self.register_buffer(
            "stats_last_nonzero", torch.zeros(n_features, dtype=torch.long)
        )

    @classmethod
    def _update_stats(cls, stats_last_nonzero: Float[torch.Tensor, "f"]) -> None:
        cls.stats_last_nonzero = stats_last_nonzero

    def _unit_norm_decoder(self: "TmsFastAutoencoder") -> None:
        """Normalize latent directions of decoder to unit norm."""
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

    def encode(self, x: at.TmsActivations):
        topk_idxs, topk_vals, auxk_idxs, auxk_vals, _stats = fast_encoder_autograd(
            x,
            self.pre_bias,
            self.W_enc,
            self.latent_bias,
            self.k,
            self.auxk,
            self.stats_last_nonzero,
            self.dead_steps_threshold,
        )
        self._update_stats(_stats)
        return topk_idxs, topk_vals, auxk_idxs, auxk_vals

    def decode(
        self,
        topk_idxs: Float[torch.Tensor, "*b k"],
        topk_vals: Float[torch.Tensor, "*b k"],
    ) -> at.TmsActivations:
        return sparse_dense_matmul(topk_idxs, topk_vals, self.W_dec, self.pre_bias)

    def forward(
        self, x: at.TmsActivations
    ) -> Tuple[at.TmsActivations, Dict[str, Float[torch.Tensor, "*b auxk"] | None]]:
        topk_idxs, topk_vals, auxk_idxs, auxk_vals = self.encode(x)
        recons: at.TmsActivations = self.decode(topk_idxs, topk_vals)
        return recons, dict(auxk_idxs=auxk_idxs, auxk_vals=auxk_vals)
