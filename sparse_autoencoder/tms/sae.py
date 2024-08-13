# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.

from typing import Any, Tuple

import torch
from jaxtyping import Float
from torch import autograd
from torch import nn
from torch.amp import custom_fwd, custom_bwd

import sparse_autoencoder.array_typing as at
from sparse_autoencoder.modules.activations import _topk_fwd_kernel
from sparse_autoencoder.modules.sparse_matmul import dense_transpose_sparse_matmul
from sparse_autoencoder.modules.utils import contiguous


def auxk_mask_(
    x: at.TmsFeatures,
    stats_last_nonzero: Float[torch.Tensor, "f"],
    dead_steps_threshold: int,
):
    dead_mask = stats_last_nonzero > dead_steps_threshold
    x *= dead_mask  # inplace to save memory
    return x


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
        Float[torch.Tensor, "b k"],
        Float[torch.Tensor, "b k"],
        Float[torch.Tensor, "b auxk"],
        Float[torch.Tensor, "b auxk"],
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
        return topk_idxs, topk_vals, auxk_idxs, auxk_vals

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, dtopk_idxs, dtopk_vals, dauxk_idxs, dauxk_vals) -> Tuple[
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
        dW_enc = (x - pre_bias) @ dy: dense.T @ sparse
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
        dW_enc = dense_transpose_sparse_matmul(
            x_pre_bias_diff, all_idxs, all_dvals, n_features
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
        self.latent_bias: at.TmsSaeLatentBias = nn.Parameter(torch.zeros(d_model))
        self.W_enc: at.TmsSaeEncoderWeights = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(d_model, n_features))
        )
        self.W_dec: at.TmsSaeDecoderWeights = nn.Parameter(self.W_enc.data.clone().T)
        self._unit_norm_decoder()

        # Last step where a neuron was noted to be nonzero.
        self.register_buffer(
            "stats_last_nonzero", torch.zeros(n_features, dtype=torch.long)
        )

    def _unit_norm_decoder(self) -> None:
        """Normalize latent directions of decoder to unit norm."""
        self.W_dec.data /= self.W_dec.data.norm(dim=1)

    def forward(self, x: at.TmsActivations):
        return fast_encoder_autograd(
            x,
            self.pre_bias,
            self.W_enc,
            self.latent_bias,
            self.k,
            self.auxk,
            self.stats_last_nonzero,
            self.dead_steps_threshold,
        )
