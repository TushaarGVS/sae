# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.

from typing import Any, Tuple, Dict

import torch
import triton
from torch import autograd
from torch import nn
from torch.amp import custom_fwd, custom_bwd

import sparse_autoencoder.array_typing as at
from sparse_autoencoder.modules.activations import _topk_fwd_kernel
from sparse_autoencoder.modules.norm import _unit_normalize_w_bwd_kernel
from sparse_autoencoder.modules.sparse_matmul import (
    sparse_dense_matmul,
    _dense_transpose_sparse_matmul_fwd_kernel,
)
from sparse_autoencoder.modules.utils import contiguous

_zeros = lambda size: nn.Parameter(torch.zeros(size))
_xavier_empty = lambda size: nn.Parameter(nn.init.xavier_normal_(torch.empty(size)))


class FastEncoderAutograd(autograd.Function):
    """Computes: topk[(x - pre_bias) @ W_enc + latent_bias]."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    @at.typed
    def forward(
        ctx: Any,
        x: at.Fl("*bl d"),
        pre_bias: at.SaePreBias,
        W_enc: at.SaeEncoderWeights,
        latent_bias: at.SaeLatentBias,
        k: int,
        auxk: int | None,
        stats_last_nonzero: at.Fl("f"),
        dead_steps_threshold: int,
    ) -> Tuple[
        at.Fl("*bl k"),
        at.Fl("*bl k"),
        at.Fl("*bl auxk") | None,
        at.Fl("*bl auxk") | None,
        at.Fl("f"),
    ]:
        x -= pre_bias
        latents_pre_act: at.Features = x @ W_enc
        topk_idxs, topk_vals = _topk_fwd_kernel(latents_pre_act, k, latent_bias)

        # Compute num times feature was in topk and nonzero.
        num_times_feat_nonzero = torch.zeros_like(stats_last_nonzero)
        num_times_feat_nonzero.scatter_add_(
            dim=0,
            index=topk_idxs.view(-1),
            src=(topk_vals > 1e-3).to(stats_last_nonzero.dtype).view(-1),
        )
        stats_last_nonzero *= 1 - num_times_feat_nonzero.clamp(max=1)  # 0 for active
        stats_last_nonzero += 1

        # For auxk loss: recons error using top-auxk dead neurons.
        auxk_idxs, auxk_vals = None, None
        if auxk is not None:
            dead_mask = stats_last_nonzero > dead_steps_threshold
            latents_pre_act *= dead_mask
            auxk_idxs, auxk_vals = _topk_fwd_kernel(latents_pre_act, auxk)

        ctx.save_for_backward(x, W_enc, topk_idxs, auxk_idxs)
        return topk_idxs, topk_vals, auxk_idxs, auxk_vals, stats_last_nonzero

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    @at.typed
    def backward(
        ctx: Any, _, dtopk_vals: at.Fl("*bl k"), __, dauxk_vals: at.Fl("*bl auxk"), ___
    ) -> Tuple[None, at.Fl("d"), at.Fl("d f"), at.Fl("f"), None, None, None, None]:
        """
        dpre_bias = -(dy @ W_enc.T).sum(0) = -dy.sum(0) @ W_enc.T
        dW_enc = x_pre_bias_diff @ dy
        dlatent_bias = dy.sum(0)
        """
        x_pre_bias_diff, W_enc, topk_idxs, auxk_idxs = ctx.saved_tensors
        if auxk_idxs is not None:
            topk_idxs: at.Fl("*bl k_auxk") = torch.cat([topk_idxs, auxk_idxs], -1)
            dtopk_vals: at.Fl("*bl k_auxk") = torch.cat([dtopk_vals, dauxk_vals], -1)

        n_features = W_enc.shape[-1]
        dy_sum0 = torch.zeros(n_features, dtype=torch.float32, device=dtopk_vals.device)
        dy_sum0.scatter_add_(
            dim=0, index=topk_idxs.view(-1), src=dtopk_vals.view(-1).to(torch.float32)
        )

        dim_b, dim_k_auxk = topk_idxs.shape
        dim_d = x_pre_bias_diff.shape[-1]
        dW_enc = torch.zeros(
            dim_d, n_features, device=dtopk_vals.device, dtype=dtopk_vals.dtype
        )
        BLOCK_B = triton.next_power_of_2(dim_b)
        BLOCK_D = triton.next_power_of_2(dim_d)
        _dense_transpose_sparse_matmul_fwd_kernel[(dim_k_auxk,)](
            dense_ptr=x_pre_bias_diff,
            sparse_idxs_ptr=topk_idxs,
            sparse_vals_ptr=dtopk_vals,
            out_ptr=dW_enc,
            stride_dense_n=x_pre_bias_diff.stride(0),
            stride_dense_a=x_pre_bias_diff.stride(1),
            stride_sparse_idxs_n=topk_idxs.stride(0),
            stride_sparse_idxs_k=topk_idxs.stride(1),
            stride_sparse_vals_n=dtopk_vals.stride(0),
            stride_sparse_vals_k=dtopk_vals.stride(1),
            stride_out_a=dW_enc.stride(0),
            stride_out_b=dW_enc.stride(1),
            dim_n=dim_b,
            dim_a=dim_d,
            BLOCK_N=BLOCK_B,
            BLOCK_A=BLOCK_D,
        )
        return None, -dy_sum0 @ W_enc.T, dW_enc, dy_sum0, None, None, None, None


fast_encoder_autograd = FastEncoderAutograd.apply


class FastAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_features: int,
        k: int,
        dead_steps_threshold: int,
        auxk: int | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_features = n_features
        self.k = k
        self.dead_steps_threshold = dead_steps_threshold
        self.auxk = auxk

        self.pre_bias: at.SaePreBias = _zeros(d_model)
        self.latent_bias: at.SaeLatentBias = _zeros(n_features)
        self.W_enc: at.SaeEncoderWeights = _xavier_empty([d_model, n_features])
        self.W_dec: at.SaeDecoderWeights = nn.Parameter(self.W_enc.data.clone().T)
        self._dec_norms = None
        self.unit_norm_decoder_()

        self.register_buffer(
            "stats_last_nonzero", torch.zeros(n_features, dtype=torch.long)
        )

    def save_pretrained(self, model_filepath: str) -> None:
        torch.save(self.state_dict(), model_filepath)

    def from_pretrained(self, model_filepath: str):
        state_dict = torch.load(
            model_filepath, map_location=torch.device("cpu"), weights_only=True
        )
        self.load_state_dict(state_dict)

    def unit_norm_decoder_(self: "FastAutoEncoder") -> None:
        self._dec_norms = self.W_dec.data.norm(dim=-1, keepdim=True)  # for grad bwd
        self.W_dec.data /= self._dec_norms

    def unit_norm_decoder_grad_adjustment_(self: "FastAutoEncoder") -> None:
        assert self.W_dec.grad is not None and self._dec_norms is not None
        W_dW_sum = (
            (self.W_dec.data * self.W_dec.grad)
            .sum(dim=1, keepdim=True)
            .broadcast_to(self.n_features, self.d_model)
        )
        BLOCK_A = 64
        BLOCK_B = 64
        grid = lambda META: (
            triton.cdiv(self.n_features, META["BLOCK_A"]),
            triton.cdiv(self.d_model, META["BLOCK_B"]),
        )
        _unit_normalize_w_bwd_kernel[grid](
            unit_norm_w_ptr=self.W_dec.data,
            dw_ptr=self.W_dec.grad,
            w_dw_sum_ptr=W_dW_sum,
            stride_unit_norm_w_a=self.W_dec.data.stride(0),
            stride_unit_norm_w_b=self.W_dec.data.stride(1),
            stride_dw_a=self.W_dec.grad.stride(0),
            stride_dw_b=self.W_dec.grad.stride(1),
            stride_w_dw_sum_a=W_dW_sum.stride(0),
            stride_w_dw_sum_b=W_dW_sum.stride(1),
            dim_a=self.n_features,
            dim_b=self.d_model,
            BLOCK_A=BLOCK_A,
            BLOCK_B=BLOCK_B,
        )
        self.W_dec.grad /= self._dec_norms

    @at.typed
    def encode(self, x: at.Fl("*bl d")) -> Tuple[
        at.Fl("*bl k"),
        at.Fl("*bl k"),
        at.Fl("*bl auxk") | None,
        at.Fl("*bl auxk") | None,
    ]:
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
        self.stats_last_nonzero = _stats
        return topk_idxs, topk_vals, auxk_idxs, auxk_vals

    @at.typed
    def decode(
        self, topk_idxs: at.Fl("*bl k"), topk_vals: at.Fl("*bl k")
    ) -> at.Fl("*bl d"):
        return sparse_dense_matmul(topk_idxs, topk_vals, self.W_dec, self.pre_bias)

    @at.typed
    def forward(
        self, x: at.Fl("*bl d")
    ) -> Tuple[at.Fl("*bl d"), Dict[str, at.Fl("*bl auxk") | None]]:
        topk_idxs, topk_vals, auxk_idxs, auxk_vals = self.encode(x)
        recons: at.Fl("*bl d") = self.decode(topk_idxs, topk_vals)
        return recons, dict(auxk_idxs=auxk_idxs, auxk_vals=auxk_vals)
