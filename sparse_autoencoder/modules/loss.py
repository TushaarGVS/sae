# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.
from typing import Any, Tuple

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Float32
from torch import autograd
from torch.amp import custom_fwd, custom_bwd

from sparse_autoencoder.modules.utils import contiguous


@triton.jit
def _unnorm_mse(target_output_diff: tl.tensor) -> tl.tensor:
    return tl.sum(target_output_diff * target_output_diff)


@triton.jit
def _mse(target_output_diff: tl.tensor, norm: int) -> tl.tensor:
    _one = 1.0
    if norm is None:
        norm = _one.to(tl.float32)
    return tl.sum(target_output_diff * target_output_diff) / norm


@triton.jit
def _mse_loss_fp16_fwd_kernel(
    output_ptr,
    target_ptr,
    loss_ptr,
    stride_output_batch: int,
    stride_output_d: int,
    stride_target_batch: int,
    stride_target_d: int,
    batch: int,
    dim_d: int,
    BLOCK_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)

    output_ptr += pid_batch * stride_output_batch
    target_ptr += pid_batch * stride_target_batch

    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < dim_d

    output_ptrs = output_ptr + offsets_d * stride_output_d
    output = tl.load(output_ptrs, mask=mask_d).to(tl.float32)
    target_ptrs = target_ptr + offsets_d * stride_target_d
    target = tl.load(target_ptrs, mask=mask_d).to(tl.float32)

    mse = _mse(output - target, dim_d * batch)
    tl.atomic_add(loss_ptr, mse)


class MseLoss(autograd.Function):
    """Fuses fp32 cast and MSE computation to save memory."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any, output: Float[torch.Tensor, "B D"], target: Float[torch.Tensor, "B D"]
    ) -> Float32[torch.Tensor, "1"]:
        assert output.shape == target.shape
        assert len(output.shape) == 2
        batch, dim_d = output.shape

        loss = torch.tensor(0.0, dtype=torch.float32, device=output.device)
        BLOCK_D = triton.next_power_of_2(dim_d)
        grid = lambda META: (batch,)
        _mse_loss_fp16_fwd_kernel[grid](
            output_ptr=output,
            target_ptr=target,
            loss_ptr=loss,
            stride_output_batch=output.stride(0),
            stride_output_d=output.stride(1),
            stride_target_batch=target.stride(0),
            stride_target_d=target.stride(1),
            batch=batch,
            dim_d=dim_d,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(output - target)
        return loss

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, dy: Float[torch.Tensor, "1"]
    ) -> Tuple[Float32[torch.Tensor, "B D"], None]:
        (output_target_diff,) = ctx.saved_tensors
        output_target_diff = output_target_diff.float()

        batch, dim_d = output_target_diff.shape
        doutput = 2 * dy.item() * output_target_diff / (batch * dim_d)
        return doutput, None


mse_loss = MseLoss.apply


@triton.jit
def _mse_auxk_loss_fp16_fwd_kernel(
    acts_ptr,
    model_recons_ptr,
    auxk_recons_ptr,
    recons_loss_ptr,
    auxk_loss_num_ptr,
    auxk_loss_denom_ptr,
    stride_acts_batch: int,
    stride_acts_d: int,
    stride_model_recons_batch: int,
    stride_model_recons_d: int,
    stride_auxk_recons_batch: int,
    stride_auxk_recons_d: int,
    batch: int,
    dim_d: int,
    BLOCK_D: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    pid_d = tl.program_id(0)

    offsets_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < dim_d

    acts_model_recons_diff_dim0_mu = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for batch_idx in range(0, batch, BLOCK_BATCH):
        offsets_batch = batch_idx + tl.arange(0, BLOCK_BATCH)
        mask_batch = offsets_batch < batch
        mask = mask_d[None] & mask_batch[:, None]

        acts_ptrs = (
            acts_ptr
            + offsets_d[None] * stride_acts_d
            + offsets_batch[:, None] * stride_acts_batch
        )
        model_recons_ptrs = (
            model_recons_ptr
            + offsets_d[None] * stride_model_recons_d
            + offsets_batch[:, None] * stride_model_recons_batch
        )
        acts = tl.load(acts_ptrs, mask=mask).to(tl.float32)
        model_recons = tl.load(model_recons_ptrs, mask=mask).to(tl.float32)
        acts_model_recons_diff = acts - model_recons
        acts_model_recons_diff_dim0_mu += tl.sum(acts_model_recons_diff, axis=0) / batch

    for batch_idx in range(0, batch, BLOCK_BATCH):
        offsets_batch = batch_idx + tl.arange(0, BLOCK_BATCH)
        mask_batch = offsets_batch < batch
        mask = mask_d[None] & mask_batch[:, None]

        acts_ptrs = (
            acts_ptr
            + offsets_d[None] * stride_acts_d
            + offsets_batch[:, None] * stride_acts_batch
        )
        model_recons_ptrs = (
            model_recons_ptr
            + offsets_d[None] * stride_model_recons_d
            + offsets_batch[:, None] * stride_model_recons_batch
        )
        auxk_recons_ptrs = (
            auxk_recons_ptr
            + offsets_d[None] * stride_auxk_recons_d
            + offsets_batch[:, None] * stride_auxk_recons_batch
        )
        acts = tl.load(acts_ptrs, mask=mask).to(tl.float32)
        model_recons = tl.load(model_recons_ptrs, mask=mask).to(tl.float32)
        auxk_recons = tl.load(auxk_recons_ptrs, mask=mask).to(tl.float32)

        acts_model_recons_diff = acts - model_recons
        mse = _mse(acts_model_recons_diff, batch * dim_d)
        tl.atomic_add(recons_loss_ptr, mse)

        auxk_loss_num = _unnorm_mse(auxk_recons - acts_model_recons_diff)
        tl.atomic_add(auxk_loss_num_ptr, auxk_loss_num)
        # Note: `acts_model_recons_diff_dim0_mu` is populated across the whole grid,
        # it's important to remove the `mask` values.
        auxk_loss_denom = _unnorm_mse(
            acts_model_recons_diff_dim0_mu * mask - acts_model_recons_diff
        )
        tl.atomic_add(auxk_loss_denom_ptr, auxk_loss_denom)


class MseAuxKLoss(autograd.Function):
    """
    Computes MSE loss on recons and auxk loss on auxk_recons within the same kernel.
    Also, the kernel fuses fp32 with the loss computation to save memory.

    The normalization is computed per token, because the scale of the error changes
    throughout training:
    - model_recon_error = target - output
    - mse_loss = mse(output, target)
    - auxk_loss = mse(auxk_recons, model_recon_error) /
                    mse(model_recon_error.mean(dim=0), model_recon_error)
    """

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        model_recons: Float[torch.Tensor, "B D"],
        auxk_recons: Float[torch.Tensor, "B D"],
        acts: Float[torch.Tensor, "B D"],
    ) -> Tuple[Float32[torch.Tensor, "1"], Float32[torch.Tensor, "1"]]:
        assert model_recons.shape == acts.shape
        assert len(model_recons.shape) == 2
        batch, dim_d = model_recons.shape

        fp32, device = torch.float32, model_recons.device
        recons_loss = torch.tensor(0.0, dtype=fp32, device=device)
        auxk_loss_num = torch.tensor(0.0, dtype=fp32, device=device)
        auxk_loss_denom = torch.tensor(0.0, dtype=fp32, device=device)

        BLOCK_BATCH = min(triton.next_power_of_2(batch), 512)
        BLOCK_D = 64
        grid = lambda META: (triton.cdiv(dim_d, BLOCK_D),)
        _mse_auxk_loss_fp16_fwd_kernel[grid](
            acts_ptr=acts,
            model_recons_ptr=model_recons,
            auxk_recons_ptr=auxk_recons,
            recons_loss_ptr=recons_loss,
            auxk_loss_num_ptr=auxk_loss_num,
            auxk_loss_denom_ptr=auxk_loss_denom,
            stride_acts_batch=acts.stride(0),
            stride_acts_d=acts.stride(1),
            stride_model_recons_batch=model_recons.stride(0),
            stride_model_recons_d=model_recons.stride(1),
            stride_auxk_recons_batch=auxk_recons.stride(0),
            stride_auxk_recons_d=auxk_recons.stride(1),
            batch=batch,
            dim_d=dim_d,
            BLOCK_D=BLOCK_D,
            BLOCK_BATCH=BLOCK_BATCH,
        )
        auxk_loss = (auxk_loss_num / auxk_loss_denom).nan_to_num(0)

        ctx.save_for_backward(model_recons, auxk_recons, acts, auxk_loss_denom)
        return recons_loss, auxk_loss

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any,
        drecons_loss: Float32[torch.Tensor, "1"],
        dauxk_loss: Float32[torch.Tensor, "1"],
    ) -> Tuple[Float[torch.Tensor, "B D"], Float[torch.Tensor, "B D"], None]:
        """
        No backpropagation through (acts - model_recons) in auxk_loss computation.

        dmodel_recons = drecons_loss * [(1 / batch * dim_d) * 2 * -acts_recons_diff]
        dauxk_recons = dauxk_loss * 2 * auxk_model_err_diff * (1 / auxk_loss_denom)
        """
        model_recons, auxk_recons, acts, auxk_loss_denom = ctx.saved_tensors
        batch, dim_d = model_recons.shape
        acts_recons_diff = (acts - model_recons).float()
        auxk_model_err_diff = (auxk_recons - acts_recons_diff).float()

        dmodel_recons = -2 * drecons_loss.item() * acts_recons_diff / (batch * dim_d)
        dauxk_recons = 2 * dauxk_loss.item() * auxk_model_err_diff / auxk_loss_denom
        return dmodel_recons, dauxk_recons, None


mse_auxk_loss = MseAuxKLoss.apply
