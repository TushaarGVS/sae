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
def _mse_loss_fp16_fwd_kernel(
    output_ptr,
    target_ptr,
    loss_ptr,
    output_target_diff_fp32_ptr,
    stride_output_batch: int,
    stride_output_d: int,
    stride_target_batch: int,
    stride_target_d: int,
    stride_output_target_diff_fp32_batch: int,
    stride_output_target_diff_fp32_d: int,
    batch: int,
    dim_d: int,
    BLOCK_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)

    output_ptr += pid_batch * stride_output_batch
    target_ptr += pid_batch * stride_target_batch
    output_target_diff_fp32_ptr += pid_batch * stride_output_target_diff_fp32_batch

    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < dim_d

    output_ptrs = output_ptr + offsets_d * stride_output_d
    output = tl.load(output_ptrs, mask=mask_d).to(tl.float32)
    target_ptrs = target_ptr + offsets_d * stride_target_d
    target = tl.load(target_ptrs, mask=mask_d).to(tl.float32)

    output_target_diff_fp32 = output - target
    mse = tl.sum(output_target_diff_fp32 * output_target_diff_fp32) / (dim_d * batch)
    tl.atomic_add(loss_ptr, mse)

    output_target_diff_fp32_ptrs = (
        output_target_diff_fp32_ptr + offsets_d * stride_output_target_diff_fp32_d
    )
    tl.store(output_target_diff_fp32_ptrs, output_target_diff_fp32, mask=mask_d)


class MseLoss(autograd.Function):
    """Fuses fp32 cast and MSE computation to save memory."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any, output: Float[torch.Tensor, "B D"], target: Float[torch.Tensor, "B D"]
    ) -> Float32[torch.Tensor, "B"]:
        assert output.shape == target.shape
        assert len(output.shape) == 2
        batch, dim_d = output.shape

        loss = torch.tensor([0.0], dtype=torch.float32, device=output.device)
        # Store (output - target) in fp32 for backward pass.
        output_target_diff_fp32 = torch.zeros(
            batch, dim_d, dtype=torch.float32, device=output.device
        )

        BLOCK_D = triton.next_power_of_2(dim_d)
        grid = lambda META: (batch,)
        _mse_loss_fp16_fwd_kernel[grid](
            output_ptr=output,
            target_ptr=target,
            loss_ptr=loss,
            output_target_diff_fp32_ptr=output_target_diff_fp32,
            stride_output_batch=output.stride(0),
            stride_output_d=output.stride(1),
            stride_target_batch=target.stride(0),
            stride_target_d=target.stride(1),
            stride_output_target_diff_fp32_batch=output_target_diff_fp32.stride(0),
            stride_output_target_diff_fp32_d=output_target_diff_fp32.stride(1),
            batch=batch,
            dim_d=dim_d,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(output_target_diff_fp32)
        return loss

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, dy: Float[torch.Tensor, "1"]
    ) -> Tuple[Float32[torch.Tensor, "B D"], None]:
        (output_target_diff_fp32,) = ctx.saved_tensors

        batch, dim_d = output_target_diff_fp32.shape
        doutput = 2 * dy.item() * output_target_diff_fp32 / (batch * dim_d)
        return doutput, None


mse_loss = MseLoss.apply
