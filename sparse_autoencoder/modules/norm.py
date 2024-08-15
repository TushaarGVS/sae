from typing import Any, Tuple

import torch
import triton
import triton.language as tl
from jaxtyping import Float
from torch import autograd
from torch.amp import custom_fwd, custom_bwd

from sparse_autoencoder.modules.utils import contiguous


@triton.jit
def _unit_normalize_w_bwd_kernel(
    unit_norm_w_ptr,
    dw_ptr,
    w_dw_sum_ptr,
    stride_unit_norm_w_a: int,
    stride_unit_norm_w_b: int,
    stride_dw_a: int,
    stride_dw_b: int,
    stride_w_dw_sum_a: int,
    stride_w_dw_sum_b: int,
    dim_a: int,
    dim_b: int,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
) -> None:
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)

    offsets_a = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    mask_a = offsets_a < dim_a
    offsets_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offsets_b < dim_b
    mask = mask_a[:, None] & mask_b[None]

    unit_norm_w_ptrs = (
        unit_norm_w_ptr
        + offsets_a[:, None] * stride_unit_norm_w_a
        + offsets_b[None] * stride_unit_norm_w_b
    )
    dw_ptrs = dw_ptr + offsets_a[:, None] * stride_dw_a + offsets_b[None] * stride_dw_b
    w_dw_sum_ptrs = (
        w_dw_sum_ptr
        + offsets_a[:, None] * stride_w_dw_sum_a
        + offsets_b[None] * stride_w_dw_sum_b
    )
    unit_norm_w = tl.load(unit_norm_w_ptrs, mask=mask).to(tl.float32)
    dw = tl.load(dw_ptrs, mask=mask).to(tl.float32)
    w_dw_sum = tl.load(w_dw_sum_ptrs, mask=mask).to(tl.float32)

    dw -= unit_norm_w * w_dw_sum
    tl.store(dw_ptrs, dw.to(dw_ptr.dtype.element_ty), mask=mask)


class UnitNormalizeW(autograd.Function):
    """Unit normalization of `W` along a given dimension."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any, w: Float[torch.Tensor, "A B"], dim: int = -1
    ) -> Float[torch.Tensor, "A B"]:
        dim_norms = w.norm(dim=dim, keepdim=True)
        unit_norm_w = w / dim_norms
        ctx.save_for_backward(unit_norm_w, dim_norms)
        ctx.dim = dim
        return unit_norm_w

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, dw: Float[torch.Tensor, "A B"]
    ) -> Tuple[Float[torch.Tensor, "A B"], None]:
        """
        dW = (dW / norm) - (W * (W * dW).sum(dim, keepdim=True) / norm**3))
           = (dW - (W/norm * (W/norm * dW).sum(dim, keepdim=True))) / norm
        """
        unit_norm_w, dim_norms = ctx.saved_tensors
        dim = ctx.dim
        dim_a, dim_b = unit_norm_w.shape
        w_dw_sum = (unit_norm_w * dw).sum(dim, keepdim=True).broadcast_to(dw.shape)

        BLOCK_A = 64
        BLOCK_B = 64
        grid = lambda META: (
            triton.cdiv(dim_a, META["BLOCK_A"]),
            triton.cdiv(dim_b, META["BLOCK_B"]),
        )
        _unit_normalize_w_bwd_kernel[grid](
            unit_norm_w_ptr=unit_norm_w,
            dw_ptr=dw,
            w_dw_sum_ptr=w_dw_sum,
            stride_unit_norm_w_a=unit_norm_w.stride(0),
            stride_unit_norm_w_b=unit_norm_w.stride(1),
            stride_dw_a=dw.stride(0),
            stride_dw_b=dw.stride(1),
            stride_w_dw_sum_a=w_dw_sum.stride(0),
            stride_w_dw_sum_b=w_dw_sum.stride(1),
            dim_a=dim_a,
            dim_b=dim_b,
            BLOCK_A=BLOCK_A,
            BLOCK_B=BLOCK_B,
        )
        dw /= dim_norms
        return dw, None


unit_normalize_w = UnitNormalizeW.apply
