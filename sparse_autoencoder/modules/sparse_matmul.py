# Adapted from openai/sparse_autoencoder: https://github.com/openai/sparse_autoencoder.

from typing import Any
from typing import Tuple

import torch
import triton
import triton.language as tl
from jaxtyping import Float
from torch import autograd
from torch.amp import custom_fwd, custom_bwd

from sparse_autoencoder.modules.utils import contiguous

# Performance enhancement, but slight loss in precision.
torch.set_float32_matmul_precision("high")


@triton.jit
def _get_elem(tensor: tl.tensor, idx: tl.int64, dtype: tl.dtype):
    """
    Returns `tensor[idx]`.

    Array indexing is not supported in triton=3.0.0:
    https://github.com/triton-lang/triton/issues/974.
    """
    return tl.sum(
        tl.where(
            tl.arange(0, tensor.numel) == idx,
            tensor,
            tl.zeros((tensor.numel,), dtype=dtype),
        )
    )


@triton.jit
def _coo_sparse_dense_matmul_fwd_kernel(
    coo_idxs_ptr,
    coo_vals_ptr,
    dense_ptr,
    out_ptr,
    stride_coo_idxs_row: int,
    stride_coo_idxs_k: int,
    stride_coo_vals_k: int,
    stride_dense_n: int,
    stride_dense_b: int,
    stride_out_a: int,
    stride_out_b: int,
    dim_k: int,
    dim_b: int,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
) -> None:
    """Computes coo_sparse @ dense."""
    pid_k = tl.program_id(0)

    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    coo_idxs_ptrs_row1 = coo_idxs_ptr + offsets_k * stride_coo_idxs_k
    coo_idxs_ptrs_row2 = (
        coo_idxs_ptr + stride_coo_idxs_row + offsets_k * stride_coo_idxs_k
    )
    mask_k = offsets_k < dim_k
    coo_idxs_row1 = tl.load(coo_idxs_ptrs_row1, mask=mask_k)
    coo_idxs_row2 = tl.load(coo_idxs_ptrs_row2, mask=mask_k)
    coo_vals_ptrs = coo_vals_ptr + offsets_k * stride_coo_vals_k
    coo_vals = tl.load(coo_vals_ptrs, mask=mask_k)

    offsets_b = tl.arange(0, BLOCK_B)
    mask_b = offsets_b < dim_b

    accum = tl.zeros((BLOCK_B,), dtype=tl.float32)
    last_coo_idxs_row1_idx = tl.min(coo_idxs_row1)  # reset `accum` when this changes
    for block_idx in range(BLOCK_K):
        dim_idx = block_idx + pid_k * BLOCK_K
        if dim_idx < dim_k:
            coo_idxs_row1_idx = _get_elem(coo_idxs_row1, block_idx, coo_idxs_row1.dtype)
            coo_idxs_row2_idx = _get_elem(coo_idxs_row2, block_idx, coo_idxs_row2.dtype)
            coo_val = _get_elem(coo_vals, block_idx, coo_vals.dtype)

            dense_ptrs = (
                dense_ptr
                + coo_idxs_row2_idx * stride_dense_n
                + offsets_b * stride_dense_b
            )
            dense_row = tl.load(dense_ptrs, mask=mask_b)

            if coo_idxs_row1_idx != last_coo_idxs_row1_idx:
                out_ptrs = (
                    out_ptr
                    + last_coo_idxs_row1_idx * stride_out_a
                    + offsets_b * stride_out_b
                )
                tl.atomic_add(out_ptrs, accum.to(out_ptr.dtype.element_ty), mask=mask_b)
                accum *= 0  # reset
                last_coo_idxs_row1_idx = coo_idxs_row1_idx

            accum += coo_val * dense_row

    out_ptrs = (
        out_ptr + last_coo_idxs_row1_idx * stride_out_a + offsets_b * stride_out_b
    )
    tl.atomic_add(out_ptrs, accum.to(out_ptr.dtype.element_ty), mask=mask_b)


@triton.jit
def _coo_sparse_dense_matmul_bwd_kernel(
    coo_idxs_ptr,
    coo_vals_ptr,
    dy_ptr,
    dw_ptr,
    stride_coo_idxs_row: int,
    stride_coo_idxs_k: int,
    stride_coo_vals_k: int,
    stride_dy_a: int,
    stride_dy_b: int,
    stride_dw_n: int,
    stride_dw_b: int,
    dim_k: int,
    dim_b: int,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
) -> None:
    """Compute coo_sparse.T @ dense."""
    pid_k = tl.program_id(0)

    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    coo_idxs_ptrs_row1 = coo_idxs_ptr + offsets_k * stride_coo_idxs_k
    coo_idxs_ptrs_row2 = (
        coo_idxs_ptr + stride_coo_idxs_row + offsets_k * stride_coo_idxs_k
    )
    mask_k = offsets_k < dim_k
    coo_idxs_row1 = tl.load(coo_idxs_ptrs_row1, mask=mask_k)
    coo_idxs_row2 = tl.load(coo_idxs_ptrs_row2, mask=mask_k)
    coo_vals_ptrs = coo_vals_ptr + offsets_k * stride_coo_vals_k
    coo_vals = tl.load(coo_vals_ptrs, mask=mask_k)

    offsets_b = tl.arange(0, BLOCK_B)
    mask_b = offsets_b < dim_b

    accum = tl.zeros((BLOCK_B,), dtype=tl.float32)
    last_coo_idxs_row2_idx = tl.min(coo_idxs_row2)  # reset `accum` when this changes
    for block_idx in range(BLOCK_K):
        dim_idx = block_idx + pid_k * BLOCK_K
        if dim_idx < dim_k:
            coo_idxs_row1_idx = _get_elem(coo_idxs_row1, block_idx, coo_idxs_row1.dtype)
            coo_idxs_row2_idx = _get_elem(coo_idxs_row2, block_idx, coo_idxs_row2.dtype)
            coo_val = _get_elem(coo_vals, block_idx, coo_vals.dtype)

            dy_ptrs = dy_ptr + coo_idxs_row1_idx * stride_dy_a + offsets_b * stride_dy_b
            dy_row = tl.load(dy_ptrs, mask=mask_b)

            if coo_idxs_row2_idx != last_coo_idxs_row2_idx:
                dw_ptrs = (
                    dw_ptr
                    + last_coo_idxs_row2_idx * stride_dw_n
                    + offsets_b * stride_dw_b
                )
                tl.atomic_add(dw_ptrs, accum.to(dw_ptr.dtype.element_ty), mask=mask_b)
                accum *= 0  # reset
                last_coo_idxs_row2_idx = coo_idxs_row2_idx

            accum += coo_val * dy_row

    dw_ptrs = dw_ptr + last_coo_idxs_row2_idx * stride_dw_n + offsets_b * stride_dw_b
    tl.atomic_add(dw_ptrs, accum.to(dw_ptr.dtype.element_ty), mask=mask_b)


class CooSparseDenseMatmul(autograd.Function):
    """Computes (x: [A N] @ dense: [N B] = out: [A B])."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: Float[torch.Tensor, "A N"],
        dense: Float[torch.Tensor, "N B"],
    ) -> Float[torch.Tensor, "A B"]:
        assert x.shape[1] == dense.shape[0]

        x_coo = x.to_sparse_coo()
        coo_idxs: Float[torch.Tensor, "2 K"] = x_coo.indices()
        coo_vals: Float[torch.Tensor, "K"] = x_coo.values()
        ctx.save_for_backward(coo_idxs, coo_vals, dense)
        ctx.dim_n = dense.shape[0]

        dim_k = coo_idxs.shape[1]
        dim_b = dense.shape[1]
        out = torch.zeros(x.shape[0], dim_b, device=dense.device, dtype=coo_vals.dtype)

        BLOCK_K = 128
        BLOCK_B = triton.next_power_of_2(dim_b)
        grid = lambda META: (triton.cdiv(dim_k, META["BLOCK_K"]),)
        _coo_sparse_dense_matmul_fwd_kernel[grid](
            coo_idxs_ptr=coo_idxs,
            coo_vals_ptr=coo_vals,
            dense_ptr=dense,
            out_ptr=out,
            stride_coo_idxs_row=coo_idxs.stride(0),
            stride_coo_idxs_k=coo_idxs.stride(1),
            stride_coo_vals_k=coo_vals.stride(0),
            stride_dense_n=dense.stride(0),
            stride_dense_b=dense.stride(1),
            stride_out_a=out.stride(0),
            stride_out_b=out.stride(1),
            dim_k=dim_k,
            dim_b=dim_b,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
        )
        return out

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx, dy: Float[torch.Tensor, "A B"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        d(XW)/dW: [N B] = X.T: [N A] @ dy [A B]
        d(XW)/dX: [A N] = dy [A B] @ W.T [B N]
        """
        coo_idxs, coo_vals, dense = ctx.saved_tensors
        dim_k = coo_idxs.shape[1]
        dim_b = dy.shape[1]
        dw = torch.zeros(ctx.dim_n, dim_b, device=dy.device, dtype=coo_vals.dtype)

        # Calling `stride()` in strict mode torch compilation is illegal:
        # https://github.com/pytorch/pytorch/issues/115344#issuecomment-1846229103.
        stride_dy_a = dy.shape[-1]
        stride_dy_b = 1
        stride_dw_n = dw.shape[-1]
        stride_dw_b = 1

        BLOCK_K = 128
        BLOCK_B = triton.next_power_of_2(dim_b)
        grid = lambda META: (triton.cdiv(dim_k, META["BLOCK_K"]),)
        _coo_sparse_dense_matmul_bwd_kernel[grid](
            coo_idxs_ptr=coo_idxs,
            coo_vals_ptr=coo_vals,
            dy_ptr=dy,
            dw_ptr=dw,
            stride_coo_idxs_row=coo_idxs.stride(0),
            stride_coo_idxs_k=coo_idxs.stride(1),
            stride_coo_vals_k=coo_vals.stride(0),
            stride_dy_a=stride_dy_a,
            stride_dy_b=stride_dy_b,
            stride_dw_n=stride_dw_n,
            stride_dw_b=stride_dw_b,
            dim_k=dim_k,
            dim_b=dim_b,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
        )

        dx = dy @ dense.transpose(-2, -1)
        return dx, dw


coo_sparse_dense_matmul = CooSparseDenseMatmul.apply


@triton.jit
def _sparse_dense_matmul_fwd_kernel(
    sparse_idxs_ptr,
    sparse_vals_ptr,
    dense_ptr,
    out_ptr,
    stride_sparse_idxs_a: int,
    stride_sparse_idxs_k: int,
    stride_sparse_vals_a: int,
    stride_sparse_vals_k: int,
    dim_k: int,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
) -> None:
    # Grid: (dim_a,) blocks with `dim_k` elements per block.
    pid_a = tl.program_id(0)
    offsets_k = tl.arange(0, BLOCK_K)
    mask = offsets_k < dim_k

    sparse_idxs_ptr += pid_a * stride_sparse_idxs_a
    sparse_vals_ptr += pid_a * stride_sparse_vals_a

    sparse_idxs_ptrs = sparse_idxs_ptr + offsets_k * stride_sparse_idxs_k
    sparse_vals_ptrs = sparse_vals_ptr + offsets_k * stride_sparse_vals_k
    sparse_idxs = tl.load(sparse_idxs_ptrs, mask=mask)
    sparse_vals = tl.load(sparse_vals_ptrs, mask=mask)


class SparseDenseMatmul(autograd.Function):
    """Computes (sparse: [A N] @ dense [N B] = out: [A B])."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        sparse_idxs: Float[torch.Tensor, "A K"],
        sparse_vals: Float[torch.Tensor, "A K"],
        dense: Float[torch.Tensor, "N B"],
    ) -> Float[torch.Tensor, "A B"]:
        assert sparse_idxs.shape == sparse_vals.shape

        dim_a = sparse_idxs.shape[0]
        dim_k = sparse_idxs.shape[1]
        dim_b = dense.shape[1]
        out = torch.empty(dim_a, dim_b, device=dense.device, dtype=sparse_vals.dtype)

        BLOCK_K = triton.next_power_of_2(dim_k)
        _sparse_dense_matmul_fwd_kernel[(dim_a,)](
            sparse_idxs_ptr=sparse_idxs,
            sparse_vals_ptr=sparse_vals,
            dense_ptr=dense,
            out_ptr=out,
            stride_sparse_idxs_a=sparse_idxs.stride(0),
            stride_sparse_idxs_k=sparse_idxs.stride(1),
            stride_sparse_vals_a=sparse_vals.stride(0),
            stride_sparse_vals_k=sparse_vals.stride(1),
            dim_k=dim_k,
            BLOCK_K=BLOCK_K,
        )
        return out

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx, dy: torch.Tensor) -> Any:
        pass


sparse_dense_matmul = SparseDenseMatmul.apply
