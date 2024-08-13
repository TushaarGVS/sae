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
    for k in range(BLOCK_K):
        if k + pid_k * BLOCK_K < dim_k:
            coo_idxs_row1_idx = _get_elem(coo_idxs_row1, k, coo_idxs_row1.dtype)
            if coo_idxs_row1_idx != last_coo_idxs_row1_idx:
                out_ptrs = (
                    out_ptr
                    + last_coo_idxs_row1_idx * stride_out_a
                    + offsets_b * stride_out_b
                )
                tl.atomic_add(out_ptrs, accum.to(out_ptr.dtype.element_ty), mask=mask_b)
                accum *= 0  # reset
                last_coo_idxs_row1_idx = coo_idxs_row1_idx

            coo_val = _get_elem(coo_vals, k, coo_vals.dtype)
            # The following check is not needed for strict coo-sparse tensors, but is
            # needed when converting sparse indices and values from topk.
            if coo_val != 0:
                coo_idxs_row2_idx = _get_elem(coo_idxs_row2, k, coo_idxs_row2.dtype)
                dense_ptrs = (
                    dense_ptr
                    + coo_idxs_row2_idx * stride_dense_n
                    + offsets_b * stride_dense_b
                )
                dense_row = tl.load(dense_ptrs, mask=mask_b)
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
    ddense_ptr,
    stride_coo_idxs_row: int,
    stride_coo_idxs_k: int,
    stride_coo_vals_k: int,
    stride_dy_a: int,
    stride_dy_b: int,
    stride_ddense_n: int,
    stride_ddense_b: int,
    dim_k: int,
    dim_b: int,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
) -> None:
    """Computes coo_sparse.T @ dense."""
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
    for k in range(BLOCK_K):
        if k + pid_k * BLOCK_K < dim_k:
            coo_idxs_row2_idx = _get_elem(coo_idxs_row2, k, coo_idxs_row2.dtype)
            if coo_idxs_row2_idx != last_coo_idxs_row2_idx:
                ddense_ptrs = (
                    ddense_ptr
                    + last_coo_idxs_row2_idx * stride_ddense_n
                    + offsets_b * stride_ddense_b
                )
                tl.atomic_add(
                    ddense_ptrs, accum.to(ddense_ptr.dtype.element_ty), mask=mask_b
                )
                accum *= 0  # reset
                last_coo_idxs_row2_idx = coo_idxs_row2_idx

            coo_val = _get_elem(coo_vals, k, coo_vals.dtype)
            if coo_val != 0:
                coo_idxs_row1_idx = _get_elem(coo_idxs_row1, k, coo_idxs_row1.dtype)
                dy_ptrs = (
                    dy_ptr + coo_idxs_row1_idx * stride_dy_a + offsets_b * stride_dy_b
                )
                dy_row = tl.load(dy_ptrs, mask=mask_b)
                accum += coo_val * dy_row

    ddense_ptrs = (
        ddense_ptr
        + last_coo_idxs_row2_idx * stride_ddense_n
        + offsets_b * stride_ddense_b
    )
    tl.atomic_add(ddense_ptrs, accum.to(ddense_ptr.dtype.element_ty), mask=mask_b)


class CooSparseDenseMatmul(autograd.Function):
    """Computes (x: [A N] @ dense: [N B] = out: [A B])."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
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
        ctx: Any, dy: Float[torch.Tensor, "A B"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        d(XW)/dW: [N B] = X.T: [N A] @ dy [A B]
        d(XW)/dX: [A N] = dy [A B] @ W.T [B N]
        """
        coo_idxs, coo_vals, dense = ctx.saved_tensors
        dim_k = coo_idxs.shape[1]
        dim_b = dy.shape[1]
        ddense = torch.zeros(ctx.dim_n, dim_b, device=dy.device, dtype=coo_vals.dtype)

        # Calling `stride()` in strict mode torch compilation is illegal:
        # https://github.com/pytorch/pytorch/issues/115344#issuecomment-1846229103.
        stride_dy_a = dy.shape[-1]
        stride_dy_b = 1
        stride_ddense_n = ddense.shape[-1]
        stride_ddense_b = 1

        BLOCK_K = 128
        BLOCK_B = triton.next_power_of_2(dim_b)
        grid = lambda META: (triton.cdiv(dim_k, META["BLOCK_K"]),)
        _coo_sparse_dense_matmul_bwd_kernel[grid](
            coo_idxs_ptr=coo_idxs,
            coo_vals_ptr=coo_vals,
            dy_ptr=dy,
            ddense_ptr=ddense,
            stride_coo_idxs_row=coo_idxs.stride(0),
            stride_coo_idxs_k=coo_idxs.stride(1),
            stride_coo_vals_k=coo_vals.stride(0),
            stride_dy_a=stride_dy_a,
            stride_dy_b=stride_dy_b,
            stride_ddense_n=stride_ddense_n,
            stride_ddense_b=stride_ddense_b,
            dim_k=dim_k,
            dim_b=dim_b,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
        )

        dx = dy @ dense.transpose(-2, -1)
        return dx, ddense


coo_sparse_dense_matmul = CooSparseDenseMatmul.apply


@triton.jit
def _dense_transpose_sparse_matmul_fwd_kernel(
    dense_ptr,
    sparse_idxs_ptr,
    sparse_vals_ptr,
    out_ptr,
    stride_dense_n: int,
    stride_dense_a: int,
    stride_sparse_idxs_n: int,
    stride_sparse_idxs_k: int,
    stride_sparse_vals_n: int,
    stride_sparse_vals_k: int,
    stride_out_a: int,
    stride_out_b: int,
    dim_n: int,
    dim_a: int,
    BLOCK_N: tl.constexpr,
    BLOCK_A: tl.constexpr,
) -> None:
    """Computes dense.T @ sparse."""
    pid_k = tl.program_id(0)

    offsets_n = tl.arange(0, BLOCK_N)
    mask_n = offsets_n < dim_n

    sparse_idxs_ptrs = (
        sparse_idxs_ptr
        + pid_k * stride_sparse_idxs_k
        + offsets_n * stride_sparse_idxs_n
    )
    sparse_vals_ptrs = (
        sparse_vals_ptr
        + pid_k * stride_sparse_vals_k
        + offsets_n * stride_sparse_vals_n
    )
    sparse_idxs = tl.load(sparse_idxs_ptrs, mask=mask_n)
    sparse_vals = tl.load(sparse_vals_ptrs, mask=mask_n)

    offsets_a = tl.arange(0, BLOCK_A)
    mask_a = offsets_a < dim_a

    for n in range(dim_n):
        sparse_val = _get_elem(sparse_vals, n, sparse_vals.dtype)
        if sparse_val != 0:
            dense_ptrs = dense_ptr + n * stride_dense_n + offsets_a * stride_dense_a
            dense_row = tl.load(dense_ptrs, mask=mask_a)

            sparse_idx = _get_elem(sparse_idxs, n, sparse_idxs.dtype)
            out_ptrs = out_ptr + sparse_idx * stride_out_b + offsets_a * stride_out_a
            tl.atomic_add(
                out_ptrs,
                (sparse_val * dense_row).to(out_ptr.dtype.element_ty),
                mask=mask_a,
            )


class DenseTransposeSparseMatmul(autograd.Function):
    """Computes (dense.T: [A N] @ sparse [N B] = out: [A B])."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        dense: Float[torch.Tensor, "N A"],
        sparse_idxs: Float[torch.Tensor, "N K"],
        sparse_vals: Float[torch.Tensor, "N K"],
        dim_b: int,
    ) -> Float[torch.Tensor, "A B"]:
        assert sparse_idxs.shape == sparse_vals.shape
        assert dense.shape[0] == sparse_vals.shape[0]

        dim_n, dim_k = sparse_idxs.shape
        dim_a = dense.shape[1]
        out = torch.zeros(dim_a, dim_b, device=dense.device, dtype=sparse_vals.dtype)

        # `out` columns are processed in parallel.
        BLOCK_N = triton.next_power_of_2(dim_n)
        BLOCK_A = triton.next_power_of_2(dim_a)
        _dense_transpose_sparse_matmul_fwd_kernel[(dim_k,)](
            dense_ptr=dense,
            sparse_idxs_ptr=sparse_idxs,
            sparse_vals_ptr=sparse_vals,
            out_ptr=out,
            stride_dense_n=dense.stride(0),
            stride_dense_a=dense.stride(1),
            stride_sparse_idxs_n=sparse_idxs.stride(0),
            stride_sparse_idxs_k=sparse_idxs.stride(1),
            stride_sparse_vals_n=sparse_vals.stride(0),
            stride_sparse_vals_k=sparse_vals.stride(1),
            stride_out_a=out.stride(0),
            stride_out_b=out.stride(1),
            dim_n=dim_n,
            dim_a=dim_a,
            BLOCK_N=BLOCK_N,
            BLOCK_A=BLOCK_A,
        )
        return out

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, dy: torch.Tensor) -> Any:
        pass


dense_transpose_sparse_matmul = DenseTransposeSparseMatmul.apply
