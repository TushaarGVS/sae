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
def _insert_at_idx(
    numel: tl.int64, idx: tl.int64, value: tl.float32, dtype: tl.dtype
) -> None:
    """Returns a tensor of `numel` zeros with `tensor[idx] = value`."""
    return tl.where(tl.arange(0, numel) == idx, value, tl.zeros((numel,), dtype=dtype))


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
def _sparse_dense_matmul_fwd_kernel(
    sparse_idxs_ptr,
    sparse_vals_ptr,
    dense_ptr,
    bias_ptr,
    out_ptr,
    stride_sparse_idxs_a: int,
    stride_sparse_idxs_k: int,
    stride_sparse_vals_a: int,
    stride_sparse_vals_k: int,
    stride_dense_n: int,
    stride_dense_b: int,
    stride_bias_b: int,
    stride_out_a: int,
    stride_out_b: int,
    dim_k: int,
    dim_b: int,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Computes sparse: [A N] @ dense: [N B] + bias: [B] = [A B]."""
    pid_a = tl.program_id(0)

    offsets_k = tl.arange(0, BLOCK_K)
    mask_k = offsets_k < dim_k

    sparse_idxs_ptrs = (
        sparse_idxs_ptr
        + pid_a * stride_sparse_idxs_a
        + offsets_k * stride_sparse_idxs_k
    )
    sparse_vals_ptrs = (
        sparse_vals_ptr
        + pid_a * stride_sparse_vals_a
        + offsets_k * stride_sparse_vals_k
    )
    sparse_idxs = tl.load(sparse_idxs_ptrs, mask=mask_k)
    sparse_vals = tl.load(sparse_vals_ptrs, mask=mask_k)

    offsets_b = tl.arange(0, BLOCK_B)
    mask_b = offsets_b < dim_b

    accum = tl.zeros((BLOCK_B,), dtype=tl.float32)
    for k in range(dim_k):
        sparse_val = _get_elem(sparse_vals, k, dtype=sparse_vals.dtype)
        if sparse_val != 0:
            sparse_idx = _get_elem(sparse_idxs, k, dtype=sparse_idxs.dtype)
            dense_ptrs = (
                dense_ptr + sparse_idx * stride_dense_n + offsets_b * stride_dense_b
            )
            dense_row = tl.load(dense_ptrs, mask=mask_b)
            accum += sparse_val * dense_row

    if bias_ptr:
        bias_ptrs = bias_ptr + offsets_b * stride_bias_b
        bias = tl.load(bias_ptrs, mask=mask_b)
        accum = accum + bias

    out_ptrs = out_ptr + pid_a * stride_out_a + offsets_b * stride_out_b
    tl.store(out_ptrs, accum.to(out_ptr.dtype.element_ty), mask=mask_b)


@triton.jit
def _dense_dense_sparseout_matmul(
    dense1_ptr,
    dense2_ptr,
    at_idxs_ptr,
    out_ptr,
    stride_dense1_a: int,
    stride_dense1_n: int,
    stride_dense2_n: int,
    stride_dense2_b: int,
    stride_at_idxs_a: int,
    stride_at_idxs_k: int,
    stride_out_a: int,
    stride_out_k: int,
    dim_k: int,
    dim_n: int,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Computes (dense1: [A N] @ dense2: [N B]).gather(1, at_idxs: [A K]) = [A K]."""
    pid_a = tl.program_id(0)

    offsets_k = tl.arange(0, BLOCK_K)
    mask_k = offsets_k < dim_k
    at_idxs_ptrs = at_idxs_ptr + pid_a * stride_at_idxs_a + offsets_k * stride_at_idxs_k
    at_idxs = tl.load(at_idxs_ptrs, mask=mask_k)

    offsets_n = tl.arange(0, BLOCK_N)
    mask_n = offsets_n < dim_n
    dense1_ptrs = dense1_ptr + pid_a * stride_dense1_a + offsets_n * stride_dense1_n
    dense1_row = tl.load(dense1_ptrs, mask=mask_n)

    accum = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(dim_k):  # inefficient for large `dim_k`
        at_idx = _get_elem(at_idxs, k, at_idxs.dtype)
        dense2_ptrs = (
            dense2_ptr + at_idx * stride_dense2_b + offsets_n * stride_dense2_n
        )
        dense2_col = tl.load(dense2_ptrs, mask=mask_n)
        accum += _insert_at_idx(BLOCK_K, k, tl.sum(dense1_row * dense2_col), tl.int64)

    out_ptrs = out_ptr + pid_a * stride_out_a + offsets_k * stride_out_k
    tl.store(out_ptrs, accum.to(out_ptr.dtype.element_ty), mask=mask_k)


@triton.jit
def _sparse_transpose_dense_matmul(
    sparse_idxs_ptr,
    sparse_vals_ptr,
    dense_ptr,
    out_ptr,
    stride_sparse_idxs_n: int,
    stride_sparse_idxs_k: int,
    stride_sparse_vals_n: int,
    stride_sparse_vals_k: int,
    stride_dense_n: int,
    stride_dense_b: int,
    stride_out_a: int,
    stride_out_b: int,
    dim_k: int,
    dim_b: int,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """
    Computes (sparse.T: [A N] @ dense: [N B] = [A B]).
    - sparse_idxs, sparse_vals: [N K]
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offsets_k = tl.arange(0, BLOCK_K)
    mask_k = offsets_k < dim_k
    sparse_idxs_ptrs = (
        sparse_idxs_ptr
        + pid_n * stride_sparse_idxs_n
        + offsets_k * stride_sparse_idxs_k
    )
    sparse_vals_ptrs = (
        sparse_vals_ptr
        + pid_n * stride_sparse_vals_n
        + offsets_k * stride_sparse_vals_k
    )
    sparse_idxs = tl.load(sparse_idxs_ptrs, mask=mask_k)
    sparse_vals = tl.load(sparse_vals_ptrs, mask=mask_k)

    offsets_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offsets_b < dim_b
    dense_ptrs = dense_ptr + pid_n * stride_dense_n + offsets_b * stride_dense_b
    dense_row = tl.load(dense_ptrs, mask=mask_b)

    # 2D block pointers of `sparse_idxs` rows and all `dim_b` columns.
    out_ptrs = (
        out_ptr + sparse_idxs[:, None] * stride_out_a + offsets_b[None] * stride_out_b
    )
    accum = sparse_vals[:, None] * dense_row[None]
    tl.atomic_add(out_ptrs, accum.to(out_ptr.dtype.element_ty), mask=mask_b[None])


class SparseDenseMatmul(autograd.Function):
    """Computes (sparse: [A N] @ dense: [N B] + bias: [B] = [A B])."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        sparse_idxs: Float[torch.Tensor, "A K"],
        sparse_vals: Float[torch.Tensor, "A K"],
        dense: Float[torch.Tensor, "N B"],
        bias: Float[torch.Tensor, "B"] | None = None,
    ) -> Float[torch.Tensor, "A B"]:
        assert sparse_idxs.shape == sparse_vals.shape
        assert bias is None or bias.shape[-1] == dense.shape[1]
        ctx.save_for_backward(sparse_idxs, sparse_vals, dense)
        ctx.bias = True if bias is not None else False

        dim_a, dim_k = sparse_vals.shape
        dim_b = dense.shape[-1]
        out = torch.zeros(dim_a, dim_b, device=dense.device, dtype=sparse_vals.dtype)

        BLOCK_K = triton.next_power_of_2(dim_k)
        BLOCK_B = triton.next_power_of_2(dim_b)
        _sparse_dense_matmul_fwd_kernel[(dim_a,)](
            sparse_idxs_ptr=sparse_idxs,
            sparse_vals_ptr=sparse_vals,
            dense_ptr=dense,
            bias_ptr=bias,
            out_ptr=out,
            stride_sparse_idxs_a=sparse_idxs.stride(0),
            stride_sparse_idxs_k=sparse_idxs.stride(1),
            stride_sparse_vals_a=sparse_vals.stride(0),
            stride_sparse_vals_k=sparse_vals.stride(1),
            stride_dense_n=dense.stride(0),
            stride_dense_b=dense.stride(1),
            stride_bias_b=bias.stride(0) if bias is not None else 0,
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
    def backward(ctx: Any, dy: Float[torch.Tensor, "A B"]) -> Tuple[
        None,
        Float[torch.Tensor, "A K"],
        Float[torch.Tensor, "N B"],
        Float[torch.Tensor, "B"] | None,
    ]:
        """
        d(sparse @ dense + bias)/dsparse = dy @ dense.T
        d(sparse @ dense + bias)/ddense = sparse.T @ dy
        sparse @ dense + bias)/dbias = dy.sum(0)
        """
        sparse_idxs, sparse_vals, dense = ctx.saved_tensors
        dbias = dy.sum(0) if ctx.bias else None
        dim_a, dim_b = dy.shape
        dim_k = sparse_vals.shape[1]
        dim_n = dense.shape[0]

        if dim_k > 512:
            # Note: naive implementation is faster for large `dim_k`.
            dsparse: Float[torch.Tensor, "A K"] = (dy @ dense.T).gather(1, sparse_idxs)
        else:
            dense_tr = dense.T.contiguous()
            dsparse = torch.zeros(dim_a, dim_k, device=dy.device, dtype=dy.dtype)
            BLOCK_K = triton.next_power_of_2(dim_k)
            BLOCK_B = triton.next_power_of_2(dim_b)
            _dense_dense_sparseout_matmul[(dim_a,)](
                dense1_ptr=dy,
                dense2_ptr=dense_tr,
                at_idxs_ptr=sparse_idxs,
                out_ptr=dsparse,
                stride_dense1_a=dy.stride(0),
                stride_dense1_n=dy.stride(1),
                stride_dense2_n=dense_tr.stride(0),
                stride_dense2_b=dense_tr.stride(1),
                stride_at_idxs_a=sparse_idxs.stride(0),
                stride_at_idxs_k=sparse_idxs.stride(1),
                stride_out_a=dsparse.stride(0),
                stride_out_k=dsparse.stride(1),
                dim_k=dim_k,
                dim_n=dim_b,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_B,
            )

        ddense = torch.zeros(dim_n, dim_b, device=dy.device, dtype=dy.dtype)
        BLOCK_K = triton.next_power_of_2(dim_k)
        # Note: when BLOCK_K > 256, dim = 4096, Triton numel limits are exceeded.
        BLOCK_B = 512
        grid = lambda META: (dim_a, triton.cdiv(dim_b, META["BLOCK_B"]))
        _sparse_transpose_dense_matmul[grid](
            sparse_idxs_ptr=sparse_idxs,
            sparse_vals_ptr=sparse_vals,
            dense_ptr=dy,
            out_ptr=ddense,
            stride_sparse_idxs_n=sparse_idxs.stride(0),
            stride_sparse_idxs_k=sparse_idxs.stride(1),
            stride_sparse_vals_n=sparse_vals.stride(0),
            stride_sparse_vals_k=sparse_vals.stride(1),
            stride_dense_n=dy.stride(0),
            stride_dense_b=dy.stride(1),
            stride_out_a=ddense.stride(0),
            stride_out_b=ddense.stride(1),
            dim_k=dim_k,
            dim_b=dim_b,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
        )

        return None, dsparse, ddense, dbias


sparse_dense_matmul = SparseDenseMatmul.apply


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
    """Computes dense.T: [A N] @ sparse: [N B] = [A B]."""
    pid_k = tl.program_id(0)

    offsets_a = tl.arange(0, BLOCK_A)
    mask_a = offsets_a < dim_a

    for block_n in range(0, dim_n, BLOCK_N):
        offsets_n = block_n + tl.arange(0, BLOCK_N)
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

        for n in range(block_n, block_n + BLOCK_N):
            if n < dim_n:
                # Note: `sparse_val` and `sparse_idx` use relative `n`, `dense_ptrs`
                # uses absolute `n`.
                sparse_val = _get_elem(sparse_vals, n - block_n, sparse_vals.dtype)
                if sparse_val != 0:
                    dense_ptrs = (
                        dense_ptr + n * stride_dense_n + offsets_a * stride_dense_a
                    )
                    dense_row = tl.load(dense_ptrs, mask=mask_a)

                    sparse_idx = _get_elem(sparse_idxs, n - block_n, sparse_idxs.dtype)
                    out_ptrs = (
                        out_ptr + sparse_idx * stride_out_b + offsets_a * stride_out_a
                    )
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
        ctx.save_for_backward(dense, sparse_idxs, sparse_vals)

        dim_n, dim_k = sparse_idxs.shape
        dim_a = dense.shape[1]
        out = torch.zeros(dim_a, dim_b, device=dense.device, dtype=sparse_vals.dtype)

        # `out` columns are processed in parallel.
        BLOCK_N = min(triton.next_power_of_2(dim_n), 2)
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
    def backward(
        ctx: Any, dy: Float[torch.Tensor, "A B"]
    ) -> Tuple[Float[torch.Tensor, "N B"], None, Float[torch.Tensor, "N K"], None]:
        """
        d(dense.T @ sparse)/ddense = sparse: [N B] @ dy.T: [B A] = [N A]
        d(dense.T @ sparse)/dsparse = dense: [N A] @ dy: [A B] = [N B]
        """
        dense, sparse_idxs, sparse_vals = ctx.saved_tensors
        dim_n, dim_a = dense.shape
        dim_k = sparse_idxs.shape[1]

        ddense = torch.zeros(dim_n, dim_a, device=dy.device, dtype=sparse_vals.dtype)
        dy_tr = dy.T.contiguous()  # [B A]
        BLOCK_K = triton.next_power_of_2(dim_k)
        BLOCK_A = triton.next_power_of_2(dim_a)
        _sparse_dense_matmul_fwd_kernel[(dim_n,)](
            sparse_idxs_ptr=sparse_idxs,
            sparse_vals_ptr=sparse_vals,
            dense_ptr=dy_tr,
            bias_ptr=None,
            out_ptr=ddense,
            stride_sparse_idxs_a=sparse_idxs.stride(0),
            stride_sparse_idxs_k=sparse_idxs.stride(1),
            stride_sparse_vals_a=sparse_vals.stride(0),
            stride_sparse_vals_k=sparse_vals.stride(1),
            stride_dense_n=dy_tr.stride(0),
            stride_dense_b=dy_tr.stride(1),
            stride_bias_b=0,
            stride_out_a=ddense.stride(0),
            stride_out_b=ddense.stride(1),
            dim_k=dim_k,
            dim_b=dim_a,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_A,
        )

        if dim_k > 512:
            dsparse: Float[torch.Tensor, "N K"] = (dense @ dy).gather(1, sparse_idxs)
        else:
            dsparse = torch.zeros(dim_n, dim_k, device=dy.device, dtype=dense.dtype)
            BLOCK_K = triton.next_power_of_2(dim_k)
            BLOCK_A = triton.next_power_of_2(dim_a)
            _dense_dense_sparseout_matmul[(dim_n,)](
                dense1_ptr=dense,
                dense2_ptr=dy,
                at_idxs_ptr=sparse_idxs,
                out_ptr=dsparse,
                stride_dense1_a=dense.stride(0),
                stride_dense1_n=dense.stride(1),
                stride_dense2_n=dy.stride(0),
                stride_dense2_b=dy.stride(1),
                stride_at_idxs_a=sparse_idxs.stride(0),
                stride_at_idxs_k=sparse_idxs.stride(1),
                stride_out_a=dsparse.stride(0),
                stride_out_k=dsparse.stride(1),
                dim_k=dim_k,
                dim_n=dim_a,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_A,
            )

        return ddense, None, dsparse, None


dense_transpose_sparse_matmul = DenseTransposeSparseMatmul.apply
