from typing import Tuple, Any

import torch
import triton
import triton.language as tl
from torch import autograd
from torch.amp import custom_fwd, custom_bwd

from sparse_autoencoder.modules.utils import contiguous


@triton.jit
def _relu(x: tl.tensor) -> tl.tensor:
    return tl.where(x > 0.0, x, 0.0)


@triton.jit
def _relu_grad(x: tl.tensor) -> tl.tensor:
    return tl.where(x > 0.0, 1.0, 0.0)


@triton.jit
def _relu_fwd_kernel(
    x_ptr,
    y_ptr,
    stride_x_m: int,
    stride_y_m: int,
    x_numel: int,
    BLOCK_M: tl.constexpr,
) -> None:
    # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
    pid_m = tl.program_id(0)
    offsets_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
    # The `mask` is not needed when `BLOCK_M=dim_m` and `dim_m` is a power of 2.
    mask = offsets_m < x_numel

    x_ptrs = x_ptr + offsets_m * stride_x_m
    x = tl.load(x_ptrs, mask=mask)

    y_ptrs = y_ptr + offsets_m * stride_y_m
    y = _relu(x)
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _relu_bwd_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    stride_x_m: int,
    stride_dy_m: int,
    stride_dx_m: int,
    x_numel: int,
    BLOCK_M: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    offsets_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
    mask = offsets_m < x_numel

    x_ptrs = x_ptr + offsets_m * stride_x_m
    dy_ptrs = dy_ptr + offsets_m * stride_dy_m
    dx_ptrs = dx_ptr + offsets_m * stride_dx_m

    x = tl.load(x_ptrs, mask=mask)
    dy = tl.load(dy_ptrs, mask=mask)
    dx = _relu_grad(x) * dy
    tl.store(dx_ptrs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class ReLU(autograd.Function):
    """Computes relu(x)."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)

        x_numel, dim_m = x.numel(), x.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        y = torch.empty_like(x)
        # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
        grid = lambda META: (triton.cdiv(x_numel, META["BLOCK_M"]),)
        _relu_fwd_kernel[grid](
            x_ptr=x,
            y_ptr=y,
            stride_x_m=x.stride(-1),
            stride_y_m=y.stride(-1),
            x_numel=x_numel,
            BLOCK_M=BLOCK_M,
        )
        return y

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, dy: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors

        x_numel, dim_m = x.numel(), x.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        dx = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(x_numel, META["BLOCK_M"]),)
        _relu_bwd_kernel[grid](
            x_ptr=x,
            dy_ptr=dy,
            dx_ptr=dx,
            stride_x_m=x.stride(-1),
            stride_dy_m=dy.stride(-1),
            stride_dx_m=dx.stride(-1),
            x_numel=x_numel,
            BLOCK_M=BLOCK_M,
        )
        return dx


relu = ReLU.apply


@triton.jit
def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


@triton.jit
def _jumprelu(x: tl.tensor, threshold: tl.tensor) -> tl.tensor:
    return x * tl.where(x > threshold, 1.0, 0.0)


@triton.jit
def _jumprelu_grad(
    x: tl.tensor, threshold: tl.tensor, bandwidth: float
) -> Tuple[tl.tensor, tl.tensor]:
    x_grad = tl.where(x > threshold, 1.0, 0.0)
    threshold_grad = -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth)
    return x_grad, threshold_grad


@triton.jit
def _jumprelu_fwd_kernel(
    x_ptr,
    log_threshold_ptr,
    y_ptr,
    stride_x_m: int,
    stride_log_threshold_m: int,
    stride_y_m: int,
    dim_m: int,
    x_numel: int,
    BLOCK_M: tl.constexpr,
) -> None:
    # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
    pid_m = tl.program_id(0)
    offsets_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
    mask = offsets_m < x_numel

    x_ptrs = x_ptr + offsets_m * stride_x_m
    x = tl.load(x_ptrs, mask=mask)

    # Thresholds are one per `dim_m`, hence, circular access is needed.
    log_threshold_offsets_m = (offsets_m * stride_log_threshold_m) % dim_m
    log_threshold_ptrs = log_threshold_ptr + log_threshold_offsets_m
    log_threshold = tl.load(log_threshold_ptrs, mask=mask)
    # Triton `tl.exp()` doesn't work for `float16` or `bfloat16`, see:
    # https://github.com/triton-lang/triton/issues/1516.
    threshold = tl.exp(log_threshold.to(tl.float32))

    y_ptrs = y_ptr + offsets_m * stride_y_m
    y = _jumprelu(x=x, threshold=threshold)
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _jumprelu_bwd_kernel(
    x_ptr,
    log_threshold_ptr,
    dy_ptr,
    dx_ptr,
    dlog_threshold_ptr,
    stride_x_m: int,
    stride_log_threshold_m: int,
    stride_dy_m: int,
    stride_dx_m: int,
    stride_dlog_threshold_m: int,
    bandwidth: float,
    dim_m: int,
    x_numel: int,
    BLOCK_M: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    offsets_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
    mask = offsets_m < x_numel

    x_ptrs = x_ptr + offsets_m * stride_x_m
    dy_ptrs = dy_ptr + offsets_m * stride_dy_m
    x = tl.load(x_ptrs, mask=mask)
    dy = tl.load(dy_ptrs, mask=mask)

    log_threshold_offsets_m = (offsets_m * stride_log_threshold_m) % dim_m
    log_threshold_ptrs = log_threshold_ptr + log_threshold_offsets_m
    log_threshold = tl.load(log_threshold_ptrs, mask=mask)
    threshold = tl.exp(log_threshold.to(tl.float32))

    dx_ptrs = dx_ptr + offsets_m * stride_dx_m
    dlog_threshold_offsets_m = (offsets_m * stride_dlog_threshold_m) % dim_m
    dlog_threshold_ptrs = dlog_threshold_ptr + dlog_threshold_offsets_m
    dx, dthreshold = _jumprelu_grad(x, threshold, bandwidth=bandwidth)
    dx = dx * dy
    dlog_threshold = tl.exp(log_threshold) * dthreshold * dy
    tl.store(dx_ptrs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(
        dlog_threshold_ptrs,
        dlog_threshold.to(dlog_threshold_ptr.dtype.element_ty),
        mask=mask,
    )


class JumpReLU(autograd.Function):
    """Computes jumprelu(x, log_threshold): https://arxiv.org/pdf/2407.14435."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: torch.Tensor, log_threshold: torch.Tensor) -> torch.Tensor:
        x_numel, dim_m = x.numel(), x.shape[-1]
        assert dim_m == log_threshold.shape[0]
        BLOCK_M = triton.next_power_of_2(dim_m)

        ctx.save_for_backward(x, log_threshold)

        y = torch.empty_like(x)
        # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
        grid = lambda META: (triton.cdiv(x_numel, META["BLOCK_M"]),)
        _jumprelu_fwd_kernel[grid](
            x_ptr=x,
            log_threshold_ptr=log_threshold,
            y_ptr=y,
            stride_x_m=x.stride(-1),
            stride_log_threshold_m=log_threshold.stride(-1),
            stride_y_m=y.stride(-1),
            dim_m=dim_m,
            x_numel=x_numel,
            BLOCK_M=BLOCK_M,
        )
        return y

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, log_threshold = ctx.saved_tensors

        bandwidth = 0.001
        x_numel, dim_m = x.numel(), x.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        dx = torch.empty_like(x)
        dlog_threshold = torch.empty_like(log_threshold)
        grid = lambda META: (triton.cdiv(x_numel, META["BLOCK_M"]),)
        _jumprelu_bwd_kernel[grid](
            x_ptr=x,
            log_threshold_ptr=log_threshold,
            dy_ptr=dy,
            dx_ptr=dx,
            dlog_threshold_ptr=dlog_threshold,
            stride_x_m=x.stride(-1),
            stride_log_threshold_m=log_threshold.stride(-1),
            stride_dy_m=dy.stride(-1),
            stride_dx_m=dx.stride(-1),
            stride_dlog_threshold_m=dlog_threshold.stride(-1),
            bandwidth=bandwidth,
            dim_m=dim_m,
            x_numel=x_numel,
            BLOCK_M=BLOCK_M,
        )
        return dx, dlog_threshold


jumprelu = JumpReLU.apply


@torch.compile(fullgraph=True)
def _topk_fwd_kernel(
    x: torch.Tensor, k: int, dim: int, BLOCK_M: tl.constexpr | None = None
) -> torch.Tensor:
    topk_ = x.topk(k=k, dim=dim)

    topk_vals = topk_.values
    act_topk_vals = torch.empty_like(topk_vals)
    # Grid: (topk_vals_numel/dim_m,) blocks, with `dim_m` elements per block.
    if BLOCK_M is None:
        BLOCK_M = triton.next_power_of_2(topk_vals.shape[-1])
    _relu_fwd_kernel[(triton.cdiv(topk_vals.numel(), BLOCK_M),)](
        x_ptr=topk_vals,
        y_ptr=act_topk_vals,
        stride_x_m=topk_vals.stride(-1),
        stride_y_m=act_topk_vals.stride(-1),
        x_numel=topk_vals.numel(),
        BLOCK_M=BLOCK_M,
    )

    y = torch.zeros_like(x)
    y.scatter_(dim=-1, index=topk_.indices, src=act_topk_vals)
    return y


class TopK(autograd.Function):
    """Computes relu(topk(x, k)): https://github.com/openai/sparse_autoencoder."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: torch.Tensor, k: int, dim: int = -1) -> Any:
        assert k <= x.size(dim)
        y = _topk_fwd_kernel(x=x, k=k, dim=dim)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, dy: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (y,) = ctx.saved_tensors

        y_numel, dim_m = y.numel(), y.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        dx = torch.empty_like(y)
        grid = lambda META: (triton.cdiv(y_numel, META["BLOCK_M"]),)
        _relu_bwd_kernel[grid](
            x_ptr=y,
            dy_ptr=dy,
            dx_ptr=dx,
            stride_x_m=y.stride(-1),
            stride_dy_m=dy.stride(-1),
            stride_dx_m=dx.stride(-1),
            x_numel=y_numel,
            BLOCK_M=BLOCK_M,
        )
        return dx, None, None


topk = TopK.apply


class BatchTopK(autograd.Function):
    """
    Computes batched topk(x, k):
    https://www.alignmentforum.org/posts/Nkx6yWZNbAsfvic98/batchtopk-a-simple-improvement-for-topk-saes.
    """

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
        assert k <= x.size(dim)
        batch, seq_len, dim_m = x.shape
        y = _topk_fwd_kernel(
            x=x.view(seq_len, -1),
            k=(k * batch),
            dim=dim,
            BLOCK_M=triton.next_power_of_2(dim_m),
        )
        y = y.view(x.shape)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, dy: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (y,) = ctx.saved_tensors

        y_ = y.reshape(y.shape[1], -1)
        y_numel, dim_m = y_.numel(), y_.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        dx = torch.empty_like(y_)
        grid = lambda META: (triton.cdiv(y_numel, META["BLOCK_M"]),)
        _relu_bwd_kernel[grid](
            x_ptr=y_,
            dy_ptr=dy,
            dx_ptr=dx,
            stride_x_m=y_.stride(-1),
            stride_dy_m=dy.stride(-1),
            stride_dx_m=dx.stride(-1),
            x_numel=y_numel,
            BLOCK_M=BLOCK_M,
        )
        dx = dx.reshape(y.shape)
        return dx, None, None


batchtopk = BatchTopK.apply


@triton.jit
def _prolu(x: tl.tensor, bias: tl.tensor) -> tl.tensor:
    return tl.where(((x + bias) > 0) & (x > 0), x, 0.0)


@triton.jit
def _prolu_fwd_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    stride_x_m: int,
    stride_bias_m: int,
    stride_y_m: int,
    dim_m: int,
    x_numel: int,
    BLOCK_M: tl.constexpr,
) -> None:
    # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
    pid_m = tl.program_id(0)
    offsets_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
    mask = offsets_m < x_numel

    x_ptrs = x_ptr + offsets_m * stride_x_m
    bias_offsets_m = (offsets_m * stride_bias_m) % dim_m
    bias_ptrs = bias_ptr + bias_offsets_m
    x = tl.load(x_ptrs, mask=mask)
    bias = tl.load(bias_ptrs, mask=mask)

    y_ptrs = y_ptr + offsets_m * stride_y_m
    y = _prolu(x=x, bias=bias)
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


class ProLU(autograd.Function):
    """
    Computes prolu(x, b):
    https://www.alignmentforum.org/posts/HEpufTdakGTTKgoYF/prolu-a-nonlinearity-for-sparse-autoencoders.
    """

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x_numel, dim_m = x.numel(), x.shape[-1]
        assert dim_m == bias.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        y = torch.empty_like(x)
        # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
        grid = lambda META: (triton.cdiv(x_numel, META["BLOCK_M"]),)
        _prolu_fwd_kernel[grid](
            x_ptr=x,
            bias_ptr=bias,
            y_ptr=y,
            stride_x_m=x.stride(-1),
            stride_bias_m=bias.stride(-1),
            stride_y_m=y.stride(-1),
            dim_m=dim_m,
            x_numel=x_numel,
            BLOCK_M=BLOCK_M,
        )
        ctx.save_for_backward(y)
        return y

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (y,) = ctx.saved_tensors

        y_numel, dim_m = y.numel(), y.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        dx = torch.empty_like(y)
        grid = lambda META: (triton.cdiv(y_numel, META["BLOCK_M"]),)
        _relu_bwd_kernel[grid](
            x_ptr=y,
            dy_ptr=dy,
            dx_ptr=dx,
            stride_x_m=y.stride(-1),
            stride_dy_m=dy.stride(-1),
            stride_dx_m=dx.stride(-1),
            x_numel=y_numel,
            BLOCK_M=BLOCK_M,
        )
        return dx.clone(), dx.clone()


prolu = ProLU.apply

ACTIVATION_FUNCTIONS = {
    "relu": relu,
    "jumprelu": jumprelu,
    "topk": topk,
    "batchtopk": batchtopk,
    "prolu": prolu,
}
