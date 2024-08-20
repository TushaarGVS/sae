from typing import Tuple, Any

import torch
import triton
import triton.language as tl
from einops import rearrange
from jaxtyping import Float
from torch import autograd
from torch.amp import custom_fwd, custom_bwd

from sparse_autoencoder.modules.utils import contiguous


@triton.jit
def _relu(x: tl.tensor) -> tl.tensor:
    _zero = 0.0
    return tl.where(x > 0.0, x, _zero.to(x.dtype))


@triton.jit
def _relu_grad(x: tl.tensor) -> tl.tensor:
    _zero, _one = 0.0, 1.0
    return tl.where(x > 0.0, _one.to(x.dtype), _zero.to(x.dtype))


@triton.jit
def _relu_fwd_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    stride_x_m: int,
    stride_bias_m: int,
    stride_y_m: int,
    x_numel: int,
    dim_m: int,
    BLOCK_M: tl.constexpr,
) -> None:
    # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
    pid_m = tl.program_id(0)
    offsets_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
    # The `mask` is not needed when `BLOCK_M=dim_m` and `dim_m` is a power of 2.
    mask = offsets_m < x_numel

    x_ptrs = x_ptr + offsets_m * stride_x_m
    x = tl.load(x_ptrs, mask=mask)
    if bias_ptr:
        bias_offsets_m = (offsets_m * stride_bias_m) % dim_m
        bias_ptrs = bias_ptr + bias_offsets_m
        bias = tl.load(bias_ptrs, mask=mask)
        x = x + bias

    y_ptrs = y_ptr + offsets_m * stride_y_m
    y = _relu(x)
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _relu_bwd_kernel(
    y_ptr,
    dy_ptr,
    dx_ptr,
    stride_y_m: int,
    stride_dy_m: int,
    stride_dx_m: int,
    y_numel: int,
    BLOCK_M: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    offsets_m = BLOCK_M * pid_m + tl.arange(0, BLOCK_M)
    mask = offsets_m < y_numel

    y_ptrs = y_ptr + offsets_m * stride_y_m
    dy_ptrs = dy_ptr + offsets_m * stride_dy_m
    dx_ptrs = dx_ptr + offsets_m * stride_dx_m

    y = tl.load(y_ptrs, mask=mask)
    dy = tl.load(dy_ptrs, mask=mask)
    dx = _relu_grad(y) * dy
    tl.store(dx_ptrs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class ReLU(autograd.Function):
    """Computes relu(x + bias)."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: Float[torch.Tensor, "*b m"],
        bias: Float[torch.Tensor, "m"] | None = None,
    ) -> Float[torch.Tensor, "*b m"]:
        x_numel, dim_m = x.numel(), x.shape[-1]
        assert bias is None or dim_m == bias.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        y = torch.empty_like(x)
        # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
        grid = lambda META: (triton.cdiv(x_numel, META["BLOCK_M"]),)
        _relu_fwd_kernel[grid](
            x_ptr=x,
            bias_ptr=bias,
            y_ptr=y,
            stride_x_m=x.stride(-1),
            stride_bias_m=(bias.stride(-1) if bias is not None else 0),
            stride_y_m=y.stride(-1),
            x_numel=x_numel,
            dim_m=dim_m,
            BLOCK_M=BLOCK_M,
        )
        ctx.save_for_backward(y)
        ctx.bias = True if bias is not None else False
        return y

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, dy: Float[torch.Tensor, "*b m"]
    ) -> Tuple[Float[torch.Tensor, "*b m"], Float[torch.Tensor, "m"] | None]:
        (y,) = ctx.saved_tensors

        y_numel, dim_m = y.numel(), y.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        dx: Float[torch.Tensor, "*b m"] = torch.empty_like(y)
        # Calling `stride()` in strict mode torch compilation is illegal:
        # https://github.com/pytorch/pytorch/issues/115344#issuecomment-1846229103.
        stride_dy_m = 1
        stride_dx_m = 1

        grid = lambda META: (triton.cdiv(y_numel, META["BLOCK_M"]),)
        _relu_bwd_kernel[grid](
            y_ptr=y,
            dy_ptr=dy,
            dx_ptr=dx,
            stride_y_m=y.stride(-1),
            stride_dy_m=stride_dy_m,
            stride_dx_m=stride_dx_m,
            y_numel=y_numel,
            BLOCK_M=BLOCK_M,
        )
        dbias = dx.clone().view(-1, dim_m).sum(dim=0) if ctx.bias else None
        return dx, dbias


relu = ReLU.apply


@triton.jit
def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


@triton.jit
def _jumprelu(x: tl.tensor, threshold: tl.tensor) -> tl.tensor:
    _zero, _one = 0.0, 1.0
    return x * tl.where(x > threshold, _one.to(x.dtype), _zero.to(x.dtype))


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
    bias_ptr,
    log_threshold_ptr,
    y_ptr,
    stride_x_m: int,
    stride_bias_m: int,
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
    if bias_ptr:
        bias_offsets_m = (offsets_m * stride_bias_m) % dim_m
        bias_ptrs = bias_ptr + bias_offsets_m
        bias = tl.load(bias_ptrs, mask=mask)
        x = x + bias

    # Thresholds are one per `dim_m`, hence, circular access is needed.
    log_threshold_offsets_m = (offsets_m * stride_log_threshold_m) % dim_m
    log_threshold_ptrs = log_threshold_ptr + log_threshold_offsets_m
    # Triton `tl.exp()` doesn't work for `float16` or `bfloat16`, see:
    # https://github.com/triton-lang/triton/issues/1516.
    log_threshold = tl.load(log_threshold_ptrs, mask=mask).to(tl.float32)
    threshold = tl.exp(log_threshold)

    y_ptrs = y_ptr + offsets_m * stride_y_m
    y = _jumprelu(x=x, threshold=threshold)
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _jumprelu_bwd_kernel(
    x_ptr,
    bias_ptr,
    log_threshold_ptr,
    dy_ptr,
    dx_ptr,
    dlog_threshold_ptr,
    stride_x_m: int,
    stride_bias_m: int,
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
    x = tl.load(x_ptrs, mask=mask)
    if bias_ptr:
        bias_offsets_m = (offsets_m * stride_bias_m) % dim_m
        bias_ptrs = bias_ptr + bias_offsets_m
        bias = tl.load(bias_ptrs, mask=mask)
        x = x + bias
    dy_ptrs = dy_ptr + offsets_m * stride_dy_m
    dy = tl.load(dy_ptrs, mask=mask)

    log_threshold_offsets_m = (offsets_m * stride_log_threshold_m) % dim_m
    log_threshold_ptrs = log_threshold_ptr + log_threshold_offsets_m
    log_threshold = tl.load(log_threshold_ptrs, mask=mask).to(tl.float32)
    threshold = tl.exp(log_threshold)

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
    """Computes jumprelu(x + bias, log_threshold): https://arxiv.org/pdf/2407.14435."""

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: Float[torch.Tensor, "*b m"],
        log_threshold: Float[torch.Tensor, "m"],
        bias: Float[torch.Tensor, "m"] | None = None,
    ) -> Float[torch.Tensor, "*b m"]:
        x_numel, dim_m = x.numel(), x.shape[-1]
        assert dim_m == log_threshold.shape[0]
        BLOCK_M = triton.next_power_of_2(dim_m)

        ctx.save_for_backward(x, log_threshold, bias)

        y = torch.empty_like(x)
        # Grid: (x_numel/dim_m,) blocks, with `dim_m` elements per block.
        grid = lambda META: (triton.cdiv(x_numel, META["BLOCK_M"]),)
        _jumprelu_fwd_kernel[grid](
            x_ptr=x,
            bias_ptr=bias,
            log_threshold_ptr=log_threshold,
            y_ptr=y,
            stride_x_m=x.stride(-1),
            stride_bias_m=bias.stride(-1) if bias is not None else 0,
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
    def backward(ctx: Any, dy: Float[torch.Tensor, "*b m"]) -> Tuple[
        Float[torch.Tensor, "*b m"],
        Float[torch.Tensor, "m"],
        Float[torch.Tensor, "m"] | None,
    ]:
        x, log_threshold, bias = ctx.saved_tensors

        bandwidth = 0.001
        x_numel, dim_m = x.numel(), x.shape[-1]
        BLOCK_M = triton.next_power_of_2(dim_m)

        dx = torch.empty_like(x)
        dlog_threshold = torch.empty_like(log_threshold)
        grid = lambda META: (triton.cdiv(x_numel, META["BLOCK_M"]),)
        _jumprelu_bwd_kernel[grid](
            x_ptr=x,
            bias_ptr=bias,
            log_threshold_ptr=log_threshold,
            dy_ptr=dy,
            dx_ptr=dx,
            dlog_threshold_ptr=dlog_threshold,
            stride_x_m=x.stride(-1),
            stride_bias_m=bias.stride(-1) if bias is not None else 0,
            stride_log_threshold_m=log_threshold.stride(-1),
            stride_dy_m=dy.stride(-1),
            stride_dx_m=dx.stride(-1),
            stride_dlog_threshold_m=dlog_threshold.stride(-1),
            bandwidth=bandwidth,
            dim_m=dim_m,
            x_numel=x_numel,
            BLOCK_M=BLOCK_M,
        )
        dbias = dx.clone().view(-1, dim_m).sum(dim=0) if bias is not None else None
        return dx, dlog_threshold, dbias


jumprelu = JumpReLU.apply


@torch.compile
def _topk_fwd_kernel(
    x: Float[torch.Tensor, "*b m"],
    k: int,
    bias: Float[torch.Tensor, "m"] | None = None,
    BLOCK_K: tl.constexpr | None = None,
) -> Tuple[Float[torch.Tensor, "*b k"], Float[torch.Tensor, "*b k"]]:
    if bias is not None:
        x = x + bias
    topk_ = x.topk(k=k, dim=-1)

    topk_vals = topk_.values
    act_topk_vals = torch.empty_like(topk_vals)
    # Grid: (topk_vals_numel/dim_m,) blocks, with `dim_m` elements per block.
    if BLOCK_K is None:
        BLOCK_K = triton.next_power_of_2(topk_vals.shape[-1])
    grid = lambda META: (triton.cdiv(topk_vals.numel(), META["BLOCK_M"]),)
    _relu_fwd_kernel[grid](
        x_ptr=topk_vals,
        bias_ptr=None,
        y_ptr=act_topk_vals,
        stride_x_m=topk_vals.stride(-1),
        stride_bias_m=0,
        stride_y_m=act_topk_vals.stride(-1),
        x_numel=topk_vals.numel(),
        dim_m=topk_vals.shape[-1],
        BLOCK_M=BLOCK_K,
    )

    # Note: `act_topk_vals` could have zero values from both `topk` and `relu`; this is
    # unlike coo tensors, where zero values are not stored at all.
    return topk_.indices, act_topk_vals


class TopK(autograd.Function):
    """
    Computes relu(topk(x + bias, k)): https://github.com/openai/sparse_autoencoder.
    """

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: Float[torch.Tensor, "*b m"],
        k: int,
        bias: Float[torch.Tensor, "m"] | None = None,
    ) -> Tuple[Float[torch.Tensor, "*b k"], Float[torch.Tensor, "*b k"]]:
        assert k <= x.size(-1)

        topk_idxs, act_topk_vals = _topk_fwd_kernel(x=x, k=k, bias=bias)

        ctx.save_for_backward(topk_idxs, act_topk_vals)
        ctx.bias = True if bias is not None else False
        ctx.dim_m = x.shape[-1]

        return topk_idxs, act_topk_vals

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any,
        dtopk_idxs: Float[torch.Tensor, "*b k"] | None,
        dact_topk_vals: Float[torch.Tensor, "*b k"],
    ) -> Tuple[Float[torch.Tensor, "*b m"], None, Float[torch.Tensor, "m"] | None]:
        topk_idxs, act_topk_vals = ctx.saved_tensors
        dim_m = ctx.dim_m

        batch, dim_k = act_topk_vals.shape[0], act_topk_vals.shape[-1]
        act_topk_vals_numel = act_topk_vals.numel()
        BLOCK_K = triton.next_power_of_2(dim_k)

        topk_dx: Float[torch.Tensor, "*b k"] = torch.empty_like(act_topk_vals)
        grid = lambda META: (triton.cdiv(act_topk_vals_numel, META["BLOCK_M"]),)
        _relu_bwd_kernel[grid](
            y_ptr=act_topk_vals,
            dy_ptr=dact_topk_vals,
            dx_ptr=topk_dx,
            stride_y_m=act_topk_vals.stride(-1),
            stride_dy_m=dact_topk_vals.stride(-1),
            stride_dx_m=topk_dx.stride(-1),
            y_numel=act_topk_vals_numel,
            BLOCK_M=BLOCK_K,
        )

        dx_size = list(act_topk_vals.shape)  # make mutable
        dx_size[-1] = dim_m
        dx: Float[torch.Tensor, "*b m"] = torch.zeros(
            dx_size, dtype=act_topk_vals.dtype, device=dact_topk_vals.device
        )
        dx.scatter_(dim=-1, index=topk_idxs, src=topk_dx)
        dbias = dx.clone().view(-1, dim_m).sum(dim=0) if ctx.bias else None
        return dx, None, dbias


topk = TopK.apply


class BatchTopK(autograd.Function):
    """
    Computes batched topk(x, k):
    https://www.alignmentforum.org/posts/Nkx6yWZNbAsfvic98/batchtopk-a-simple-improvement-for-topk-saes.
    """

    @staticmethod
    @contiguous
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: Float[torch.Tensor, "b l m"],
        k: int,
        bias: Float[torch.Tensor, "m"] | None = None,
    ) -> Tuple[Float[torch.Tensor, "l bk"], Float[torch.Tensor, "l bk"]]:
        assert k <= x.size(-1)
        if len(x.shape) == 2:
            x: Float[torch.Tensor, "b 1 m"] = x[:, None]

        batch, seq_len, dim_m = x.shape
        # Note: Do not use `view()` or `reshape()` to rearrange: you get wrong
        # arrangements depending on the tensor stride.
        topk_idxs, act_topk_vals = _topk_fwd_kernel(
            x=rearrange(x, "b l m -> l (b m)", m=dim_m),
            k=(k * batch),
            bias=(None if bias is None else bias.broadcast_to(batch, dim_m).flatten()),
            BLOCK_K=triton.next_power_of_2(dim_m),
        )

        ctx.save_for_backward(topk_idxs, act_topk_vals)
        ctx.bias = True if bias is not None else False
        ctx.batch = batch
        ctx.dim_m = dim_m

        return topk_idxs.squeeze(0), act_topk_vals.squeeze(0)

    @staticmethod
    @contiguous
    @custom_bwd(device_type="cuda")
    def backward(
        ctx: Any,
        dtopk_idxs: Float[torch.Tensor, "l bk"] | None,
        dact_topk_vals: Float[torch.Tensor, "l bk"],
    ) -> Tuple[Float[torch.Tensor, "b l m"], None, Float[torch.Tensor, "m"] | None]:
        topk_idxs, act_topk_vals = ctx.saved_tensors
        if len(act_topk_vals.shape) == 1:
            topk_idxs: Float[torch.Tensor, "1 bk"] = topk_idxs[None]
            act_topk_vals: Float[torch.Tensor, "1 bk"] = act_topk_vals[None]
        batch = ctx.batch
        dim_m = ctx.dim_m

        seq_len, dim_batch_k = act_topk_vals.shape[0], act_topk_vals.shape[-1]
        act_topk_vals_numel = act_topk_vals.numel()
        BLOCK_BK = triton.next_power_of_2(dim_batch_k)

        topk_dx: Float[torch.Tensor, "l bk"] = torch.empty_like(act_topk_vals)
        grid = lambda META: (triton.cdiv(act_topk_vals_numel, META["BLOCK_M"]),)
        _relu_bwd_kernel[grid](
            y_ptr=act_topk_vals,
            dy_ptr=dact_topk_vals,
            dx_ptr=topk_dx,
            stride_y_m=act_topk_vals.stride(-1),
            stride_dy_m=dact_topk_vals.stride(-1),
            stride_dx_m=topk_dx.stride(-1),
            y_numel=act_topk_vals_numel,
            BLOCK_M=BLOCK_BK,
        )

        dx: Float[torch.Tensor, "l bm"] = torch.zeros(
            seq_len,
            batch * dim_m,
            dtype=act_topk_vals.dtype,
            device=dact_topk_vals.device,
        )
        dx.scatter_(dim=-1, index=topk_idxs, src=topk_dx)
        dx: Float[torch.Tensor, "b l m"] = rearrange(dx, "l (b m) -> b l m", m=dim_m)
        dbias = dx.clone().contiguous().view(-1, dim_m).sum(dim=0) if ctx.bias else None
        return dx.squeeze(1), None, dbias


batchtopk = BatchTopK.apply


@triton.jit
def _prolu(x: tl.tensor, bias: tl.tensor) -> tl.tensor:
    _zero = 0.0
    return tl.where(((x + bias) > 0) & (x > 0), x, _zero.to(x.dtype))


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
    def forward(
        ctx: Any, x: Float[torch.Tensor, "*b m"], bias: Float[torch.Tensor, "m"]
    ) -> Float[torch.Tensor, "*b m"]:
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
            y_ptr=y,
            dy_ptr=dy,
            dx_ptr=dx,
            stride_y_m=y.stride(-1),
            stride_dy_m=dy.stride(-1),
            stride_dx_m=dx.stride(-1),
            y_numel=y_numel,
            BLOCK_M=BLOCK_M,
        )
        dbias = dx.clone().sum(dim=0)
        return dx, dbias


prolu = ProLU.apply
