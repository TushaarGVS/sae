from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import autograd

from sparse_autoencoder.modules.activations import (
    relu,
    jumprelu,
    topk,
    batchtopk,
    prolu,
)
from sparse_autoencoder.modules.test_utils import get_fl_tensor

_factory_kwargs = {"dtype": torch.float16}


def _y_from_topk(
    size: torch.Size | Tuple,
    dtype: torch.dtype,
    device: torch.device,
    topk_idxs: torch.Tensor,
    topk_vals: torch.Tensor,
):
    y = torch.zeros(size, dtype=dtype, device=device)
    y.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
    return y


def _topk_wrapper(
    x: torch.Tensor, k: int, bias: torch.Tensor | None = None
) -> torch.Tensor:
    # Note: Do not change the order of the arguments in the function signature.
    topk_idxs, topk_vals = topk(x, k, bias)
    return _y_from_topk(
        size=x.size(),
        dtype=x.dtype,
        device=x.device,
        topk_idxs=topk_idxs,
        topk_vals=topk_vals,
    )


def _batchtopk_wrapper(
    x: torch.Tensor, k: int, bias: torch.Tensor | None = None
) -> torch.Tensor:
    # Note: Do not change the order of the arguments in the function signature.
    topk_idxs, topk_vals = batchtopk(x, k, bias)
    if len(x.shape) == 2:
        y_size = torch.Size([x.shape[0] * x.shape[1]])
    else:
        y_size = torch.Size([x.shape[1], x.shape[0] * x.shape[2]])
    y = _y_from_topk(
        size=y_size,
        dtype=x.dtype,
        device=x.device,
        topk_idxs=topk_idxs,
        topk_vals=topk_vals,
    )
    return rearrange(y, "... (b m) -> b ... m", m=x.shape[-1]).squeeze(1)


def _test_activation(
    x: torch.Tensor,
    activation_fn: Callable,
    y_ref: torch.Tensor,
    activation_fn_name: str | None = None,
    *args: Any,
) -> None:
    if activation_fn_name is None:
        activation_fn_name = activation_fn.__name__

    # Note: `torch.autograd.Function.apply()` doesn't allow for kwargs.
    y = activation_fn(x, *args)
    max_abs_err_y = torch.max(torch.abs(y - y_ref)).item()

    dy = autograd.grad(torch.sum(y), x, retain_graph=True)[0]
    dy_ref = autograd.grad(torch.sum(y_ref), x, retain_graph=True)[0]
    max_abs_err_dy = torch.max(torch.abs(dy - dy_ref)).item()

    print(f"x_numel={x.numel()}, {x.dtype=}, {x.shape=}")
    print(f"{y.dtype=}, {y.shape=}")
    print(f"[{activation_fn_name}] {max_abs_err_y=}, {max_abs_err_dy=}\n--")


def _test_activation_with_bias(
    x: torch.Tensor,
    bias: torch.Tensor,
    activation_fn: Callable,
    y_ref: torch.Tensor,
    activation_fn_name: str | None = None,
    *args: Any,
) -> None:
    if activation_fn_name is None:
        activation_fn_name = activation_fn.__name__

    y = activation_fn(x, *args, bias)
    max_abs_err_y = torch.max(torch.abs(y - y_ref)).item()

    dx = autograd.grad(torch.sum(y), x, retain_graph=True)[0]
    dx_ref = autograd.grad(torch.sum(y_ref), x, retain_graph=True)[0]
    max_abs_err_dx = torch.max(torch.abs(dx - dx_ref)).item()

    dbias = autograd.grad(torch.sum(y), bias, retain_graph=True)[0]
    dbias_ref = autograd.grad(torch.sum(y_ref), bias, retain_graph=True)[0]
    max_abs_err_dbias = torch.max(torch.abs(dbias - dbias_ref)).item()

    print(f"x_numel={x.numel()}, {x.dtype=}, {x.shape=}, {bias.shape=}")
    print(f"{y.dtype=}, {y.shape=}")
    print(f"{dx.shape=}, {dx_ref.shape=}")
    print(f"{dbias.shape=}, {dbias_ref.shape=}")
    print(
        f"[{activation_fn_name}] "
        f"{max_abs_err_y=}, "
        f"{max_abs_err_dx=}, "
        f"{max_abs_err_dbias=}\n--"
    )


def test_relu() -> None:
    x = get_fl_tensor(torch.Size([64, 32, 4096]), factory_kwargs=_factory_kwargs)
    y_ref = F.relu(x)
    _test_activation(x, relu, y_ref, "relu")

    bias = get_fl_tensor(torch.Size([4096]), factory_kwargs=_factory_kwargs)
    y_ref = F.relu(x + bias)
    _test_activation_with_bias(x, bias, relu, y_ref, "relu+bias")


def test_jumprelu() -> None:
    x = get_fl_tensor(torch.Size([64, 32, 4096]), factory_kwargs=_factory_kwargs)
    log_threshold = get_fl_tensor(torch.Size([4096]))
    y_ref = x * torch.where(x > torch.exp(log_threshold), 1.0, 0.0)
    _test_activation(x, jumprelu, y_ref, "jumprelu", log_threshold)

    bias = get_fl_tensor(torch.Size([4096]), factory_kwargs=_factory_kwargs)
    x_ = x + bias
    y_ref = x_ * torch.where(x_ > torch.exp(log_threshold), 1.0, 0.0)
    _test_activation_with_bias(x, bias, jumprelu, y_ref, "jumprelu+bias", log_threshold)


def test_topk() -> None:
    x = get_fl_tensor(torch.Size([64, 32, 4096]), factory_kwargs=_factory_kwargs)
    k = 500
    topk_ref = x.topk(k=k, dim=-1)
    y_ref = _y_from_topk(
        size=x.size(),
        dtype=x.dtype,
        device=x.device,
        topk_idxs=topk_ref.indices,
        topk_vals=F.relu(topk_ref.values),
    )
    _test_activation(x, _topk_wrapper, y_ref, "topk", k)

    bias = get_fl_tensor(torch.Size([4096]), factory_kwargs=_factory_kwargs)
    topk_ref = (x + bias).topk(k=k, dim=-1)
    y_ref = _y_from_topk(
        size=x.size(),
        dtype=x.dtype,
        device=x.device,
        topk_idxs=topk_ref.indices,
        topk_vals=F.relu(topk_ref.values),
    )
    _test_activation_with_bias(x, bias, _topk_wrapper, y_ref, "topk+bias", k)


def test_batchtopk() -> None:
    x = get_fl_tensor(torch.Size([16, 32, 4096]), factory_kwargs=_factory_kwargs)
    k = 500  # note: torch.topk runs out of memory for large k (= k * batch)
    batch, seq_len, dim_m = x.shape
    batch_k = batch * k
    topk_ref = rearrange(x, "b l d -> l (b d)", d=dim_m).topk(k=batch_k, dim=-1)
    y_ref = _y_from_topk(
        size=torch.Size([seq_len, batch * dim_m]),
        dtype=x.dtype,
        device=x.device,
        topk_idxs=topk_ref.indices,
        topk_vals=F.relu(topk_ref.values),
    )
    y_ref = rearrange(y_ref, "l (b d) -> b l d", d=dim_m)
    _test_activation(x, _batchtopk_wrapper, y_ref, "batchtopk-3d", k)

    bias = get_fl_tensor(torch.Size([4096]), factory_kwargs=_factory_kwargs)
    topk_ref = rearrange(x + bias, "b l d -> l (b d)", d=dim_m).topk(k=batch_k, dim=-1)
    y_ref = _y_from_topk(
        size=torch.Size([seq_len, batch * dim_m]),
        dtype=x.dtype,
        device=x.device,
        topk_idxs=topk_ref.indices,
        topk_vals=F.relu(topk_ref.values),
    )
    y_ref = rearrange(y_ref, "l (b d) -> b l d", d=dim_m)
    _test_activation_with_bias(x, bias, _batchtopk_wrapper, y_ref, "batchtopk+bias", k)

    x = get_fl_tensor(torch.Size([16, 4096]), factory_kwargs=_factory_kwargs)
    k = 500
    batch, dim_m = x.shape
    batch_k = batch * k
    topk_ref = rearrange(x, "b d -> (b d)", d=dim_m).topk(k=batch_k, dim=-1)
    y_ref = _y_from_topk(
        size=torch.Size([batch * dim_m]),
        dtype=x.dtype,
        device=x.device,
        topk_idxs=topk_ref.indices,
        topk_vals=F.relu(topk_ref.values),
    )
    y_ref = rearrange(y_ref, "(b d) -> b d", d=dim_m)
    _test_activation(x, _batchtopk_wrapper, y_ref, "batchtopk-2d", k)


def test_prolu() -> None:
    x = get_fl_tensor(torch.Size([64, 32, 4096 * 16]), factory_kwargs=_factory_kwargs)
    bias = get_fl_tensor(torch.Size([4096 * 16]))
    y_ref = torch.where(((x + bias) > 0) & (x > 0), x, 0.0)
    _test_activation(x, prolu, y_ref, "prolu", bias)


if __name__ == "__main__":
    test_relu()
    test_jumprelu()
    test_topk()
    test_batchtopk()
    test_prolu()
