from typing import Dict, Any, Callable

import torch
import torch.nn.functional as F
from torch import autograd

from sparse_autoencoder.modules.activations import (
    relu,
    jumprelu,
    topk,
    batchtopk,
    prolu,
)


def _get_fl_tensor(size: torch.Size, factory_kwargs: Dict[str, Any] | None = None):
    default_factory_kwargs = {
        "dtype": torch.float32,
        "device": torch.device("cuda"),
        "requires_grad": True,
    }
    if factory_kwargs is not None:
        default_factory_kwargs.update(factory_kwargs)
    return torch.randn(size, **default_factory_kwargs)


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
    print(f"[{activation_fn_name}] {max_abs_err_y=} {max_abs_err_dy=}\n--")


def test_relu() -> None:
    x = _get_fl_tensor(torch.Size([64, 32, 4096]))
    y_ref = F.relu(x)
    _test_activation(x, relu, y_ref, "relu")


def test_jumprelu() -> None:
    x = _get_fl_tensor(torch.Size([64, 32, 4096]))
    log_threshold = _get_fl_tensor(torch.Size([4096]))
    y_ref = x * torch.where(x > torch.exp(log_threshold), 1.0, 0.0)
    _test_activation(x, jumprelu, y_ref, "jumprelu", log_threshold)


def test_topk() -> None:
    x = _get_fl_tensor(torch.Size([64, 32, 4096]))
    k = 500
    dim = -1
    topk_ref = x.topk(k=k, dim=dim)
    y_ref = torch.zeros_like(x)
    y_ref.scatter_(dim=dim, index=topk_ref.indices, src=F.relu(topk_ref.values))
    _test_activation(x, topk, y_ref, "topk", k, dim)


def test_batchtopk() -> None:
    x = _get_fl_tensor(torch.Size([16, 32, 4096]))
    k = 500  # note: torch.topk runs out of memory for large k (= k * batch)
    dim = -1
    topk_ref = x.reshape(x.shape[1], -1).topk(k=(k * x.shape[0]), dim=dim)
    y_ref = torch.zeros_like(x.reshape(x.shape[1], -1))
    y_ref.scatter_(dim=dim, index=topk_ref.indices, src=F.relu(topk_ref.values))
    y_ref = y_ref.reshape(x.shape)
    _test_activation(x, batchtopk, y_ref, "batchtopk", k, dim)


def test_prolu() -> None:
    x = _get_fl_tensor(torch.Size([64, 32, 4096 * 16]))
    bias = _get_fl_tensor(torch.Size([4096 * 16]))
    y_ref = torch.where(((x + bias) > 0) & (x > 0), x, 0.0)
    _test_activation(x, prolu, y_ref, "prolu", bias)


if __name__ == "__main__":
    test_relu()
    test_jumprelu()
    test_topk()
    test_batchtopk()
    test_prolu()
