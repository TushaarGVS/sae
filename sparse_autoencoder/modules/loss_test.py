from typing import Callable, Any

import torch
from torch import autograd

from sparse_autoencoder.modules.loss import mse_loss
from sparse_autoencoder.modules.test_utils import get_fl_tensor

_factory_kwargs = {"dtype": torch.float16}


def _test_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Callable,
    loss_ref: torch.Tensor,
    loss_fn_name: str | None = None,
    *args: Any,
):
    if loss_fn_name is None:
        loss_fn_name = loss_fn.__name__

    loss = loss_fn(output, target, *args)
    max_abs_err_loss = torch.abs(loss - loss_ref).item()

    dloss = autograd.grad(loss, output, retain_graph=True)[0]
    dloss_ref = autograd.grad(loss_ref, output, retain_graph=True)[0]
    max_abs_err_dloss = torch.max(torch.abs(dloss - dloss_ref)).item()

    print(f"output_numel={output.numel()}, {output.dtype=}, {output.shape=}")
    print(f"{loss.dtype=}, {loss.shape=}")
    print(f"{dloss.dtype=}, {dloss.shape=}")
    print(f"[{loss_fn_name}] {max_abs_err_loss=}, {max_abs_err_dloss=}\n--")


def test_mse_loss() -> None:
    output = get_fl_tensor(torch.Size([8192, 16_384]), factory_kwargs=_factory_kwargs)
    target = get_fl_tensor(torch.Size([8192, 16_384]), factory_kwargs=_factory_kwargs)
    loss_ref = (output.float() - target.float()).pow(2).mean()
    _test_loss(output, target, mse_loss, loss_ref, "mse_loss")


if __name__ == "__main__":
    test_mse_loss()
