from typing import Callable, Any

import torch
from torch import autograd

from sparse_autoencoder.modules.sparse_matmul import coo_sparse_dense_matmul
from sparse_autoencoder.modules.test_utils import get_fl_tensor


def _test_matmul(
    x: torch.Tensor,
    dense: torch.Tensor,
    matmul_fn: Callable,
    y_ref: torch.Tensor,
    matmul_fn_name: str | None = None,
    *args: Any,
):
    if matmul_fn_name is None:
        matmul_fn_name = matmul_fn.__name__

    y = matmul_fn(x, dense, *args)
    max_abs_err_y = torch.max(torch.abs(y - y_ref)).item()

    dydw = autograd.grad(torch.sum(y), dense, retain_graph=True)[0]
    dydw_ref = autograd.grad(torch.sum(y_ref), dense, retain_graph=True)[0]
    max_abs_err_dydw = torch.max(torch.abs(dydw - dydw_ref)).item()

    dydx = autograd.grad(torch.sum(y), x, retain_graph=True)[0]
    dydx_ref = autograd.grad(torch.sum(y_ref), x, retain_graph=True)[0]
    max_abs_err_dydx = torch.max(torch.abs(dydx - dydx_ref)).item()

    print(f"x_nonzero={x.count_nonzero().item()}, {x.dtype=}, {x.shape=}")
    print(f"dense_numel={dense.numel()}, {dense.dtype=}, {dense.shape=}")
    print(
        f"[{matmul_fn_name}] "
        f"{max_abs_err_y=}, "
        f"{max_abs_err_dydw=}, "
        f"{max_abs_err_dydx=}\n"
        f"--"
    )


def test_coo_sparse_dense_matmul() -> None:
    x = get_fl_tensor(torch.Size([4096, 8192]), sparsity=0.0)
    dense = get_fl_tensor(torch.Size([8192, 2560]))
    y_ref = x @ dense
    _test_matmul(x, dense, coo_sparse_dense_matmul, y_ref, "coo_sparse_dense_matmul")


if __name__ == "__main__":
    test_coo_sparse_dense_matmul()
