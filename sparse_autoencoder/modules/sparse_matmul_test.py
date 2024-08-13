import torch
from jaxtyping import Float
from torch import autograd

from sparse_autoencoder.modules.sparse_matmul import (
    coo_sparse_dense_matmul,
    dense_transpose_sparse_matmul,
)
from sparse_autoencoder.modules.test_utils import get_fl_tensor


def test_coo_sparse_dense_matmul() -> None:
    x = get_fl_tensor(torch.Size([4096, 8192]), sparsity=0.0)
    dense = get_fl_tensor(torch.Size([8192, 2560]))
    y_ref = x @ dense

    y = coo_sparse_dense_matmul(x, dense)
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
        f"[coo_sparse_dense_matmul] "
        f"{max_abs_err_y=}, "
        f"{max_abs_err_dydw=}, "
        f"{max_abs_err_dydx=}\n"
        f"--"
    )


def test_dense_transpose_sparse_matmul() -> None:
    x: Float[torch.Tensor, "*b f"] = get_fl_tensor(torch.Size([8192, 50_000]))
    dense: Float[torch.Tensor, "*b d"] = get_fl_tensor(torch.Size([8192, 4096]))

    k, auxk = 32, 512
    x_topk = x.topk(k=(k + auxk), dim=-1)
    x_idxs, x_vals = x_topk.indices, x_topk.values
    x_sparse = torch.zeros_like(x)
    x_sparse.scatter_(dim=-1, index=x_idxs, src=x_vals)
    y_ref: Float[torch.Tensor, "d f"] = dense.T @ x_sparse

    y = dense_transpose_sparse_matmul(dense, x_idxs, x_vals, x.shape[1])
    max_abs_err_y = torch.max(torch.abs(y - y_ref)).item()

    print(f"{x_vals.dtype=}, {x_vals.shape=}")
    print(f"dense_numel={dense.numel()}, {dense.dtype=}, {dense.shape=}")
    print(f"[dense_transpose_sparse_matmul] {max_abs_err_y=}\n--")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")  # only for testing

    test_coo_sparse_dense_matmul()
    test_dense_transpose_sparse_matmul()
