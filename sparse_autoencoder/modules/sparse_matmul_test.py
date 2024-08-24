import torch
from jaxtyping import Float
from torch import autograd
from torch.profiler import profile, ProfilerActivity, record_function

from sparse_autoencoder.modules.sparse_matmul import (
    coo_sparse_dense_matmul,
    dense_transpose_sparse_matmul,
    sparse_dense_matmul,
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

    k, auxk = 32, 256
    x_topk = x.topk(k=(k + auxk), dim=-1)
    x_idxs, x_vals = x_topk.indices, x_topk.values
    x_sparse = torch.zeros_like(x)
    x_sparse.scatter_(dim=-1, index=x_idxs, src=x_vals)
    y_ref: Float[torch.Tensor, "d f"] = dense.T @ x_sparse

    y = dense_transpose_sparse_matmul(dense, x_idxs, x_vals, x.shape[1])
    max_abs_err_y = torch.max(torch.abs(y - y_ref)).item()

    dydw = autograd.grad(torch.sum(y), dense, retain_graph=True)[0]
    dydw_ref = autograd.grad(torch.sum(y_ref), dense, retain_graph=True)[0]
    max_abs_err_dydw = torch.max(torch.abs(dydw - dydw_ref)).item()

    dydsparse = autograd.grad(torch.sum(y), x_vals, retain_graph=True)[0]
    dydsparse_ref = autograd.grad(torch.sum(y_ref), x_sparse, retain_graph=True)[0]
    dydsparse_ref = dydsparse_ref.gather(1, x_idxs)
    max_abs_err_dydsparse = torch.max(torch.abs(dydsparse - dydsparse_ref)).item()

    print(f"{x_vals.dtype=}, {x_vals.shape=}")
    print(f"dense_numel={dense.numel()}, {dense.dtype=}, {dense.shape=}")
    print(f"{dydw.shape=}, {dydw_ref.shape=}")
    print(f"{dydsparse.shape=}, {dydsparse_ref.shape=}")
    print(
        f"[dense_transpose_sparse_matmul] "
        f"{max_abs_err_y=}, "
        f"{max_abs_err_dydw=}, "
        f"{max_abs_err_dydsparse=}\n"
        f"--"
    )


def test_sparse_dense_matmul() -> None:
    x: Float[torch.Tensor, "*b f"] = get_fl_tensor(torch.Size([8192, 50_000]))
    dense: Float[torch.Tensor, "f d"] = get_fl_tensor(torch.Size([50_000, 4096]))
    bias: Float[torch.Tensor, "d"] = get_fl_tensor(torch.Size([4096]))

    k, auxk = 32, 256
    x_topk = x.topk(k=(k + auxk), dim=-1)
    x_idxs, x_vals = x_topk.indices, x_topk.values
    x_sparse = torch.zeros_like(x)
    x_sparse.scatter_(dim=-1, index=x_idxs, src=x_vals)
    y_ref: Float[torch.Tensor, "*b d"] = (x_sparse @ dense) + bias

    y = sparse_dense_matmul(x_idxs, x_vals, dense, bias)
    max_abs_err_y = torch.max(torch.abs(y - y_ref)).item()

    dydw = autograd.grad(torch.sum(y), dense, retain_graph=True)[0]
    dydw_ref = autograd.grad(torch.sum(y_ref), dense, retain_graph=True)[0]
    max_abs_err_dydw = torch.max(torch.abs(dydw - dydw_ref)).item()

    dydsparse = autograd.grad(torch.sum(y), x_vals, retain_graph=True)[0]
    dydsparse_ref = autograd.grad(torch.sum(y_ref), x_sparse, retain_graph=True)[0]
    dydsparse_ref = dydsparse_ref.gather(1, x_idxs)
    max_abs_err_dydsparse = torch.max(torch.abs(dydsparse - dydsparse_ref)).item()

    dydbias = autograd.grad(torch.sum(y), bias, retain_graph=True)[0]
    dydbias_ref = autograd.grad(torch.sum(y_ref), bias, retain_graph=True)[0]
    max_abs_err_dydbias = torch.max(torch.abs(dydbias - dydbias_ref)).item()

    print(f"{x_vals.dtype=}, {x_vals.shape=}")
    print(f"dense_numel={dense.numel()}, {dense.dtype=}, {dense.shape=}")
    print(f"{dydw.shape=}, {dydw_ref.shape=}")
    print(f"{dydsparse.shape=}, {dydsparse_ref.shape=}")
    print(f"{dydbias.shape=}, {dydbias_ref.shape=}")
    print(
        f"[sparse_dense_matmul] "
        f"{max_abs_err_y=}, "
        f"{max_abs_err_dydw=}, "
        f"{max_abs_err_dydsparse=}, "
        f"{max_abs_err_dydbias=}\n"
        f"--"
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")  # only for testing

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as proton:
        with record_function("coo_sparse_dense_matmul"):
            test_coo_sparse_dense_matmul()
        with record_function("dense_transpose_sparse_matmul"):
            test_dense_transpose_sparse_matmul()
        with record_function("sparse_dense_matmul"):
            test_sparse_dense_matmul()

    proton.export_chrome_trace("artefacts/sparse_matmul_test_trace.json")
