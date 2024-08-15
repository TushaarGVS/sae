import torch
from torch import autograd

from sparse_autoencoder.modules.norm import unit_normalize_w
from sparse_autoencoder.modules.test_utils import get_fl_tensor


def test_unit_normalize_w():
    dense = get_fl_tensor(torch.Size([8192, 4096]))
    norm_ref = dense / dense.norm(dim=1, keepdim=True)
    norm_dense = unit_normalize_w(dense)
    max_abs_err_y = torch.max(torch.abs(norm_dense - norm_ref)).item()

    dydw = autograd.grad(torch.sum(norm_dense), dense, retain_graph=True)[0]
    dydw_ref = autograd.grad(torch.sum(norm_ref), dense, retain_graph=True)[0]
    max_abs_err_dydw = torch.max(torch.abs(dydw - dydw_ref)).item()

    print(f"dense_numel={dense.numel()}, {dense.dtype=}, {dense.shape=}")
    print(f"[unit_normalize_w] {max_abs_err_y=}, {max_abs_err_dydw=}\n--")


if __name__ == "__main__":
    test_unit_normalize_w()
