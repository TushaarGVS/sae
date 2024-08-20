import torch
from torch import autograd
from torch.profiler import profile, record_function, ProfilerActivity

from sparse_autoencoder.modules.loss import mse_loss, mse_auxk_loss
from sparse_autoencoder.modules.test_utils import get_fl_tensor

_factory_kwargs = {"dtype": torch.float16}


def _mse_ref(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output.float() - target.float()).pow(2).mean()


def _unnorm_mse_ref(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output.float() - target.float()).pow(2).sum()


def test_mse_loss() -> None:
    output = get_fl_tensor(torch.Size([8192, 16_384]), factory_kwargs=_factory_kwargs)
    target = get_fl_tensor(torch.Size([8192, 16_384]), factory_kwargs=_factory_kwargs)
    target.requires_grad = False
    loss_ref = _mse_ref(output, target)

    loss = mse_loss(output, target)
    max_abs_err_loss = torch.abs(loss - loss_ref).item()

    dloss = autograd.grad(loss, output, retain_graph=True)[0]
    dloss_ref = autograd.grad(loss_ref, output, retain_graph=True)[0]
    max_abs_err_dloss = torch.max(torch.abs(dloss - dloss_ref)).item()

    print(f"output_numel={output.numel()}, {output.dtype=}, {output.shape=}")
    print(f"{loss.dtype=}, {loss.shape=}")
    print(f"{dloss.dtype=}, {dloss.shape=}")
    print(f"[mse_loss] {max_abs_err_loss=}, {max_abs_err_dloss=}\n--")


def test_mse_aux_loss() -> None:
    batch, dim_d = 8192, 16_384
    recons = get_fl_tensor(torch.Size([batch, dim_d]), factory_kwargs=_factory_kwargs)
    auxk_recons = get_fl_tensor(
        torch.Size([batch, dim_d]), factory_kwargs=_factory_kwargs
    )
    acts = get_fl_tensor(torch.Size([batch, dim_d]), factory_kwargs=_factory_kwargs)
    acts.requires_grad = False
    recons_loss_ref = _mse_ref(recons, acts)
    auxk_loss_ref = _unnorm_mse_ref(auxk_recons, acts - recons) / _unnorm_mse_ref(
        auxk_recons.mean(dim=0)[None].broadcast_to(recons.shape), acts - recons.detach()
    )

    recons_loss, auxk_loss = mse_auxk_loss(recons, auxk_recons, acts)
    max_abs_err_loss = torch.abs(recons_loss - recons_loss_ref).item()
    max_abs_err_auxk = torch.abs(auxk_loss - auxk_loss_ref).item()

    dloss = autograd.grad(recons_loss, recons, retain_graph=True)[0]
    dloss_ref = autograd.grad(recons_loss_ref, recons, retain_graph=True)[0]
    max_abs_err_drecons = torch.max(torch.abs(dloss - dloss_ref)).item()

    dauxk = autograd.grad(auxk_loss, auxk_recons, retain_graph=True)[0]
    dauxk_ref = autograd.grad(auxk_loss_ref, auxk_recons, retain_graph=True)[0]
    max_abs_err_dauxk = torch.max(torch.abs(dauxk - dauxk_ref)).item()

    print(f"recons_numel={recons.numel()}, {recons.dtype=}, {recons.shape=}")
    print(f"{recons_loss.dtype=}, {recons_loss.shape=}")
    print(f"{auxk_loss.dtype=}, {auxk_loss.shape=}")
    print(
        f"[mse_aux_loss] "
        f"{max_abs_err_loss=}, "
        f"{max_abs_err_auxk=}, "
        f"{max_abs_err_drecons=}, "
        f"{max_abs_err_dauxk=}\n"
        f"--"
    )


if __name__ == "__main__":
    test_mse_loss()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as proton:
        with record_function("fused_mse_aux_loss"):
            test_mse_aux_loss()

    print(proton.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(proton.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    proton.export_chrome_trace("artefacts/fused_mse_aux_loss_test_trace.json")
