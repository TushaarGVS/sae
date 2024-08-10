from typing import Dict, Any, Tuple

import torch


def _fl_tensor(
    size: torch.Size, factory_kwargs: Dict[str, Any] | None = None
) -> torch.Tensor:
    default_factory_kwargs = {
        "dtype": torch.float32,
        "device": torch.device("cuda"),
        "requires_grad": True,
    }
    if factory_kwargs is not None:
        default_factory_kwargs.update(factory_kwargs)
    return torch.randn(size, **default_factory_kwargs)


def get_fl_tensor(
    size: torch.Size | Tuple,
    sparsity: float = 0.0,
    factory_kwargs: Dict[str, Any] | None = None,
) -> torch.Tensor:
    tensor = _fl_tensor(size, factory_kwargs)
    if sparsity == 0:
        return tensor

    flat_tensor = tensor.reshape(-1)
    k = int(flat_tensor.numel() * (1 - sparsity))
    tensor_topk = flat_tensor.topk(k=k, dim=-1)
    y = torch.zeros_like(flat_tensor)
    y.scatter_(dim=-1, index=tensor_topk.indices, src=tensor_topk.values)
    return y.reshape(tensor.shape)


if __name__ == "__main__":
    assert (get_fl_tensor((10, 20), sparsity=0.5)).count_nonzero().item() <= 100
