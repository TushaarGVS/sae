import functools

import torch

from sparse_autoencoder.array_typing import F


def contiguous(function: F):
    @functools.wraps(function)
    def wrapper(ctx, *args, **kwargs):
        return function(
            ctx,
            *(
                arg if not isinstance(arg, torch.Tensor) else arg.contiguous()
                for arg in args
            ),
            **{
                key: (val if not isinstance(val, torch.Tensor) else val.contiguous())
                for key, val in kwargs.items()
            },
        )

    return wrapper


def is_power_of_2(n: int) -> bool:
    """Check if `n` is a power of 2."""
    return n and not (n & (n - 1))


def next_power_of_2(n: int) -> bool:
    """Compute the next power of 2 greater or equal to `n`."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n
