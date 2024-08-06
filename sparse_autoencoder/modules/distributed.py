from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup


@dataclass
class Comm:
    group: ProcessGroup

    def all_reduce(
        self, x: torch.Tensor, op: ReduceOp = ReduceOp.SUM, async_op: bool = False
    ):
        return dist.all_reduce(tensor=x, op=op, group=self.group, async_op=async_op)

    def all_gather(
        self, x_list: List[torch.Tensor], x: torch.Tensor, async_op: bool = False
    ):
        return dist.all_gather(
            tensor_list=list(x_list), tensor=x, group=self.group, async_op=async_op
        )

    def broadcast(self):
        return dist.broadcast