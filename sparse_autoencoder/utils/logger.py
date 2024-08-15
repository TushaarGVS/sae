from typing import Dict, Any

import torch
import wandb


class Logger:
    def __init__(
        self,
        config: Dict[str, Any] | None,
        wandb_entity: str | None,
        run_id: str | None,
        RANK: int = -1,
    ):
        self.vals = {}
        self.enabled = RANK == 0
        if self.enabled:
            self.wandb_run = wandb.init(
                entity=wandb_entity, project="sae", name=run_id, config=config
            )

    def eager_log_kv(self, key: Any, val: Any) -> Any:
        if self.enabled:
            wandb.log({key: val.detach() if isinstance(val, torch.Tensor) else val})
        return val

    def lazy_log_kv(self, key: Any, val: Any) -> Any:
        if self.enabled:
            self.vals[key] = val.detach() if isinstance(val, torch.Tensor) else val
        return val

    def dump_lazy_logged_kvs(self):
        if self.enabled:
            wandb.log(self.vals)
            self.vals = {}

    def done(self):
        self.vals = {}
        wandb.finish()
