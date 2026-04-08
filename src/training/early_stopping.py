from __future__ import annotations

from dataclasses import dataclass

import copy

import torch


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    best_loss: float = float("inf")
    counter: int = 0
    best_state: dict[str, torch.Tensor] | None = None

    def step(self, loss: float, model: torch.nn.Module) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
