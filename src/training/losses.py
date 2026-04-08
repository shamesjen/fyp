from __future__ import annotations

import torch


def smoothness_penalty(predictions: torch.Tensor) -> torch.Tensor:
    if predictions.shape[1] < 2:
        return torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    diff = predictions[:, 1:] - predictions[:, :-1]
    return (diff**2).mean()


def no_arb_hook(predictions: torch.Tensor) -> torch.Tensor:
    if predictions.shape[1] < 3:
        return torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    second_diff = predictions[:, 2:] - 2.0 * predictions[:, 1:-1] + predictions[:, :-2]
    return torch.relu(-second_diff).pow(2).mean()


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str = "mse",
    vega_weights: torch.Tensor | None = None,
    smoothness_weight: float = 0.0,
    no_arb_weight: float = 0.0,
) -> torch.Tensor:
    errors = predictions - targets
    if loss_name == "huber":
        base = torch.nn.functional.huber_loss(predictions, targets, reduction="none")
    else:
        base = errors.pow(2)

    if vega_weights is not None:
        base = base * vega_weights

    loss = base.mean()
    if smoothness_weight > 0:
        loss = loss + smoothness_weight * smoothness_penalty(predictions)
    if no_arb_weight > 0:
        loss = loss + no_arb_weight * no_arb_hook(predictions)
    return loss
