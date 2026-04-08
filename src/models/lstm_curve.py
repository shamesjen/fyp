from __future__ import annotations

import torch
from torch import nn


class LSTMCurveForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling_mode: str = "last",
    ):
        super().__init__()
        self.pooling_mode = str(pooling_mode).lower()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        if self.pooling_mode not in {"last", "attention"}:
            raise ValueError(f"Unsupported LSTM pooling_mode {pooling_mode!r}")
        self.attention_score = nn.Linear(hidden_size, 1) if self.pooling_mode == "attention" else None
        self.pool_norm = nn.LayerNorm(hidden_size) if self.pooling_mode == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        if self.pooling_mode == "attention":
            scores = self.attention_score(outputs).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled_hidden = torch.sum(weights * outputs, dim=1)
            pooled_hidden = self.pool_norm(pooled_hidden)
        else:
            pooled_hidden = outputs[:, -1, :]
        return self.head(self.dropout(pooled_hidden))
