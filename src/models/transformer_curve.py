from __future__ import annotations

import torch
from torch import nn


class TransformerCurveForecaster(nn.Module):
    """Lightweight encoder-style transformer for curve forecasting.

    The model consumes the observed input window only, so a bidirectional encoder
    over the available history is appropriate. A learnable positional embedding is
    used because the context length is fixed and short in the main 5-minute setup.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        context_length: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        pooling_mode: str = "last",
    ) -> None:
        super().__init__()
        if pooling_mode not in {"last", "mean"}:
            raise ValueError(f"Unsupported transformer pooling mode: {pooling_mode!r}")
        self.pooling_mode = pooling_mode
        self.input_projection = nn.Linear(input_size, embedding_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, context_length, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embedding_dim, output_size)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs) + self.position_embedding[:, : inputs.size(1), :]
        hidden = self.encoder(hidden)
        if self.pooling_mode == "mean":
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden[:, -1, :]
        pooled = self.output_norm(pooled)
        return self.head(self.output_dropout(pooled))
