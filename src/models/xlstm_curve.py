from __future__ import annotations

import torch
from torch import nn

from xlstm import mLSTMBlockConfig, mLSTMLayerConfig, xLSTMBlockStack, xLSTMBlockStackConfig


class XLSTMCurveForecaster(nn.Module):
    """CPU-safe xLSTM forecaster using an mLSTM block stack.

    The official xlstm package supports mixed mLSTM/sLSTM stacks, but the sLSTM
    path relies on CUDA-specific kernels in the current package version. For
    this repo we therefore use the official mLSTM block stack so the experiment
    remains reproducible on the existing environment.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        context_length: int,
        embedding_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        proj_factor: float = 2.0,
        conv1d_kernel_size: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, embedding_dim, bias=bias)
        mlstm_layer = mLSTMLayerConfig(
            embedding_dim=embedding_dim,
            context_length=context_length,
            num_heads=num_heads,
            proj_factor=proj_factor,
            conv1d_kernel_size=conv1d_kernel_size,
            dropout=dropout,
            bias=bias,
        )
        stack_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mlstm_layer),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            dropout=dropout,
            slstm_at=[],
            bias=bias,
        )
        self.backbone = xLSTMBlockStack(stack_config)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embedding_dim, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        hidden = self.backbone(hidden)
        final_hidden = hidden[:, -1, :]
        final_hidden = self.output_norm(final_hidden)
        return self.head(self.output_dropout(final_hidden))
