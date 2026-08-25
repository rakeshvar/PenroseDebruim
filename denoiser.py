"""The single fixed De Bruijn geometry transformer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import ModelConfig
from debruijn import CONDITION_DIM, GAMMA_DIM, TARGET_DIM, tile_count


class DebruijnTransformer(nn.Module):
    def __init__(self, config: ModelConfig, kmax: int) -> None:
        super().__init__()
        self.kmax = kmax
        self.num_tiles = tile_count(kmax)
        self.num_global_tokens = config.num_global_tokens

        self.tile_latents = nn.Parameter(torch.randn(self.num_tiles, TARGET_DIM) * 0.02)
        self.geometry_embedding = nn.Linear(TARGET_DIM, config.d_model, bias=False)
        self.condition_projection = nn.Sequential(
            nn.Linear(CONDITION_DIM, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.global_tokens = nn.Parameter(
            torch.randn(config.num_global_tokens, config.d_model) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, γ: torch.Tensor, θoffset: float | torch.Tensor = 0.5) -> torch.Tensor:
        if γ.ndim != 2 or γ.shape[1] != GAMMA_DIM:
            raise ValueError(f"γ must have shape (batch, {GAMMA_DIM}), got {γ.shape}")
        θoffset = torch.as_tensor(θoffset, dtype=γ.dtype, device=γ.device)
        if θoffset.ndim == 0:
            θoffset = θoffset.expand(γ.shape[0])
        if θoffset.shape != (γ.shape[0],):
            raise ValueError(f"θoffset must be scalar or have shape ({γ.shape[0]},), got {tuple(θoffset.shape)}")
        condition = torch.cat((γ, θoffset[:, None]), dim=1)
        batch_size = γ.shape[0]
        tiles = self.geometry_embedding(self.tile_latents)
        tiles = tiles.unsqueeze(0).expand(batch_size, -1, -1)
        globals_ = self.global_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = torch.cat((globals_, tiles), dim=1)
        hidden = hidden + self.condition_projection(condition).unsqueeze(1)
        hidden = self.encoder(hidden)
        tile_hidden = hidden[:, self.num_global_tokens :]
        return F.linear(self.output_norm(tile_hidden), self.geometry_embedding.weight.T)
