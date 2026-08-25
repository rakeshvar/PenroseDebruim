"""One-pass inference and prediction-to-target comparison SVGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from checkpoint import load_checkpoint
from config import Config, load_config
from debruijn import (
    CONDITION_DIM,
    GAMMA_DIM,
    TARGET_DIM,
    debruijn_torch,
    scaled_target_torch,
    unscale_target_torch,
    xy_scale,
)
from denoiser import DebruijnTransformer
from svg import save_comparison_svg


@torch.no_grad()
def save_model_comparison(
    model: DebruijnTransformer,
    γ: torch.Tensor,
    θoffset: float | torch.Tensor,
    kmax: int,
    xy_scale: float,
    path: str | Path,
    *,
    target_θoffset: float | torch.Tensor | None = None,
) -> Path:
    model.eval()
    if γ.ndim == 1:
        γ = γ.unsqueeze(0)
    if γ.shape[0] != 1:
        raise ValueError("an SVG comparison requires exactly one γ vector")
    θoffset = torch.as_tensor(θoffset, dtype=γ.dtype, device=γ.device).reshape(-1)
    if θoffset.numel() != 1:
        raise ValueError("an SVG comparison requires exactly one θoffset value")
    if target_θoffset is None:
        target_θoffset = θoffset
    exact_geometry = debruijn_torch(kmax, γ, target_θoffset)
    exact_scaled = scaled_target_torch(exact_geometry, xy_scale)
    prediction_scaled = model(γ, θoffset)
    prediction = unscale_target_torch(prediction_scaled, xy_scale)[0]
    exact = unscale_target_torch(exact_scaled, xy_scale)[0]
    predicted_colors = (prediction[:, 3] + 1) / 2
    return save_comparison_svg(
        path,
        prediction[:, :3].detach().cpu().numpy(),
        exact[:, :3].detach().cpu().numpy(),
        predicted_colors.detach().cpu().numpy(),
        exact_geometry.colors[0].detach().cpu().numpy(),
        γ=γ[0].detach().cpu().numpy(),
        θoffset=float(θoffset.item()),
        target_θoffset=float(exact_geometry.θoffset[0].item()),
    )


def _sample_γ(
    config: Config, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    if config.sample.γ is not None:
        return torch.tensor(config.sample.γ, device=device, dtype=torch.float32).unsqueeze(0)
    return torch.rand((config.sample.n, GAMMA_DIM), device=device, dtype=torch.float32, generator=generator)


def main() -> None:
    config, _ = load_config()
    if not config.output.resume:
        raise ValueError("sampler requires --resume CHECKPOINT")
    device = torch.device(config.train.device)
    checkpoint = load_checkpoint(config.output.resume, device)
    if int(checkpoint.get("condition_dim", -1)) != CONDITION_DIM:
        raise ValueError(
            f"checkpoint is not compatible with condition_dim={CONDITION_DIM}"
        )
    if int(checkpoint.get("target_dim", -1)) != TARGET_DIM or checkpoint.get("tied_geometry_embedding") is not True:
        raise ValueError("checkpoint does not use the required tied four-dimensional geometry embedding")
    model = DebruijnTransformer(config.model, config.data.kmax).to(device)
    model.load_state_dict(checkpoint["model"])
    scale = xy_scale(config.data.kmax)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.sample.seed)
    γ_batch = _sample_γ(config, device, generator)
    θoffset_batch = torch.full((γ_batch.shape[0],), config.sample.θoffset, device=device, dtype=torch.float32)
    target_θoffset = 0.5 if config.train.no_rotation else None
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (γ, θoffset) in enumerate(zip(γ_batch, θoffset_batch)):
        path = output_dir / f"{checkpoint['identifier']}_sample{index:03d}.svg"
        save_model_comparison(model, γ, θoffset, config.data.kmax, scale, path, target_θoffset=target_θoffset)
        print(path)


if __name__ == "__main__":
    main()
