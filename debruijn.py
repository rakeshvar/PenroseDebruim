"""Vectorized PyTorch De Bruijn generation used during training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


SCALED_ANGLE_HALF_PERIOD = math.sqrt(3.0)
SCALED_ANGLE_PERIOD = 2.0 * SCALED_ANGLE_HALF_PERIOD
ANGLE_SCALE = SCALED_ANGLE_HALF_PERIOD / math.pi
XY_SCALE_RESIDUAL = 5.65
GAMMA_DIM = 5
CONDITION_DIM = 6
TARGET_DIM = 4
θ0_HALF_RANGE = math.pi / 10


@dataclass(frozen=True)
class TorchGeometry:
    γ: torch.Tensor
    θoffset: torch.Tensor
    θ0: torch.Tensor
    centers: torch.Tensor
    angles: torch.Tensor
    colors: torch.Tensor


@dataclass(frozen=True)
class GeometryLoss:
    total: torch.Tensor
    xy: torch.Tensor
    angle: torch.Tensor
    color: torch.Tensor


def tile_count(kmax: int) -> int:
    if kmax < 0:
        raise ValueError("kmax must be nonnegative")
    return 10 * (2 * kmax + 1) ** 2


def xy_scale(kmax: int) -> float:
    """Calibrated shared x/y population scale for raw centers."""
    tile_count(kmax)
    return math.sqrt((25.0 / 6.0) * kmax * (kmax + 1) + XY_SCALE_RESIDUAL)


def θ0_from_offset(θoffset):
    """Map θoffset from [0, 1] to θ0 in [-π/10, π/10]."""
    return (θoffset - 0.5) * (2 * θ0_HALF_RANGE)


def wrap_angle_torch(θ: torch.Tensor) -> torch.Tensor:
    return torch.remainder(θ + math.pi, 2 * math.pi) - math.pi


def wrapped_error(θ_hat: torch.Tensor, θ_star: torch.Tensor, half_period: float = math.pi) -> torch.Tensor:
    """Return θ_star - θ_hat wrapped to [-half_period, half_period)."""
    Δθ = θ_star - θ_hat
    return torch.remainder(Δθ + half_period, 2 * half_period) - half_period


def wrap_scaled_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(
        angle + SCALED_ANGLE_HALF_PERIOD,
        SCALED_ANGLE_PERIOD,
    ) - SCALED_ANGLE_HALF_PERIOD


def scaled_angle_delta(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return wrap_scaled_angle(target - source)


def canonicalize_scaled_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim < 1 or target.shape[-1] not in (3, 4):
        raise ValueError(
            f"target must have final dimension 3 or 4, got {tuple(target.shape)}"
        )
    values = [target[..., :2], wrap_scaled_angle(target[..., 2:3])]
    if target.shape[-1] == 4:
        values.append(target[..., 3:4])
    return torch.cat(values, dim=-1)


def geometry_loss(prediction: torch.Tensor, target: torch.Tensor) -> GeometryLoss:
    if prediction.shape != target.shape or prediction.shape[-1] != TARGET_DIM:
        raise ValueError(
            f"prediction and target must share final dimension {TARGET_DIM}, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    xy = (prediction[..., :2] - target[..., :2]).square().mean()
    angle = scaled_angle_delta(prediction[..., 2], target[..., 2]).square().mean()
    color = (prediction[..., 3] - target[..., 3]).square().mean()
    return GeometryLoss(total=(2 * xy + angle + color) / 4, xy=xy, angle=angle, color=color)


def _index_data(kmax: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    k = torch.arange(-kmax, kmax + 1, dtype=torch.int64, device=device)
    family_i, family_j = torch.triu_indices(5, 5, offset=1, device=device)
    line_i, line_j = torch.meshgrid(k, k, indexing="ij")
    pair_size = line_i.numel()
    line_i = line_i.reshape(1, -1).expand(10, -1).reshape(-1)
    line_j = line_j.reshape(1, -1).expand(10, -1).reshape(-1)
    family_i = family_i.repeat_interleave(pair_size)
    family_j = family_j.repeat_interleave(pair_size)
    return family_i, family_j, line_i, line_j


def _prepare_inputs(γ: torch.Tensor, θoffset: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    squeeze = γ.ndim == 1
    if squeeze:
        γ = γ.unsqueeze(0)
    if γ.ndim != 2 or γ.shape[1] != GAMMA_DIM:
        raise ValueError(f"γ must have shape ({GAMMA_DIM},) or (batch, {GAMMA_DIM}), got {γ.shape}")
    θoffset = torch.as_tensor(θoffset, dtype=γ.dtype, device=γ.device)
    if θoffset.ndim == 0:
        θoffset = θoffset.expand(γ.shape[0])
    elif θoffset.ndim == 1 and θoffset.shape[0] == γ.shape[0]:
        pass
    else:
        raise ValueError(f"θoffset must be scalar or have shape ({γ.shape[0]},), got {tuple(θoffset.shape)}")
    if bool(torch.any((θoffset < 0) | (θoffset > 1))):
        raise ValueError("θoffset must lie in [0, 1]")
    return γ, θoffset, squeeze


def debruijn_torch(kmax: int, γ: torch.Tensor, θoffset: float | torch.Tensor = 0.5) -> TorchGeometry:
    """Generate an ordered batch on ``γ.device`` using float64 geometry."""
    tile_count(kmax)
    γ, θoffset, squeeze = _prepare_inputs(γ, θoffset)
    output_dtype = γ.dtype
    work_γ = γ.to(torch.float64)
    work_θoffset = θoffset.to(torch.float64)
    θ0 = θ0_from_offset(work_θoffset)
    θ = (2 * math.pi / 5) * torch.arange(5, dtype=torch.float64, device=γ.device)[None, :] + θ0[:, None]
    directions = torch.stack((θ.cos(), θ.sin()), dim=-1)
    family_i, family_j, line_i, line_j = _index_data(kmax, γ.device)

    ui = directions[:, family_i]
    uj = directions[:, family_j]
    determinant = ui[..., 0] * uj[..., 1] - ui[..., 1] * uj[..., 0]
    ci = line_i[None, :] - work_γ[:, family_i]
    cj = line_j[None, :] - work_γ[:, family_j]
    x = (ci * uj[..., 1] - cj * ui[..., 1]) / determinant
    y = (ui[..., 0] * cj - uj[..., 0] * ci) / determinant
    intersections = torch.stack((x, y), dim=-1)

    regions = torch.ceil(torch.einsum("bnd,bfd->bnf", intersections, directions) + work_γ[:, None, :])
    regions.scatter_(2, family_i[None, :, None].expand(regions.shape[0], -1, -1), line_i[None, :, None].expand(regions.shape[0], -1, -1).to(torch.float64))
    regions.scatter_(2, family_j[None, :, None].expand(regions.shape[0], -1, -1), line_j[None, :, None].expand(regions.shape[0], -1, -1).to(torch.float64))

    axis = directions[:, family_i] + directions[:, family_j]
    centers = torch.einsum("bnf,bfd->bnd", regions, directions) + axis / 2
    θ1 = torch.atan2(axis[..., 1], axis[..., 0])
    parity = regions.sum(dim=-1).to(torch.int64).remainder(2).bool()
    angles = wrap_angle_torch(torch.where(parity, θ1, θ1 + math.pi))
    colors = ((family_j - family_i == 2) | (family_j - family_i == 3)).to(torch.uint8)
    colors = colors[None, :].expand(angles.shape[0], -1)

    centers = centers.to(output_dtype)
    angles = angles.to(output_dtype)
    θ0 = θ0.to(output_dtype)
    if squeeze:
        return TorchGeometry(γ[0], θoffset[0], θ0[0], centers[0], angles[0], colors[0])
    return TorchGeometry(γ, θoffset, θ0, centers, angles, colors)


def scaled_target_torch(geometry: TorchGeometry, xy_scale: float) -> torch.Tensor:
    scaled_colors = geometry.colors.to(geometry.centers.dtype) * 2 - 1
    return torch.cat((geometry.centers / xy_scale, (wrap_angle_torch(geometry.angles) * ANGLE_SCALE).unsqueeze(-1), scaled_colors.unsqueeze(-1)), dim=-1)


def unscale_target_torch(target: torch.Tensor, xy_scale: float) -> torch.Tensor:
    target = canonicalize_scaled_target(target)
    values = [
        target[..., :2] * xy_scale,
        (target[..., 2] / ANGLE_SCALE).unsqueeze(-1),
    ]
    if target.shape[-1] == 4:
        values.append(target[..., 3:4])
    return torch.cat(values, dim=-1)
