"""Readable NumPy reference implementation of De Bruijn generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from debruijn import ANGLE_SCALE, GAMMA_DIM, θ0_from_offset, tile_count


@dataclass(frozen=True)
class NumpyGeometry:
    γ: np.ndarray
    θoffset: np.ndarray
    θ0: np.ndarray
    centers: np.ndarray
    angles: np.ndarray
    colors: np.ndarray


def wrap_angle_numpy(θ: np.ndarray) -> np.ndarray:
    return (θ + np.pi) % (2 * np.pi) - np.pi


def _index_data(kmax: int) -> tuple[np.ndarray, ...]:
    k = np.arange(-kmax, kmax + 1, dtype=np.int64)
    family_i, family_j = np.triu_indices(5, k=1)
    line_i, line_j = np.meshgrid(k, k, indexing="ij")
    line_i = np.broadcast_to(line_i.ravel(), (10, line_i.size)).reshape(-1)
    line_j = np.broadcast_to(line_j.ravel(), (10, line_j.size)).reshape(-1)
    family_i = np.repeat(family_i, (2 * kmax + 1) ** 2)
    family_j = np.repeat(family_j, (2 * kmax + 1) ** 2)
    return family_i, family_j, line_i, line_j


def _prepare_inputs(γ: np.ndarray, θoffset: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    γ = np.asarray(γ, dtype=np.float64)
    squeeze = γ.ndim == 1
    if squeeze:
        γ = γ[None, :]
    if γ.ndim != 2 or γ.shape[1] != GAMMA_DIM:
        raise ValueError(f"γ must have shape ({GAMMA_DIM},) or (batch, {GAMMA_DIM}), got {γ.shape}")
    θoffset = np.asarray(θoffset, dtype=np.float64)
    if θoffset.ndim == 0:
        θoffset = np.full(γ.shape[0], float(θoffset))
    elif θoffset.ndim == 1 and θoffset.shape[0] == γ.shape[0]:
        pass
    else:
        raise ValueError(f"θoffset must be scalar or have shape ({γ.shape[0]},), got {θoffset.shape}")
    if np.any((θoffset < 0) | (θoffset > 1)):
        raise ValueError("θoffset must lie in [0, 1]")
    return γ, θoffset, squeeze


def debruijn_numpy(kmax: int, γ: np.ndarray, θoffset: float | np.ndarray = 0.5) -> NumpyGeometry:
    """Generate an ordered batch of uncropped De Bruijn rhombi."""
    tile_count(kmax)
    γ, θoffset, squeeze = _prepare_inputs(γ, θoffset)
    θ0 = θ0_from_offset(θoffset)
    θ = (2 * math.pi / 5) * np.arange(5, dtype=np.float64)[None, :] + θ0[:, None]
    directions = np.stack((np.cos(θ), np.sin(θ)), axis=-1)
    family_i, family_j, line_i, line_j = _index_data(kmax)

    ui = directions[:, family_i]
    uj = directions[:, family_j]
    determinant = ui[..., 0] * uj[..., 1] - ui[..., 1] * uj[..., 0]
    ci = line_i[None, :] - γ[:, family_i]
    cj = line_j[None, :] - γ[:, family_j]
    x = (ci * uj[..., 1] - cj * ui[..., 1]) / determinant
    y = (ui[..., 0] * cj - uj[..., 0] * ci) / determinant
    intersections = np.stack((x, y), axis=-1)

    regions = np.ceil(np.einsum("bnd,bfd->bnf", intersections, directions) + γ[:, None, :])
    positions = np.arange(regions.shape[1])
    regions[:, positions, family_i] = line_i
    regions[:, positions, family_j] = line_j

    axis = directions[:, family_i] + directions[:, family_j]
    centers = np.einsum("bnf,bfd->bnd", regions, directions) + axis / 2
    θ1 = np.arctan2(axis[..., 1], axis[..., 0])
    parity = regions.sum(axis=-1).astype(np.int64) % 2 == 1
    angles = wrap_angle_numpy(np.where(parity, θ1, θ1 + np.pi))
    colors = ((family_j - family_i == 2) | (family_j - family_i == 3)).astype(np.uint8)
    colors = np.broadcast_to(colors, angles.shape).copy()

    if squeeze:
        return NumpyGeometry(γ[0], θoffset[0], θ0[0], centers[0], angles[0], colors[0])
    return NumpyGeometry(γ, θoffset, θ0, centers, angles, colors)


def scaled_target_numpy(geometry: NumpyGeometry, xy_scale: float) -> np.ndarray:
    scaled_colors = geometry.colors.astype(geometry.centers.dtype) * 2 - 1
    return np.concatenate((geometry.centers / xy_scale, (wrap_angle_numpy(geometry.angles) * ANGLE_SCALE)[..., None], scaled_colors[..., None]), axis=-1)


def unscale_target_numpy(target: np.ndarray, xy_scale: float) -> np.ndarray:
    result = np.asarray(target).copy()
    result[..., :2] *= xy_scale
    result[..., 2] = wrap_angle_numpy(result[..., 2] / ANGLE_SCALE)
    return result
