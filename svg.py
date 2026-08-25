"""SVG rendering for exact and predicted De Bruijn rhombi."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np


FILL_RGB = np.array(((76, 120, 168), (242, 142, 43)), dtype=np.float64)


def _condition_text(γ: np.ndarray | None, θoffset: float | None, target_θoffset: float | None = None) -> str:
    if γ is None:
        return ""
    values = ", ".join(f"{value:.6f}" for value in np.asarray(γ))
    parts = [f"γ=[{values}]"]
    if θoffset is not None:
        parts.extend((f"θoffset={θoffset:.6f}", f"θ0={(θoffset - 0.5) * np.pi / 5:.6f}"))
    if target_θoffset is not None and target_θoffset != θoffset:
        parts.extend((f"target θoffset={target_θoffset:.6f}", f"target θ0={(target_θoffset - 0.5) * np.pi / 5:.6f}"))
    return f"<desc>{html.escape('; '.join(parts))}</desc>"


def rhombus_vertices(
    geometry: np.ndarray, colors: np.ndarray, side: float = 1.0
) -> np.ndarray:
    geometry = np.asarray(geometry, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    centers = geometry[..., :2]
    angles = geometry[..., 2]
    top_angles = (3 - 2 * colors) * np.pi / 5
    short = side * np.cos(top_angles / 2)
    long = side * np.sin(top_angles / 2)
    along = np.stack((np.cos(angles), np.sin(angles)), axis=-1) * long[..., None]
    across = np.stack((-np.sin(angles), np.cos(angles)), axis=-1) * short[..., None]
    return centers[..., None, :] + np.stack((along, across, -along, -across), axis=-2)


def _points(vertices: np.ndarray) -> str:
    return " ".join(f"{x:.6f},{y:.6f}" for x, y in vertices)


def _fill_color(color: float) -> str:
    color = float(color)
    red, green, blue = np.rint((1 - color) * FILL_RGB[0] + color * FILL_RGB[1]).astype(int)
    return f"rgb({red} {green} {blue})"


def save_geometry_svg(
    path: str | Path,
    geometry: np.ndarray,
    colors: np.ndarray,
    *,
    γ: np.ndarray | None = None,
    θoffset: float | None = None,
    side: float = 1.0,
) -> Path:
    """Render one generated rhombus patch without comparison overlays."""
    path = Path(path)
    geometry = np.asarray(geometry, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError("geometry must have shape (N, 3)")
    if colors.shape != (geometry.shape[0],):
        raise ValueError("colors must have shape (N,)")

    vertices = rhombus_vertices(geometry, colors, side)
    points = vertices.reshape(-1, 2)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = max(float(span.max()) * 0.04, side)
    x0, y0 = minimum - margin
    width, height = span + 2 * margin
    display_height = 1000
    display_width = display_height * width / height
    stroke_width = max(float(span.max()) / 900, 0.008)
    condition_text = _condition_text(γ, θoffset)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_width:.0f}" height="{display_height}" viewBox="{x0:.6f} '
            f'{-y0-height:.6f} {width:.6f} {height:.6f}">'
        ),
        condition_text,
        f'<rect x="{x0:.6f}" y="{-y0-height:.6f}" width="{width:.6f}" height="{height:.6f}" fill="#000000"/>',
        '<g transform="scale(1,-1)" stroke="#f3f4f6" stroke-linejoin="round">',
    ]
    for polygon, color in zip(vertices, colors):
        lines.append(
            f'<polygon class="generated" points="{_points(polygon)}" '
            f'fill="{_fill_color(color)}" stroke-width="{stroke_width:.6f}"/>'
        )
    lines.extend(("</g>", "</svg>"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path


def save_comparison_svg(
    path: str | Path,
    predicted: np.ndarray,
    exact: np.ndarray,
    predicted_colors: np.ndarray,
    exact_colors: np.ndarray,
    *,
    γ: np.ndarray | None = None,
    θoffset: float | None = None,
    target_θoffset: float | None = None,
    side: float = 1.0,
) -> Path:
    """Overlay both sets and connect corresponding centers with one line each."""
    path = Path(path)
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    predicted_colors = np.asarray(predicted_colors, dtype=np.float64)
    exact_colors = np.asarray(exact_colors, dtype=np.float64)
    if predicted.shape != exact.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("predicted and exact must both have shape (N, 3)")
    if predicted_colors.shape != (predicted.shape[0],) or exact_colors.shape != (predicted.shape[0],):
        raise ValueError("predicted_colors and exact_colors must have shape (N,)")

    predicted_vertices = rhombus_vertices(predicted, predicted_colors, side)
    exact_vertices = rhombus_vertices(exact, exact_colors, side)
    all_points = np.concatenate(
        (predicted_vertices.reshape(-1, 2), exact_vertices.reshape(-1, 2)), axis=0
    )
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = max(float(span.max()) * 0.04, side)
    x0, y0 = minimum - margin
    width, height = span + 2 * margin
    display_height = 1000
    display_width = display_height * width / height
    stroke_width = max(float(span.max()) / 900, 0.008)

    condition_text = _condition_text(γ, θoffset, target_θoffset)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_width:.0f}" height="{display_height}" viewBox="{x0:.6f} '
            f'{-y0-height:.6f} {width:.6f} {height:.6f}">'
        ),
        condition_text,
        f'<rect x="{x0:.6f}" y="{-y0-height:.6f}" width="{width:.6f}" height="{height:.6f}" fill="#000000"/>',
        f'<g transform="scale(1,-1)" stroke-linejoin="round" stroke-linecap="round">',
        '<g id="correspondences" stroke="#9ca3af" opacity="0.65" fill="none">',
    ]
    for predicted_center, exact_center in zip(predicted[:, :2], exact[:, :2]):
        lines.append(
            '<line class="correspondence" '
            f'x1="{predicted_center[0]:.6f}" y1="{predicted_center[1]:.6f}" '
            f'x2="{exact_center[0]:.6f}" y2="{exact_center[1]:.6f}" '
            f'stroke-width="{stroke_width:.6f}"/>'
        )
    lines.append("</g>")
    lines.append('<g id="exact" stroke="#f3f4f6" opacity="0.45">')
    for vertices, color in zip(exact_vertices, exact_colors):
        lines.append(
            f'<polygon class="exact" points="{_points(vertices)}" '
            f'fill="{_fill_color(color)}" stroke-width="{stroke_width:.6f}"/>'
        )
    lines.append("</g>")
    lines.append('<g id="predicted" stroke="#ef4444" fill-opacity="0.55">')
    for vertices, color in zip(predicted_vertices, predicted_colors):
        lines.append(
            f'<polygon class="predicted" points="{_points(vertices)}" '
            f'fill="{_fill_color(color)}" stroke-width="{2 * stroke_width:.6f}"/>'
        )
    lines.extend(("</g>", "</g>", "</svg>"))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path
