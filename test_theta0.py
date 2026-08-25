"""Verify θ0 rotation and write two fixed-γ SVG examples."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from debruijn import θ0_from_offset
from debruijn_np import debruijn_numpy, wrap_angle_numpy
from svg import save_geometry_svg


ROOT = Path(__file__).parent
DEFAULT_OUTPUT = ROOT / "tests" / "theta0"
γ = np.array([0.11, 0.27, 0.43, 0.59, 0.73], dtype=np.float64)
θoffset_VALUES = (0.25, 0.75)


def generated_xya(kmax: int, γ: np.ndarray, θoffset: float) -> tuple[np.ndarray, np.ndarray]:
    geometry = debruijn_numpy(kmax, γ, θoffset)
    xya = np.concatenate((geometry.centers, geometry.angles[:, None]), axis=1)
    return xya, geometry.colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmax", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generated = []
    for θoffset in θoffset_VALUES:
        xya, colors = generated_xya(args.kmax, γ, θoffset)
        path = args.output / f"thetaoffset_{θoffset:.2f}.svg"
        save_geometry_svg(path, xya, colors, γ=γ, θoffset=θoffset)
        generated.append((θoffset, xya, colors, path))
        print(path)

    low_θoffset, low_xya, low_colors, _ = generated[0]
    high_θoffset, high_xya, high_colors, _ = generated[1]
    np.testing.assert_array_equal(low_colors, high_colors)
    θ0_delta = float(θ0_from_offset(high_θoffset) - θ0_from_offset(low_θoffset))
    cosine = math.cos(θ0_delta)
    sine = math.sin(θ0_delta)
    rotation_transpose = np.array([[cosine, sine], [-sine, cosine]])
    np.testing.assert_allclose(
        low_xya[:, :2] @ rotation_transpose,
        high_xya[:, :2],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        wrap_angle_numpy(high_xya[:, 2] - low_xya[:, 2]),
        θ0_delta,
        atol=1e-12,
    )
    print(f"θ0 rotation delta={θ0_delta:.12g} radians")


if __name__ == "__main__":
    main()
