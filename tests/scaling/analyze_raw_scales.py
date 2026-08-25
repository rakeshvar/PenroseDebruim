"""Reproduce raw-center x/y scale estimates without patch mean subtraction."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from debruijn_np import debruijn_numpy  # noqa: E402


def _moments(total: float, square_total: float, count: int) -> dict[str, float]:
    mean = total / count
    second_moment = square_total / count
    population_variance = second_moment - mean * mean
    sample_variance = population_variance * count / (count - 1)
    return {
        "mean": mean,
        "variance": population_variance,
        "sample_variance": sample_variance,
        "second_moment": second_moment,
    }


def estimate_raw_statistics(
    kmax: int, samples: int, seed: int, chunk_size: int
) -> tuple[dict[str, float | int], np.ndarray]:
    rng = np.random.default_rng(seed)
    coordinate_sum = np.zeros(2, dtype=np.float64)
    coordinate_square_sum = np.zeros(2, dtype=np.float64)
    angle_sum = 0.0
    angle_square_sum = 0.0
    angle_histogram = np.zeros(20, dtype=np.int64)
    angle_step = np.pi / 10
    count = 0
    remaining = samples
    while remaining:
        size = min(chunk_size, remaining)
        γ = rng.uniform(0.0, 1.0, size=(size, 5))
        θoffset = rng.uniform(0.0, 1.0, size=size)
        geometry = debruijn_numpy(kmax, γ, θoffset)
        centers = geometry.centers
        coordinate_sum += centers.sum(axis=(0, 1), dtype=np.float64)
        coordinate_square_sum += np.square(centers).sum(
            axis=(0, 1), dtype=np.float64
        )
        angles = geometry.angles
        angle_sum += float(angles.sum(dtype=np.float64))
        angle_square_sum += float(np.square(angles).sum(dtype=np.float64))
        angle_bin = np.floor((angles + np.pi) / angle_step).astype(np.int64)
        angle_bin = np.clip(angle_bin, 0, 19)
        angle_histogram += np.bincount(angle_bin.ravel(), minlength=20)
        count += centers.shape[0] * centers.shape[1]
        remaining -= size
    x = _moments(float(coordinate_sum[0]), float(coordinate_square_sum[0]), count)
    y = _moments(float(coordinate_sum[1]), float(coordinate_square_sum[1]), count)
    xy = _moments(
        float(coordinate_sum.sum()), float(coordinate_square_sum.sum()), 2 * count
    )
    angle = _moments(angle_sum, angle_square_sum, count)
    row: dict[str, float | int] = {
        "kmax": kmax,
        "condition_samples": samples,
        "tiles_per_condition": 10 * (2 * kmax + 1) ** 2,
        "observation_count": count,
        "seed": seed,
    }
    for name, values in (("x", x), ("y", y), ("xy", xy), ("angle", angle)):
        for statistic, value in values.items():
            row[f"{statistic}_{name}"] = value
    leading = (25.0 / 6.0) * kmax * (kmax + 1)
    row["leading_variance_xy"] = leading
    row["second_moment_residual_xy"] = xy["second_moment"] - leading
    return row, angle_histogram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--kmax", type=int, nargs="+", default=list(range(6)))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "scaling" / "raw_statistics.csv",
    )
    parser.add_argument(
        "--angle-output",
        type=Path,
        default=ROOT / "tests" / "scaling" / "angle_histogram.csv",
    )
    args = parser.parse_args()
    rows = []
    angle_rows = []
    for kmax in args.kmax:
        row, histogram = estimate_raw_statistics(
            kmax, args.samples, args.seed, args.chunk_size
        )
        rows.append(row)
        total = int(histogram.sum())
        for offset, count in enumerate(histogram):
            index = offset - 10
            angle_rows.append(
                {
                    "kmax": kmax,
                    "angle_bin_index": index,
                    "angle_lower_radians": index * np.pi / 10,
                    "angle_upper_radians": (index + 1) * np.pi / 10,
                    "count": int(count),
                    "fraction": int(count) / total,
                }
            )
    fieldnames = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    angle_temporary = args.angle_output.with_suffix(args.angle_output.suffix + ".tmp")
    with angle_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(angle_rows[0]))
        writer.writeheader()
        writer.writerows(angle_rows)
    angle_temporary.replace(args.angle_output)
    print(args.output)
    print(args.angle_output)
    for row in rows:
        print(
            f"kmax={row['kmax']} "
            f"second_moment_xy={row['second_moment_xy']:.12g} "
            f"residual={row['second_moment_residual_xy']:.12g} "
            f"second_moment_angle={row['second_moment_angle']:.12g}"
        )


if __name__ == "__main__":
    main()
