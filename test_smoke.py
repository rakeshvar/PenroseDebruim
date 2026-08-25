"""Local smoke test for generation, training, SVGs, and fresh-process resume."""

from __future__ import annotations

import csv
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import capture_rng, restore_rng, retain_newest_and_best
from config import load_config
from debruijn import (
    ANGLE_SCALE,
    GAMMA_DIM,
    SCALED_ANGLE_HALF_PERIOD,
    SCALED_ANGLE_PERIOD,
    TARGET_DIM,
    debruijn_torch,
    geometry_loss,
    scaled_angle_delta,
    scaled_target_torch,
    tile_count,
    unscale_target_torch,
    wrapped_error,
    xy_scale,
)
from debruijn_np import debruijn_numpy
from denoiser import DebruijnTransformer
from svg import rhombus_vertices, save_comparison_svg


ROOT = Path(__file__).parent
SMOKE_SVG_DIR = ROOT / "tests" / "smoke"


def test_generation_and_scaling() -> None:
    rng = np.random.default_rng(7)
    statistics_path = ROOT / "tests" / "scaling" / "raw_statistics.csv"
    with statistics_path.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["kmax"]): row for row in csv.DictReader(handle)}
    assert set(rows) == set(range(6))
    for kmax in range(0, 6):
        γ = rng.uniform(size=(2, GAMMA_DIM))
        θoffset = rng.uniform(size=2)
        numpy_geometry = debruijn_numpy(kmax, γ, θoffset)
        torch_geometry = debruijn_torch(kmax, torch.tensor(γ, dtype=torch.float64), torch.tensor(θoffset, dtype=torch.float64))
        assert numpy_geometry.centers.shape == (2, tile_count(kmax), 2)
        np.testing.assert_allclose(
            numpy_geometry.centers, torch_geometry.centers.numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            numpy_geometry.angles, torch_geometry.angles.numpy(), atol=1e-12
        )
        np.testing.assert_array_equal(
            numpy_geometry.colors, torch_geometry.colors.numpy()
        )
        patch_means = numpy_geometry.centers.mean(axis=1)
        assert np.all(np.linalg.norm(patch_means, axis=1) > 1e-9)
        row = rows[kmax]
        assert int(row["condition_samples"]) == 100_000
        assert int(row["seed"]) == 20260823
        assert int(row["tiles_per_condition"]) == tile_count(kmax)
        measured_raw_scale = math.sqrt(float(row["second_moment_xy"]))
        assert math.isclose(
            xy_scale(kmax),
            measured_raw_scale,
            rel_tol=0.0,
            abs_tol=1e-3,
        )
        assert abs(float(row["mean_x"])) < 1e-3
        assert abs(float(row["mean_y"])) < 1e-3
        assert abs(float(row["mean_angle"])) < 5e-3
        assert math.isclose(
            float(row["second_moment_angle"]),
            math.pi**2 / 3,
            rel_tol=0.0,
            abs_tol=1e-2,
        )
        if kmax == 0:
            assert tile_count(kmax) == 10


def test_angle_histogram() -> None:
    statistics_path = ROOT / "tests" / "scaling" / "raw_statistics.csv"
    histogram_path = ROOT / "tests" / "scaling" / "angle_histogram.csv"
    with statistics_path.open(newline="", encoding="utf-8") as handle:
        statistics = {int(row["kmax"]): row for row in csv.DictReader(handle)}
    with histogram_path.open(newline="", encoding="utf-8") as handle:
        histogram_rows = list(csv.DictReader(handle))
    for kmax in range(6):
        rows = [row for row in histogram_rows if int(row["kmax"]) == kmax]
        assert len(rows) == 20
        assert sum(int(row["count"]) for row in rows) == int(
            statistics[kmax]["observation_count"]
        )
        assert math.isclose(
            sum(float(row["fraction"]) for row in rows),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert all(int(row["count"]) > 0 for row in rows)
        if kmax >= 1:
            assert all(
                abs(float(row["fraction"]) - 0.05) < 1e-3 for row in rows
            )


def test_wrapped_error_scales() -> None:
    angles_hat = torch.tensor([-math.pi + 0.01])
    angles_star = torch.tensor([math.pi - 0.01])
    scaled_angles_hat = angles_hat * ANGLE_SCALE
    scaled_angles_star = angles_star * ANGLE_SCALE
    np.testing.assert_allclose(wrapped_error(angles_hat, angles_star).abs().numpy(), [0.02], atol=1e-6)
    np.testing.assert_allclose(wrapped_error(angles_hat / math.pi, angles_star / math.pi, 1.0).abs().numpy(), [0.02 / math.pi], atol=1e-6)
    np.testing.assert_allclose(wrapped_error(scaled_angles_hat, scaled_angles_star, SCALED_ANGLE_HALF_PERIOD).abs().numpy(), [0.02 * ANGLE_SCALE], atol=1e-6)
    forward = scaled_angle_delta(torch.tensor(-1.70), torch.tensor(1.70))
    backward = scaled_angle_delta(torch.tensor(1.70), torch.tensor(-1.70))
    assert 0.0 < forward.item() < 0.1
    assert -0.1 < backward.item() < 0.0
    target = torch.tensor([[[0.0, 0.0, -1.70, 1.0]]])
    equivalent = target.clone()
    equivalent[..., 2] += SCALED_ANGLE_PERIOD
    assert geometry_loss(equivalent, target).angle.item() < 1e-12


def test_config_and_optimizer_step() -> None:
    sampling_config, _ = load_config(["--gamma", "0.1", "0.2", "0.3", "0.4", "0.5", "--theta-offset", "0.75"])
    assert sampling_config.sample.γ == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert sampling_config.sample.θoffset == 0.75
    assert not sampling_config.train.no_rotation
    config, _ = load_config(
        [
            "--kmax",
            "0",
            "-m",
            "d_model=16",
            "-m",
            "num_heads=4",
            "-m",
            "num_layers=1",
            "-m",
            "num_global_tokens=2",
            "--no-rotation",
        ]
    )
    assert config.train.no_rotation
    model = DebruijnTransformer(config.model, config.data.kmax)
    assert model.tile_latents.shape == (tile_count(0), TARGET_DIM)
    assert model.geometry_embedding.weight.shape == (config.model.d_model, TARGET_DIM)
    assert model.geometry_embedding.bias is None
    assert not hasattr(model, "output_projection")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    γ = torch.rand(2, GAMMA_DIM)
    θoffset = torch.rand(2)
    geometry = debruijn_torch(0, γ)
    np.testing.assert_allclose(geometry.centers.numpy(), debruijn_torch(0, γ, 0.5).centers.numpy())
    assert not torch.allclose(geometry.centers, debruijn_torch(0, γ, θoffset).centers)
    target = scaled_target_torch(geometry, xy_scale(0))
    assert target.shape == (2, tile_count(0), 4)
    assert set(target[..., 3].unique().tolist()) == {-1.0, 1.0}
    normalized_tiles = []
    hook = model.output_norm.register_forward_hook(lambda module, inputs, output: normalized_tiles.append(output))
    prediction = model(γ, θoffset)
    hook.remove()
    assert prediction.shape == target.shape
    torch.testing.assert_close(prediction, F.linear(normalized_tiles[0], model.geometry_embedding.weight.T))
    loss = F.mse_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.geometry_embedding.weight.grad is not None
    optimizer.step()

    try:
        load_config(["-m", "unknown=1"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown configuration keys must be rejected")
    try:
        load_config(["--kmax", "-1"])
    except ValueError:
        pass
    else:
        raise AssertionError("negative kmax must be rejected")


def test_svg_and_retention() -> None:
    continuous_vertices = rhombus_vertices(np.array([[0.0, 0.0, 0.0]]), np.array([0.5]))
    np.testing.assert_allclose(continuous_vertices[0, 0], [math.sin(math.pi / 5), 0.0], atol=1e-12)
    extrapolated_vertices = rhombus_vertices(np.array([[0.0, 0.0, 0.0]]), np.array([1.5]))
    np.testing.assert_allclose(extrapolated_vertices[0, 0], [0.0, 0.0], atol=1e-12)
    γ = np.full(GAMMA_DIM, 0.25)
    θoffset = 0.25
    geometry = debruijn_numpy(0, γ, θoffset)
    exact = np.concatenate((geometry.centers, geometry.angles[:, None]), axis=1)
    predicted = exact.copy()
    predicted[:, 0] += 0.1
    SMOKE_SVG_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = SMOKE_SVG_DIR / "comparison.svg"
    predicted_colors = 1 - geometry.colors
    save_comparison_svg(svg_path, predicted, exact, predicted_colors, geometry.colors, γ=γ, θoffset=θoffset)
    tree = ET.parse(svg_path)
    svg_root = tree.getroot()
    assert svg_root.attrib["height"] == "1000"
    assert int(svg_root.attrib["width"]) > 0
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    lines = tree.findall(".//svg:line[@class='correspondence']", namespace)
    assert len(lines) == tile_count(0) == 10
    backgrounds = tree.findall(".//svg:rect[@fill='#000000']", namespace)
    assert len(backgrounds) == 1
    predicted_polygons = tree.findall(".//svg:polygon[@class='predicted']", namespace)
    exact_polygons = tree.findall(".//svg:polygon[@class='exact']", namespace)
    assert all("fill" in polygon.attrib for polygon in predicted_polygons + exact_polygons)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        identifier = "debruim_test"
        for epoch in range(4):
            (root / f"{identifier}_e{epoch:03d}.pt").touch()
        retain_newest_and_best(root, identifier, newest_epoch=3, best_epoch=1)
        assert sorted(path.name for path in root.glob("*.pt")) == [
            f"{identifier}_e001.pt",
            f"{identifier}_e003.pt",
        ]


def test_rng_round_trip() -> None:
    training = torch.Generator().manual_seed(10)
    sampling = torch.Generator().manual_seed(20)
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    state = capture_rng(training, sampling)
    expected = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
        float(torch.rand((), generator=training)),
        float(torch.rand((), generator=sampling)),
    )
    restore_rng(state, training, sampling)
    actual = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
        float(torch.rand((), generator=training)),
        float(torch.rand((), generator=sampling)),
    )
    assert expected == actual


def _run(command: list[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment.pop("WANDB_API_KEY", None)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def test_fresh_process_resume() -> None:
    python = sys.executable
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "outputs"
        common = [
            python,
            "train.py",
            "-d",
            "kmax=0",
            "-m",
            "d_model=16",
            "-m",
            "num_heads=4",
            "-m",
            "num_layers=1",
            "-m",
            "num_global_tokens=2",
            "-t",
            "batch_size=2",
            "-t",
            "samples_per_epoch=4",
            "-t",
            "device=cpu",
            "-w",
            "enable=false",
            "--no-rotation",
            "-o",
            f"directory={output}",
        ]
        _run(common + ["-t", "num_epochs=1"], ROOT)
        checkpoints = list(output.glob("*/checkpoints/*_e000.pt"))
        assert len(checkpoints) == 1
        first = checkpoints[0]
        run_dir = first.parents[1]
        assert len(list((run_dir / "svg").glob("*_e000.svg"))) == 1
        assert len(list((run_dir / "svg").glob("*_e000_latents.svg"))) == 1

        _run(
            [
                python,
                "train.py",
                "--resume",
                str(first),
                "-t",
                "num_epochs=2",
            ],
            ROOT,
        )
        epoch_one = list((run_dir / "checkpoints").glob("*_e001.pt"))
        assert len(epoch_one) == 1
        checkpoint = torch.load(epoch_one[0], map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 1
        assert checkpoint["no_rotation"] is True
        assert checkpoint["identifier"] in epoch_one[0].name
        training_svgs = list((run_dir / "svg").glob("*_e001.svg"))
        assert len(training_svgs) == 1
        latent_svgs = list((run_dir / "svg").glob("*_e001_latents.svg"))
        assert len(latent_svgs) == 1
        SMOKE_SVG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(training_svgs[0], SMOKE_SVG_DIR / "training.svg")
        shutil.copy2(latent_svgs[0], SMOKE_SVG_DIR / "latents.svg")
        sample_dir = Path(temporary) / "samples"
        _run(
            [
                python,
                "sampler.py",
                "--resume",
                str(epoch_one[0]),
                "--sample-n",
                "2",
                "--output-directory",
                str(sample_dir),
            ],
            ROOT,
        )
        sample_svgs = sorted(sample_dir.glob("*_sample*.svg"))
        assert len(sample_svgs) == 2
        shutil.copy2(sample_svgs[0], SMOKE_SVG_DIR / "sample.svg")

        scaled = torch.zeros(1, tile_count(0), 3)
        assert unscale_target_torch(scaled, xy_scale(0)).shape == scaled.shape


def main() -> None:
    test_generation_and_scaling()
    test_angle_histogram()
    test_wrapped_error_scales()
    test_config_and_optimizer_step()
    test_svg_and_retention()
    test_rng_round_trip()
    test_fresh_process_resume()
    print("PenroseDebruim smoke test passed")


if __name__ == "__main__":
    main()
