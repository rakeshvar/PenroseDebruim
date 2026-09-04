"""Export retained DeBruijnTransformer checkpoints for ONNX Runtime Web."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config  # noqa: E402
from denoiser import DebruijnTransformer  # noqa: E402


DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT.parent
    / "cloud/runs/0824-penrose-debruim/debruim-k1-32x4"
    / "debruim_0824_2130_32x4_kmax1_mse/checkpoints"
    / "debruim_0824_2130_32x4_kmax1_mse_e050.pt",
    PROJECT_ROOT.parent
    / "cloud/runs/0824-penrose-debruim/debruim-k1-256x16"
    / "debruim_0824_2220_256x16_kmax1_mse/checkpoints"
    / "debruim_0824_2220_256x16_kmax1_mse_e196.pt",
)


class ConditionModel(nn.Module):
    def __init__(self, model: DebruijnTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.model(condition[:, :5], condition[:, 5])


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(checkpoint_path: Path, output_directory: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config.from_dict(checkpoint["config"])
    if config.data.kmax != 1 or int(checkpoint["num_tiles"]) != 90:
        raise ValueError(f"{checkpoint_path} is not the expected kmax=1, 90-tile model")

    model = DebruijnTransformer(config.model, config.data.kmax)
    model.load_state_dict(checkpoint["model"])
    wrapped = ConditionModel(model.eval())
    torch.backends.mha.set_fastpath_enabled(False)
    identifier = f"debruim-k1-{config.model.d_model}x{config.model.num_layers}"
    output_path = output_directory / f"{identifier}.onnx"
    condition = torch.tensor(
        [[0.125, 0.25, 0.375, 0.5, 0.75, 0.625]], dtype=torch.float32
    )

    with torch.inference_mode():
        expected = wrapped(condition).numpy()
        torch.onnx.export(
            wrapped,
            (condition,),
            output_path,
            input_names=["condition"],
            output_names=["prediction"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx.checker.check_model(onnx.load(output_path))
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(["prediction"], {"condition": condition.numpy()})[0]
    max_absolute_error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=2e-4, atol=2e-5):
        raise RuntimeError(
            f"{identifier} ONNX parity failed: max error {max_absolute_error:.8g}"
        )

    size = output_path.stat().st_size
    print(
        f"{identifier}: {size / 1024 / 1024:.2f} MiB, "
        f"max |PyTorch-ONNX|={max_absolute_error:.3g}"
    )
    return {
        "id": identifier,
        "label": f"{config.model.d_model} × {config.model.num_layers}",
        "file": output_path.name,
        "epoch": int(checkpoint["epoch"]),
        "kmax": config.data.kmax,
        "tiles": int(checkpoint["num_tiles"]),
        "dModel": config.model.d_model,
        "layers": config.model.num_layers,
        "heads": config.model.num_heads,
        "globalTokens": config.model.num_global_tokens,
        "xyScale": float(checkpoint["xy_scale"]),
        "angleScale": float(checkpoint["angle_scale"]),
        "bytes": size,
        "sha256": checksum(output_path),
        "maxParityError": max_absolute_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoints",
        type=Path,
        nargs="*",
        default=DEFAULT_CHECKPOINTS,
        help="checkpoint paths (defaults to the selected 0824 runs)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).parent / "public/models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    models = [
        export_checkpoint(path.resolve(), args.output_directory)
        for path in args.checkpoints
    ]
    manifest = {
        "format": 1,
        "input": {"name": "condition", "shape": [1, 6]},
        "output": {"name": "prediction", "shape": [1, 90, 4]},
        "models": models,
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
