"""Atomic checkpoints, RNG restoration, and newest/best retention."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


EPOCH_PATTERN = re.compile(r"_e(\d+)\.pt$")


def capture_rng(
    training_generator: torch.Generator, sampling_generator: torch.Generator
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "training_generator": training_generator.get_state(),
        "sampling_generator": sampling_generator.get_state(),
    }


def restore_rng(
    state: dict[str, Any],
    training_generator: torch.Generator,
    sampling_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    training_generator.set_state(state["training_generator"].cpu())
    sampling_generator.set_state(state["sampling_generator"].cpu())


def atomic_save(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def checkpoint_epoch(path: Path) -> int | None:
    match = EPOCH_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def retain_newest_and_best(
    checkpoint_dir: str | Path,
    identifier: str,
    newest_epoch: int,
    best_epoch: int,
) -> None:
    keep = {newest_epoch, best_epoch}
    for path in Path(checkpoint_dir).glob(f"{identifier}_e*.pt"):
        epoch = checkpoint_epoch(path)
        if epoch is not None and epoch not in keep:
            path.unlink()


def load_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)
