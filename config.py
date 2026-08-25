"""Strict YAML configuration and command-line overrides."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass
class DataConfig:
    kmax: int


@dataclass
class ModelConfig:
    d_model: int
    num_heads: int
    num_layers: int
    num_global_tokens: int
    dropout: float


@dataclass
class TrainConfig:
    batch_size: int
    samples_per_epoch: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    min_lr_factor: float
    grad_clip: float
    seed: int
    device: str
    no_rotation: bool


@dataclass
class SampleConfig:
    n: int
    seed: int
    γ: list[float] | None
    θoffset: float


@dataclass
class WandbConfig:
    enable: bool
    project: str
    run_name: str | None
    run_id: str | None


@dataclass
class OutputConfig:
    directory: str
    resume: str | None


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    sample: SampleConfig
    wandb: WandbConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Config":
        section_types = {
            "data": DataConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "sample": SampleConfig,
            "wandb": WandbConfig,
            "output": OutputConfig,
        }
        if set(value) != set(section_types):
            unknown = set(value) - set(section_types)
            missing = set(section_types) - set(value)
            raise ValueError(
                f"configuration sections differ: unknown={unknown}, missing={missing}"
            )
        for section, section_type in section_types.items():
            expected = {field.name for field in fields(section_type)}
            if not isinstance(value[section], dict):
                raise ValueError(f"{section} must be a mapping")
            actual = set(value[section])
            if actual != expected:
                raise ValueError(
                    f"{section} keys differ: unknown={actual - expected}, "
                    f"missing={expected - actual}"
                )
        return cls(
            data=DataConfig(**value["data"]),
            model=ModelConfig(**value["model"]),
            train=TrainConfig(**value["train"]),
            sample=SampleConfig(**value["sample"]),
            wandb=WandbConfig(**value["wandb"]),
            output=OutputConfig(**value["output"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SECTIONS = ("data", "model", "train", "sample", "wandb", "output")
IMMUTABLE_ON_RESUME = (
    "data.kmax",
    "model.d_model",
    "model.num_heads",
    "model.num_layers",
    "model.num_global_tokens",
    "model.dropout",
    "train.no_rotation",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _strict_merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            _strict_merge(base[key], value)
        else:
            base[key] = value


def _set_override(config: dict[str, Any], section: str, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"override must be KEY=VALUE: {expression}")
    key, raw = expression.split("=", 1)
    if section not in config or key not in config[section]:
        raise ValueError(f"unknown configuration key: {section}.{key}")
    config[section][key] = yaml.safe_load(raw)


def _checkpoint_config(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    value = checkpoint.get("config")
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {path} has no resolved config")
    return value


def _validate(config: Config) -> None:
    if config.data.kmax < 0:
        raise ValueError("data.kmax must be nonnegative")
    if config.model.d_model % config.model.num_heads:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if config.train.batch_size < 1 or config.train.samples_per_epoch < 1:
        raise ValueError("batch_size and samples_per_epoch must be positive")
    if config.train.num_epochs < 1 or config.sample.n < 1:
        raise ValueError("num_epochs and sample.n must be positive")
    if not isinstance(config.train.no_rotation, bool):
        raise ValueError("train.no_rotation must be boolean")
    if config.sample.γ is not None and len(config.sample.γ) != 5:
        raise ValueError("sample.γ must contain five phase values")
    if not 0 <= config.sample.θoffset <= 1:
        raise ValueError("sample.θoffset must lie in [0, 1]")


def validate_resume_immutables(
    current: Config, checkpoint_config: dict[str, Any]
) -> None:
    current_dict = current.to_dict()
    for dotted in IMMUTABLE_ON_RESUME:
        section, key = dotted.split(".")
        expected = checkpoint_config[section][key]
        actual = current_dict[section][key]
        if actual != expected:
            raise ValueError(
                f"cannot override immutable {dotted} on resume: {expected!r} -> {actual!r}"
            )


def load_config(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Train PenroseDebruim")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    for short, section in zip(("-d", "-m", "-t", "-s", "-w", "-o"), SECTIONS):
        parser.add_argument(
            short, f"--{section}", action="append", default=[], metavar="KEY=VALUE"
        )
    parser.add_argument("--kmax", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output-directory")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--sample-n", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--no-rotation", action="store_true", default=None)
    parser.add_argument(
        "--gamma",
        type=float,
        nargs=5,
        metavar=("γ0", "γ1", "γ2", "γ3", "γ4"),
    )
    parser.add_argument("--theta-offset", dest="θoffset", type=float)
    args = parser.parse_args(argv)

    base = _read_yaml(DEFAULT_CONFIG)
    if args.resume:
        base = copy.deepcopy(_checkpoint_config(args.resume))
    if args.config:
        _strict_merge(base, _read_yaml(args.config))
    for section in SECTIONS:
        for expression in getattr(args, section):
            _set_override(base, section, expression)

    direct = {
        ("data", "kmax"): args.kmax,
        ("train", "batch_size"): args.batch_size,
        ("train", "num_epochs"): args.epochs,
        ("train", "learning_rate"): args.learning_rate,
        ("train", "no_rotation"): args.no_rotation,
        ("output", "directory"): args.output_directory,
        ("wandb", "project"): args.wandb_project,
        ("wandb", "run_name"): args.wandb_run_name,
        ("sample", "n"): args.sample_n,
        ("sample", "seed"): args.sample_seed,
        ("sample", "γ"): args.gamma,
        ("sample", "θoffset"): args.θoffset,
    }
    for (section, key), value in direct.items():
        if value is not None:
            base[section][key] = value
    if args.resume:
        base["output"]["resume"] = str(args.resume)

    config = Config.from_dict(base)
    _validate(config)
    print(yaml.safe_dump(config.to_dict(), sort_keys=False).rstrip())
    return config, args
