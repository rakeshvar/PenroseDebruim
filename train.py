"""Train the deterministic De Bruijn geometry regressor."""

from __future__ import annotations

import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from checkpoint import (
    atomic_save,
    capture_rng,
    load_checkpoint,
    restore_rng,
    retain_newest_and_best,
)
from config import Config, load_config, validate_resume_immutables
from debruijn import (
    ANGLE_SCALE,
    CONDITION_DIM,
    GAMMA_DIM,
    TARGET_DIM,
    θ0_HALF_RANGE,
    debruijn_torch,
    geometry_loss,
    scaled_target_torch,
    tile_count,
    unscale_target_torch,
    xy_scale,
)
from denoiser import DebruijnTransformer
from sampler import save_model_comparison
from svg import save_geometry_svg


def build_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, min_lr_factor: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = min(10, math.floor(0.05 * epochs))

    def factor(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return 0.01 + 0.99 * epoch / warmup_epochs
        decay_epochs = max(1, epochs - warmup_epochs - 1)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / decay_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_identifier(config: Config) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%m%d_%H%M")
    architecture = f"{config.model.d_model}x{config.model.num_layers}"
    return f"debruim_{timestamp}_{architecture}_kmax{config.data.kmax}_mse"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def init_wandb(
    config: Config,
    identifier: str,
    model: torch.nn.Module,
    resume: bool,
    run_id: str | None,
) -> Any | None:
    if not config.wandb.enable:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("WandB disabled: WANDB_API_KEY is not set")
        return None
    try:
        import wandb
    except ImportError:
        print("WandB disabled: package is not installed")
        return None
    wandb_config = config.to_dict()
    wandb_config["parameter_counts"] = parameter_counts(model)
    return wandb.init(
        project=config.wandb.project,
        name=config.wandb.run_name or identifier,
        id=run_id,
        resume="must" if resume else None,
        config=wandb_config,
    )


def train(config: Config) -> Path:
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_everything(config.train.seed)
    geometry_scale = xy_scale(config.data.kmax)

    model = DebruijnTransformer(config.model, config.data.kmax).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer, config.train.num_epochs, config.train.min_lr_factor
    )
    training_generator = torch.Generator(device=device)
    sampling_generator = torch.Generator(device=device)
    training_generator.manual_seed(config.train.seed + 1000)
    sampling_generator.manual_seed(config.sample.seed)

    start_epoch = 0
    global_step = 0
    best_epoch = -1
    best_primary_metric = math.inf
    checkpoint: dict[str, Any] | None = None
    if config.output.resume:
        checkpoint = load_checkpoint(config.output.resume, device)
        validate_resume_immutables(config, checkpoint["config"])
        if int(checkpoint.get("condition_dim", -1)) != CONDITION_DIM:
            raise ValueError(
                f"checkpoint condition_dim does not match required {CONDITION_DIM}"
            )
        if int(checkpoint.get("target_dim", -1)) != TARGET_DIM or checkpoint.get("tied_geometry_embedding") is not True:
            raise ValueError("checkpoint does not use the required tied four-dimensional geometry embedding")
        if not math.isclose(
            float(checkpoint["xy_scale"]),
            geometry_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("checkpoint x/y scale does not match the scaling formula")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(
            checkpoint["rng"], training_generator, sampling_generator
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_epoch = int(checkpoint["best_epoch"])
        best_primary_metric = float(checkpoint["best_primary_metric"])
        identifier = checkpoint["identifier"]
        run_dir = Path(checkpoint["output_directory"])
    else:
        identifier = make_identifier(config)
        run_dir = Path(config.output.directory) / identifier

    checkpoint_dir = run_dir / "checkpoints"
    svg_dir = run_dir / "svg"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    run_id = checkpoint.get("wandb_run_id") if checkpoint else config.wandb.run_id
    wandb_run = init_wandb(
        config, identifier, model, checkpoint is not None, run_id
    )

    steps_per_epoch = math.ceil(
        config.train.samples_per_epoch / config.train.batch_size
    )
    last_checkpoint: Path | None = None
    for epoch in range(start_epoch, config.train.num_epochs):
        model.train()
        totals = {"loss": 0.0, "xy_mse": 0.0, "scaled_angle_mse": 0.0, "color_mse": 0.0, "color_accuracy": 0.0, "grad_norm": 0.0}
        seen = 0
        for step in range(steps_per_epoch):
            remaining = config.train.samples_per_epoch - step * config.train.batch_size
            batch_size = min(config.train.batch_size, remaining)
            γ = torch.rand((batch_size, GAMMA_DIM), device=device, dtype=torch.float32, generator=training_generator)
            θoffset = torch.rand(batch_size, device=device, dtype=torch.float32, generator=training_generator)
            target_θoffset = 0.5 if config.train.no_rotation else θoffset
            geometry = debruijn_torch(config.data.kmax, γ, target_θoffset)
            target = scaled_target_torch(geometry, geometry_scale)
            prediction = model(γ, θoffset)
            geometry_terms = geometry_loss(prediction, target)
            xy_mse = geometry_terms.xy
            scaled_angle_mse = geometry_terms.angle
            color_mse = geometry_terms.color
            color_accuracy = (prediction[..., 3].sign() == target[..., 3]).float().mean()
            loss = geometry_terms.total

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clip
            )
            optimizer.step()

            totals["loss"] += float(loss.detach()) * batch_size
            totals["xy_mse"] += float(xy_mse.detach()) * batch_size
            totals["scaled_angle_mse"] += float(scaled_angle_mse.detach()) * batch_size
            totals["color_mse"] += float(color_mse.detach()) * batch_size
            totals["color_accuracy"] += float(color_accuracy.detach()) * batch_size
            totals["grad_norm"] += float(grad_norm) * batch_size
            seen += batch_size
            global_step += 1

        metrics = {
            "average_training_loss": totals["loss"] / seen,
            "xy_mse": totals["xy_mse"] / seen,
            "scaled_angle_mse": totals["scaled_angle_mse"] / seen,
            "color_mse": totals["color_mse"] / seen,
            "color_accuracy": totals["color_accuracy"] / seen,
            "gradient_norm": totals["grad_norm"] / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if metrics["average_training_loss"] < best_primary_metric:
            best_primary_metric = metrics["average_training_loss"]
            best_epoch = epoch

        sample_γ = torch.rand((1, GAMMA_DIM), device=device, dtype=torch.float32, generator=sampling_generator)
        sample_θoffset = torch.rand(1, device=device, dtype=torch.float32, generator=sampling_generator)
        sample_target_θoffset = 0.5 if config.train.no_rotation else sample_θoffset
        svg_path = svg_dir / f"{identifier}_e{epoch:03d}.svg"
        save_model_comparison(model, sample_γ, sample_θoffset, config.data.kmax, geometry_scale, svg_path, target_θoffset=sample_target_θoffset)
        latent_values = unscale_target_torch(model.tile_latents.detach(), geometry_scale)
        latent_colors = (latent_values[:, 3] + 1) / 2
        save_geometry_svg(svg_dir / f"{identifier}_e{epoch:03d}_latents.svg", latent_values[:, :3].cpu().numpy(), latent_colors.cpu().numpy())
        scheduler.step()

        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "primary_metric": metrics["average_training_loss"],
            "best_epoch": best_epoch,
            "best_primary_metric": best_primary_metric,
            "config": config.to_dict(),
            "kmax": config.data.kmax,
            "num_tiles": tile_count(config.data.kmax),
            "tile_order": "family_pair_then_line_i_then_line_j",
            "xy_scale": geometry_scale,
            "angle_scale": ANGLE_SCALE,
            "color_encoding": (-1.0, 1.0),
            "gamma_dim": GAMMA_DIM,
            "condition_dim": CONDITION_DIM,
            "target_dim": TARGET_DIM,
            "tied_geometry_embedding": True,
            "θoffset_range": (0.0, 1.0),
            "θ0_range": (-θ0_HALF_RANGE, θ0_HALF_RANGE),
            "no_rotation": config.train.no_rotation,
            "identifier": identifier,
            "output_directory": str(run_dir),
            "wandb_run_id": getattr(wandb_run, "id", run_id),
            "rng": capture_rng(training_generator, sampling_generator),
        }
        last_checkpoint = checkpoint_dir / f"{identifier}_e{epoch:03d}.pt"
        atomic_save(payload, last_checkpoint)
        retain_newest_and_best(checkpoint_dir, identifier, epoch, best_epoch)

        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch)
        print(
            f"epoch={epoch} loss={metrics['average_training_loss']:.8f} "
            f"xy={metrics['xy_mse']:.8f} scaled_angle={metrics['scaled_angle_mse']:.8f} "
            f"color={metrics['color_mse']:.8f} color_acc={metrics['color_accuracy']:.4f} "
            f"lr={metrics['learning_rate']:.8g}"
        )

    if wandb_run is not None:
        wandb_run.finish()
    if last_checkpoint is None:
        raise ValueError(
            f"checkpoint already completed configured num_epochs={config.train.num_epochs}"
        )
    return last_checkpoint


def main() -> None:
    config, _ = load_config()
    path = train(config)
    print(path)


if __name__ == "__main__":
    main()
