#!/usr/bin/env python
"""
Train the CaloGAN WGAN on prepared calorimeter datasets.

This entrypoint is intentionally limited to the WGAN workflow:
- it consumes already-prepared fresh and aged event datasets
- it optionally evaluates against externally-provided aging targets
- it does not generate synthetic aging factors inside the repository
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from comet_ml import Experiment
except ImportError:  # pragma: no cover
    Experiment = None

from utils.wgan_core import CaloDataset, Discriminator, Generator, save_checkpoint
from utils.wgan_data import (
    build_event_tensor_samples,
    build_target_coords,
    load_prepared_training_data,
    load_target_aging_tensor,
    trim_events_for_debug,
)
from utils.wgan_metrics import compute_alignment_metrics, save_alignment_plots


@dataclass(slots=True)
class IOConfig:
    fresh_data_path: str = "datasets/train_prepared_new.zip"
    aged_data_path: str = "datasets/train_prepared_old.zip"
    target_aging_path: str | None = None


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 512
    n_epochs: int = 100
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    save_interval: int = 5
    seed: int = 42
    use_mask: bool = True
    make_old_to_new: bool = False
    precision: str = "fp32"
    metrics_interval: int = 20

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)


@dataclass(slots=True)
class DataConfig:
    event_column: str = "event"
    x_column: str = "x"
    y_column: str = "y"
    z_column: str = "z"
    fresh_energy_column: str = "E_cal"
    aged_energy_column: str = "E_aged_cal"
    target_value_column: str = "aging_factor"
    shape_z: int = 40
    shape_x: int = 24
    shape_y: int = 24

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return (self.shape_z, self.shape_x, self.shape_y)


@dataclass(slots=True)
class CometConfig:
    enabled: bool = False
    api_key: str | None = None
    project_name: str | None = None
    workspace: str | None = None


@dataclass(slots=True)
class MiscConfig:
    debug: bool = False
    debug_max_events: int = 128


def _load_section(dataclass_type: type[Any], raw: dict[str, Any]) -> Any:
    allowed = {field_info.name for field_info in fields(dataclass_type)}
    payload = {key: value for key, value in raw.items() if key in allowed}
    return dataclass_type(**payload)


@dataclass(slots=True)
class Config:
    io: IOConfig = field(default_factory=IOConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    comet: CometConfig = field(default_factory=CometConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        return cls(
            io=_load_section(IOConfig, raw.get("io", {})),
            train=_load_section(TrainingConfig, raw.get("train", {})),
            data=_load_section(DataConfig, raw.get("data", {})),
            comet=_load_section(CometConfig, raw.get("comet", {})),
            misc=_load_section(MiscConfig, raw.get("misc", {})),
        )

    def dump(self, dst: str | Path) -> None:
        with open(dst, "w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(self), handle, sort_keys=False)


def create_experiment_folder(base_dir: str = "results", exp_name: str = "train") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_dir) / f"{timestamp}_{exp_name}"
    full_path.mkdir(parents=True, exist_ok=False)
    print(f"[info] Experiment results folder: {full_path.resolve()}")
    return full_path


def prepare_directories(config_name: str, res_dir: str = "results") -> tuple[Path, Path, Path, Path, Path]:
    config_stem = Path(config_name).stem
    exp_dir = create_experiment_folder(res_dir, config_stem)
    indices_dir = exp_dir / "indices"
    checkpoints_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figs"
    log_dir = exp_dir / "logs"
    for directory in (indices_dir, checkpoints_dir, fig_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return exp_dir, indices_dir, checkpoints_dir, fig_dir, log_dir


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clamp_generator_weights(generator: Generator) -> None:
    with torch.no_grad():
        generator.W.clamp_(0.0, 1.0)


def write_data_manifest(cfg: Config, log_dir: Path) -> None:
    manifest = {
        "fresh_data_path": str(Path(cfg.io.fresh_data_path).resolve()),
        "aged_data_path": str(Path(cfg.io.aged_data_path).resolve()),
        "target_aging_path": (
            str(Path(cfg.io.target_aging_path).resolve()) if cfg.io.target_aging_path else None
        ),
    }
    with open(log_dir / "data_sources.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)


def train(cfg: Config, cfg_file: str) -> None:
    cfg_name = Path(cfg_file).name
    exp_dir, _, checkpoints_dir, fig_dir, log_dir = prepare_directories(cfg_name, res_dir="results")
    cfg.dump(exp_dir / "resolved_config.yaml")
    write_data_manifest(cfg, log_dir)

    seed_everything(cfg.train.seed)
    device = resolve_device()
    print(f"[info] Device: {device}")

    precision = str(cfg.train.precision).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(
            f"Unsupported train.precision={cfg.train.precision!r}. Expected 'fp32' or 'bf16'."
        )

    use_bf16 = False
    if precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        use_bf16 = True
        print("[info] Precision mode: bf16 autocast")
    else:
        print("[info] Precision mode: fp32")

    def amp_autocast():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    df_new, df_old = load_prepared_training_data(
        cfg.io.fresh_data_path,
        cfg.io.aged_data_path,
        event_column=cfg.data.event_column,
        x_column=cfg.data.x_column,
        y_column=cfg.data.y_column,
        z_column=cfg.data.z_column,
        fresh_energy_column=cfg.data.fresh_energy_column,
        aged_energy_column=cfg.data.aged_energy_column,
    )

    if cfg.misc.debug:
        df_new = trim_events_for_debug(
            df_new,
            event_column=cfg.data.event_column,
            max_events=cfg.misc.debug_max_events,
        )
        df_old = trim_events_for_debug(
            df_old,
            event_column=cfg.data.event_column,
            max_events=cfg.misc.debug_max_events,
        )

    coords = build_target_coords(
        df_new,
        df_old,
        x_column=cfg.data.x_column,
        y_column=cfg.data.y_column,
        z_column=cfg.data.z_column,
    )
    target_tensor = load_target_aging_tensor(
        cfg.io.target_aging_path,
        shape=cfg.data.tensor_shape,
        x_column=cfg.data.x_column,
        y_column=cfg.data.y_column,
        z_column=cfg.data.z_column,
        value_column=cfg.data.target_value_column,
    )

    samples_new = build_event_tensor_samples(
        df_new,
        event_column=cfg.data.event_column,
        x_column=cfg.data.x_column,
        y_column=cfg.data.y_column,
        z_column=cfg.data.z_column,
        energy_column=cfg.data.fresh_energy_column,
        sample_key="calorimeterCellsNew",
        shape=cfg.data.tensor_shape,
    )
    samples_old = build_event_tensor_samples(
        df_old,
        event_column=cfg.data.event_column,
        x_column=cfg.data.x_column,
        y_column=cfg.data.y_column,
        z_column=cfg.data.z_column,
        energy_column=cfg.data.aged_energy_column,
        sample_key="calorimeterCellsOld",
        shape=cfg.data.tensor_shape,
    )

    if not samples_new or not samples_old:
        raise ValueError("Prepared datasets produced no event samples")

    g = Generator(cfg.data.tensor_shape).to(device)
    d = Discriminator(in_dim=cfg.data.shape_z).to(device)
    g_opt = torch.optim.Adam(g.parameters(), lr=cfg.train.lr, betas=cfg.train.betas)
    d_opt = torch.optim.Adam(d.parameters(), lr=cfg.train.lr, betas=cfg.train.betas)

    experiment = None
    if cfg.comet.enabled and Experiment is not None and cfg.comet.api_key:
        experiment = Experiment(
            api_key=cfg.comet.api_key,
            project_name=cfg.comet.project_name,
            workspace=cfg.comet.workspace,
            auto_output_logging="simple",
            log_code=True,
        )
        experiment.log_parameters(asdict(cfg))

    metrics_log: list[dict[str, float | int]] = []
    step_cnt = 0
    initial_dl_new = DataLoader(CaloDataset(samples_new), batch_size=cfg.train.batch_size, shuffle=True)
    initial_dl_old = DataLoader(CaloDataset(samples_old), batch_size=cfg.train.batch_size, shuffle=True)
    len_iter = min(len(initial_dl_new), len(initial_dl_old))
    if len_iter == 0:
        raise ValueError("Prepared datasets produced zero training batches")

    with tqdm(total=cfg.train.n_epochs * len_iter, colour="#36b24e") as pbar:
        for epoch in range(cfg.train.n_epochs):
            dl_new = DataLoader(CaloDataset(samples_new), batch_size=cfg.train.batch_size, shuffle=True)
            dl_old = DataLoader(CaloDataset(samples_old), batch_size=cfg.train.batch_size, shuffle=True)

            for batch_new, batch_old in zip(dl_new, dl_old):
                e_new = batch_new["calorimeterCellsNew"].to(device)
                e_old = batch_old["calorimeterCellsOld"].to(device)

                if e_new.size(0) != e_old.size(0):
                    min_bsz = min(e_new.size(0), e_old.size(0))
                    e_new = e_new[:min_bsz]
                    e_old = e_old[:min_bsz]

                if cfg.train.use_mask:
                    mask = (e_new != 0) & (e_old != 0)
                    e_new = e_new * mask
                    e_old = e_old * mask

                d_opt.zero_grad(set_to_none=True)
                if cfg.train.make_old_to_new:
                    with amp_autocast():
                        e_new_fake = g(e_old, aged=True).detach()
                        d_loss = -torch.mean(d(e_new)) + torch.mean(d(e_new_fake))
                    d_loss.backward()
                    d_opt.step()
                    for parameter in d.parameters():
                        parameter.data.clamp_(-0.01, 0.01)

                    g_opt.zero_grad(set_to_none=True)
                    with amp_autocast():
                        g_loss = -torch.mean(d(g(e_old, aged=True)))
                    g_loss.backward()
                    g_opt.step()
                else:
                    with amp_autocast():
                        e_old_fake = g(e_new, aged=False).detach()
                        d_loss = -torch.mean(d(e_old)) + torch.mean(d(e_old_fake))
                    d_loss.backward()
                    d_opt.step()
                    for parameter in d.parameters():
                        parameter.data.clamp_(-0.01, 0.01)

                    g_opt.zero_grad(set_to_none=True)
                    with amp_autocast():
                        g_loss = -torch.mean(d(g(e_new, aged=False)))
                    g_loss.backward()
                    g_opt.step()

                clamp_generator_weights(g)

                if experiment is not None:
                    experiment.log_metric("d_loss", d_loss.item(), step=step_cnt, epoch=epoch)
                    experiment.log_metric("g_loss", g_loss.item(), step=step_cnt, epoch=epoch)

                if target_tensor is not None and step_cnt % cfg.train.metrics_interval == 0:
                    metrics = compute_alignment_metrics(g.W.detach(), target_tensor, coords)
                    if experiment is not None:
                        experiment.log_metric("RMSE", metrics["rmse"], step=step_cnt, epoch=epoch)
                        experiment.log_metric("MAE_aging_factor", metrics["mae"], step=step_cnt, epoch=epoch)
                        experiment.log_metric("R^2", metrics["r2"], step=step_cnt, epoch=epoch)

                    metrics_log.append(
                        {
                            "epoch": epoch,
                            "step": step_cnt,
                            "rmse": metrics["rmse"],
                            "mae": metrics["mae"],
                            "r2": metrics["r2"],
                        }
                    )

                step_cnt += 1
                pbar.update(1)

            if epoch % cfg.train.save_interval == 0 or epoch == cfg.train.n_epochs - 1:
                save_checkpoint(epoch, step_cnt, g, d, g_opt, d_opt, checkpoints_dir)
                if experiment is not None:
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmpfile:
                        torch.save(g.state_dict(), tmpfile.name)
                        experiment.log_model(
                            name=f"generator_epoch_{epoch}",
                            file_or_folder=tmpfile.name,
                            overwrite=True,
                        )
                        os.remove(tmpfile.name)

    learned_aging_factors = g.W.detach().cpu()
    torch.save(learned_aging_factors, exp_dir / "learned_aging_factors.pt")
    np.save(exp_dir / "learned_aging_factors.npy", learned_aging_factors.numpy())
    print(f"[info] Saved learned aging factors to {exp_dir}")

    if metrics_log:
        metrics_df = pd.DataFrame(metrics_log)
        metrics_df.to_csv(exp_dir / "training_metrics.csv", index=False)

    if target_tensor is not None:
        final_metrics = save_alignment_plots(learned_aging_factors, target_tensor, coords, fig_dir)
        print(
            "[info] Final alignment metrics: "
            f"RMSE={final_metrics['rmse']:.5f}, "
            f"MAE={final_metrics['mae']:.5f}, "
            f"R2={final_metrics['r2']:.5f}"
        )
        if experiment is not None:
            experiment.log_metrics(
                {
                    "final_rmse": final_metrics["rmse"],
                    "final_mae": final_metrics["mae"],
                    "final_r2": final_metrics["r2"],
                }
            )

    if experiment is not None:
        experiment.end()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CaloGAN WGAN from prepared datasets")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (see config/train.yaml)",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config.from_yaml(args.config)
    train(cfg, args.config)


if __name__ == "__main__":
    main()
