from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score


def load_generator_weights(ckpt_path: Path) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("generator_state_dict", {})
    weights = state.get("W")
    if weights is None:
        raise KeyError(f"W not found in {ckpt_path}")
    return weights.detach().cpu().numpy()


def load_df(df_path: Path) -> pd.DataFrame:
    if not df_path.is_file():
        raise FileNotFoundError(f"Provided dataset path does not exist: {df_path}")
    return pd.read_csv(df_path)


def list_checkpoint_epochs(chkpt_dir: Path) -> list[int]:
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    epochs: list[int] = []
    for path in chkpt_dir.glob("checkpoint_epoch_*.pth"):
        match = pattern.search(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def nearest_lower_epoch(target_epoch: int, available_epochs: list[int]) -> int | None:
    candidates = [epoch for epoch in available_epochs if epoch <= target_epoch]
    return max(candidates) if candidates else None


def pick_epoch(results_dir: Path, metric: str | None, explicit_epoch: int | None) -> int:
    ckpt_dir = results_dir / "checkpoints"
    available_epochs = list_checkpoint_epochs(ckpt_dir)
    if not available_epochs:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    if explicit_epoch is not None:
        target_epoch = int(explicit_epoch)
    else:
        if metric is None:
            raise ValueError("Provide --epoch or --metric {r2, rmse}.")
        metrics_path = results_dir / "training_metrics.csv"
        metrics_df = load_df(metrics_path)
        if metric == "r2":
            if "r2" not in metrics_df.columns:
                raise KeyError("Column 'r2' not found in training_metrics.csv")
            target_epoch = int(metrics_df.loc[metrics_df["r2"].idxmax(), "epoch"])
        elif metric == "rmse":
            if "rmse" not in metrics_df.columns:
                raise KeyError("Column 'rmse' not found in training_metrics.csv")
            target_epoch = int(metrics_df.loc[metrics_df["rmse"].idxmin(), "epoch"])
        else:
            raise ValueError("metric must be 'r2' or 'rmse'")

    chosen_epoch = nearest_lower_epoch(target_epoch, available_epochs)
    if chosen_epoch is None:
        raise FileNotFoundError(
            f"No checkpoint <= target epoch {target_epoch}. "
            f"Available epochs: {available_epochs}"
        )

    if chosen_epoch != target_epoch:
        print(
            f"[info] Target epoch {target_epoch} was not saved; "
            f"using nearest lower checkpoint {chosen_epoch}."
        )
    return chosen_epoch


def predict_aging_factors(
    df: pd.DataFrame,
    generator_weights: np.ndarray,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
) -> np.ndarray:
    return np.asarray(
        [generator_weights[z, x, y] for x, y, z in zip(df[x_col], df[y_col], df[z_col])]
    )


def build_cell_results(
    df: pd.DataFrame,
    cellid_col: str = "cellid",
    real_col: str = "aging_factor",
    pred_col: str = "pred_aging_factor",
) -> pd.DataFrame:
    return df[[cellid_col, real_col, pred_col]].drop_duplicates(subset=[cellid_col]).copy()


def compute_metrics(real: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    r2 = float(r2_score(real, pred))
    rmse = float(np.sqrt(mean_squared_error(real, pred)))
    return r2, rmse


def plot_pred_vs_real(
    real: np.ndarray,
    pred: np.ndarray,
    out_dir: Path,
    title: str,
    stem: str,
    legacy_stem: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(real, pred, s=4, alpha=0.5)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], "r--")
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close()

    if legacy_stem is None:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(real, pred, alpha=0.7, edgecolor="black")
    upper = max(float(np.max(pred)), float(np.max(real)))
    lower = min(float(np.min(pred)), float(np.min(real)))
    plt.plot([upper, lower], [upper, lower], "b-", label="y=x")
    plt.title("True vs Predicted Aging Factors")
    plt.xlabel("True Aging Factors")
    plt.ylabel("Predicted Aging Factors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{legacy_stem}.png", format="png", dpi=300)
    plt.savefig(out_dir / f"{legacy_stem}.pdf", format="pdf", dpi=300)
    plt.close()


def plot_residuals(
    real: np.ndarray,
    pred: np.ndarray,
    out_dir: Path,
    title: str,
    stem: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    residuals = real - pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=100, alpha=0.7)
    plt.title(title)
    plt.xlabel("Residuals (True - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close()


def plot_metric_summary(
    out_dir: Path,
    title: str,
    r2: float,
    rmse: float,
    stem: str = "r2_score_box",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.text(
        0.5,
        0.5,
        f"R2 Score: {r2:.4f}\nRMSE: {rmse:.4f}",
        fontsize=15,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
    )
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}.png")
    plt.savefig(out_dir / f"{stem}.pdf")
    plt.close()
