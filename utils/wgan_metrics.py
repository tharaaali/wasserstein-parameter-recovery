from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .evaluation import plot_metric_summary, plot_pred_vs_real, plot_residuals


def _as_numpy(weights: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(weights, torch.Tensor):
        return weights.detach().cpu().numpy()
    return np.asarray(weights)


def extract_alignment_arrays(
    generator_weights: torch.Tensor | np.ndarray,
    target_tensor: torch.Tensor,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if coords.size == 0:
        raise ValueError("No coordinates available for alignment metrics")

    predicted = _as_numpy(generator_weights)
    target = target_tensor.detach().cpu().numpy()

    z_idx = coords[:, 0].astype(int)
    x_idx = coords[:, 1].astype(int)
    y_idx = coords[:, 2].astype(int)

    real_values = target[z_idx, x_idx, y_idx].flatten()
    predicted_values = predicted[z_idx, x_idx, y_idx].flatten()
    return real_values, predicted_values


def compute_alignment_metrics(
    generator_weights: torch.Tensor | np.ndarray,
    target_tensor: torch.Tensor,
    coords: np.ndarray,
) -> dict[str, float]:
    real_values, predicted_values = extract_alignment_arrays(generator_weights, target_tensor, coords)
    return {
        "rmse": float(np.sqrt(mean_squared_error(real_values, predicted_values))),
        "mae": float(mean_absolute_error(real_values, predicted_values)),
        "r2": float(r2_score(real_values, predicted_values)),
    }


def save_alignment_plots(
    generator_weights: torch.Tensor | np.ndarray,
    target_tensor: torch.Tensor,
    coords: np.ndarray,
    fig_dir: str | Path,
) -> dict[str, float]:
    fig_path = Path(fig_dir)
    real_values, predicted_values = extract_alignment_arrays(generator_weights, target_tensor, coords)
    metrics = compute_alignment_metrics(generator_weights, target_tensor, coords)

    plot_pred_vs_real(
        real_values,
        predicted_values,
        fig_path,
        "Predicted vs Real aging factors",
        stem="predicted_vs_real",
    )
    plot_residuals(
        real_values,
        predicted_values,
        fig_path,
        "Histogram of residuals (Real - Predicted)",
        stem="residuals_hist",
    )
    plot_metric_summary(
        fig_path,
        "Final aging-factor alignment",
        metrics["r2"],
        metrics["rmse"],
        stem="final_alignment_metrics",
    )
    return metrics
