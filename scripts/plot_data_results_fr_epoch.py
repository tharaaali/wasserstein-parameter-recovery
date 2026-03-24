import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score, mean_squared_error
import re  # NEW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import evaluation as eval_utils

# ---------- CKPT HELPERS ----------

def list_checkpoint_epochs(chkpt_dir: Path) -> list[int]:
    """Return sorted list of epochs that have checkpoints."""
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    epochs: list[int] = []
    for p in chkpt_dir.glob("checkpoint_epoch_*.pth"):
        m = pattern.search(p.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)

def nearest_lower_epoch(target_epoch: int, available_epochs: list[int]) -> int | None:
    """Pick the closest epoch <= target_epoch. If none, return None."""
    candidates = [e for e in available_epochs if e <= target_epoch]
    return max(candidates) if candidates else None

def pick_epoch(results_dir: Path, metric: str | None, explicit_epoch: int | None) -> int:
    """
    Decide which epoch to use.
    - If explicit_epoch provided, use that (and validate it exists or find nearest lower).
    - Else read training_metrics.csv and pick best by metric ('r2' max, 'rmse' min).
    Then snap to nearest lower checkpoint epoch.
    """
    ckpt_dir = results_dir / "checkpoints"
    available = list_checkpoint_epochs(ckpt_dir)
    if not available:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    if explicit_epoch is not None:
        target = int(explicit_epoch)
    else:
        if metric is None:
            raise ValueError("Provide --epoch or --metric {r2, rmse}.")
        metrics_path = results_dir / "training_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        df = pd.read_csv(metrics_path)
        if metric == "r2":
            if "r2" not in df.columns:
                raise KeyError("Column 'r2' not found in training_metrics.csv")
            target = int(df.loc[df["r2"].idxmax(), "epoch"])
        elif metric == "rmse":
            if "rmse" not in df.columns:
                raise KeyError("Column 'rmse' not found in training_metrics.csv")
            target = int(df.loc[df["rmse"].idxmin(), "epoch"])
        else:
            raise ValueError("metric must be 'r2' or 'rmse'")

    chosen = nearest_lower_epoch(target, available)
    if chosen is None:
        # Tell the truth: we refuse to go forward or â€œuse lastâ€.
        raise FileNotFoundError(
            f"No checkpoint <= target epoch {target}. "
            f"Available epochs: {available}"
        )
    if chosen != target:
        print(f"[i] Target epoch {target} not saved; using nearest lower checkpoint {chosen}.")
    else:
        print(f"[i] Using exact checkpoint at epoch {chosen}.")
    return chosen

# ---------- EXISTING CODE ----------

def load_generator_weights(ckpt_path: Path) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("generator_state_dict", {})
    W = state.get("W")
    if W is None:
        raise KeyError(f"W not found in {ckpt_path}")
    return W.detach().cpu().numpy()

def load_df(df_path: Path) -> pd.DataFrame:
    assert df_path.is_file(), f"Provided dataset path does not exist: {df_path}"
    return pd.read_csv(df_path)

def plot_pred_vs_real(real: np.ndarray, pred: np.ndarray, epoch: int, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(real, pred, s=4, alpha=0.5)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'r--')
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path/ f"pred_vs_real_epoch_{epoch}.png", bbox_inches="tight")
    plt.savefig(out_path/ f"pred_vs_real_epoch_{epoch}.pdf", bbox_inches="tight")
    plt.close()
    
    plt.scatter(real, pred, alpha=0.7, edgecolor='black')
    p1 = max(max(pred), max(real))
    p2 = min(min(pred), min(real))
    plt.plot([p1, p2], [p1, p2], 'b-',label='y=x')
    plt.title('True vs Predicted Aging Factors')
    plt.xlabel('True Aging Factors')
    plt.ylabel('Predicted Aging Factors')
    plt.savefig(out_path/f"true_vs_predicted_recreated_{epoch}.png", format='png', dpi=300)
    plt.legend()
    plt.close()

def plot_residuals(real: np.ndarray, pred: np.ndarray,epoch: int, out_path: Path, title: str) -> None:
    residuals = real - pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=100, alpha=0.7)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals (True - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(out_path / f"residuals_epoch_{epoch}.png", bbox_inches="tight")
    plt.savefig(out_path / f"residuals_epoch_{epoch}.pdf", bbox_inches="tight")
    plt.close()

def plot_r2(fig_dir: Path, title: str, r2: float, rmse: float) -> None:
    plt.figure()
    plt.text(0.5, 0.5, f"R2 Score: {r2:.4f}\nRMSE: {rmse:.4f}", fontsize=15,
             ha='center', va='center', bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_dir / "r2_score_box.png")
    plt.close()

def plot_energy_sum_distributions(
    df_events_new: pd.DataFrame,
    df_events_old: pd.DataFrame,
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    out_path: Path,
    epoch: int
) -> None:
    bins = 35
    ran = [0, 100]

    plt.figure(figsize=(8, 6))
    plt.hist(df_events_new.Esum / 1000, bins=bins,
             label=f"Undamaged ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.Esum_aged / 1000, bins=bins,
             label=f"Damaged ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.E_sum_aged_pred / 1000, bins=bins, 
             label=f"Calibrated ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "EnergySumComparison_Undamaged_Damaged_Calibrated.png", bbox_inches="tight")
    plt.close()

    plt.hist(df_events_new.Esum / 1000, bins=bins, range= ran,
             label=f"Undamaged ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.Esum_aged / 1000, bins=bins, range= ran,
             label=f"Damaged ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.E_sum_aged_pred / 1000, bins=bins, range= ran,
             label=f"Calibrated ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / f"EnergySumComparison_Undamaged_Damaged_Calibrated_Zoomed_{epoch}.png", bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.hist(df_events_new.Esum / 1000, bins=bins, 
             label=f"Undamaged ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.E_sum_aged_pred / 1000, bins=bins,  
             label=f"Calibrated ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution - Undamaged vs Calibrated')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "EnergySumComparison-Undamaged-Calibrate.png", bbox_inches="tight")
    plt.close()

    plt.hist(df_events_new.Esum / 1000, bins=bins,range= ran,
             label=f"Undamaged ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.E_sum_aged_pred / 1000, bins=bins, range= ran,
             label=f"Calibrated ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution - Undamaged vs Calibrated')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / f"EnergySumComparison-Undamaged-Calibrate_Zoomed_{epoch}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(df_events_new.Esum / 1000, bins=bins,
             label=f"Undamaged group new ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.Esum / 1000, bins=bins,
             label=f"Undamaged group old ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution - Undamaged vs Damaged')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "EnergySumComparison-new-orrect_old.png", bbox_inches="tight")
    plt.close()
    
    plt.hist(df_events_new.Esum / 1000, bins=bins,range= ran,
             label=f"Undamaged ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.E_sum_aged_pred / 1000, bins=bins, range= ran,
             label=f"Calibrated ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution - Undamaged vs Calibrated')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / f"EnergySumComparison-Undamaged-Calibrate_Zoomed_{epoch}.png", bbox_inches="tight")
    plt.close()

    plt.hist(df_events_new.Esum / 1000, bins=bins,range= ran,
             label=f"Undamaged ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.Esum_aged / 1000, bins=bins, range= ran,
             label=f"Damaged ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution - Undamaged vs Damaged')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / f"EnergySumComparison-Undamaged-Damaged_Zoomed_{epoch}.png", bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.hist(df_events_new.Esum_aged / 1000, bins=bins,
             label=f"damaged group new ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.Esum_aged / 1000, bins=bins,
             label=f"damaged group old ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution - Damaged new vs Damaged old')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "EnergySumComparison-damaged-new-old.png", bbox_inches="tight")
    plt.close()
    
    plt.hist(df_events_new.Esum_aged / 1000, bins=bins,range= ran,
             label=f"damaged group new ({df_new['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old.Esum_aged / 1000, bins=bins,range= ran,
             label=f"damaged group old ({df_old['event'].nunique()} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution - Damaged new vs Damaged old')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / f"EnergySumComparison-damaged-new-old_Zoomed_{epoch}.png", bbox_inches="tight")
    plt.close()

    # Filter events with >3 unique cellids
    valid_events_old = (
        df_old.groupby('event')['cellid']
        .nunique()
        .loc[lambda x: x > 3]
        .index
    )
    valid_events_new = (
        df_new.groupby('event')['cellid']
        .nunique()
        .loc[lambda x: x > 3]
        .index
    )

    df_events_old_filtered = df_events_old[df_events_old['event'].isin(valid_events_old)]
    df_events_new_filtered = df_events_new[df_events_new['event'].isin(valid_events_new)]
    
    plt.figure(figsize=(8, 6))
    plt.hist(df_events_new_filtered.Esum / 1000, bins=bins,
             label=f"Undamaged (>3 cells, {len(valid_events_new)} events)",
             histtype='step', linewidth=2)
    plt.hist(df_events_old_filtered.E_sum_aged_pred / 1000, bins=bins,
             label=f"Calibrated (>3 cells, {len(valid_events_old)} events)",
             histtype='step', linewidth=2)
    plt.title('Energy Sum Distribution (Only Events with > 3 Active Cells)')
    plt.xlabel('$E_{sum}$ [GeV]', fontsize=12)
    plt.ylabel('Entries / bin', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "EnergySumComparison_OnlyEvents_GT3Cells.png", bbox_inches="tight")
    plt.close()

def plot_sparse_events_per_layer(df: pd.DataFrame, max_cells: int = 3, out_path: Path = None, tag: str = ""):
    group = df.groupby(['event', 'z'])['cellid'].nunique().reset_index(name='active_cells')
    sparse_counts = group[group['active_cells'] <= max_cells].groupby('z')['event'].nunique()
    plt.figure(figsize=(10, 5))
    plt.bar(sparse_counts.index, sparse_counts.values)
    plt.xlabel('Layer z')
    plt.ylabel(f'Number of sparse events (â‰¤ {max_cells} cells)')
    plt.title(f'Sparse Event Counts per Layer (â‰¤ {max_cells} active cells)')
    plt.tight_layout()
    if out_path:
        filename = f"sparse_events_per_layer_{tag}_max{max_cells}.png"
        plt.savefig(out_path / filename, bbox_inches="tight")
    plt.show()
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generator weights from specific epoch")
    parser.add_argument("results_dir", type=Path, help="Experiment results directory (under results/)")
    parser.add_argument("--df", type=Path, default=None, help="Optional: custom path to training DataFrame (CSV)")

    group = parser.add_mutually_exclusive_group(required=False)  # CHANGED
    group.add_argument("--epoch", type=int, help="Epoch number to load generator checkpoint")
    group.add_argument("--metric", choices=["r2", "rmse"],
                       help="Pick best epoch by metric from training_metrics.csv (r2=max, rmse=min)")

    args = parser.parse_args()

    results_dir = args.results_dir
    fig_dir = results_dir / "figs"
    logs_dir= results_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolve epoch (explicit or from metrics) and snap to nearest lower checkpoint
    chosen_epoch = eval_utils.pick_epoch(results_dir, metric=args.metric, explicit_epoch=args.epoch)

    ckpt_path = results_dir / "checkpoints" / f"checkpoint_epoch_{chosen_epoch}.pth"
    if not ckpt_path.exists():
        # Should not happen, pick_epoch already validated.
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    df_new_path = logs_dir / "train_prepared_new.zip"
    df_old_path = logs_dir / "train_prepared_old.zip"
    if not df_old_path.exists():
        raise FileNotFoundError(f"Old dataset not found: {df_old_path}")
    if not df_new_path.exists():
        raise FileNotFoundError(f"New dataset not found: {df_new_path}")

    df_old = eval_utils.load_df(df_old_path)
    df_new = eval_utils.load_df(df_new_path)

    # Normalize column names to the expected ones
    df_new = df_new.rename(columns={'E_cal': 'E_cal_norm','E_aged_cal': 'E_aged_norm','E_cal_org': 'E_cal'})
    df_old = df_old.rename(columns={'E_cal': 'E_cal_norm','E_aged_cal': 'E_aged_norm','E_cal_org': 'E_cal'})
    df_new['E_aged_cal'] = df_new['E_cal'] * df_new['aging_factor']
    df_old['E_aged_cal'] = df_old['E_cal'] * df_old['aging_factor']

    g_weights = eval_utils.load_generator_weights(ckpt_path)
    df_old["pred_aging_factor"] = eval_utils.predict_aging_factors(df_old, g_weights)

    results_df = eval_utils.build_cell_results(df_old)
    true_value = results_df["aging_factor"].to_numpy()
    predicted_value = results_df["pred_aging_factor"].to_numpy()

    r2, rmse = eval_utils.compute_metrics(true_value, predicted_value)

    eval_utils.plot_pred_vs_real(
        true_value,
        predicted_value,
        fig_dir,
        f"Predicted vs Real (Epoch {chosen_epoch})",
        stem=f"pred_vs_real_epoch_{chosen_epoch}",
        legacy_stem=f"true_vs_predicted_recreated_{chosen_epoch}",
    )
    eval_utils.plot_residuals(
        true_value,
        predicted_value,
        fig_dir,
        f"Residuals (Epoch {chosen_epoch})",
        stem=f"residuals_epoch_{chosen_epoch}",
    )
    eval_utils.plot_metric_summary(fig_dir, f"Model Summary (Epoch {chosen_epoch})", r2, rmse)
    
    df_old['E_cal_pred'] = df_old['E_aged_cal'] / df_old['pred_aging_factor']

    # --------- AGGREGATION ---------
    df_events_old = df_old.groupby('event').agg(
        Esum=('E_cal', 'sum'),
        Esum_aged=('E_aged_cal', 'sum'),
        E_sum_aged_pred=('E_cal_pred', 'sum')
    ).reset_index()
    df_events_new = df_new.groupby('event').agg(
        Esum=('E_cal', 'sum'),
        Esum_aged=('E_aged_cal', 'sum')
    ).reset_index()

    plot_energy_sum_distributions(
        df_events_new=df_events_new,
        df_events_old=df_events_old,
        df_old=df_old,
        df_new=df_new,
        out_path=fig_dir,
        epoch=chosen_epoch
    )
    plot_sparse_events_per_layer(df_old, max_cells=3, out_path=fig_dir, tag="old")
    plot_sparse_events_per_layer(df_new, max_cells=3, out_path=fig_dir, tag="new")

    # Console output
    print(f"Epoch {chosen_epoch} evaluation")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")

if __name__ == "__main__":
    main()


