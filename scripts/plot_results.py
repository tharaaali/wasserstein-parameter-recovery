import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluation import (
    build_cell_results,
    compute_metrics,
    list_checkpoint_epochs,
    load_df,
    load_generator_weights,
    nearest_lower_epoch,
    plot_metric_summary,
    plot_pred_vs_real,
    plot_residuals,
    predict_aging_factors,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predictions vs real values for CaloGAN")
    parser.add_argument("results_dir", type=Path, help="Path to a single experiment directory under results/")
    parser.add_argument("--df", type=Path, required=True, help="Path to the saved training DataFrame (CSV)")

    args = parser.parse_args()

    results_dir = args.results_dir
    metrics_path = results_dir / "training_metrics.csv"
    ckpt_dir = results_dir / "checkpoints"
    fig_dir = results_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_df(metrics_path)
    if "epoch" not in metrics_df.columns or "rmse" not in metrics_df.columns or "r2" not in metrics_df.columns:
        raise KeyError("training_metrics.csv must contain 'epoch', 'rmse', and 'r2' columns")

    df_old = load_df(args.df)
    available_epochs = list_checkpoint_epochs(ckpt_dir)

    best_rmse_row = metrics_df.loc[metrics_df["rmse"].idxmin()]
    best_r2_row = metrics_df.loc[metrics_df["r2"].idxmax()]

    for suffix, row in {"best_rmse": best_rmse_row, "best_r2": best_r2_row}.items():
        target_epoch = int(row["epoch"])
        epoch = nearest_lower_epoch(target_epoch, available_epochs)
        if epoch is None:
            print(f"No checkpoint found at or before epoch {target_epoch}, skipping {suffix}")
            continue
        if epoch != target_epoch:
            print(f"Target epoch {target_epoch} was not saved; using checkpoint {epoch} for {suffix}")
        ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"

        df_eval = df_old.copy()
        g_weights = load_generator_weights(ckpt_path)
        df_eval["pred_aging_factor"] = predict_aging_factors(df_eval, g_weights)

        results_df = build_cell_results(df_eval)
        true_value = results_df["aging_factor"].to_numpy()
        predicted_value = results_df["pred_aging_factor"].to_numpy()

        r2, rmse = compute_metrics(true_value, predicted_value)

        plot_pred_vs_real(
            true_value,
            predicted_value,
            fig_dir,
            f"Predicted vs Real ({suffix})",
            stem=f"pred_vs_real_{suffix}",
        )
        plot_residuals(
            true_value,
            predicted_value,
            fig_dir,
            f"Residuals ({suffix})",
            stem=f"residuals_{suffix}",
        )
        plot_metric_summary(
            fig_dir,
            f"Model Evaluation Summary ({suffix})",
            r2,
            rmse,
            stem=f"r2_score_box_{suffix}",
        )

        print(f"[{suffix}] epoch={epoch}")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
