import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluation import (
    build_cell_results,
    compute_metrics,
    load_df,
    load_generator_weights,
    plot_metric_summary,
    plot_pred_vs_real,
    plot_residuals,
    predict_aging_factors,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generator weights from a specific epoch")
    parser.add_argument("results_dir", type=Path, help="Experiment results directory under results/")
    parser.add_argument("--df", type=Path, default=None, help="Optional path to the training DataFrame (CSV)")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load")

    args = parser.parse_args()

    results_dir = args.results_dir
    fig_dir = results_dir / "figs"
    logs_dir = results_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = results_dir / "checkpoints" / f"checkpoint_epoch_{args.epoch}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    df_path = args.df if args.df is not None else logs_dir / "train_prepared.zip"
    if not df_path.exists():
        raise FileNotFoundError(f"Dataset not found: {df_path}")

    df_old = load_df(df_path)
    g_weights = load_generator_weights(ckpt_path)
    df_old["pred_aging_factor"] = predict_aging_factors(df_old, g_weights)

    results_df = build_cell_results(df_old)
    true_value = results_df["aging_factor"].to_numpy()
    predicted_value = results_df["pred_aging_factor"].to_numpy()

    r2, rmse = compute_metrics(true_value, predicted_value)

    plot_pred_vs_real(
        true_value,
        predicted_value,
        fig_dir,
        f"Predicted vs Real (Epoch {args.epoch})",
        stem=f"pred_vs_real_epoch_{args.epoch}",
        legacy_stem=f"true_vs_predicted_recreated_{args.epoch}",
    )
    plot_residuals(
        true_value,
        predicted_value,
        fig_dir,
        f"Residuals (Epoch {args.epoch})",
        stem=f"residuals_epoch_{args.epoch}",
    )
    plot_metric_summary(fig_dir, f"Model Summary (Epoch {args.epoch})", r2, rmse)

    print(f"Epoch {args.epoch} evaluation")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")


if __name__ == "__main__":
    main()
