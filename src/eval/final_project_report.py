import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ChessWindowDataset
from src.train.trainer import (
    ChessTrainingModel,
    collate_batch,
    get_device,
    move_batch_to_device,
    run_epoch,
)
from src.train.config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate train/val/test metrics for the final project demo.")
    parser.add_argument("--checkpoint", default="experiments/run2/model.pt")
    parser.add_argument(
        "--data-path",
        default="data/processed/games_sample_25000_stockfish_trainonly.parquet",
    )
    parser.add_argument("--output-dir", default="experiments/run2/final_demo")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def build_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**checkpoint["config"])
    model = ChessTrainingModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def evaluate_split(model, config, data_path, split, batch_size, device):
    dataset = ChessWindowDataset(
        data_path,
        split=split,
        window_size=config.window_size,
        stride=config.stride,
        min_window_length=config.min_window_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    metrics = run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        loss_weights=config.loss_weights,
        train=False,
    )
    metrics["split"] = split
    metrics["num_windows"] = len(dataset)
    return metrics


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, checkpoint = build_model(args.checkpoint)
    device = get_device()
    model = model.to(device)

    rows = []
    for split in ("train", "val", "test"):
        rows.append(
            evaluate_split(
                model=model,
                config=config,
                data_path=args.data_path,
                split=split,
                batch_size=args.batch_size,
                device=device,
            )
        )

    summary = pd.DataFrame(rows)[
        [
            "split",
            "num_windows",
            "loss_total",
            "rating_rmse",
            "expected_cp_rmse",
            "residual_cp_rmse",
            "residual_cp_rmse_clipped_300",
            "phase_residual_rmse",
        ]
    ]
    summary_path = output_dir / "train_val_test_metrics.parquet"
    summary.to_parquet(summary_path, index=False)

    markdown_lines = [
        "# Final Demo Metrics",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- data: `{args.data_path}`",
        "",
        "## Split Results",
        "",
        "| split | windows | loss_total | rating_rmse | expected_cp_rmse | residual_cp_rmse | residual_cp_rmse_clipped_300 | phase_residual_rmse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        markdown_lines.append(
            f"| {row.split} | {row.num_windows} | {row.loss_total:.3f} | {row.rating_rmse:.2f} | {row.expected_cp_rmse:.2f} | {row.residual_cp_rmse:.2f} | {row.residual_cp_rmse_clipped_300:.2f} | {row.phase_residual_rmse:.2f} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are leak-free metrics computed on the train-only relabeled 25k dataset.",
            "- The main project target is `phase_residual_rmse`.",
            "- The strongest baseline on validation is `hist_gbdt` with phase residual RMSE `49.63`.",
            "- The transformer validation phase residual RMSE should be compared against that baseline.",
        ]
    )

    markdown_path = output_dir / "final_demo_metrics.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    history_path = output_dir / "training_history.pt"
    torch.save(checkpoint.get("history", []), history_path)

    print("Saved split metrics:", summary_path)
    print("Saved markdown summary:", markdown_path)
    print("Saved training history:", history_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
