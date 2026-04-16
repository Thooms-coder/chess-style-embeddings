import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize residual weakness patterns by rating bucket and phase.")
    parser.add_argument(
        "--input-file",
        default="data/processed/games_sample_200_stockfish.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/run1/plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_file)
    rows = []
    for row in df.itertuples(index=False):
        for bucket, phase, residual, valid in zip(
            row.move_rating_bucket,
            row.move_phase_name,
            row.move_residual_cp_loss,
            row.move_engine_valid,
        ):
            if not valid or residual is None or pd.isna(residual):
                continue
            rows.append(
                {
                    "rating_bucket": bucket,
                    "phase_name": phase,
                    "residual_cp_loss": float(residual),
                }
            )

    flat_df = pd.DataFrame(rows)
    summary = (
        flat_df.groupby(["rating_bucket", "phase_name"])["residual_cp_loss"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary.to_parquet(output_dir / "residual_summary.parquet", index=False)

    heatmap_df = summary.pivot(index="rating_bucket", columns="phase_name", values="mean").sort_index()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="coolwarm", center=0.0)
    plt.title("Mean Residual CP Loss by Rating Bucket and Phase")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_heatmap.png", dpi=200)
    plt.close()

    print("Saved summary:", output_dir / "residual_summary.parquet")
    print("Saved plot:", output_dir / "residual_heatmap.png")
    print("Rows summarized:", len(flat_df))


if __name__ == "__main__":
    main()
