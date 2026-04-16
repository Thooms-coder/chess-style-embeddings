import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stockfish import attach_expected_and_residuals, compute_expected_loss_tables


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute expected and residual move-quality labels from existing engine annotations."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Existing parquet with move_cp_loss, move_engine_valid, move_rating_bucket, and move_phase_name.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output parquet with recomputed expected/residual labels.",
    )
    parser.add_argument(
        "--baseline-split",
        default="train",
        help="Split used to fit expected-quality baseline tables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    required_columns = {
        "move_cp_loss",
        "move_engine_valid",
        "move_rating_bucket",
        "move_phase_name",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for relabeling: {sorted(missing)}")

    phase_table, bucket_table = compute_expected_loss_tables(df, source_split=args.baseline_split)
    relabeled_df = attach_expected_and_residuals(df, phase_table, bucket_table)
    relabeled_df.to_parquet(output_path, index=False)

    print("Saved relabeled dataset:", output_path)
    print("Total games:", len(relabeled_df))
    print("Baseline split:", args.baseline_split)
    print("Phase baseline entries:", len(phase_table))
    print("Rating-bucket baseline entries:", len(bucket_table))


if __name__ == "__main__":
    main()
