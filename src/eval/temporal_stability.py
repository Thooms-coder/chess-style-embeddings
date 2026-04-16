import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PHASE_ORDER = ["opening", "middlegame", "endgame"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure temporal stability of player phase-residual profiles."
    )
    parser.add_argument(
        "--input-file",
        default="data/processed/games_sample_10000_stockfish.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/run2/stability",
    )
    parser.add_argument(
        "--min-valid-moves",
        type=int,
        default=20,
        help="Minimum valid engine-labeled moves per player and split.",
    )
    parser.add_argument(
        "--residual-clip",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return np.nan
    return float(np.dot(a, b) / (a_norm * b_norm))


def build_player_phase_profiles(df, residual_clip):
    records = []

    for row in df.itertuples(index=False):
        for ply_index, is_valid in enumerate(row.move_engine_valid):
            if not is_valid:
                continue

            residual = row.move_residual_cp_loss[ply_index]
            if residual is None:
                continue

            player_hash = row.white_player_hash if ply_index % 2 == 0 else row.black_player_hash
            if player_hash is None:
                continue

            phase_name = row.move_phase_name[ply_index]
            clipped_residual = float(np.clip(float(residual), -residual_clip, residual_clip))
            records.append(
                {
                    "player_hash": player_hash,
                    "split": row.split,
                    "phase_name": phase_name,
                    "residual": clipped_residual,
                }
            )

    if not records:
        return pd.DataFrame()

    move_df = pd.DataFrame(records)
    grouped = (
        move_df.groupby(["player_hash", "split", "phase_name"])["residual"]
        .agg(["mean", "count"])
        .reset_index()
    )

    mean_wide = (
        grouped.pivot_table(
            index=["player_hash", "split"],
            columns="phase_name",
            values="mean",
        )
        .reindex(columns=PHASE_ORDER)
        .reset_index()
    )
    count_wide = (
        grouped.pivot_table(
            index=["player_hash", "split"],
            columns="phase_name",
            values="count",
            fill_value=0,
        )
        .reindex(columns=PHASE_ORDER, fill_value=0)
        .reset_index()
    )

    profile_df = mean_wide.merge(
        count_wide,
        on=["player_hash", "split"],
        suffixes=("_mean", "_count"),
    )
    return profile_df


def compare_split_profiles(profile_df, earlier_split, later_split, min_valid_moves, seed):
    earlier = profile_df[profile_df["split"] == earlier_split].copy()
    later = profile_df[profile_df["split"] == later_split].copy()

    count_columns = [f"{phase}_count" for phase in PHASE_ORDER]
    earlier["total_valid_moves"] = earlier[count_columns].sum(axis=1)
    later["total_valid_moves"] = later[count_columns].sum(axis=1)

    earlier = earlier[earlier["total_valid_moves"] >= min_valid_moves]
    later = later[later["total_valid_moves"] >= min_valid_moves]

    merged = earlier.merge(
        later,
        on="player_hash",
        suffixes=("_earlier", "_later"),
    )
    if merged.empty:
        return pd.DataFrame(), {}

    phase_mean_earlier = [f"{phase}_mean_earlier" for phase in PHASE_ORDER]
    phase_mean_later = [f"{phase}_mean_later" for phase in PHASE_ORDER]

    comparisons = []
    rng = np.random.default_rng(seed)
    later_vectors = merged[phase_mean_later].to_numpy(dtype=float)

    for row_idx, row in merged.iterrows():
        earlier_vector = row[phase_mean_earlier].to_numpy(dtype=float)
        later_vector = row[phase_mean_later].to_numpy(dtype=float)
        earlier_vector = np.nan_to_num(earlier_vector, nan=0.0)
        later_vector = np.nan_to_num(later_vector, nan=0.0)
        random_idx = int(rng.integers(0, len(later_vectors)))
        random_later = np.nan_to_num(later_vectors[random_idx], nan=0.0)

        comparisons.append(
            {
                "player_hash": row["player_hash"],
                "earlier_split": earlier_split,
                "later_split": later_split,
                "earlier_valid_moves": int(row["total_valid_moves_earlier"]),
                "later_valid_moves": int(row["total_valid_moves_later"]),
                "temporal_cosine": cosine_similarity(earlier_vector, later_vector),
                "temporal_rmse": float(np.sqrt(np.mean((earlier_vector - later_vector) ** 2))),
                "random_cosine": cosine_similarity(earlier_vector, random_later),
                "random_rmse": float(np.sqrt(np.mean((earlier_vector - random_later) ** 2))),
            }
        )

    comparison_df = pd.DataFrame(comparisons)
    summary = {
        "earlier_split": earlier_split,
        "later_split": later_split,
        "players_compared": int(len(comparison_df)),
        "mean_temporal_cosine": float(comparison_df["temporal_cosine"].mean()),
        "mean_random_cosine": float(comparison_df["random_cosine"].mean()),
        "mean_temporal_rmse": float(comparison_df["temporal_rmse"].mean()),
        "mean_random_rmse": float(comparison_df["random_rmse"].mean()),
    }
    return comparison_df, summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_file)
    profile_df = build_player_phase_profiles(df, residual_clip=args.residual_clip)
    if profile_df.empty:
        raise ValueError("No valid engine-labeled residual profiles could be built.")

    profile_path = output_dir / "player_phase_profiles.parquet"
    profile_df.to_parquet(profile_path, index=False)

    comparisons = []
    summaries = []
    for earlier_split, later_split in (("train", "val"), ("val", "test"), ("train", "test")):
        comparison_df, summary = compare_split_profiles(
            profile_df,
            earlier_split=earlier_split,
            later_split=later_split,
            min_valid_moves=args.min_valid_moves,
            seed=args.seed,
        )
        if comparison_df.empty:
            continue
        comparison_path = output_dir / f"temporal_{earlier_split}_vs_{later_split}.parquet"
        comparison_df.to_parquet(comparison_path, index=False)
        comparisons.append(comparison_path)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "temporal_stability_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)

    print("Saved profiles:", profile_path)
    print("Saved summary:", summary_path)
    for comparison_path in comparisons:
        print("Saved comparison:", comparison_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
