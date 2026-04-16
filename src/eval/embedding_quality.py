import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.embed_analysis import build_model, parse_args as _unused  # noqa: F401
from src.eval.personalization_validity import export_observed_profiles


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare embedding-based neighbor quality against rating-only neighbors."
    )
    parser.add_argument("--embedding-file", default="experiments/run2/player_embeddings_val.parquet")
    parser.add_argument("--data-path", default="data/processed/games_sample_10000_stockfish.parquet")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="experiments/run2/embedding_quality")
    parser.add_argument("--min-windows", type=int, default=1)
    return parser.parse_args()


def cosine_similarity_matrix(x):
    x = np.nan_to_num(x.astype(np.float64, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalized = x / norms
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    similarities = normalized @ normalized.T
    return np.nan_to_num(similarities, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_df = pd.read_parquet(args.embedding_file)
    embedding_df = embedding_df[embedding_df["num_windows"] >= args.min_windows].copy()
    observed_df = export_observed_profiles(args.data_path, split=args.split)
    observed_df = observed_df[observed_df["num_windows"] >= args.min_windows].copy()

    merged = embedding_df.merge(observed_df, on=["side", "player_hash"], suffixes=("_embed", "_obs"))
    if len(merged) < 3:
        raise ValueError("Need at least 3 overlapping player profiles for neighbor comparison.")

    embeddings = np.asarray(merged["embedding"].tolist(), dtype=float)
    ratings = merged["rating"].to_numpy(dtype=float)
    observed = merged[["obs_opening", "obs_middlegame", "obs_endgame"]].to_numpy(dtype=float)
    similarity = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(similarity, -np.inf)

    comparisons = []
    for idx in range(len(merged)):
        rating_diffs = np.abs(ratings - ratings[idx])
        rating_diffs[idx] = np.inf
        rating_neighbor = int(np.argmin(rating_diffs))
        embedding_neighbor = int(np.argmax(similarity[idx]))

        target = observed[idx]
        rating_profile = observed[rating_neighbor]
        embedding_profile = observed[embedding_neighbor]

        comparisons.append(
            {
                "side": merged.iloc[idx]["side"],
                "player_hash": merged.iloc[idx]["player_hash"],
                "rating": float(ratings[idx]),
                "num_windows": int(merged.iloc[idx]["num_windows_embed"]),
                "embedding_neighbor_hash": merged.iloc[embedding_neighbor]["player_hash"],
                "rating_neighbor_hash": merged.iloc[rating_neighbor]["player_hash"],
                "embedding_neighbor_similarity": float(similarity[idx, embedding_neighbor]),
                "rating_neighbor_gap": float(rating_diffs[rating_neighbor]),
                "embedding_neighbor_rmse": rmse(target, embedding_profile),
                "rating_neighbor_rmse": rmse(target, rating_profile),
            }
        )

    comparison_df = pd.DataFrame(comparisons)
    summary_df = pd.DataFrame(
        [
            {
                "split": args.split,
                "players_compared": int(len(comparison_df)),
                "mean_embedding_neighbor_rmse": float(comparison_df["embedding_neighbor_rmse"].mean()),
                "mean_rating_neighbor_rmse": float(comparison_df["rating_neighbor_rmse"].mean()),
                "embedding_beats_rating_rate": float(
                    (comparison_df["embedding_neighbor_rmse"] < comparison_df["rating_neighbor_rmse"]).mean()
                ),
                "mean_embedding_neighbor_similarity": float(
                    comparison_df["embedding_neighbor_similarity"].mean()
                ),
                "mean_rating_neighbor_gap": float(comparison_df["rating_neighbor_gap"].mean()),
            }
        ]
    )

    comparison_path = output_dir / f"embedding_vs_rating_{args.split}.parquet"
    summary_path = output_dir / "embedding_quality_summary.parquet"
    comparison_df.to_parquet(comparison_path, index=False)
    summary_df.to_parquet(summary_path, index=False)

    print("Saved comparison:", comparison_path)
    print("Saved summary:", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
