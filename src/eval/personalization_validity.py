import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ChessWindowDataset
from src.train.trainer import ChessTrainingModel, collate_batch, get_device, move_batch_to_device
from src.train.config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate whether earlier predicted phase weaknesses match later observed weaknesses."
    )
    parser.add_argument("--checkpoint", default="experiments/run2/model.pt")
    parser.add_argument("--data-path", default="data/processed/games_sample_10000_stockfish.parquet")
    parser.add_argument("--earlier-split", default="train")
    parser.add_argument("--later-split", default="val")
    parser.add_argument("--output-dir", default="experiments/run2/personalization")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-windows", type=int, default=2)
    return parser.parse_args()


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return np.nan
    return float(np.dot(a, b) / (a_norm * b_norm))


def build_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**checkpoint["config"])
    model = ChessTrainingModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def export_predicted_profiles(model, config, data_path, split, batch_size):
    device = get_device()
    model = model.to(device)
    dataset = ChessWindowDataset(
        data_path,
        split=split,
        window_size=config.window_size,
        stride=config.stride,
        min_window_length=config.min_window_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    player_predictions = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            _, outputs = model(batch)

            white_predictions = outputs["white_phase_residual"].detach().cpu().numpy()
            black_predictions = outputs["black_phase_residual"].detach().cpu().numpy()

            for player_hash, vector in zip(batch["white_player_hash"], white_predictions):
                if player_hash is not None:
                    player_predictions[("white", player_hash)].append(vector)
            for player_hash, vector in zip(batch["black_player_hash"], black_predictions):
                if player_hash is not None:
                    player_predictions[("black", player_hash)].append(vector)

    rows = []
    for (side, player_hash), vectors in player_predictions.items():
        mean_vector = np.mean(np.asarray(vectors, dtype=float), axis=0)
        rows.append(
            {
                "side": side,
                "player_hash": player_hash,
                "num_windows": len(vectors),
                "pred_opening": float(mean_vector[0]),
                "pred_middlegame": float(mean_vector[1]),
                "pred_endgame": float(mean_vector[2]),
            }
        )

    return pd.DataFrame(rows)


def export_observed_profiles(data_path, split):
    dataset = ChessWindowDataset(data_path, split=split)
    grouped = defaultdict(list)

    for sample in dataset.samples:
        for side in ("white", "black"):
            player_hash = sample[f"{side}_player_hash"]
            if player_hash is None:
                continue
            mask = sample[f"{side}_phase_mask"].numpy()
            values = sample[f"{side}_phase_residual"].numpy()
            grouped[(side, player_hash)].append((values, mask))

    rows = []
    for (side, player_hash), entries in grouped.items():
        sums = np.zeros(3, dtype=float)
        counts = np.zeros(3, dtype=float)
        for values, mask in entries:
            sums += values * mask
            counts += mask.astype(float)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        rows.append(
            {
                "side": side,
                "player_hash": player_hash,
                "num_windows": len(entries),
                "obs_opening": float(means[0]),
                "obs_middlegame": float(means[1]),
                "obs_endgame": float(means[2]),
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = build_model(args.checkpoint)
    predicted_df = export_predicted_profiles(
        model,
        config,
        data_path=args.data_path,
        split=args.earlier_split,
        batch_size=args.batch_size,
    )
    observed_df = export_observed_profiles(args.data_path, split=args.later_split)

    predicted_df = predicted_df[predicted_df["num_windows"] >= args.min_windows]
    observed_df = observed_df[observed_df["num_windows"] >= args.min_windows]

    merged = predicted_df.merge(observed_df, on=["side", "player_hash"], suffixes=("_pred", "_obs"))
    if merged.empty:
        raise ValueError("No overlapping players met the minimum window threshold.")

    comparisons = []
    obs_vectors = merged[["obs_opening", "obs_middlegame", "obs_endgame"]].to_numpy(dtype=float)
    rng = np.random.default_rng(42)

    for _, row in merged.iterrows():
        pred = row[["pred_opening", "pred_middlegame", "pred_endgame"]].to_numpy(dtype=float)
        obs = row[["obs_opening", "obs_middlegame", "obs_endgame"]].to_numpy(dtype=float)
        random_obs = obs_vectors[int(rng.integers(0, len(obs_vectors)))]
        comparisons.append(
            {
                "side": row["side"],
                "player_hash": row["player_hash"],
                "earlier_windows": int(row["num_windows_pred"]),
                "later_windows": int(row["num_windows_obs"]),
                "personal_cosine": cosine_similarity(pred, obs),
                "random_cosine": cosine_similarity(pred, random_obs),
                "personal_rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
                "random_rmse": float(np.sqrt(np.mean((pred - random_obs) ** 2))),
            }
        )

    comparison_df = pd.DataFrame(comparisons)
    summary = pd.DataFrame(
        [
            {
                "earlier_split": args.earlier_split,
                "later_split": args.later_split,
                "players_compared": int(len(comparison_df)),
                "mean_personal_cosine": float(comparison_df["personal_cosine"].mean()),
                "mean_random_cosine": float(comparison_df["random_cosine"].mean()),
                "mean_personal_rmse": float(comparison_df["personal_rmse"].mean()),
                "mean_random_rmse": float(comparison_df["random_rmse"].mean()),
            }
        ]
    )

    comparison_path = output_dir / f"personalization_{args.earlier_split}_to_{args.later_split}.parquet"
    summary_path = output_dir / "personalization_summary.parquet"
    comparison_df.to_parquet(comparison_path, index=False)
    summary.to_parquet(summary_path, index=False)

    print("Saved comparison:", comparison_path)
    print("Saved summary:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
