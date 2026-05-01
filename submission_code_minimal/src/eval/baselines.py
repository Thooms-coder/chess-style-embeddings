import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


PHASE_ORDER = ["opening", "middlegame", "endgame"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run non-neural baselines for phase-residual weakness prediction."
    )
    parser.add_argument(
        "--input-file",
        default="data/processed/games_sample_25000_stockfish.parquet",
    )
    parser.add_argument(
        "--train-split",
        default="train",
    )
    parser.add_argument(
        "--eval-split",
        default="val",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/run2/baselines",
    )
    parser.add_argument(
        "--rating-bucket-size",
        type=int,
        default=100,
    )
    return parser.parse_args()


def init_stats():
    return {
        "game_count": 0,
        "rapid_games": 0,
        "classical_games": 0,
        "rating_sum": 0.0,
        "opponent_rating_sum": 0.0,
        "num_moves_sum": 0.0,
        "valid_moves": 0,
        "cp_loss_sum": 0.0,
        "expected_cp_sum": 0.0,
        "phase_cp_sum": {phase: 0.0 for phase in PHASE_ORDER},
        "phase_cp_count": {phase: 0 for phase in PHASE_ORDER},
        "phase_expected_sum": {phase: 0.0 for phase in PHASE_ORDER},
        "phase_expected_count": {phase: 0 for phase in PHASE_ORDER},
        "phase_residual_sum": {phase: 0.0 for phase in PHASE_ORDER},
        "phase_residual_count": {phase: 0 for phase in PHASE_ORDER},
    }


def build_player_side_table(df):
    aggregates = defaultdict(init_stats)

    for row in df.itertuples(index=False):
        for side_name, player_hash, player_rating, opponent_rating in (
            ("white", row.white_player_hash, row.white_rating, row.black_rating),
            ("black", row.black_player_hash, row.black_rating, row.white_rating),
        ):
            if player_hash is None:
                continue
            key = (row.split, side_name, player_hash)
            stats = aggregates[key]
            stats["game_count"] += 1
            stats["rating_sum"] += float(player_rating)
            stats["opponent_rating_sum"] += float(opponent_rating)
            stats["num_moves_sum"] += float(row.num_moves)
            if row.time_class == "Rapid":
                stats["rapid_games"] += 1
            elif row.time_class == "Classical":
                stats["classical_games"] += 1

        for ply_index, is_valid in enumerate(row.move_engine_valid):
            if not is_valid:
                continue
            phase_name = row.move_phase_name[ply_index]
            if phase_name not in PHASE_ORDER:
                continue

            side_name = "white" if ply_index % 2 == 0 else "black"
            player_hash = row.white_player_hash if side_name == "white" else row.black_player_hash
            if player_hash is None:
                continue

            key = (row.split, side_name, player_hash)
            stats = aggregates[key]
            cp_loss = float(row.move_cp_loss[ply_index])
            expected_cp = float(row.move_expected_cp_loss[ply_index])
            residual_cp = float(row.move_residual_cp_loss[ply_index])

            stats["valid_moves"] += 1
            stats["cp_loss_sum"] += cp_loss
            stats["expected_cp_sum"] += expected_cp
            stats["phase_cp_sum"][phase_name] += cp_loss
            stats["phase_cp_count"][phase_name] += 1
            stats["phase_expected_sum"][phase_name] += expected_cp
            stats["phase_expected_count"][phase_name] += 1
            stats["phase_residual_sum"][phase_name] += residual_cp
            stats["phase_residual_count"][phase_name] += 1

    rows = []
    for (split, side, player_hash), stats in aggregates.items():
        if stats["game_count"] == 0:
            continue

        row = {
            "split": split,
            "side": side,
            "player_hash": player_hash,
            "game_count": stats["game_count"],
            "rapid_frac": stats["rapid_games"] / stats["game_count"],
            "classical_frac": stats["classical_games"] / stats["game_count"],
            "mean_rating": stats["rating_sum"] / stats["game_count"],
            "mean_opponent_rating": stats["opponent_rating_sum"] / stats["game_count"],
            "mean_num_moves": stats["num_moves_sum"] / stats["game_count"],
            "valid_moves": stats["valid_moves"],
            "mean_cp_loss": stats["cp_loss_sum"] / max(stats["valid_moves"], 1),
            "mean_expected_cp_loss": stats["expected_cp_sum"] / max(stats["valid_moves"], 1),
        }

        for phase in PHASE_ORDER:
            row[f"{phase}_move_frac"] = stats["phase_cp_count"][phase] / max(stats["valid_moves"], 1)
            row[f"{phase}_mean_cp_loss"] = stats["phase_cp_sum"][phase] / max(
                stats["phase_cp_count"][phase], 1
            )
            row[f"{phase}_mean_expected_cp_loss"] = stats["phase_expected_sum"][phase] / max(
                stats["phase_expected_count"][phase], 1
            )
            row[f"target_{phase}"] = stats["phase_residual_sum"][phase] / max(
                stats["phase_residual_count"][phase], 1
            )
            row[f"target_{phase}_count"] = stats["phase_residual_count"][phase]

        rows.append(row)

    return pd.DataFrame(rows)


def feature_columns():
    columns = [
        "mean_rating",
        "mean_opponent_rating",
        "rapid_frac",
        "classical_frac",
        "mean_num_moves",
        "valid_moves",
        "mean_cp_loss",
        "mean_expected_cp_loss",
    ]
    for phase in PHASE_ORDER:
        columns.append(f"{phase}_move_frac")
    return columns


def target_columns():
    return [f"target_{phase}" for phase in PHASE_ORDER]


def prepare_matrix(df, columns):
    matrix = df[columns].to_numpy(dtype=np.float64, copy=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(matrix, -1e6, 1e6)


def evaluate_predictions(name, y_true, y_pred):
    per_phase = {}
    rmses = []
    for idx, phase in enumerate(PHASE_ORDER):
        rmse = float(np.sqrt(np.mean((y_true[:, idx] - y_pred[:, idx]) ** 2)))
        per_phase[f"{phase}_rmse"] = float(rmse)
        rmses.append(rmse)
    return {
        "model": name,
        "phase_residual_rmse": float(np.mean(rmses)),
        **per_phase,
    }


def rating_baseline(train_df, eval_df, bucket_size):
    train_df = train_df.copy()
    eval_df = eval_df.copy()
    train_df["rating_bucket"] = (train_df["mean_rating"] // bucket_size).astype(int)
    eval_df["rating_bucket"] = (eval_df["mean_rating"] // bucket_size).astype(int)

    bucket_means = train_df.groupby("rating_bucket")[target_columns()].mean()
    global_mean = train_df[target_columns()].mean()

    predictions = []
    for row in eval_df.itertuples(index=False):
        if row.rating_bucket in bucket_means.index:
            pred = bucket_means.loc[row.rating_bucket].to_numpy(dtype=float)
        else:
            pred = global_mean.to_numpy(dtype=float)
        predictions.append(pred)
    return np.asarray(predictions, dtype=float)


def fit_linear_baseline(train_df, eval_df):
    model = Ridge(alpha=1.0)
    model = MultiOutputRegressor(model)
    model.fit(prepare_matrix(train_df, feature_columns()), prepare_matrix(train_df, target_columns()))
    return model.predict(prepare_matrix(eval_df, feature_columns()))


def fit_tree_baseline(train_df, eval_df):
    base = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=42,
    )
    model = MultiOutputRegressor(base)
    model.fit(prepare_matrix(train_df, feature_columns()), prepare_matrix(train_df, target_columns()))
    return model.predict(prepare_matrix(eval_df, feature_columns()))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_file)
    player_df = build_player_side_table(df)

    train_df = player_df[player_df["split"] == args.train_split].reset_index(drop=True)
    eval_df = player_df[player_df["split"] == args.eval_split].reset_index(drop=True)
    y_eval = eval_df[target_columns()].to_numpy(dtype=float)

    summaries = []

    rating_pred = rating_baseline(train_df, eval_df, bucket_size=args.rating_bucket_size)
    summaries.append(evaluate_predictions("rating_heuristic", y_eval, rating_pred))

    linear_pred = fit_linear_baseline(train_df, eval_df)
    summaries.append(evaluate_predictions("ridge", y_eval, linear_pred))

    tree_pred = fit_tree_baseline(train_df, eval_df)
    summaries.append(evaluate_predictions("hist_gbdt", y_eval, tree_pred))

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / f"baseline_summary_{args.train_split}_to_{args.eval_split}.parquet"
    player_path = output_dir / f"baseline_player_table_{args.train_split}_to_{args.eval_split}.parquet"
    summary_df.to_parquet(summary_path, index=False)
    player_df.to_parquet(player_path, index=False)

    print("Saved summary:", summary_path)
    print("Saved player table:", player_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
