import argparse
from pathlib import Path
import sys

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.move_tokenizer import MoveTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate processed chess games with Stockfish move quality labels."
    )
    parser.add_argument(
        "--input-file",
        default="data/processed/games_sample_200.parquet",
        help="Input parquet produced by scripts/build_dataset.py.",
    )
    parser.add_argument(
        "--output-file",
        default="data/processed/games_sample_200_stockfish.parquet",
        help="Output parquet with engine annotations.",
    )
    parser.add_argument(
        "--engine-path",
        required=True,
        help="Path to a UCI-compatible Stockfish binary.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on number of games to annotate.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Analysis depth for each position.",
    )
    parser.add_argument(
        "--skip-opening-plies",
        type=int,
        default=10,
        help="Mark early plies as opening-book moves and exclude them from residual stats.",
    )
    parser.add_argument(
        "--cp-loss-clip",
        type=int,
        default=1000,
        help="Upper clip for centipawn loss to reduce extreme outliers.",
    )
    parser.add_argument(
        "--rating-bucket-size",
        type=int,
        default=100,
        help="Bucket size used for expected-quality baselines.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write a checkpoint parquet every N annotated games.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint file if present.",
    )
    return parser.parse_args()


def evaluate_position(engine, board, pov_color, depth):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(pov_color).score(mate_score=10000)
    if score is None:
        return 0
    return int(score)


def annotate_game(row, engine, depth, skip_opening_plies, cp_loss_clip, tokenizer):
    board = chess.Board()
    encoded_moves = tokenizer.encode_game(
        moves=row.moves,
        white_rating=row.white_rating,
        black_rating=row.black_rating,
        time_class=getattr(row, "time_class", None),
        result_label=getattr(row, "result_label", "unknown"),
    )

    eval_before = []
    eval_after = []
    cp_loss = []
    engine_valid = []
    mover_rating = []
    mover_bucket = []
    phase_name = []

    for ply_index, move_uci in enumerate(row.moves):
        mover_color = board.turn
        before_cp = evaluate_position(engine, board, mover_color, depth)
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        after_cp = evaluate_position(engine, board, mover_color, depth)

        loss = max(0, before_cp - after_cp)
        clipped_loss = min(loss, cp_loss_clip)
        valid = ply_index >= skip_opening_plies

        move_features = encoded_moves[ply_index]
        eval_before.append(before_cp)
        eval_after.append(after_cp)
        cp_loss.append(clipped_loss)
        engine_valid.append(valid)
        mover_rating.append(move_features["player_rating"])
        mover_bucket.append(move_features["rating_bucket"])
        phase_name.append(move_features["phase_name"])

    return {
        "move_eval_before_cp": eval_before,
        "move_eval_after_cp": eval_after,
        "move_cp_loss": cp_loss,
        "move_engine_valid": engine_valid,
        "move_player_rating": mover_rating,
        "move_rating_bucket": mover_bucket,
        "move_phase_name": phase_name,
    }


def compute_expected_loss_tables(df, source_split="train"):
    records = []
    baseline_df = df
    if "split" in df.columns:
        split_filtered = df[df["split"] == source_split]
        if not split_filtered.empty:
            baseline_df = split_filtered

    for row in baseline_df.itertuples(index=False):
        for idx, loss in enumerate(row.move_cp_loss):
            if not row.move_engine_valid[idx]:
                continue
            records.append(
                {
                    "rating_bucket": row.move_rating_bucket[idx],
                    "phase_name": row.move_phase_name[idx],
                    "cp_loss": loss,
                }
            )

    if not records:
        return {}, {}

    loss_df = pd.DataFrame(records)
    phase_table = (
        loss_df.groupby(["rating_bucket", "phase_name"])["cp_loss"].mean().to_dict()
    )
    bucket_table = loss_df.groupby("rating_bucket")["cp_loss"].mean().to_dict()

    return phase_table, bucket_table


def attach_expected_and_residuals(df, phase_table, bucket_table):
    expected_column = []
    residual_column = []

    for row in df.itertuples(index=False):
        expected_values = []
        residual_values = []

        for idx, loss in enumerate(row.move_cp_loss):
            if not row.move_engine_valid[idx]:
                expected = None
                residual = None
            else:
                key = (row.move_rating_bucket[idx], row.move_phase_name[idx])
                expected = phase_table.get(key, bucket_table.get(row.move_rating_bucket[idx], 0.0))
                residual = float(loss - expected)

            expected_values.append(expected)
            residual_values.append(residual)

        expected_column.append(expected_values)
        residual_column.append(residual_values)

    df = df.copy()
    df["move_expected_cp_loss"] = expected_column
    df["move_residual_cp_loss"] = residual_column
    return df


def write_output(df, output_path, baseline_split="train"):
    phase_table, bucket_table = compute_expected_loss_tables(df, source_split=baseline_split)
    enriched_df = attach_expected_and_residuals(df, phase_table, bucket_table)
    enriched_df.to_parquet(output_path, index=False)
    return enriched_df, phase_table, bucket_table


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if args.max_games is not None:
        df = df.head(args.max_games).copy()

    tokenizer = MoveTokenizer(rating_bucket_size=args.rating_bucket_size)
    output_path = Path(args.output_file)
    checkpoint_path = output_path.with_suffix(".checkpoint.parquet")

    start_index = 0
    annotated_rows = []

    if args.resume and checkpoint_path.exists():
        checkpoint_df = pd.read_parquet(checkpoint_path)
        annotated_columns = {
            "move_eval_before_cp",
            "move_eval_after_cp",
            "move_cp_loss",
            "move_engine_valid",
            "move_player_rating",
            "move_rating_bucket",
            "move_phase_name",
        }
        if annotated_columns.issubset(set(checkpoint_df.columns)):
            start_index = len(checkpoint_df)
            annotated_rows = checkpoint_df[list(annotated_columns)].to_dict("records")
            print(f"Resuming from checkpoint: {checkpoint_path} ({start_index} games)")
            df = df.iloc[start_index:].reset_index(drop=True)

    with chess.engine.SimpleEngine.popen_uci(args.engine_path) as engine:
        progress = tqdm(df.itertuples(index=False), total=len(df), desc="Annotating games")
        for relative_index, row in enumerate(progress, start=1):
            annotations = annotate_game(
                row=row,
                engine=engine,
                depth=args.depth,
                skip_opening_plies=args.skip_opening_plies,
                cp_loss_clip=args.cp_loss_clip,
                tokenizer=tokenizer,
            )
            annotated_rows.append(annotations)

            if args.checkpoint_every > 0 and relative_index % args.checkpoint_every == 0:
                completed_count = start_index + relative_index
                partial_source_df = pd.read_parquet(input_path).head(completed_count).reset_index(
                    drop=True
                )
                partial_annotations_df = pd.DataFrame(annotated_rows)
                partial_merged_df = pd.concat(
                    [partial_source_df.reset_index(drop=True), partial_annotations_df],
                    axis=1,
                )
                _, phase_table, bucket_table = write_output(
                    partial_merged_df,
                    checkpoint_path,
                    baseline_split="train",
                )
                progress.set_postfix(
                    completed=completed_count,
                    phase_baselines=len(phase_table),
                    bucket_baselines=len(bucket_table),
                )

    annotations_df = pd.DataFrame(annotated_rows)
    source_df = pd.read_parquet(input_path)
    if args.max_games is not None:
        source_df = source_df.head(args.max_games).copy()
    merged_df = pd.concat(
        [source_df.reset_index(drop=True), annotations_df.reset_index(drop=True)],
        axis=1,
    )

    merged_df, phase_table, bucket_table = write_output(
        merged_df,
        output_path,
        baseline_split="train",
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("Saved annotated dataset:", output_path)
    print("Total games:", len(merged_df))
    print("Phase baseline entries:", len(phase_table))
    print("Rating-bucket baseline entries:", len(bucket_table))


if __name__ == "__main__":
    main()
