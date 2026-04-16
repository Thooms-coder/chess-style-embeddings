import argparse
import io
import sys
from pathlib import Path

import chess.pgn
import zstandard as zstd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pgn_parser import PGNParser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count how many games in a PGN archive survive the parser filters."
    )
    parser.add_argument(
        "--input-file",
        default="data/raw/lichess_db_standard_rated_2026-02.pgn.zst",
        help="Path to the input .pgn.zst archive.",
    )
    parser.add_argument("--min-rating", type=int, default=1200)
    parser.add_argument("--max-rating", type=int, default=2200)
    parser.add_argument("--min-moves", type=int, default=10)
    parser.add_argument(
        "--allowed-time-controls",
        nargs="+",
        default=["Rapid", "Classical"],
        help="Allowed time classes.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Update progress every N games scanned.",
    )
    return parser.parse_args()


def count_moves(game):
    moves = 0
    node = game
    while not node.is_end():
        node = node.variation(0)
        moves += 1
    return moves


def main():
    args = parse_args()
    parser_helper = PGNParser(
        input_file=args.input_file,
        max_games=1,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        allowed_time_controls=tuple(args.allowed_time_controls),
        min_moves=args.min_moves,
    )

    total_seen = 0
    total_valid = 0
    valid_by_time_class = {time_class: 0 for time_class in args.allowed_time_controls}

    with open(args.input_file, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

        progress = tqdm(unit="games", desc="Scanning archive")
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            total_seen += 1
            headers = game.headers

            if total_seen % args.progress_every == 0:
                progress.update(args.progress_every)
                progress.set_postfix(valid=total_valid)

            if not (
                parser_helper._valid_rating(headers.get("WhiteElo"))
                and parser_helper._valid_rating(headers.get("BlackElo"))
            ):
                continue

            time_class = parser_helper._normalize_time_class(headers)
            if time_class not in args.allowed_time_controls:
                continue

            if count_moves(game) < args.min_moves:
                continue

            total_valid += 1
            valid_by_time_class[time_class] = valid_by_time_class.get(time_class, 0) + 1

        remaining = total_seen % args.progress_every
        if remaining:
            progress.update(remaining)
        progress.set_postfix(valid=total_valid)
        progress.close()

    print("Input file:", args.input_file)
    print("Total games scanned:", total_seen)
    print("Total games passing filters:", total_valid)
    if total_seen:
        print("Filter pass rate:", round(total_valid / total_seen, 6))
    print("Valid games by time class:", valid_by_time_class)


if __name__ == "__main__":
    main()
