import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pgn_parser import PGNParser


def parse_args():
    parser = argparse.ArgumentParser(description="Build a processed chess dataset from PGN.")
    parser.add_argument(
        "--input-file",
        default="data/raw/lichess_db_standard_rated_2026-02.pgn.zst",
        help="Path to the input .pgn.zst archive.",
    )
    parser.add_argument(
        "--output-file",
        default="data/processed/games_sample.parquet",
        help="Path to the output parquet file.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=100000,
        help="Maximum number of filtered games to parse.",
    )
    parser.add_argument("--min-rating", type=int, default=1200)
    parser.add_argument("--max-rating", type=int, default=2200)
    parser.add_argument("--min-moves", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parser = PGNParser(
        input_file=args.input_file,
        max_games=args.max_games,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        min_moves=args.min_moves,
    )
    df = parser.parse()
    df.to_parquet(output_path, index=False)

    print("Saved dataset:", output_path)
    print("Total games:", len(df))
    if not df.empty:
        print("Columns:", ", ".join(df.columns))
        print("Splits:", df["split"].value_counts().to_dict())
        print("Time classes:", df["time_class"].value_counts().to_dict())
        print(
            "Date range:",
            df["game_datetime"].min(),
            "to",
            df["game_datetime"].max(),
        )


if __name__ == "__main__":
    main()
