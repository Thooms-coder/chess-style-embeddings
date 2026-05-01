import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ChessWindowDataset
from src.train.trainer import ChessTrainingModel, collate_batch, get_device, move_batch_to_device
from src.train.config import TrainConfig


PHASE_NAMES = ["opening", "middlegame", "endgame"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 5 test predictions for the final project demo.")
    parser.add_argument("--checkpoint", default="experiments/run2/model.pt")
    parser.add_argument(
        "--data-path",
        default="data/processed/games_sample_25000_stockfish_trainonly.parquet",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", default="experiments/run2/final_demo")
    return parser.parse_args()


def build_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig(**checkpoint["config"])
    model = ChessTrainingModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def format_vector(vector, mask=None):
    items = []
    for idx, phase_name in enumerate(PHASE_NAMES):
        if mask is not None and not bool(mask[idx]):
            items.append(f"{phase_name}=n/a")
        else:
            items.append(f"{phase_name}={float(vector[idx]):.2f}")
    return ", ".join(items)


def select_demo_indices(dataset, num_samples):
    candidates = []
    for idx, sample in enumerate(dataset.samples):
        white_valid = int(sample["white_phase_mask"].sum())
        black_valid = int(sample["black_phase_mask"].sum())
        valid_moves = int(sample.get("move_engine_valid", torch.zeros(1, dtype=torch.bool)).sum())
        score = white_valid + black_valid + (1 if valid_moves >= 32 else 0)
        if score >= 3:
            candidates.append((score, idx))
    candidates.sort(reverse=True)
    selected = [idx for _, idx in candidates[:num_samples]]
    if len(selected) < num_samples:
        selected.extend(list(range(len(selected), min(len(dataset), num_samples))))
    return selected[:num_samples]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = build_model(args.checkpoint)
    device = get_device()
    model = model.to(device)

    dataset = ChessWindowDataset(
        args.data_path,
        split=args.split,
        window_size=config.window_size,
        stride=config.stride,
        min_window_length=config.min_window_length,
    )

    indices = select_demo_indices(dataset, args.num_samples)
    batch = collate_batch([dataset[idx] for idx in indices])
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        _, outputs = model(batch)

    white_pred = outputs["white_phase_residual"].detach().cpu()
    black_pred = outputs["black_phase_residual"].detach().cpu()
    white_rating_pred = outputs["white_rating"].detach().cpu()
    black_rating_pred = outputs["black_rating"].detach().cpu()

    rows = []
    for i, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        move_preview = " ".join(move for move in sample["move_uci"][:12] if move != "<pad>")
        row = {
            "sample_index": sample_idx,
            "game_id": sample["game_id"],
            "window_start": sample["window_start"],
            "window_end": sample["window_end"],
            "window_length": sample["window_length"],
            "time_class": sample["time_class"],
            "result_label": sample["result_label"],
            "white_player_hash": sample["white_player_hash"],
            "black_player_hash": sample["black_player_hash"],
            "white_rating_true": sample["white_rating"],
            "black_rating_true": sample["black_rating"],
            "white_rating_pred": float(white_rating_pred[i]),
            "black_rating_pred": float(black_rating_pred[i]),
            "white_phase_pred": [float(x) for x in white_pred[i].tolist()],
            "black_phase_pred": [float(x) for x in black_pred[i].tolist()],
            "white_phase_true": [float(x) for x in sample["white_phase_residual"].tolist()],
            "black_phase_true": [float(x) for x in sample["black_phase_residual"].tolist()],
            "white_phase_mask": [bool(x) for x in sample["white_phase_mask"].tolist()],
            "black_phase_mask": [bool(x) for x in sample["black_phase_mask"].tolist()],
            "move_preview": move_preview,
        }
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_path = output_dir / f"demo_predictions_{args.split}.parquet"
    result_df.to_parquet(result_path, index=False)

    markdown_lines = [
        "# Final Demo: Test Predictions",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- data: `{args.data_path}`",
        f"- split: `{args.split}`",
        "",
    ]

    for sample_number, row in enumerate(rows, start=1):
        markdown_lines.extend(
            [
                f"## Sample {sample_number}",
                "",
                f"- game_id: `{row['game_id']}`",
                f"- window: `{row['window_start']}:{row['window_end']}`",
                f"- time class: `{row['time_class']}`",
                f"- result: `{row['result_label']}`",
                f"- white rating: true `{row['white_rating_true']}`, predicted `{row['white_rating_pred']:.2f}`",
                f"- black rating: true `{row['black_rating_true']}`, predicted `{row['black_rating_pred']:.2f}`",
                f"- white predicted weakness: {format_vector(row['white_phase_pred'])}",
                f"- white observed weakness: {format_vector(row['white_phase_true'], row['white_phase_mask'])}",
                f"- black predicted weakness: {format_vector(row['black_phase_pred'])}",
                f"- black observed weakness: {format_vector(row['black_phase_true'], row['black_phase_mask'])}",
                f"- opening moves preview: `{row['move_preview']}`",
                "",
            ]
        )

    markdown_path = output_dir / f"demo_predictions_{args.split}.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    print("Saved demo predictions:", result_path)
    print("Saved markdown demo summary:", markdown_path)
    for idx, row in enumerate(rows, start=1):
        print(f"Sample {idx}: game_id={row['game_id']} window={row['window_start']}:{row['window_end']}")
        print("  White predicted:", format_vector(row["white_phase_pred"]))
        print("  White observed :", format_vector(row["white_phase_true"], row["white_phase_mask"]))
        print("  Black predicted:", format_vector(row["black_phase_pred"]))
        print("  Black observed :", format_vector(row["black_phase_true"], row["black_phase_mask"]))
        print("  Moves:", row["move_preview"])


if __name__ == "__main__":
    main()
