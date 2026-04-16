import argparse
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ChessWindowDataset
from src.train.config import TrainConfig
from src.train.trainer import ChessTrainingModel, collate_batch, get_device, move_batch_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Export player embeddings from a saved checkpoint.")
    parser.add_argument("--checkpoint", default="experiments/run1/model.pt")
    parser.add_argument("--data-path", default="data/processed/games_sample_200_stockfish.parquet")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-file", default="experiments/run1/player_embeddings_val.parquet")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def build_model(checkpoint):
    config = TrainConfig(**checkpoint["config"])
    model = ChessTrainingModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, config = build_model(checkpoint)
    device = get_device()
    model = model.to(device)

    dataset = ChessWindowDataset(
        args.data_path,
        split=args.split,
        window_size=config.window_size,
        stride=config.stride,
        min_window_length=config.min_window_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    player_vectors = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            backbone_outputs, _ = model(batch)

            white_embeddings = F.normalize(backbone_outputs["white_embedding"], dim=-1)
            black_embeddings = F.normalize(backbone_outputs["black_embedding"], dim=-1)
            white_embeddings = torch.nan_to_num(white_embeddings, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()
            black_embeddings = torch.nan_to_num(black_embeddings, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()

            for player_hash, rating, embedding in zip(
                batch["white_player_hash"], batch["white_rating"].detach().cpu().tolist(), white_embeddings
            ):
                if player_hash is not None:
                    player_vectors[("white", player_hash, rating)].append(embedding)

            for player_hash, rating, embedding in zip(
                batch["black_player_hash"], batch["black_rating"].detach().cpu().tolist(), black_embeddings
            ):
                if player_hash is not None:
                    player_vectors[("black", player_hash, rating)].append(embedding)

    rows = []
    for (side, player_hash, rating), embeddings in player_vectors.items():
        mean_embedding = torch.stack(embeddings).mean(dim=0)
        mean_embedding = F.normalize(mean_embedding.unsqueeze(0), dim=-1).squeeze(0)
        mean_embedding = torch.nan_to_num(mean_embedding, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(
            {
                "side": side,
                "player_hash": player_hash,
                "rating": rating,
                "num_windows": len(embeddings),
                "embedding_norm": float(mean_embedding.norm().item()),
                "embedding": mean_embedding.tolist(),
            }
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print("Saved embeddings:", output_path)
    print("Players exported:", len(rows))


if __name__ == "__main__":
    main()
