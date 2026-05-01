import argparse
from pathlib import Path
import sys
import math
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, BatchSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ChessWindowDataset
from src.eval.metrics import engine_target_metrics, rating_rmse
from src.models.heads import MultiTaskHeads
from src.models.losses import total_loss
from src.models.transformer import ChessStyleTransformer
from src.train.config import ensure_output_dir, load_config


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_batch(batch):
    collated = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], bool):
            collated[key] = torch.tensor(values, dtype=torch.bool)
        elif isinstance(values[0], int):
            collated[key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], float):
            collated[key] = torch.tensor(values, dtype=torch.float32)
        else:
            collated[key] = values
    return collated


class PlayerAwareBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, positive_group_size=2, positive_batch_ratio=0.75):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_group_size = max(2, positive_group_size)
        self.positive_batch_ratio = positive_batch_ratio
        self.num_samples = len(dataset)
        self.all_indices = list(range(self.num_samples))
        self.player_to_indices = self._build_player_index()
        self.player_keys = list(self.player_to_indices.keys())

    def _build_player_index(self):
        mapping = {}
        for idx, sample in enumerate(self.dataset.samples):
            for side in ("white", "black"):
                player_hash = sample.get(f"{side}_player_hash")
                if player_hash is None:
                    continue
                key = (side, player_hash)
                mapping.setdefault(key, []).append(idx)

        return {
            key: indices
            for key, indices in mapping.items()
            if len(indices) >= self.positive_group_size
        }

    def __iter__(self):
        num_batches = len(self)
        for _ in range(num_batches):
            batch_indices = []
            if self.player_keys and random.random() < self.positive_batch_ratio:
                key = random.choice(self.player_keys)
                candidate_indices = self.player_to_indices[key]
                take = min(self.positive_group_size, len(candidate_indices), self.batch_size)
                batch_indices.extend(random.sample(candidate_indices, take))

            if len(batch_indices) < self.batch_size:
                remaining_pool = [idx for idx in self.all_indices if idx not in batch_indices]
                fill_count = self.batch_size - len(batch_indices)
                if remaining_pool:
                    batch_indices.extend(random.sample(remaining_pool, min(fill_count, len(remaining_pool))))

            if len(batch_indices) < self.batch_size:
                refill_count = self.batch_size - len(batch_indices)
                batch_indices.extend(random.choices(self.all_indices, k=refill_count))

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)


class ChessTrainingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = ChessStyleTransformer(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_sequence_length=config.window_size,
        )
        self.heads = MultiTaskHeads(
            hidden_dim=config.hidden_dim,
            residual_clip=config.loss_weights.get("residual_clip", 300.0),
        )

    def forward(self, batch):
        backbone_outputs = self.backbone(batch)
        head_outputs = self.heads(backbone_outputs)
        return backbone_outputs, head_outputs


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(model, loader, optimizer, device, loss_weights, train=True):
    model.train(train)
    running = {
        "loss_total": 0.0,
        "loss_rating": 0.0,
        "loss_expected": 0.0,
        "loss_residual": 0.0,
        "loss_contrastive": 0.0,
        "rating_rmse": 0.0,
        "expected_cp_rmse": 0.0,
        "residual_cp_rmse": 0.0,
        "residual_cp_rmse_clipped_300": 0.0,
        "phase_residual_rmse": 0.0,
    }
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            backbone_outputs, head_outputs = model(batch)
            loss, metrics = total_loss(backbone_outputs, head_outputs, batch, loss_weights)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        for key, value in metrics.items():
            running[key] += value
        eval_metrics = {}
        eval_metrics.update(rating_rmse(head_outputs, batch))
        eval_metrics.update(engine_target_metrics(head_outputs, batch))
        running["rating_rmse"] += eval_metrics["rating_rmse"]
        running["expected_cp_rmse"] += eval_metrics["expected_cp_rmse"]
        running["residual_cp_rmse"] += eval_metrics["residual_cp_rmse"]
        running["residual_cp_rmse_clipped_300"] += eval_metrics["residual_cp_rmse_clipped_300"]
        running["phase_residual_rmse"] += eval_metrics["phase_residual_rmse"]
        num_batches += 1

    if num_batches == 0:
        return running
    return {key: value / num_batches for key, value in running.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the chess style transformer.")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    device = get_device()

    train_dataset = ChessWindowDataset(
        config.data_path,
        split=config.split_train,
        window_size=config.window_size,
        stride=config.stride,
        min_window_length=config.min_window_length,
    )
    val_dataset = ChessWindowDataset(
        config.data_path,
        split=config.split_val,
        window_size=config.window_size,
        stride=config.stride,
        min_window_length=config.min_window_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PlayerAwareBatchSampler(
            train_dataset,
            batch_size=config.batch_size,
        ),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = ChessTrainingModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("Using device:", device)
    print("Train windows:", len(train_dataset))
    print("Val windows:", len(val_dataset))

    history = []
    for epoch in range(1, config.num_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config.loss_weights,
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            config.loss_weights,
            train=False,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(f"Epoch {epoch}: train={train_metrics} val={val_metrics}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "history": history,
        },
        output_dir / "model.pt",
    )
    print("Saved checkpoint:", output_dir / "model.pt")


if __name__ == "__main__":
    main()
