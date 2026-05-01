from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.move_tokenizer import MoveTokenizer


@dataclass
class WindowSpec:
    window_size: int = 128
    stride: int = 64
    min_window_length: int = 32


class ChessWindowDataset(Dataset):
    def __init__(
        self,
        dataframe,
        split=None,
        window_size=128,
        stride=64,
        min_window_length=32,
        rating_bucket_size=100,
    ):
        if isinstance(dataframe, str):
            dataframe = pd.read_parquet(dataframe)

        self.dataframe = dataframe.copy()
        if split is not None:
            self.dataframe = self.dataframe[self.dataframe["split"] == split].reset_index(
                drop=True
            )

        self.window_spec = WindowSpec(
            window_size=window_size,
            stride=stride,
            min_window_length=min_window_length,
        )
        self.tokenizer = MoveTokenizer(rating_bucket_size=rating_bucket_size)
        self.samples = self._build_samples()

    def _window_starts(self, sequence_length):
        if sequence_length <= self.window_spec.window_size:
            return [0] if sequence_length >= self.window_spec.min_window_length else []

        starts = list(
            range(0, sequence_length - self.window_spec.window_size + 1, self.window_spec.stride)
        )
        last_start = sequence_length - self.window_spec.window_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _pad_window(self, window_moves):
        pad_length = self.window_spec.window_size - len(window_moves)

        feature_names = [
            "from_square",
            "to_square",
            "promotion",
            "piece_type",
            "is_capture",
            "is_check",
            "moving_side",
            "ply_index",
            "phase_id",
            "player_rating",
            "opponent_rating",
            "rating_bucket",
            "time_class_id",
            "result_id",
        ]

        tensors = {}
        for name in feature_names:
            values = [move[name] for move in window_moves]
            values.extend([0] * pad_length)
            tensors[name] = torch.tensor(values, dtype=torch.long)

        move_text = [move["move_uci"] for move in window_moves]
        move_text.extend(["<pad>"] * pad_length)

        tensors["attention_mask"] = torch.tensor(
            [1] * len(window_moves) + [0] * pad_length,
            dtype=torch.bool,
        )
        tensors["move_uci"] = move_text

        return tensors

    def _pad_optional_numeric_window(self, values, pad_value=0.0, dtype=torch.float32):
        pad_length = self.window_spec.window_size - len(values)
        padded_values = list(values) + [pad_value] * pad_length
        return torch.tensor(padded_values, dtype=dtype)

    def _phase_residual_targets(self, window_moves, residual_values=None, valid_values=None):
        phase_targets = {
            "white_phase_residual": [0.0, 0.0, 0.0],
            "black_phase_residual": [0.0, 0.0, 0.0],
            "white_phase_mask": [False, False, False],
            "black_phase_mask": [False, False, False],
        }
        if residual_values is None or valid_values is None:
            return phase_targets

        buckets = {
            "white": {0: [], 1: [], 2: []},
            "black": {0: [], 1: [], 2: []},
        }
        for move, residual, is_valid in zip(window_moves, residual_values, valid_values):
            if not is_valid or residual is None:
                continue
            side_name = "white" if move["moving_side"] == 0 else "black"
            buckets[side_name][move["phase_id"]].append(float(residual))

        for side_name in ("white", "black"):
            for phase_id in range(3):
                values = buckets[side_name][phase_id]
                if values:
                    phase_targets[f"{side_name}_phase_residual"][phase_id] = sum(values) / len(values)
                    phase_targets[f"{side_name}_phase_mask"][phase_id] = True

        return phase_targets

    def _build_samples(self):
        samples = []

        for row in self.dataframe.itertuples(index=False):
            encoded_moves = self.tokenizer.encode_game(
                moves=row.moves,
                white_rating=row.white_rating,
                black_rating=row.black_rating,
                time_class=getattr(row, "time_class", None),
                result_label=getattr(row, "result_label", "unknown"),
            )

            for start in self._window_starts(len(encoded_moves)):
                end = min(start + self.window_spec.window_size, len(encoded_moves))
                window_moves = encoded_moves[start:end]
                if len(window_moves) < self.window_spec.min_window_length:
                    continue

                padded = self._pad_window(window_moves)
                sample = {
                    **padded,
                    "game_id": int(row.game_id),
                    "window_start": start,
                    "window_end": end,
                    "window_length": len(window_moves),
                    "split": row.split,
                    "time_class": getattr(row, "time_class", None),
                    "event": getattr(row, "event", None),
                    "result_label": getattr(row, "result_label", None),
                    "game_datetime": getattr(row, "game_datetime", None),
                    "white_player_hash": getattr(row, "white_player_hash", None),
                    "black_player_hash": getattr(row, "black_player_hash", None),
                    "white_rating": int(row.white_rating),
                    "black_rating": int(row.black_rating),
                }

                optional_sequence_columns = {
                    "move_cp_loss": torch.float32,
                    "move_expected_cp_loss": torch.float32,
                    "move_residual_cp_loss": torch.float32,
                }

                for column_name, dtype in optional_sequence_columns.items():
                    if hasattr(row, column_name):
                        raw_values = list(getattr(row, column_name)[start:end])
                        normalized_values = [
                            0.0 if value is None else float(value) for value in raw_values
                        ]
                        sample[column_name] = self._pad_optional_numeric_window(
                            normalized_values,
                            pad_value=0.0,
                            dtype=dtype,
                        )

                if hasattr(row, "move_engine_valid"):
                    valid_values = list(getattr(row, "move_engine_valid")[start:end])
                    valid_values.extend([False] * (self.window_spec.window_size - len(valid_values)))
                    sample["move_engine_valid"] = torch.tensor(valid_values, dtype=torch.bool)
                else:
                    valid_values = None

                residual_values = None
                if hasattr(row, "move_residual_cp_loss"):
                    residual_values = list(getattr(row, "move_residual_cp_loss")[start:end])

                phase_targets = self._phase_residual_targets(
                    window_moves,
                    residual_values=residual_values,
                    valid_values=valid_values[: len(window_moves)] if valid_values is not None else None,
                )
                sample["white_phase_residual"] = torch.tensor(
                    phase_targets["white_phase_residual"],
                    dtype=torch.float32,
                )
                sample["black_phase_residual"] = torch.tensor(
                    phase_targets["black_phase_residual"],
                    dtype=torch.float32,
                )
                sample["white_phase_mask"] = torch.tensor(
                    phase_targets["white_phase_mask"],
                    dtype=torch.bool,
                )
                sample["black_phase_mask"] = torch.tensor(
                    phase_targets["black_phase_mask"],
                    dtype=torch.bool,
                )

                samples.append(
                    sample
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
