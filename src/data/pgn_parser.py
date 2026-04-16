import hashlib
import io

import chess.pgn
import pandas as pd
import zstandard as zstd
from tqdm import tqdm


class PGNParser:
    def __init__(
        self,
        input_file,
        max_games=100000,
        min_rating=1200,
        max_rating=2200,
        allowed_time_controls=("Rapid", "Classical"),
        min_moves=10,
    ):
        self.input_file = input_file
        self.max_games = max_games
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.allowed_time_controls = allowed_time_controls
        self.min_moves = min_moves

    def _hash_player(self, username):
        if not username or username == "?":
            return None
        return hashlib.sha256(username.strip().lower().encode("utf-8")).hexdigest()

    def _valid_rating(self, rating):
        try:
            r = int(rating)
            return self.min_rating <= r <= self.max_rating
        except (TypeError, ValueError):
            return False

    def _normalize_time_class(self, headers):
        time_class = headers.get("TimeClass")
        if time_class:
            return time_class

        event = headers.get("Event", "")
        for candidate in ("Bullet", "Blitz", "Rapid", "Classical", "Correspondence"):
            if candidate.lower() in event.lower():
                return candidate

        return None

    def _parse_datetime(self, headers):
        date_value = headers.get("UTCDate") or headers.get("Date")
        time_value = headers.get("UTCTime") or headers.get("Time") or "00:00:00"

        if not date_value or "?" in date_value:
            return None

        date_str = date_value.replace(".", "-")
        datetime_str = f"{date_str} {time_value}"

        return pd.to_datetime(datetime_str, errors="coerce", utc=True)

    def _result_to_label(self, result):
        if result == "1-0":
            return "white_win"
        if result == "0-1":
            return "black_win"
        if result == "1/2-1/2":
            return "draw"
        return "unknown"

    def _parse_moves(self, game):
        moves = []
        node = game

        while not node.is_end():
            node = node.variation(0)
            moves.append(node.move.uci())

        return moves

    def _assign_chronological_split(self, df):
        if df.empty:
            return df

        df = df.sort_values(["game_datetime", "game_id"], na_position="last").reset_index(
            drop=True
        )

        train_end = int(len(df) * 0.7)
        val_end = int(len(df) * 0.85)

        split = pd.Series("test", index=df.index)
        split.iloc[:train_end] = "train"
        split.iloc[train_end:val_end] = "val"
        df["split"] = split

        return df

    def parse(self):

        games = []
        parsed = 0
        total_seen = 0

        with open(self.input_file, "rb") as f:

            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

            pbar = tqdm(total=self.max_games, desc="Parsing games")

            while parsed < self.max_games:

                game = chess.pgn.read_game(text_stream)

                if game is None:
                    break

                total_seen += 1

                headers = game.headers

                white_rating = headers.get("WhiteElo")
                black_rating = headers.get("BlackElo")

                if not (
                    self._valid_rating(white_rating)
                    and self._valid_rating(black_rating)
                ):
                    continue

                time_class = self._normalize_time_class(headers)

                if self.allowed_time_controls:
                    if time_class not in self.allowed_time_controls:
                        continue

                moves = self._parse_moves(game)

                if len(moves) < self.min_moves:
                    continue

                game_datetime = self._parse_datetime(headers)

                games.append(
                    {
                        "game_id": total_seen,
                        "source_file": self.input_file,
                        "site": headers.get("Site"),
                        "event": headers.get("Event"),
                        "round": headers.get("Round"),
                        "result": headers.get("Result"),
                        "result_label": self._result_to_label(headers.get("Result")),
                        "termination": headers.get("Termination"),
                        "time_control": headers.get("TimeControl"),
                        "time_class": time_class,
                        "game_date": headers.get("Date"),
                        "utc_date": headers.get("UTCDate"),
                        "utc_time": headers.get("UTCTime"),
                        "game_datetime": game_datetime,
                        "white_player_hash": self._hash_player(headers.get("White")),
                        "black_player_hash": self._hash_player(headers.get("Black")),
                        "white_rating": int(white_rating),
                        "black_rating": int(black_rating),
                        "moves": moves,
                        "num_moves": len(moves),
                    }
                )

                parsed += 1
                pbar.update(1)

            pbar.close()

        df = pd.DataFrame(games)
        df = self._assign_chronological_split(df)

        return df
