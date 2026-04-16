import torch
from torch import nn


class ChessStyleTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_sequence_length=128,
        max_rating_bucket=32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.from_square_embedding = nn.Embedding(65, hidden_dim)
        self.to_square_embedding = nn.Embedding(65, hidden_dim)
        self.promotion_embedding = nn.Embedding(8, hidden_dim)
        self.piece_type_embedding = nn.Embedding(8, hidden_dim)
        self.capture_embedding = nn.Embedding(2, hidden_dim)
        self.check_embedding = nn.Embedding(2, hidden_dim)
        self.side_embedding = nn.Embedding(2, hidden_dim)
        self.phase_embedding = nn.Embedding(3, hidden_dim)
        self.rating_bucket_embedding = nn.Embedding(max_rating_bucket, hidden_dim)
        self.time_class_embedding = nn.Embedding(3, hidden_dim)
        self.result_embedding = nn.Embedding(5, hidden_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)

        self.numeric_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _masked_mean(self, embeddings, mask):
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (embeddings * mask).sum(dim=1) / denom

    def forward(self, batch):
        from_square = batch["from_square"]
        to_square = batch["to_square"]
        promotion = batch["promotion"]
        piece_type = batch["piece_type"]
        is_capture = batch["is_capture"]
        is_check = batch["is_check"]
        moving_side = batch["moving_side"]
        phase_id = batch["phase_id"]
        rating_bucket = batch["rating_bucket"].clamp(min=0, max=self.rating_bucket_embedding.num_embeddings - 1)
        time_class_id = batch["time_class_id"].clamp(min=0, max=self.time_class_embedding.num_embeddings - 1)
        result_id = batch["result_id"].clamp(min=0, max=self.result_embedding.num_embeddings - 1)
        attention_mask = batch["attention_mask"]

        batch_size, seq_len = from_square.shape
        position_ids = torch.arange(seq_len, device=from_square.device).unsqueeze(0).expand(batch_size, -1)

        token_embeddings = (
            self.from_square_embedding(from_square)
            + self.to_square_embedding(to_square)
            + self.promotion_embedding(promotion)
            + self.piece_type_embedding(piece_type)
            + self.capture_embedding(is_capture)
            + self.check_embedding(is_check)
            + self.side_embedding(moving_side)
            + self.phase_embedding(phase_id)
            + self.rating_bucket_embedding(rating_bucket)
            + self.time_class_embedding(time_class_id)
            + self.result_embedding(result_id)
            + self.position_embedding(position_ids)
        )

        numeric_features = torch.stack(
            [
                batch["ply_index"].float(),
                batch["player_rating"].float(),
                batch["opponent_rating"].float(),
            ],
            dim=-1,
        )
        token_embeddings = token_embeddings + self.numeric_projection(numeric_features)
        token_embeddings = self.dropout(self.input_norm(token_embeddings))

        encoded = self.encoder(
            token_embeddings,
            src_key_padding_mask=~attention_mask,
        )
        encoded = self.output_norm(encoded)

        white_mask = attention_mask & (moving_side == 0)
        black_mask = attention_mask & (moving_side == 1)
        white_embedding = self._masked_mean(encoded, white_mask)
        black_embedding = self._masked_mean(encoded, black_mask)

        white_context = white_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        black_context = black_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        player_context = torch.where(
            moving_side.unsqueeze(-1) == 0,
            white_context,
            black_context,
        )

        return {
            "token_embeddings": encoded,
            "sequence_embedding": self._masked_mean(encoded, attention_mask),
            "white_embedding": white_embedding,
            "black_embedding": black_embedding,
            "player_context": player_context,
            "white_mask": white_mask,
            "black_mask": black_mask,
        }
