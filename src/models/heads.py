import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim=256, residual_clip=300.0):
        super().__init__()
        self.residual_clip = residual_clip
        token_input_dim = hidden_dim * 2
        self.expected_quality_head = nn.Sequential(
            nn.Linear(token_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(token_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.rating_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.phase_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, backbone_outputs):
        token_embeddings = backbone_outputs["token_embeddings"]
        player_context = backbone_outputs["player_context"]
        white_embedding = backbone_outputs["white_embedding"]
        black_embedding = backbone_outputs["black_embedding"]
        token_inputs = torch.cat([token_embeddings, player_context], dim=-1)
        expected_quality = self.expected_quality_head(token_inputs).squeeze(-1)
        residual = self.residual_head(token_inputs).squeeze(-1)

        return {
            "expected_quality": torch.nn.functional.softplus(expected_quality),
            "residual": torch.tanh(residual) * self.residual_clip,
            "white_phase_residual": torch.tanh(self.phase_residual_head(white_embedding)) * self.residual_clip,
            "black_phase_residual": torch.tanh(self.phase_residual_head(black_embedding)) * self.residual_clip,
            "white_rating": self.rating_head(white_embedding).squeeze(-1),
            "black_rating": self.rating_head(black_embedding).squeeze(-1),
        }
