import torch
import torch.nn.functional as F


def masked_mse_loss(predictions, targets, mask):
    valid_mask = mask & torch.isfinite(targets)
    if valid_mask.sum() == 0:
        return predictions.new_tensor(0.0)
    diff = predictions[valid_mask] - targets[valid_mask]
    return (diff ** 2).mean()


def masked_huber_loss(predictions, targets, mask, delta=1.0):
    valid_mask = mask & torch.isfinite(targets)
    if valid_mask.sum() == 0:
        return predictions.new_tensor(0.0)
    return F.smooth_l1_loss(predictions[valid_mask], targets[valid_mask], beta=delta)


def phase_residual_loss(outputs, batch, cp_scale=100.0, residual_clip=300.0, huber_delta=1.0):
    white_targets = torch.clamp(
        batch["white_phase_residual"],
        min=-residual_clip,
        max=residual_clip,
    )
    black_targets = torch.clamp(
        batch["black_phase_residual"],
        min=-residual_clip,
        max=residual_clip,
    )
    white_predictions = torch.clamp(
        outputs["white_phase_residual"],
        min=-residual_clip,
        max=residual_clip,
    )
    black_predictions = torch.clamp(
        outputs["black_phase_residual"],
        min=-residual_clip,
        max=residual_clip,
    )
    white_loss = masked_huber_loss(
        white_predictions / cp_scale,
        white_targets / cp_scale,
        batch["white_phase_mask"],
        delta=huber_delta,
    )
    black_loss = masked_huber_loss(
        black_predictions / cp_scale,
        black_targets / cp_scale,
        batch["black_phase_mask"],
        delta=huber_delta,
    )
    return 0.5 * (white_loss + black_loss)


def rating_loss(outputs, batch, rating_scale=1000.0):
    white_targets = batch["white_rating"].float() / rating_scale
    black_targets = batch["black_rating"].float() / rating_scale
    white_predictions = outputs["white_rating"] / rating_scale
    black_predictions = outputs["black_rating"] / rating_scale
    white_loss = F.mse_loss(white_predictions, white_targets)
    black_loss = F.mse_loss(black_predictions, black_targets)
    return 0.5 * (white_loss + black_loss)


def token_losses(outputs, batch, cp_scale=100.0, residual_clip=300.0, huber_delta=1.0):
    engine_mask = batch["attention_mask"] & batch["move_engine_valid"]
    expected_loss = masked_huber_loss(
        outputs["expected_quality"] / cp_scale,
        batch["move_expected_cp_loss"] / cp_scale,
        engine_mask,
        delta=huber_delta,
    )
    residual_loss = phase_residual_loss(
        outputs,
        batch,
        cp_scale=cp_scale,
        residual_clip=residual_clip,
        huber_delta=huber_delta,
    )
    return expected_loss, residual_loss


def contrastive_player_loss(backbone_outputs, batch, temperature=0.1):
    losses = []
    for side_name, embedding_key, hash_key in (
        ("white", "white_embedding", "white_player_hash"),
        ("black", "black_embedding", "black_player_hash"),
    ):
        embeddings = F.normalize(backbone_outputs[embedding_key], dim=-1)
        hashes = batch[hash_key]
        if len(hashes) < 2:
            continue

        labels = torch.tensor(
            [0 if h is None else hash(h) for h in hashes],
            device=embeddings.device,
            dtype=torch.long,
        )
        similarity = embeddings @ embeddings.t() / temperature
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask.fill_diagonal_(False)

        for idx in range(similarity.size(0)):
            positives = similarity[idx][positive_mask[idx]]
            if positives.numel() == 0:
                continue
            denominator = torch.logsumexp(similarity[idx], dim=0)
            losses.append(-(positives - denominator).mean())

    if not losses:
        return backbone_outputs["sequence_embedding"].new_tensor(0.0)

    return torch.stack(losses).mean()


def total_loss(backbone_outputs, outputs, batch, weights=None):
    weights = weights or {}
    rating_weight = weights.get("rating", 1.0)
    expected_weight = weights.get("expected", 1.0)
    residual_weight = weights.get("residual", 1.0)
    contrastive_weight = weights.get("contrastive", 0.1)
    rating_scale = weights.get("rating_scale", 1000.0)
    cp_scale = weights.get("cp_scale", 100.0)
    residual_clip = weights.get("residual_clip", 300.0)
    huber_delta = weights.get("huber_delta", 1.0)

    loss_rating = rating_loss(outputs, batch, rating_scale=rating_scale)
    loss_expected, loss_residual = token_losses(
        outputs,
        batch,
        cp_scale=cp_scale,
        residual_clip=residual_clip,
        huber_delta=huber_delta,
    )
    loss_contrastive = contrastive_player_loss(backbone_outputs, batch)

    loss = (
        rating_weight * loss_rating
        + expected_weight * loss_expected
        + residual_weight * loss_residual
        + contrastive_weight * loss_contrastive
    )
    return loss, {
        "loss_total": float(loss.detach().cpu()),
        "loss_rating": float(loss_rating.detach().cpu()),
        "loss_expected": float(loss_expected.detach().cpu()),
        "loss_residual": float(loss_residual.detach().cpu()),
        "loss_contrastive": float(loss_contrastive.detach().cpu()),
    }
