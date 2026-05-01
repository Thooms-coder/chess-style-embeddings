import torch


def masked_rmse(predictions, targets, mask):
    valid_mask = mask & torch.isfinite(targets)
    if valid_mask.sum() == 0:
        return 0.0
    mse = ((predictions[valid_mask] - targets[valid_mask]) ** 2).mean()
    return float(torch.sqrt(mse).detach().cpu())


def rating_rmse(outputs, batch):
    white_rmse = torch.sqrt(
        ((outputs["white_rating"] - batch["white_rating"].float()) ** 2).mean()
    )
    black_rmse = torch.sqrt(
        ((outputs["black_rating"] - batch["black_rating"].float()) ** 2).mean()
    )
    return {
        "white_rating_rmse": float(white_rmse.detach().cpu()),
        "black_rating_rmse": float(black_rmse.detach().cpu()),
        "rating_rmse": float(((white_rmse + black_rmse) * 0.5).detach().cpu()),
    }


def masked_rmse_clipped(predictions, targets, mask, clip_value):
    valid_mask = mask & torch.isfinite(targets)
    if valid_mask.sum() == 0:
        return 0.0
    clipped_predictions = torch.clamp(predictions[valid_mask], -clip_value, clip_value)
    clipped_targets = torch.clamp(targets[valid_mask], -clip_value, clip_value)
    mse = ((clipped_predictions - clipped_targets) ** 2).mean()
    return float(torch.sqrt(mse).detach().cpu())


def phase_residual_rmse(outputs, batch, clip_value=300.0):
    white_mask = batch["white_phase_mask"]
    black_mask = batch["black_phase_mask"]
    white_rmse = masked_rmse_clipped(
        outputs["white_phase_residual"],
        batch["white_phase_residual"],
        white_mask,
        clip_value=clip_value,
    )
    black_rmse = masked_rmse_clipped(
        outputs["black_phase_residual"],
        batch["black_phase_residual"],
        black_mask,
        clip_value=clip_value,
    )
    return {
        "phase_residual_rmse": 0.5 * (white_rmse + black_rmse),
        "white_phase_residual_rmse": white_rmse,
        "black_phase_residual_rmse": black_rmse,
    }


def engine_target_metrics(outputs, batch):
    engine_mask = batch["attention_mask"] & batch["move_engine_valid"]
    metrics = {
        "expected_cp_rmse": masked_rmse(
            outputs["expected_quality"], batch["move_expected_cp_loss"], engine_mask
        ),
        "residual_cp_rmse": masked_rmse(
            outputs["residual"], batch["move_residual_cp_loss"], engine_mask
        ),
        "residual_cp_rmse_clipped_300": masked_rmse_clipped(
            outputs["residual"], batch["move_residual_cp_loss"], engine_mask, clip_value=300.0
        ),
        "valid_engine_targets": int(engine_mask.sum().detach().cpu()),
    }
    metrics.update(phase_residual_rmse(outputs, batch))
    return metrics
