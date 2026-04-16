from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ChessWindowDataset
from src.train.config import TrainConfig
from src.train.trainer import ChessTrainingModel, collate_batch, get_device, move_batch_to_device


DATA_PATH = ROOT / "data/processed/games_sample_25000_stockfish_trainonly.parquet"
CHECKPOINT_PATH = ROOT / "experiments/run2/model.pt"
FINAL_METRICS_PATH = ROOT / "experiments/run2/final_demo/train_val_test_metrics.parquet"
BASELINE_PATH = ROOT / "experiments/run2/baselines_25000_trainonly/baseline_summary_train_to_val.parquet"
EMBED_QUALITY_PATH = ROOT / "experiments/run2/embedding_quality/embedding_quality_summary.parquet"
STABILITY_PATH = ROOT / "experiments/run2/stability_trainonly/temporal_stability_summary.parquet"
PERSONALIZATION_PATH = ROOT / "experiments/run2/personalization/train_to_val_min1/personalization_summary.parquet"
PROJECTION_PATH = ROOT / "experiments/run2/plots/embedding_projection.parquet"
EMBEDDINGS_PATH = ROOT / "experiments/run2/player_embeddings_val.parquet"

PHASE_NAMES = ["Opening", "Middlegame", "Endgame"]


st.set_page_config(
    page_title="Chess Weakness Modeling Demo",
    page_icon="♟️",
    layout="wide",
)


@st.cache_data
def load_artifact(path: str):
    return pd.read_parquet(path)


@st.cache_resource
def load_model_and_config():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config = TrainConfig(**checkpoint["config"])
    model = ChessTrainingModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = get_device()
    model = model.to(device)
    return model, config, device


@st.cache_resource
def load_dataset(split: str):
    _, config, _ = load_model_and_config()
    return ChessWindowDataset(
        str(DATA_PATH),
        split=split,
        window_size=config.window_size,
        stride=config.stride,
        min_window_length=config.min_window_length,
    )


def truncate_hash(value: str, width: int = 10):
    if value is None:
        return "n/a"
    return f"{value[:width]}..."


def build_phase_comparison(pred_values, true_values, mask_values, side_label):
    rows = []
    for phase_name, pred, true, mask in zip(PHASE_NAMES, pred_values, true_values, mask_values):
        rows.append(
            {
                "phase": phase_name,
                "series": f"{side_label} predicted",
                "value": float(pred),
            }
        )
        rows.append(
            {
                "phase": phase_name,
                "series": f"{side_label} observed",
                "value": float(true) if bool(mask) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def predict_sample(split: str, sample_index: int):
    model, _, device = load_model_and_config()
    dataset = load_dataset(split)
    sample = dataset[sample_index]
    batch = collate_batch([sample])
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        _, outputs = model(batch)

    result = {
        "sample": sample,
        "white_rating_pred": float(outputs["white_rating"].detach().cpu()[0]),
        "black_rating_pred": float(outputs["black_rating"].detach().cpu()[0]),
        "white_phase_pred": outputs["white_phase_residual"].detach().cpu()[0].tolist(),
        "black_phase_pred": outputs["black_phase_residual"].detach().cpu()[0].tolist(),
    }
    return result


def cosine_neighbors(embeddings_df: pd.DataFrame, player_hash: str, side: str, top_k: int = 5):
    selected = embeddings_df[
        (embeddings_df["player_hash"] == player_hash) & (embeddings_df["side"] == side)
    ]
    if selected.empty:
        return None

    target = np.asarray(selected.iloc[0]["embedding"], dtype=float)
    matrix = np.asarray(embeddings_df["embedding"].tolist(), dtype=float)
    target_norm = np.linalg.norm(target)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = np.clip(target_norm * matrix_norm, 1e-8, None)
    similarities = (matrix @ target) / denom

    neighbors = embeddings_df.copy()
    neighbors["cosine_similarity"] = similarities
    neighbors = neighbors[
        ~((neighbors["player_hash"] == player_hash) & (neighbors["side"] == side))
    ].sort_values("cosine_similarity", ascending=False)
    return neighbors.head(top_k)


def render_overview():
    metrics_df = load_artifact(str(FINAL_METRICS_PATH))
    baseline_df = load_artifact(str(BASELINE_PATH))
    embed_quality_df = load_artifact(str(EMBED_QUALITY_PATH))
    stability_df = load_artifact(str(STABILITY_PATH))
    personalization_df = load_artifact(str(PERSONALIZATION_PATH))

    st.subheader("Clean 25k Run")
    col1, col2, col3, col4 = st.columns(4)
    val_row = metrics_df[metrics_df["split"] == "val"].iloc[0]
    with col1:
        st.metric("Validation Phase RMSE", f"{val_row['phase_residual_rmse']:.2f}")
    with col2:
        st.metric("Validation Rating RMSE", f"{val_row['rating_rmse']:.2f}")
    with col3:
        st.metric("Validation Expected CP RMSE", f"{val_row['expected_cp_rmse']:.2f}")
    with col4:
        st.metric("Test Phase RMSE", f"{metrics_df[metrics_df['split'] == 'test'].iloc[0]['phase_residual_rmse']:.2f}")

    st.markdown("**Train / Validation / Test**")
    st.dataframe(
        metrics_df[
            [
                "split",
                "num_windows",
                "rating_rmse",
                "expected_cp_rmse",
                "phase_residual_rmse",
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Baseline Comparison on Validation**")
    baseline_plot = go.Figure()
    baseline_plot.add_bar(
        x=baseline_df["model"],
        y=baseline_df["phase_residual_rmse"],
        marker_color=["#c7d2fe", "#93c5fd", "#60a5fa"],
        name="Baselines",
    )
    baseline_plot.add_bar(
        x=["transformer"],
        y=[val_row["phase_residual_rmse"]],
        marker_color="#111827",
        name="Transformer",
    )
    baseline_plot.update_layout(
        height=360,
        yaxis_title="Phase Residual RMSE",
        barmode="group",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(baseline_plot, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Beyond Rating Alone**")
        eq = embed_quality_df.iloc[0]
        st.write(
            f"Embeddings beat rating-only nearest neighbors on weakness-profile RMSE "
            f"({eq['mean_embedding_neighbor_rmse']:.2f} vs {eq['mean_rating_neighbor_rmse']:.2f}) "
            f"with a win rate of {eq['embedding_beats_rating_rate'] * 100:.1f}%."
        )
        st.dataframe(embed_quality_df.round(3), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Stability and Personalization**")
        st.dataframe(stability_df.round(3), use_container_width=True, hide_index=True)
        st.dataframe(personalization_df.round(3), use_container_width=True, hide_index=True)


def render_sample_explorer():
    split = st.selectbox("Split", ["test", "val", "train"], index=0)
    dataset = load_dataset(split)
    curated_indices = [0, 1, 2, 3, 4, 10, 25, 50, 100]
    max_index = len(dataset) - 1
    curated_indices = [idx for idx in curated_indices if idx <= max_index]

    selected_mode = st.radio("Selection mode", ["Curated sample", "Manual index"], horizontal=True)
    if selected_mode == "Curated sample":
        sample_index = st.selectbox("Sample index", curated_indices, index=0)
    else:
        sample_index = int(st.number_input("Sample index", min_value=0, max_value=max_index, value=0, step=1))

    result = predict_sample(split, sample_index)
    sample = result["sample"]

    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.metric("Game ID", sample["game_id"])
    meta2.metric("Window", f"{sample['window_start']}:{sample['window_end']}")
    meta3.metric("Time Class", sample["time_class"] or "n/a")
    meta4.metric("Result", sample["result_label"] or "n/a")

    st.caption(
        f"White {truncate_hash(sample['white_player_hash'])} | "
        f"Black {truncate_hash(sample['black_player_hash'])}"
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Rating Prediction**")
        rating_rows = pd.DataFrame(
            [
                {"side": "White", "true_rating": sample["white_rating"], "predicted_rating": result["white_rating_pred"]},
                {"side": "Black", "true_rating": sample["black_rating"], "predicted_rating": result["black_rating_pred"]},
            ]
        )
        fig = go.Figure()
        fig.add_bar(name="True", x=rating_rows["side"], y=rating_rows["true_rating"], marker_color="#93c5fd")
        fig.add_bar(name="Predicted", x=rating_rows["side"], y=rating_rows["predicted_rating"], marker_color="#1d4ed8")
        fig.update_layout(height=320, barmode="group", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Move Preview**")
        moves = [move for move in sample["move_uci"] if move != "<pad>"]
        st.code(" ".join(moves[:20]), language="text")
        st.write(f"Window length: `{sample['window_length']}` plies")

    white_df = build_phase_comparison(
        result["white_phase_pred"],
        sample["white_phase_residual"].tolist(),
        sample["white_phase_mask"].tolist(),
        "White",
    )
    black_df = build_phase_comparison(
        result["black_phase_pred"],
        sample["black_phase_residual"].tolist(),
        sample["black_phase_mask"].tolist(),
        "Black",
    )
    phase_df = pd.concat([white_df, black_df], ignore_index=True)

    st.markdown("**Predicted vs Observed Phase Weakness**")
    phase_chart = px.bar(
        phase_df,
        x="phase",
        y="value",
        color="series",
        barmode="group",
        height=420,
        color_discrete_sequence=["#1d4ed8", "#93c5fd", "#111827", "#9ca3af"],
    )
    phase_chart.update_layout(margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Residual weakness")
    st.plotly_chart(phase_chart, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "side": "White",
                    "predicted": ", ".join(f"{name}={value:.2f}" for name, value in zip(PHASE_NAMES, result["white_phase_pred"])),
                    "observed": ", ".join(
                        f"{name}={value:.2f}" if mask else f"{name}=n/a"
                        for name, value, mask in zip(PHASE_NAMES, sample["white_phase_residual"].tolist(), sample["white_phase_mask"].tolist())
                    ),
                },
                {
                    "side": "Black",
                    "predicted": ", ".join(f"{name}={value:.2f}" for name, value in zip(PHASE_NAMES, result["black_phase_pred"])),
                    "observed": ", ".join(
                        f"{name}={value:.2f}" if mask else f"{name}=n/a"
                        for name, value, mask in zip(PHASE_NAMES, sample["black_phase_residual"].tolist(), sample["black_phase_mask"].tolist())
                    ),
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_embedding_explorer():
    projection_df = load_artifact(str(PROJECTION_PATH))
    embeddings_df = load_artifact(str(EMBEDDINGS_PATH))

    projection_mode = st.radio("Projection", ["PCA", "t-SNE"], horizontal=True)
    x_col, y_col = ("pca_x", "pca_y") if projection_mode == "PCA" else ("tsne_x", "tsne_y")

    plot_df = projection_df.copy()
    plot_df["label"] = plot_df.apply(
        lambda row: f"{row['side']} | rating {int(row['rating'])} | {truncate_hash(row['player_hash'])}",
        axis=1,
    )
    scatter = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="rating",
        hover_data=["label", "num_windows"],
        height=500,
        color_continuous_scale="Blues",
    )
    scatter.update_traces(marker=dict(size=6, opacity=0.7))
    scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(scatter, use_container_width=True)

    top_players = embeddings_df.sort_values(["num_windows", "rating"], ascending=[False, False]).copy()
    top_players["display"] = top_players.apply(
        lambda row: f"{row['side']} | rating {int(row['rating'])} | windows {int(row['num_windows'])} | {truncate_hash(row['player_hash'])}",
        axis=1,
    )
    selected_display = st.selectbox("Select a validation embedding", top_players["display"].tolist(), index=0)
    selected_row = top_players[top_players["display"] == selected_display].iloc[0]

    neighbors = cosine_neighbors(
        embeddings_df,
        player_hash=selected_row["player_hash"],
        side=selected_row["side"],
        top_k=5,
    )

    st.markdown("**Selected Player**")
    st.write(
        f"{selected_row['side']} player {truncate_hash(selected_row['player_hash'])}, "
        f"rating {int(selected_row['rating'])}, windows {int(selected_row['num_windows'])}"
    )

    if neighbors is not None:
        display_neighbors = neighbors[["side", "player_hash", "rating", "num_windows", "cosine_similarity"]].copy()
        display_neighbors["player_hash"] = display_neighbors["player_hash"].map(truncate_hash)
        st.markdown("**Nearest Embedding Neighbors**")
        st.dataframe(display_neighbors.round(4), use_container_width=True, hide_index=True)


def main():
    st.title("Chess Weakness Modeling Demo")
    st.write(
        "Interactive showcase for the leak-free 25k-game transformer run: "
        "results, held-out sample predictions, and embedding structure."
    )

    tab1, tab2, tab3 = st.tabs(["Results Overview", "Sample Explorer", "Embedding Explorer"])
    with tab1:
        render_overview()
    with tab2:
        render_sample_explorer()
    with tab3:
        render_embedding_explorer()

    st.caption(
        "Artifacts: clean 25k train-only relabeled dataset, run2 checkpoint, "
        "baseline comparisons, temporal stability, personalization, and embedding projections."
    )


if __name__ == "__main__":
    main()
