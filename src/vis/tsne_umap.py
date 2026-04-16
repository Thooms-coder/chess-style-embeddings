import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize player embeddings with projection plots.")
    parser.add_argument(
        "--input-file",
        default="experiments/run1/player_embeddings_val.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/run1/plots",
    )
    parser.add_argument(
        "--min-windows",
        type=int,
        default=1,
    )
    return parser.parse_args()


def sanitize_embeddings(df: pd.DataFrame) -> np.ndarray:
    embeddings = pd.DataFrame(df["embedding"].tolist())
    values = embeddings.to_numpy(dtype=np.float64, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, -10.0, 10.0)

    # Embeddings are already normalized upstream. Center only to avoid
    # exploding low-variance dimensions during per-feature standardization.
    values -= values.mean(axis=0, keepdims=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(values, dtype=np.float64)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_file)
    df = df[df["num_windows"] >= args.min_windows].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 embeddings to project.")

    embedding_values = sanitize_embeddings(df)
    pca = PCA(n_components=2, svd_solver="full")
    pca_coords = pca.fit_transform(embedding_values)

    tsne_input_dims = min(50, embedding_values.shape[1], len(df) - 1)
    tsne_input = embedding_values
    if tsne_input_dims >= 2:
        tsne_input = PCA(n_components=tsne_input_dims, svd_solver="full").fit_transform(embedding_values)

    perplexity = max(5, min(30, len(df) // 10))
    perplexity = min(perplexity, len(df) - 1)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
    )
    tsne_coords = tsne.fit_transform(tsne_input)

    plot_df = df.copy()
    plot_df["pca_x"] = pca_coords[:, 0]
    plot_df["pca_y"] = pca_coords[:, 1]
    plot_df["tsne_x"] = tsne_coords[:, 0]
    plot_df["tsne_y"] = tsne_coords[:, 1]

    projection_path = output_dir / "embedding_projection.parquet"
    plot_df.to_parquet(projection_path, index=False)

    pca_fig = px.scatter(
        plot_df,
        x="pca_x",
        y="pca_y",
        color="rating",
        symbol="side",
        hover_data=["player_hash", "num_windows"],
        title="PCA of Player Embeddings",
    )
    pca_html = output_dir / "pca_embeddings.html"
    pca_fig.write_html(pca_html)

    tsne_fig = px.scatter(
        plot_df,
        x="tsne_x",
        y="tsne_y",
        color="rating",
        symbol="side",
        hover_data=["player_hash", "num_windows"],
        title="t-SNE of Player Embeddings",
    )
    tsne_html = output_dir / "tsne_embeddings.html"
    tsne_fig.write_html(tsne_html)

    print("Saved projections:", projection_path)
    print("Saved plots:", pca_html, tsne_html)
    print("Players visualized:", len(plot_df))


if __name__ == "__main__":
    main()
