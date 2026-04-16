# Chess Style Embeddings: Current Results

## Goal

Learn player representations and weakness signals that are **invariant to rating**.

Core idea:
- model expected move quality given rating
- compute residual weakness relative to that expectation
- learn player embeddings that capture stable structure beyond raw skill

## Data Pipeline

Completed pipeline:
- raw Lichess PGN `.zst` parsing
- filters for rated Standard `Rapid` and `Classical`
- player rating filter `1200` to `2200`
- hashed player IDs
- chronological `train` / `val` / `test` splits
- structured move encoding
- Stockfish centipawn-loss annotation
- expected-quality baselines by rating bucket and phase
- residual target generation
- fixed `128`-ply windows for training

Largest current annotated dataset:
- [data/processed/games_sample_25000_stockfish.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/data/processed/games_sample_25000_stockfish.parquet)

## Modeling

Implemented:
- transformer encoder over move windows
- pooled white/black player embeddings
- rating prediction head
- expected move-quality head
- phase-aggregated residual weakness head
- contrastive player objective

Important modeling change:
- raw per-move residual regression was too noisy
- switched to **phase-aggregated residual weakness** per player-side per window
- this became the first residual formulation that clearly learned useful signal

## Best Current Training Run

Checkpoint:
- [experiments/run2/model.pt](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/model.pt)

Dataset:
- `25000` annotated games
- `16542` train windows
- `3527` validation windows

Final validation metrics:
- rating RMSE: `35.13`
- expected CP RMSE: `2.58`
- phase residual RMSE: `45.76`

Comparison across scale:
- `5000` games: `49.10`
- `10000` games: `47.07`
- `25000` games: `45.76`

## Evaluation Results

### 1. Beyond Rating Alone

Files:
- [embedding_quality_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/embedding_quality/embedding_quality_summary.parquet)
- [embedding_vs_rating_val.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/embedding_quality/embedding_vs_rating_val.parquet)

Validation result:
- players compared: `6528`
- mean embedding-neighbor RMSE: `55.06`
- mean rating-neighbor RMSE: `61.98`
- embedding beats rating rate: `57.38%`

Interpretation:
- embeddings capture player phase-weakness structure better than rating-only matching
- this is the strongest current evidence for the “beyond rating alone” claim

### 2. Temporal Residual Stability

Files:
- [temporal_stability_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/stability/temporal_stability_summary.parquet)

Results:
- `train -> val`
  - players compared: `2068`
  - temporal cosine: `0.2516`
  - random cosine: `0.2036`
  - temporal RMSE: `32.08`
  - random RMSE: `33.58`
- `val -> test`
  - players compared: `1516`
  - temporal cosine: `0.2068`
  - random cosine: `0.1565`
  - temporal RMSE: `35.62`
  - random RMSE: `36.80`
- `train -> test`
  - players compared: `1420`
  - temporal cosine: `0.2178`
  - random cosine: `0.1873`
  - temporal RMSE: `32.55`
  - random RMSE: `34.50`

Interpretation:
- short-horizon stability is clearly present
- long-horizon stability is now also directionally positive on the larger dataset

### 3. Personalization Validity

Files:
- [train_to_val_min1 summary](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/personalization/train_to_val_min1/personalization_summary.parquet)
- [train_to_test_min1 summary](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/personalization/train_to_test_min1/personalization_summary.parquet)

Results:
- `train -> val`
  - players compared: `2386`
  - personal cosine: `0.0802`
  - random cosine: `0.0549`
  - personal RMSE: `46.66`
  - random RMSE: `47.30`
- `train -> test`
  - not yet rerun on the `25k` checkpoint in the current summary

Interpretation:
- same-player future weakness is now better than random on both cosine and RMSE in the short-horizon `train -> val` setting
- the personalization effect is still modest, but stronger than on the `10k` run

## Visualization Artifacts

- [PCA embeddings](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/pca_embeddings.html)
- [t-SNE embeddings](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/tsne_embeddings.html)
- [Residual heatmap](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/residual_heatmap.png)

## Recommended Presentation Narrative

Use this claim structure:

1. Rating-aware preprocessing and engine labels allow residual weakness modeling.
2. Raw move-level residuals are too noisy, but phase-aggregated residual weakness is learnable.
3. The learned embeddings capture structure beyond rating alone.
4. Weakness profiles show short-term temporal stability.
5. Personalized future prediction is positive but still modest.

## Honest Limitations

- dataset is now `25k` annotated games, which is much stronger than the original prototype but still small relative to the full monthly pool
- personalization validity is positive but still not strong enough for a high-confidence deployment claim
- evaluation is strongest for relative structure, not precise future weakness forecasting

## Best One-Sentence Conclusion

The project successfully moved from isolated mistake detection to **rating-aware, phase-level weakness modeling**, with growing evidence that embeddings capture player-specific structure beyond rating and that weakness profiles remain meaningfully stable over time, though personalized forecasting remains modest.
