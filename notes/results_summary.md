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
- [data/processed/games_sample_25000_stockfish_trainonly.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/data/processed/games_sample_25000_stockfish_trainonly.parquet)

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
- rating RMSE: `37.53`
- expected CP RMSE: `2.28`
- phase residual RMSE: `47.12`

Comparison across scale:
- `5000` games: `49.10`
- `10000` games: `47.07`
- `25000` games: `47.12`

Leak-free baseline comparison on `25000` validation:
- `rating_heuristic`: `67.77`
- `ridge`: `58.85`
- `hist_gbdt`: `49.63`
- transformer: `47.12`

## Evaluation Results

### 1. Beyond Rating Alone

Files:
- [embedding_quality_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/embedding_quality/embedding_quality_summary.parquet)
- [embedding_vs_rating_val.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/embedding_quality/embedding_vs_rating_val.parquet)

Validation result:
- players compared: `6528`
- mean embedding-neighbor RMSE: `54.80`
- mean rating-neighbor RMSE: `61.99`
- embedding beats rating rate: `57.7%`

Interpretation:
- embeddings capture player phase-weakness structure better than rating-only matching
- this is the strongest current evidence for the “beyond rating alone” claim

### 2. Temporal Residual Stability

Files:
- [temporal_stability_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/stability_trainonly/temporal_stability_summary.parquet)

Results:
- `train -> val`
  - players compared: `2068`
  - temporal cosine: `0.2551`
  - random cosine: `0.2043`
  - temporal RMSE: `32.08`
  - random RMSE: `33.57`
- `val -> test`
  - players compared: `1516`
  - temporal cosine: `0.2085`
  - random cosine: `0.1576`
  - temporal RMSE: `35.65`
  - random RMSE: `36.82`
- `train -> test`
  - players compared: `1420`
  - temporal cosine: `0.2191`
  - random cosine: `0.1890`
  - temporal RMSE: `32.57`
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
  - personal cosine: `0.1109`
  - random cosine: `0.0702`
  - personal RMSE: `45.36`
  - random RMSE: `46.16`

Interpretation:
- same-player future weakness is better than random on both cosine and RMSE in the short-horizon `train -> val` setting
- the personalization effect is still modest, but now cleaner after fixing residual-label leakage

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
- fixing residual-label leakage reduced the headline model metrics, but made the claims substantially more defensible
- personalization validity is positive but still not strong enough for a high-confidence deployment claim
- evaluation is strongest for relative structure, not precise future weakness forecasting

## Best One-Sentence Conclusion

The project successfully moved from isolated mistake detection to **rating-aware, phase-level weakness modeling**, with growing evidence that embeddings capture player-specific structure beyond rating and that weakness profiles remain meaningfully stable over time, though personalized forecasting remains modest.
