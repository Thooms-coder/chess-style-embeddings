# Final Deliverable

## Title

Latent Style Embeddings and Rating-Conditioned Residual Weakness Modeling for Chess Players

## Deliverable Type

This repository now supports a complete final-project deliverable consisting of:
- a working end-to-end codebase
- a reproducible experimental pipeline
- a written report narrative
- presentation-ready results and artifacts

The strongest deliverable that matches the current state of the project is:

**A research-style final project report and presentation package built around a leak-free 25k-game experiment, with baselines, chronological evaluation, and visualization artifacts.**

## What Was Actually Built

The implemented system is no longer a proposal or scaffold. It now includes:

### 1. Data Pipeline
- parsing rated Lichess Standard PGN archives from `.pgn.zst`
- filtering to `Rapid` and `Classical`
- filtering players to the `1200` to `2200` rating range
- hashing player identities during preprocessing
- storing structured game metadata in parquet
- assigning chronological `train`, `val`, and `test` splits

### 2. Engine Annotation
- Stockfish-based move annotation
- move-level centipawn loss
- expected move quality by rating bucket and phase
- residual move quality labels
- checkpointing and resume support for long annotation runs

### 3. Modeling
- structured move tokenization
- fixed `128`-ply windows
- transformer encoder over move sequences
- pooled player embeddings
- multitask learning for:
  - rating prediction
  - expected move quality prediction
  - phase-level residual weakness prediction
- contrastive player objective

### 4. Evaluation
- embedding vs rating baseline
- temporal stability over chronological splits
- personalization validity
- non-neural baselines:
  - rating heuristic
  - ridge regression
  - histogram gradient-boosted trees

### 5. Outputs
- trained checkpoint
- player embedding parquet
- PCA and t-SNE projections
- residual summary tables
- results summary, abstract, and speaker notes

## Recommended Final Submission Framing

The deliverable should be framed as a **research prototype with empirical results**, not as a fully proven production system.

That framing matches the code and results best:
- the system is complete end to end
- the experiments are real and reproducible
- the core idea is supported empirically
- the personalization result is promising but still limited

## Core Problem Statement

Standard chess analysis tools identify mistakes, but they do not cleanly separate:
- overall player strength
- player-specific structural weakness

This project addresses that gap by modeling **residual weakness relative to rating-conditioned expected performance**, then learning embeddings that capture persistent player-specific structure beyond rating alone.

## Main Contribution

The core contribution is not a new transformer architecture.

The contribution is the **framework**:
- estimate expected move quality conditioned on rating
- compute residual weakness relative to that expectation
- aggregate residual weakness by phase
- learn player embeddings from structured move sequences
- test whether those weakness profiles are stable and personalized over time

## Final Experimental Setup

### Dataset
- source: rated Lichess Standard games
- filtered to `Rapid` and `Classical`
- rating range: `1200` to `2200`
- largest clean annotated dataset: `25,000` games
- leak-free residual labels computed from `train` only

### Train/Validation Scale
- train windows: `16,542`
- validation windows: `3,527`

### Main Model
- transformer encoder over `128`-ply windows
- structured move inputs
- multitask heads for rating, expected quality, and phase residual weakness

### Main Target
- opening residual weakness
- middlegame residual weakness
- endgame residual weakness

This phase-aggregated target replaced raw move-level residual prediction because the raw target was too noisy.

## Best Clean Results

### Main Model
- rating RMSE: `37.53`
- expected CP RMSE: `2.28`
- phase residual RMSE: `47.12`

### Baselines on the Same Leak-Free `25k` Validation Setup
- rating heuristic: `67.77`
- ridge regression: `58.85`
- histogram gradient boosting: `49.63`
- transformer: `47.12`

Interpretation:
- the transformer beats all tested baselines on the main phase-residual target
- the margin over the strongest tree baseline is real but not huge
- that makes the result credible rather than overstated

### Beyond Rating Alone
- players compared: `6528`
- embedding-neighbor RMSE: `54.80`
- rating-neighbor RMSE: `61.99`
- embedding beats rating rate: `57.7%`

Interpretation:
- learned embeddings match weakness profiles better than rating-only similarity

### Temporal Stability
- `train -> val`
  - temporal cosine: `0.2551`
  - random cosine: `0.2043`
  - temporal RMSE: `32.08`
  - random RMSE: `33.57`
- `val -> test`
  - temporal cosine: `0.2085`
  - random cosine: `0.1576`
  - temporal RMSE: `35.65`
  - random RMSE: `36.82`
- `train -> test`
  - temporal cosine: `0.2191`
  - random cosine: `0.1890`
  - temporal RMSE: `32.57`
  - random RMSE: `34.50`

Interpretation:
- weakness profiles are more stable over time than random baselines

### Personalization
- `train -> val`
  - players compared: `2386`
  - personal cosine: `0.1109`
  - random cosine: `0.0702`
  - personal RMSE: `45.36`
  - random RMSE: `46.16`

Interpretation:
- personalization is positive but modest
- this is the weakest part of the system, but it still supports a careful claim

## Strongest Defensible Claim

The strongest final claim supported by the current project is:

**Rating-aware residual modeling combined with transformer-based player embeddings can recover meaningful phase-level behavioral structure beyond rating alone, and those weakness profiles show measurable temporal stability over time.**

## Claims To Avoid

Do not present this as:
- a fundamentally new transformer architecture
- a solved personalization system
- a production-ready coaching product
- a publication-proof result without caveats

Those claims are too strong for the current evidence.

## Best Submission Package

If this is being turned in or presented, the best package is:

### 1. Written Report
Use:
- [notes/results_summary.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/notes/results_summary.md)
- [notes/final_abstract.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/notes/final_abstract.md)

Structure:
- problem
- data and preprocessing
- Stockfish-based residual labeling
- transformer and multitask learning
- baselines
- results
- limitations
- future work

### 2. Presentation
Use:
- [notes/final_speaker_notes.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/notes/final_speaker_notes.md)
- [notes/Blue and White Hand-Drawn Strategy Presentation.pdf](/Users/mutsamungoshi/Desktop/chess-style-embeddings/notes/Blue%20and%20White%20Hand-Drawn%20Strategy%20Presentation.pdf)

### 3. Code Artifacts
- checkpoint: [experiments/run2/model.pt](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/model.pt)
- clean annotated dataset: [data/processed/games_sample_25000_stockfish_trainonly.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/data/processed/games_sample_25000_stockfish_trainonly.parquet)
- baseline summary: [baseline_summary_train_to_val.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/baselines_25000_trainonly/baseline_summary_train_to_val.parquet)
- embedding quality summary: [embedding_quality_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/embedding_quality/embedding_quality_summary.parquet)
- temporal stability summary: [temporal_stability_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/stability_trainonly/temporal_stability_summary.parquet)
- personalization summary: [personalization_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/personalization/train_to_val_min1/personalization_summary.parquet)

### 4. Visual Artifacts
- [PCA embeddings](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/pca_embeddings.html)
- [t-SNE embeddings](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/tsne_embeddings.html)
- [Residual heatmap](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/residual_heatmap.png)

## Final Deliverable Statement

If a single concise deliverable statement is needed, use this:

**This project delivers a leak-free, end-to-end deep learning system for rating-aware chess weakness modeling, including large-scale Lichess preprocessing, Stockfish residual labeling, transformer-based player embeddings, baseline comparisons, and chronological evaluation showing structure beyond rating alone.**

## If You Need To Present It In One Sentence

This is a **research-grade prototype for rating-aware chess weakness analysis**, with a complete pipeline, strong baseline comparisons, and credible evidence that player embeddings capture behavioral structure beyond rating.
