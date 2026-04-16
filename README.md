# Latent Style Embeddings and Residual Weakness Modeling

Rating-aware personalized chess analysis from Lichess game archives.

## Overview

This project studies whether chess weaknesses can be modeled in a way that is more informative than raw rating alone. Instead of treating mistakes as isolated blunders, the pipeline estimates expected move quality for a given rating and then analyzes residual deviations from that expectation.

The implemented system:
- parses rated Lichess Standard PGN archives
- filters to `Rapid` and `Classical` players between `1200` and `2200`
- hashes player identities during preprocessing
- annotates games with Stockfish centipawn-loss labels
- builds fixed `128`-ply windows with structured move features
- trains a transformer to learn player embeddings and phase-level weakness targets
- evaluates whether those embeddings capture structure beyond rating alone

The strongest current result is that the model learns **phase-aggregated residual weakness profiles** that improve with scale and produce player embeddings that outperform rating-only matching on weakness-profile similarity.

## Current Status

The repository is no longer a scaffold. It supports:
- PGN parsing from `.pgn.zst`
- filtered parquet dataset creation
- Stockfish annotation with checkpoints and resume
- windowed PyTorch dataset generation
- transformer training
- embedding export and projection
- temporal stability, embedding-quality, and personalization evaluation

Current best annotated dataset:
- [data/processed/games_sample_25000_stockfish.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/data/processed/games_sample_25000_stockfish.parquet)

Current best checkpoint:
- [experiments/run2/model.pt](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/model.pt)

Project notes and final materials:
- [notes/results_summary.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/notes/results_summary.md)
- [notes/final_abstract.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/notes/final_abstract.md)
- [notes/final_speaker_notes.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/notes/final_speaker_notes.md)

## Dataset

Source:
- Lichess rated Standard PGN archives

Filtering:
- time classes: `Rapid`, `Classical`
- player ratings: `1200` to `2200`
- minimum move count threshold in preprocessing

Stored metadata:
- hashed `white_player_hash`, `black_player_hash`
- ratings
- result and result label
- time control and time class
- UTC date/time and parsed `game_datetime`
- move sequence
- chronological split assignment

Annotated labels:
- move-level centipawn loss
- expected move quality by rating bucket and phase
- residual move quality
- phase-aggregated residual weakness targets

## Methods

### 1. Preprocessing

Raw `.pgn.zst` archives are parsed into parquet with a stable schema. Usernames are removed and replaced with hashes during preprocessing. Games are split chronologically into `train`, `val`, and `test`.

### 2. Move Representation

Each move is encoded with structured features:
- from-square
- to-square
- promotion
- piece type
- capture flag
- check flag
- moving side
- game phase
- rating bucket
- time-class ID
- result ID

Games are then windowed into fixed `128`-ply segments with padding and attention masks.

### 3. Engine Labeling

Stockfish is used to annotate each move with:
- evaluation before the move
- evaluation after the move
- clipped centipawn loss
- expected loss for the player’s rating bucket and game phase
- residual loss relative to that expectation

Long runs are checkpointed to parquet and can be resumed.

### 4. Model

The backbone is a transformer encoder over structured move windows. It produces:
- token embeddings
- pooled white-player embedding
- pooled black-player embedding
- player-conditioned token context

Heads are used for:
- rating prediction
- expected move-quality prediction
- phase-aggregated residual weakness prediction

Training includes:
- multitask supervised losses
- robust residual loss
- contrastive player objective
- gradient clipping

### 5. Residual Formulation

An important empirical finding was that raw move-level residual prediction was too noisy to learn well. The project therefore switched to **phase-aggregated residual weakness** across:
- opening
- middlegame
- endgame

That phase-level target became the most reliable weakness objective in the project.

## Results

### Best Training Run

Dataset:
- `25,000` annotated games
- `16,542` train windows
- `3,527` validation windows

Final validation metrics:
- rating RMSE: `35.13`
- expected CP RMSE: `2.58`
- phase residual RMSE: `45.76`

Scale trend:
- `5k` annotated games: phase residual RMSE `49.10`
- `10k` annotated games: phase residual RMSE `47.07`
- `25k` annotated games: phase residual RMSE `45.76`

### Beyond Rating Alone

Files:
- [experiments/run2/embedding_quality/embedding_quality_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/embedding_quality/embedding_quality_summary.parquet)
- [experiments/run2/embedding_quality/embedding_vs_rating_val.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/embedding_quality/embedding_vs_rating_val.parquet)

Validation result on `25k`:
- players compared: `6528`
- mean embedding-neighbor RMSE: `55.06`
- mean rating-neighbor RMSE: `61.98`
- embedding beats rating rate: `57.38%`

Interpretation:
- the learned embeddings capture phase-weakness structure better than rating-only matching

### Temporal Stability

File:
- [experiments/run2/stability/temporal_stability_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/stability/temporal_stability_summary.parquet)

Results on `25k`:
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
- weakness profiles are more stable over time than random baselines

### Personalization Validity

File:
- [experiments/run2/personalization/train_to_val_min1/personalization_summary.parquet](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/personalization/train_to_val_min1/personalization_summary.parquet)

Short-horizon `train -> val` result on `25k`:
- players compared: `2386`
- personal cosine: `0.0802`
- random cosine: `0.0549`
- personal RMSE: `46.66`
- random RMSE: `47.30`

Interpretation:
- future personalized weakness prediction is positive but still modest

## Discussion

### What Worked

- Rating-conditioned preprocessing and Stockfish labels produced a workable residual-analysis dataset.
- Phase-level aggregation made residual weakness learnable.
- More annotated data consistently improved the phase-residual objective.
- Embeddings now show clear evidence of capturing structure beyond rating alone.
- Temporal stability results are positive across chronological splits on the `25k` run.

### What Did Not Work Well

- Raw move-level residual regression was too noisy.
- Player-aware contrastive batching helped activate the contrastive loss, but did not meaningfully fix residual prediction by itself.
- Personalization validity improved, but is still the weakest part of the system.

### Main Takeaway

The main contribution is not a fundamentally new transformer architecture. The real contribution is the **rating-aware residual weakness framework**:
- expected move quality conditioned on rating
- residual weakness relative to that expectation
- phase-level weakness prediction
- temporal evaluation of persistence

The current evidence supports a careful claim that player-specific behavioral structure exists beyond rating and can be recovered with latent embeddings, though personalized forecasting still needs more scale for stronger claims.

## Visualization Artifacts

- [PCA embeddings](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/pca_embeddings.html)
- [t-SNE embeddings](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/tsne_embeddings.html)
- [Residual heatmap](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/residual_heatmap.png)

## Running the Pipeline

### 1. Build Filtered Dataset

```bash
python3 scripts/build_dataset.py \
  --max-games 25000 \
  --output-file data/processed/games_sample_25000.parquet
```

### 2. Annotate with Stockfish

```bash
python3 scripts/run_stockfish.py \
  --input-file data/processed/games_sample_25000.parquet \
  --output-file data/processed/games_sample_25000_stockfish.parquet \
  --engine-path "$(which stockfish)" \
  --depth 12 \
  --checkpoint-every 25 \
  --resume
```

### 3. Train

```bash
python3 src/train/trainer.py --config configs/base.yaml
```

### 4. Export Embeddings

```bash
python3 src/eval/embed_analysis.py \
  --checkpoint experiments/run2/model.pt \
  --data-path data/processed/games_sample_25000_stockfish.parquet \
  --split val \
  --output-file experiments/run2/player_embeddings_val.parquet
```

### 5. Run Evaluation

```bash
python3 src/eval/temporal_stability.py \
  --input-file data/processed/games_sample_25000_stockfish.parquet \
  --output-dir experiments/run2/stability

python3 src/eval/embedding_quality.py \
  --embedding-file experiments/run2/player_embeddings_val.parquet \
  --data-path data/processed/games_sample_25000_stockfish.parquet \
  --split val \
  --output-dir experiments/run2/embedding_quality
```

## Repo Structure

```text
chess-style-embeddings/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
├── notes/
├── scripts/
├── src/
│   ├── data/
│   ├── eval/
│   ├── features/
│   ├── models/
│   ├── train/
│   └── vis/
└── README.md
```

## Limitations

- `25k` annotated games is much stronger than the original prototype, but still small relative to the full monthly Lichess pool.
- The architecture is not novel by itself; the novelty is mainly in the residual formulation and evaluation framing.
- The strongest evidence is for relative structure and temporal consistency, not for precise long-horizon personalized forecasting.

## Conclusion

This project successfully moved from isolated mistake detection to **rating-aware, phase-level weakness modeling**, with growing evidence that latent embeddings capture player-specific structure beyond rating and that weakness profiles remain meaningfully stable over time.
