# Final Slide Deck Text

This is deck-ready text matched to the course rubric. Each slide has:
- slide title
- slide bullets
- short speaker guidance

## Slide 1 — Title

**Latent Style Embeddings and Rating-Conditioned Residual Weakness Modeling**

- Deep Learning Final Project
- Lichess + Stockfish + Transformer
- Goal: model chess weaknesses beyond rating alone

Speaker note:
- “Our project studies whether chess player weaknesses can be modeled in a way that is more informative than rating alone.”

## Slide 2 — Problem Overview

- Standard chess tools detect blunders well
- They usually treat mistakes as isolated events
- They do not clearly separate:
  - overall strength
  - persistent structural weakness
- Two players with the same rating can still have very different weakness patterns

Speaker note:
- “The main problem is that rating summarizes strength, but it does not explain *how* players tend to go wrong.”

## Slide 3 — Practical Application

- Personalized chess training
- Adaptive coaching systems
- Rating-aware diagnostic tools
- Player profiling beyond simple blunder counts

Speaker note:
- “A practical application is personalized training. Two 1500-rated players may need completely different training recommendations.”

## Slide 4 — Dataset

- Source: publicly available rated Lichess Standard games
- Filters:
  - Rapid and Classical only
  - player ratings `1200–2200`
- Final clean annotated dataset:
  - `25,000` games
  - chronological `train / val / test`
- Privacy:
  - usernames replaced with hashes during preprocessing

Speaker note:
- “We used a cleaned, chronological, privacy-preserving dataset so the evaluation reflects future generalization rather than random mixing.”

## Slide 5 — Data Pipeline

- PGN archive parsing
- Structured parquet dataset
- Stockfish annotation per move
- Expected move quality by rating and phase
- Residual weakness = actual minus expected
- Windowing into fixed `128`-ply sequences

Speaker note:
- “The core label construction step is rating-conditioned residual modeling: we estimate what move quality should look like at a given rating, then measure the deviation from that expectation.”

## Slide 6 — Method / Architecture

- Structured move encoding:
  - from-square, to-square
  - promotion, piece type
  - capture and check flags
  - side to move, phase, rating bucket
- Transformer encoder over `128`-ply windows
- Pooled player embeddings
- Multi-task heads for:
  - rating prediction
  - expected move quality
  - phase residual weakness

Speaker note:
- “The transformer learns player-conditioned sequence representations from structured chess decisions, rather than raw text or only board states.”

## Slide 7 — Important Iteration / Extra Component

- Initial target: raw move-level residual prediction
- Problem: too noisy to learn reliably
- Final target: **phase-aggregated residual weakness**
  - opening
  - middlegame
  - endgame
- Final evaluation also fixed a train/test leakage issue by computing residual baselines from `train` only

Speaker note:
- “Our extra contribution beyond a straightforward implementation was improving the target formulation and cleaning the evaluation setup.”

## Slide 8 — Training, Validation, Test Results

- Main target: phase residual RMSE

| Split | Windows | Phase Residual RMSE |
|---|---:|---:|
| Train | 16,542 | 46.62 |
| Validation | 3,527 | 47.12 |
| Test | 3,531 | 47.65 |

- Additional clean metrics:
  - rating RMSE: about `37`
  - expected CP RMSE: about `2.25`

Speaker note:
- “The train, validation, and test phase-residual errors are close, which suggests the model generalizes reasonably well.”

## Slide 9 — Baseline Comparison

Validation phase residual RMSE:

- Rating heuristic: `67.77`
- Ridge regression: `58.85`
- HistGradientBoosting: `49.63`
- Transformer: `47.12`

Takeaway:
- the transformer beats all tested non-neural baselines
- the gain over the strongest tree baseline is real but modest

Speaker note:
- “This is important because it shows the transformer is doing more than simple rating heuristics or tabular regression.”

## Slide 10 — Beyond Rating Alone

- Embedding-based neighbor evaluation on validation
- Players compared: `6528`
- Embedding-neighbor RMSE: `54.80`
- Rating-neighbor RMSE: `61.99`
- Embedding beats rating rate: `57.7%`

Takeaway:
- learned embeddings capture player structure beyond rating alone

Speaker note:
- “This is the strongest evidence for the project’s central claim: the learned representation contains behavior not explained by rating alone.”

## Slide 11 — Temporal Stability and Personalization

Temporal stability:
- `train -> val`
  - temporal cosine `0.2551`
  - random cosine `0.2043`
  - temporal RMSE `32.08`
  - random RMSE `33.57`

Personalization:
- `train -> val`
  - personal cosine `0.1109`
  - random cosine `0.0702`
  - personal RMSE `45.36`
  - random RMSE `46.16`

Takeaway:
- weakness profiles are temporally stable
- personalization is positive but still modest

Speaker note:
- “The model captures persistence over time, but precise personalized forecasting is still the weakest part of the system.”

## Slide 12 — Demo / Example Predictions

Include 2 to 3 examples from:
- [demo_predictions_test.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/demo_predictions_test.md)

Suggested layout:
- game/window metadata
- predicted white and black phase weakness
- observed white and black phase weakness
- short move preview

Speaker note:
- “These are held-out test windows showing predicted and observed weakness profiles. The goal is to demonstrate structured outputs on real unseen samples.”

## Slide 13 — Conclusion

- Rating-aware residual modeling is workable
- Phase aggregation made residual weakness learnable
- Transformer beats strong baselines on the main task
- Embeddings capture player structure beyond rating alone
- Temporal stability is positive
- Personalization is promising but still limited

Speaker note:
- “The project’s strongest contribution is a rating-aware residual weakness framework, not a fundamentally new transformer architecture.”

## Slide 14 — Future Work

- Scale beyond `25k` annotated games
- Learn a stronger expected-quality model
- Add hierarchical modeling across games
- Improve personalization with richer player-history modeling
- Build an interpretable coaching interface

Speaker note:
- “The next major step is to model players across sequences of games instead of isolated windows.”

## Final One-Sentence Claim

Use this exact closing sentence:

“Our project delivers a leak-free transformer-based system for rating-aware chess weakness modeling, with evidence that player embeddings capture meaningful structure beyond rating alone and that phase-level weakness profiles remain stable over time.”
