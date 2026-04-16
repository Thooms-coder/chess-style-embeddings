# 8-Minute Slide Outline

This outline is matched directly to the course presentation rubric.

## Slide 1 — Title and Problem

Title:
- Latent Style Embeddings and Rating-Conditioned Residual Weakness Modeling

What to say:
- Standard chess tools detect blunders well.
- They do not separate rating from player-specific structural weakness.
- Our goal is to model weaknesses that persist beyond raw skill level.

## Slide 2 — Practical Application

Include:
- adaptive coaching
- personalized chess training
- player profiling beyond rating

What to say:
- Two players with the same rating can need different training.
- This system aims to identify those structural differences.

## Slide 3 — Dataset Description

Include:
- source: public Lichess rated Standard games
- filters: `Rapid`, `Classical`
- ratings: `1200–2200`
- final clean annotated set: `25,000` games
- chronological `train / val / test`
- usernames hashed

What to say:
- We use publicly available game data and preserve privacy through hashing.
- We split chronologically because the project cares about persistence over time.

## Slide 4 — Data Pipeline

Visual pipeline:
- PGN archive
- parser
- parquet dataset
- Stockfish annotation
- expected move quality
- residual labels
- windowed sequences

What to say:
- We parse PGNs, annotate moves with Stockfish, estimate expected move quality conditioned on rating, and compute residual weakness.

## Slide 5 — Method and Architecture

Include:
- structured move encoding
- `128`-ply windows
- transformer encoder
- pooled player embeddings
- multitask heads

Tasks:
- rating prediction
- expected move quality
- phase residual weakness

What to say:
- The model learns player representations and predicts phase-level weakness rather than only raw move quality.

## Slide 6 — Extra Component / Iteration

This is important for the course.

Include:
- initial raw move-level residual target was too noisy
- switched to phase-aggregated residual weakness
- fixed residual-label leakage by computing expectations from train only
- added strong non-neural baselines

What to say:
- The extra engineering/research contribution was not just training one model.
- We iterated on the target, cleaned the evaluation, and added baseline comparisons.

## Slide 7 — Results: Training, Validation, Test

Use the table from:
- [final_demo_metrics.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/final_demo_metrics.md)

Highlight:
- train phase residual RMSE: `46.62`
- val phase residual RMSE: `47.12`
- test phase residual RMSE: `47.65`

What to say:
- The train/val/test numbers are close, so the model generalizes reasonably well.

## Slide 8 — Baseline Comparison

Include:
- rating heuristic: `67.77`
- ridge: `58.85`
- hist gradient boosting: `49.63`
- transformer: `47.12`

What to say:
- This is the key evidence that the transformer is doing more than simple tabular regression.
- The margin is not huge, which makes the result credible.

## Slide 9 — Beyond Rating Alone

Include:
- embedding-neighbor RMSE: `54.80`
- rating-neighbor RMSE: `61.99`
- embedding beats rating rate: `57.7%`

What to say:
- Embeddings capture player structure better than rating-only similarity.
- This is the strongest evidence for the central claim.

## Slide 10 — Temporal Stability and Personalization

Include:
- temporal stability beats random across splits
- short-horizon personalization is positive but modest

Suggested text:
- `train -> val` temporal cosine `0.2551` vs random `0.2043`
- personalization RMSE `45.36` vs random `46.16`

What to say:
- Weakness profiles show measurable persistence over time.
- Personalization is promising, but weaker than the structure and stability results.

## Slide 11 — Demo / Test Predictions

Show:
- 2 or 3 examples from [demo_predictions_test.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/demo_predictions_test.md)

What to say:
- These are held-out test windows with model-predicted and observed phase weakness profiles.
- The purpose is to show that the model produces structured outputs on real unseen samples.

## Slide 12 — Conclusion and Future Work

Conclusion:
- rating-aware residual modeling is workable
- phase aggregation made weakness prediction learnable
- embeddings capture behavior beyond rating
- transformer beats strong baselines on the main target

Future work:
- scale beyond `25k`
- hierarchical across-game player modeling
- better personalization
- improved expected-quality model

## Recommended Speaking Time

- Slides 1 to 3: 2 minutes
- Slides 4 to 6: 2 minutes
- Slides 7 to 10: 3 minutes
- Slides 11 to 12: 1 minute

## Final Claim To Use

“Our project delivers a leak-free transformer-based system for rating-aware chess weakness modeling, with evidence that player embeddings capture meaningful structure beyond rating alone and that phase-level weakness profiles remain stable over time.”
