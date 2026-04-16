# Final Speaker Notes

## Slide 1 — Title

“Today I’m presenting a project on learning **rating-aware chess style embeddings** and **residual weakness models**.

The main goal is to move beyond isolated blunder detection and instead learn deeper patterns in how players make decisions, especially weaknesses that persist even after accounting for rating.”

## Slide 2 — Problem

“Current chess tools are very good at flagging blunders, but they usually treat mistakes as isolated events.

That misses an important distinction: two players with the same rating can still have very different structural weaknesses. One player may consistently struggle in endgames, while another may struggle in tactical middlegames.

So the key limitation is that standard analysis often mixes together overall strength and player-specific decision tendencies.”

## Slide 3 — Core Question

“The core question is:

Can we disentangle player strength from structural decision-making tendencies by modeling errors relative to what is expected at that rating?

Instead of asking, ‘Was this move bad?’ we ask, ‘Was this move worse than what we would expect from a player at this rating?’

That residual lets us isolate structural weakness rather than just raw error frequency.”

## Slide 4 — Data

“We use publicly available rated Standard games from the Lichess open database.

We filter to Rapid and Classical games, and to players between 1200 and 2200 rating. That reduces noise and keeps decision contexts more comparable.

We parse PGN archives, hash all usernames during preprocessing, and split the data chronologically for train, validation, and test evaluation.”

## Slide 5 — Representation and Labels

“Each move is converted into structured model inputs, including from-square, to-square, promotion, piece type, capture and check flags, game phase, and rating bucket.

We then use Stockfish to compute centipawn loss for each move.

From that we derive expected move quality by rating and phase, and then compute residual weakness as actual minus expected.

In practice, raw move-level residuals turned out to be too noisy, so the most successful version of the model predicts **phase-aggregated weakness profiles** across opening, middlegame, and endgame.”

## Slide 6 — Model

“The pipeline is:

PGN to structured move encoding, then a transformer, then pooled player embeddings, then multitask heads.

The transformer models 128-ply move windows using self-attention, and the output is used for three main objectives:
- rating prediction
- expected move quality prediction
- phase-level residual weakness prediction

We also use a contrastive objective to encourage more consistent player embeddings across games.”

## Slide 7 — Key Innovation

“The key innovation is not a brand-new transformer architecture.

The contribution is the **framework**:
- conditioning performance on rating
- modeling residual weakness relative to rating-level expectation
- aggregating that weakness by phase
- and testing whether those patterns remain stable over time

So the novelty is mainly in the formulation of rating-adjusted diagnosis and the way we evaluate it.”

## Slide 8 — Main Results

“The current best run uses 25,000 Stockfish-annotated games.

On that run:
- validation phase-residual RMSE improved to 45.76
- embedding-based neighbors match player weakness profiles better than rating-only neighbors
- and weakness profiles are more stable over time than random baselines

This means the model is capturing player-specific structure beyond rating alone.”

## Slide 9 — Beyond Rating

“The strongest result is the ‘beyond rating alone’ evaluation.

On the validation split, embedding-based nearest neighbors produced lower weakness-profile error than rating-only nearest neighbors, and embeddings beat rating in about 57% of pairwise comparisons.

That is the clearest evidence that the learned representation contains behavioral structure not explained by rating alone.”

## Slide 10 — Temporal Stability

“We also tested whether weakness profiles are stable over time using chronological splits.

For train-to-validation and validation-to-test comparisons, the same player’s weakness profile was more similar over time than a random player baseline, both in cosine similarity and RMSE.

On the 25k run, even train-to-test stability became directionally positive.

So the system is not just fitting noise; it is recovering some persistent player structure.”

## Slide 11 — Personalization Validity

“The weakest but still promising result is personalization.

When we compare earlier predicted weakness profiles to later observed weaknesses for the same player, the effect is positive but modest.

On the 25k run, short-horizon train-to-validation personalization outperformed random on both cosine similarity and RMSE, but the margin is still small.

So personalized forecasting is the least mature part of the system and likely needs more annotated data.”

## Slide 12 — Limitations

“There are several limitations.

First, 25,000 annotated games is much larger than the initial prototype, but still small relative to the full monthly Lichess pool.

Second, the architecture itself is not fundamentally novel; the contribution is mainly the rating-conditioned residual framework.

Third, the strongest evidence is for relative structure and stability, not for precise long-horizon personalized prediction.”

## Slide 13 — Contribution

“So the most accurate claim is:

This project introduces a **rating-aware residual weakness framework** for chess analysis, and provides empirical evidence that transformer-based player embeddings can capture meaningful behavioral structure beyond rating alone.

It is best understood as a strong prototype with promising results, rather than a fully proven research system.”

## Slide 14 — Closing

“The overall takeaway is that we can move from isolated mistake detection toward structured, rating-aware modeling of player tendencies.

That opens the door to more personalized and interpretable chess training systems.”
