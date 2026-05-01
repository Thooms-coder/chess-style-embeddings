# Manus-Ready Final Slide Deck

Use this document as the primary source for generating the final presentation in Manus AI.

## Deck Instructions for Manus

Create a clean, academic deep learning final-project slide deck. The presentation should feel like a technical research talk, not a marketing pitch.

Visual style:
- Use a restrained blue, white, charcoal, and light gray palette.
- Use chessboard, move-sequence, and model-pipeline visuals where helpful.
- Avoid cartoonish chess art, heavy gradients, or crowded decorative backgrounds.
- Prefer diagrams, compact tables, and clean metric cards.
- Use one main idea per slide.
- Keep slide text concise; put explanation in speaker notes.
- Use clear labels for all metrics.

Important terminology:
- This is a regression project, so use **RMSE**, not accuracy.
- RMSE is measured in **centipawns**, not percent.
- `100 centipawns = 1 pawn`.
- Positive residual weakness means worse than expected for rating and phase.
- Negative residual weakness means better than expected.

Core claim:
Rating-aware residual modeling with transformer-based player embeddings can recover meaningful phase-level chess weakness structure beyond rating alone.

Do not overclaim:
- Do not call this production-ready.
- Do not claim perfect personalized coaching.
- Do not say the architecture is fundamentally new.
- Present it as a research prototype with real empirical evidence.

---

## Slide 1 — Title

**Latent Style Embeddings and Rating-Conditioned Residual Weakness Modeling**

Deep Learning Final Project  
Lichess + Stockfish + Transformer

**Goal:** model chess weaknesses beyond rating alone.

Visual direction:
- Use a clean chessboard or chess-position background.
- Add a small pipeline subtitle: `Games -> Engine Labels -> Residuals -> Embeddings`.

Speaker notes:
Today I am presenting a project on learning rating-aware chess style embeddings and residual weakness models. The goal is not to build the strongest chess engine. The goal is to understand whether player weakness patterns can be modeled beyond rating alone.

---

## Slide 2 — Problem

**Rating tells us strength, but not how a player tends to struggle.**

- Chess tools detect blunders well.
- They usually treat mistakes as isolated events.
- They do not cleanly separate:
  - overall player strength
  - persistent structural weakness
- Two players with the same rating can have different weakness profiles.

Visual direction:
- Show two player cards with the same rating but different phase weaknesses.
- Example: Player A: opening weakness; Player B: endgame weakness.

Speaker notes:
The main limitation is that rating is a single number. It summarizes strength, but it does not explain how a player tends to go wrong. Two players can both be rated 1500 but need very different training.

---

## Slide 3 — Core Idea

**Model weakness relative to what is expected at that rating.**

Instead of asking:

`Was this move bad?`

Ask:

`Was this move worse than expected for a player at this rating and phase?`

Residual weakness:

`actual centipawn loss - expected centipawn loss`

Visual direction:
- Use a simple equation graphic.
- Show rating and game phase feeding into expected move quality.
- Then subtract expected from actual to produce residual weakness.

Speaker notes:
The key idea is rating-conditioned residual modeling. A 60-centipawn mistake means something different for a 1300-rated player than for a 2100-rated player. So we estimate expected move quality by rating and phase, then model the residual.

---

## Slide 4 — Practical Application

**Toward personalized chess training**

Potential uses:
- rating-aware diagnostics
- personalized training plans
- adaptive chess coaching
- player profiling beyond blunder counts

Example:
- two equal-rated players
- different phase weaknesses
- different recommended training focus

Visual direction:
- Show a compact coaching dashboard mockup with phase bars for opening, middlegame, and endgame.

Speaker notes:
A practical application is personalized training. If one player is weaker in endgames and another is weaker in tactical middlegames, a rating-only system cannot distinguish them. A residual weakness profile can.

---

## Slide 5 — Dataset

**Public Lichess rated Standard games**

Filtering:
- Rapid and Classical games only
- player ratings from `1200` to `2200`
- chronological train / validation / test split
- usernames replaced with hashes

Final clean annotated dataset:

| Split | Games |
|---|---:|
| Train | 17,500 |
| Validation | 3,750 |
| Test | 3,750 |
| Total | 25,000 |

Visual direction:
- Use a dataset funnel: raw PGN archive -> filters -> clean 25k set.

Speaker notes:
The dataset comes from public Lichess games. We filtered to Rapid and Classical because those games are more comparable than blitz or bullet. We also hash usernames and split chronologically so validation and test represent future games, not random leakage.

---

## Slide 6 — Pipeline

**End-to-end system**

1. Parse `.pgn.zst` Lichess archive.
2. Filter games and hash player IDs.
3. Annotate moves with Stockfish.
4. Compute expected centipawn loss by rating bucket and phase.
5. Compute residual weakness.
6. Train transformer on fixed `128`-ply windows.
7. Evaluate embeddings and weakness profiles.

Visual direction:
- Make this a horizontal pipeline diagram.
- Use icons for PGN file, filter, Stockfish, residual equation, transformer, metrics.

Speaker notes:
This is a full pipeline, not just a model file. It starts from raw Lichess archives and ends with trained embeddings, predictions, baselines, and evaluation artifacts.

---

## Slide 7 — Model Architecture

**Transformer over structured move windows**

Inputs per move:
- from-square and to-square
- promotion and piece type
- capture and check flags
- side to move
- game phase
- rating bucket
- time class and result

Outputs:
- white player embedding
- black player embedding
- rating prediction
- expected move-quality prediction
- phase residual weakness prediction

Visual direction:
- Show a sequence of moves entering a transformer block.
- Show pooled white and black embeddings branching into output heads.

Speaker notes:
The model uses structured chess features instead of raw text. A transformer encoder processes 128-ply windows, then we pool representations for each side and predict multiple targets.

---

## Slide 8 — Key Iteration

**Raw move-level residuals were too noisy.**

Initial target:
- predict residual weakness for every move

Problem:
- individual move labels were high variance
- single-game windows were noisy

Final target:
- phase-aggregated residual weakness
- opening
- middlegame
- endgame

Leakage fix:
- expected-quality baselines computed from `train` only

Visual direction:
- Show noisy per-move residual line collapsing into three phase-level bars.

Speaker notes:
The most important iteration was changing the target. Raw move-level residual prediction was too noisy, so we aggregated weakness by phase. This made the task more learnable and more aligned with the coaching use case.

---

## Slide 9 — Training Traces and Main Metrics

**Regression metric: phase residual RMSE in centipawns**

Lower is better.

| Split | Windows | Phase Residual RMSE |
|---|---:|---:|
| Train | 16,542 | 46.62 |
| Validation | 3,527 | 47.12 |
| Test | 3,531 | 47.65 |

Additional metrics:
- rating RMSE: about `37`
- expected centipawn-loss RMSE: about `2.25`

Visual direction:
- Use a table plus a small bar chart.
- Add annotation: `47 centipawns ~= 0.47 pawns`.

Speaker notes:
Since this is regression, the metric is RMSE rather than classification accuracy. The train, validation, and test scores are close, which suggests the model is not simply memorizing the training data.

Demo evidence:
- File: `experiments/run2/final_demo/final_demo_metrics.md`
- Checkpoint: `experiments/run2/model.pt`

---

## Slide 10 — Baseline Comparison

**Validation phase residual RMSE**

| Model | RMSE |
|---|---:|
| Rating heuristic | 67.77 |
| Ridge regression | 58.85 |
| Histogram gradient boosting | 49.63 |
| Transformer | 47.12 |

Takeaway:
- transformer beats all tested baselines
- gain over strongest tree baseline is real but modest

Visual direction:
- Use a descending bar chart.
- Highlight the transformer bar.

Speaker notes:
This baseline comparison is important because it shows the transformer is doing more than rating-only matching or simple tabular regression. The improvement over the strongest baseline is not huge, but it is meaningful on a noisy residual target.

---

## Slide 11 — Beyond Rating Alone

**Do learned embeddings capture player structure beyond rating?**

Validation embedding-neighbor evaluation:

| Metric | Value |
|---|---:|
| Players compared | 6,528 |
| Embedding-neighbor RMSE | 54.80 |
| Rating-neighbor RMSE | 61.99 |
| Embedding beats rating rate | 57.7% |

Takeaway:
- learned embeddings match weakness profiles better than rating-only neighbors

Visual direction:
- Show two nearest-neighbor paths:
  - rating-only neighbor
  - embedding neighbor
- Show embedding neighbor with lower RMSE.

Speaker notes:
This is the strongest evidence for the central claim. If rating explained everything, rating-nearest players should match weakness profiles as well as embedding-nearest players. But embeddings do better.

---

## Slide 12 — Temporal Stability

**Are weakness profiles persistent over time?**

Chronological profile comparison:

| Comparison | Temporal RMSE | Random RMSE | Temporal Cosine | Random Cosine |
|---|---:|---:|---:|---:|
| Train -> Val | 32.08 | 33.57 | 0.2551 | 0.2043 |
| Val -> Test | 35.65 | 36.82 | 0.2085 | 0.1576 |
| Train -> Test | 32.57 | 34.50 | 0.2191 | 0.1890 |

Takeaway:
- same-player profiles are more stable than random baselines

Visual direction:
- Use a line or paired-bar chart comparing temporal vs random.

Speaker notes:
This evaluates whether the weakness profile is persistent, not just fit noise. Across chronological splits, same-player profiles are more similar than random player profiles.

---

## Slide 13 — Personalization

**Personalized future prediction is positive but modest.**

Train -> Validation personalization:

| Metric | Personal | Random |
|---|---:|---:|
| Cosine similarity | 0.1109 | 0.0702 |
| RMSE | 45.36 | 46.16 |

Players compared: `2,386`

Takeaway:
- same-player future weakness is better than random
- effect size is still small

Visual direction:
- Use two compact metric cards: Personal vs Random.
- Add small caution label: `promising, not solved`.

Speaker notes:
Personalization is positive, but it is the weakest part of the system. The right claim is that the signal exists, not that this is ready for high-confidence individualized coaching.

---

## Slide 14 — Demo: Held-Out Test Predictions

**Model outputs phase-level residual weakness profiles.**

Show 2 or 3 examples from the test split.

For each example:
- game ID and window
- true vs predicted ratings
- predicted phase residual weakness
- observed phase residual weakness
- short move preview

Interpretation:
- positive = worse than expected
- negative = better than expected
- individual windows are noisy
- aggregate RMSE is the main evidence

Visual direction:
- Use one example card with opening / middlegame / endgame bars.
- Show predicted and observed bars side by side.

Speaker notes:
These examples demonstrate that the trained checkpoint produces structured outputs on held-out test games. I do not want to overclaim individual examples, because one window can be noisy. The important evidence is the full split metric and the baseline comparison.

Demo evidence:
- File: `experiments/run2/final_demo/demo_predictions_test.md`
- Command: `python3 scripts/demo_final_project.py --num-samples 5`

---

## Slide 15 — Limitations

**What this project does not prove yet**

- `25k` annotated games is useful, but still small relative to Lichess scale.
- Stockfish residual labels are noisy at the individual-window level.
- The transformer architecture itself is not novel.
- Personalization is positive but still modest.
- This is not production-ready coaching.

Visual direction:
- Use a simple limitations checklist.

Speaker notes:
The honest interpretation is that the system works as a research prototype. It shows measurable behavioral structure, but precise personalized forecasting needs more scale and better player-history modeling.

---

## Slide 16 — Conclusion

**Main result**

Rating-aware residual modeling can recover phase-level player weakness structure beyond rating alone.

What worked:
- phase aggregation made residual weakness learnable
- transformer beat tested baselines
- embeddings captured structure beyond rating
- weakness profiles showed temporal stability

Final claim:

**This project delivers a leak-free transformer-based system for rating-aware chess weakness modeling, with evidence that player embeddings capture meaningful structure beyond rating alone.**

Visual direction:
- End with a clean summary diagram: Rating-aware residuals + Transformer embeddings -> stable weakness profiles.

Speaker notes:
The strongest contribution is the framework: rating-conditioned expected quality, residual weakness, player embeddings, and chronological evaluation. The project moves from isolated mistake detection toward structured, rating-aware modeling of player tendencies.

---

# Backup Slide A — Live Demo Commands

Use if the instructor asks how the demo was produced.

```bash
python3 src/eval/final_project_report.py \
  --checkpoint experiments/run2/model.pt \
  --data-path data/processed/games_sample_25000_stockfish_trainonly.parquet \
  --output-dir experiments/run2/final_demo
```

```bash
python3 scripts/demo_final_project.py \
  --checkpoint experiments/run2/model.pt \
  --data-path data/processed/games_sample_25000_stockfish_trainonly.parquet \
  --split test \
  --num-samples 5 \
  --output-dir experiments/run2/final_demo
```

Speaker notes:
Full training takes too long for a live Zoom demo, so the demo loads the saved checkpoint, recomputes metrics, and generates predictions from held-out test samples.

---

# Backup Slide B — How to Explain RMSE

RMSE is not a score out of 100.

- It is measured in centipawns.
- `100 centipawns = 1 pawn`.
- Test RMSE around `47.65` means roughly `0.48 pawns` of residual prediction error.
- Lower is better.
- This is useful for phase-level profiling, not exact move-level coaching.

Speaker notes:
If asked whether 47 is good, explain that it is good relative to baselines and stable across train, validation, and test. But it is not precise enough to claim production-level coaching.

