# Final 12-Slide Deck

This version is compressed for a clean final presentation. Each section is the exact content to place on a slide.

## Slide 1 — Title

**Latent Style Embeddings and Rating-Conditioned Residual Weakness Modeling**

- Deep Learning Final Project
- Lichess + Stockfish + Transformer
- Goal: model chess weaknesses beyond rating alone

## Slide 2 — Problem

- Chess tools detect blunders well
- They treat many mistakes as isolated events
- They do not clearly separate:
  - overall skill
  - structural player weakness
- Two players with the same rating can have different weakness profiles

## Slide 3 — Practical Use

- Personalized chess training
- Adaptive coaching
- Rating-aware diagnostics
- Player profiling beyond blunder counts

## Slide 4 — Dataset

- Source: rated Lichess Standard games
- Filters:
  - Rapid and Classical
  - player ratings `1200–2200`
- Final clean annotated set:
  - `25,000` games
  - chronological `train / val / test`
- Usernames hashed during preprocessing

## Slide 5 — Data Pipeline

- PGN parsing
- Structured parquet dataset
- Stockfish centipawn-loss annotation
- Expected move quality by rating and phase
- Residual weakness = actual minus expected
- Fixed `128`-ply windows

## Slide 6 — Model

- Structured move encoding
- Transformer encoder over move windows
- Pooled player embeddings
- Multi-task heads for:
  - rating prediction
  - expected move quality
  - phase residual weakness

## Slide 7 — Extra Component / Iteration

- Raw move-level residual target was too noisy
- Switched to **phase-aggregated residual weakness**
  - opening
  - middlegame
  - endgame
- Fixed residual-label leakage:
  - expectations computed from `train` only
- Added strong non-neural baselines

## Slide 8 — Main Results

**Clean `25k` split results**

| Split | Windows | Phase Residual RMSE |
|---|---:|---:|
| Train | 16,542 | 46.62 |
| Validation | 3,527 | 47.12 |
| Test | 3,531 | 47.65 |

- rating RMSE: about `37`
- expected CP RMSE: about `2.25`

## Slide 9 — Baselines

**Validation phase residual RMSE**

- Rating heuristic: `67.77`
- Ridge regression: `58.85`
- HistGradientBoosting: `49.63`
- Transformer: `47.12`

**Takeaway:** transformer beats all tested baselines on the main target

## Slide 10 — Beyond Rating Alone

- Players compared: `6528`
- Embedding-neighbor RMSE: `54.80`
- Rating-neighbor RMSE: `61.99`
- Embedding beats rating rate: `57.7%`

**Takeaway:** learned embeddings capture player structure beyond rating alone

## Slide 11 — Stability and Personalization

**Temporal stability**
- `train -> val`: temporal cosine `0.2551` vs random `0.2043`
- `train -> val`: temporal RMSE `32.08` vs random `33.57`

**Personalization**
- personal cosine `0.1109` vs random `0.0702`
- personal RMSE `45.36` vs random `46.16`

**Takeaway:** weakness profiles are stable; personalization is positive but modest

## Slide 12 — Conclusion and Future Work

### Conclusion
- Rating-aware residual modeling is workable
- Phase aggregation made weakness prediction learnable
- Transformer beats strong baselines
- Embeddings capture structure beyond rating

### Future Work
- Scale beyond `25k`
- Model players across games, not only windows
- Improve expected-quality modeling
- Strengthen personalization

**Final claim:** a leak-free transformer-based system can recover meaningful chess weakness structure beyond rating alone
