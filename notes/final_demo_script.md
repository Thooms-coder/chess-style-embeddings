# 5-Minute Demo Script

## Goal

Show, within 5 minutes, that:
- the code runs
- the model was really trained
- the reported metrics are real
- the model can produce predictions on at least 5 test samples

Use this exact order.

## Demo Setup

Have these ready before the presentation:
- terminal in repo root
- [experiments/run2/final_demo/final_demo_metrics.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/final_demo_metrics.md)
- [experiments/run2/final_demo/demo_predictions_test.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/demo_predictions_test.md)
- [experiments/run2/plots/pca_embeddings.html](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/pca_embeddings.html)
- [experiments/run2/plots/tsne_embeddings.html](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/tsne_embeddings.html)

## Demo Timing

### 0:00 to 0:45 — Show the pipeline is real

Say:

“This project is a full end-to-end deep learning pipeline. We parse Lichess PGNs, annotate moves with Stockfish, build 128-ply windows, train a transformer, and evaluate it on chronological splits.”

Then show these commands:

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

If you do not want to rerun live, say:

“Because the full model evaluation is already saved, I’ll show the generated traces and outputs from the exact checkpoint used in our final results.”

### 0:45 to 1:45 — Show training / validation / test metrics

Open:
- [final_demo_metrics.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/final_demo_metrics.md)

Say:

“Our main target is phase residual weakness prediction. On the clean leak-free 25k setup, the phase residual RMSE is 46.62 on train, 47.12 on validation, and 47.65 on test. That shows the model generalizes reasonably well across splits.”

Then say:

“On validation, the strongest non-neural baseline was histogram gradient boosting at 49.63, while the transformer achieved 47.12.”

### 1:45 to 3:45 — Show 5 test predictions

Open:
- [demo_predictions_test.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/demo_predictions_test.md)

For each sample, say one short sentence:

1. “This sample shows the model predicting white and black phase weaknesses from the move window.”
2. “Here the predicted weakness direction roughly matches the observed middlegame and endgame trend.”
3. “This one shows a case where the model captures the sign and phase pattern better for White than for Black.”
4. “This sample is useful because it shows the model can also predict a strong endgame weakness signal.”
5. “This last example shows that the model is not perfect, but it still produces structured phase-level predictions rather than noise.”

Do not overclaim per-sample accuracy. The point is to show:
- real test windows
- real predictions
- real observed targets

### 3:45 to 4:30 — Show representation result

Open one of:
- [pca_embeddings.html](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/pca_embeddings.html)
- [tsne_embeddings.html](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/plots/tsne_embeddings.html)

Say:

“Beyond direct prediction, the learned embeddings also capture behavioral structure beyond rating alone. In our evaluation, embedding-based neighbors matched weakness profiles better than rating-only neighbors.”

### 4:30 to 5:00 — Close the demo

Say:

“So the demo shows three things: the code is real, the final metrics are reproducible, and the model can make structured predictions on held-out test windows.”

## Short Backup Version

If time gets tight, only show:
1. [final_demo_metrics.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/final_demo_metrics.md)
2. first 2 samples from [demo_predictions_test.md](/Users/mutsamungoshi/Desktop/chess-style-embeddings/experiments/run2/final_demo/demo_predictions_test.md)
3. one embedding plot

## One-Sentence Demo Summary

“This demo verifies that our transformer was trained on a leak-free 25k-game dataset, outperforms strong baselines on phase residual weakness prediction, and produces structured predictions on held-out test samples.”
