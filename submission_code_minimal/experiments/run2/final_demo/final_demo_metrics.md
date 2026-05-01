# Final Demo Metrics

- checkpoint: `experiments/run2/model.pt`
- data: `data/processed/games_sample_25000_stockfish_trainonly.parquet`

## Split Results

| split | windows | loss_total | rating_rmse | expected_cp_rmse | residual_cp_rmse | residual_cp_rmse_clipped_300 | phase_residual_rmse |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 16542 | 0.146 | 37.13 | 2.27 | 167.84 | 127.89 | 46.62 |
| val | 3527 | 0.149 | 37.53 | 2.28 | 167.09 | 127.91 | 47.12 |
| test | 3531 | 0.148 | 37.13 | 2.24 | 169.04 | 128.47 | 47.65 |

## Notes

- These are leak-free metrics computed on the train-only relabeled 25k dataset.
- The main project target is `phase_residual_rmse`.
- The strongest baseline on validation is `hist_gbdt` with phase residual RMSE `49.63`.
- The transformer validation phase residual RMSE should be compared against that baseline.