# Minimal Runnable Code Package

This folder contains only the code and configuration needed to run the project pipeline and demo scripts.

Large files are intentionally not included:
- raw Lichess PGN archive
- processed parquet dataset
- trained checkpoint
- full experiment artifacts

Those files are required to reproduce the exact final run, but they are too large for a code attachment.

## Included

- `src/`: core data, feature, model, training, and evaluation code
- `scripts/`: dataset build, Stockfish annotation, residual relabeling, and demo prediction scripts
- `configs/base.yaml`: training configuration
- `streamlit_app.py`: optional interactive demo
- `requirements.txt`: Python dependencies
- `README.md`: full project overview
- `experiments/run2/final_demo/*.md`: lightweight generated demo outputs for reference

## Main Commands

From the full project repo root:

```bash
cd /Users/mutsamungoshi/Desktop/chess-style-embeddings
pyenv shell 3.9.6
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Train, assuming the processed dataset exists at the path in `configs/base.yaml`:

```bash
python3 src/train/trainer.py --config configs/base.yaml
```

Show the saved training trace from the checkpoint:

```bash
python3 -c "import torch; ck=torch.load('experiments/run2/model.pt', map_location='cpu'); print(ck['history'][-1])"
```

Recompute train/validation/test metrics from a saved checkpoint:

```bash
python3 src/eval/final_project_report.py
```

Generate five held-out test predictions:

```bash
python3 scripts/demo_final_project.py --num-samples 5
```

Run the interactive demo:

```bash
streamlit run streamlit_app.py
```

Create the minimal code zip for sending:

```bash
zip -r submission_code_minimal.zip submission_code_minimal
```

## Most Important Files

- `src/models/transformer.py`
- `src/models/heads.py`
- `src/models/losses.py`
- `src/train/trainer.py`
- `src/data/dataset.py`
- `src/features/move_tokenizer.py`
- `scripts/run_stockfish.py`
- `src/eval/final_project_report.py`
- `scripts/demo_final_project.py`
