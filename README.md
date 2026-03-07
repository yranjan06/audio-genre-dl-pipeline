# Messy Mashup: End-to-End Project Pipeline

This repository now contains a complete training and inference pipeline for the Kaggle
competition and viva requirements.

## What is covered

- EDA and baseline notebooks (`notebooks/`)
- Classical model: MFCC/mel/chroma features + XGBoost fallback
- Scratch deep models: CNN and CRNN
- Pretrained model: HuBERT fine-tuning
- Macro-F1 based validation tracking
- Inference-only submission generation
- CSV majority-vote ensemble utility

## Repository layout

- `notebooks/milestones/`: milestone-wise experimentation notebooks
- `scripts/dataset.py`: data loading, stem mixing, augmentation, test datasets
- `scripts/model.py`: CNN, CRNN, HuBERT model builders
- `scripts/train.py`: train entrypoint (`xgb`, `cnn`, `crnn`, `hubert`)
- `scripts/predict.py`: inference entrypoint and submission writer
- `scripts/runners/`: milestone-specific convenience runners
- `scripts/ensemble.py`: majority-vote ensembling over submission files
- `scripts/run_pipeline.py`: one-command orchestrator (train + predict + ensemble)

```text
dl-genai-project-26-t1/
├── configs/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/        # ignored in git
│   └── raw/              # ignored in git
├── notebooks/
│   ├── milestones/
│   └── archive/
├── scripts/
│   └── runners/
├── artifacts/            # ignored in git (checkpoints/metrics/logs)
├── submissions/
├── reports/
│   └── figures/
└── tests/
```

## Milestone notebooks

- `notebooks/milestones/m1_eda_random_baseline.ipynb`
- `notebooks/milestones/m2_classical_ml.ipynb`
- `notebooks/milestones/m3_m5_deep_models.ipynb`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

W&B key should be passed via environment variable (never hardcode):

```bash
export WANDB_API_KEY="<your_wandb_api_key>"
```

## Dataset expectation

`--base-dir` must point to Kaggle competition folder:

```text
/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup
```

Expected inside base dir:

- `genres_stems/`
- `mashups/`
- `test.csv`
- `ESC-50-master/audio` or `ESC-50/audio`

## One-command full run

```bash
python -m scripts.run_pipeline \
  --base-dir /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --output-root artifacts/full_run
```

Use `--skip-hubert` (or other `--skip-*`) when runtime is tight.

## Training commands

### 1) Classical model (Milestone 2)

```bash
python -m scripts.runners.train_m2_xgb \
  --base-dir /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --output-dir artifacts/m2_xgb
```

### 2) CNN from scratch (Milestone 3)

```bash
python -m scripts.runners.train_m3_cnn \
  --base-dir /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --epochs 12 \
  --batch-size 32 \
  --output-dir artifacts/m3_cnn
```

### 3) CRNN (Milestone 4)

```bash
python -m scripts.runners.train_m4_crnn \
  --base-dir /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --epochs 12 \
  --batch-size 24 \
  --output-dir artifacts/m4_crnn
```

### 4) HuBERT fine-tuning (Milestone 5)

```bash
python -m scripts.runners.train_m5_hubert \
  --base-dir /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --epochs 5 \
  --batch-size 8 \
  --lr 2e-5 \
  --output-dir artifacts/m5_hubert
```

## Inference-only submission (recommended for Kaggle submit notebook)

### XGB

```bash
python -m scripts.predict \
  --base-dir /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --model-type xgb \
  --checkpoint artifacts/m2_xgb/best_xgb.pkl \
  --output-csv submissions/submission_xgb.csv
```

### CNN/CRNN/HuBERT

```bash
python -m scripts.runners.predict_submission \
  --base-dir /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --model-type hubert \
  --checkpoint artifacts/m5_hubert/best_hubert.pth \
  --output-csv submissions/submission_hubert.csv
```

Replace `hubert` with `cnn` or `crnn` and matching checkpoint path as needed.

## Ensemble

```bash
python -m scripts.ensemble \
  --inputs submissions/submission_xgb.csv submissions/submission_crnn.csv submissions/submission_hubert.csv \
  --output submissions/submission_ensemble.csv
```

## Notes for Kaggle runtime

- Keep training and submission in separate notebooks.
- Use `num_workers=2` and moderate batch sizes for stability.
- Prefer fewer epochs + better augmentation over very long runs.
- Track `val_macro_f1` for model selection.
