# Music Genre Classification — Messy Mashup Challenge

**Roll No**: 22f1001611  
**Course**: BSDA2001P — Introduction to DL and GenAI  
**Project**: T1 2026 Kaggle Competition

## Overview

Classify music genres from noisy mashup audio files. Training data contains clean stem files (vocals, drums, bass, other) organized by genre; test data consists of mashups combining stems from different songs with added environmental noise.

## Project Structure

```
project-name/
├── notebooks/
│   ├── milestone-1.ipynb      # EDA + Random Baseline
│   ├── milestone-2.ipynb      # XGBoost (MFCC + Chroma + Spectral)
│   └── final_notebook.ipynb   # CNN + CRNN + HuBERT + Ensemble
├── src/
│   ├── train.py               # Model definitions + training loops
│   ├── inference.py           # Test inference + TTA + ensemble
│   └── utils.py               # Constants, configs, audio utilities
├── reports/
│   ├── milestone-1-report.pdf
│   ├── milestone-2-report.pdf
│   └── final-report.pdf
├── models/                    # Saved model weights (.pth) — gitignored
├── requirements.txt
└── README.md
```

## Branching Strategy

| Branch | Contents |
|--------|----------|
| `main` | Latest stable code, all milestones merged |
| `milestone-1` | EDA + Random Baseline |
| `milestone-2` | XGBoost classical ML baseline |
| `milestone-3` | CNN + CRNN + HuBERT + Ensemble (Final) |

## Approach

### Milestone 1 — EDA + Random Baseline
- Class distribution analysis, audio stats, waveform/spectrogram visualization
- Random genre assignment (~10% accuracy baseline)

### Milestone 2 — XGBoost
- Hand-crafted features: MFCC (20), Chroma (12), Spectral Contrast, Centroid, Bandwidth, Rolloff, ZCR, Tempo
- Noise augmentation with ESC-50 environmental sounds
- XGBoost classifier with WandB logging

### Milestone 3 — Deep Learning + Ensemble
- **CNN** (from scratch): 3-layer CNN with BatchNorm + AdaptiveAvgPool on mel spectrograms
- **CRNN**: CNN feature extractor → Bidirectional LSTM for temporal patterns
- **HuBERT**: Fine-tuned `superb/hubert-base-superb-ks` with phased training (frozen → unfrozen)
- **Key Innovation**: Cross-song stem mixing — each stem from a DIFFERENT song of same genre (3x dataset multiplication)
- **Ensemble**: Weighted soft voting (HuBERT 0.65 + CRNN 0.20 + CNN 0.15) with systematic TTA (5 overlapping 10s windows)
- **Target**: 0.80+ Macro F1

## How to Run

```bash
pip install -r requirements.txt
```

Run notebooks in order on Kaggle (GPU required for Milestone 3):
1. `notebooks/milestone-1.ipynb`
2. `notebooks/milestone-2.ipynb`
3. `notebooks/final_notebook.ipynb`

## WandB

All training runs logged to project `22f1001611-t12026`.
