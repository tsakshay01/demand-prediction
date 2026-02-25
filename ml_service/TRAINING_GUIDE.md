# Model Training Guide - DemandAI

## What Needs Training?

| Component | Pre-trained? | Needs Training? | Why |
|-----------|:---:|:---:|-----|
| **MobileNetV2** | ✅ ImageNet | ❌ No | Frozen feature extractor (1280-dim) |
| **SentenceTransformer** | ✅ all-MiniLM-L6-v2 | ❌ No | Frozen text embedder (384-dim) |
| **Fusion Network** | ❌ | ✅ **YES** | LSTM + Gated Fusion + Dense → 7-day forecast |

**Only the Fusion Network needs training.** It learns to combine image, text, and time-series features into demand predictions.

## Dataset: H&M from Kaggle

**Download:** [kaggle.com/competitions/h-and-m-personalized-fashion-recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

Files needed:
- `articles.csv` (product descriptions)
- `transactions_train.csv` (purchase dates/quantities)
- `images/` folder (product photos) — optional but recommended

## Steps

1. Download H&M dataset from Kaggle
2. Place files in a folder (e.g., `hm_data/`)
3. Run `train_on_hm.py` (below) — produces `model_weights.weights.h5`
4. Copy `.h5` file to `ml_service/` in the project
5. Restart the ML service — it auto-loads the weights

## Training Parameters

- **Epochs:** 50 (recommended) — more = better accuracy
- **Products:** Top 2000+ by sales volume
- **Window:** 30 days input → 7 days output
- **Log normalization:** `log1p()` applied to sales data
- **Validation split:** 20%
