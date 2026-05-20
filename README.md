# Recommendation Project

A production-style recommendation pipeline for top-K item recommendation using a
**two-stage architecture**: multi-source candidate generation → LambdaMART reranker.

Primary metric: **Precision@10** (PIR scoring).

---

## Table of Contents

1. [Project Goals](#1-project-goals)
2. [Repository Structure](#2-repository-structure)
3. [System Architecture](#3-system-architecture)
4. [Data Requirements](#4-data-requirements)
5. [Environment Setup](#5-environment-setup)
6. [Quick Start (run_pipeline.sh)](#6-quick-start)
7. [Manual Step-by-Step](#7-manual-step-by-step)
8. [Evaluation and Metrics](#8-evaluation-and-metrics)
9. [Outputs and Artifacts](#9-outputs-and-artifacts)
10. [Configuration Reference](#10-configuration-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Suggested Workflow](#12-suggested-workflow)

## 1. Project Goals

Predict top-10 items each customer will purchase in January 2026, trained on 2025 transaction history.

| Split | Months | Purpose |
|---|---|---|
| Train | Jan – Oct 2025 (1–10) | Feature computation + ranker labels |
| Validation | Nov 2025 (11) | Offline evaluation |
| Test / labels | Dec 2025 (12) | Final ranker training labels |
| **Prediction target** | **Jan 2026** | **Submission output** |

---

## 2. Repository Structure

```text
.
├── run_pipeline.sh              ← end-to-end script (train + predict)
├── train_twostage.py            ← two-stage training entry point
├── predict_twostage.py          ← two-stage inference entry point
├── train.py                     ← single-stage ALS / popularity baseline
├── predict.py                   ← single-stage inference
├── src/
│   ├── config.py                ← ALL hyperparameters and paths
│   ├── data/loader.py           ← data loading + temporal split
│   ├── candidates/
│   │   ├── history.py           ← candidate source 1: purchase history
│   │   ├── covisitation.py      ← candidate source 2: item co-occurrence (BᵀB)
│   │   └── word2vec_cands.py    ← candidate source 3: W2V nearest neighbours
│   ├── features/
│   │   ├── user_features.py     ← ~25 user-level statistical features
│   │   ├── item_features.py     ← ~20 item-level features + metadata encoding
│   │   └── cross_features.py   ← user×item interaction features (incl. category affinity)
│   ├── ranker/
│   │   └── lgbm_ranker.py       ← LambdaMART / binary classifier wrapper
│   └── evaluation/metrics.py   ← PIR metrics + recall / NDCG
├── pipeline/
│   ├── stage1_candidates.py     ← orchestrate 3-source candidate generation
│   └── stage2_reranking.py      ← build feature matrix + sample training pairs
└── outputs/
    ├── checkpoints/             ← saved artifacts (covisit, w2v, ranker)
    └── predictions/             ← JSON prediction files + logs
```

---

## 3. System Architecture

### Stage 1 — Candidate Generation

For each user, up to `MAX_CANDIDATES = 1200` candidate items are assembled from three sources:

| Source | Config key | Description |
|---|---|---|
| **History** | `HISTORY_MAX_ITEMS = 200` | Items the user has bought before (recency-sorted) |
| **Covisitation** | `COVISIT_CANDS = 300` | Items that co-occur in the same bill as history items (BᵀB sparse matrix) |
| **Word2Vec** | `W2V_CANDS = 200` | ANN nearest neighbours from a recency-weighted user embedding |

Candidates are deduplicated in priority order (history → covisitation → Word2Vec) and filtered
to active items (`sale_status == 1`, ~6 850 items).

### Stage 2 — LambdaMART Reranking

The ranker is trained on ~1.5 M labelled `(user, candidate_item)` pairs.

#### Feature Groups (~58 features)

| Group | Count | Key signals |
|---|---|---|
| **User features** | ~25 | transaction count, recency, spend, monthly activity patterns |
| **Item features** | ~20 | popularity rank, buyer diversity, trend score, category / brand encoding |
| **Cross features** | ~13 | history interaction stats, covisit score, W2V score, user-category affinity, stage-1 rank |

##### Cross features added in May 2026

| Feature | Description |
|---|---|
| `ui_user_cat1_count` | Times the user purchased from the candidate item's `category_l1` |
| `ui_user_cat2_count` | Same for `category_l2` (more specific) |
| `ui_user_cat1_pct` | Fraction of user's total purchases in `category_l1` |
| `ui_user_cat2_pct` | Fraction in `category_l2` |
| `stage1_rank` | Position of the item in the Stage-1 candidate list (1 = highest priority) |

#### Ranker Model — LambdaMART

| Parameter | Value | Notes |
|---|---|---|
| `objective` | `lambdarank` | Directly optimises NDCG (listwise) |
| `n_estimators` | 1 000 | Boosting rounds |
| `learning_rate` | 0.05 | |
| `max_depth` | 6 | |
| `num_leaves` | 63 | |
| `min_child_samples` | 10 | Lower than LGBM default for better ranking granularity |

> **Why LambdaMART instead of binary classification?**
> Binary cross-entropy treats each `(user, item)` pair independently — the model has no knowledge
> that items belong to the same user.  LambdaMART optimises NDCG *within each user's candidate
> list*, which directly improves Precision@10 and MAP.

#### Training Procedure

1. Sample `RANKER_TRAIN_USERS = 100 000` users from the training period.
2. Run Stage-1 candidate generation for those users (up to 1 200 candidates each).
3. Build feature matrix (~58 columns: user + item + cross features).
4. Label pairs: `label = 1` if the item was purchased in the label period.
5. Downsample negatives: keep `NEG_RATIO × n_positives` random negatives.
6. Sort by `customer_id`, build LightGBM group array, train `LGBMRanker(lambdarank)`.

---

## 4. Data Requirements

Place the following parquet files in the project root:

```
transaction_full_2025.parquet
event_full_2025.parquet
items.parquet
```

**Transaction columns:** `customer_id`, `item_id`, `bill_id`, `quantity`, `price`, `discount`, `updated_date` (datetime)

**Item metadata columns:** `item_id`, `sale_status` (1 = active), `category_l1`, `category_l2`, `brand`, `price`

---

## 5. Environment Setup

```bash
pip install uv        # if uv is not yet installed
uv sync               # create .venv + install all dependencies
```

Core libraries: `polars`, `numpy`, `scipy`, `lightgbm`, `gensim`, `scikit-learn`

---

## 6. Quick Start

The recommended way to run the full pipeline is `run_pipeline.sh`.

### 6.1 Full retrain → Jan 2026 prediction (submission)

```bash
bash run_pipeline.sh
```

This runs:
1. `train_twostage.py --final-2025 ...` — Stage-1 artifacts + LambdaMART ranker using
   months 1–11 as history and month 12 as label data.
2. `predict_twostage.py --target-split jan2026 ...` — top-10 predictions for all ~862 k users.

### 6.2 Retrain ranker only (reuse existing covisit / W2V)

```bash
bash run_pipeline.sh --skip-stage1
```

Skips the expensive Stage-1 artifact building (~15–20 min).
Use this when iterating on features or hyperparameters.

### 6.3 Evaluate on validation set

```bash
bash run_pipeline.sh --eval
```

Trains on months 1–10 and evaluates Precision@10 / MAP / IoU / MRR on month 11.

### 6.4 Environment variable overrides

```bash
WORKERS=8 N_USERS=50000 bash run_pipeline.sh --skip-stage1 --eval
```

| Variable | Default | Description |
|---|---|---|
| `WORKERS` | `12` | CPU threads for training and inference |
| `N_USERS` | `100000` | Training users sampled for the ranker |
| `BATCH_SIZE` | `3000` | Users per Stage-1 processing batch |
| `EVAL_USERS` | `10000` | Users evaluated during `--eval` mode |
| `RANKER_TYPE` | `lambdarank` | `lambdarank` or `binary` |
| `NEG_RATIO` | `20` | Negatives per positive in training |

---

## 7. Manual Step-by-Step

### 7.1 Two-Stage Training

Full training from scratch (months 1–11 history, month 12 labels):

```bash
uv run python train_twostage.py \
    --final-2025 \
    --n-users 100000 \
    --stage1-batch-size 3000 \
    --w2v-workers 12 \
    --ranker-type lambdarank \
    --no-eval
```

Retrain ranker only (reuse saved Stage-1 artifacts):

```bash
uv run python train_twostage.py \
    --skip-stage1 \
    --n-users 100000 \
    --stage1-batch-size 3000 \
    --w2v-workers 12 \
    --ranker-type lambdarank \
    --eval-max-users 10000
```

Quick debug run (50 k users, no W2V):

```bash
uv run python train_twostage.py --skip-stage1 --n-users 50000 --no-w2v --ranker-type binary
```

#### `train_twostage.py` flags

| Flag | Default | Description |
|---|---|---|
| `--final-2025` | off | Months 1–11 history, month 12 labels (for Jan-2026 prediction) |
| `--skip-stage1` | off | Load covisit + W2V from checkpoints instead of rebuilding |
| `--n-users` | 100 000 | Users sampled for ranker training |
| `--stage1-batch-size` | 2 000 | Users per candidate-generation batch |
| `--w2v-workers` | auto | Word2Vec training threads |
| `--ranker-type` | `lambdarank` | `lambdarank` (recommended) or `binary` |
| `--neg-ratio` | 20 | Negatives per positive |
| `--eval-max-users` | 20 000 | Users for validation evaluation (0 = all) |
| `--no-eval` | off | Skip validation evaluation after training |
| `--no-w2v` | off | Skip Word2Vec (faster, lower recall) |

### 7.2 Two-Stage Prediction

Jan 2026 prediction:

```bash
uv run python predict_twostage.py \
    --target-split jan2026 \
    --batch-size 3000 \
    --workers 12 \
    --save-every-batches 1
```

Predict and evaluate validation split:

```bash
uv run python predict_twostage.py \
    --target-split val \
    --batch-size 3000 \
    --workers 12 \
    --quick-metrics
```

Resume an interrupted run (default — no extra flag needed):

```bash
uv run python predict_twostage.py --target-split jan2026
```

Force restart from scratch:

```bash
uv run python predict_twostage.py --target-split jan2026 --no-resume
```

#### `predict_twostage.py` flags

| Flag | Default | Description |
|---|---|---|
| `--target-split` | `test` | `val`, `test`, or `jan2026` |
| `--batch-size` | 5 000 | Users processed per batch |
| `--workers` | 0 (auto) | CPU threads |
| `--save-every-batches` | 1 | Checkpoint partial results every N batches |
| `--no-resume` | off | Ignore existing partial checkpoint |
| `--max-users` | 0 (all) | Limit to first N users (smoke tests) |
| `--quick-metrics` | off | Compute PIR metrics after prediction (val only) |
| `--no-w2v` | off | Skip W2V candidates at inference |
| `--log-file` | auto | Path to log file |
| `--log-overwrite` | off | Overwrite log instead of appending |

### 7.3 Single-Stage Baseline

```bash
uv run python train.py                           # ALS
uv run python train.py --model popularity        # popularity baseline
uv run python predict.py                         # predict test split
uv run python predict.py --target-split val      # predict val split
```

---

## 8. Evaluation and Metrics

All metrics are computed in `src/evaluation/metrics.py` via `compute_pir_metrics()`.

| Metric | Output key | Description |
|---|---|---|
| **Precision@10** ⭐ | `precision_at_10` | Fraction of top-10 predictions matching ground truth; denominator = `min(10, \|GT\|)` |
| **MAP** | `map` | Mean Average Precision over the full ranked list |
| **IoU** | `iou` | Intersection-over-Union of recommended set vs ground-truth set, averaged across users |
| **MRR** | `reciprocal_rank_first_hit` | Mean Reciprocal Rank — `1 / rank` of the first correct item |
| Recall@K | `recall@10` | Standard recall at cut-off K |
| NDCG@K | `ndcg@10` | Normalised Discounted Cumulative Gain |
| Hit Rate@K | `hit_rate@10` | Fraction of users with ≥ 1 correct item in top-K |

> Precision@10 is the **primary metric** for Kiosk recommendation scoring.

Metrics are printed to stdout during training (when `--no-eval` is not set) and written to
`outputs/val_metrics_twostage.json`.

---

## 9. Outputs and Artifacts

### Checkpoints (`outputs/checkpoints/`)

| File | Description |
|---|---|
| `covisit.pkl` | Sparse item-item co-occurrence matrix (Stage-1) |
| `w2v/w2v.model` | Trained Gensim Word2Vec model |
| `w2v/emb_artifacts.pkl` | Item embedding matrix + item list |
| `lgbm_ranker.pkl` | Trained `LGBMRanker` (LambdaMART) with saved feature list |
| `items_df.pkl` | Items metadata snapshot used during training |

### Predictions (`outputs/predictions/`)

| File | Description |
|---|---|
| `predictions_twostage_jan2026.json` | Final Jan-2026 submission |
| `predictions_twostage_val.json` | Validation-split predictions |
| `predictions_twostage_test.json` | Test-split predictions |
| `*.partial.json` | Resumable checkpoint during inference |
| `predict_twostage_<split>.log` | Per-run log file |
| `val_metrics_twostage.json` | Saved evaluation metrics |

**Prediction format:**

```json
{
  "123456": ["item_A", "item_B", "item_C"],
  "789012": ["item_X", "item_Y"]
}
```

Keys = `customer_id` as strings; values = ordered `item_id` list (best first, max 10).

---

## 10. Configuration Reference

All tuneable constants are in `src/config.py`.

### Stage-1 parameters

| Constant | Default | Description |
|---|---|---|
| `HISTORY_MAX_ITEMS` | 200 | Max items from user purchase history |
| `COVISIT_TOP_K` | 50 | Top-K co-occurring items kept per item |
| `COVISIT_CANDS` | 300 | Covisitation candidates per user |
| `W2V_CANDS` | 200 | Word2Vec candidates per user |
| `W2V_RECENT` | 20 | Recent items used to build W2V query vector |
| `MAX_CANDIDATES` | 1 200 | Total Stage-1 candidate cap per user |

### Stage-2 / ranker parameters

| Constant | Default | Description |
|---|---|---|
| `RANKER_OBJECTIVE` | `lambdarank` | Ranker loss function |
| `RANKER_N_ESTIMATORS` | 1 000 | Boosting rounds |
| `RANKER_LEARNING_RATE` | 0.05 | Shrinkage |
| `RANKER_MAX_DEPTH` | 6 | Max tree depth |
| `RANKER_NUM_LEAVES` | 63 | Max leaves per tree |
| `RANKER_SUBSAMPLE` | 0.8 | Row subsampling ratio |
| `RANKER_COLSAMPLE` | 0.8 | Column subsampling ratio |
| `RANKER_MIN_CHILD_SAMPLES` | 10 | Min samples per leaf |
| `RANKER_TRAIN_USERS` | 100 000 | Users sampled for ranker training |
| `RANKER_NEG_RATIO` | 20 | Negatives per positive |
| `RANKER_TOP_K_OUTPUT` | 10 | Items in final output per user |

---

## 11. Troubleshooting

### `FileNotFoundError` — missing checkpoint
Run training first. Use `--skip-stage1` only after a prior full training run.

### `LightGBM` import error
```bash
uv sync
```

### Memory error during inference
Reduce `--batch-size` (e.g., `--batch-size 1000`). Inference is fully resumable.

### Slow training
- Use `--skip-stage1` when only the ranker needs updating.
- Use `--n-users 50000` for fast iteration cycles.
- Use `--no-w2v` to skip Word2Vec (lowers recall, ~3× faster Stage-1).

### Feature count mismatch at inference (`LGBMRanker` error)
Happens when loading a ranker saved before new features were added, or when switching
between `lambdarank` and `binary` objectives.

**Fix:** delete `outputs/checkpoints/lgbm_ranker.pkl` and retrain.

---

## 12. Suggested Workflow

```
1. First run (from scratch, ~1–2 hours):
   bash run_pipeline.sh

2. Fast iteration (ranker only, with evaluation, ~20 min):
   bash run_pipeline.sh --skip-stage1 --eval
   # → inspect Precision@10 / MAP / IoU / MRR in stdout

3. Tune src/config.py or features, then repeat step 2.

4. Final submission:
   bash run_pipeline.sh
   # → outputs/predictions/predictions_twostage_jan2026.json
```
