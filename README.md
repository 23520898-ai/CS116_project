# Recommendation Project

A production-style recommendation pipeline built for a machine-learning project, with two modeling tracks:

- Single-stage baseline: ALS collaborative filtering and popularity baseline.
- Two-stage recommender: multi-source candidate generation + LightGBM reranking.

The codebase is optimized for large-scale offline inference with batching, checkpointing, and reusable artifacts.

## Table of Contents

1. Project Goals
2. Repository Structure
3. System Architecture
4. Data Requirements
5. Environment Setup
6. How to Run
7. Evaluation and Metrics
8. Outputs and Artifacts
9. Configuration Reference
10. Troubleshooting
11. Suggested Workflow

## 1. Project Goals

This project solves top-K item recommendation for customers with a temporal split:

- Train period: months 1-10 (Jan-Oct 2025)
- Validation period: month 11 (Nov 2025)
- Test period: month 12 (Dec 2025)

Main objectives:

- Generate high-quality personalized recommendations.
- Compare a strong collaborative-filtering baseline against a richer two-stage pipeline.
- Produce submission-ready JSON predictions.

## 2. Repository Structure

```text
.
|-- train.py                     # Train single-stage model (ALS or popularity)
|-- predict.py                   # Inference for single-stage model
|-- train_twostage.py            # Train two-stage system
|-- predict_twostage.py          # Inference for two-stage system
|-- requirements.txt
|-- pyproject.toml
|-- Data exam.ipynb              # Notebook for exploration/experiments
|-- pipeline/
|   |-- stage1_candidates.py     # Merge 3 candidate sources
|   `-- stage2_reranking.py      # Build features + sample training pairs
|-- src/
|   |-- config.py                # Paths, split config, hyperparameters
|   |-- data/loader.py           # Data loading and temporal split utilities
|   |-- candidates/
|   |   |-- history.py           # Candidate source 1
|   |   |-- covisitation.py      # Candidate source 2
|   |   `-- word2vec_cands.py    # Candidate source 3
|   |-- features/
|   |   |-- builder.py           # User-item matrix builder (single-stage)
|   |   |-- user_features.py
|   |   |-- item_features.py
|   |   `-- cross_features.py
|   |-- models/
|   |   |-- als_model.py
|   |   |-- popularity.py
|   |   `-- base.py
|   |-- ranker/lgbm_ranker.py
|   `-- evaluation/metrics.py
`-- outputs/
    |-- checkpoints/
    `-- predictions/
```

## 3. System Architecture

### 3.1 Single-Stage Baseline

1. Load and split transactions by month.
2. Build sparse user-item matrix from purchase quantity (and optional event weights).
3. Train model:
   - ALS (`implicit`) or
   - Popularity baseline.
4. Predict top-K items per target user.
5. Evaluate on validation split (optional).

When to use:

- Fast baseline benchmarking.
- Simpler pipeline with lower engineering complexity.

### 3.2 Two-Stage Recommender

#### Stage 1: Candidate Generation

Three candidate sources are merged per user:

- History candidates: most recently purchased items.
- Covisitation candidates: graph-style co-occurrence from bills (`B^T B`).
- Word2Vec candidates: nearest neighbors from recency-weighted user embedding.

Candidates are deduplicated and capped (`MAX_CANDIDATES`).

#### Stage 2: Reranking

1. Build feature matrix by joining:
   - User features (~25)
   - Item features (~20)
   - User-item cross features
2. Attach labels using validation purchases for training.
3. Negative sampling with configurable ratio.
4. Train LightGBM binary classifier (`LGBMClassifier`).
5. Score candidates and keep top-K per user.

When to use:

- Better ranking quality than pure collaborative filtering.
- Flexible feature engineering and model iteration.

## 4. Data Requirements

Place the following parquet files in the project root:

- `transaction_full_2025.parquet`
- `event_full_2025.parquet`
- `items.parquet`

### 4.1 Required Transaction Columns

Expected by training/feature modules:

- `customer_id` (int-like)
- `item_id` (string-like)
- `bill_id`
- `quantity`
- `price`
- `discount`
- `updated_date` (datetime)

### 4.2 Required Event Columns

Expected by optional event integration:

- `customer_id`
- `item_id`
- `event_type` (e.g., `view`, `wishlist`, `add_to_cart`, `purchase`)
- `event_date` (datetime)

### 4.3 Required Item Metadata Columns

Expected by item feature builder:

- `item_id`
- `sale_status` (1 means currently active/sellable)
- `category_l1`
- `category_l2`
- `brand`
- `price`

## 5. Environment Setup

### 5.1 Python Version

- Python 3.11+

### 5.2 Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

Core libraries used:

- `polars`, `numpy`, `scipy`
- `implicit` (ALS)
- `gensim` (Word2Vec)
- `lightgbm` (reranker)
- `scikit-learn`

## 6. How to Run

All commands assume you run from project root.

### 6.1 Single-Stage Training

Default ALS:

```bash
uv run python train.py
```

Train popularity baseline:

```bash
uv run python train.py --model popularity
```

Use event behavioral signals:

```bash
uv run python train.py --use-events
```

Useful options:

- `--top-k 20`
- `--factors 128`
- `--iterations 20`
- `--alpha 40`
- `--no-eval`

### 6.2 Single-Stage Prediction

Predict for test month (month 12):

```bash
uv run python predict.py
```

Predict for validation month:

```bash
uv run python predict.py --target-split val
```

Use popularity model:

```bash
uv run python predict.py --model-type popularity --top-k 20
```

### 6.3 Two-Stage Training

Full training:

```bash
uv run python train_twostage.py
```

Faster debug run (fewer users, no Word2Vec):

```bash
uv run python train_twostage.py --n-users 50000 --top-k 10 --no-w2v
```

Reuse existing Stage-1 artifacts:

```bash
uv run python train_twostage.py --skip-stage1
```

Useful options:

- `--stage1-batch-size 2000`
- `--neg-ratio 20`
- `--eval-max-users 20000`
- `--no-eval`
- `--w2v-workers <n>`

### 6.4 Two-Stage Prediction

Predict for test split:

```bash
uv run python predict_twostage.py
```

Predict validation split + quick metrics:

```bash
uv run python predict_twostage.py --target-split val --quick-metrics
```

Memory-friendly / resumable run:

```bash
uv run python predict_twostage.py --batch-size 5000 --save-every-batches 1
```

Useful options:

- `--max-users 10000` for quick smoke tests
- `--no-resume` to ignore partial checkpoint
- `--workers <n>` to cap runtime threads
- `--no-w2v` to skip W2V candidates at inference

## 7. Evaluation and Metrics

Evaluation is implemented in `src/evaluation/metrics.py`.

Reported metrics include:

- `recall@K`
- `precision@K`
- `ndcg@K`
- `hit_rate@K`
- PIR-style metrics:
  - `total_correct_recommendations`
  - `precision_at_10`
  - `map`
  - `iou`
  - `reciprocal_rank_first_hit`

Notes:

- For `K=10`, precision key is aligned with PIR denominator behavior.
- Quick validation metrics can be computed directly from `predict_twostage.py --quick-metrics`.

## 8. Outputs and Artifacts

### 8.1 Checkpoints (`outputs/checkpoints/`)

Single-stage:

- `als_model.pkl` or `popularity_model.pkl`
- `artifacts.pkl` (mappings + interaction matrix)

Two-stage:

- `covisit.pkl`
- `w2v/w2v.model`
- `w2v/emb_artifacts.pkl`
- `lgbm_ranker.pkl`
- `items_df.pkl`

### 8.2 Predictions (`outputs/predictions/`)

- `predictions_test.json`
- `predictions_val.json`
- `predictions_twostage_test.json`
- `predictions_twostage_val.json`
- `*.partial.json` during resumable two-stage inference
- optional log files: `predict_twostage_<split>.log`

Prediction format:

```json
{
  "<customer_id>": ["<item_id_1>", "<item_id_2>", "..."],
  "...": ["..."]
}
```

## 9. Configuration Reference

Main constants are in `src/config.py`:

- Paths: data files and output folders.
- Time split: train/val/test months.
- Single-stage params: `TOP_K`, ALS hyperparameters.
- Stage-1 params:
  - candidate limits (`HISTORY_MAX_ITEMS`, `COVISIT_CANDS`, `W2V_CANDS`)
  - cap (`MAX_CANDIDATES`)
  - batching (`STAGE1_BATCH_SIZE`)
  - Word2Vec hyperparameters.
- Stage-2 params:
  - LightGBM defaults.
  - sampling defaults (`RANKER_TRAIN_USERS`, `RANKER_NEG_RATIO`).

If you need to tune behavior globally, start from `src/config.py` before changing scripts.

## 10. Troubleshooting

### Missing model/checkpoint files

If prediction fails with missing artifacts:

- Run the corresponding training script first.
- Ensure files are saved under `outputs/checkpoints/`.

### LightGBM / Gensim import errors

Install missing libraries:

```bash
uv sync
```

### Memory pressure during two-stage inference

- Reduce `--batch-size`.
- Use `--workers` to limit thread count.
- Use resume mode (default) so interrupted jobs can continue.

### Slow run time

- Disable W2V temporarily with `--no-w2v`.
- Use smaller `--n-users` for experimentation.
- Use `--eval-max-users` for quicker validation cycles.

## 11. Suggested Workflow

1. Run single-stage ALS to verify data and baseline quality.
2. Train two-stage system with moderate user sample.
3. Inspect feature importance from ranker logs.
4. Iterate on feature engineering and hyperparameters.
5. Run full-scale two-stage prediction for final output.

---

If needed, this README can be extended with:

- experiment tracking template,
- parameter sweep recipes,
- and leaderboard/report-ready result tables.
