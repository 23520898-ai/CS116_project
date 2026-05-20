"""
Main training pipeline.

Usage
-----
python train.py                           # ALS model (default)
python train.py --model popularity        # popularity baseline
python train.py --use-events              # add event behavioural signals
python train.py --top-k 20               # change recommendation size
python train.py --no-eval                 # skip validation evaluation
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time

import numpy as np

from src.config import (
    OUTPUTS_DIR, CHECKPOINTS_DIR, PREDICTIONS_DIR, TOP_K,
    ALS_FACTORS, ALS_ITERATIONS, ALS_REGULARIZATION, ALS_ALPHA,
)
from src.data.loader import (
    load_transactions, load_events,
    split_transactions, split_events,
    print_split_stats,
)
from src.features.builder import build_user_item_matrix
from src.evaluation.metrics import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a recommendation model.")
    p.add_argument("--model", choices=["als", "popularity"], default="als",
                   help="Model type (default: als)")
    p.add_argument("--use-events", action="store_true",
                   help="Include event data as additional signals.")
    p.add_argument("--top-k", type=int, default=TOP_K,
                   help=f"Number of recommendations per user (default: {TOP_K})")
    p.add_argument("--factors", type=int, default=ALS_FACTORS)
    p.add_argument("--iterations", type=int, default=ALS_ITERATIONS)
    p.add_argument("--alpha", type=float, default=ALS_ALPHA)
    p.add_argument("--no-eval", action="store_true",
                   help="Skip validation evaluation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load & split transactions ─────────────────────────────────────────────
    log.info("Loading transaction data …")
    t0 = time.time()
    trans_train, trans_val, trans_test = split_transactions(load_transactions())
    log.info("Loaded in %.1f s", time.time() - t0)
    print_split_stats(trans_train, trans_val, trans_test, "Transactions")

    # ── Optionally load events ────────────────────────────────────────────────
    event_train = None
    if args.use_events:
        log.info("Loading event data …")
        event_train, _, _ = split_events(load_events())
        log.info("Event train rows: %d", len(event_train))

    # ── Build interaction matrix ──────────────────────────────────────────────
    log.info("Building user-item interaction matrix …")
    t0 = time.time()
    matrix, user2idx, item2idx = build_user_item_matrix(trans_train, event_train)
    idx2item = {v: k for k, v in item2idx.items()}
    idx2user = {v: k for k, v in user2idx.items()}
    log.info("Matrix built in %.1f s — shape=%s  nnz=%d", time.time() - t0, matrix.shape, matrix.nnz)

    # ── Train model ───────────────────────────────────────────────────────────
    if args.model == "als":
        from src.models.als_model import ALSRecommender
        model = ALSRecommender(
            factors=args.factors,
            iterations=args.iterations,
            regularization=ALS_REGULARIZATION,
            alpha=args.alpha,
        )
    else:
        from src.models.popularity import PopularityRecommender
        model = PopularityRecommender(top_k=args.top_k)

    log.info("Training %s model …", args.model)
    t0 = time.time()
    model.fit(matrix, idx2item=idx2item)
    log.info("Training done in %.1f s", time.time() - t0)

    # ── Persist model + artifacts ─────────────────────────────────────────────
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path     = CHECKPOINTS_DIR / f"{args.model}_model.pkl"
    artifacts_path = CHECKPOINTS_DIR / "artifacts.pkl"

    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    with open(artifacts_path, "wb") as fh:
        pickle.dump({
            "user2idx": user2idx, "item2idx": item2idx,
            "idx2item": idx2item, "idx2user": idx2user,
            "matrix":   matrix,
        }, fh)
    log.info("Model saved → %s", model_path)

    # ── Evaluate on validation set ────────────────────────────────────────────
    if not args.no_eval:
        log.info("Evaluating on validation set (month 11) …")
        val_users = [u for u in trans_val["customer_id"].unique().to_list() if u in user2idx]
        log.info("Scoring %d users …", len(val_users))

        t0 = time.time()
        preds = model.recommend(
            user_ids=val_users,
            user2idx=user2idx,
            idx2item=idx2item,
            top_k=args.top_k,
            filter_purchased=matrix,
        )
        log.info("Recommendations generated in %.1f s", time.time() - t0)

        metrics = evaluate(preds, trans_val, k=args.top_k)

        print("\n── Validation metrics ──────────────────────────────────────")
        for name, val in metrics.items():
            if isinstance(val, float):
                print(f"  {name:25s}: {val:.4f}")
            else:
                print(f"  {name:25s}: {val:,}")
        print("────────────────────────────────────────────────────────────\n")

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = OUTPUTS_DIR / "val_metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        log.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    main()
