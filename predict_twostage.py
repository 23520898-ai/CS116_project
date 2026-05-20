"""
Two-Stage Recommendation – Prediction Script
=============================================
Loads all trained artifacts (covisit, W2V, LGBMRanker) and generates
final top-K recommendations for test customers (month 12).

Output: outputs/predictions/predictions_twostage_<split>.json
        {customer_id (str): [item_id, ...], ...}

Usage
-----
python predict_twostage.py                       # test split (month 12)
python predict_twostage.py --target-split val    # validation (month 11)
python predict_twostage.py --top-k 20 --batch-size 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from src.config import (
    CHECKPOINTS_DIR, PREDICTIONS_DIR,
    RANKER_TOP_K_OUTPUT,
)
from src.data.loader import load_transactions, split_transactions, load_items
from src.candidates.covisitation   import load_covisit
from src.candidates.word2vec_cands import load_w2v_artifacts
from src.ranker.lgbm_ranker        import LGBMRanker
from src.evaluation.metrics        import evaluate
from src.features.user_features    import build_user_features
from src.features.item_features    import build_item_features

from pipeline.stage1_candidates import generate_candidates_for_users, candidates_to_dataframe
from pipeline.stage2_reranking  import build_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict_twostage")


def _set_runtime_workers(workers: int) -> None:
    """Best-effort runtime thread limits for numeric backends."""
    n = max(1, int(workers))
    val = str(n)
    os.environ["OMP_NUM_THREADS"] = val
    os.environ["MKL_NUM_THREADS"] = val
    os.environ["OPENBLAS_NUM_THREADS"] = val
    os.environ["NUMEXPR_NUM_THREADS"] = val
    os.environ["POLARS_MAX_THREADS"] = val
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target-split", choices=["val", "test", "jan2026"], default="test",
                   help="val=month11, test=month12, jan2026=predict January 2026 using full 2025 history")
    p.add_argument("--top-k",        type=int, default=RANKER_TOP_K_OUTPUT)
    p.add_argument("--batch-size",   type=int, default=5_000,
                   help="Users processed per batch (controls peak memory).")
    p.add_argument("--max-users",    type=int, default=0,
                   help="Only predict first N users for quick checks (0 = all users).")
    p.add_argument("--quick-metrics", action="store_true",
                   help="When target-split=val, compute and print validation metrics.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing partial checkpoint and start from scratch.")
    p.add_argument("--save-every-batches", type=int, default=1,
                   help="Persist partial predictions every N batches.")
    p.add_argument("--workers", type=int, default=0,
                   help="CPU workers/threads (0 = library default).")
    p.add_argument("--log-file", type=str, default="",
                   help="Path to log file. Empty = outputs/predictions/predict_twostage_<split>.log")
    p.add_argument("--log-overwrite", action="store_true",
                   help="Overwrite log file instead of append mode.")
    p.add_argument("--no-w2v",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Configure persistent file logging.
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    default_log_path = PREDICTIONS_DIR / f"predict_twostage_{args.target_split}.log"
    log_path = Path(args.log_file) if args.log_file else default_log_path
    file_handler = logging.FileHandler(log_path, mode=("w" if args.log_overwrite else "a"), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)
    log.info("File logging enabled → %s", log_path)

    if args.workers and args.workers > 0:
        _set_runtime_workers(args.workers)
        log.info("Runtime workers set to %d", args.workers)

    # ── Load artifacts ────────────────────────────────────────────────────────
    log.info("Loading Stage-1 artifacts …")
    covisit = load_covisit(CHECKPOINTS_DIR / "covisit.pkl")

    if not args.no_w2v and (CHECKPOINTS_DIR / "w2v" / "w2v.model").exists():
        w2v_model, emb_matrix, item_list = load_w2v_artifacts(CHECKPOINTS_DIR / "w2v")
    else:
        w2v_model = emb_matrix = item_list = None
        log.info("W2V artifacts not found or skipped.")

    log.info("Loading LGBMRanker …")
    ranker    = LGBMRanker.load(CHECKPOINTS_DIR / "lgbm_ranker.pkl")
    if args.workers and args.workers > 0:
        ranker.n_jobs = args.workers
        if getattr(ranker, "_model", None) is not None:
            try:
                ranker._model.set_params(n_jobs=args.workers)
            except Exception:
                pass
    with open(CHECKPOINTS_DIR / "items_df.pkl", "rb") as fh:
        items_df = pickle.load(fh)
    active_items = set(
        items_df.filter(pl.col("sale_status") == 1)["item_id"].cast(pl.Utf8).to_list()
    )
    log.info("Active items (sale_status=1): %d", len(active_items))

    # ── Load transaction data ─────────────────────────────────────────────────
    log.info("Loading transaction data …")
    trans_train, trans_val, trans_test = split_transactions(load_transactions())

    # Choose target split and history
    if args.target_split == "test":
        target_df    = trans_test
        # Use months 1-11 as history for test prediction (more data = better)
        history_df   = pl.concat([trans_train, trans_val])
        split_label  = "test (month 12)"
    elif args.target_split == "jan2026":
        target_df    = trans_test
        # Use all 12 months 2025 as history for January 2026 prediction
        history_df   = pl.concat([trans_train, trans_val, trans_test])
        split_label  = "jan2026 (use full 2025 as history)"
    else:
        target_df    = trans_val
        history_df   = trans_train
        split_label  = "val (month 11)"

    target_users = target_df["customer_id"].unique().to_list()
    if args.max_users and args.max_users > 0:
        target_users = target_users[: args.max_users]
    log.info("Target users (%s): %d", split_label, len(target_users))

    # Build static features once, then reuse for every batch.
    log.info("Precomputing user/item features once for all batches …")
    user_feat = build_user_features(history_df)
    item_feat = build_item_features(history_df, items_df)

    # ── Output paths / optional resume ───────────────────────────────────────
    if args.max_users and args.max_users > 0:
        out_path = PREDICTIONS_DIR / (
            f"predictions_twostage_{args.target_split}_{args.max_users}users.json"
        )
    else:
        out_path = PREDICTIONS_DIR / f"predictions_twostage_{args.target_split}.json"
    partial_path = out_path.with_name(out_path.stem + ".partial.json")

    # ── Batch prediction ──────────────────────────────────────────────────────
    all_predictions: dict[int, list[str]] = {}
    if not args.no_resume and partial_path.exists():
        with open(partial_path, "r", encoding="utf-8") as fh:
            resumed = json.load(fh)
        all_predictions = {int(k): v for k, v in resumed.items()}
        log.info("Resuming from %s with %d users done.", partial_path, len(all_predictions))
    else:
        log.info("No resume checkpoint found at %s", partial_path)

    done_users = set(all_predictions.keys())
    if done_users:
        target_users = [u for u in target_users if u not in done_users]
        log.info("Remaining users after resume: %d", len(target_users))

    batch_size = args.batch_size
    n_batches  = (len(target_users) + batch_size - 1) // batch_size

    save_every = max(1, args.save_every_batches)
    for batch_idx in range(n_batches):
        batch_users = target_users[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        log.info(
            "Batch %d/%d  |  %d users …",
            batch_idx + 1, n_batches, len(batch_users),
        )

        # Stage 1
        cand_results = generate_candidates_for_users(
            user_ids   = batch_users,
            trans_df   = history_df,
            covisit    = covisit,
            w2v_model  = w2v_model,
            emb_matrix = emb_matrix,
            item_list  = item_list,
            allowed_items=active_items,
        )
        cands_df    = candidates_to_dataframe(cand_results)
        cov_scores  = {uid: r["covisit_scores"] for uid, r in cand_results.items()}
        w2v_scs     = {uid: r["w2v_scores"]     for uid, r in cand_results.items()}

        if cands_df.height == 0:
            log.info("Batch %d/%d has no valid candidates after filters; writing empty predictions.", batch_idx + 1, n_batches)
            for uid in batch_users:
                all_predictions[uid] = []
            continue

        # Stage 2: features + rank
        feat_df = build_feature_matrix(
            candidates_df  = cands_df,
            trans_df       = history_df,
            items_df       = items_df,
            covisit_scores = cov_scores,
            w2v_scores     = w2v_scs,
            user_feat      = user_feat,
            item_feat      = item_feat,
        )
        if feat_df.height == 0:
            log.info("Batch %d/%d feature matrix is empty; writing empty predictions.", batch_idx + 1, n_batches)
            for uid in batch_users:
                all_predictions[uid] = []
            continue

        batch_preds = ranker.rank(feat_df, top_k=args.top_k)
        for uid in batch_users:
            all_predictions[uid] = batch_preds.get(uid, [])

        if (batch_idx + 1) % save_every == 0:
            serialisable_partial = {str(k): v for k, v in all_predictions.items()}
            with open(partial_path, "w", encoding="utf-8") as fh:
                json.dump(serialisable_partial, fh)

    # ── Save ──────────────────────────────────────────────────────────────────
    serialisable = {str(k): v for k, v in all_predictions.items()}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh)
    if partial_path.exists():
        partial_path.unlink()

    log.info(
        "Saved %d predictions → %s",
        len(all_predictions), out_path,
    )

    if args.quick_metrics and args.target_split == "val":
        metrics = evaluate(all_predictions, target_df, k=args.top_k)
        print("\n── Quick validation metrics ───────────────────────────────────")
        for name, val in metrics.items():
            if isinstance(val, float):
                print(f"  {name:25s}: {val:.4f}")
            else:
                print(f"  {name:25s}: {val:,}")
        print("────────────────────────────────────────────────────────────────\n")

        metrics_path = PREDICTIONS_DIR / "val_metrics_twostage_quick.json"
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        log.info("Saved quick metrics → %s", metrics_path)

    print(f"\nSample (first 3):")
    for uid, items in list(all_predictions.items())[:3]:
        print(f"  customer {uid:>10}: {items}")


if __name__ == "__main__":
    main()
