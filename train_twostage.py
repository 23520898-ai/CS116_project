"""
Two-Stage Recommendation Training Pipeline
===========================================

Giai đoạn 1 – Candidate Generation:
  • Lưới 1: History  (past purchases)
  • Lưới 2: Covisitation matrix (bill-based item co-occurrence)
  • Lưới 3: Word2Vec (sequence-based deep embeddings)

Giai đoạn 2 – Reranking:
  • Feature engineering: user / item / user×item (~60 features)
  • LightGBM binary classifier  → top-K per user

Usage
-----
python train_twostage.py
python train_twostage.py --n-users 50000 --top-k 10 --no-w2v
python train_twostage.py --skip-stage1   # reload saved Stage-1 artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.config import (
    CHECKPOINTS_DIR, OUTPUTS_DIR, PREDICTIONS_DIR,
    STAGE1_BATCH_SIZE,
    COVISIT_TOP_K, COVISIT_MAX_BILL,
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS, W2V_WORKERS,
    RANKER_N_ESTIMATORS, RANKER_LEARNING_RATE, RANKER_MAX_DEPTH,
    RANKER_NUM_LEAVES, RANKER_SUBSAMPLE, RANKER_COLSAMPLE,
    RANKER_OBJECTIVE, RANKER_MIN_CHILD_SAMPLES,
    RANKER_TRAIN_USERS, RANKER_NEG_RATIO, RANKER_TOP_K_OUTPUT,
)
from src.data.loader import (
    load_transactions, load_items,
    split_transactions, print_split_stats,
)
from src.candidates.covisitation  import build_covisitation_matrix, save_covisit, load_covisit
from src.candidates.word2vec_cands import train_word2vec, build_embedding_matrix, save_w2v_artifacts, load_w2v_artifacts
from src.ranker.lgbm_ranker        import LGBMRanker
from src.evaluation.metrics        import evaluate, compute_pir_metrics
from src.features.user_features    import build_user_features
from src.features.item_features    import build_item_features

from pipeline.stage1_candidates import generate_candidates_for_users, candidates_to_dataframe
from pipeline.stage2_reranking  import build_feature_matrix, sample_training_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_twostage")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-users",      type=int,   default=RANKER_TRAIN_USERS,
                   help="Users sampled for ranker training (default: 100 000)")
    p.add_argument("--stage1-batch-size", type=int, default=STAGE1_BATCH_SIZE,
                   help="Users per batch for stage-1 candidate generation.")
    p.add_argument("--w2v-workers", type=int, default=W2V_WORKERS,
                   help="Word2Vec workers (CPU threads).")
    p.add_argument("--eval-max-users", type=int, default=20_000,
                   help="Max users for validation evaluation (0 = all available).")
    p.add_argument("--top-k",        type=int,   default=RANKER_TOP_K_OUTPUT)
    p.add_argument("--neg-ratio",    type=int,   default=RANKER_NEG_RATIO)
    p.add_argument("--ranker-type",  choices=["lambdarank", "binary"],
                   default=RANKER_OBJECTIVE,
                   help="Ranker objective: lambdarank (default, recommended) or binary.")
    p.add_argument("--no-w2v",       action="store_true",
                   help="Skip Word2Vec (faster, lower recall)")
    p.add_argument("--skip-stage1",  action="store_true",
                   help="Load pre-built Stage-1 artifacts from checkpoints/")
    p.add_argument("--final-2025", action="store_true",
                   help="Final retrain mode: use Jan-Nov as history and Dec as labels (for Jan-2026 prediction).")
    p.add_argument("--no-eval",      action="store_true")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stage1_artifact_paths() -> dict[str, Path]:
    return {
        "covisit": CHECKPOINTS_DIR / "covisit.pkl",
        "w2v_dir": CHECKPOINTS_DIR / "w2v",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    # 0. Load data
    # ════════════════════════════════════════════════════════════════════════
    log.info("Loading transaction data …")
    t0 = time.time()
    trans_train, trans_val, trans_test = split_transactions(load_transactions())
    items_df = load_items()
    active_items = set(
        items_df.filter(pl.col("sale_status") == 1)["item_id"].cast(pl.Utf8).to_list()
    )
    log.info("Active items (sale_status=1): %d", len(active_items))
    log.info("Loaded in %.1f s", time.time() - t0)
    print_split_stats(trans_train, trans_val, trans_test, "Transactions")

    if args.final_2025:
        log.info("Final-2025 mode enabled: history=Jan-Nov, labels=Dec")
        history_train_df = pl.concat([trans_train, trans_val])
        label_df = trans_test
        if not args.no_eval:
            log.info("Disabling evaluation in final-2025 mode (no future holdout in 2025).")
        args.no_eval = True
    else:
        history_train_df = trans_train
        label_df = trans_val

    # ════════════════════════════════════════════════════════════════════════
    # 1. Stage 1 – Build / Load artifacts
    # ════════════════════════════════════════════════════════════════════════
    paths = _stage1_artifact_paths()

    if args.skip_stage1:
        log.info("Loading Stage-1 artifacts from disk …")
        covisit = load_covisit(paths["covisit"])
        if not args.no_w2v:
            w2v_model, emb_matrix, item_list = load_w2v_artifacts(paths["w2v_dir"])
        else:
            w2v_model = emb_matrix = item_list = None
    else:
        # ── Lưới 2: Covisitation matrix ──────────────────────────────────────
        t0 = time.time()
        covisit = build_covisitation_matrix(
            history_train_df,
            top_k_per_item=COVISIT_TOP_K,
            max_bill_size=COVISIT_MAX_BILL,
        )
        save_covisit(covisit, paths["covisit"])
        log.info("Covisitation built in %.1f s", time.time() - t0)

        # ── Lưới 3: Word2Vec ─────────────────────────────────────────────────
        if not args.no_w2v:
            t0 = time.time()
            w2v_model = train_word2vec(
                history_train_df,
                vector_size=W2V_VECTOR_SIZE,
                window=W2V_WINDOW,
                min_count=W2V_MIN_COUNT,
                epochs=W2V_EPOCHS,
                workers=max(1, args.w2v_workers),
            )
            emb_matrix, item_list = build_embedding_matrix(w2v_model)
            save_w2v_artifacts(w2v_model, emb_matrix, item_list, paths["w2v_dir"])
            log.info("Word2Vec built in %.1f s", time.time() - t0)
        else:
            w2v_model = emb_matrix = item_list = None
            log.info("Word2Vec skipped (--no-w2v).")

    # ════════════════════════════════════════════════════════════════════════
    # 2. Generate candidates for sampled training users
    #    (train on months 1-10, label with month 11)
    # ════════════════════════════════════════════════════════════════════════

    # --- Sample users for ranker training (e.g. 100k) ---
    n_train_users = max(1, args.n_users)
    rng = np.random.default_rng(42)
    all_train_users = history_train_df["customer_id"].unique().to_list()
    n_sample = min(n_train_users, len(all_train_users))
    sampled_users = rng.choice(all_train_users, size=n_sample, replace=False).tolist()
    log.info(
        "Generating Stage-1 candidates for %d training users (batch %d) …",
        n_sample,
        args.stage1_batch_size,
    )

    # --- Batch candidate generation ---
    batch_size = max(100, args.stage1_batch_size)
    cand_results = {}
    t0 = time.time()
    for i in range(0, n_sample, batch_size):
        batch_users = sampled_users[i:i+batch_size]
        batch_cand = generate_candidates_for_users(
            user_ids   = batch_users,
            trans_df   = history_train_df,
            covisit    = covisit,
            w2v_model  = w2v_model,
            emb_matrix = emb_matrix,
            item_list  = item_list,
            allowed_items=active_items,
        )
        cand_results.update(batch_cand)
        log.info(f"  ...done {i+len(batch_users):5d} / {n_sample} users ({(i+len(batch_users))/n_sample*100:.1f}%)")
    log.info("Candidates generated in %.1f s", time.time() - t0)

    # Flatten to DataFrame
    cands_df = candidates_to_dataframe(cand_results)

    # Attach covisit / w2v scores for cross features
    cov_scores = {uid: res["covisit_scores"] for uid, res in cand_results.items()}
    w2v_scs    = {uid: res["w2v_scores"]     for uid, res in cand_results.items()}

    # Ground truth for validation split (label=1 if purchased in month 11)
    val_pos = label_df.filter(
        pl.col("customer_id").is_in(sampled_users)
    ).select(["customer_id", "item_id"])

    # ════════════════════════════════════════════════════════════════════════
    # 3. Stage 2 – Feature engineering
    # ════════════════════════════════════════════════════════════════════════
    log.info("Building feature matrix …")
    t0 = time.time()
    log.info("Precomputing user/item features once for train/eval …")
    user_feat = build_user_features(history_train_df)
    item_feat = build_item_features(history_train_df, items_df)

    feature_df = build_feature_matrix(
        candidates_df  = cands_df,
        trans_df       = history_train_df,
        items_df       = items_df,
        covisit_scores = cov_scores,
        w2v_scores     = w2v_scs,
        label_df       = val_pos,
        user_feat      = user_feat,
        item_feat      = item_feat,
    )
    log.info("Feature matrix built in %.1f s  shape=%s", time.time() - t0, feature_df.shape)

    # Negative sampling to balance training set
    train_data = sample_training_pairs(feature_df, neg_ratio=args.neg_ratio)

    # ════════════════════════════════════════════════════════════════════════
    # 4. Train LGBMRanker
    # ════════════════════════════════════════════════════════════════════════
    ranker = LGBMRanker(
        n_estimators  = RANKER_N_ESTIMATORS,
        learning_rate = RANKER_LEARNING_RATE,
        max_depth     = RANKER_MAX_DEPTH,
        num_leaves    = RANKER_NUM_LEAVES,
        subsample     = RANKER_SUBSAMPLE,
        colsample     = RANKER_COLSAMPLE,
        objective     = args.ranker_type,
        min_child_samples = RANKER_MIN_CHILD_SAMPLES,
    )
    log.info("Training LGBMRanker …")
    t0 = time.time()
    ranker.fit(train_data, exclude_cols=["customer_id", "item_id"])
    log.info("Ranker trained in %.1f s", time.time() - t0)

    # Save ranker
    ranker.save(CHECKPOINTS_DIR / "lgbm_ranker.pkl")
    with open(CHECKPOINTS_DIR / "items_df.pkl", "wb") as fh:
        pickle.dump(items_df, fh)

    # Feature importance
    fi = ranker.feature_importance()
    print("\n── Top-20 feature importances ──────────────────────────────────")
    for row in fi.head(20).iter_rows(named=True):
        print(f"  {row['feature']:40s}  {row['importance']:>6}")
    print("────────────────────────────────────────────────────────────────\n")

    # ════════════════════════════════════════════════════════════════════════
    # 5. Evaluate on validation set
    # ════════════════════════════════════════════════════════════════════════
    if not args.no_eval:
        log.info("Evaluating on validation set …")
        # Score all candidates for sampled validation users

        sampled_users_set = set(sampled_users)
        val_users = [u for u in trans_val["customer_id"].unique().to_list() if u in sampled_users_set]
        if args.eval_max_users > 0:
            val_users = val_users[: args.eval_max_users]

        preds: dict[int, list[str]] = {}
        eval_batch_size = max(100, args.stage1_batch_size)
        n_eval = len(val_users)
        for i in range(0, n_eval, eval_batch_size):
            batch_users = val_users[i:i+eval_batch_size]
            log.info(
                "  eval progress: %d / %d users (%.1f%%)",
                i + len(batch_users),
                n_eval,
                ((i + len(batch_users)) / n_eval * 100.0) if n_eval else 100.0,
            )
            val_cand_results = generate_candidates_for_users(
                user_ids   = batch_users,
                trans_df   = history_train_df,
                covisit    = covisit,
                w2v_model  = w2v_model,
                emb_matrix = emb_matrix,
                item_list  = item_list,
                allowed_items=active_items,
            )
            val_cands_df  = candidates_to_dataframe(val_cand_results)
            val_cov_sc    = {uid: r["covisit_scores"] for uid, r in val_cand_results.items()}
            val_w2v_sc    = {uid: r["w2v_scores"]     for uid, r in val_cand_results.items()}
            val_feat_df = build_feature_matrix(
                candidates_df  = val_cands_df,
                trans_df       = history_train_df,
                items_df       = items_df,
                covisit_scores = val_cov_sc,
                w2v_scores     = val_w2v_sc,
                user_feat      = user_feat,
                item_feat      = item_feat,
            )
            preds.update(ranker.rank(val_feat_df, top_k=args.top_k))
        metrics = evaluate(preds, trans_val, k=args.top_k)

        print("── Validation metrics ──────────────────────────────────────────")
        pir_keys = ["precision_at_10", "map", "iou", "reciprocal_rank_first_hit",
                    "total_correct_recommendations"]
        print("  [ PIR / competition metrics ]")
        for k in pir_keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    print(f"  {k:35s}: {v:.6f}")
                else:
                    print(f"  {k:35s}: {v:,}")
        print("  [ additional ranking metrics ]")
        for name, val in metrics.items():
            if name in pir_keys:
                continue
            if isinstance(val, float):
                print(f"  {name:35s}: {val:.4f}")
            else:
                print(f"  {name:35s}: {val:,}")
        print("────────────────────────────────────────────────────────────────\n")

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUTS_DIR / "val_metrics_twostage.json", "w") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
