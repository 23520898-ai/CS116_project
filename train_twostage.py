"""
Two-Stage Recommendation Training Pipeline (FIXED VERSION)
==========================================================
Fixed data leakage issues:
- Feature computation uses ONLY pre-label period data
- Candidate generation uses ONLY pre-label period history
- Proper temporal split for training/validation

Giai đoạn 1 – Candidate Generation:
  • Lưới 1: History  (past purchases - pre-label only)
  • Lưới 2: Covisitation matrix (bill-based item co-occurrence - pre-label only)
  • Lưới 3: Word2Vec (sequence-based deep embeddings - pre-label only)

Giai đoạn 2 – Reranking:
  • Feature engineering: user / item / user×item (~60 features)
  • Features computed from pre-label period only
  • LightGBM binary classifier → top-K per user

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
from typing import Optional, Tuple

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
    USE_EXTEND_LABELS, USE_HARD_NEGATIVES, HARD_NEG_FRACTION,
    USE_TEMPORAL_FEATURES, TEMPORAL_DECAY_RATE,
    USE_SESSION_FEATURES, SESSION_GAP_HOURS,
    USE_CATEGORY_AFFINITY, CAT_AFFINITY_DECAY_RATE,
    USE_ITEM_TRENDS, ITEM_TREND_WINDOWS,
    USE_UI_HISTORY, ENSEMBLE_SEEDS, ENSEMBLE_METHOD,
)
from src.data.loader import (
    load_transactions, load_items,
    split_transactions, print_split_stats,
)
from src.candidates.covisitation import build_covisitation_matrix, save_covisit, load_covisit
from src.candidates.word2vec_cands import train_word2vec, build_embedding_matrix, save_w2v_artifacts, load_w2v_artifacts
from src.ranker.lgbm_ranker import LGBMRanker
from src.evaluation.metrics import evaluate
from src.evaluation.ensemble import ensemble_rankers
from src.features.user_features import (
    build_user_features, build_temporal_decay_features, build_session_features,
)
from src.features.item_features import build_item_features, build_item_trend_features
from src.features.cross_features import (
    precompute_cross_stats, build_category_affinity_temporal, build_user_item_history_features,
)

from pipeline.stage1_candidates import generate_candidates_for_users, candidates_to_dataframe
from pipeline.stage2_reranking import build_feature_matrix, sample_training_pairs

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
    # ── Feature improvement flags ─────────────────────────────────────────────
    p.add_argument("--extend-labels",       action="store_true", default=USE_EXTEND_LABELS,
                   help="Use grade-0/1/2 labels (soft positives from same category).")
    p.add_argument("--use-hard-negatives",  action="store_true", default=USE_HARD_NEGATIVES,
                   help="Prioritise hard negatives (popular / same-category) when sampling.")
    p.add_argument("--use-temporal-features", action="store_true", default=USE_TEMPORAL_FEATURES,
                   help="Add exponential-decay temporal user/item features.")
    p.add_argument("--use-session-features", action="store_true", default=USE_SESSION_FEATURES,
                   help="Add session-based basket-size and frequency features.")
    p.add_argument("--use-category-affinity", action="store_true", default=USE_CATEGORY_AFFINITY,
                   help="Add temporal category affinity user/cross features.")
    p.add_argument("--use-item-trends",     action="store_true", default=USE_ITEM_TRENDS,
                   help="Add multi-window item popularity trend features.")
    p.add_argument("--use-ui-history",      action="store_true", default=USE_UI_HISTORY,
                   help="Add user-item purchase history features.")
    p.add_argument("--ensemble-seeds",      type=str, default="",
                   help="Comma-separated random seeds for ensemble training (e.g. 42,123,456).")
    p.add_argument("--ensemble-method",     type=str, default=ENSEMBLE_METHOD,
                   choices=["reciprocal_rank", "borda_count", "weighted_vote"],
                   help="Fusion method for ensemble predictions.")
    p.add_argument("--debug",               action="store_true",
                   help="Quick debug run with 1 000 users (skips full training).")
    # ── Resume / checkpoint flags ─────────────────────────────────────────────
    p.add_argument("--skip-candidates",     action="store_true",
                   help="Load Stage-1 candidates from checkpoints/ (skip generation).")
    p.add_argument("--skip-feature-matrix", action="store_true",
                   help="Load feature matrix from checkpoints/ (skip all feature engineering).")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stage1_artifact_paths() -> dict[str, Path]:
    return {
        "covisit": CHECKPOINTS_DIR / "covisit.pkl",
        "w2v_dir": CHECKPOINTS_DIR / "w2v",
    }


def get_temporal_splits(
    trans_train: pl.DataFrame,
    trans_val: pl.DataFrame,
    trans_test: pl.DataFrame,
    final_2025: bool = False
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create proper temporal splits avoiding data leakage.
    
    Returns:
        history_for_artifacts: Data for building Stage-1 artifacts (covisit, W2V)
        history_for_features: Data strictly BEFORE label period (for feature computation)
        label_df: Label period transactions
        label_start_date: Minimum date in label period (cutoff for history)
    """
    if final_2025:
        # Final mode: history = Jan-Nov, labels = Dec
        history_all = pl.concat([trans_train, trans_val])
        label_df = trans_test
        
        # Get the earliest date in label period
        label_start_date = label_df["updated_date"].min()
        log.info(f"Label period starts: {label_start_date}")
        
        # For artifacts (covisit, W2V), use full pre-label history
        history_for_artifacts = history_all.filter(
            pl.col("updated_date") < label_start_date
        )
        
        # For features, also use pre-label only
        history_for_features = history_for_artifacts.clone()
        
    else:
        # Standard mode: history = months 1-10, labels = month 11
        history_all = trans_train
        label_df = trans_val
        
        # Get the earliest date in label period
        label_start_date = label_df["updated_date"].min()
        log.info(f"Label period starts: {label_start_date}")
        
        # For artifacts, use pre-label history
        history_for_artifacts = history_all.filter(
            pl.col("updated_date") < label_start_date
        )
        
        # For features, same as artifacts
        history_for_features = history_for_artifacts.clone()
    
    log.info(f"History for artifacts: {history_for_artifacts.shape[0]:,} transactions")
    log.info(f"History max date: {history_for_artifacts['updated_date'].max()}")
    log.info(f"Label period: {label_df.shape[0]:,} transactions")
    log.info(f"Label date range: {label_df['updated_date'].min()} to {label_df['updated_date'].max()}")
    
    # Verify no overlap
    max_history_date = history_for_artifacts["updated_date"].max()
    min_label_date = label_df["updated_date"].min()
    assert max_history_date < min_label_date, \
        f"DATA LEAKAGE: History max date ({max_history_date}) >= Label min date ({min_label_date})"
    
    return history_for_artifacts, history_for_features, label_df, label_start_date


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    # 0. Load data and create temporal splits
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

    # FIX: Create proper temporal splits to avoid data leakage
    history_artifacts, history_features, label_df, label_start = get_temporal_splits(
        trans_train, trans_val, trans_test, args.final_2025
    )

    if args.final_2025:
        log.info("Final-2025 mode enabled: history=Jan-Oct, labels=Nov (artifact) + Dec (label)")
        if not args.no_eval:
            log.warning(
                "Evaluation in final-2025 mode uses Dec as labels. "
                "No future holdout available in 2025 data."
            )
    else:
        log.info("Standard mode: history=Jan-Oct (pre-label), labels=Nov")

    # ════════════════════════════════════════════════════════════════════════
    # 1. Stage 1 – Build / Load artifacts (using pre-label history only)
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
        # FIX: Use pre-label history for building artifacts
        log.info("Building Stage-1 artifacts from pre-label history only …")
        
        # ── Lưới 2: Covisitation matrix ──────────────────────────────────────
        t0 = time.time()
        covisit = build_covisitation_matrix(
            history_artifacts,  # FIX: pre-label only
            top_k_per_item=COVISIT_TOP_K,
            max_bill_size=COVISIT_MAX_BILL,
        )
        save_covisit(covisit, paths["covisit"])
        
        # Debug: check sparsity
        avg_neighbors = covisit.getnnz(axis=1).mean() if hasattr(covisit, 'getnnz') else 0
        log.info("Covisitation built in %.1f s | avg neighbors per item: %.1f", 
                time.time() - t0, avg_neighbors)

        # ── Lưới 3: Word2Vec ─────────────────────────────────────────────────
        if not args.no_w2v:
            t0 = time.time()
            w2v_model = train_word2vec(
                history_artifacts,  # FIX: pre-label only
                vector_size=W2V_VECTOR_SIZE,
                window=W2V_WINDOW,
                min_count=W2V_MIN_COUNT,
                epochs=W2V_EPOCHS,
                workers=max(1, args.w2v_workers),
            )
            emb_matrix, item_list = build_embedding_matrix(w2v_model)
            save_w2v_artifacts(w2v_model, emb_matrix, item_list, paths["w2v_dir"])
            log.info("Word2Vec built in %.1f s | vocab size: %d", 
                    time.time() - t0, len(item_list) if item_list else 0)
        else:
            w2v_model = emb_matrix = item_list = None
            log.info("Word2Vec skipped (--no-w2v).")

    # ════════════════════════════════════════════════════════════════════════
    # 2. Generate candidates for sampled training users
    #    FIX: Use pre-label history for candidate generation
    # ════════════════════════════════════════════════════════════════════════

    # --- Sample users for ranker training ---
    n_train_users = max(1, args.n_users if not args.debug else 1_000)
    rng = np.random.default_rng(42)
    all_train_users = history_features["customer_id"].unique().to_list()
    n_sample = min(n_train_users, len(all_train_users))
    sampled_users = rng.choice(all_train_users, size=n_sample, replace=False).tolist()
    log.info(
        "Generating Stage-1 candidates for %d training users (batch %d) …%s",
        n_sample, args.stage1_batch_size,
        "  [DEBUG MODE]" if args.debug else "",
    )

    _ckpt_cands   = CHECKPOINTS_DIR / "train_cands.parquet"
    _ckpt_scores  = CHECKPOINTS_DIR / "train_scores.pkl"
    _ckpt_val_pos = CHECKPOINTS_DIR / "train_val_pos.parquet"

    if args.skip_candidates and _ckpt_cands.exists() and _ckpt_scores.exists():
        log.info("[RESUME] Loading candidates from %s …", _ckpt_cands)
        cands_df = pl.read_parquet(_ckpt_cands)
        with open(_ckpt_scores, "rb") as _fh:
            _scores_bundle = pickle.load(_fh)
        cov_scores    = _scores_bundle["cov_scores"]
        w2v_scs       = _scores_bundle["w2v_scs"]
        sampled_users = _scores_bundle["sampled_users"]
        val_pos = pl.read_parquet(_ckpt_val_pos)
        log.info("[RESUME] Loaded %d candidates, %d positive labels, %d users",
                 cands_df.shape[0], val_pos.shape[0], len(sampled_users))
    else:
        # --- Batch candidate generation ---
        batch_size = max(100, args.stage1_batch_size)
        cand_results = {}
        t0 = time.time()
        for i in range(0, n_sample, batch_size):
            batch_users = sampled_users[i:i+batch_size]
            batch_cand = generate_candidates_for_users(
                user_ids     = batch_users,
                trans_df     = history_features,
                covisit      = covisit,
                w2v_model    = w2v_model,
                emb_matrix   = emb_matrix,
                item_list    = item_list,
                allowed_items= active_items,
            )
            cand_results.update(batch_cand)
            if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= n_sample:
                log.info("  ...done %5d / %d users (%.1f%%)",
                         min(i+len(batch_users), n_sample), n_sample,
                         min(i+len(batch_users), n_sample)/n_sample*100)
        log.info("Candidates generated in %.1f s", time.time() - t0)

        cands_df   = candidates_to_dataframe(cand_results)
        cov_scores = {uid: res["covisit_scores"] for uid, res in cand_results.items()}
        w2v_scs    = {uid: res["w2v_scores"]     for uid, res in cand_results.items()}

        val_pos = label_df.filter(
            pl.col("customer_id").is_in(sampled_users)
        ).select(["customer_id", "item_id"])

        # ── Save candidates checkpoint ──────────────────────────────────────
        log.info("Saving candidates checkpoint …")
        cands_df.write_parquet(_ckpt_cands)
        val_pos.write_parquet(_ckpt_val_pos)
        with open(_ckpt_scores, "wb") as _fh:
            pickle.dump({"cov_scores": cov_scores, "w2v_scs": w2v_scs,
                         "sampled_users": sampled_users}, _fh)
        log.info("Candidates checkpoint saved → %s", _ckpt_cands)

    log.info("Positive labels in training: %d", val_pos.shape[0])

    # ════════════════════════════════════════════════════════════════════════
    # 3. Stage 2 – Feature engineering (using pre-label history only)
    # ════════════════════════════════════════════════════════════════════════
    log.info("Precomputing user/item features from pre-label history …")
    t0 = time.time()

    user_feat = build_user_features(history_features)
    item_feat = build_item_features(history_features, items_df)
    log.info("User features: %d users, %d cols | Item features: %d items, %d cols",
             user_feat.shape[0], user_feat.shape[1],
             item_feat.shape[0], item_feat.shape[1])

    # Pre-compute cross-feature stats + optional precomputed extras
    pre_computed = precompute_cross_stats(history_features, items_df)

    # ── Improvement 4: Temporal Decay Features ───────────────────────────────
    temporal_user_feat = temporal_item_feat = None
    if args.use_temporal_features:
        log.info("Building temporal decay features …")
        temporal_user_feat, temporal_item_feat = build_temporal_decay_features(
            history_features, decay_rate=TEMPORAL_DECAY_RATE
        )

    # ── Improvement 6: Session Features ─────────────────────────────────────
    session_feat = None
    if args.use_session_features:
        log.info("Building session features …")
        session_feat = build_session_features(history_features, session_gap_hours=SESSION_GAP_HOURS)

    # ── Improvement 7: Category Affinity ────────────────────────────────────
    if args.use_category_affinity:
        log.info("Building category affinity features …")
        cat_affinity_feat, cross_cat_feat = build_category_affinity_temporal(
            history_features, items_df, decay_rate=CAT_AFFINITY_DECAY_RATE
        )
        if cat_affinity_feat.height > 0:
            user_feat = user_feat.join(cat_affinity_feat, on="customer_id", how="left")
        if cross_cat_feat.height > 0:
            pre_computed["cross_cat_feat"] = cross_cat_feat
            # Also store item_cats for the cross-feature join
            if "item_cats" not in pre_computed:
                pre_computed["item_cats"] = items_df.select(["item_id", "category_l1", "category_l2"])

    # ── Improvement 8: Item Trend Features ──────────────────────────────────
    item_trend_feat = None
    if args.use_item_trends:
        log.info("Building item trend features …")
        item_trend_feat = build_item_trend_features(history_features, windows=ITEM_TREND_WINDOWS)

    # ── Improvement 9: UI History Features ──────────────────────────────────
    if args.use_ui_history:
        log.info("Building user-item history features …")
        ui_hist_feat = build_user_item_history_features(history_features, items_df)
        pre_computed["ui_history_feat"] = ui_hist_feat

    log.info("All feature groups computed in %.1f s", time.time() - t0)

    _ckpt_feat_matrix = CHECKPOINTS_DIR / "train_feature_matrix.parquet"

    if args.skip_feature_matrix and _ckpt_feat_matrix.exists():
        log.info("[RESUME] Loading feature matrix from %s …", _ckpt_feat_matrix)
        t0 = time.time()
        feature_df = pl.read_parquet(_ckpt_feat_matrix)
        log.info("[RESUME] Feature matrix loaded in %.1f s  shape=%s",
                 time.time() - t0, feature_df.shape)
    else:
        log.info("Building training feature matrix …")
        t0 = time.time()
        feature_df = build_feature_matrix(
            candidates_df       = cands_df,
            trans_df            = history_features,
            items_df            = items_df,
            covisit_scores      = cov_scores,
            w2v_scores          = w2v_scs,
            label_df            = val_pos,
            user_feat           = user_feat,
            item_feat           = item_feat,
            pre_computed        = pre_computed,
            temporal_user_feat  = temporal_user_feat,
            temporal_item_feat  = temporal_item_feat,
            session_feat        = session_feat,
            item_trend_feat     = item_trend_feat,
            extend_labels       = args.extend_labels,
            use_hard_negatives  = args.use_hard_negatives,
        )
        log.info("Feature matrix built in %.1f s  shape=%s", time.time() - t0, feature_df.shape)
        # ── Save feature matrix checkpoint ──────────────────────────────────
        log.info("Saving feature matrix checkpoint …")
        feature_df.write_parquet(_ckpt_feat_matrix)
        log.info("Feature matrix checkpoint saved → %s", _ckpt_feat_matrix)

    if "label" in feature_df.columns:
        pos_count = int(feature_df["label"].sum())
        log.info("Positive samples: %d (%.2f%%)", pos_count,
                 pos_count / max(1, feature_df.shape[0]) * 100)

    # Negative sampling to balance training set
    train_data = sample_training_pairs(
        feature_df,
        neg_ratio=args.neg_ratio,
        use_hard_negatives=args.use_hard_negatives,
        hard_neg_fraction=HARD_NEG_FRACTION,
    )
    log.info("Training data after sampling: %d rows", train_data.shape[0])

    # ════════════════════════════════════════════════════════════════════════
    # 4. Train LGBMRanker (single or ensemble)
    # ════════════════════════════════════════════════════════════════════════

    # Columns to exclude from feature matrix during training
    _exclude = ["customer_id", "item_id", "hard_neg_type"]

    # Parse ensemble seeds
    ensemble_seed_list: list[int] = []
    if args.ensemble_seeds:
        try:
            ensemble_seed_list = [int(s.strip()) for s in args.ensemble_seeds.split(",") if s.strip()]
        except ValueError:
            log.warning("Could not parse --ensemble-seeds %r – falling back to single model",
                        args.ensemble_seeds)

    if not ensemble_seed_list:
        # ── Single model ─────────────────────────────────────────────────────
        ranker = LGBMRanker(
            n_estimators      = RANKER_N_ESTIMATORS,
            learning_rate     = RANKER_LEARNING_RATE,
            max_depth         = RANKER_MAX_DEPTH,
            num_leaves        = RANKER_NUM_LEAVES,
            subsample         = RANKER_SUBSAMPLE,
            colsample         = RANKER_COLSAMPLE,
            objective         = args.ranker_type,
            min_child_samples = RANKER_MIN_CHILD_SAMPLES,
        )
        log.info("Training LGBMRanker (single model) …")
        t0 = time.time()
        ranker.fit(train_data, exclude_cols=_exclude)
        log.info("Ranker trained in %.1f s", time.time() - t0)

        ranker.save(CHECKPOINTS_DIR / "lgbm_ranker.pkl")
        _ensemble_rankers: list[LGBMRanker] = [ranker]

    else:
        # ── Ensemble: train one model per seed ───────────────────────────────
        log.info("Training ensemble with %d seeds: %s", len(ensemble_seed_list), ensemble_seed_list)
        _ensemble_rankers = []
        for seed in ensemble_seed_list:
            r = LGBMRanker(
                n_estimators      = RANKER_N_ESTIMATORS,
                learning_rate     = RANKER_LEARNING_RATE,
                max_depth         = RANKER_MAX_DEPTH,
                num_leaves        = RANKER_NUM_LEAVES,
                subsample         = RANKER_SUBSAMPLE,
                colsample         = RANKER_COLSAMPLE,
                objective         = args.ranker_type,
                min_child_samples = RANKER_MIN_CHILD_SAMPLES,
                random_state      = seed,
            )
            t_seed = time.time()
            seed_data = sample_training_pairs(feature_df, neg_ratio=args.neg_ratio, seed=seed,
                                              use_hard_negatives=args.use_hard_negatives,
                                              hard_neg_fraction=HARD_NEG_FRACTION)
            r.fit(seed_data, exclude_cols=_exclude)
            log.info("  seed=%d trained in %.1f s", seed, time.time() - t_seed)
            r.save(CHECKPOINTS_DIR / f"lgbm_ranker_seed{seed}.pkl")
            _ensemble_rankers.append(r)
        # Save primary (first-seed) model as default
        _ensemble_rankers[0].save(CHECKPOINTS_DIR / "lgbm_ranker.pkl")
        ranker = _ensemble_rankers[0]

    with open(CHECKPOINTS_DIR / "items_df.pkl", "wb") as fh:
        pickle.dump(items_df, fh)

    # Save feature list for inference
    feature_cols = [c for c in train_data.columns if c not in _exclude + ["label"]]
    with open(CHECKPOINTS_DIR / "feature_columns.json", "w") as fh:
        json.dump(feature_cols, fh)

    # Feature importance (primary model)
    fi = ranker.feature_importance()
    print("\n── Top-20 feature importances ──────────────────────────────────")
    for row in fi.head(20).iter_rows(named=True):
        print(f"  {row['feature']:40s}  {row['importance']:>6}")
    print("────────────────────────────────────────────────────────────────\n")

    # ════════════════════════════════════════════════════════════════════════
    # 5. Evaluate on validation set (using pre-label history)
    # ════════════════════════════════════════════════════════════════════════
    if not args.no_eval:
        log.info("Evaluating on validation set …")

        sampled_users_set = set(sampled_users)
        val_users = [u for u in label_df["customer_id"].unique().to_list()
                     if u in sampled_users_set]
        if args.eval_max_users > 0:
            val_users = val_users[: args.eval_max_users]

        eval_batch_size = max(100, args.stage1_batch_size)
        n_eval = len(val_users)
        log.info("Evaluating %d validation users …", n_eval)

        # Collect per-model predictions for ensemble
        all_model_preds: list[dict] = [dict() for _ in _ensemble_rankers]

        for i in range(0, n_eval, eval_batch_size):
            batch_users = val_users[i:i + eval_batch_size]

            val_cand_results = generate_candidates_for_users(
                user_ids     = batch_users,
                trans_df     = history_features,
                covisit      = covisit,
                w2v_model    = w2v_model,
                emb_matrix   = emb_matrix,
                item_list    = item_list,
                allowed_items= active_items,
            )
            val_cands_df = candidates_to_dataframe(val_cand_results)
            val_cov_sc   = {uid: r["covisit_scores"] for uid, r in val_cand_results.items()}
            val_w2v_sc   = {uid: r["w2v_scores"]     for uid, r in val_cand_results.items()}

            # Build val feature matrix (no labels needed for inference)
            val_feat_df = build_feature_matrix(
                candidates_df      = val_cands_df,
                trans_df           = history_features,
                items_df           = items_df,
                covisit_scores     = val_cov_sc,
                w2v_scores         = val_w2v_sc,
                user_feat          = user_feat,
                item_feat          = item_feat,
                pre_computed       = pre_computed,
                temporal_user_feat = temporal_user_feat,
                temporal_item_feat = temporal_item_feat,
                session_feat       = session_feat,
                item_trend_feat    = item_trend_feat,
            )

            for mi, r_model in enumerate(_ensemble_rankers):
                all_model_preds[mi].update(r_model.rank(val_feat_df, top_k=args.top_k))

            if (i + eval_batch_size) % (eval_batch_size * 5) == 0:
                log.info("  eval progress: %d / %d users",
                         min(i + len(batch_users), n_eval), n_eval)

        # Fuse predictions (single model = identity, ensemble = RRF/etc.)
        if len(_ensemble_rankers) > 1:
            log.info("Fusing %d model predictions via '%s' …",
                     len(_ensemble_rankers), args.ensemble_method)
            preds = ensemble_rankers(all_model_preds, method=args.ensemble_method, top_k=args.top_k)
        else:
            preds = all_model_preds[0]

        metrics = evaluate(preds, label_df, k=args.top_k)

        print("── Validation metrics ──────────────────────────────────────────")
        pir_keys = ["precision_at_10", "map", "iou", "reciprocal_rank_first_hit",
                    "total_correct_recommendations"]
        print("  [ PIR / competition metrics ]")
        for k in pir_keys:
            if k in metrics:
                v = metrics[k]
                print(f"  {k:35s}: {v:.6f}" if isinstance(v, float) else f"  {k:35s}: {v:,}")
        print("  [ additional ranking metrics ]")
        for name, val in metrics.items():
            if name in pir_keys:
                continue
            print(f"  {name:35s}: {val:.4f}" if isinstance(val, float) else f"  {name:35s}: {val:,}")
        print("────────────────────────────────────────────────────────────────\n")

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_file = OUTPUTS_DIR / "val_metrics_twostage.json"
        with open(metrics_file, "w") as fh:
            json.dump(metrics, fh, indent=2)
        log.info("Saved validation metrics to %s", metrics_file)


if __name__ == "__main__":
    main()