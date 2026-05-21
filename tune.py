"""
Hyperparameter Tuning for the Two-Stage Recommendation System (FIXED VERSION)
==============================================================================
Fixed data leakage: features and candidates computed from pre-label history only.
Added early stopping via n_estimators tuning, better search spaces, and trial pruning.

Uses Optuna to find the best LGBMRanker configuration, optimising
Precision@10 on the validation split.

Usage
-----
python tune.py                              # 30 trials, 30k train users
python tune.py --n-trials 50 --n-users 50000
python tune.py --study-name my_study --storage sqlite:///outputs/tune.db
python tune.py --skip-stage1               # reuse existing Stage-1 artifacts
python tune.py --tune-neg-ratio            # also tune neg_ratio
python tune.py --apply                     # save best params → outputs/best_params.json
python tune.py --prune                     # enable MedianPruner for faster tuning
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl

try:
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as exc:
    raise ImportError(
        "Optuna is required for tuning.\n"
        "Install: uv add optuna   or   pip install optuna"
    ) from exc

from src.config import (
    CHECKPOINTS_DIR, OUTPUTS_DIR,
    STAGE1_BATCH_SIZE,
    COVISIT_TOP_K, COVISIT_MAX_BILL,
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS, W2V_WORKERS,
    RANKER_TRAIN_USERS, RANKER_NEG_RATIO,
    RANKER_N_ESTIMATORS, RANKER_LEARNING_RATE, RANKER_MAX_DEPTH,
    RANKER_NUM_LEAVES, RANKER_SUBSAMPLE, RANKER_COLSAMPLE,
    RANKER_OBJECTIVE, RANKER_MIN_CHILD_SAMPLES, RANKER_TOP_K_OUTPUT,
    USE_EXTEND_LABELS, USE_HARD_NEGATIVES, HARD_NEG_FRACTION,
)
from src.data.loader import load_transactions, load_items, split_transactions
from src.candidates.covisitation import (
    build_covisitation_matrix, load_covisit, save_covisit, build_covisit_sparse,
)
from src.candidates.word2vec_cands import (
    train_word2vec, build_embedding_matrix,
    save_w2v_artifacts, load_w2v_artifacts,
)
from src.ranker.lgbm_ranker import LGBMRanker
from src.evaluation.metrics import evaluate
from src.features.user_features import build_user_features
from src.features.item_features import build_item_features
from src.features.cross_features import precompute_cross_stats
from pipeline.stage1_candidates import (
    generate_candidates_for_users_fast, candidates_to_dataframe,
)
from pipeline.stage2_reranking import build_feature_matrix, sample_training_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tune")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna for the two-stage ranker."
    )
    p.add_argument("--n-trials",      type=int, default=30,
                   help="Number of Optuna trials (default: 30).")
    p.add_argument("--n-users",       type=int, default=30_000,
                   help="Training users sampled for candidate gen (default: 30k).")
    p.add_argument("--n-eval-users",  type=int, default=10_000,
                   help="Val users used to compute Precision@10 per trial (default: 10k).")
    p.add_argument("--stage1-batch-size", type=int, default=STAGE1_BATCH_SIZE,
                   help="Users per Stage-1 generation batch.")
    p.add_argument("--top-k",         type=int, default=RANKER_TOP_K_OUTPUT)
    p.add_argument("--study-name",    type=str, default="twostage_ranker_v2",
                   help="Optuna study name (default: twostage_ranker_v2).")
    p.add_argument("--storage",       type=str, default="",
                   help="Optuna storage URL (e.g. sqlite:///outputs/tune.db). "
                        "Empty = in-memory (results lost on exit).")
    p.add_argument("--skip-stage1",   action="store_true",
                   help="Load Stage-1 artifacts from disk instead of rebuilding.")
    p.add_argument("--no-w2v",        action="store_true",
                   help="Skip Word2Vec during Stage-1 build.")
    p.add_argument("--w2v-workers",   type=int, default=W2V_WORKERS)
    p.add_argument("--tune-neg-ratio", action="store_true",
                   help="Also tune the neg_ratio training-data balance parameter.")
    p.add_argument("--prune",         action="store_true",
                   help="Enable MedianPruner to stop unpromising trials early.")
    p.add_argument("--apply",         action="store_true",
                   help="Save best params to outputs/best_params.json after tuning.")
    p.add_argument("--extend-labels",      action="store_true", default=USE_EXTEND_LABELS,
                   help="Use grade-0/1/2 soft labels during tuning.")
    p.add_argument("--use-hard-negatives", action="store_true", default=USE_HARD_NEGATIVES,
                   help="Include hard negatives in training pairs during tuning.")
    return p.parse_args()


# ── Temporal split helper (FIX: no data leakage) ──────────────────────────────

def get_temporal_splits(
    trans_train: pl.DataFrame,
    trans_val: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    FIX: Create proper temporal splits for tuning.
    
    Returns:
        history_artifacts: Data BEFORE label period (for building covisit, W2V)
        history_features: Same as artifacts (for feature computation)
        label_df: Validation period transactions (month 11)
    """
    label_df = trans_val
    
    # Get the earliest date in label period
    label_start_date = label_df["updated_date"].min()
    log.info(f"Label period starts: {label_start_date}")
    
    # Use only pre-label history for artifacts and features
    history_for_artifacts = trans_train.filter(
        pl.col("updated_date") < label_start_date
    )
    
    log.info(f"Pre-label history: {history_for_artifacts.shape[0]:,} transactions")
    log.info(f"History date range: {history_for_artifacts['updated_date'].min()} to "
             f"{history_for_artifacts['updated_date'].max()}")
    log.info(f"Label date range: {label_df['updated_date'].min()} to "
             f"{label_df['updated_date'].max()}")
    
    # Verify no temporal overlap
    max_history_date = history_for_artifacts["updated_date"].max()
    min_label_date = label_df["updated_date"].min()
    if max_history_date >= min_label_date:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: History max date ({max_history_date}) >= "
            f"Label min date ({min_label_date}). "
            f"Check your data split - transactions may span multiple months in same file."
        )
    
    log.info("✓ Temporal split verified - no data leakage")
    
    return history_for_artifacts, history_for_artifacts, label_df


# ── Stage-1 artifact helpers ──────────────────────────────────────────────────

def _build_or_load_stage1(
    args:        argparse.Namespace,
    history_df:  pl.DataFrame,  # FIX: pre-label history
    items_df:    pl.DataFrame,
) -> tuple:
    """Return (covisit, w2v_model, emb_matrix, item_list, active_items)."""
    active_items = set(
        items_df.filter(pl.col("sale_status") == 1)["item_id"].cast(pl.Utf8).to_list()
    )
    paths = {
        "covisit": CHECKPOINTS_DIR / "covisit.pkl",
        "w2v_dir": CHECKPOINTS_DIR / "w2v",
    }

    if args.skip_stage1:
        log.info("Loading Stage-1 artifacts from disk …")
        covisit = load_covisit(paths["covisit"])
        if not args.no_w2v and (paths["w2v_dir"] / "w2v.model").exists():
            w2v_model, emb_matrix, item_list = load_w2v_artifacts(paths["w2v_dir"])
        else:
            w2v_model = emb_matrix = item_list = None
    else:
        log.info("Building covisitation matrix from pre-label history …")
        t0 = time.time()
        covisit = build_covisitation_matrix(
            history_df,  # FIX: pre-label only
            top_k_per_item=COVISIT_TOP_K,
            max_bill_size=COVISIT_MAX_BILL,
        )
        save_covisit(covisit, paths["covisit"])
        
        # Debug info
        if hasattr(covisit, 'getnnz'):
            avg_neighbors = covisit.getnnz(axis=1).mean()
            log.info(f"Covisitation built in {time.time() - t0:.1f}s | "
                    f"avg neighbors per item: {avg_neighbors:.1f}")

        if not args.no_w2v:
            log.info("Training Word2Vec from pre-label history …")
            t0 = time.time()
            w2v_model = train_word2vec(
                history_df,  # FIX: pre-label only
                vector_size=W2V_VECTOR_SIZE,
                window=W2V_WINDOW,
                min_count=W2V_MIN_COUNT,
                epochs=W2V_EPOCHS,
                workers=args.w2v_workers,
            )
            emb_matrix, item_list = build_embedding_matrix(w2v_model)
            save_w2v_artifacts(w2v_model, emb_matrix, item_list, paths["w2v_dir"])
            log.info(f"Word2Vec built in {time.time() - t0:.1f}s | "
                    f"vocab size: {len(item_list) if item_list else 0}")
        else:
            w2v_model = emb_matrix = item_list = None

    return covisit, w2v_model, emb_matrix, item_list, active_items


def _generate_candidates_batched(
    users:                  list[int],
    trans_df:               pl.DataFrame,
    covisit:                dict,
    w2v_model,
    emb_matrix,
    item_list,
    active_items:           set[str],
    batch_size:             int,
    covisit_sparse_data:    tuple,
    item_to_emb_idx:        dict | None,
) -> dict:
    """Run Stage-1 in batches and return merged candidate dict."""
    results: dict = {}
    n = len(users)
    for i in range(0, n, batch_size):
        batch = users[i: i + batch_size]
        batch_cands = generate_candidates_for_users_fast(
            user_ids=batch,
            trans_df=trans_df,
            covisit=covisit,
            w2v_model=w2v_model,
            emb_matrix=emb_matrix,
            item_list=item_list,
            allowed_items=active_items,
            _prebuilt_covisit_sparse=covisit_sparse_data,
            _item_to_emb_idx=item_to_emb_idx,
        )
        results.update(batch_cands)
        if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= n:
            log.info(
                "  Stage-1 candidates: %d / %d  (%.1f%%)",
                min(i + batch_size, n), n,
                min(i + batch_size, n) / n * 100,
            )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading transaction data …")
    trans_train, trans_val, _ = split_transactions(load_transactions())
    items_df = load_items()

    # FIX: Create proper temporal splits
    history_artifacts, history_features, label_df = get_temporal_splits(
        trans_train, trans_val
    )

    # ── Build / load Stage-1 artifacts (FIX: pre-label history) ──────────────
    covisit, w2v_model, emb_matrix, item_list, active_items = _build_or_load_stage1(
        args, history_artifacts, items_df
    )

    # Pre-build structures used across all trials
    log.info("Pre-building covisit sparse matrix …")
    covisit_sparse_data = build_covisit_sparse(covisit)
    item_to_emb_idx = (
        {item: i for i, item in enumerate(item_list)} if item_list else None
    )

    # FIX: Precompute features from pre-label history
    log.info("Precomputing user / item features from pre-label history …")
    user_feat = build_user_features(history_features)
    item_feat = build_item_features(history_features, items_df)
    log.info(f"User features: {user_feat.shape[0]:,} users, {user_feat.shape[1]} features")
    log.info(f"Item features: {item_feat.shape[0]:,} items, {item_feat.shape[1]} features")

    log.info("Precomputing cross-feature stats …")
    pre_computed = precompute_cross_stats(history_features, items_df)

    # ── Sample fixed sets of train & eval users ───────────────────────────────
    rng = np.random.default_rng(42)

    # FIX: Sample from pre-label history users only
    all_train_users = history_features["customer_id"].unique().to_list()
    n_train = min(args.n_users, len(all_train_users))
    train_users = rng.choice(all_train_users, size=n_train, replace=False).tolist()
    log.info(f"Sampled {n_train:,} training users from {len(all_train_users):,} available")

    # FIX: Eval users from validation period (label_df)
    all_val_users = label_df["customer_id"].unique().to_list()
    n_eval = min(args.n_eval_users, len(all_val_users))
    eval_users = rng.choice(all_val_users, size=n_eval, replace=False).tolist()
    log.info(f"Sampled {n_eval:,} eval users from {len(all_val_users):,} available")

    # ── Stage-1 candidates for training users (once) ──────────────────────────
    log.info("Generating Stage-1 candidates for %d training users …", n_train)
    t0 = time.time()
    train_cand_results = _generate_candidates_batched(
        users=train_users,
        trans_df=history_features,  # FIX: pre-label history
        covisit=covisit,
        w2v_model=w2v_model,
        emb_matrix=emb_matrix,
        item_list=item_list,
        active_items=active_items,
        batch_size=args.stage1_batch_size,
        covisit_sparse_data=covisit_sparse_data,
        item_to_emb_idx=item_to_emb_idx,
    )
    log.info("Train Stage-1 done in %.1f s", time.time() - t0)

    train_cands_df  = candidates_to_dataframe(train_cand_results)
    train_cov_sc    = {uid: r["covisit_scores"] for uid, r in train_cand_results.items()}
    train_w2v_sc    = {uid: r["w2v_scores"]     for uid, r in train_cand_results.items()}

    # Ground-truth labels (purchased in label period)
    val_pos = label_df.filter(
        pl.col("customer_id").is_in(train_users)
    ).select(["customer_id", "item_id"])
    log.info(f"Positive labels in training: {val_pos.shape[0]:,}")

    log.info("Building train feature matrix …")
    t0 = time.time()
    train_feat_df = build_feature_matrix(
        candidates_df=train_cands_df,
        trans_df=history_features,  # FIX: pre-label history
        items_df=items_df,
        covisit_scores=train_cov_sc,
        w2v_scores=train_w2v_sc,
        label_df=val_pos,
        user_feat=user_feat,
        item_feat=item_feat,
        pre_computed=pre_computed,
        extend_labels=args.extend_labels,
        use_hard_negatives=args.use_hard_negatives,
    )
    log.info("Train feature matrix: %s  (%.1f s)", train_feat_df.shape, time.time() - t0)
    
    # Debug class balance
    if 'label' in train_feat_df.columns:
        pos_ratio = train_feat_df['label'].mean()
        log.info(f"Positive ratio in train features: {pos_ratio:.4f} ({pos_ratio*100:.2f}%)")

    # ── Stage-1 candidates for eval users (once) ──────────────────────────────
    log.info("Generating Stage-1 candidates for %d eval users …", n_eval)
    t0 = time.time()
    eval_cand_results = _generate_candidates_batched(
        users=eval_users,
        trans_df=history_features,  # FIX: pre-label history
        covisit=covisit,
        w2v_model=w2v_model,
        emb_matrix=emb_matrix,
        item_list=item_list,
        active_items=active_items,
        batch_size=args.stage1_batch_size,
        covisit_sparse_data=covisit_sparse_data,
        item_to_emb_idx=item_to_emb_idx,
    )
    log.info("Eval Stage-1 done in %.1f s", time.time() - t0)

    eval_cands_df = candidates_to_dataframe(eval_cand_results)
    eval_cov_sc   = {uid: r["covisit_scores"] for uid, r in eval_cand_results.items()}
    eval_w2v_sc   = {uid: r["w2v_scores"]     for uid, r in eval_cand_results.items()}

    log.info("Building eval feature matrix …")
    t0 = time.time()
    eval_feat_df = build_feature_matrix(
        candidates_df=eval_cands_df,
        trans_df=history_features,  # FIX: pre-label history
        items_df=items_df,
        covisit_scores=eval_cov_sc,
        w2v_scores=eval_w2v_sc,
        user_feat=user_feat,
        item_feat=item_feat,
        pre_computed=pre_computed,
    )
    log.info("Eval feature matrix: %s  (%.1f s)", eval_feat_df.shape, time.time() - t0)

    # FIX: Evaluate against label_df (validation period)
    eval_label_df = label_df.filter(pl.col("customer_id").is_in(eval_users))
    log.info(f"Eval label transactions: {eval_label_df.shape[0]:,}")

    # ── Cache feature columns ─────────────────────────────────────────────────
    feature_cols = [c for c in train_feat_df.columns 
                    if c not in ["customer_id", "item_id", "label"]]
    log.info(f"Feature columns: {len(feature_cols)}")

    # ── Determine LGBMRanker supported params ─────────────────────────────────
    # Inspect the wrapper to see what params it accepts
    import inspect
    ranker_params = inspect.signature(LGBMRanker.__init__).parameters
    supported_params = set(ranker_params.keys()) - {'self', 'args', 'kwargs'}
    log.info(f"LGBMRanker supported params: {sorted(supported_params)}")
    
    # Check if objective is included
    has_objective = 'objective' in supported_params
    has_early_stopping = 'early_stopping_rounds' in supported_params

    # ── Optuna objective (only ranker changes per trial) ──────────────────────
    def objective(trial: optuna.Trial) -> float:
        # Build ranker kwargs based on what the wrapper supports
        ranker_kwargs = {}
        
        # Core params (should always be supported)
        ranker_kwargs['n_estimators'] = trial.suggest_int("n_estimators", 300, 2000, step=100)
        ranker_kwargs['learning_rate'] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        ranker_kwargs['max_depth'] = trial.suggest_int("max_depth", 4, 12)
        
        # Constrain num_leaves based on max_depth
        max_possible_leaves = min(255, 2 ** ranker_kwargs['max_depth'])
        ranker_kwargs['num_leaves'] = trial.suggest_int(
            "num_leaves", 15, max_possible_leaves, log=True
        )
        
        ranker_kwargs['subsample'] = trial.suggest_float("subsample", 0.6, 1.0)
        ranker_kwargs['colsample'] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        ranker_kwargs['min_child_samples'] = trial.suggest_int(
            "min_child_samples", 5, 50, log=True
        )
        
        # Only add objective if wrapper supports it
        if has_objective:
            ranker_kwargs['objective'] = RANKER_OBJECTIVE
        
        # Only add early_stopping_rounds if wrapper supports it
        if has_early_stopping:
            ranker_kwargs['early_stopping_rounds'] = 50
        
        # Neg ratio (separate from ranker)
        neg_ratio = (
            trial.suggest_int("neg_ratio", 5, 50)
            if args.tune_neg_ratio else RANKER_NEG_RATIO
        )

        # Sample training pairs with current neg_ratio
        train_data = sample_training_pairs(
            train_feat_df,
            neg_ratio=neg_ratio,
            use_hard_negatives=args.use_hard_negatives,
            hard_neg_fraction=HARD_NEG_FRACTION,
        )

        # Create and train ranker
        ranker = LGBMRanker(**ranker_kwargs)
        ranker.fit(train_data, exclude_cols=["customer_id", "item_id", "hard_neg_type"])

        # Predict and evaluate
        preds = ranker.rank(eval_feat_df, top_k=args.top_k)
        metrics = evaluate(preds, eval_label_df, k=args.top_k)
        precision = float(metrics.get("precision_at_10", 0.0))

        # Report for pruning
        trial.report(precision, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Build log message
        log_parts = [
            f"n_est={ranker_kwargs['n_estimators']}",
            f"lr={ranker_kwargs['learning_rate']:.4f}",
            f"depth={ranker_kwargs['max_depth']}",
            f"leaves={ranker_kwargs['num_leaves']}",
            f"sub={ranker_kwargs['subsample']:.2f}",
            f"col={ranker_kwargs['colsample']:.2f}",
            f"min_child={ranker_kwargs['min_child_samples']}",
            f"neg={neg_ratio}",
            f"P@10={precision:.6f}",
        ]
        log.info("Trial %3d  |  %s", trial.number, "  ".join(log_parts))
        
        return precision

    # ── Create / resume Optuna study ──────────────────────────────────────────
    storage = args.storage if args.storage else None
    
    # Add pruner if requested
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
        interval_steps=1,
    ) if args.prune else None
    
    # Note: multivariate might not work well with pruning
    sampler_kwargs = {
        "seed": 42,
        "n_startup_trials": 10,
    }
    
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(**sampler_kwargs),
        pruner=pruner,
    )

    log.info("Starting Optuna tuning: %d trials …", args.n_trials)
    if args.prune:
        log.info("MedianPruner enabled - unpromising trials will be stopped early")
    
    t0 = time.time()
    try:
        study.optimize(
            objective, 
            n_trials=args.n_trials, 
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        log.warning("Tuning interrupted by user. Saving partial results...")
    
    total_time = time.time() - t0
    log.info(f"Tuning completed in {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Print results ─────────────────────────────────────────────────────────
    if len(study.trials) == 0:
        log.error("No trials completed. Exiting.")
        return
    
    # Filter completed trials
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    
    if len(completed) == 0:
        log.error("All trials failed. Check errors above.")
        return
    
    best = study.best_trial
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  Best trial #{best.number}  |  Precision@10 = {best.value:.6f}")
    print(sep)
    print("  Best hyperparameters:")
    for k, v in sorted(best.params.items()):
        if isinstance(v, float):
            print(f"    {k:28s} = {v:.6f}")
        else:
            print(f"    {k:28s} = {v}")
    print(sep)

    # Trial statistics
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    print(f"\n  Trial statistics:")
    print(f"    Completed: {len(completed)}")
    print(f"    Pruned:    {len(pruned)}")
    print(f"    Failed:    {len(failed)}")
    
    if completed:
        values = [t.value for t in completed]
        print(f"    Mean P@10: {np.mean(values):.6f} ± {np.std(values):.6f}")
        print(f"    Min P@10:  {np.min(values):.6f}")
        print(f"    Max P@10:  {np.max(values):.6f}")

    # Parameter importance (only if enough completed trials)
    if len(completed) >= 10:
        try:
            importances = optuna.importance.get_param_importances(study)
            if importances:
                print(f"\n  Parameter importances:")
                for param, importance in sorted(importances.items(), 
                                               key=lambda x: x[1], reverse=True):
                    print(f"    {param:28s}: {importance:.4f}")
        except Exception as e:
            log.debug(f"Could not compute importances: {e}")

    # ── Config update suggestions ─────────────────────────────────────────────
    p = best.params
    print(f"\n  Suggested updates for src/config.py:")
    print(f"    RANKER_N_ESTIMATORS        = {p.get('n_estimators', RANKER_N_ESTIMATORS)}")
    print(f"    RANKER_LEARNING_RATE       = {p.get('learning_rate', RANKER_LEARNING_RATE):.6f}")
    print(f"    RANKER_MAX_DEPTH           = {p.get('max_depth', RANKER_MAX_DEPTH)}")
    print(f"    RANKER_NUM_LEAVES          = {p.get('num_leaves', RANKER_NUM_LEAVES)}")
    print(f"    RANKER_SUBSAMPLE           = {p.get('subsample', RANKER_SUBSAMPLE):.4f}")
    print(f"    RANKER_COLSAMPLE           = {p.get('colsample_bytree', RANKER_COLSAMPLE):.4f}")
    print(f"    RANKER_MIN_CHILD_SAMPLES   = {p.get('min_child_samples', RANKER_MIN_CHILD_SAMPLES)}")
    if args.tune_neg_ratio:
        print(f"    RANKER_NEG_RATIO           = {p.get('neg_ratio', RANKER_NEG_RATIO)}")
    print()

    # ── Optionally save best params ───────────────────────────────────────────
    if args.apply:
        best_params_path = OUTPUTS_DIR / "best_params.json"
        payload = {
            "best_precision_at_10": best.value,
            "params": best.params,
            "study_name": args.study_name,
            "n_trials_completed": len(completed),
            "n_trials_pruned": len(pruned),
            "n_trials_failed": len(failed),
            "total_time_seconds": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ranker_params_supported": sorted(supported_params),
        }
        with open(best_params_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        log.info("Best params saved → %s", best_params_path)

        # Print top-5 trials
        completed_sorted = sorted(completed, key=lambda t: t.value, reverse=True)
        print("  Top-5 trials:")
        print(f"  {'Rank':<6} {'Trial':<8} {'P@10':<12} {'Key Params'}")
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*50}")
        for rank, t in enumerate(completed_sorted[:5], 1):
            key_params = []
            for k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves']:
                if k in t.params:
                    key_params.append(f"{k}={t.params[k]}")
            params_str = ", ".join(key_params)
            print(f"  {rank:<6} #{t.number:<7} {t.value:<12.6f} {params_str}")
        print()


if __name__ == "__main__":
    main()