"""
Stage-2 Reranker – LightGBM binary classifier
==============================================
Trains on labelled (user, item) candidate pairs, scores them, and
returns the top-K items per user.

Label  = 1  if the item was actually purchased in the target period
       = 0  otherwise (negative sample from Stage-1 candidates)
"""
from __future__ import annotations

import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

# Feature columns used for training (filled in dynamically at fit time)
_FEATURE_COLS: list[str] = []


class LGBMRanker:
    """
    LightGBM ranker supporting two objectives:
      - ``lambdarank`` (default): LambdaMART – group-aware, directly optimises
        NDCG.  Significantly better Precision@k / MAP than binary classification.
      - ``binary``: backward-compatible pointwise binary classifier.

    The public interface (fit / predict_proba / rank / save / load) is identical
    for both objectives, so callers do not need to change.
    """

    def __init__(
        self,
        n_estimators:      int   = 1000,
        learning_rate:     float = 0.05,
        max_depth:         int   = 6,
        num_leaves:        int   = 63,
        subsample:         float = 0.8,
        colsample:         float = 0.8,
        random_state:      int   = 42,
        n_jobs:            int   = -1,
        objective:         str   = "lambdarank",
        min_child_samples: int   = 10,
    ) -> None:
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.num_leaves        = num_leaves
        self.subsample         = subsample
        self.colsample         = colsample
        self.random_state      = random_state
        self.n_jobs            = n_jobs
        self.objective         = objective
        self.min_child_samples = min_child_samples
        self._model: Any       = None
        self.feature_cols: list[str] = []

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df:     pl.DataFrame,
        label_col:    str = "label",
        exclude_cols: list[str] | None = None,
    ) -> None:
        """
        Train on a labelled candidate DataFrame.

        Parameters
        ----------
        train_df     : DataFrame with feature columns + label_col
        label_col    : name of the binary label column (0/1)
        exclude_cols : non-feature columns to drop before training
                       (e.g., customer_id, item_id, label)
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required. Install: pip install lightgbm"
            ) from exc

        skip = set(exclude_cols or []) | {label_col}
        self.feature_cols = [c for c in train_df.columns if c not in skip]

        if self.objective == "lambdarank":
            # ── LambdaMART: group-aware listwise ranking ──────────────────────
            # Data must be sorted by customer_id so LightGBM group boundaries
            # align with the X / y arrays.
            train_sorted = train_df.sort("customer_id")

            # Group sizes in the same (sorted) order as train_sorted rows.
            groups: list[int] = (
                train_sorted
                .group_by("customer_id", maintain_order=True)
                .agg(pl.len().alias("_cnt"))
                ["_cnt"]
                .to_list()
            )

            X = train_sorted.select(self.feature_cols).to_numpy().astype(np.float32)
            y = train_sorted[label_col].to_numpy().astype(np.int32)

            log.info(
                "Training LGBMRanker(lambdarank)  |"
                "  samples=%d  positives=%d  features=%d  groups=%d",
                len(y), int(y.sum()), len(self.feature_cols), len(groups),
            )
            log.info(
                "LambdaRanker params: n_est=%d  lr=%.4f  depth=%d  leaves=%d  "
                "sub=%.2f  col=%.2f  min_child=%d",
                self.n_estimators, self.learning_rate, self.max_depth,
                self.num_leaves, self.subsample, self.colsample,
                self.min_child_samples,
            )

            self._model = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                eval_at=[10],   # was ndcg_eval_at – renamed in LightGBM ≥ 4.0
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                subsample=self.subsample,
                colsample_bytree=self.colsample,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            self._model.fit(X, y, group=groups)

        else:
            # ── Binary classification (backward-compat pointwise mode) ────────
            X = train_df.select(self.feature_cols).to_numpy().astype(np.float32)
            y = train_df[label_col].to_numpy().astype(np.int32)

            log.info(
                "Training LGBMRanker(binary)  |"
                "  samples=%d  positives=%d  features=%d",
                len(y), int(y.sum()), len(self.feature_cols),
            )

            self._model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                subsample=self.subsample,
                colsample_bytree=self.colsample,
                objective="binary",
                metric="auc",
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            self._model.fit(X, y)

        log.info("LGBMRanker trained.")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """
        Return a ranking score for each row.

        For ``lambdarank`` mode the score is a raw LambdaMART output (higher =
        more relevant).  For ``binary`` mode it is the positive-class
        probability.  Both are suitable for sorting within a user group.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if df.height == 0:
            return np.array([], dtype=np.float32)
        _t0 = time.perf_counter()
        X = df.select(self.feature_cols).to_numpy().astype(np.float32)
        _t1 = time.perf_counter()
        if X.shape[0] == 0:
            return np.array([], dtype=np.float32)
        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None  # type: ignore
        # Inference-time tree truncation: using fewer trees cuts predict time
        # proportionally with minimal quality loss for well-converged models.
        try:
            from src.config import RANKER_PREDICT_NUM_ITERATION as _N_ITER
        except Exception:
            _N_ITER = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if lgb is not None and isinstance(self._model, lgb.LGBMRanker):
                result = self._model.booster_.predict(
                    X, num_threads=-1, num_iteration=_N_ITER
                ).astype(np.float32)
            else:
                result = self._model.predict_proba(X)[:, 1].astype(np.float32)
        _t2 = time.perf_counter()
        if _t2 - _t0 > 1.0:   # only log when it matters
            log.info(
                "  [ranker] to_numpy=%.1fs  predict=%.1fs  rows=%d  cols=%d",
                _t1 - _t0, _t2 - _t1, X.shape[0], X.shape[1],
            )
        return result

    def rank(
        self,
        candidates_df: pl.DataFrame,
        top_k:         int = 10,
    ) -> dict[int, list[str]]:
        """
        Score all (user, item) pairs and return top-K items per user.

        Parameters
        ----------
        candidates_df : DataFrame with columns [customer_id, item_id, *features]
        top_k         : items to keep per user

        Returns
        -------
        dict  customer_id → [item_id, ...]  (length ≤ top_k)
        """
        if candidates_df.height == 0:
            return {}

        scores = self.predict_proba(candidates_df)
        scored = candidates_df.select(["customer_id", "item_id"]).with_columns(
            pl.Series("score", scores, dtype=pl.Float32)
        )
        # Sort by score desc, take top_k per user
        ranked = (
            scored
            .sort("score", descending=True)
            .group_by("customer_id")
            .agg(pl.col("item_id").head(top_k).alias("items"))
        )
        return {row["customer_id"]: row["items"] for row in ranked.iter_rows(named=True)}

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self) -> pl.DataFrame:
        """Return a DataFrame of feature importances sorted descending."""
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        imp = self._model.feature_importances_
        return (
            pl.DataFrame({"feature": self.feature_cols, "importance": imp.tolist()})
            .sort("importance", descending=True)
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        log.info("LGBMRanker saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "LGBMRanker":
        with open(path, "rb") as fh:
            return pickle.load(fh)
