"""
ALS collaborative filtering recommender backed by the ``implicit`` library.

Install dependency
------------------
pip install implicit
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.models.base import BaseRecommender

log = logging.getLogger(__name__)

_BATCH_SIZE = 10_000   # users per batch during recommendation


class ALSRecommender(BaseRecommender):
    """
    Alternating Least Squares with implicit feedback.

    The interaction matrix values are treated as raw counts / weights; the
    model internally scales them to confidence values:
        C_ui = 1 + alpha * r_ui
    """

    def __init__(
        self,
        factors:        int   = 128,
        iterations:     int   = 20,
        regularization: float = 0.01,
        alpha:          float = 40.0,
        random_state:   int   = 42,
    ) -> None:
        self.factors        = factors
        self.iterations     = iterations
        self.regularization = regularization
        self.alpha          = alpha
        self.random_state   = random_state
        self._model: Any    = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        interaction_matrix: sp.csr_matrix,
        **kwargs: Any,
    ) -> None:
        """
        Fit the ALS model.

        Parameters
        ----------
        interaction_matrix : (n_users, n_items) float32 CSR matrix
            Raw interaction weights (purchase quantities, event weights, …).
        """
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise ImportError(
                "The 'implicit' library is required. "
                "Install it with:  pip install implicit"
            ) from exc

        confidence = (interaction_matrix * self.alpha).astype(np.float32)

        self._model = AlternatingLeastSquares(
            factors        = self.factors,
            iterations     = self.iterations,
            regularization = self.regularization,
            random_state   = self.random_state,
        )

        log.info(
            "Fitting ALS — factors=%d  iterations=%d  alpha=%.1f  "
            "matrix=%s  nnz=%d",
            self.factors, self.iterations, self.alpha,
            confidence.shape, confidence.nnz,
        )
        self._model.fit(confidence)
        log.info("ALS model fitted.")

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_ids:         list[int],
        user2idx:         dict[int, int],
        idx2item:         dict[int, str],
        top_k:            int = 10,
        filter_purchased: sp.csr_matrix | None = None,
    ) -> dict[int, list[str]]:
        """
        Generate top-K recommendations.  Users absent from *user2idx* are
        skipped (they were not seen during training).
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Map external IDs → matrix indices
        valid: list[tuple[int, int]] = [
            (uid, user2idx[uid]) for uid in user_ids if uid in user2idx
        ]
        if not valid:
            return {}

        user_items = filter_purchased if filter_purchased is not None else sp.csr_matrix(
            (len(user2idx), len(idx2item)), dtype=np.float32
        )
        filter_flag = filter_purchased is not None

        result: dict[int, list[str]] = {}

        # Process in batches to cap peak memory usage
        for batch_start in range(0, len(valid), _BATCH_SIZE):
            batch = valid[batch_start: batch_start + _BATCH_SIZE]
            uid_ext_arr  = [uid  for uid, _    in batch]
            uidx_arr     = np.array([uidx for _,    uidx in batch], dtype=np.int32)

            ids_2d, _ = self._model.recommend(
                uidx_arr,
                user_items[uidx_arr],
                N=top_k,
                filter_already_liked_items=filter_flag,
            )
            # ids_2d shape: (batch_size, top_k)
            for k_pos, uid in enumerate(uid_ext_arr):
                result[uid] = [
                    idx2item[i] for i in ids_2d[k_pos] if i in idx2item
                ]

        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        log.info("ALSRecommender saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "ALSRecommender":
        with open(path, "rb") as fh:
            return pickle.load(fh)
