"""Popularity-based recommender: recommend globally most-purchased items."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.models.base import BaseRecommender

log = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):
    """
    Recommend the globally top-K items by total interaction weight,
    filtering out items the user has already purchased.
    """

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self._popular_items: list[str] = []   # ordered by descending popularity

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        interaction_matrix: sp.csr_matrix,
        idx2item: dict[int, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Rank items by column-sum (total interaction weight).

        Parameters
        ----------
        interaction_matrix : (n_users, n_items) CSR matrix
        idx2item           : col_index → item_id mapping
        """
        col_sums    = np.asarray(interaction_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(col_sums)[::-1]

        if idx2item:
            self._popular_items = [idx2item[i] for i in top_indices if i in idx2item]
        else:
            self._popular_items = [str(i) for i in top_indices]

        log.info(
            "PopularityRecommender fitted — top item: %s (%.0f interactions)",
            self._popular_items[0], col_sums[top_indices[0]],
        )

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_ids:         list[int],
        user2idx:         dict[int, int],
        idx2item:         dict[int, str],
        top_k:            int | None = None,
        filter_purchased: sp.csr_matrix | None = None,
    ) -> dict[int, list[str]]:
        k = top_k or self.top_k
        result: dict[int, list[str]] = {}

        for uid in user_ids:
            uidx = user2idx.get(uid)
            if uidx is not None and filter_purchased is not None:
                already = {idx2item[i] for i in filter_purchased[uidx].indices if i in idx2item}
            else:
                already = set()

            result[uid] = [it for it in self._popular_items if it not in already][:k]

        return result
