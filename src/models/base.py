"""Abstract base class that every recommender must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import scipy.sparse as sp


class BaseRecommender(ABC):
    """Common interface for all recommendation models."""

    @abstractmethod
    def fit(self, interaction_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Train the model on the (n_users × n_items) interaction matrix."""
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        user_ids:         list[int],
        user2idx:         dict[int, int],
        idx2item:         dict[int, str],
        top_k:            int,
        filter_purchased: sp.csr_matrix | None = None,
    ) -> dict[int, list[str]]:
        """
        Generate top-K item recommendations for each user.

        Parameters
        ----------
        user_ids         : external customer IDs to score
        user2idx         : customer_id → row index in the interaction matrix
        idx2item         : col index   → item_id
        top_k            : number of items to return per user
        filter_purchased : if provided, already-purchased items are excluded

        Returns
        -------
        dict  customer_id → [item_id, …]  (length ≤ top_k)
        """
        raise NotImplementedError
