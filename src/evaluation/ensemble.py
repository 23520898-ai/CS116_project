"""
Ensemble utilities for combining predictions from multiple ranker models.
=========================================================================
Supports three fusion methods:

  reciprocal_rank  (default / recommended)
      score_i = Σ  w_j / rank(i, model_j)
      Classic Reciprocal Rank Fusion (RRF).  Rank-normalised, robust to
      score-scale differences between models.

  borda_count
      score_i = Σ  w_j * (n_j - rank(i, model_j) + 1)
      Linear rank weighting.  Sensitive to list length differences.

  weighted_vote
      score_i = Σ  w_j * I(item_i in top_k of model_j)
      Simple ballot: items that appear in multiple top-k lists win.

Why ensembling works
---------------------
Each model trained with a different random seed explores a slightly different
part of the loss landscape.  Combining them via rank-fusion averages out
individual model variance without requiring access to raw scores or logits.
The expected gain over a single model is typically +1 to +3 percentage points
on Precision@10 for this type of problem.

Usage
-----
from src.evaluation.ensemble import ensemble_rankers

# Three separate model predictions
preds_list = [preds_seed42, preds_seed123, preds_seed456]
final = ensemble_rankers(preds_list, method="reciprocal_rank", top_k=10)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def ensemble_rankers(
    predictions_list: list[dict[int, list[str]]],
    weights: list[float] | None = None,
    method: str = "reciprocal_rank",
    top_k: int = 10,
    rrf_k: int = 60,
) -> dict[int, list[str]]:
    """
    Fuse multiple per-user ranked prediction lists into a single ranked list.

    Parameters
    ----------
    predictions_list : list of  {customer_id: [item_id, ...]}  dicts
    weights          : per-model importance weights (default: uniform)
    method           : one of "reciprocal_rank", "borda_count", "weighted_vote"
    top_k            : items to return per user
    rrf_k            : RRF constant  (reciprocal_rank only).  Larger values
                       dampen the advantage of rank-1 items.  RRF default = 60.

    Returns
    -------
    dict  customer_id → [item_id, ...]  (up to top_k items)
    """
    if not predictions_list:
        return {}

    n_models = len(predictions_list)
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        total = sum(weights)
        weights = [w / (total + 1e-12) for w in weights]

    all_users: set[int] = set()
    for preds in predictions_list:
        all_users.update(preds.keys())

    result: dict[int, list[str]] = {}

    for user_id in all_users:
        item_scores: dict[str, float] = {}

        for model_idx, preds in enumerate(predictions_list):
            w = weights[model_idx]
            user_items = preds.get(user_id, [])
            n = len(user_items)

            for rank, item_id in enumerate(user_items, start=1):
                if item_id not in item_scores:
                    item_scores[item_id] = 0.0

                if method == "reciprocal_rank":
                    item_scores[item_id] += w / (rrf_k + rank)
                elif method == "borda_count":
                    item_scores[item_id] += w * (n - rank + 1)
                elif method == "weighted_vote":
                    if rank <= top_k:
                        item_scores[item_id] += w
                else:
                    raise ValueError(
                        f"Unknown ensemble method {method!r}. "
                        "Choose from: reciprocal_rank, borda_count, weighted_vote."
                    )

        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        result[user_id] = [iid for iid, _ in sorted_items[:top_k]]

    log.info(
        "Ensemble (%s, %d models, top_k=%d): %d users",
        method, n_models, top_k, len(result),
    )
    return result


def blend_predictions(
    base_preds: dict[int, list[str]],
    boost_preds: dict[int, list[str]],
    boost_weight: float = 0.3,
    top_k: int = 10,
) -> dict[int, list[str]]:
    """
    Lightweight two-model blend: combine a primary model with a booster model.

    The booster contributes ``boost_weight`` to each item's score.
    Items only in the base model get weight ``(1 - boost_weight) / rank``.
    Items only in the booster get weight ``boost_weight / rank``.

    Parameters
    ----------
    base_preds   : primary model predictions
    boost_preds  : secondary model predictions
    boost_weight : weight given to boost_preds (0 = ignore booster)
    top_k        : items to return per user

    Returns
    -------
    dict  customer_id → [item_id, ...]
    """
    return ensemble_rankers(
        [base_preds, boost_preds],
        weights=[1.0 - boost_weight, boost_weight],
        method="reciprocal_rank",
        top_k=top_k,
    )
