"""Ranking evaluation metrics for top-K recommendation and PIR scoring."""
from __future__ import annotations

import math
from typing import Hashable

import numpy as np
import polars as pl


def _ground_truth(df: pl.DataFrame) -> dict[int, set[str]]:
    """Extract  customer_id → set[item_id]  from a transaction DataFrame."""
    grouped = (
        df.group_by("customer_id")
        .agg(pl.col("item_id").unique().alias("items"))
    )
    return {row["customer_id"]: set(row["items"]) for row in grouped.iter_rows(named=True)}


def _ground_truth_lists(df: pl.DataFrame) -> dict[int, list[str]]:
    """Extract customer_id -> unique item list for PIR-compatible metrics."""
    grouped = (
        df.group_by("customer_id")
        .agg(pl.col("item_id").unique().alias("items"))
    )
    return {row["customer_id"]: list(row["items"]) for row in grouped.iter_rows(named=True)}


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def compute_pir_metrics(
    predictions: dict[Hashable, list[str]],
    ground_truth: dict[Hashable, list[str]],
    k: int = 10,
) -> dict[str, float | int]:
    """
    Compute PIR metrics.

    precision_at_10 uses denominator min(k, |actual|), matching PIR scorer.
    """
    eligible_customers = [
        customer_id for customer_id, items in ground_truth.items() if len(items) > 0
    ]
    if not eligible_customers:
        return {
            "total_correct_recommendations": 0,
            "precision_at_10": 0.0,
            "map": 0.0,
            "iou": 0.0,
            "reciprocal_rank_first_hit": 0.0,
        }

    total_correct_recommendations = 0
    precision_scores: list[float] = []
    ap_scores: list[float] = []
    iou_scores: list[float] = []
    rr_scores: list[float] = []

    for customer_id in eligible_customers:
        actual_items = _unique_keep_order(ground_truth.get(customer_id, []))
        pred_items = _unique_keep_order(predictions.get(customer_id, []))

        actual_set = set(actual_items)
        pred_set = set(pred_items)
        total_correct_recommendations += len(actual_set & pred_set)

        # precision@10 with PIR denominator
        topk = pred_items[:k]
        hit_count = len(set(topk) & actual_set)
        denom = min(k, len(actual_set)) if actual_set else 1
        precision_scores.append(hit_count / denom)

        # AP
        hit_seen = 0
        precision_sum = 0.0
        for rank, item_id in enumerate(pred_items, start=1):
            if item_id in actual_set:
                hit_seen += 1
                precision_sum += hit_seen / rank
        ap = precision_sum / len(actual_set) if actual_set else 0.0
        ap_scores.append(ap)

        # IOU
        union_size = len(actual_set | pred_set)
        iou = len(actual_set & pred_set) / union_size if union_size > 0 else 0.0
        iou_scores.append(iou)

        # Reciprocal rank of first hit
        rr = 0.0
        for rank, item_id in enumerate(pred_items, start=1):
            if item_id in actual_set:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    n = len(eligible_customers)
    return {
        "total_correct_recommendations": int(total_correct_recommendations),
        "precision_at_10": round(sum(precision_scores) / n, 6),
        "map": round(sum(ap_scores) / n, 6),
        "iou": round(sum(iou_scores) / n, 6),
        "reciprocal_rank_first_hit": round(sum(rr_scores) / n, 6),
    }


# ── Individual metrics ────────────────────────────────────────────────────────

def recall_at_k(
    predictions:  dict[int, list[str]],
    ground_truth: dict[int, set[str]],
    k: int,
) -> float:
    """Mean recall@K over users that have ground-truth purchases."""
    scores: list[float] = []
    for uid, pred in predictions.items():
        gt = ground_truth.get(uid, set())
        if not gt:
            continue
        hits = len(set(pred[:k]) & gt)
        scores.append(hits / min(len(gt), k))
    return float(np.mean(scores)) if scores else 0.0


def precision_at_k(
    predictions:  dict[int, list[str]],
    ground_truth: dict[int, set[str]],
    k: int,
) -> float:
    """Mean precision@K over users that have ground-truth purchases."""
    scores: list[float] = []
    for uid, pred in predictions.items():
        gt = ground_truth.get(uid, set())
        if not gt:
            continue
        hits = len(set(pred[:k]) & gt)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(
    predictions:  dict[int, list[str]],
    ground_truth: dict[int, set[str]],
    k: int,
) -> float:
    """Mean NDCG@K over users that have ground-truth purchases."""
    scores: list[float] = []
    for uid, pred in predictions.items():
        gt = ground_truth.get(uid, set())
        if not gt:
            continue
        dcg  = sum(1.0 / math.log2(i + 2) for i, item in enumerate(pred[:k]) if item in gt)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def hit_rate_at_k(
    predictions:  dict[int, list[str]],
    ground_truth: dict[int, set[str]],
    k: int,
) -> float:
    """Fraction of users who have at least one correct item in their top-K."""
    if not predictions:
        return 0.0
    hits = sum(
        1 for uid, pred in predictions.items()
        if ground_truth.get(uid, set()) & set(pred[:k])
    )
    return hits / len(predictions)


# ── Combined evaluator ────────────────────────────────────────────────────────

def evaluate(
    predictions: dict[int, list[str]],
    val_df:      pl.DataFrame,
    k:           int = 10,
) -> dict[str, float | int]:
    """
    Evaluate predictions against a validation DataFrame.

    Parameters
    ----------
    predictions : customer_id → list of recommended item_ids
    val_df      : validation transaction DataFrame
    k           : rank cut-off

    Returns
    -------
    dict with recall@k, precision@k, ndcg@k, hit_rate@k, n_users_evaluated
    """
    gt = _ground_truth(val_df)
    gt_list = _ground_truth_lists(val_df)

    # For partial prediction runs (e.g., --max-users quick checks),
    # evaluate PIR on the same user scope as predictions.
    gt_list_scoped = {uid: gt_list[uid] for uid in predictions if uid in gt_list}
    pir = compute_pir_metrics(predictions, gt_list_scoped, k=k)

    metrics: dict[str, float | int] = {
        f"recall@{k}":       recall_at_k(predictions, gt, k),
        # Keep key for compatibility, but align with PIR definition.
        f"precision@{k}":    pir["precision_at_10"] if k == 10 else precision_at_k(predictions, gt, k),
        f"ndcg@{k}":         ndcg_at_k(predictions, gt, k),
        f"hit_rate@{k}":     hit_rate_at_k(predictions, gt, k),
        "n_users_predicted": len(predictions),
        "n_users_in_gt":     len([u for u in predictions if u in gt]),
    }
    metrics.update(pir)
    return metrics
