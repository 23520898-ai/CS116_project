"""Build sparse user-item interaction matrices for collaborative filtering."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp

log = logging.getLogger(__name__)

# Default behavioural signal weights relative to a single transaction
DEFAULT_EVENT_WEIGHTS: dict[str, float] = {
    "purchase":    1.0,
    "add_to_cart": 0.4,
    "wishlist":    0.2,
    "view":        0.1,
}


def _event_weight_expr(weights: dict[str, float]) -> pl.Expr:
    """
    Build a polars expression that maps ``event_type`` → float weight.
    Unlisted event types receive weight 0.
    """
    # Build nested when/then/otherwise bottom-up
    expr: pl.Expr = pl.lit(0.0, dtype=pl.Float32)
    for etype, w in reversed(list(weights.items())):
        expr = (
            pl.when(pl.col("event_type") == etype)
            .then(pl.lit(float(w), dtype=pl.Float32))
            .otherwise(expr)
        )
    return expr


def build_user_item_matrix(
    trans_df:      pl.DataFrame,
    event_df:      pl.DataFrame | None = None,
    event_weights: dict[str, float] | None = None,
    user2idx:      dict[int, int]  | None = None,
    item2idx:      dict[str, int]  | None = None,
) -> tuple[sp.csr_matrix, dict[int, int], dict[str, int]]:
    """
    Aggregate purchase quantities (+ optional event signals) into a
    sparse **user × item** confidence matrix.

    Parameters
    ----------
    trans_df      : transaction DataFrame with columns
                    ``customer_id``, ``item_id``, ``quantity``
    event_df      : optional event DataFrame with columns
                    ``customer_id``, ``item_id``, ``event_type``
    event_weights : weight per event_type; defaults to DEFAULT_EVENT_WEIGHTS
    user2idx      : pre-built customer_id → row index (for inference reuse)
    item2idx      : pre-built item_id → col index

    Returns
    -------
    matrix   : (n_users, n_items) float32 CSR matrix
    user2idx : customer_id → row index
    item2idx : item_id     → col index
    """
    weights = event_weights or DEFAULT_EVENT_WEIGHTS

    # ── Collect (customer_id, item_id, weight) pairs ──────────────────────────
    frames: list[pl.DataFrame] = [
        trans_df.select([
            pl.col("customer_id"),
            pl.col("item_id"),
            pl.col("quantity").cast(pl.Float32).alias("weight"),
        ])
    ]

    if event_df is not None and len(event_df) > 0:
        e_frame = (
            event_df
            .select(["customer_id", "item_id", "event_type"])
            .with_columns(_event_weight_expr(weights).alias("weight"))
            .filter(pl.col("weight") > 0)
            .select(["customer_id", "item_id", "weight"])
        )
        frames.append(e_frame)

    combined = pl.concat(frames, how="diagonal_relaxed")

    # ── Aggregate per (user, item) pair ───────────────────────────────────────
    agg = (
        combined
        .group_by(["customer_id", "item_id"])
        .agg(pl.col("weight").sum())
    )

    # ── Build index mappings ──────────────────────────────────────────────────
    if user2idx is None:
        unique_users = sorted(agg["customer_id"].unique().to_list())
        user2idx = {uid: i for i, uid in enumerate(unique_users)}

    if item2idx is None:
        unique_items = sorted(agg["item_id"].unique().to_list())
        item2idx = {iid: i for i, iid in enumerate(unique_items)}

    # ── Build COO arrays ──────────────────────────────────────────────────────
    cust_list   = agg["customer_id"].to_list()
    item_list   = agg["item_id"].to_list()
    weight_arr  = agg["weight"].to_numpy().astype(np.float32)

    row_idx = np.fromiter((user2idx.get(u,  -1) for u  in cust_list), dtype=np.int32)
    col_idx = np.fromiter((item2idx.get(it, -1) for it in item_list), dtype=np.int32)

    mask = (row_idx >= 0) & (col_idx >= 0)
    row_idx, col_idx, weight_arr = row_idx[mask], col_idx[mask], weight_arr[mask]

    matrix = sp.csr_matrix(
        (weight_arr, (row_idx, col_idx)),
        shape=(len(user2idx), len(item2idx)),
        dtype=np.float32,
    )

    log.info(
        "Interaction matrix: %d users × %d items | nnz=%d | density=%.5f%%",
        matrix.shape[0], matrix.shape[1], matrix.nnz,
        100.0 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
    )
    return matrix, user2idx, item2idx


def get_user_purchased_items(trans_df: pl.DataFrame) -> dict[int, set[str]]:
    """Return a dict mapping ``customer_id`` → set of purchased ``item_id``s."""
    grouped = (
        trans_df
        .group_by("customer_id")
        .agg(pl.col("item_id").unique().alias("items"))
    )
    return {row["customer_id"]: set(row["items"]) for row in grouped.iter_rows(named=True)}
