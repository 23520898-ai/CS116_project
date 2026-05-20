"""
Lưới 2 – Covisitation Matrix (Graph-based Candidates)
=======================================================
Nếu user đã mua item A, thì các item B thường xuất hiện cùng bill với A
(hay nhiều user khác cũng mua A rồi mua B) sẽ được đề xuất.

Workflow
--------
1. build_covisitation_matrix(trans_df)  →  covisit dict
2. get_covisit_candidates(history, covisit)  →  scored candidate list

Thuật toán: B (bills × items) sparse matrix, covisitation = Bᵀ B.
Mỗi entry covisit[A][B] = số bill chứa cả A lẫn B.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

log = logging.getLogger(__name__)


# ── Build ─────────────────────────────────────────────────────────────────────

def build_covisitation_matrix(
    trans_df:       pl.DataFrame,
    top_k_per_item: int = 50,
    max_bill_size:  int = 20,
) -> dict[str, list[tuple[str, float]]]:
    """
    Build item–item covisitation matrix from transaction data.

    Items sharing a bill_id are "co-visited".  Bills with more than
    `max_bill_size` items are capped to avoid combinatorial explosion.

    Returns
    -------
    dict  item_id → [(co_item_id, score), ...]  sorted by score descending
    """
    log.info("Building covisitation matrix …")

    # ── Unique (bill_id, item_id) pairs, capped per bill ─────────────────────
    bill_item = (
        trans_df
        .select(["bill_id", "item_id"])
        .unique()
        .sort(["bill_id", "item_id"])
        .group_by("bill_id")
        .agg(pl.col("item_id").head(max_bill_size).alias("item_id"))
        .explode("item_id")
    )

    log.info(
        "Bills: %d | Items: %d | bill-item pairs: %d",
        bill_item["bill_id"].n_unique(),
        bill_item["item_id"].n_unique(),
        len(bill_item),
    )

    # ── Build integer indices via join (fast, avoids Python loops) ────────────
    bills_sorted = bill_item["bill_id"].unique().sort()
    items_sorted = bill_item["item_id"].unique().sort()

    bill_idx_df = pl.DataFrame({
        "bill_id":  bills_sorted,
        "bill_idx": pl.arange(len(bills_sorted), eager=True),
    })
    item_idx_df = pl.DataFrame({
        "item_id":  items_sorted,
        "item_idx": pl.arange(len(items_sorted), eager=True),
    })

    indexed = (
        bill_item
        .join(bill_idx_df, on="bill_id")
        .join(item_idx_df, on="item_id")
    )

    row_arr = indexed["bill_idx"].to_numpy()
    col_arr = indexed["item_idx"].to_numpy()
    n_bills = len(bills_sorted)
    n_items = len(items_sorted)

    # ── B (bills × items) sparse matrix ──────────────────────────────────────
    B = sp.csr_matrix(
        (np.ones(len(row_arr), dtype=np.float32), (row_arr, col_arr)),
        shape=(n_bills, n_items),
    )

    # ── Covisitation = Bᵀ B (items × items) ──────────────────────────────────
    log.info("Computing BᵀB (%d × %d) …", n_items, n_items)
    BtB = (B.T @ B).astype(np.float32)   # remains sparse

    # Zero out self-covisitation
    BtB = BtB.tolil()
    BtB.setdiag(0)
    BtB = BtB.tocsr()

    idx2item: dict[int, str] = dict(enumerate(items_sorted.to_list()))

    # ── Extract top-K per item ────────────────────────────────────────────────
    log.info("Extracting top-%d neighbours per item …", top_k_per_item)
    covisit: dict[str, list[tuple[str, float]]] = {}

    for i in range(n_items):
        row = np.asarray(BtB.getrow(i).todense()).flatten()
        top_idx = np.argsort(row)[::-1][:top_k_per_item]
        top_idx = top_idx[row[top_idx] > 0]
        covisit[idx2item[i]] = [(idx2item[j], float(row[j])) for j in top_idx]

    log.info("Covisitation matrix ready  (%d items covered).", len(covisit))
    return covisit


# ── Inference ─────────────────────────────────────────────────────────────────

def get_covisit_candidates(
    history_items: list[str],
    covisit:       dict[str, list[tuple[str, float]]],
    n_candidates:  int = 300,
    history_set:   set[str] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """
    Aggregate covisitation scores for all items reachable from `history_items`
    and return the top `n_candidates`.

    Returns
    -------
    candidates : list of item_ids sorted by descending score
    scores     : dict item_id → aggregated covisitation score
    """
    scores: dict[str, float] = {}
    for hist_item in history_items:
        for cand, score in covisit.get(hist_item, []):
            scores[cand] = scores.get(cand, 0.0) + score

    if history_set:
        scores = {k: v for k, v in scores.items() if k not in history_set}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [item for item, _ in ranked[:n_candidates]]
    return candidates, {k: v for k, v in ranked[:n_candidates]}


# ── Persistence ───────────────────────────────────────────────────────────────

def save_covisit(covisit: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(covisit, fh)
    log.info("Covisitation matrix saved → %s", path)


def load_covisit(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
