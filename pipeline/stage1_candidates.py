"""
Stage 1 Orchestrator – Candidate Generation
============================================
Merges candidates from all three "lưới" (filters) for each user,
deduplicates, and caps at MAX_CANDIDATES.

Return format per user
----------------------
{
  "history":          [item_id, ...]
  "covisit":          [item_id, ...]
  "w2v":              [item_id, ...]
  "all_candidates":   [item_id, ...]   ← union, capped
  "covisit_scores":   {item_id: float}
  "w2v_scores":       {item_id: float}
}
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from src.candidates.history      import get_history_candidates
from src.candidates.covisitation import get_covisit_candidates
from src.candidates.word2vec_cands import get_w2v_candidates
from src.config import (
    HISTORY_MAX_ITEMS, COVISIT_CANDS, W2V_CANDS,
    W2V_RECENT, MAX_CANDIDATES,
)

log = logging.getLogger(__name__)

CandidateResult = dict[str, Any]


def generate_candidates_for_users(
    user_ids:      list[int],
    trans_df:      pl.DataFrame,
    covisit:       dict[str, list[tuple[str, float]]],
    w2v_model:     Any,
    emb_matrix:    np.ndarray,
    item_list:     list[str],
    max_candidates: int = MAX_CANDIDATES,
    allowed_items: set[str] | None = None,
) -> dict[int, CandidateResult]:
    """
    Generate stage-1 candidates for a list of users.

    Parameters
    ----------
    user_ids    : list of customer_id values
    trans_df    : training transaction DataFrame (or history slice)
    covisit     : prebuilt covisitation dict
    w2v_model   : trained Word2Vec model (or None to skip)
    emb_matrix  : L2-normalised item embedding matrix
    item_list   : item_id for each row of emb_matrix
    max_candidates : cap on total candidates per user
    allowed_items : optional whitelist of item_id allowed for prediction
                    (e.g., items with sale_status == 1)

    Returns
    -------
    dict  customer_id → CandidateResult
    """
    # ── Precompute history lookup ─────────────────────────────────────────────
    log.info("Computing history candidates for %d users …", len(user_ids))
    history_map = get_history_candidates(
        trans_df.filter(pl.col("customer_id").is_in(user_ids)),
        max_items=HISTORY_MAX_ITEMS,
    )

    results: dict[int, CandidateResult] = {}

    for uid in user_ids:
        history = history_map.get(uid, [])
        if allowed_items is not None:
            history = [it for it in history if it in allowed_items]
        history_set = set(history)

        # ── Lưới 2: Covisitation ─────────────────────────────────────────────
        cov_cands, cov_scores = get_covisit_candidates(
            history, covisit, n_candidates=COVISIT_CANDS, history_set=history_set
        )
        if allowed_items is not None:
            cov_cands = [it for it in cov_cands if it in allowed_items]
            cov_scores = {k: v for k, v in cov_scores.items() if k in allowed_items}

        # ── Lưới 3: Word2Vec ─────────────────────────────────────────────────
        if w2v_model is not None:
            w2v_cands, w2v_sim = get_w2v_candidates(
                history, w2v_model, emb_matrix, item_list,
                n_candidates=W2V_CANDS,
                history_set=history_set,
                n_recent=W2V_RECENT,
            )
            if allowed_items is not None:
                w2v_cands = [it for it in w2v_cands if it in allowed_items]
                w2v_sim = {k: v for k, v in w2v_sim.items() if k in allowed_items}
        else:
            w2v_cands, w2v_sim = [], {}

        # ── Union & dedup (history → covisit → w2v priority) ─────────────────
        seen: set[str] = set()
        all_cands: list[str] = []
        for item in history + cov_cands + w2v_cands:
            if item not in seen:
                seen.add(item)
                all_cands.append(item)
            if len(all_cands) >= max_candidates:
                break

        results[uid] = {
            "history":        history,
            "covisit":        cov_cands,
            "w2v":            w2v_cands,
            "all_candidates": all_cands,
            "covisit_scores": cov_scores,
            "w2v_scores":     w2v_sim,
        }

    return results


def candidates_to_dataframe(
    candidate_results: dict[int, CandidateResult],
) -> pl.DataFrame:
    """
    Flatten the candidate dict into a (customer_id, item_id) DataFrame
    with binary source flags and stage-1 rank used as cross features.

    Columns: customer_id, item_id, from_history, from_covisit, from_w2v,
             stage1_rank
    """
    rows = []
    for uid, res in candidate_results.items():
        hist_set    = set(res["history"])
        covisit_set = set(res["covisit"])
        w2v_set     = set(res["w2v"])
        for rank, item in enumerate(res["all_candidates"], start=1):
            rows.append((
                uid,
                item,
                int(item in hist_set),
                int(item in covisit_set),
                int(item in w2v_set),
                rank,
            ))

    return pl.DataFrame(
        rows,
        schema={
            "customer_id": pl.Int32,
            "item_id":     pl.Utf8,
            "from_history":  pl.Int8,
            "from_covisit":  pl.Int8,
            "from_w2v":      pl.Int8,
            "stage1_rank":   pl.Int32,
        },
        orient="row",
    )
