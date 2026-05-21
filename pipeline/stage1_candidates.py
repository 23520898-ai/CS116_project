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
from src.candidates.covisitation import get_covisit_candidates, build_covisit_sparse, get_covisit_candidates_batch
from src.candidates.word2vec_cands import get_w2v_candidates, get_w2v_candidates_batch
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
    with binary source flags, stage-1 rank, and pre-joined covisit / w2v
    scores (avoids a separate dict→DataFrame conversion + join in Stage 2).

    Columns: customer_id, item_id, from_history, from_covisit, from_w2v,
             stage1_rank, ui_covisit_score, ui_w2v_score
    """
    rows = []
    for uid, res in candidate_results.items():
        hist_set    = set(res["history"])
        covisit_set = set(res["covisit"])
        w2v_set     = set(res["w2v"])
        cov_scores  = res.get("covisit_scores", {})
        w2v_scores  = res.get("w2v_scores", {})
        for rank, item in enumerate(res["all_candidates"], start=1):
            rows.append((
                uid,
                item,
                int(item in hist_set),
                int(item in covisit_set),
                int(item in w2v_set),
                rank,
                float(cov_scores.get(item, 0.0)),
                float(w2v_scores.get(item, 0.0)),
            ))

    return pl.DataFrame(
        rows,
        schema={
            "customer_id":      pl.Int32,
            "item_id":          pl.Utf8,
            "from_history":     pl.Int8,
            "from_covisit":     pl.Int8,
            "from_w2v":         pl.Int8,
            "stage1_rank":      pl.Int32,
            "ui_covisit_score": pl.Float32,
            "ui_w2v_score":     pl.Float32,
        },
        orient="row",
    )


def generate_candidates_for_users_fast(
    user_ids:       list[int],
    trans_df:       pl.DataFrame,
    covisit:        dict[str, list[tuple[str, float]]],
    w2v_model:      Any,
    emb_matrix:     np.ndarray | None,
    item_list:      list[str] | None,
    max_candidates: int = MAX_CANDIDATES,
    allowed_items:  set[str] | None = None,
    _prebuilt_covisit_sparse: tuple | None = None,
    _item_to_emb_idx:         dict[str, int] | None = None,
) -> dict[int, CandidateResult]:
    """
    Drop-in replacement for ``generate_candidates_for_users`` using vectorised
    batch operations for both covisitation and Word2Vec.

    Key differences vs. the original:
    - Covisitation uses scipy sparse row-sum instead of Python dict accumulation.
    - W2V uses a single batched matrix multiply (user_vecs @ emb_matrix.T) for
      all users at once instead of per-user dot products.

    Parameters
    ----------
    _prebuilt_covisit_sparse : optional pre-built (sparse_mat, item_list, item_to_idx)
        from ``build_covisit_sparse``.  Pass this when calling from a loop to
        avoid rebuilding the sparse matrix every batch.
    _item_to_emb_idx : optional pre-built {item_id: emb_row} mapping.
        Pass this together with *_prebuilt_covisit_sparse* for maximum speed.
    """
    # ── History (vectorised via polars) ───────────────────────────────────────
    history_map = get_history_candidates(
        trans_df.filter(pl.col("customer_id").is_in(user_ids)),
        max_items=HISTORY_MAX_ITEMS,
    )

    user_histories: list[list[str]] = []
    for uid in user_ids:
        history = history_map.get(uid, [])
        if allowed_items is not None:
            history = [it for it in history if it in allowed_items]
        user_histories.append(history)

    history_sets = [set(h) for h in user_histories]

    # ── Covisitation (sparse matrix batch) ───────────────────────────────────
    if _prebuilt_covisit_sparse is not None:
        cov_sparse, cov_item_list, cov_item_to_idx = _prebuilt_covisit_sparse
    else:
        cov_sparse, cov_item_list, cov_item_to_idx = build_covisit_sparse(covisit)

    cov_item_arr = np.array(cov_item_list)
    covisit_results = get_covisit_candidates_batch(
        user_history_list=user_histories,
        covisit_sparse=cov_sparse,
        item_to_idx=cov_item_to_idx,
        item_arr=cov_item_arr,
        n_candidates=COVISIT_CANDS,
        history_sets=history_sets,
    )
    if allowed_items is not None:
        covisit_results = [
            (
                [it for it in cands if it in allowed_items],
                {k: v for k, v in scores.items() if k in allowed_items},
            )
            for cands, scores in covisit_results
        ]

    # ── Word2Vec (batched matmul) ─────────────────────────────────────────────
    if w2v_model is not None and emb_matrix is not None and item_list is not None:
        if _item_to_emb_idx is None:
            _item_to_emb_idx = {item: i for i, item in enumerate(item_list)}

        w2v_results = get_w2v_candidates_batch(
            user_history_list=user_histories,
            emb_matrix=emb_matrix,
            item_list=item_list,
            item_to_emb_idx=_item_to_emb_idx,
            n_candidates=W2V_CANDS,
            history_sets=history_sets,
            n_recent=W2V_RECENT,
        )
        if allowed_items is not None:
            w2v_results = [
                (
                    [it for it in cands if it in allowed_items],
                    {k: v for k, v in scores.items() if k in allowed_items},
                )
                for cands, scores in w2v_results
            ]
    else:
        w2v_results = [([], {})] * len(user_ids)

    # ── Assemble per-user results ─────────────────────────────────────────────
    results: dict[int, CandidateResult] = {}
    for i, uid in enumerate(user_ids):
        history = user_histories[i]
        cov_cands, cov_scores = covisit_results[i]
        w2v_cands, w2v_sim = w2v_results[i]

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
