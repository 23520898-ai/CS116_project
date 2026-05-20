"""
Lưới 3 – Word2Vec / Deep-Learning Candidates
=============================================
Mỗi chuỗi mua hàng của user (sắp xếp theo thời gian) được coi là một
"câu", mỗi item là một "từ".  Sau khi train Word2Vec (skip-gram), mỗi
item có vector embedding 64-chiều.

Tại inference:
- Tính vector đại diện của user = weighted average của N item gần nhất
- Tìm top-K item có cosine similarity cao nhất với vector đó

Không cần ANN library nặng: với ~20K items × 64 dims, phép nhân ma trận
(sklearn / numpy) chạy đủ nhanh.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


# ── Build sequences ───────────────────────────────────────────────────────────

def build_sequences(trans_df: pl.DataFrame) -> list[list[str]]:
    """
    One purchase sequence per customer, sorted by date ascending.
    Items repeated across dates are kept (Word2Vec benefits from repetition).
    """
    df = (
        trans_df
        .sort("updated_date")
        .group_by("customer_id")
        .agg(pl.col("item_id").alias("items"))
    )
    return [row["items"] for row in df.iter_rows(named=True)]


# ── Train ─────────────────────────────────────────────────────────────────────

def train_word2vec(
    trans_df:    pl.DataFrame,
    vector_size: int = 64,
    window:      int = 5,
    min_count:   int = 3,
    epochs:      int = 10,
    workers:     int = 8,
) -> Any:
    """
    Train a Word2Vec (skip-gram) model on item purchase sequences.
    Requires: pip install gensim
    """
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError(
            "gensim is required for Word2Vec candidates.\n"
            "Install: pip install gensim"
        ) from exc

    sequences = build_sequences(trans_df)
    log.info("Training Word2Vec on %d customer sequences …", len(sequences))

    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,       # skip-gram (better for sparse / long-tail items)
    )
    log.info("Word2Vec ready  |  vocab: %d items", len(model.wv))
    return model


# ── Embedding matrix ──────────────────────────────────────────────────────────

def build_embedding_matrix(w2v_model: Any) -> tuple[np.ndarray, list[str]]:
    """
    Build a L2-normalised embedding matrix for fast cosine-similarity lookup.

    Returns
    -------
    emb_matrix : (n_items, vector_size) float32, each row is unit-normalised
    item_list  : item_id for each row
    """
    item_list = list(w2v_model.wv.key_to_index.keys())
    vectors   = np.array([w2v_model.wv[it] for it in item_list], dtype=np.float32)

    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1.0, norms)
    vectors = vectors / norms
    return vectors, item_list


# ── Inference ─────────────────────────────────────────────────────────────────

def get_w2v_candidates(
    history_items: list[str],
    w2v_model:     Any,
    emb_matrix:    np.ndarray,
    item_list:     list[str],
    n_candidates:  int = 200,
    history_set:   set[str] | None = None,
    n_recent:      int = 20,
) -> tuple[list[str], dict[str, float]]:
    """
    Generate candidates nearest to the user's recency-weighted item vector.

    Recency weighting: most recent item → weight 1.0,
    each subsequent item's weight decays by factor 0.9.

    Returns
    -------
    candidates  : list[item_id]
    similarities: dict item_id → cosine similarity
    """
    vocab = w2v_model.wv.key_to_index
    # Take at most n_recent items that are in the W2V vocabulary
    query_items = [it for it in history_items[:n_recent] if it in vocab]
    if not query_items:
        return [], {}

    query_vecs = np.array([w2v_model.wv[it] for it in query_items], dtype=np.float32)

    # Exponential recency decay: index 0 = most recent
    weights = np.power(0.9, np.arange(len(query_items)))
    weights /= weights.sum()
    user_vec = (weights[:, None] * query_vecs).sum(axis=0)

    norm = np.linalg.norm(user_vec)
    if norm == 0:
        return [], {}
    user_vec /= norm

    # Cosine similarities (emb_matrix is already L2-normalised)
    sims = (emb_matrix @ user_vec).astype(float)  # (n_items,)

    # Mask out history items
    hs = history_set or set()
    item_arr = np.array(item_list)
    mask = np.fromiter((it not in hs for it in item_list), dtype=bool)
    sims = np.where(mask, sims, -np.inf)

    top_idx = np.argsort(sims)[::-1][:n_candidates]
    top_idx = top_idx[sims[top_idx] > -np.inf]

    candidates  = item_arr[top_idx].tolist()
    similarities = {candidates[k]: float(sims[top_idx[k]]) for k in range(len(candidates))}
    return candidates, similarities


# ── Persistence ───────────────────────────────────────────────────────────────

def save_w2v_artifacts(
    w2v_model:  Any,
    emb_matrix: np.ndarray,
    item_list:  list[str],
    dir_path:   Path,
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    w2v_model.save(str(dir_path / "w2v.model"))
    with open(dir_path / "emb_artifacts.pkl", "wb") as fh:
        pickle.dump({"emb_matrix": emb_matrix, "item_list": item_list}, fh)
    log.info("W2V artifacts saved → %s", dir_path)


def load_w2v_artifacts(dir_path: Path) -> tuple[Any, np.ndarray, list[str]]:
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError("pip install gensim") from exc

    w2v_model = Word2Vec.load(str(dir_path / "w2v.model"))
    with open(dir_path / "emb_artifacts.pkl", "rb") as fh:
        art = pickle.load(fh)
    return w2v_model, art["emb_matrix"], art["item_list"]
