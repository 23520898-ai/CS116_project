"""
Stage 2 Orchestrator – Feature Engineering + Reranking
========================================================
1. Merges user features, item features, and cross features onto the
   candidate DataFrame produced by Stage 1.
2. Optionally labels candidates (positive = purchased in target period).
3. Trains or applies the LGBMRanker.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.features.user_features  import build_user_features
from src.features.item_features  import build_item_features
from src.features.cross_features import build_cross_features

log = logging.getLogger(__name__)

# Columns that are identifiers / labels – excluded from the feature matrix
_NON_FEATURE_COLS = ["customer_id", "item_id", "label"]


def build_feature_matrix(
    candidates_df:  pl.DataFrame,
    trans_df:       pl.DataFrame,
    items_df:       pl.DataFrame,
    covisit_scores: dict[int, dict[str, float]] | None = None,
    w2v_scores:     dict[int, dict[str, float]] | None = None,
    label_df:       pl.DataFrame | None = None,
    user_feat:      pl.DataFrame | None = None,
    item_feat:      pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Attach all feature groups to the candidate pairs.

    Parameters
    ----------
    candidates_df  : (customer_id, item_id, from_history, from_covisit,
                      from_w2v)  from Stage 1
    trans_df       : training transaction DataFrame
    items_df       : items metadata DataFrame
    covisit_scores : {customer_id: {item_id: covisit_score}}
    w2v_scores     : {customer_id: {item_id: cosine_sim}}
    label_df       : if provided, a DataFrame with [customer_id, item_id]
                     representing positive (actually-purchased) pairs;
                     a "label" column (0/1) is added

    Returns
    -------
    Full feature DataFrame (one row per candidate pair)
    """
    if user_feat is None:
        log.info("Building user features …")
        user_feat = build_user_features(trans_df)

    if item_feat is None:
        log.info("Building item features …")
        item_feat = build_item_features(trans_df, items_df)

    log.info("Building cross features …")
    df = build_cross_features(candidates_df, trans_df, covisit_scores, w2v_scores,
                              items_df=items_df)

    # ── Join user and item features ───────────────────────────────────────────
    df = df.join(user_feat, on="customer_id", how="left")
    df = df.join(item_feat, on="item_id",     how="left")

    # ── Price ratio: item avg price / user avg price ──────────────────────────
    if "i_avg_price" in df.columns and "u_avg_price" in df.columns:
        df = df.with_columns(
            (pl.col("i_avg_price") / (pl.col("u_avg_price") + 1e-6))
            .alias("ui_price_ratio")
        )

    # ── Optionally attach labels ──────────────────────────────────────────────
    if label_df is not None:
        pos = label_df.select(["customer_id", "item_id"]).with_columns(
            pl.lit(1).cast(pl.Int8).alias("label")
        )
        df = df.join(pos, on=["customer_id", "item_id"], how="left").with_columns(
            pl.col("label").fill_null(0).cast(pl.Int8)
        )

    # Fill any remaining nulls with 0 / -1 for numeric columns
    df = df.with_columns(
        [
            pl.col(c).fill_null(0)
            for c in df.columns
            if c not in _NON_FEATURE_COLS
            and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64,
                                pl.UInt32, pl.Int8, pl.UInt8)
        ]
    )

    return df


def sample_training_pairs(
    feature_df: pl.DataFrame,
    neg_ratio:  int = 20,
    seed:       int = 42,
) -> pl.DataFrame:
    """
    Downsample negatives to `neg_ratio` negatives per positive.
    Assumes a "label" column (0/1) is present.
    """
    positives = feature_df.filter(pl.col("label") == 1)
    negatives = feature_df.filter(pl.col("label") == 0)

    n_neg_keep = len(positives) * neg_ratio
    if n_neg_keep < len(negatives):
        negatives = negatives.sample(n=n_neg_keep, seed=seed)

    sampled = pl.concat([positives, negatives]).sample(fraction=1.0, seed=seed)
    log.info(
        "Training pairs: %d positives + %d negatives (ratio 1:%d)",
        len(positives), len(negatives.head(n_neg_keep)), neg_ratio,
    )
    return sampled
