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
import time

import numpy as np
import polars as pl

from src.features.user_features  import build_user_features
from src.features.item_features  import build_item_features
from src.features.cross_features import build_cross_features

log = logging.getLogger(__name__)

# Columns that are identifiers / labels – excluded from the feature matrix
_NON_FEATURE_COLS = ["customer_id", "item_id", "label"]


# ── Improvement 2: Extended Labels ───────────────────────────────────────────

def create_extended_labels(
    candidates_df: pl.DataFrame,
    items_df: pl.DataFrame,
    label_df: pl.DataFrame,
    use_soft_labels: bool = True,
) -> pl.DataFrame:
    """
    Extend binary labels to relevance grades for LambdaRank.

    Grades
    ------
    2 : hard positive  – item was purchased in the label period
    1 : soft positive  – item shares category_l2 with a purchased item
                         (only when use_soft_labels=True)
    0 : negative

    Why effective
    -------------
    LambdaRank with multi-grade relevance provides richer gradient signal than
    binary 0/1.  Soft positives reduce the effective class-imbalance and teach
    the model to rank category-relevant items above completely irrelevant ones,
    even when they were not the exact item bought.

    Returns
    -------
    candidates_df with an integer 'label' column (grades 0/1/2).
    """
    t0 = time.perf_counter()

    pos = label_df.select(["customer_id", "item_id"]).with_columns(
        pl.lit(2).cast(pl.Int32).alias("label")
    )
    df = (
        candidates_df
        .join(pos, on=["customer_id", "item_id"], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int32))
    )

    if not use_soft_labels or "category_l2" not in items_df.columns:
        n2 = int((df["label"] == 2).sum())
        log.info("Labels: %d hard positives (grade-2), soft labels disabled  [%.1fs]",
                 n2, time.perf_counter() - t0)
        return df

    item_cats = items_df.select(["item_id", "category_l2"]).filter(
        pl.col("category_l2").is_not_null()
    )
    # Category of each purchased item per user
    purchased_cats = (
        label_df.select(["customer_id", "item_id"])
        .join(item_cats, on="item_id", how="inner")
        .select(["customer_id", "category_l2"])
        .unique()
    )
    # Candidates whose category matches any purchased category for that user
    soft_mask = (
        df.select(["customer_id", "item_id"])
        .join(item_cats, on="item_id", how="left")
        .join(purchased_cats, on=["customer_id", "category_l2"], how="inner")
        .select(["customer_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int32).alias("_soft"))
    )
    df = df.join(soft_mask, on=["customer_id", "item_id"], how="left")
    df = df.with_columns(
        pl.when(pl.col("label") >= 2).then(2)
        .when(pl.col("_soft") == 1).then(1)
        .otherwise(0)
        .cast(pl.Int32).alias("label")
    ).drop("_soft")

    n2 = int((df["label"] == 2).sum())
    n1 = int((df["label"] == 1).sum())
    n0 = int((df["label"] == 0).sum())
    log.info(
        "Extended labels: grade2=%d  grade1=%d  grade0=%d  "
        "positive_rate=%.2f%%  [%.1fs]",
        n2, n1, n0, (n2 + n1) / max(1, len(df)) * 100,
        time.perf_counter() - t0,
    )
    return df


# ── Improvement 3: Hard Negative Mining ──────────────────────────────────────

def add_hard_negatives(
    feature_df: pl.DataFrame,
    items_df: pl.DataFrame,
    n_hard_per_user: int = 15,
) -> pl.DataFrame:
    """
    Mark hard negatives within the existing candidate pool.

    Adds column 'hard_neg_type' (Int8):
      0 : regular (easy) negative
      1 : popular negative – item is in the global top-10% by purchase count
      2 : same-category negative – shares category_l2 with a positive item

    Why effective
    -------------
    Random negatives are mostly easy (unpopular, wrong category).  Hard
    negatives force the model to learn fine-grained discriminative signals
    rather than simply separating popular from obscure items.  Training with
    a mix of easy + hard negatives is known to accelerate convergence and
    improve precision on difficult cases.

    Returns
    -------
    feature_df with additional 'hard_neg_type' column.
    """
    t0 = time.perf_counter()

    if "label" not in feature_df.columns:
        log.warning("add_hard_negatives: 'label' column missing – skipping")
        return feature_df.with_columns(pl.lit(0).cast(pl.Int8).alias("hard_neg_type"))

    feature_df = feature_df.with_columns(pl.lit(0).cast(pl.Int8).alias("hard_neg_type"))

    # Type 1: popular negatives (top-10% by i_n_purchases)
    if "i_n_purchases" in feature_df.columns:
        pop_thresh = float(feature_df.filter(pl.col("label") == 0)["i_n_purchases"].quantile(0.90))
        feature_df = feature_df.with_columns(
            pl.when((pl.col("label") == 0) & (pl.col("i_n_purchases") >= pop_thresh))
            .then(1)
            .otherwise(pl.col("hard_neg_type"))
            .cast(pl.Int8).alias("hard_neg_type")
        )

    # Type 2: same-category negatives
    if "i_category_l2_enc" in feature_df.columns:
        pos_cats = (
            feature_df
            .filter(pl.col("label") >= 1)
            .select(["customer_id", "i_category_l2_enc"])
            .unique()
        )
        same_cat = (
            feature_df
            .filter(pl.col("label") == 0)
            .select(["customer_id", "item_id", "i_category_l2_enc"])
            .join(pos_cats, on=["customer_id", "i_category_l2_enc"], how="inner")
            .select(["customer_id", "item_id"])
            .unique()
            .with_columns(pl.lit(2).cast(pl.Int8).alias("_t2"))
        )
        feature_df = feature_df.join(same_cat, on=["customer_id", "item_id"], how="left")
        feature_df = feature_df.with_columns(
            pl.when((pl.col("label") == 0) & (pl.col("_t2") == 2))
            .then(2)
            .otherwise(pl.col("hard_neg_type"))
            .cast(pl.Int8).alias("hard_neg_type")
        ).drop("_t2")

    n1 = int((feature_df["hard_neg_type"] == 1).sum())
    n2 = int((feature_df["hard_neg_type"] == 2).sum())
    log.info(
        "Hard negatives: type1(popular)=%d  type2(same-cat)=%d  [%.1fs]",
        n1, n2, time.perf_counter() - t0,
    )
    return feature_df


def build_feature_matrix(
    candidates_df:  pl.DataFrame,
    trans_df:       pl.DataFrame,
    items_df:       pl.DataFrame,
    covisit_scores: dict[int, dict[str, float]] | None = None,
    w2v_scores:     dict[int, dict[str, float]] | None = None,
    label_df:       pl.DataFrame | None = None,
    user_feat:      pl.DataFrame | None = None,
    item_feat:      pl.DataFrame | None = None,
    pre_computed:   dict | None = None,
    # ── New optional feature groups ──────────────────────────────────────────
    temporal_user_feat: pl.DataFrame | None = None,
    temporal_item_feat: pl.DataFrame | None = None,
    session_feat:       pl.DataFrame | None = None,
    item_trend_feat:    pl.DataFrame | None = None,
    extend_labels:      bool = False,
    use_hard_negatives: bool = False,
) -> pl.DataFrame:
    """
    Attach all feature groups to the candidate pairs.

    Parameters
    ----------
    candidates_df       : (customer_id, item_id, from_history, from_covisit,
                           from_w2v)  from Stage 1
    trans_df            : training transaction DataFrame
    items_df            : items metadata DataFrame
    covisit_scores      : {customer_id: {item_id: covisit_score}}
    w2v_scores          : {customer_id: {item_id: cosine_sim}}
    label_df            : positive (purchased) pairs; adds "label" column
    pre_computed        : dict from precompute_cross_stats().
                          Optionally contains extra keys:
                            "cross_cat_feat"  – from build_category_affinity_temporal
                            "ui_history_feat" – from build_user_item_history_features
    temporal_user_feat  : DataFrame from build_temporal_decay_features()[0]
    temporal_item_feat  : DataFrame from build_temporal_decay_features()[1]
    session_feat        : DataFrame from build_session_features()
    item_trend_feat     : DataFrame from build_item_trend_features()
    extend_labels       : if True, use create_extended_labels() (grades 0/1/2)
                          instead of binary 0/1 labels
    use_hard_negatives  : if True, add 'hard_neg_type' column via add_hard_negatives()

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
    _tc0 = time.perf_counter()
    df = build_cross_features(
        candidates_df,
        trans_df=trans_df,
        covisit_scores=covisit_scores,
        w2v_scores=w2v_scores,
        items_df=items_df,
        pre_computed=pre_computed,
    )
    _tc1 = time.perf_counter()

    # ── Join user and item features ───────────────────────────────────────────
    _batch_users = df["customer_id"].unique()
    df = df.join(
        user_feat.filter(pl.col("customer_id").is_in(_batch_users)),
        on="customer_id", how="left",
    )
    df = df.join(item_feat, on="item_id", how="left")

    # ── Join optional extended feature groups ─────────────────────────────────
    if temporal_user_feat is not None and temporal_user_feat.height > 0:
        df = df.join(
            temporal_user_feat.filter(pl.col("customer_id").is_in(_batch_users)),
            on="customer_id", how="left",
        )
    if temporal_item_feat is not None and temporal_item_feat.height > 0:
        df = df.join(temporal_item_feat, on="item_id", how="left")
    if session_feat is not None and session_feat.height > 0:
        df = df.join(
            session_feat.filter(pl.col("customer_id").is_in(_batch_users)),
            on="customer_id", how="left",
        )
    if item_trend_feat is not None and item_trend_feat.height > 0:
        df = df.join(item_trend_feat, on="item_id", how="left")

    _tc2 = time.perf_counter()

    # ── Price ratio: item avg price / user avg price ──────────────────────────
    if "i_avg_price" in df.columns and "u_avg_price" in df.columns:
        df = df.with_columns(
            (pl.col("i_avg_price") / (pl.col("u_avg_price") + 1e-6))
            .alias("ui_price_ratio")
        )

    # ── Attach labels ─────────────────────────────────────────────────────────
    if label_df is not None:
        if extend_labels:
            # Grades 0/1/2 for LambdaRank (requires items_df for soft labels)
            df = create_extended_labels(df, items_df, label_df, use_soft_labels=True)
        else:
            pos = label_df.select(["customer_id", "item_id"]).with_columns(
                pl.lit(1).cast(pl.Int8).alias("label")
            )
            df = df.join(pos, on=["customer_id", "item_id"], how="left").with_columns(
                pl.col("label").fill_null(0).cast(pl.Int8)
            )

    # ── Mark hard negatives (for smarter sampling in sample_training_pairs) ──
    if use_hard_negatives and "label" in df.columns:
        df = add_hard_negatives(df, items_df)

    # ── Fill remaining numeric nulls ──────────────────────────────────────────
    df = df.with_columns(
        [
            pl.col(c).fill_null(0)
            for c in df.columns
            if c not in _NON_FEATURE_COLS
            and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64,
                                pl.UInt32, pl.Int8, pl.UInt8)
        ]
    )
    _tc3 = time.perf_counter()
    log.info(
        "  [feat timing] cross=%.1fs  ufeat/ifeat=%.1fs  fill=%.1fs  rows=%d  cols=%d",
        _tc1 - _tc0, _tc2 - _tc1, _tc3 - _tc2, df.height, df.width,
    )
    return df


def sample_training_pairs(
    feature_df:          pl.DataFrame,
    neg_ratio:           int   = 20,
    seed:                int   = 42,
    use_hard_negatives:  bool  = False,
    hard_neg_fraction:   float = 0.3,
) -> pl.DataFrame:
    """
    Downsample negatives to ``neg_ratio`` negatives per positive.

    When ``use_hard_negatives=True``, reserves a ``hard_neg_fraction`` of the
    negative budget for hard negatives (popular or same-category items) so the
    model always sees some challenging examples.

    Supports multi-grade labels (0/1/2) from ``create_extended_labels``.
    """
    positives = feature_df.filter(pl.col("label") >= 1)
    negatives = feature_df.filter(pl.col("label") == 0)

    n_neg_keep = len(positives) * neg_ratio

    if use_hard_negatives and "hard_neg_type" in negatives.columns:
        hard_negs = negatives.filter(pl.col("hard_neg_type") > 0)
        easy_negs = negatives.filter(pl.col("hard_neg_type") == 0)

        n_hard = min(len(hard_negs), int(n_neg_keep * hard_neg_fraction))
        n_easy = max(0, n_neg_keep - n_hard)

        sampled_hard = hard_negs.sample(n=min(n_hard, len(hard_negs)), seed=seed)
        sampled_easy = easy_negs.sample(n=min(n_easy, len(easy_negs)), seed=seed)
        sampled_negs = pl.concat([sampled_hard, sampled_easy])
        log.info(
            "Training pairs: %d positives + %d hard-neg + %d easy-neg  (ratio 1:%.1f)",
            len(positives), len(sampled_hard), len(sampled_easy),
            len(sampled_negs) / max(1, len(positives)),
        )
    else:
        if n_neg_keep < len(negatives):
            negatives = negatives.sample(n=n_neg_keep, seed=seed)
        sampled_negs = negatives
        log.info(
            "Training pairs: %d positives + %d negatives (ratio 1:%d)",
            len(positives), len(sampled_negs), neg_ratio,
        )

    return pl.concat([positives, sampled_negs]).sample(fraction=1.0, seed=seed)
