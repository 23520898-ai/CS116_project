"""
User × Item cross features (interaction features).
====================================================
These features capture the relationship between a specific user and a
specific candidate item, combining signals from all three Stage-1 sources.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def build_cross_features(
    candidates_df:   pl.DataFrame,
    trans_df:        pl.DataFrame,
    covisit_scores:  dict[int, dict[str, float]] | None = None,
    w2v_scores:      dict[int, dict[str, float]] | None = None,
    items_df:        pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Build cross features for a DataFrame of (customer_id, item_id) candidate
    pairs.

    Parameters
    ----------
    candidates_df  : DataFrame with columns [customer_id, item_id,
                     from_history (int8), from_covisit (int8),
                     from_w2v (int8), stage1_rank (int32)]
    trans_df       : training transactions
    covisit_scores : {customer_id: {item_id: score}}  (from stage-1)
    w2v_scores     : {customer_id: {item_id: cosine_sim}}  (from stage-1)
    items_df       : items metadata (optional). When provided, adds
                     user×category affinity features:
                     ui_user_cat1_count, ui_user_cat2_count,
                     ui_user_cat1_pct,   ui_user_cat2_pct.

    Returns
    -------
    candidates_df with additional feature columns
    """
    ref = trans_df["updated_date"].max()

    # ── User–item interaction stats from training history ─────────────────────
    ui_stats = (
        trans_df
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().cast(pl.UInt32).alias("ui_history_count"),
            pl.col("updated_date").max().alias("_ui_last_date"),
        ])
        .with_columns(
            pl.lit(1).cast(pl.Int8).alias("ui_in_history"),
            (
                (pl.lit(ref) - pl.col("_ui_last_date")).dt.total_days().cast(pl.Float32)
            ).alias("ui_history_last_days"),
        )
        .drop("_ui_last_date")
    )

    out = candidates_df.join(ui_stats, on=["customer_id", "item_id"], how="left")
    out = out.with_columns([
        pl.col("ui_in_history").fill_null(0).cast(pl.Int8),
        pl.col("ui_history_count").fill_null(0),
        pl.col("ui_history_last_days").fill_null(9999.0),
    ])

    # ── Covisitation scores ────────────────────────────────────────────────────
    if covisit_scores:
        cov_rows = [
            (uid, iid, score)
            for uid, item_scores in covisit_scores.items()
            for iid, score in item_scores.items()
        ]
        if cov_rows:
            cov_df = pl.DataFrame(
                cov_rows,
                schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "ui_covisit_score": pl.Float32},
                orient="row",
            )
            out = out.join(cov_df, on=["customer_id", "item_id"], how="left")
    if "ui_covisit_score" not in out.columns:
        out = out.with_columns(pl.lit(0.0).alias("ui_covisit_score"))
    out = out.with_columns(pl.col("ui_covisit_score").fill_null(0.0))

    # ── Word2Vec similarity scores ─────────────────────────────────────────────
    if w2v_scores:
        w2v_rows = [
            (uid, iid, sim)
            for uid, item_sims in w2v_scores.items()
            for iid, sim in item_sims.items()
        ]
        if w2v_rows:
            w2v_df = pl.DataFrame(
                w2v_rows,
                schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "ui_w2v_score": pl.Float32},
                orient="row",
            )
            out = out.join(w2v_df, on=["customer_id", "item_id"], how="left")
    if "ui_w2v_score" not in out.columns:
        out = out.with_columns(pl.lit(0.0).alias("ui_w2v_score"))
    out = out.with_columns(pl.col("ui_w2v_score").fill_null(0.0))

    # ── User×Category affinity features (requires items metadata) ─────────────
    # For each (user, candidate_item) pair:
    #   - How many times has the user bought from category_l1 / _l2 of this item?
    #   - What fraction of the user's purchases are in that category?
    if items_df is not None and "category_l1" in items_df.columns:
        item_cats = items_df.select(["item_id", "category_l1", "category_l2"])

        # Count user purchases per category (trans_df × item categories)
        trans_cats = trans_df.select(["customer_id", "item_id"]).join(
            item_cats, on="item_id", how="left"
        )
        user_cat1_cnt = (
            trans_cats
            .filter(pl.col("category_l1").is_not_null())
            .group_by(["customer_id", "category_l1"])
            .agg(pl.len().cast(pl.Float32).alias("_uc1_cnt"))
        )
        user_cat2_cnt = (
            trans_cats
            .filter(pl.col("category_l2").is_not_null())
            .group_by(["customer_id", "category_l2"])
            .agg(pl.len().cast(pl.Float32).alias("_uc2_cnt"))
        )
        user_total_cnt = (
            trans_df
            .group_by("customer_id")
            .agg(pl.len().cast(pl.Float32).alias("_u_total"))
        )

        # Add item categories to candidates, then look up affinity counts
        out = (
            out
            .join(item_cats, on="item_id", how="left")
            .join(user_cat1_cnt, on=["customer_id", "category_l1"], how="left")
            .join(user_cat2_cnt, on=["customer_id", "category_l2"], how="left")
            .join(user_total_cnt, on="customer_id", how="left")
            .with_columns([
                pl.col("_uc1_cnt").fill_null(0.0).alias("ui_user_cat1_count"),
                pl.col("_uc2_cnt").fill_null(0.0).alias("ui_user_cat2_count"),
                (
                    pl.col("_uc1_cnt").fill_null(0.0)
                    / (pl.col("_u_total").fill_null(1.0) + 1e-6)
                ).alias("ui_user_cat1_pct"),
                (
                    pl.col("_uc2_cnt").fill_null(0.0)
                    / (pl.col("_u_total").fill_null(1.0) + 1e-6)
                ).alias("ui_user_cat2_pct"),
            ])
            .drop(["category_l1", "category_l2", "_uc1_cnt", "_uc2_cnt", "_u_total"])
        )

    return out
