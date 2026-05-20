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
) -> pl.DataFrame:
    """
    Build cross features for a DataFrame of (customer_id, item_id) candidate
    pairs.

    Parameters
    ----------
    candidates_df  : DataFrame with columns [customer_id, item_id,
                     from_history (int8), from_covisit (int8),
                     from_w2v (int8)]
    trans_df       : training transactions
    covisit_scores : {customer_id: {item_id: score}}  (from stage-1)
    w2v_scores     : {customer_id: {item_id: cosine_sim}}  (from stage-1)

    Returns
    -------
    candidates_df with additional columns:
      ui_in_history, ui_history_count, ui_history_last_days,
      ui_covisit_score, ui_w2v_score,
      ui_price_ratio  (item avg price / user avg price),
      from_history, from_covisit, from_w2v  (already present, kept)
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

    return out
