"""Item-side features computed from training transactions + items metadata."""
from __future__ import annotations

import polars as pl


def build_item_features(
    trans_df: pl.DataFrame,
    items_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute ~20 statistical + metadata features per item.

    Parameters
    ----------
    trans_df : training transaction DataFrame
    items_df : items metadata DataFrame

    Returns
    -------
    polars DataFrame indexed by item_id with columns:
      i_n_purchases, i_n_unique_buyers, i_n_bills,
      i_total_quantity, i_avg_price, i_avg_discount, i_avg_quantity,
      i_purchase_last_30d, i_purchase_last_90d,
      i_popularity_rank, i_buyer_diversity,
      i_trend_score, i_is_popular,
      i_sale_status, i_category_l1_enc, i_category_l2_enc, i_brand_enc
    """
    ref = trans_df["updated_date"].max()

    # ── Transaction-side aggregations ─────────────────────────────────────────
    base = trans_df.group_by("item_id").agg([
        pl.len().alias("i_n_purchases"),
        pl.col("customer_id").n_unique().alias("i_n_unique_buyers"),
        pl.col("bill_id").n_unique().alias("i_n_bills"),
        pl.col("quantity").sum().cast(pl.Float32).alias("i_total_quantity"),
        pl.col("price").cast(pl.Float32).mean().alias("i_avg_price"),
        pl.col("discount").cast(pl.Float32).mean().alias("i_avg_discount"),
        pl.col("quantity").cast(pl.Float32).mean().alias("i_avg_quantity"),
        pl.col("updated_date").min().alias("_first_seen"),
    ])

    # Popularity rank + buyer diversity
    base = base.with_columns([
        pl.col("i_n_purchases").rank(method="ordinal", descending=True)
            .cast(pl.Float32).alias("i_popularity_rank"),
        (pl.col("i_n_unique_buyers").cast(pl.Float32) /
         (pl.col("i_n_purchases").cast(pl.Float32) + 1)).alias("i_buyer_diversity"),
    ])

    # Mark top-10% most purchased items
    thresh = base["i_n_purchases"].quantile(0.90)
    base = base.with_columns(
        (pl.col("i_n_purchases") >= thresh).cast(pl.Int8).alias("i_is_popular")
    )

    # Recent purchase counts
    for days, col in [(30, "i_purchase_last_30d"), (90, "i_purchase_last_90d")]:
        cutoff = ref - pl.duration(days=days)
        recent = (
            trans_df
            .filter(pl.col("updated_date") >= cutoff)
            .group_by("item_id")
            .agg(pl.len().cast(pl.UInt32).alias(col))
        )
        base = base.join(recent, on="item_id", how="left").with_columns(
            pl.col(col).fill_null(0)
        )

    # Trend score: purchases last 30d / (avg monthly purchases over full period)
    span_months = max(1.0, (ref - base["_first_seen"].min()).days / 30.0)
    base = base.with_columns(
        (
            pl.col("i_purchase_last_30d").cast(pl.Float32) /
            (pl.col("i_n_purchases").cast(pl.Float32) / span_months + 1)
        ).alias("i_trend_score")
    ).drop("_first_seen")

    # ── Metadata from items table ─────────────────────────────────────────────
    # Label-encode categorical columns
    for col in ["category_l1", "category_l2", "brand"]:
        cats = items_df[col].unique().sort()
        cat_map = pl.DataFrame({col: cats, f"{col}_idx": pl.arange(len(cats), eager=True)})
        items_df = items_df.join(cat_map, on=col, how="left")

    meta = items_df.select([
        "item_id",
        pl.col("sale_status").cast(pl.Int32).alias("i_sale_status"),
        pl.col("category_l1_idx").cast(pl.Int32).alias("i_category_l1_enc"),
        pl.col("category_l2_idx").cast(pl.Int32).alias("i_category_l2_enc"),
        pl.col("brand_idx").cast(pl.Int32).alias("i_brand_enc"),
        pl.col("price").cast(pl.Float32).alias("i_list_price"),
    ])

    return base.join(meta, on="item_id", how="left")
