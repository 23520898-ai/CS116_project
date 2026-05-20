"""User-side features computed from the training transaction history."""
from __future__ import annotations

import polars as pl

# Reference point for recency: end of last training month (Oct 31, 2025)
_TRAIN_END = pl.lit("2025-10-31 23:59:59").str.to_datetime()


def build_user_features(trans_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute ~25 statistical features per user from transaction history.

    Parameters
    ----------
    trans_df : training transaction DataFrame

    Returns
    -------
    polars DataFrame with one row per customer_id and feature columns:
      u_n_transactions, u_n_unique_items, u_n_unique_categories_l1,
      u_n_unique_bills, u_total_quantity, u_total_spend,
      u_avg_price, u_avg_discount, u_avg_bill_size,
      u_n_days_active, u_span_days, u_days_since_last,
      u_purchase_freq, u_purchases_last_30d, u_purchases_last_60d,
      u_purchases_last_90d, u_is_new_user,
      u_month_{1..10}_count
    """
    ref = trans_df["updated_date"].max()  # end of training period

    # Base aggregations
    base = trans_df.group_by("customer_id").agg([
        pl.len().alias("u_n_transactions"),
        pl.col("item_id").n_unique().alias("u_n_unique_items"),
        pl.col("bill_id").n_unique().alias("u_n_unique_bills"),
        pl.col("quantity").sum().cast(pl.Float32).alias("u_total_quantity"),
        (pl.col("price").cast(pl.Float64) * pl.col("quantity").cast(pl.Float64))
            .sum().cast(pl.Float32).alias("u_total_spend"),
        pl.col("price").cast(pl.Float32).mean().alias("u_avg_price"),
        pl.col("discount").cast(pl.Float32).mean().alias("u_avg_discount"),
        pl.col("updated_date").min().alias("_first_date"),
        pl.col("updated_date").max().alias("_last_date"),
        pl.col("updated_date").dt.date().n_unique().alias("u_n_days_active"),
    ])

    # Derived time features
    base = base.with_columns([
        (
            (pl.lit(ref) - pl.col("_last_date")).dt.total_days().cast(pl.Float32)
        ).alias("u_days_since_last"),
        (
            (pl.col("_last_date") - pl.col("_first_date")).dt.total_days().cast(pl.Float32)
        ).alias("u_span_days"),
    ])

    base = base.with_columns(
        (pl.col("u_n_transactions") / (pl.col("u_span_days") + 1)).alias("u_purchase_freq"),
        (pl.col("u_total_spend") / pl.col("u_n_unique_bills").cast(pl.Float32)).alias("u_avg_bill_size"),
        (pl.col("u_span_days") < 60).cast(pl.Int8).alias("u_is_new_user"),
    ).drop(["_first_date"])

    # Recent activity counts
    for days, col in [(30, "u_purchases_last_30d"), (60, "u_purchases_last_60d"), (90, "u_purchases_last_90d")]:
        cutoff = ref - pl.duration(days=days)
        recent = (
            trans_df
            .filter(pl.col("updated_date") >= cutoff)
            .group_by("customer_id")
            .agg(pl.len().cast(pl.UInt32).alias(col))
        )
        base = base.join(recent, on="customer_id", how="left").with_columns(
            pl.col(col).fill_null(0)
        )

    # Monthly purchase counts (months 1-10)
    for m in range(1, 11):
        col = f"u_month_{m}_count"
        monthly = (
            trans_df
            .filter(pl.col("updated_date").dt.month() == m)
            .group_by("customer_id")
            .agg(pl.len().cast(pl.UInt32).alias(col))
        )
        base = base.join(monthly, on="customer_id", how="left").with_columns(
            pl.col(col).fill_null(0)
        )

    return base.drop("_last_date")
