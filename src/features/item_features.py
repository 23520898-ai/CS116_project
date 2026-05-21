"""Item-side features computed from training transactions + items metadata."""
from __future__ import annotations

import logging
import time

import polars as pl

log = logging.getLogger(__name__)


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


# ── Improvement 8: Item Popularity Trend Features ────────────────────────────

def build_item_trend_features(
    trans_df: pl.DataFrame,
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """
    Compute multi-window popularity trend features per item.

    For each window W (default: 7, 30, 90 days):
      i_pop_last_Wd     : purchase count in last W days
      i_pop_prev_Wd     : purchase count in the W days before that
      i_trend_ratio_Wd  : recent / previous  (> 1 = trending up)

    Derived:
      i_trend_acceleration  : trend_7d / trend_30d  (short vs medium momentum)
      i_volatility          : std of monthly purchase counts
      i_n_active_months     : months with at least one sale
      i_seasonal_score      : peak-month count / mean monthly count

    Why effective
    -------------
    Static popularity ignores direction.  A rising item should rank above a
    stagnant one with the same total purchases.  Seasonal items need to be
    recommended at the right time.

    Returns
    -------
    DataFrame keyed by item_id with trend feature columns.
    """
    if windows is None:
        windows = [7, 30, 90]
    t0 = time.perf_counter()
    ref = trans_df["updated_date"].max()

    result = trans_df.select("item_id").unique()

    for w in windows:
        cutoff_recent = ref - pl.duration(days=w)
        cutoff_prev   = ref - pl.duration(days=w * 2)

        recent = (
            trans_df.filter(pl.col("updated_date") >= cutoff_recent)
            .group_by("item_id")
            .agg(pl.len().cast(pl.Float32).alias(f"i_pop_last_{w}d"))
        )
        previous = (
            trans_df.filter(
                (pl.col("updated_date") >= cutoff_prev)
                & (pl.col("updated_date") < cutoff_recent)
            )
            .group_by("item_id")
            .agg(pl.len().cast(pl.Float32).alias(f"i_pop_prev_{w}d"))
        )
        result = (
            result
            .join(recent,   on="item_id", how="left")
            .join(previous, on="item_id", how="left")
            .with_columns([
                pl.col(f"i_pop_last_{w}d").fill_null(0.0),
                pl.col(f"i_pop_prev_{w}d").fill_null(0.0),
            ])
            .with_columns([
                (pl.col(f"i_pop_last_{w}d") / (pl.col(f"i_pop_prev_{w}d") + 1e-6))
                .cast(pl.Float32).alias(f"i_trend_ratio_{w}d"),
            ])
        )

    # Trend acceleration: short-term vs medium-term momentum
    if 7 in windows and 30 in windows:
        result = result.with_columns([
            (pl.col("i_trend_ratio_7d") / (pl.col("i_trend_ratio_30d") + 1e-6))
            .cast(pl.Float32).alias("i_trend_acceleration"),
        ])

    # Monthly stats: volatility and seasonal score
    monthly = (
        trans_df
        .with_columns(pl.col("updated_date").dt.month().alias("_month"))
        .group_by(["item_id", "_month"])
        .agg(pl.len().cast(pl.Float32).alias("_mc"))
    )
    volatility = (
        monthly.group_by("item_id").agg([
            pl.col("_mc").std().fill_null(0.0).cast(pl.Float32).alias("i_volatility"),
            pl.col("_month").n_unique().cast(pl.Float32).alias("i_n_active_months"),
        ])
    )
    seasonal = (
        monthly.group_by("item_id").agg([
            pl.col("_mc").max().alias("_peak"),
            pl.col("_mc").mean().alias("_avg"),
        ])
        .with_columns([
            (pl.col("_peak") / (pl.col("_avg") + 1e-6)).cast(pl.Float32).alias("i_seasonal_score"),
        ])
        .select(["item_id", "i_seasonal_score"])
    )

    result = (
        result
        .join(volatility, on="item_id", how="left")
        .join(seasonal,   on="item_id", how="left")
    )

    # Fill remaining nulls
    for c in result.columns:
        if c != "item_id":
            result = result.with_columns(pl.col(c).fill_null(0.0))

    log.info(
        "Item trend features: %d items  %d features  [%.1fs]",
        len(result), result.width - 1, time.perf_counter() - t0,
    )
    return result
