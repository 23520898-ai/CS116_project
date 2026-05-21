"""User-side features computed from the training transaction history."""
from __future__ import annotations

import logging
import math
import time

import polars as pl

log = logging.getLogger(__name__)

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


# ── Improvement 4: Temporal Decay Features ────────────────────────────────────

def build_temporal_decay_features(
    trans_df: pl.DataFrame,
    decay_rate: float = 0.9,
    ref_date=None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build user and item features with exponential time decay.

    Weight  w = decay_rate ^ (days_since_purchase / 30)

    USER features returned
    ----------------------
    u_weighted_purchase_count   : sum of decayed weights (recency-adjusted activity)
    u_purchase_momentum_30d     : purchases last 30d / prior 30d (> 1 = increasing)
    u_recent_unique_items       : unique items with decay weight > 0.7
    u_days_since_last           : days from latest purchase to ref_date
    u_purchases_per_day         : transactions / active-span days
    u_weekend_ratio             : fraction of purchases on weekend
    u_purchase_recency_score    : mean decay weight (overall freshness)

    ITEM features returned
    ----------------------
    i_weighted_popularity       : sum of decayed purchase weights
    i_trend_60d                 : purchases last 60d / prior 60d (> 1 = trending)
    i_days_since_last_sale      : days since most recent sale
    i_month_diversity           : number of distinct months item was sold

    Why effective
    -------------
    Recommendations should favor items matching the user's *recent* preferences.
    Items gaining popularity recently are more likely to be purchased again.

    Returns
    -------
    (user_temporal_df, item_temporal_df) – join on customer_id / item_id.
    """
    t0 = time.perf_counter()

    ref = trans_df["updated_date"].max() if ref_date is None else ref_date
    # Decay coefficient: ln(decay_rate) / 30 so weight = exp(coeff * days_ago)
    ln_decay = math.log(max(1e-9, decay_rate))
    coeff = ln_decay / 30.0

    df = (
        trans_df.with_columns([
            (pl.lit(ref) - pl.col("updated_date"))
            .dt.total_days()
            .cast(pl.Float32)
            .alias("_days_ago"),
        ])
        .with_columns([
            # w = decay_rate^(days/30) = exp(ln(decay_rate) * days / 30)
            (pl.col("_days_ago") * coeff).exp().cast(pl.Float32).alias("_w"),
            # Weekend flag: weekday() returns Mon=0 … Sun=6
            (pl.col("updated_date").dt.weekday() >= 5).cast(pl.Int8).alias("_is_weekend"),
        ])
    )

    # ── User temporal aggregations ────────────────────────────────────────────
    user_temp = df.group_by("customer_id").agg([
        pl.col("_w").sum().cast(pl.Float32).alias("u_weighted_purchase_count"),
        pl.col("_days_ago").min().cast(pl.Float32).alias("u_days_since_last"),
        pl.col("_is_weekend").mean().cast(pl.Float32).alias("u_weekend_ratio"),
        pl.len().cast(pl.Float32).alias("_u_total"),
        (pl.col("_days_ago").max() - pl.col("_days_ago").min() + 1).alias("_u_span"),
    ])

    user_temp = user_temp.with_columns([
        (pl.col("_u_total") / (pl.col("_u_span") + 1.0)).cast(pl.Float32).alias("u_purchases_per_day"),
        (pl.col("u_weighted_purchase_count") / (pl.col("_u_total") + 1e-6))
        .cast(pl.Float32).alias("u_purchase_recency_score"),
    ]).drop(["_u_total", "_u_span"])

    # Purchase momentum: last-30d count / prev-30d count
    cutoff_30 = ref - pl.duration(days=30)
    cutoff_60 = ref - pl.duration(days=60)

    cnt_30 = (
        trans_df.filter(pl.col("updated_date") >= cutoff_30)
        .group_by("customer_id")
        .agg(pl.len().cast(pl.Float32).alias("_cnt_30"))
    )
    cnt_prev = (
        trans_df.filter(
            (pl.col("updated_date") >= cutoff_60) & (pl.col("updated_date") < cutoff_30)
        ).group_by("customer_id")
        .agg(pl.len().cast(pl.Float32).alias("_cnt_prev"))
    )

    user_temp = (
        user_temp
        .join(cnt_30, on="customer_id", how="left")
        .join(cnt_prev, on="customer_id", how="left")
        .with_columns([
            pl.col("_cnt_30").fill_null(0.0),
            pl.col("_cnt_prev").fill_null(0.0),
        ])
        .with_columns([
            (pl.col("_cnt_30") / (pl.col("_cnt_prev") + 1e-6))
            .cast(pl.Float32).alias("u_purchase_momentum_30d"),
        ])
        .drop(["_cnt_30", "_cnt_prev"])
    )

    # Recent unique items (weight > 0.7 ≈ purchased within ~10 days)
    recent_unique = (
        df.filter(pl.col("_w") > 0.7)
        .group_by("customer_id")
        .agg(pl.col("item_id").n_unique().cast(pl.Float32).alias("u_recent_unique_items"))
    )
    user_temp = (
        user_temp
        .join(recent_unique, on="customer_id", how="left")
        .with_columns(pl.col("u_recent_unique_items").fill_null(0.0))
    )

    # ── Item temporal aggregations ────────────────────────────────────────────
    item_temp = df.group_by("item_id").agg([
        pl.col("_w").sum().cast(pl.Float32).alias("i_weighted_popularity"),
        pl.col("_days_ago").min().cast(pl.Float32).alias("i_days_since_last_sale"),
        pl.col("updated_date").dt.month().n_unique().cast(pl.Float32).alias("i_month_diversity"),
    ])

    cutoff_60d = ref - pl.duration(days=60)
    cutoff_120d = ref - pl.duration(days=120)

    i_cnt_60 = (
        trans_df.filter(pl.col("updated_date") >= cutoff_60d)
        .group_by("item_id").agg(pl.len().cast(pl.Float32).alias("_i_60"))
    )
    i_cnt_prev = (
        trans_df.filter(
            (pl.col("updated_date") >= cutoff_120d) & (pl.col("updated_date") < cutoff_60d)
        ).group_by("item_id").agg(pl.len().cast(pl.Float32).alias("_i_prev"))
    )

    item_temp = (
        item_temp
        .join(i_cnt_60, on="item_id", how="left")
        .join(i_cnt_prev, on="item_id", how="left")
        .with_columns([
            pl.col("_i_60").fill_null(0.0),
            pl.col("_i_prev").fill_null(0.0),
        ])
        .with_columns([
            (pl.col("_i_60") / (pl.col("_i_prev") + 1e-6)).cast(pl.Float32).alias("i_trend_60d"),
        ])
        .drop(["_i_60", "_i_prev"])
    )

    elapsed = time.perf_counter() - t0
    log.info(
        "Temporal decay features: %d users (%d cols) | %d items (%d cols)  [%.1fs]",
        len(user_temp), user_temp.width - 1,
        len(item_temp), item_temp.width - 1,
        elapsed,
    )
    return user_temp, item_temp


# ── Improvement 6: Session-Based Features ─────────────────────────────────────

def build_session_features(
    trans_df: pl.DataFrame,
    session_gap_hours: float = 2.0,
) -> pl.DataFrame:
    """
    Group transactions into shopping sessions (gap > session_gap_hours = new session).

    USER features returned
    ----------------------
    u_avg_basket_size       : mean items per session
    u_max_basket_size       : max items in a single session
    u_avg_spend_per_session : mean session spend
    u_total_sessions        : total number of sessions
    u_avg_session_duration  : mean session duration in minutes
    u_session_frequency     : sessions per day over user's active span

    Why effective
    -------------
    Basket size predicts how many items to recommend.
    Session frequency separates heavy shoppers from casual buyers.
    Heavy shoppers respond better to diverse recommendations.

    Returns
    -------
    DataFrame with one row per customer_id and session feature columns.
    """
    t0 = time.perf_counter()
    gap_secs = int(session_gap_hours * 3600)

    # Sort by (user, time) so shift(1).over(user) gives within-user previous time
    df_sorted = (
        trans_df
        .select(["customer_id", "item_id", "updated_date", "quantity", "price"])
        .sort(["customer_id", "updated_date"])
    )

    df_sessions = (
        df_sorted
        .with_columns([
            pl.col("updated_date").shift(1).over("customer_id").alias("_prev_time"),
        ])
        .with_columns([
            (pl.col("updated_date") - pl.col("_prev_time"))
            .dt.total_seconds()
            .fill_null(0)
            .alias("_gap_sec"),
        ])
        .with_columns([
            (pl.col("_gap_sec") > gap_secs).cast(pl.Int32).alias("_new_session"),
        ])
        .with_columns([
            pl.col("_new_session").cum_sum().over("customer_id").alias("_session_id"),
        ])
    )

    # Session-level aggregations
    session_agg = (
        df_sessions.group_by(["customer_id", "_session_id"])
        .agg([
            pl.len().cast(pl.Int32).alias("_basket_size"),
            (pl.col("price").cast(pl.Float64) * pl.col("quantity").cast(pl.Float64))
            .sum().alias("_session_spend"),
            pl.col("updated_date").min().alias("_sess_start"),
            pl.col("updated_date").max().alias("_sess_end"),
        ])
        .with_columns([
            (pl.col("_sess_end") - pl.col("_sess_start"))
            .dt.total_minutes()
            .cast(pl.Float32)
            .alias("_duration_min"),
        ])
    )

    # User-level aggregations over sessions
    user_span = (
        trans_df.group_by("customer_id").agg([
            ((pl.col("updated_date").max() - pl.col("updated_date").min())
             .dt.total_days().cast(pl.Float32) + 1).alias("_span_days"),
        ])
    )

    user_sess = (
        session_agg.group_by("customer_id").agg([
            pl.col("_basket_size").mean().cast(pl.Float32).alias("u_avg_basket_size"),
            pl.col("_basket_size").max().cast(pl.Float32).alias("u_max_basket_size"),
            pl.col("_session_spend").mean().cast(pl.Float32).alias("u_avg_spend_per_session"),
            pl.len().cast(pl.Int32).alias("u_total_sessions"),
            pl.col("_duration_min").mean().cast(pl.Float32).alias("u_avg_session_duration"),
        ])
        .join(user_span, on="customer_id", how="left")
        .with_columns([
            (pl.col("u_total_sessions").cast(pl.Float32) / (pl.col("_span_days") + 1e-6))
            .cast(pl.Float32).alias("u_session_frequency"),
        ])
        .drop("_span_days")
    )

    log.info(
        "Session features: %d users  %d features  [%.1fs]",
        len(user_sess), user_sess.width - 1, time.perf_counter() - t0,
    )
    return user_sess
