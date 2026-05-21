"""
User x Item cross features (interaction features).
====================================================
These features capture the relationship between a specific user and a
specific candidate item, combining signals from all three Stage-1 sources.
"""
from __future__ import annotations

import logging
import math
import time

import polars as pl

log = logging.getLogger(__name__)


def precompute_cross_stats(
    trans_df:  pl.DataFrame,
    items_df:  pl.DataFrame | None = None,
) -> dict:
    """
    Pre-compute the expensive aggregations that ``build_cross_features`` would
    otherwise re-run on the full transaction table for every batch.

    Call this once before any prediction/evaluation loop and pass the returned
    dict as ``pre_computed`` to ``build_cross_features``.

    Returns
    -------
    dict with keys:
        "ui_stats"       - (customer_id, item_id) purchase stats DataFrame
        "item_cats"      - item_id -> (category_l1, category_l2)
        "user_cat1_cnt"  - (customer_id, category_l1) -> purchase count
        "user_cat2_cnt"  - (customer_id, category_l2) -> purchase count
        "user_total_cnt" - customer_id -> total purchase count
    """
    ref = trans_df["updated_date"].max()

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

    result: dict = {"ui_stats": ui_stats}

    if items_df is not None and "category_l1" in items_df.columns:
        item_cats = items_df.select(["item_id", "category_l1", "category_l2"])
        trans_cats = (
            trans_df.select(["customer_id", "item_id"])
            .join(item_cats, on="item_id", how="left")
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
        result["item_cats"] = item_cats
        result["user_cat1_cnt"] = user_cat1_cnt
        result["user_cat2_cnt"] = user_cat2_cnt
        result["user_total_cnt"] = user_total_cnt

    return result


def build_cross_features(
    candidates_df:   pl.DataFrame,
    trans_df:        pl.DataFrame | None = None,
    covisit_scores:  dict[int, dict[str, float]] | None = None,
    w2v_scores:      dict[int, dict[str, float]] | None = None,
    items_df:        pl.DataFrame | None = None,
    pre_computed:    dict | None = None,
) -> pl.DataFrame:
    """
    Build cross features for a DataFrame of (customer_id, item_id) candidate
    pairs.

    Parameters
    ----------
    candidates_df  : DataFrame with columns [customer_id, item_id,
                     from_history (int8), from_covisit (int8),
                     from_w2v (int8), stage1_rank (int32)]
    trans_df       : training transactions (used when pre_computed is None)
    covisit_scores : {customer_id: {item_id: score}}  (from stage-1)
    w2v_scores     : {customer_id: {item_id: cosine_sim}}  (from stage-1)
    items_df       : items metadata (optional, used when pre_computed is None)
    pre_computed   : dict returned by precompute_cross_stats().  Pass this to
                     avoid recomputing expensive aggregations every batch.

    Returns
    -------
    candidates_df with additional cross feature columns
    """
    # Unique users in this batch – used to slice all large precomputed tables
    # so hash-join right-side is small (hundreds of rows) instead of millions.
    _batch_users = candidates_df["customer_id"].unique()

    # ---- Resolve pre-computed vs. on-the-fly tables -------------------------
    if pre_computed is not None:
        # Slice to batch users only → tiny hash table for the join
        ui_stats = pre_computed["ui_stats"].filter(
            pl.col("customer_id").is_in(_batch_users)
        )
        _use_precomputed_cats = True
    else:
        assert trans_df is not None, \
            "Either trans_df or pre_computed must be provided."
        ref = trans_df["updated_date"].max()
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
        _use_precomputed_cats = False

    # ---- User-item interaction stats ----------------------------------------
    out = candidates_df.join(ui_stats, on=["customer_id", "item_id"], how="left")
    out = out.with_columns([
        pl.col("ui_in_history").fill_null(0).cast(pl.Int8),
        pl.col("ui_history_count").fill_null(0),
        pl.col("ui_history_last_days").fill_null(9999.0),
    ])

    # ---- Covisitation scores ------------------------------------------------
    # If candidates_df already has ui_covisit_score (embedded by
    # candidates_to_dataframe), use it directly – skip dict→DataFrame + join.
    if "ui_covisit_score" not in out.columns:
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
            out = out.with_columns(pl.lit(0.0).cast(pl.Float32).alias("ui_covisit_score"))
    out = out.with_columns(pl.col("ui_covisit_score").fill_null(0.0))

    # ---- Word2Vec similarity scores -----------------------------------------
    # Same: skip dict→DataFrame + join if scores already embedded.
    if "ui_w2v_score" not in out.columns:
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
            out = out.with_columns(pl.lit(0.0).cast(pl.Float32).alias("ui_w2v_score"))
    out = out.with_columns(pl.col("ui_w2v_score").fill_null(0.0))

    # ---- User x Category affinity features ----------------------------------
    if _use_precomputed_cats and "item_cats" in pre_computed:
        item_cats      = pre_computed["item_cats"]          # item-keyed – already small
        # Slice user-keyed tables to this batch's users
        user_cat1_cnt  = pre_computed["user_cat1_cnt"].filter(
            pl.col("customer_id").is_in(_batch_users)
        )
        user_cat2_cnt  = pre_computed["user_cat2_cnt"].filter(
            pl.col("customer_id").is_in(_batch_users)
        )
        user_total_cnt = pre_computed["user_total_cnt"].filter(
            pl.col("customer_id").is_in(_batch_users)
        )

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
    elif items_df is not None and "category_l1" in items_df.columns:
        # Fallback: compute on the fly
        item_cats = items_df.select(["item_id", "category_l1", "category_l2"])
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

    # ── Optional: category affinity cross features ────────────────────────────
    # Injected via pre_computed["cross_cat_feat"] = (customer_id, category_l2,
    # ui_cat_affinity_score, ui_cat_affinity_rank) from build_category_affinity_temporal
    if pre_computed is not None and "cross_cat_feat" in pre_computed:
        cross_cat = pre_computed["cross_cat_feat"].filter(
            pl.col("customer_id").is_in(_batch_users)
        )
        item_cats_for_cross = (
            pre_computed.get("item_cats")
            if pre_computed.get("item_cats") is not None
            else None
        )
        if item_cats_for_cross is not None and "category_l2" in item_cats_for_cross.columns:
            # Join candidates → item category → cross affinity
            out = (
                out
                .join(item_cats_for_cross.select(["item_id", "category_l2"]),
                      on="item_id", how="left")
                .join(cross_cat, on=["customer_id", "category_l2"], how="left")
                .with_columns([
                    pl.col("ui_cat_affinity_score").fill_null(0.0),
                    pl.col("ui_cat_affinity_rank").fill_null(9999.0),
                    (pl.col("ui_cat_affinity_rank") == 1.0).cast(pl.Int8).alias("ui_is_top_category"),
                ])
                .drop("category_l2")
            )

    # ── Optional: user-item history cross features ────────────────────────────
    # Injected via pre_computed["ui_history_feat"]
    if pre_computed is not None and "ui_history_feat" in pre_computed:
        ui_hist = pre_computed["ui_history_feat"].filter(
            pl.col("customer_id").is_in(_batch_users)
        )
        out = out.join(ui_hist, on=["customer_id", "item_id"], how="left")
        # Fill history features for unseen pairs
        for col in ["ui_times_purchased", "ui_last_purchase_days", "ui_purchase_interval",
                    "ui_is_repurchase", "ui_quantity_total", "ui_avg_quantity_per_purchase",
                    "ui_discount_usage"]:
            if col in out.columns:
                fill_val = 0 if col == "ui_is_repurchase" else (9999.0 if "days" in col else 0.0)
                out = out.with_columns(pl.col(col).fill_null(fill_val))

    return out


# ── Improvement 7: Category Affinity with Temporal Decay ─────────────────────

def build_category_affinity_temporal(
    trans_df: pl.DataFrame,
    items_df: pl.DataFrame,
    decay_rate: float = 0.95,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute user–category affinity scores with exponential time decay.

    USER features (one row per customer_id)
    ----------------------------------------
    u_cat_l1_diversity     : distinct L1 categories purchased
    u_cat_diversity        : distinct L2 categories purchased
    u_top_cat_concentration: share of purchases in user's top L2 category  (0-1)
    u_cat_switch_rate      : fraction of transitions that cross L2 categories

    CROSS features (one row per customer_id × category_l2)
    -------------------------------------------------------
    ui_cat_affinity_score  : decay-weighted purchase count in this category
    ui_cat_affinity_rank   : rank of this category for the user (1 = top)

    Why effective
    -------------
    Category preferences shift over time (seasonal, trend-driven).
    Decay-weighted affinity captures *current* category preference better than
    raw counts.  Category-switch rate separates explorers from specialists.

    Returns
    -------
    (user_cat_affinity_df, cross_cat_features_df)
    """
    t0 = time.perf_counter()

    if "category_l2" not in items_df.columns:
        log.warning("items_df missing category_l2 – skipping category affinity features")
        return pl.DataFrame(), pl.DataFrame()

    ref = trans_df["updated_date"].max()
    ln_decay = math.log(max(1e-9, decay_rate))
    coeff = ln_decay / 30.0

    item_cats = items_df.select(["item_id", "category_l1", "category_l2"])

    df = (
        trans_df.select(["customer_id", "item_id", "updated_date"])
        .join(item_cats, on="item_id", how="left")
        .with_columns([
            (pl.lit(ref) - pl.col("updated_date"))
            .dt.total_days().cast(pl.Float32).alias("_days_ago"),
        ])
        .with_columns([
            (pl.col("_days_ago") * coeff).exp().cast(pl.Float32).alias("_w"),
        ])
    )

    # ── User-level category diversity stats ───────────────────────────────────
    user_cat_stats = df.group_by("customer_id").agg([
        pl.col("category_l1").n_unique().cast(pl.Float32).alias("u_cat_l1_diversity"),
        pl.col("category_l2").n_unique().cast(pl.Float32).alias("u_cat_diversity"),
    ])

    # Top category concentration: max category_l2 weighted share
    cat2_w = (
        df.filter(pl.col("category_l2").is_not_null())
        .group_by(["customer_id", "category_l2"])
        .agg(pl.col("_w").sum().alias("_c2w"))
    )
    user_total_w = (
        df.group_by("customer_id")
        .agg(pl.col("_w").sum().alias("_utw"))
    )
    top_conc = (
        cat2_w
        .join(user_total_w, on="customer_id", how="left")
        .with_columns((pl.col("_c2w") / (pl.col("_utw") + 1e-6)).alias("_pct"))
        .group_by("customer_id")
        .agg(pl.col("_pct").max().cast(pl.Float32).alias("u_top_cat_concentration"))
    )
    user_cat_stats = user_cat_stats.join(top_conc, on="customer_id", how="left")

    # Category switch rate
    cat_seq = (
        trans_df.select(["customer_id", "item_id", "updated_date"])
        .sort(["customer_id", "updated_date"])
        .join(item_cats.select(["item_id", "category_l2"]), on="item_id", how="left")
        .with_columns([
            pl.col("category_l2").shift(1).over("customer_id").alias("_prev_cat"),
        ])
        .with_columns([
            (
                (pl.col("category_l2") != pl.col("_prev_cat"))
                & pl.col("_prev_cat").is_not_null()
            ).cast(pl.Float32).alias("_is_switch"),
        ])
        .group_by("customer_id")
        .agg([
            pl.col("_is_switch").sum().alias("_n_sw"),
            pl.len().cast(pl.Float32).alias("_n_t"),
        ])
        .with_columns(
            (pl.col("_n_sw") / (pl.col("_n_t") + 1e-6)).cast(pl.Float32).alias("u_cat_switch_rate")
        )
        .select(["customer_id", "u_cat_switch_rate"])
    )
    user_cat_stats = user_cat_stats.join(cat_seq, on="customer_id", how="left")

    # ── Cross features: (customer_id, category_l2) affinity ──────────────────
    cross_cat = (
        cat2_w
        .join(user_total_w, on="customer_id", how="left")
        .with_columns([
            (pl.col("_c2w") / (pl.col("_utw") + 1e-6))
            .cast(pl.Float32).alias("ui_cat_affinity_score"),
        ])
        .select(["customer_id", "category_l2", "ui_cat_affinity_score"])
        .with_columns([
            pl.col("ui_cat_affinity_score")
            .rank(method="ordinal", descending=True)
            .over("customer_id")
            .cast(pl.Float32)
            .alias("ui_cat_affinity_rank"),
        ])
    )

    log.info(
        "Category affinity: %d users (%d user-cols) | %d user×cat pairs  [%.1fs]",
        len(user_cat_stats), user_cat_stats.width - 1,
        len(cross_cat), time.perf_counter() - t0,
    )
    return user_cat_stats, cross_cat


# ── Improvement 9: User-Item Interaction History Features ────────────────────

def build_user_item_history_features(
    trans_df: pl.DataFrame,
    items_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Build history features for each (user, item) pair seen in trans_df.

    Columns returned
    ----------------
    ui_times_purchased           : how many times user bought this item
    ui_last_purchase_days        : days since most recent purchase
    ui_purchase_interval         : mean days between consecutive purchases
    ui_is_repurchase             : 1 if purchased more than once
    ui_quantity_total            : total units bought
    ui_avg_quantity_per_purchase : mean units per transaction
    ui_discount_usage            : fraction of purchases made with discount > 0

    Why effective
    -------------
    Repurchase patterns are among the strongest signals for consumable goods.
    Purchase interval predicts *when* a user will buy again, enabling timely
    recommendations.  Discount sensitivity affects buy probability.

    Returns
    -------
    DataFrame keyed by (customer_id, item_id) with history feature columns.
    """
    t0 = time.perf_counter()
    ref = trans_df["updated_date"].max()

    ui_hist = (
        trans_df.group_by(["customer_id", "item_id"]).agg([
            pl.len().cast(pl.Float32).alias("ui_times_purchased"),
            (pl.lit(ref) - pl.col("updated_date").max())
            .dt.total_days().cast(pl.Float32).alias("ui_last_purchase_days"),
            pl.col("quantity").cast(pl.Float32).sum().alias("ui_quantity_total"),
            pl.col("quantity").cast(pl.Float32).mean().alias("ui_avg_quantity_per_purchase"),
            pl.col("updated_date").min().alias("_first"),
            pl.col("updated_date").max().alias("_last"),
        ])
        .with_columns([
            (pl.col("ui_times_purchased") > 1).cast(pl.Int8).alias("ui_is_repurchase"),
            (
                (pl.col("_last") - pl.col("_first")).dt.total_days().cast(pl.Float32)
                / (pl.col("ui_times_purchased") - 1 + 1e-6)
            ).alias("ui_purchase_interval"),
        ])
        .drop(["_first", "_last"])
    )

    if "discount" in trans_df.columns:
        disc = (
            trans_df.group_by(["customer_id", "item_id"]).agg([
                (pl.col("discount").cast(pl.Float32) > 0)
                .mean().cast(pl.Float32).alias("ui_discount_usage"),
            ])
        )
        ui_hist = ui_hist.join(disc, on=["customer_id", "item_id"], how="left")
    else:
        ui_hist = ui_hist.with_columns(pl.lit(0.0).cast(pl.Float32).alias("ui_discount_usage"))

    log.info(
        "UI history features: %d pairs  %d features  [%.1fs]",
        len(ui_hist), ui_hist.width - 2, time.perf_counter() - t0,
    )
    return ui_hist
