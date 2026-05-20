"""
Lưới 1 – History Candidates
============================
Trả lại toàn bộ sản phẩm mà user đã từng mua trong giai đoạn train,
sắp xếp theo thứ tự mua gần nhất đến xa nhất (most-recent first).
"""
from __future__ import annotations

import polars as pl


def get_history_candidates(
    trans_df: pl.DataFrame,
    max_items: int = 200,
) -> dict[int, list[str]]:
    """
    Parameters
    ----------
    trans_df  : transaction DataFrame (customer_id, item_id, updated_date)
    max_items : maximum number of items returned per user

    Returns
    -------
    dict  customer_id → [item_id, ...]   (most-recently purchased first)
    """
    # Most-recent purchase date per (customer, item) pair
    recency = (
        trans_df
        .group_by(["customer_id", "item_id"])
        .agg(pl.col("updated_date").max().alias("last_date"))
    )
    # Per customer: sort items by recency descending, take top max_items
    df = (
        recency
        .group_by("customer_id")
        .agg(
            pl.col("item_id")
            .sort_by("last_date", descending=True)
            .head(max_items)
            .alias("items")
        )
    )
    return {row["customer_id"]: row["items"] for row in df.iter_rows(named=True)}
