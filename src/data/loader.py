"""Data loading and temporal split utilities."""
from __future__ import annotations

import logging

import polars as pl

from src.config import (
    TRANSACTION_FILE, EVENT_FILE, ITEMS_FILE,
    TRAIN_MONTHS, VAL_MONTH, TEST_MONTH, EVENT_YEAR,
)

log = logging.getLogger(__name__)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_transactions() -> pl.LazyFrame:
    """Lazy-load the full transaction parquet."""
    return pl.scan_parquet(str(TRANSACTION_FILE))


def load_events() -> pl.LazyFrame:
    """Lazy-load the full event parquet."""
    return pl.scan_parquet(str(EVENT_FILE))


def load_items() -> pl.DataFrame:
    """Load the items metadata table (small enough to fit in memory)."""
    return pl.scan_parquet(str(ITEMS_FILE)).collect()


# ── Splitters ─────────────────────────────────────────────────────────────────

def split_transactions(
    lf: pl.LazyFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Temporal split of transaction data by the month of ``updated_date``.

    Returns
    -------
    train : months  1–10
    val   : month  11
    test  : month  12
    """
    if lf is None:
        lf = load_transactions()

    month = pl.col("updated_date").dt.month()
    train = lf.filter(month.is_between(TRAIN_MONTHS[0], TRAIN_MONTHS[-1])).collect()
    val   = lf.filter(month == VAL_MONTH).collect()
    test  = lf.filter(month == TEST_MONTH).collect()

    log.info(
        "Transaction split — train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


def split_events(
    lf: pl.LazyFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Temporal split of event data (year 2025 only) by the month of ``event_date``.

    Returns
    -------
    train : months  1–10
    val   : month  11
    test  : month  12
    """
    if lf is None:
        lf = load_events()

    lf_2025 = lf.filter(pl.col("event_date").dt.year() == EVENT_YEAR)
    month   = pl.col("event_date").dt.month()
    train   = lf_2025.filter(month.is_between(TRAIN_MONTHS[0], TRAIN_MONTHS[-1])).collect()
    val     = lf_2025.filter(month == VAL_MONTH).collect()
    test    = lf_2025.filter(month == TEST_MONTH).collect()

    log.info(
        "Event split — train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_split_stats(
    train: pl.DataFrame,
    val:   pl.DataFrame,
    test:  pl.DataFrame,
    label: str = "Dataset",
) -> None:
    """Print a summary table of the three splits."""
    total = len(train) + len(val) + len(test)
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    for name, df in [
        ("Train (months 1-10)", train),
        ("Val   (month  11)  ", val),
        ("Test  (month  12)  ", test),
    ]:
        pct = len(df) / total * 100
        print(
            f"  {name}: {len(df):>12,} rows  ({pct:5.1f}%)"
            f"  | {df['customer_id'].n_unique():>8,} customers"
            f"  | {df['item_id'].n_unique():>6,} items"
        )
    print(f"{'─' * 60}")
