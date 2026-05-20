"""
Generate final predictions and output the submission dictionary.

Loads the trained model + saved artifacts, then produces top-K recommendations
for every customer who was active in the target split (val or test).

Output
------
outputs/predictions/predictions_<split>.json
    {customer_id (str): [item_id, …], …}

Usage
-----
python predict.py                         # test split (month 12)
python predict.py --target-split val      # validation split (month 11)
python predict.py --model-type popularity --top-k 20
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle

from src.config import CHECKPOINTS_DIR, PREDICTIONS_DIR, TOP_K
from src.data.loader import load_transactions, split_transactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate recommendation predictions.")
    p.add_argument("--model-type", choices=["als", "popularity"], default="als")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument(
        "--target-split", choices=["val", "test"], default="test",
        help="Which split to predict for ('val' for debugging, 'test' for submission).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load artifacts ────────────────────────────────────────────────────────
    model_path     = CHECKPOINTS_DIR / f"{args.model_type}_model.pkl"
    artifacts_path = CHECKPOINTS_DIR / "artifacts.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}.\n"
            "Run  python train.py  first."
        )

    log.info("Loading model from %s …", model_path)
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    with open(artifacts_path, "rb") as fh:
        artifacts = pickle.load(fh)

    user2idx = artifacts["user2idx"]
    idx2item = artifacts["idx2item"]
    matrix   = artifacts["matrix"]

    # ── Determine target users ────────────────────────────────────────────────
    log.info("Loading transaction splits to identify target customers …")
    trans_train, trans_val, trans_test = split_transactions(load_transactions())

    target_df = trans_test if args.target_split == "test" else trans_val
    all_target = target_df["customer_id"].unique().to_list()
    known_target = [u for u in all_target if u in user2idx]

    log.info(
        "Target customers in month %s: %d total  |  %d with training history",
        "12 (test)" if args.target_split == "test" else "11 (val)",
        len(all_target), len(known_target),
    )

    # ── Generate recommendations ──────────────────────────────────────────────
    log.info("Generating top-%d recommendations …", args.top_k)
    predictions = model.recommend(
        user_ids=known_target,
        user2idx=user2idx,
        idx2item=idx2item,
        top_k=args.top_k,
        filter_purchased=matrix,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"predictions_{args.target_split}.json"

    # JSON requires string keys
    serializable = {str(k): v for k, v in predictions.items()}
    with open(out_path, "w") as fh:
        json.dump(serializable, fh)

    log.info("Saved %d customer predictions → %s", len(predictions), out_path)

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\nSample predictions (first 3 customers):")
    for uid, items in list(predictions.items())[:3]:
        print(f"  customer {uid:>10}: {items}")


if __name__ == "__main__":
    main()
