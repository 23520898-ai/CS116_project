#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# Recommendation Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
WORKERS="${WORKERS:-12}"
N_USERS="${N_USERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-3000}"
EVAL_USERS="${EVAL_USERS:-10000}"
RANKER_TYPE="${RANKER_TYPE:-lambdarank}"
NEG_RATIO="${NEG_RATIO:-20}"

# ── Feature Improvement Flags (set to 1 to enable) ────────────────────────────
EXTEND_LABELS="${EXTEND_LABELS:-0}"
USE_HARD_NEGATIVES="${USE_HARD_NEGATIVES:-0}"
USE_TEMPORAL="${USE_TEMPORAL:-0}"
USE_SESSION="${USE_SESSION:-0}"
USE_CATEGORY_AFFINITY="${USE_CATEGORY_AFFINITY:-0}"
USE_ITEM_TRENDS="${USE_ITEM_TRENDS:-0}"
USE_UI_HISTORY="${USE_UI_HISTORY:-0}"
ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-}"      # e.g. "42,123,456"
ENSEMBLE_METHOD="${ENSEMBLE_METHOD:-reciprocal_rank}"

MODE="full"
SKIP_STAGE1=false
EVAL_MODE=false

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --skip-stage1)           SKIP_STAGE1=true ;;
        --eval)                  EVAL_MODE=true ;;
        --extend-labels)         EXTEND_LABELS=1 ;;
        --use-hard-negatives)    USE_HARD_NEGATIVES=1 ;;
        --use-temporal)          USE_TEMPORAL=1 ;;
        --use-session)           USE_SESSION=1 ;;
        --use-category-affinity) USE_CATEGORY_AFFINITY=1 ;;
        --use-item-trends)       USE_ITEM_TRENDS=1 ;;
        --use-ui-history)        USE_UI_HISTORY=1 ;;
        --all-features)
            EXTEND_LABELS=1; USE_HARD_NEGATIVES=1; USE_TEMPORAL=1
            USE_SESSION=1; USE_CATEGORY_AFFINITY=1; USE_ITEM_TRENDS=1; USE_UI_HISTORY=1 ;;
        *)                       echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Print configuration ───────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo " Recommendation Pipeline"
echo "════════════════════════════════════════════════════════════════════"
echo " Workers:              $WORKERS"
echo " Train users:          $N_USERS"
echo " Batch size:           $BATCH_SIZE"
echo " Ranker type:          $RANKER_TYPE"
echo " Neg ratio:            $NEG_RATIO"
echo " Skip Stage1:          $SKIP_STAGE1"
echo " Eval mode:            $EVAL_MODE"
echo " --- Feature Improvements ---"
echo " Extend labels:        $EXTEND_LABELS"
echo " Hard negatives:       $USE_HARD_NEGATIVES"
echo " Temporal features:    $USE_TEMPORAL"
echo " Session features:     $USE_SESSION"
echo " Category affinity:    $USE_CATEGORY_AFFINITY"
echo " Item trends:          $USE_ITEM_TRENDS"
echo " UI history:           $USE_UI_HISTORY"
echo " Ensemble seeds:       ${ENSEMBLE_SEEDS:-none}"
echo "════════════════════════════════════════════════════════════════════"

# Check required data files
for f in transaction_full_2025.parquet event_full_2025.parquet items.parquet; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
done

echo "✓ All required data files found"

# ── Build train command ───────────────────────────────────────────────────────
TRAIN_CMD="uv run python train_twostage.py \
    --n-users $N_USERS \
    --stage1-batch-size $BATCH_SIZE \
    --w2v-workers $WORKERS \
    --ranker-type $RANKER_TYPE \
    --neg-ratio $NEG_RATIO \
    --eval-max-users $EVAL_USERS \
    --ensemble-method $ENSEMBLE_METHOD"

[[ "$EXTEND_LABELS"       == "1" ]] && TRAIN_CMD+=" --extend-labels"
[[ "$USE_HARD_NEGATIVES"  == "1" ]] && TRAIN_CMD+=" --use-hard-negatives"
[[ "$USE_TEMPORAL"        == "1" ]] && TRAIN_CMD+=" --use-temporal-features"
[[ "$USE_SESSION"         == "1" ]] && TRAIN_CMD+=" --use-session-features"
[[ "$USE_CATEGORY_AFFINITY" == "1" ]] && TRAIN_CMD+=" --use-category-affinity"
[[ "$USE_ITEM_TRENDS"     == "1" ]] && TRAIN_CMD+=" --use-item-trends"
[[ "$USE_UI_HISTORY"      == "1" ]] && TRAIN_CMD+=" --use-ui-history"
[[ -n "$ENSEMBLE_SEEDS"         ]] && TRAIN_CMD+=" --ensemble-seeds $ENSEMBLE_SEEDS"
$SKIP_STAGE1 && TRAIN_CMD+=" --skip-stage1"

# ── Train ─────────────────────────────────────────────────────────────────────
if $EVAL_MODE; then
    echo ""
    echo ">>> Training with validation evaluation..."
    $TRAIN_CMD
else
    echo ""
    echo ">>> Training for final prediction (Jan 2026)..."
    $TRAIN_CMD --final-2025 --no-eval

    # ── Validate training artifacts ───────────────────────────────────────────
    echo ""
    echo ">>> Validating training artifacts..."
    REQUIRED_ARTIFACTS=(
        "outputs/checkpoints/covisit.pkl"
        "outputs/checkpoints/lgbm_ranker.pkl"
        "outputs/checkpoints/items_df.pkl"
        "outputs/checkpoints/feature_columns.json"
    )

    ALL_FOUND=true
    for artifact in "${REQUIRED_ARTIFACTS[@]}"; do
        if [[ -f "$artifact" ]]; then
            echo "  ✓ $artifact"
        else
            echo "  ✗ MISSING: $artifact"
            ALL_FOUND=false
        fi
    done

    if [[ -f "outputs/checkpoints/w2v/w2v.model" ]]; then
        echo "  ✓ outputs/checkpoints/w2v/w2v.model"
    else
        echo "  ⚠ W2V model not found (may have been skipped)"
    fi

    if ! $ALL_FOUND; then
        echo "ERROR: Missing required training artifacts. Training may have failed."
        exit 1
    fi

    # ── Predict Jan 2026 ──────────────────────────────────────────────────────
    echo ""
    echo ">>> Generating January 2026 predictions..."
    uv run python predict_twostage.py \
        --target-split jan2026 \
        --batch-size $BATCH_SIZE \
        --workers $WORKERS \
        --save-every-batches 1

    # ── Validate predictions ──────────────────────────────────────────────────
    PRED_FILE="outputs/predictions/predictions_twostage_jan2026.json"
    if [[ -f "$PRED_FILE" ]]; then
        NUM_PREDS=$(python -c "import json; print(len(json.load(open('$PRED_FILE'))))")
        echo ""
        echo "✓ Predictions saved: $PRED_FILE"
        echo "  Users predicted: $NUM_PREDS"
    else
        echo "ERROR: Prediction file not found: $PRED_FILE"
        exit 1
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " Pipeline completed successfully!"
echo "════════════════════════════════════════════════════════════════════"