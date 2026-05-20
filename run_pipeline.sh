#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end two-stage recommendation pipeline
# =============================================================================
#
# Usage:
#   bash run_pipeline.sh                  # full retrain + Jan-2026 prediction
#   bash run_pipeline.sh --skip-stage1    # reuse covisit/w2v, retrain ranker only
#   bash run_pipeline.sh --eval           # retrain on train-split, eval on val
#   bash run_pipeline.sh --target val     # predict & evaluate validation month
#
# Environment variable overrides (set before calling the script):
#   WORKERS=12         CPU threads for training / inference (default: 12)
#   N_USERS=100000     Training users for ranker (default: 100 000)
#   BATCH_SIZE=3000    Users per Stage-1 batch   (default: 3 000)
#   EVAL_USERS=10000   Users for val-set metrics (default: 10 000)
#   RANKER_TYPE=lambdarank   "lambdarank" or "binary"  (default: lambdarank)
#   NEG_RATIO=20       Negatives per positive in training (default: 20)
#
# Requirements:
#   - uv installed  (https://docs.astral.sh/uv/)  — OR — pip + virtualenv
#   - parquet data files in project root (see README §4)
#   - Run from project root:  bash run_pipeline.sh
# =============================================================================

set -euo pipefail

# ── Defaults (override via env vars) ─────────────────────────────────────────
WORKERS="${WORKERS:-12}"
N_USERS="${N_USERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-3000}"
EVAL_USERS="${EVAL_USERS:-10000}"
RANKER_TYPE="${RANKER_TYPE:-lambdarank}"
NEG_RATIO="${NEG_RATIO:-20}"

# ── Parse CLI flags ───────────────────────────────────────────────────────────
SKIP_STAGE1=0      # --skip-stage1  : reuse existing covisit + w2v artifacts
DO_EVAL=0          # --eval         : train on months 1-10, evaluate on month 11
TARGET_SPLIT="jan2026"  # --target <val|test|jan2026>

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-stage1)  SKIP_STAGE1=1 ;;
        --eval)         DO_EVAL=1 ;;
        --target)       shift; TARGET_SPLIT="$1" ;;
        --workers)      shift; WORKERS="$1" ;;
        --n-users)      shift; N_USERS="$1" ;;
        --batch-size)   shift; BATCH_SIZE="$1" ;;
        --ranker-type)  shift; RANKER_TYPE="$1" ;;
        --neg-ratio)    shift; NEG_RATIO="$1" ;;
        -h|--help)
            sed -n '2,30p' "$0" | grep '^#' | sed 's/^# \{0,2\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Detect Python runner ──────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    PY="uv run python"
elif [[ -f ".venv/Scripts/python" ]]; then
    PY=".venv/Scripts/python"
elif [[ -f ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="python"
fi
echo "[pipeline] Python runner: $PY"

# ── Helper: timestamped log line ──────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "────────────────────────────────────────────────────────────────"; }

# ── Banner ────────────────────────────────────────────────────────────────────
hr
log "Two-Stage Recommendation Pipeline"
log "  workers     = $WORKERS"
log "  n_users     = $N_USERS"
log "  batch_size  = $BATCH_SIZE"
log "  ranker_type = $RANKER_TYPE"
log "  neg_ratio   = $NEG_RATIO"
log "  target      = $TARGET_SPLIT"
[[ $SKIP_STAGE1 -eq 1 ]] && log "  mode        = skip-stage1 (reuse covisit/w2v)"
[[ $DO_EVAL -eq 1 ]]    && log "  mode        = eval on validation set"
hr

# =============================================================================
# STEP 1 – TRAIN
# =============================================================================
log "STEP 1/2 — Training"
hr

TRAIN_ARGS=(
    --n-users        "$N_USERS"
    --stage1-batch-size "$BATCH_SIZE"
    --w2v-workers    "$WORKERS"
    --ranker-type    "$RANKER_TYPE"
    --neg-ratio      "$NEG_RATIO"
)

if [[ $SKIP_STAGE1 -eq 1 ]]; then
    TRAIN_ARGS+=(--skip-stage1)
fi

if [[ $DO_EVAL -eq 1 ]]; then
    # Train on months 1-10, evaluate on month 11
    TRAIN_ARGS+=(--eval-max-users "$EVAL_USERS")
    log "Training mode: standard (months 1-10 → ranker → eval on month 11)"
else
    # Final mode: use all 12 months of 2025 for prediction
    TRAIN_ARGS+=(--final-2025 --no-eval)
    log "Training mode: final-2025 (months 1-11 history, month 12 labels)"
fi

log "Running: $PY train_twostage.py ${TRAIN_ARGS[*]}"
$PY train_twostage.py "${TRAIN_ARGS[@]}"

hr
log "STEP 1/2 — Training complete"
hr

# =============================================================================
# STEP 2 – PREDICT
# =============================================================================
log "STEP 2/2 — Prediction  (target-split=$TARGET_SPLIT)"
hr

PRED_ARGS=(
    --target-split   "$TARGET_SPLIT"
    --batch-size     "$BATCH_SIZE"
    --workers        "$WORKERS"
    --save-every-batches 1
)

if [[ "$TARGET_SPLIT" == "val" ]]; then
    PRED_ARGS+=(--quick-metrics)
    log "Validation split selected — metrics will be reported after prediction."
fi

log "Running: $PY predict_twostage.py ${PRED_ARGS[*]}"
$PY predict_twostage.py "${PRED_ARGS[@]}"

# =============================================================================
# DONE
# =============================================================================
hr
log "Pipeline finished."
log "Predictions saved to:  outputs/predictions/predictions_twostage_${TARGET_SPLIT}.json"
[[ "$TARGET_SPLIT" == "val" ]] && \
    log "Metrics saved to:      outputs/val_metrics_twostage.json"
log "Ranker checkpoint:     outputs/checkpoints/lgbm_ranker.pkl"
hr
