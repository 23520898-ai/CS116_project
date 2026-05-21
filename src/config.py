"""Central configuration: paths, split settings, and model hyperparameters."""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR        = PROJECT_ROOT          # .parquet files sit in the project root
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

TRANSACTION_FILE = DATA_DIR / "transaction_full_2025.parquet"
EVENT_FILE       = DATA_DIR / "event_full_2025.parquet"
ITEMS_FILE       = DATA_DIR / "items.parquet"

# ── Temporal split ────────────────────────────────────────────────────────────
TRAIN_MONTHS = list(range(1, 11))  
VAL_MONTH    = 11                  
TEST_MONTH   = 12                  
EVENT_YEAR   = 2025

# ── Recommendation settings ───────────────────────────────────────────────────
TOP_K = 10

# ── ALS hyperparameters ───────────────────────────────────────────────────────
ALS_FACTORS        = 128
ALS_ITERATIONS     = 20
ALS_REGULARIZATION = 0.01
ALS_ALPHA          = 40.0   # confidence scale:  C = 1 + alpha * r

# ─────────────────────────────────────────────────────────────────────────────
# TWO-STAGE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Candidate Generation ────────────────────────────────────────────
HISTORY_MAX_ITEMS  = 200    # items from user's own purchase history
COVISIT_TOP_K      = 50     # top-K co-occurring items kept per item in matrix
COVISIT_MAX_BILL   = 20     # cap bill size before covisit computation
COVISIT_CANDS      = 300    # covisitation candidates per user
W2V_CANDS          = 200    # Word2Vec candidates per user
W2V_RECENT         = 20     # how many recent items form the user query vector
MAX_CANDIDATES     = 1200   # total stage-1 candidates per user
STAGE1_BATCH_SIZE  = 2_000  # users per candidate-generation batch

# Word2Vec hyperparameters
W2V_VECTOR_SIZE = 64
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 3
W2V_EPOCHS      = 10
W2V_WORKERS     = max(1, (os.cpu_count() or 8) - 1)

# ── Stage 2: Reranking ────────────────────────────────────────────────────────
RANKER_N_ESTIMATORS  = 800
RANKER_LEARNING_RATE = 0.013173
RANKER_MAX_DEPTH     = 12
RANKER_NUM_LEAVES    = 138
RANKER_PREDICT_NUM_ITERATION = 800
RANKER_SUBSAMPLE     = 0.5039
RANKER_COLSAMPLE     = 0.8728
RANKER_OBJECTIVE     = "lambdarank"   # "lambdarank" (recommended) or "binary"
RANKER_MIN_CHILD_SAMPLES = 80        # lower than LGBM default (20) for ranking
RANKER_TOP_K_OUTPUT  = 10   # final output items per user

# Training sampling
RANKER_TRAIN_USERS = 100_000   # users sampled for ranker training
RANKER_NEG_RATIO   = 20        # negative samples per positive (for training)

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT FLAGS – defaults (can be overridden by CLI flags)
# ─────────────────────────────────────────────────────────────────────────────

# Improvement 2: Extended Labels
USE_EXTEND_LABELS       = False   # True → use grade-0/1/2 labels for LambdaRank
SOFT_LABEL_GRADE        = 1       # relevance grade for soft positives (same-cat)

# Improvement 3: Hard Negative Mining
USE_HARD_NEGATIVES      = False   # True → prioritise hard negs in sampling
HARD_NEG_FRACTION       = 0.3     # fraction of negative budget for hard negs

# Improvement 4: Temporal Decay Features
USE_TEMPORAL_FEATURES   = False   # True → add temporal-decay user/item features
TEMPORAL_DECAY_RATE     = 0.9     # w = decay_rate ^ (days / 30)

# Improvement 6: Session Features
USE_SESSION_FEATURES    = False   # True → add session-based user features
SESSION_GAP_HOURS       = 2.0     # time gap (hours) that defines a new session

# Improvement 7: Category Affinity with Temporal Decay
USE_CATEGORY_AFFINITY   = False   # True → add temporal category affinity features
CAT_AFFINITY_DECAY_RATE = 0.95    # decay rate for category affinity weights

# Improvement 8: Item Trend Features
USE_ITEM_TRENDS         = False   # True → add multi-window item trend features
ITEM_TREND_WINDOWS      = [7, 30, 90]   # trend windows in days

# Improvement 9: UI History Features
USE_UI_HISTORY          = False   # True → add user-item purchase history features

# Improvement 10: Ensemble
ENSEMBLE_SEEDS: list[int] = []   # e.g. [42, 123, 456] → train 3 models and ensemble
ENSEMBLE_METHOD          = "reciprocal_rank"   # "reciprocal_rank" | "borda_count"
