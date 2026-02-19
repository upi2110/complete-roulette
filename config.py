"""
Configuration constants for AI European Roulette Prediction System.
Single source of truth for all tunable parameters.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Bankroll Settings ───────────────────────────────────────────────
INITIAL_BANKROLL = 4000.0
BASE_BET = 7.50                     # THE ONE unit size — $7.50 per unit
SESSION_TARGET = 999999.0           # Effectively disabled — no profit cap, let winners run
STOP_LOSS_THRESHOLD = 3500.0        # Stop at $500 loss ($4000 - $500 = $3500)
MAX_CONSECUTIVE_LOSSES = 99         # DISABLED — never block betting due to loss streak
KELLY_FRACTION = 0.25               # (Legacy — kept for compatibility)
MAX_BET_MULTIPLIER = 15.0           # Sequence peak ×15 = $112.50 per number
RECOVERY_BET_FACTOR = 0.75          # (Legacy — kept for compatibility)

# ─── Sequence Betting Strategy (THE ONE) ────────────────────────────
# 1-3-2-6 variant: multipliers applied to BASE_BET ($7.50 unit).
# On WIN: advance to next step. On LOSS: reset to step 0.
# Marathon winner: $102.92 avg, P/R 0.152, tested on 1.57B sessions.
# Bet amounts at each step: $7.50, $45, $97.50, $112.50, $67.50
BET_SEQUENCE = [1, 6, 13, 15, 9]    # Multipliers × BASE_BET
MIN_BET = 1.0                       # Floor: never go below $1 per number
SESSION_TARGET_ENABLED = False       # No profit cap — let winners run

# ─── Confidence Thresholds (THE ONE) ─────────────────────────────────
# Only bet when AI confidence >= 65 AND should_bet = True.
# Marathon proven: patience is the edge. Skip ~57 of 60 spins.
CONFIDENCE_BET_THRESHOLD = 65.0     # Only bet when AI confidence >= 65%
CONFIDENCE_HIGH_THRESHOLD = 65.0    # Same threshold — no tiered behavior
CONFIDENCE_STRAIGHT_THRESHOLD = 65.0 # Consistent gating
FORCED_WAIT_SPINS = 0               # No forced waits
WAIT_AFTER_LOSS_SPINS = 0           # No cooldown after losses

# ─── Confidence Weights ──────────────────────────────────────────────
# Balanced 6-factor weighting. Sample size is reliably high (97/100 with 2000+ spins)
# and provides a stable baseline that keeps confidence achievable.
# Pattern strength provides a second stable anchor (~57).
# Performance factors (hit_rate, momentum, accuracy) modulate around the stable base.
WEIGHT_MODEL_AGREEMENT = 0.10       # Models rarely agree in roulette — low signal
WEIGHT_HISTORICAL_ACCURACY = 0.20   # Tracks color/dozen/number accuracy
WEIGHT_PATTERN_STRENGTH = 0.15      # Stable high score (~57) — anchors confidence
WEIGHT_SAMPLE_SIZE = 0.15           # Stable high score (~97) — anchors confidence
WEIGHT_STREAK_MOMENTUM = 0.15       # Recent win/loss streak — direct performance
WEIGHT_RECENT_HIT_RATE = 0.25       # Actual top-12 hit rate — most direct signal

# ─── ML Hyperparameters ──────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 192              # GRU hidden dimension (was 128)
LSTM_NUM_LAYERS = 2                 # GRU depth
LSTM_DROPOUT = 0.3                  # Dropout rate
LSTM_SEQUENCE_LENGTH = 30           # Context window for neural net (was 20)
LSTM_LEARNING_RATE = 0.001          # Initial learning rate
LSTM_EPOCHS = 100                   # Max epochs (early stopping will limit)
LSTM_BATCH_SIZE = 32                # Mini-batch size
LSTM_MIN_TRAINING_SPINS = 40       # Minimum spins before LSTM can train
RETRAIN_INTERVAL = 50               # Retrain every N new spins

# ─── Feature Engineering ─────────────────────────────────────────────
FEATURE_DIM = 85                    # Feature vector dimension per timestep
FEATURE_SEQUENCE_LENGTH = 30        # Same as LSTM_SEQUENCE_LENGTH

# ─── Transformer-GRU Hybrid ─────────────────────────────────────────
ATTENTION_HEADS = 4                 # Multi-head attention heads
LSTM_LABEL_SMOOTHING = 0.1          # Prevents overconfident predictions
EARLY_STOPPING_PATIENCE = 10        # Stop if val loss stalls for N epochs
LR_SCHEDULE_PATIENCE = 5            # Reduce LR after N epochs without improvement
LR_SCHEDULE_FACTOR = 0.5            # Multiply LR by this on plateau
VALIDATION_SPLIT = 0.15             # Hold out last 15% for validation

# ─── Adaptive Ensemble ───────────────────────────────────────────────
ADAPTIVE_WINDOW = 50                # Rolling window for performance tracking
ADAPTIVE_TEMPERATURE = 1.0          # Softmax temperature (lower = sharper discrimination)
ADAPTIVE_MIN_WEIGHT = 0.05          # Floor: no model drops below 5%
ADAPTIVE_MIN_OBSERVATIONS = 100     # Models must prove themselves over 100 predictions first

# ─── Confidence Calibration ──────────────────────────────────────────
CALIBRATION_BINS = 6                # Number of confidence bins for calibration
CALIBRATION_MIN_SAMPLES = 10        # Min samples per bin before calibrating

# ─── Time-Weighted Frequencies ───────────────────────────────────────
FREQUENCY_DECAY_FACTOR = 0.998      # Per-spin decay (half-life ~350 spins)
FREQUENCY_FLAT_WEIGHT = 0.50        # Backtest-optimized: 50/50 blend with window=30 gives +1.71% edge
FREQUENCY_RECENT_WEIGHT = 0.50      # Recent 30 spins captures short-term patterns — was 0.20
FREQUENCY_RECENT_WINDOW = 30        # Last 30 spins for recent frequency — sweet spot from backtest

# ─── Loss-Reactive Strategy ────────────────────────────────────────────
# After consecutive misses, diversify predictions to explore new sectors.
# Prevents AI from stubbornly predicting same failing numbers.
CONSECUTIVE_MISS_THRESHOLD = 3      # Diversify after 3 consecutive misses
EXPLORATION_BOOST_FACTOR = 0.3      # 30% exploration injection at threshold
SECTOR_COOLDOWN_WINDOW = 5          # Track misses over last N predictions

# ─── Regime Detection ────────────────────────────────────────────────
REGIME_RECENT_WINDOW = 50           # Recent spins for regime comparison
REGIME_BASELINE_WINDOW = 200        # Baseline spins for comparison
REGIME_KL_THRESHOLD = 0.15          # KL divergence threshold for regime change

# ─── Markov Chain ─────────────────────────────────────────────────────
MARKOV_ORDER_1_WEIGHT = 0.6
MARKOV_ORDER_2_WEIGHT = 0.4

# ─── Pattern Detection ────────────────────────────────────────────────
MIN_SPINS_FOR_PATTERNS = 10
HOT_NUMBER_THRESHOLD = 1.5          # Times above expected frequency
COLD_NUMBER_THRESHOLD = 0.5         # Times below expected frequency
SECTOR_SIZE = 5                     # Numbers per sector on wheel

# ─── European Roulette Wheel Layout ──────────────────────────────────
# Physical wheel order (clockwise from 0)
WHEEL_ORDER = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36,
    11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9,
    22, 18, 29, 7, 28, 12, 35, 3, 26
]

# Number properties
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
GREEN_NUMBERS = {0}

FIRST_DOZEN = set(range(1, 13))
SECOND_DOZEN = set(range(13, 25))
THIRD_DOZEN = set(range(25, 37))

FIRST_COLUMN = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34}
SECOND_COLUMN = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}
THIRD_COLUMN = {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36}

LOW_NUMBERS = set(range(1, 19))
HIGH_NUMBERS = set(range(19, 37))
ODD_NUMBERS = {n for n in range(1, 37) if n % 2 == 1}
EVEN_NUMBERS = {n for n in range(1, 37) if n % 2 == 0}

TOTAL_NUMBERS = 37  # 0-36
TOP_PREDICTIONS_MAX = 14     # 14 numbers: coverage=37.8%, V3 optimal
TOP_PREDICTIONS_COUNT = 14   # 14 numbers — best balance of hit rate vs profit per hit
ANCHOR_COUNT = 4             # Legacy — kept for compatibility
NEIGHBOURS_PER_ANCHOR = 2    # ±2 is optimal — backtest: ±2=32.87%, ±1=32.24%, ±3=31.94%

# ─── Dynamic Prediction Threshold ───────────────────────────────────
# AI picks numbers whose probability exceeds uniform (1/37) by this factor.
# factor=1.0 means pick anything above average. factor=1.3 means 30% above average.
PREDICTION_CONFIDENCE_FACTOR = 1.05  # Target ~14 numbers above threshold — V3 optimal N=14

# ─── Wheel Strategy Groups (physical wheel-based sector strategy) ─────
# Two semicircles of the European wheel, split at 0 and opposite side
WHEEL_TABLE_0 = {3, 26, 0, 32, 21, 2, 25, 27, 13, 36, 23, 10, 5, 1, 20, 14, 18, 29, 7}   # 19 numbers
WHEEL_TABLE_19 = {15, 19, 4, 17, 34, 6, 11, 30, 8, 24, 16, 33, 31, 9, 22, 28, 12, 35}     # 18 numbers

# Positive/Negative classification — alternating groups of 3 on the physical wheel
WHEEL_POSITIVE = {3, 26, 0, 32, 15, 19, 4, 27, 13, 36, 11, 30, 8, 1, 20, 14, 31, 9, 22}   # 19 numbers
WHEEL_NEGATIVE = {21, 2, 25, 17, 34, 6, 23, 10, 5, 24, 16, 33, 18, 29, 7, 28, 12, 35}     # 18 numbers

# Three balanced sets for sector-based betting
WHEEL_SET_1 = {32, 15, 25, 17, 36, 11, 5, 24, 14, 31, 7, 28}   # 12 numbers
WHEEL_SET_2 = {4, 21, 6, 27, 8, 23, 33, 1, 22, 18, 35, 3}      # 12 numbers
WHEEL_SET_3 = {0, 26, 19, 2, 34, 13, 30, 10, 16, 20, 9, 29, 12} # 13 numbers (covers remainder)

# ─── Wheel Sectors (groups of adjacent numbers on physical wheel) ────
def get_wheel_sectors():
    sectors = {}
    for i in range(0, len(WHEEL_ORDER), SECTOR_SIZE):
        sector_nums = WHEEL_ORDER[i:i + SECTOR_SIZE]
        sector_name = f"S{i // SECTOR_SIZE + 1}"
        sectors[sector_name] = sector_nums
    return sectors

WHEEL_SECTORS = get_wheel_sectors()

# Number to wheel position mapping
NUMBER_TO_POSITION = {num: idx for idx, num in enumerate(WHEEL_ORDER)}

# ─── Payout Table ─────────────────────────────────────────────────────
PAYOUTS = {
    'straight': 35,      # Single number
    'split': 17,          # 2 numbers
    'street': 11,         # 3 numbers
    'corner': 8,          # 4 numbers
    'six_line': 5,        # 6 numbers
    'dozen': 2,           # 12 numbers
    'column': 2,          # 12 numbers
    'red_black': 1,       # 18 numbers
    'odd_even': 1,        # 18 numbers
    'high_low': 1,        # 18 numbers
}

# Win probabilities (European single zero)
WIN_PROBABILITIES = {
    'straight': 1 / 37,
    'split': 2 / 37,
    'street': 3 / 37,
    'corner': 4 / 37,
    'six_line': 6 / 37,
    'dozen': 12 / 37,
    'column': 12 / 37,
    'red_black': 18 / 37,
    'odd_even': 18 / 37,
    'high_low': 18 / 37,
}

# ─── File Paths ───────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, 'data')
SESSIONS_DIR = os.path.join(DATA_DIR, 'sessions')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
USERDATA_DIR = os.path.join(BASE_DIR, 'userdata')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.pth')
MODEL_STATE_PATH = os.path.join(MODELS_DIR, 'model_state.pkl')

# ─── Server Settings ─────────────────────────────────────────────────
HOST = '0.0.0.0'
PORT = 5050
DEBUG = False                       # Was True — caused 100%+ CPU and slow responses
SECRET_KEY = 'roulette-ai-prediction-system-2024'

# ─── Ensemble Weights ────────────────────────────────────────────────
# V3 Marathon optimized (59,242 strategies, 23.5 hours, 3,781 real spins):
#   gap+tab_streak dominated all 33 models tested — 62%/38% optimal ratio
#   Best result: gap(62%)+tab_streak(38%) N=14 → +$117/session, 58% win rate
#   Frequency and wheel_strategy provide stable baseline diversity
#   Markov/LSTM still disabled (hurt predictions)
ENSEMBLE_GAP_WEIGHT = 0.35          # Gap/contrarian (mean reversion) — V3 dominant model
ENSEMBLE_TAB_STREAK_WEIGHT = 0.22   # Table streak patterns — V3 breakthrough model
ENSEMBLE_FREQUENCY_WEIGHT = 0.20    # Frequency — reliable baseline
ENSEMBLE_WHEEL_STRATEGY_WEIGHT = 0.13  # Wheel strategy (0/19, pos/neg, sets)
ENSEMBLE_MARKOV_WEIGHT = 0.00       # DISABLED: Markov hurts predictions (-2%)
ENSEMBLE_PATTERN_WEIGHT = 0.05      # Small pattern diversity
ENSEMBLE_LSTM_WEIGHT = 0.00         # DISABLED: overfits
ENSEMBLE_HOT_NUMBER_WEIGHT = 0.05   # Hot number boost

# ─── Wheel Strategy Settings ────────────────────────────────────────
# Backtest sweep found window=50, boost=1.30 gives best edge (+1.67%)
# Shorter windows (10-20) are anti-correlated → trends need time to stabilize
WHEEL_STRATEGY_RECENT_WINDOW = 50   # Last 50 spins — optimal from backtest sweep
WHEEL_STRATEGY_TREND_BOOST = 1.30   # Moderate boost — best from backtest (1.30 > 1.40)
WHEEL_STRATEGY_COLD_DAMPEN = 0.70   # Dampen cold groups (mirrors boost)

# ─── Hot Number Settings ────────────────────────────────────────────
# Numbers appearing frequently in short recent window get boosted.
# Backtest: window=15, boost=2.0 optimal at 8% ensemble weight.
HOT_NUMBER_WINDOW = 15              # Short window to detect recent repeats
HOT_NUMBER_BOOST_FACTOR = 2.0       # 2× boost for hot numbers

# ─── Conditional Betting (Signal Strength Gate) ─────────────────────
# Only bet when wheel strategy signal strength exceeds threshold.
# Backtest results:
#   thresh=25 → 34.54% hit, +2.11% edge, $1908 profit (bet 59% of spins)
#   thresh=30 → 35.92% hit, +3.48% edge, $2772 profit (bet 40% of spins)
# Default: thresh=25 for good balance of frequency and accuracy.
# Set to 0 to disable conditional betting (always bet).
CONDITIONAL_BET_THRESHOLD = 25      # Minimum signal strength to place bet (0-100, 0=always bet)

# ─── Color Mapping for UI ────────────────────────────────────────────
def get_number_color(number):
    if number in RED_NUMBERS:
        return 'red'
    elif number in BLACK_NUMBERS:
        return 'black'
    return 'green'
