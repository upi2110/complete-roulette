"""
Configuration constants for AI European Roulette Prediction System.
Single source of truth for all tunable parameters.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Bankroll Settings ───────────────────────────────────────────────
INITIAL_BANKROLL = 4000.0
BASE_BET = 1.0                      # Starting bet per number ($1)
SESSION_TARGET = 100.0
STOP_LOSS_THRESHOLD = 3000.0        # Protection mode if bankroll drops below this
MAX_CONSECUTIVE_LOSSES = 5          # After this many losses, require higher confidence to bet (was 3 — too aggressive)
KELLY_FRACTION = 0.25               # (Legacy — kept for compatibility)
MAX_BET_MULTIPLIER = 20.0           # Max bet per number = BASE_BET * this
RECOVERY_BET_FACTOR = 0.75          # (Legacy — kept for compatibility)

# ─── Incremental Betting Strategy ───────────────────────────────────
BET_INCREMENT = 1.0                 # +$1 after LOSSES_BEFORE_INCREMENT consecutive losses
BET_DECREMENT = 1.0                 # -$1 after WINS_BEFORE_DECREMENT consecutive wins
MIN_BET = 1.0                       # Floor for bet per number
LOSSES_BEFORE_INCREMENT = 3         # Consecutive losses before bet increases
WINS_BEFORE_DECREMENT = 2           # Consecutive wins before bet decreases

# ─── Confidence Thresholds ───────────────────────────────────────────
# Lowered from 45/55/65 → 40/50/60 so the system can actually bet.
# With 6-factor scoring, neutral = ~50. Requiring 55 after losses was unachievable.
CONFIDENCE_BET_THRESHOLD = 40.0     # Reduced: was 45 — too hard to reach after losses
CONFIDENCE_HIGH_THRESHOLD = 50.0    # Reduced: was 55 — loss-streak caution requires this
CONFIDENCE_STRAIGHT_THRESHOLD = 60.0  # Reduced: was 65
FORCED_WAIT_SPINS = 0               # No forced wait — AI decides via confidence

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
FREQUENCY_FLAT_WEIGHT = 0.80        # Backtest: 100% flat beat all blends — using 80/20 for some recency
FREQUENCY_RECENT_WEIGHT = 0.20      # Reduced from 0.65 — flat distribution more reliable

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
TOP_PREDICTIONS_MAX = 12     # Safety cap — AI can pick up to this many numbers
TOP_PREDICTIONS_COUNT = 12   # Legacy — kept for compatibility with tests
ANCHOR_COUNT = 4             # Legacy — kept for compatibility
NEIGHBOURS_PER_ANCHOR = 2    # Neighbours each side of anchor on wheel (±2)

# ─── Dynamic Prediction Threshold ───────────────────────────────────
# AI picks numbers whose probability exceeds uniform (1/37) by this factor.
# factor=1.0 means pick anything above average. factor=1.3 means 30% above average.
PREDICTION_CONFIDENCE_FACTOR = 1.5  # Pick numbers ≥ 1.5× the uniform probability

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
# Markov strongest in real-time play. LSTM overfits on training data (59.6%
# backtest was memorisation, not prediction). Keep LSTM meaningful but not dominant.
ENSEMBLE_FREQUENCY_WEIGHT = 0.30
ENSEMBLE_MARKOV_WEIGHT = 0.40
ENSEMBLE_PATTERN_WEIGHT = 0.05
ENSEMBLE_LSTM_WEIGHT = 0.25

# ─── Color Mapping for UI ────────────────────────────────────────────
def get_number_color(number):
    if number in RED_NUMBERS:
        return 'red'
    elif number in BLACK_NUMBERS:
        return 'black'
    return 'green'
