"""
Feature Engine — Rich feature extraction for neural network input.

Instead of feeding the GRU raw one-hot vectors (37 dims), this module
creates a rich ~85-dimensional feature vector for each spin position,
encoding wheel geometry, category memberships, gap statistics,
running context, and recency information.
"""

import math
import numpy as np
from collections import Counter

import sys
sys.path.insert(0, '.')
from config import (
    TOTAL_NUMBERS, RED_NUMBERS, BLACK_NUMBERS,
    FIRST_DOZEN, SECOND_DOZEN, THIRD_DOZEN,
    FIRST_COLUMN, SECOND_COLUMN, THIRD_COLUMN,
    LOW_NUMBERS, HIGH_NUMBERS, ODD_NUMBERS, EVEN_NUMBERS,
    WHEEL_ORDER, NUMBER_TO_POSITION, WHEEL_SECTORS,
)

# ── Pre-computed wheel angles (sin/cos for positional encoding) ──────
_WHEEL_LEN = len(WHEEL_ORDER)
_WHEEL_ANGLE = {}  # number → angle in radians
for _idx, _num in enumerate(WHEEL_ORDER):
    _WHEEL_ANGLE[_num] = 2 * math.pi * _idx / _WHEEL_LEN


def _wheel_position_features(number):
    """Sin/cos encoding of physical wheel position (2 dims)."""
    angle = _WHEEL_ANGLE.get(number, 0.0)
    return [math.sin(angle), math.cos(angle)]


def _category_features(number):
    """One-hot category features (6 dims): red, black, green, odd, even, zero."""
    return [
        1.0 if number in RED_NUMBERS else 0.0,
        1.0 if number in BLACK_NUMBERS else 0.0,
        1.0 if number == 0 else 0.0,       # green
        1.0 if number in ODD_NUMBERS else 0.0,
        1.0 if number in EVEN_NUMBERS else 0.0,
        1.0 if number == 0 else 0.0,       # is_zero (explicit flag)
    ]


def _group_features(number):
    """Dozen and column one-hot features (6 dims)."""
    return [
        1.0 if number in FIRST_DOZEN else 0.0,
        1.0 if number in SECOND_DOZEN else 0.0,
        1.0 if number in THIRD_DOZEN else 0.0,
        1.0 if number in FIRST_COLUMN else 0.0,
        1.0 if number in SECOND_COLUMN else 0.0,
        1.0 if number in THIRD_COLUMN else 0.0,
    ]


def _high_low_features(number):
    """High/Low features (2 dims)."""
    return [
        1.0 if number in HIGH_NUMBERS else 0.0,
        1.0 if number in LOW_NUMBERS else 0.0,
    ]


class FeatureEngine:
    """Converts raw spin history into rich feature sequences for the neural net."""

    # Feature dimensions breakdown (for documentation / testing):
    #   wheel_position: 2
    #   category: 6
    #   group: 6
    #   high_low: 2
    #   gap: 37
    #   running_stats: 10
    #   prev_number_encoding: 22  (top deviation from uniform, compressed)
    # Total: ~85
    STATIC_DIMS = 2 + 6 + 6 + 2        # 16 per-number features
    GAP_DIMS = TOTAL_NUMBERS            # 37
    RUNNING_DIMS = 10
    PREV_ENCODING_DIMS = 22
    FEATURE_DIM = STATIC_DIMS + GAP_DIMS + RUNNING_DIMS + PREV_ENCODING_DIMS  # 85

    def __init__(self):
        pass

    # ── Public API ───────────────────────────────────────────────────

    def extract_sequence(self, spin_history, seq_start, seq_length):
        """Extract a feature matrix for a sequence window.

        Args:
            spin_history: full list of spin results
            seq_start: index of the first spin in the window
            seq_length: number of timesteps

        Returns:
            np.ndarray of shape (seq_length, FEATURE_DIM), dtype float32
        """
        features = np.zeros((seq_length, self.FEATURE_DIM), dtype=np.float32)

        for t in range(seq_length):
            idx = seq_start + t
            number = spin_history[idx]

            # Context: all spins before this position (for gaps, running stats)
            context = spin_history[:idx]

            features[t] = self._build_feature_vector(number, context, spin_history, idx)

        return features

    def extract_single(self, number, context, full_history=None, position=None):
        """Extract features for a single spin (used for online prediction).

        Args:
            number: the spin result
            context: list of spins before this one
            full_history: optional full history (defaults to context + [number])
            position: optional index in full_history

        Returns:
            np.ndarray of shape (FEATURE_DIM,), dtype float32
        """
        if full_history is None:
            full_history = list(context) + [number]
        if position is None:
            position = len(context)
        return self._build_feature_vector(number, context, full_history, position)

    # ── Internal ─────────────────────────────────────────────────────

    def _build_feature_vector(self, number, context, full_history, idx):
        """Build the full feature vector for one timestep."""
        vec = []

        # 1. Wheel position encoding (2 dims)
        vec.extend(_wheel_position_features(number))

        # 2. Category features (6 dims)
        vec.extend(_category_features(number))

        # 3. Group features (6 dims)
        vec.extend(_group_features(number))

        # 4. High/Low (2 dims)
        vec.extend(_high_low_features(number))

        # 5. Gap features (37 dims) — spins since each number last appeared
        vec.extend(self._gap_features(context))

        # 6. Running statistics (10 dims)
        vec.extend(self._running_stats(context))

        # 7. Previous-number encoding (22 dims)
        vec.extend(self._prev_number_encoding(context))

        return np.array(vec, dtype=np.float32)

    def _gap_features(self, context):
        """For each number 0-36, how many spins since it last appeared.
        Normalized to [0, 1] by dividing by max(gap, 1).
        If a number never appeared, gap = len(context).
        """
        n = len(context)
        if n == 0:
            return [1.0] * TOTAL_NUMBERS  # No history — max gap for all

        last_seen = {}
        for i, num in enumerate(context):
            last_seen[num] = i

        gaps = []
        max_gap = max(n, 1)
        for num in range(TOTAL_NUMBERS):
            if num in last_seen:
                gap = n - last_seen[num]
            else:
                gap = n  # Never seen
            gaps.append(min(gap / max_gap, 1.0))

        return gaps

    def _running_stats(self, context):
        """10 running statistics computed from recent context.

        1. color_streak_length — current color streak (normalized /10)
        2. dozen_dominance — max(dozen_pct) - 1/3 (how skewed dozens are)
        3. sector_heat — max sector ratio in last 20 (0 if not enough data)
        4. hot_cold_ratio — (max_count - min_count) / expected
        5. repeat_rate_20 — fraction of last 20 that are repeats
        6. spins_since_repeat — spins since last consecutive repeat (/20)
        7. sector_change — 1 if last two spins in different sectors, else 0
        8. consec_same_dozen — consecutive same-dozen count (/5)
        9. gap_variance — variance of per-number gaps (normalized)
        10. entropy_20 — entropy of last 20 spins (normalized by max)
        """
        stats = [0.0] * 10
        n = len(context)
        if n < 3:
            return stats

        # 1. Color streak
        streak = 0
        last_color = self._color_of(context[-1])
        for num in reversed(context):
            if self._color_of(num) == last_color:
                streak += 1
            else:
                break
        stats[0] = min(streak / 10.0, 1.0)

        # 2. Dozen dominance (last 30)
        window = context[-30:]
        dozen_counts = [0, 0, 0]
        non_zero = 0
        for num in window:
            if num in FIRST_DOZEN:
                dozen_counts[0] += 1; non_zero += 1
            elif num in SECOND_DOZEN:
                dozen_counts[1] += 1; non_zero += 1
            elif num in THIRD_DOZEN:
                dozen_counts[2] += 1; non_zero += 1
        if non_zero > 0:
            max_pct = max(dozen_counts) / non_zero
            stats[1] = min((max_pct - 1/3) * 3, 1.0)  # 0 when uniform, ~1 when dominant

        # 3. Sector heat (last 20)
        recent_20 = context[-20:]
        if len(recent_20) >= 10:
            max_ratio = 0.0
            for sector_nums in WHEEL_SECTORS.values():
                hits = sum(1 for s in recent_20 if s in sector_nums)
                expected = len(recent_20) * len(sector_nums) / TOTAL_NUMBERS
                if expected > 0:
                    ratio = hits / expected
                    max_ratio = max(max_ratio, ratio)
            stats[2] = min((max_ratio - 1.0) / 1.0, 1.0)  # 0 if fair, 1 if 2x expected

        # 4. Hot/cold ratio
        counts = Counter(context)
        expected = n / TOTAL_NUMBERS
        if expected > 0:
            max_c = max(counts.values())
            min_c = min(counts.get(i, 0) for i in range(TOTAL_NUMBERS))
            stats[3] = min((max_c - min_c) / (expected + 1), 1.0)

        # 5. Repeat rate in last 20
        r20 = context[-20:]
        if len(r20) >= 5:
            unique = len(set(r20))
            stats[4] = 1.0 - (unique / len(r20))  # Higher = more repeats

        # 6. Spins since last consecutive repeat
        dist = 0
        for i in range(len(context) - 1, 0, -1):
            if context[i] == context[i - 1]:
                break
            dist += 1
        stats[5] = min(dist / 20.0, 1.0)

        # 7. Sector change
        if n >= 2:
            pos1 = NUMBER_TO_POSITION.get(context[-1], 0)
            pos2 = NUMBER_TO_POSITION.get(context[-2], 0)
            sector1 = pos1 // 5
            sector2 = pos2 // 5
            stats[6] = 0.0 if sector1 == sector2 else 1.0

        # 8. Consecutive same dozen
        if n >= 2:
            current_dozen = self._dozen_idx(context[-1])
            consec = 1
            for num in reversed(context[:-1]):
                if self._dozen_idx(num) == current_dozen:
                    consec += 1
                else:
                    break
            stats[7] = min(consec / 5.0, 1.0)

        # 9. Gap variance
        if n >= TOTAL_NUMBERS:
            last_seen = {}
            for i, num in enumerate(context):
                last_seen[num] = i
            gaps = [(n - last_seen.get(i, 0)) for i in range(TOTAL_NUMBERS)]
            gap_var = np.var(gaps) if gaps else 0
            stats[8] = min(gap_var / (n * 0.5 + 1), 1.0)

        # 10. Entropy of last 20
        r20 = context[-20:]
        if len(r20) >= 5:
            c20 = Counter(r20)
            total = len(r20)
            entropy = 0.0
            for count in c20.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(min(TOTAL_NUMBERS, total))
            stats[9] = entropy / max_entropy if max_entropy > 0 else 0.0

        return stats

    def _prev_number_encoding(self, context):
        """Encode the previous number as top deviations from uniform (22 dims).

        Instead of a full 37-dim one-hot, we encode:
          - One-hot of previous number projected into top 20 deviation slots
          - Plus 2 features: prev number's wheel position sin/cos

        This compresses 37 dims into 22 while retaining the most information.
        """
        encoding = [0.0] * self.PREV_ENCODING_DIMS
        if not context:
            return encoding

        prev = context[-1]

        # Sin/cos of previous number's wheel position
        angle = _WHEEL_ANGLE.get(prev, 0.0)
        encoding[0] = math.sin(angle)
        encoding[1] = math.cos(angle)

        # If enough context, encode deviation from uniform for top 20 numbers
        if len(context) >= 20:
            counts = Counter(context[-100:])  # Last 100 for stability
            total = sum(counts.values())
            expected = total / TOTAL_NUMBERS
            deviations = []
            for num in range(TOTAL_NUMBERS):
                dev = (counts.get(num, 0) - expected) / (expected + 1)
                deviations.append((num, dev))
            # Sort by absolute deviation, take top 20
            deviations.sort(key=lambda x: abs(x[1]), reverse=True)
            for i, (num, dev) in enumerate(deviations[:20]):
                encoding[2 + i] = max(-1.0, min(1.0, dev))
        else:
            # With little context, use simple one-hot-like encoding
            # Map prev number to a slot (prev % 20)
            slot = prev % 20
            encoding[2 + slot] = 1.0

        return encoding

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _color_of(n):
        if n in RED_NUMBERS:
            return 'red'
        if n in BLACK_NUMBERS:
            return 'black'
        return 'green'

    @staticmethod
    def _dozen_idx(n):
        if n in FIRST_DOZEN:
            return 0
        if n in SECOND_DOZEN:
            return 1
        if n in THIRD_DOZEN:
            return 2
        return -1  # zero
