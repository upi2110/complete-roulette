"""
Wheel Strategy Analyzer — Sector-based prediction using physical wheel layout.

Analyzes three independent grouping systems based on the European roulette wheel:
  1. Table Groups (0/19): Two semicircles of the wheel
  2. Polarity (Positive/Negative): Alternating groups of ~3 on the wheel
  3. Sets (1/2/3): Three balanced betting groups covering all 37 numbers

Each system tracks recent trends independently. The combined signal boosts
numbers that appear in currently "hot" groups across all three systems.

Example: If recent spins favour Table-0, Positive polarity, and Set-2,
numbers that belong to ALL three groups get the strongest boost.
"""

import numpy as np
from collections import Counter, deque

import sys
sys.path.insert(0, '.')
from config import (
    TOTAL_NUMBERS,
    WHEEL_TABLE_0, WHEEL_TABLE_19,
    WHEEL_POSITIVE, WHEEL_NEGATIVE,
    WHEEL_SET_1, WHEEL_SET_2, WHEEL_SET_3,
    WHEEL_STRATEGY_RECENT_WINDOW,
    WHEEL_STRATEGY_TREND_BOOST, WHEEL_STRATEGY_COLD_DAMPEN,
)


class WheelStrategyAnalyzer:
    """Generates probability distributions based on wheel sector strategy."""

    def __init__(self):
        self.spin_history = []
        # Pre-compute membership for fast lookup
        self._table_map = {}   # number → '0' or '19'
        self._polarity_map = {}  # number → 'positive' or 'negative'
        self._set_map = {}     # number → 1, 2, or 3

        for n in range(TOTAL_NUMBERS):
            if n in WHEEL_TABLE_0:
                self._table_map[n] = '0'
            else:
                self._table_map[n] = '19'

            if n in WHEEL_POSITIVE:
                self._polarity_map[n] = 'positive'
            else:
                self._polarity_map[n] = 'negative'

            if n in WHEEL_SET_1:
                self._set_map[n] = 1
            elif n in WHEEL_SET_2:
                self._set_map[n] = 2
            else:
                self._set_map[n] = 3

    def update(self, number):
        """Add a new spin result."""
        self.spin_history.append(number)

    def load_history(self, history):
        """Load full history."""
        self.spin_history = list(history)

    def get_table_trend(self, window=None):
        """Analyze which table (0 or 19) is hot in recent spins.

        Returns:
            dict with 'hot_table', 'table_0_pct', 'table_19_pct',
                  'table_0_count', 'table_19_count'
        """
        if window is None:
            window = WHEEL_STRATEGY_RECENT_WINDOW
        recent = self.spin_history[-window:] if self.spin_history else []
        if not recent:
            return {'hot_table': None, 'table_0_pct': 0.5, 'table_19_pct': 0.5,
                    'table_0_count': 0, 'table_19_count': 0}

        t0 = sum(1 for n in recent if self._table_map.get(n) == '0')
        t19 = len(recent) - t0
        total = len(recent)

        return {
            'hot_table': '0' if t0 > t19 else '19' if t19 > t0 else None,
            'table_0_pct': round(t0 / total, 3),
            'table_19_pct': round(t19 / total, 3),
            'table_0_count': t0,
            'table_19_count': t19,
        }

    def get_polarity_trend(self, window=None):
        """Analyze which polarity (positive/negative) is hot.

        Returns:
            dict with 'hot_polarity', 'positive_pct', 'negative_pct',
                  'positive_count', 'negative_count'
        """
        if window is None:
            window = WHEEL_STRATEGY_RECENT_WINDOW
        recent = self.spin_history[-window:] if self.spin_history else []
        if not recent:
            return {'hot_polarity': None, 'positive_pct': 0.5, 'negative_pct': 0.5,
                    'positive_count': 0, 'negative_count': 0}

        pos = sum(1 for n in recent if self._polarity_map.get(n) == 'positive')
        neg = len(recent) - pos
        total = len(recent)

        return {
            'hot_polarity': 'positive' if pos > neg else 'negative' if neg > pos else None,
            'positive_pct': round(pos / total, 3),
            'negative_pct': round(neg / total, 3),
            'positive_count': pos,
            'negative_count': neg,
        }

    def get_set_trend(self, window=None):
        """Analyze which set (1/2/3) is hot.

        Returns:
            dict with 'hot_set', 'set_1_pct', 'set_2_pct', 'set_3_pct',
                  'set_counts'
        """
        if window is None:
            window = WHEEL_STRATEGY_RECENT_WINDOW
        recent = self.spin_history[-window:] if self.spin_history else []
        if not recent:
            return {'hot_set': None, 'set_1_pct': 0.333, 'set_2_pct': 0.333,
                    'set_3_pct': 0.334, 'set_counts': {1: 0, 2: 0, 3: 0}}

        counts = {1: 0, 2: 0, 3: 0}
        for n in recent:
            s = self._set_map.get(n, 3)
            counts[s] += 1
        total = len(recent)

        hot = max(counts, key=counts.get)
        # Only declare hot if it's meaningfully above expected
        expected = total / 3
        if counts[hot] <= expected * 1.1:
            hot = None

        return {
            'hot_set': hot,
            'set_1_pct': round(counts[1] / total, 3),
            'set_2_pct': round(counts[2] / total, 3),
            'set_3_pct': round(counts[3] / total, 3),
            'set_counts': counts,
        }

    def get_number_probabilities(self):
        """Generate probability distribution using all three grouping systems.

        Each system independently boosts/dampens numbers based on recent trends:
          - Table: numbers in the hot table get boosted
          - Polarity: numbers with the hot polarity get boosted
          - Set: numbers in the hot set get boosted

        Boosts are multiplicative so numbers in ALL hot groups get the biggest push.
        """
        probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        if len(self.spin_history) < 5:
            return probs

        table_trend = self.get_table_trend()
        polarity_trend = self.get_polarity_trend()
        set_trend = self.get_set_trend()

        boost = WHEEL_STRATEGY_TREND_BOOST
        dampen = WHEEL_STRATEGY_COLD_DAMPEN

        for num in range(TOTAL_NUMBERS):
            multiplier = 1.0

            # Table boost/dampen
            hot_table = table_trend['hot_table']
            if hot_table is not None:
                num_table = self._table_map.get(num)
                t0_pct = table_trend['table_0_pct']
                t19_pct = table_trend['table_19_pct']
                # Scale boost by how dominant the hot table is
                # (e.g. 60% vs 40% gives moderate boost; 70% vs 30% gives strong boost)
                imbalance = abs(t0_pct - t19_pct) / 0.5  # 0 to ~1
                table_boost = 1.0 + (boost - 1.0) * imbalance
                table_dampen = 1.0 - (1.0 - dampen) * imbalance
                if num_table == hot_table:
                    multiplier *= table_boost
                else:
                    multiplier *= table_dampen

            # Polarity boost/dampen
            hot_pol = polarity_trend['hot_polarity']
            if hot_pol is not None:
                num_pol = self._polarity_map.get(num)
                pos_pct = polarity_trend['positive_pct']
                neg_pct = polarity_trend['negative_pct']
                imbalance = abs(pos_pct - neg_pct) / 0.5
                pol_boost = 1.0 + (boost - 1.0) * imbalance
                pol_dampen = 1.0 - (1.0 - dampen) * imbalance
                if num_pol == hot_pol:
                    multiplier *= pol_boost
                else:
                    multiplier *= pol_dampen

            # Set boost/dampen
            hot_set = set_trend['hot_set']
            if hot_set is not None:
                num_set = self._set_map.get(num, 3)
                # For 3 sets, use the hot set's percentage vs expected (33.3%)
                set_pcts = {
                    1: set_trend['set_1_pct'],
                    2: set_trend['set_2_pct'],
                    3: set_trend['set_3_pct'],
                }
                hot_excess = (set_pcts[hot_set] - 1/3) / (1/3)  # 0 to ~1
                hot_excess = max(0, min(1, hot_excess))
                set_boost = 1.0 + (boost - 1.0) * hot_excess
                set_dampen = 1.0 - (1.0 - dampen) * hot_excess
                if num_set == hot_set:
                    multiplier *= set_boost
                else:
                    multiplier *= set_dampen

            probs[num] *= multiplier

        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total

        return probs

    def get_strategy_strength(self):
        """Overall strategy signal strength 0-100.

        High score = strong trends detected across all three grouping systems.
        Low score = everything near expected (no useful signal).
        """
        if len(self.spin_history) < 5:
            return 0.0

        scores = []

        # Table imbalance score
        table = self.get_table_trend()
        # Expected: 0-table=19/37≈51.4%, 19-table=18/37≈48.6%
        expected_t0 = 19 / 37
        t0_dev = abs(table['table_0_pct'] - expected_t0)
        # Max realistic deviation ~0.25 (75/25 split)
        table_score = min(100, t0_dev / 0.25 * 100)
        scores.append(table_score)

        # Polarity imbalance score
        pol = self.get_polarity_trend()
        expected_pos = 19 / 37
        pos_dev = abs(pol['positive_pct'] - expected_pos)
        pol_score = min(100, pos_dev / 0.25 * 100)
        scores.append(pol_score)

        # Set imbalance score
        sets = self.get_set_trend()
        set_pcts = [sets['set_1_pct'], sets['set_2_pct'], sets['set_3_pct']]
        max_dev = max(abs(p - 1/3) for p in set_pcts)
        set_score = min(100, max_dev / 0.20 * 100)
        scores.append(set_score)

        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def get_summary(self):
        """Full strategy summary for debugging/display."""
        return {
            'table_trend': self.get_table_trend(),
            'polarity_trend': self.get_polarity_trend(),
            'set_trend': self.get_set_trend(),
            'strategy_strength': self.get_strategy_strength(),
            'total_spins': len(self.spin_history),
        }
