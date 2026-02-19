"""
Table Streak Analyzer — Table-Side Streak Pattern Model.

Tracks what follows when consecutive spins land on the same table side
(0-side or 19-side) vs when they alternate. This exploits wheel sector
bias and dealer spin patterns on Evolution Immersive Roulette.

V3 Marathon Winner: tab_streak(38%) was the second model in the best strategy,
tested across 59,242 strategies on 3,781 real spins from 8 data files.
The combination of gap(62%) + tab_streak(38%) produced +$117/session avg.
"""

import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, '.')
from config import TOTAL_NUMBERS, WHEEL_TABLE_0, WHEEL_TABLE_19


# Precompute table mapping: which half of the wheel each number belongs to
TABLE_MAP = {}
for _n in range(TOTAL_NUMBERS):
    TABLE_MAP[_n] = '0' if _n in WHEEL_TABLE_0 else '19'


class TabStreakAnalyzer:
    """Tracks what numbers tend to follow table-side streaks vs alternations.

    The wheel is split into two halves (0-side: 19 numbers, 19-side: 18 numbers).
    When consecutive spins land on the SAME side (streak), certain numbers
    tend to follow. When they alternate, different numbers tend to follow.

    This builds a transition table keyed by (table_side, streak_or_alt).
    """

    MIN_HISTORY = 100  # Need significant data for transition patterns

    def __init__(self):
        self.spin_history = []

    def update(self, number):
        """Add a new spin result."""
        self.spin_history.append(number)

    def load_history(self, history):
        """Load historical spin data."""
        self.spin_history = list(history)

    def get_number_probabilities(self):
        """Return probability distribution based on table streak patterns.

        Looks at the last two spins to determine:
        - Were they on the same table side? (streak)
        - Were they on different sides? (alternation)

        Then returns the historical distribution of what follows that pattern.

        Returns:
            np.array of shape (37,) — probability per number
        """
        probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        if len(self.spin_history) < self.MIN_HISTORY:
            return probs

        # Build transition table: (table_side, streak/alt) → number counts
        trans = defaultdict(lambda: np.ones(TOTAL_NUMBERS) * 0.5)

        for i in range(2, len(self.spin_history)):
            t1 = TABLE_MAP[self.spin_history[i - 2]]
            t2 = TABLE_MAP[self.spin_history[i - 1]]

            if t1 == t2:
                key = (t2, 'streak')
            else:
                key = (t2, 'alt')

            trans[key][self.spin_history[i]] += 1

        # Predict based on last two spins
        if len(self.spin_history) >= 2:
            t1 = TABLE_MAP[self.spin_history[-2]]
            t2 = TABLE_MAP[self.spin_history[-1]]
            key = (t2, 'streak') if t1 == t2 else (t2, 'alt')

            row = trans[key]
            total = row.sum()
            if total > 0:
                return row / total

        return probs

    def get_current_state(self):
        """Return current streak state for UI display."""
        if len(self.spin_history) < 2:
            return {
                'status': 'collecting',
                'last_side': None,
                'pattern': None,
                'streak_length': 0,
            }

        # Count current streak length
        current_side = TABLE_MAP[self.spin_history[-1]]
        streak_length = 1

        for i in range(len(self.spin_history) - 2, -1, -1):
            if TABLE_MAP[self.spin_history[i]] == current_side:
                streak_length += 1
            else:
                break

        t1 = TABLE_MAP[self.spin_history[-2]]
        t2 = TABLE_MAP[self.spin_history[-1]]
        pattern = 'streak' if t1 == t2 else 'alternation'

        return {
            'status': 'active',
            'last_side': current_side,
            'pattern': pattern,
            'streak_length': streak_length,
            'side_label': f"{'Zero' if current_side == '0' else 'Nineteen'}-side",
        }

    def get_strength(self):
        """Return a 0-100 score of how strong the streak signal is.

        Longer streaks = stronger signal (more predictable what follows).
        """
        if len(self.spin_history) < self.MIN_HISTORY:
            return 0.0

        state = self.get_current_state()
        streak_len = state.get('streak_length', 0)

        # Streak of 3+ on same side is a strong signal
        if streak_len >= 5:
            return 90.0
        elif streak_len >= 4:
            return 75.0
        elif streak_len >= 3:
            return 60.0
        elif streak_len >= 2:
            return 40.0
        else:
            return 20.0

    def get_summary(self):
        """Summary info for dashboard display."""
        state = self.get_current_state()

        if len(self.spin_history) < self.MIN_HISTORY:
            return {
                'status': 'collecting',
                'spins_needed': self.MIN_HISTORY - len(self.spin_history),
            }

        # Count how many times each side appeared in recent 20 spins
        recent = self.spin_history[-20:] if len(self.spin_history) >= 20 else self.spin_history
        side_0_count = sum(1 for n in recent if TABLE_MAP[n] == '0')
        side_19_count = len(recent) - side_0_count

        return {
            'status': 'active',
            'current_side': state.get('last_side', '?'),
            'pattern': state.get('pattern', '?'),
            'streak_length': state.get('streak_length', 0),
            'recent_0_side': side_0_count,
            'recent_19_side': side_19_count,
            'strength': round(self.get_strength(), 1),
        }
