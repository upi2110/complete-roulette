"""
Gap Analyzer — Contrarian/Mean Reversion Model.

Numbers that haven't appeared for a long time get boosted probability.
This exploits real wheel biases where neglected numbers eventually cluster.

V3 Marathon Winner: gap(62%) was the dominant model in the best strategy,
tested across 59,242 strategies on 3,781 real spins from 8 data files.
"""

import numpy as np

import sys
sys.path.insert(0, '.')
from config import TOTAL_NUMBERS


class GapAnalyzer:
    """Tracks how long since each number last appeared and boosts overdue numbers.

    The core insight: on a biased wheel, numbers don't appear uniformly.
    Numbers that have been absent longer than ~2x the expected gap
    have a slightly higher probability of appearing soon (mean reversion).
    """

    MIN_HISTORY = 50  # Need at least 50 spins before gap analysis is meaningful

    def __init__(self):
        self.spin_history = []

    def update(self, number):
        """Add a new spin result."""
        self.spin_history.append(number)

    def load_history(self, history):
        """Load historical spin data."""
        self.spin_history = list(history)

    def get_number_probabilities(self):
        """Return probability distribution based on gap analysis.

        Numbers with gaps > 2x expected (74 spins) get 1.5x boost.
        Numbers with gaps > 1.5x expected (55 spins) get 1.2x boost.
        All probabilities are normalized to sum to 1.

        Returns:
            np.array of shape (37,) — probability per number
        """
        probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        if len(self.spin_history) < self.MIN_HISTORY:
            return probs

        # Find last appearance of each number
        last_seen = {}
        for i, n in enumerate(self.spin_history):
            last_seen[n] = i

        current_pos = len(self.spin_history)

        for num in range(TOTAL_NUMBERS):
            gap = current_pos - last_seen.get(num, 0)

            # Strong boost: number hasn't appeared in 2x expected interval
            if gap > TOTAL_NUMBERS * 2:  # > 74 spins
                probs[num] *= 1.5
            # Moderate boost: number hasn't appeared in 1.5x expected interval
            elif gap > TOTAL_NUMBERS * 1.5:  # > 55 spins
                probs[num] *= 1.2

        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total

        return probs

    def get_gap_stats(self):
        """Return gap statistics for each number (for UI display)."""
        if not self.spin_history:
            return {}

        last_seen = {}
        for i, n in enumerate(self.spin_history):
            last_seen[n] = i

        current_pos = len(self.spin_history)
        stats = {}

        for num in range(TOTAL_NUMBERS):
            gap = current_pos - last_seen.get(num, 0)
            stats[num] = {
                'gap': gap,
                'overdue': gap > TOTAL_NUMBERS * 1.5,
                'very_overdue': gap > TOTAL_NUMBERS * 2,
            }

        return stats

    def get_strength(self):
        """Return a 0-100 score of how strong the gap signal is.

        Higher score = more numbers are significantly overdue.
        """
        if len(self.spin_history) < self.MIN_HISTORY:
            return 0.0

        last_seen = {}
        for i, n in enumerate(self.spin_history):
            last_seen[n] = i

        current_pos = len(self.spin_history)
        overdue_count = 0

        for num in range(TOTAL_NUMBERS):
            gap = current_pos - last_seen.get(num, 0)
            if gap > TOTAL_NUMBERS * 1.5:
                overdue_count += 1

        # Normalize: ~5-10 overdue numbers is typical, 15+ is very strong
        return min(100.0, overdue_count * 10.0)

    def get_summary(self):
        """Summary info for dashboard display."""
        if len(self.spin_history) < self.MIN_HISTORY:
            return {'status': 'collecting', 'overdue_count': 0}

        stats = self.get_gap_stats()
        overdue = [n for n, s in stats.items() if s['overdue']]
        very_overdue = [n for n, s in stats.items() if s['very_overdue']]

        return {
            'status': 'active',
            'overdue_count': len(overdue),
            'very_overdue_count': len(very_overdue),
            'overdue_numbers': sorted(overdue)[:5],  # Top 5 for display
            'very_overdue_numbers': sorted(very_overdue)[:5],
            'strength': round(self.get_strength(), 1),
        }
