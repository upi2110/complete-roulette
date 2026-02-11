"""
Frequency Analyzer — Chi-square bias detection, hot/cold number identification,
and time-weighted frequency analysis.

Uses statistical tests to detect deviations from uniform distribution.
Time-weighted frequencies give more weight to recent spins via exponential decay.
"""

import numpy as np
from scipy import stats
from collections import Counter

import sys
sys.path.insert(0, '.')
from config import (
    TOTAL_NUMBERS, HOT_NUMBER_THRESHOLD, COLD_NUMBER_THRESHOLD,
    WHEEL_SECTORS, NUMBER_TO_POSITION, RED_NUMBERS, BLACK_NUMBERS,
    FREQUENCY_DECAY_FACTOR, FREQUENCY_FLAT_WEIGHT, FREQUENCY_RECENT_WEIGHT,
)


class FrequencyAnalyzer:
    def __init__(self):
        self.spin_history = []
        self.frequency_counts = Counter()
        self.expected_frequency = 1.0 / TOTAL_NUMBERS

    def update(self, number):
        self.spin_history.append(number)
        self.frequency_counts[number] += 1

    def load_history(self, history):
        self.spin_history = list(history)
        self.frequency_counts = Counter(self.spin_history)

    def get_chi_square_result(self):
        """Chi-square goodness-of-fit test against uniform distribution."""
        if len(self.spin_history) < 10:
            return {'statistic': 0, 'p_value': 1.0, 'significant': False}

        observed = np.array([self.frequency_counts.get(i, 0) for i in range(TOTAL_NUMBERS)])
        expected = np.full(TOTAL_NUMBERS, len(self.spin_history) / TOTAL_NUMBERS)

        chi2, p_value = stats.chisquare(observed, expected)
        return {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }

    def get_hot_numbers(self, top_n=5):
        """Numbers appearing significantly more than expected."""
        if not self.spin_history:
            return []

        total = len(self.spin_history)
        expected = total / TOTAL_NUMBERS
        hot = []

        for num in range(TOTAL_NUMBERS):
            count = self.frequency_counts.get(num, 0)
            ratio = count / expected if expected > 0 else 0
            if ratio >= HOT_NUMBER_THRESHOLD:
                hot.append({
                    'number': num,
                    'count': count,
                    'ratio': round(ratio, 2),
                    'frequency': round(count / total, 4)
                })

        hot.sort(key=lambda x: x['ratio'], reverse=True)
        return hot[:top_n]

    def get_cold_numbers(self, top_n=5):
        """Numbers appearing significantly less than expected."""
        if not self.spin_history:
            return []

        total = len(self.spin_history)
        expected = total / TOTAL_NUMBERS
        cold = []

        for num in range(TOTAL_NUMBERS):
            count = self.frequency_counts.get(num, 0)
            ratio = count / expected if expected > 0 else 0
            if ratio <= COLD_NUMBER_THRESHOLD:
                cold.append({
                    'number': num,
                    'count': count,
                    'ratio': round(ratio, 2),
                    'frequency': round(count / total, 4)
                })

        cold.sort(key=lambda x: x['ratio'])
        return cold[:top_n]

    def get_sector_bias(self):
        """Analyze if certain wheel sectors are hitting more than expected."""
        if len(self.spin_history) < 20:
            return {}

        total = len(self.spin_history)
        sector_scores = {}

        for sector_name, sector_nums in WHEEL_SECTORS.items():
            sector_count = sum(self.frequency_counts.get(n, 0) for n in sector_nums)
            expected = total * len(sector_nums) / TOTAL_NUMBERS
            ratio = sector_count / expected if expected > 0 else 1.0
            sector_scores[sector_name] = {
                'numbers': sector_nums,
                'count': sector_count,
                'expected': round(expected, 1),
                'ratio': round(ratio, 2),
                'hot': ratio >= 1.3,
                'cold': ratio <= 0.7
            }

        return sector_scores

    def get_number_probabilities(self):
        """Return probability distribution based on frequency analysis.

        Blends uniform (Laplace-smoothed) counts with time-weighted counts
        so that recent data has more influence.
        """
        if len(self.spin_history) < 5:
            return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        # Standard Laplace-smoothed distribution
        total = len(self.spin_history)
        flat_probs = np.array([
            (self.frequency_counts.get(i, 0) + 1) / (total + TOTAL_NUMBERS)
            for i in range(TOTAL_NUMBERS)
        ])
        flat_probs /= flat_probs.sum()

        # Time-weighted distribution
        recent_probs = self.get_recent_probabilities()

        # Blend: 40% flat (overall) + 60% recent (time-weighted)
        blended = FREQUENCY_FLAT_WEIGHT * flat_probs + FREQUENCY_RECENT_WEIGHT * recent_probs
        blended /= blended.sum()

        return blended

    def get_recent_probabilities(self):
        """Time-weighted probability distribution using exponential decay.

        Recent spins count more than old spins.
        Decay factor 0.998 per spin → half-life ~350 spins.
        """
        if len(self.spin_history) < 5:
            return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        n = len(self.spin_history)
        decay = FREQUENCY_DECAY_FACTOR

        # Compute weighted counts
        weighted_counts = np.ones(TOTAL_NUMBERS, dtype=np.float64)  # Laplace smoothing
        for i, num in enumerate(self.spin_history):
            weight = decay ** (n - 1 - i)  # Most recent spin has weight 1.0
            weighted_counts[num] += weight

        probs = weighted_counts / weighted_counts.sum()
        return probs

    def get_color_distribution(self):
        """Analyze red/black/green distribution."""
        if not self.spin_history:
            return {'red': 0, 'black': 0, 'green': 0}

        red = sum(1 for n in self.spin_history if n in RED_NUMBERS)
        black = sum(1 for n in self.spin_history if n in BLACK_NUMBERS)
        green = sum(1 for n in self.spin_history if n == 0)
        total = len(self.spin_history)

        return {
            'red': round(red / total, 4),
            'black': round(black / total, 4),
            'green': round(green / total, 4),
            'red_count': red,
            'black_count': black,
            'green_count': green
        }

    def get_bias_score(self):
        """Overall bias score 0-100.  Higher = more detectable patterns.

        Multi-factor scoring that stays meaningful as data grows:
          1. Chi-square factor  — statistical deviation from uniform
          2. Peak deviation      — how extreme the hottest/coldest numbers are
          3. Recency factor      — are recent spins (last 200) more biased?

        Each factor is 0-100, weighted and combined.
        """
        n = len(self.spin_history)
        if n < 10:
            return 0.0

        # ── Factor 1: Chi-square (30% weight) ──────────────────────
        chi = self.get_chi_square_result()
        # Use -log10(p) to spread the scale (p=0.05→1.3, p=0.001→3.0)
        if chi['p_value'] <= 0:
            chi_factor = 100.0
        else:
            import math
            neg_log_p = -math.log10(max(chi['p_value'], 1e-10))
            # Scale: 0 → 0, 1.3 (p=0.05) → 50, 3.0 (p=0.001) → 100
            chi_factor = min(100, neg_log_p / 3.0 * 100)

        # ── Factor 2: Peak deviation (40% weight) ─────────────────
        expected = n / TOTAL_NUMBERS
        if expected > 0:
            ratios = [self.frequency_counts.get(i, 0) / expected
                      for i in range(TOTAL_NUMBERS)]
            max_over = max(ratios)   # hottest number ratio
            min_under = min(ratios)  # coldest number ratio
            # Combine over- and under-representation
            # ratio 1.0 = fair. 1.4 → score ~60, 1.8 → ~100
            peak_dev = max(max_over - 1.0, 1.0 - min_under)
            peak_factor = min(100, peak_dev / 0.6 * 100)
        else:
            peak_factor = 0.0

        # ── Factor 3: Recency (30% weight) ────────────────────────
        # Check if the LAST 200 spins are more biased than overall
        window = min(200, n)
        recent = self.spin_history[-window:]
        recent_counts = Counter(recent)
        recent_expected = window / TOTAL_NUMBERS
        if recent_expected > 0:
            recent_ratios = [recent_counts.get(i, 0) / recent_expected
                             for i in range(TOTAL_NUMBERS)]
            recent_peak = max(max(recent_ratios) - 1.0, 1.0 - min(recent_ratios))
            recency_factor = min(100, recent_peak / 0.5 * 100)
        else:
            recency_factor = 0.0

        # ── Weighted combination ──────────────────────────────────
        score = (chi_factor * 0.30 +
                 peak_factor * 0.40 +
                 recency_factor * 0.30)

        return round(min(100, max(0, score)), 1)

    def get_summary(self):
        return {
            'total_spins': len(self.spin_history),
            'chi_square': self.get_chi_square_result(),
            'hot_numbers': self.get_hot_numbers(),
            'cold_numbers': self.get_cold_numbers(),
            'bias_score': self.get_bias_score(),
            'color_distribution': self.get_color_distribution()
        }
