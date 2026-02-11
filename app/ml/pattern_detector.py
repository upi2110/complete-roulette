"""
Pattern Detector — Analyzes dozens, columns, colors, sectors, streaks,
and various roulette-specific patterns for prediction signals.

Industrial-level additions:
  - Runs test for randomness
  - Serial correlation test
  - Sector autocorrelation
  - Dealer signature detection
"""

import numpy as np
from collections import Counter
from scipy import stats as scipy_stats

import sys
sys.path.insert(0, '.')
from config import (
    TOTAL_NUMBERS, RED_NUMBERS, BLACK_NUMBERS, GREEN_NUMBERS,
    FIRST_DOZEN, SECOND_DOZEN, THIRD_DOZEN,
    FIRST_COLUMN, SECOND_COLUMN, THIRD_COLUMN,
    LOW_NUMBERS, HIGH_NUMBERS, ODD_NUMBERS, EVEN_NUMBERS,
    WHEEL_SECTORS, NUMBER_TO_POSITION, MIN_SPINS_FOR_PATTERNS,
    WHEEL_ORDER,
)


class PatternDetector:
    def __init__(self):
        self.spin_history = []

    def update(self, number):
        self.spin_history.append(number)

    def load_history(self, history):
        self.spin_history = list(history)

    def _get_streak(self, category_fn, lookback=20):
        """Get current streak length for a category function."""
        if not self.spin_history:
            return 0

        recent = self.spin_history[-lookback:]
        if not recent:
            return 0

        current_cat = category_fn(recent[-1])
        streak = 0
        for num in reversed(recent):
            if category_fn(num) == current_cat:
                streak += 1
            else:
                break
        return streak

    def _color_of(self, n):
        if n in RED_NUMBERS:
            return 'red'
        elif n in BLACK_NUMBERS:
            return 'black'
        return 'green'

    def _dozen_of(self, n):
        if n == 0:
            return 'zero'
        if n in FIRST_DOZEN:
            return '1st'
        if n in SECOND_DOZEN:
            return '2nd'
        return '3rd'

    def _column_of(self, n):
        if n == 0:
            return 'zero'
        if n in FIRST_COLUMN:
            return '1st'
        if n in SECOND_COLUMN:
            return '2nd'
        return '3rd'

    def _high_low(self, n):
        if n == 0:
            return 'zero'
        return 'high' if n in HIGH_NUMBERS else 'low'

    def _odd_even(self, n):
        if n == 0:
            return 'zero'
        return 'odd' if n in ODD_NUMBERS else 'even'

    def analyze_color_patterns(self, lookback=30):
        """Analyze red/black/green distribution and streaks."""
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 3:
            return {'red_pct': 0.486, 'black_pct': 0.486, 'streak': 0, 'streak_color': 'none'}

        colors = [self._color_of(n) for n in recent]
        total = len(colors)
        red_count = colors.count('red')
        black_count = colors.count('black')

        # Current color streak
        streak = self._get_streak(self._color_of, lookback)
        streak_color = self._color_of(recent[-1]) if recent else 'none'

        return {
            'red_pct': round(red_count / total, 3),
            'black_pct': round(black_count / total, 3),
            'red_count': red_count,
            'black_count': black_count,
            'streak': streak,
            'streak_color': streak_color,
            'last_colors': colors[-10:]
        }

    def analyze_dozen_patterns(self, lookback=30):
        """Analyze dozen distribution and streaks."""
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 3:
            return {'dominant': 'none', 'streak': 0, 'distribution': {}}

        dozens = [self._dozen_of(n) for n in recent if n != 0]
        if not dozens:
            return {'dominant': 'none', 'streak': 0, 'distribution': {}}

        total = len(dozens)
        dist = Counter(dozens)

        dominant = dist.most_common(1)[0][0]
        streak = self._get_streak(self._dozen_of, lookback)

        return {
            'dominant': dominant,
            'streak': streak,
            'streak_dozen': self._dozen_of(recent[-1]) if recent else 'none',
            'distribution': {k: round(v / total, 3) for k, v in dist.items()},
            'missing_dozen': [d for d in ['1st', '2nd', '3rd'] if d not in dist]
        }

    def analyze_column_patterns(self, lookback=30):
        """Analyze column distribution and streaks."""
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 3:
            return {'dominant': 'none', 'streak': 0, 'distribution': {}}

        columns = [self._column_of(n) for n in recent if n != 0]
        if not columns:
            return {'dominant': 'none', 'streak': 0, 'distribution': {}}

        total = len(columns)
        dist = Counter(columns)
        dominant = dist.most_common(1)[0][0]
        streak = self._get_streak(self._column_of, lookback)

        return {
            'dominant': dominant,
            'streak': streak,
            'distribution': {k: round(v / total, 3) for k, v in dist.items()}
        }

    def analyze_high_low(self, lookback=30):
        """Analyze high/low patterns."""
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 3:
            return {'high_pct': 0.5, 'low_pct': 0.5, 'streak': 0}

        hl = [self._high_low(n) for n in recent if n != 0]
        if not hl:
            return {'high_pct': 0.5, 'low_pct': 0.5, 'streak': 0}

        total = len(hl)
        high_count = hl.count('high')
        streak = self._get_streak(self._high_low, lookback)

        return {
            'high_pct': round(high_count / total, 3),
            'low_pct': round((total - high_count) / total, 3),
            'streak': streak,
            'streak_type': self._high_low(recent[-1]) if recent else 'none'
        }

    def analyze_odd_even(self, lookback=30):
        """Analyze odd/even patterns."""
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 3:
            return {'odd_pct': 0.5, 'even_pct': 0.5, 'streak': 0}

        oe = [self._odd_even(n) for n in recent if n != 0]
        if not oe:
            return {'odd_pct': 0.5, 'even_pct': 0.5, 'streak': 0}

        total = len(oe)
        odd_count = oe.count('odd')
        streak = self._get_streak(self._odd_even, lookback)

        return {
            'odd_pct': round(odd_count / total, 3),
            'even_pct': round((total - odd_count) / total, 3),
            'streak': streak,
            'streak_type': self._odd_even(recent[-1]) if recent else 'none'
        }

    def detect_repeaters(self, lookback=20):
        """Detect numbers that have repeated recently."""
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 5:
            return []

        counts = Counter(recent)
        repeaters = [
            {'number': num, 'times': count, 'rate': round(count / len(recent), 3)}
            for num, count in counts.most_common(5)
            if count >= 2
        ]
        return repeaters

    def detect_sector_patterns(self, lookback=20):
        """Analyze which wheel sectors are hot."""
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 10:
            return {}

        sector_hits = {}
        for sector_name, sector_nums in WHEEL_SECTORS.items():
            hits = sum(1 for n in recent if n in sector_nums)
            expected = len(recent) * len(sector_nums) / TOTAL_NUMBERS
            ratio = hits / expected if expected > 0 else 1.0
            sector_hits[sector_name] = {
                'hits': hits,
                'expected': round(expected, 1),
                'ratio': round(ratio, 2),
                'numbers': sector_nums,
                'hot': ratio >= 1.4
            }

        return sector_hits

    # ─── Statistical Pattern Tests ──────────────────────────────────────

    def runs_test(self, lookback=200):
        """Wald-Wolfowitz runs test for randomness.

        Classifies each spin as above/below median.  Counts runs
        (consecutive sequences of same classification).  Returns
        p-value: low p → sequence is NOT random.
        """
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 20:
            return {'p_value': 1.0, 'significant': False, 'n_runs': 0}

        median = np.median(recent)
        # Convert to binary: above median = 1, at/below = 0
        binary = [1 if x > median else 0 for x in recent]

        n1 = sum(binary)
        n2 = len(binary) - n1
        if n1 == 0 or n2 == 0:
            return {'p_value': 1.0, 'significant': False, 'n_runs': 1}

        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i - 1]:
                runs += 1

        # Expected runs and variance under null hypothesis
        n = len(binary)
        expected_runs = 1 + (2 * n1 * n2) / n
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))

        if variance <= 0:
            return {'p_value': 1.0, 'significant': False, 'n_runs': runs}

        z = (runs - expected_runs) / (variance ** 0.5)
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

        return {
            'p_value': round(float(p_value), 4),
            'significant': p_value < 0.05,
            'n_runs': runs,
            'expected_runs': round(expected_runs, 1),
            'z_score': round(float(z), 3),
        }

    def serial_correlation_test(self, lag=1, lookback=200):
        """Test if consecutive spins are correlated (autocorrelation at given lag).

        Returns: correlation coefficient and p-value.
        Significant correlation → spins are NOT independent.
        """
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 20 + lag:
            return {'correlation': 0.0, 'p_value': 1.0, 'significant': False}

        x = np.array(recent[:-lag], dtype=float)
        y = np.array(recent[lag:], dtype=float)

        if np.std(x) == 0 or np.std(y) == 0:
            return {'correlation': 0.0, 'p_value': 1.0, 'significant': False}

        corr, p_value = scipy_stats.pearsonr(x, y)

        return {
            'correlation': round(float(corr), 4),
            'p_value': round(float(p_value), 4),
            'significant': p_value < 0.05,
            'lag': lag,
        }

    def sector_autocorrelation(self, lookback=100):
        """Does the ball tend to land in the same sector repeatedly?

        Maps each spin to its sector index, then checks autocorrelation.
        """
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 20:
            return {'correlation': 0.0, 'p_value': 1.0, 'significant': False}

        # Map to sector indices
        sector_indices = []
        for num in recent:
            pos = NUMBER_TO_POSITION.get(num, 0)
            sector_idx = pos // 5
            sector_indices.append(sector_idx)

        if len(sector_indices) < 10:
            return {'correlation': 0.0, 'p_value': 1.0, 'significant': False}

        x = np.array(sector_indices[:-1], dtype=float)
        y = np.array(sector_indices[1:], dtype=float)

        if np.std(x) == 0 or np.std(y) == 0:
            return {'correlation': 0.0, 'p_value': 1.0, 'significant': False}

        corr, p_value = scipy_stats.pearsonr(x, y)

        return {
            'correlation': round(float(corr), 4),
            'p_value': round(float(p_value), 4),
            'significant': p_value < 0.05,
        }

    def dealer_signature_test(self, lookback=200):
        """Detect if spin-to-spin differences cluster around certain values.

        If the dealer has a consistent throwing style, the difference
        between consecutive numbers' wheel positions tends to cluster.
        """
        recent = self.spin_history[-lookback:] if self.spin_history else []
        if len(recent) < 30:
            return {'detected': False, 'cluster_strength': 0.0, 'common_gaps': []}

        wheel_len = len(WHEEL_ORDER)
        # Compute positional differences
        diffs = []
        for i in range(1, len(recent)):
            pos1 = NUMBER_TO_POSITION.get(recent[i - 1], 0)
            pos2 = NUMBER_TO_POSITION.get(recent[i], 0)
            diff = (pos2 - pos1) % wheel_len
            diffs.append(diff)

        if not diffs:
            return {'detected': False, 'cluster_strength': 0.0, 'common_gaps': []}

        # If uniform, each gap 0-36 equally likely.  Test with chi-square.
        diff_counts = Counter(diffs)
        observed = np.array([diff_counts.get(i, 0) for i in range(wheel_len)])
        expected = np.full(wheel_len, len(diffs) / wheel_len)

        chi2, p_value = scipy_stats.chisquare(observed, expected)

        # Find most common gaps
        common = diff_counts.most_common(3)
        expected_count = len(diffs) / wheel_len

        return {
            'detected': p_value < 0.05,
            'p_value': round(float(p_value), 4),
            'chi_square': round(float(chi2), 2),
            'cluster_strength': round(max(0, min(100, (1 - p_value) * 100)), 1),
            'common_gaps': [
                {'gap': g, 'count': c, 'ratio': round(c / expected_count, 2)}
                for g, c in common
            ],
        }

    def get_statistical_tests(self):
        """Run all statistical tests and return combined results."""
        return {
            'runs_test': self.runs_test(),
            'serial_correlation': self.serial_correlation_test(),
            'sector_autocorrelation': self.sector_autocorrelation(),
            'dealer_signature': self.dealer_signature_test(),
        }

    # ─── Probability Generation ──────────────────────────────────────

    def get_number_probabilities(self):
        """Generate probability distribution from pattern analysis.

        Now incorporates statistical test results alongside heuristic rules.
        """
        probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        if len(self.spin_history) < MIN_SPINS_FOR_PATTERNS:
            return probs

        # Boost hot sectors
        sectors = self.detect_sector_patterns()
        for sector_data in sectors.values():
            if sector_data['hot']:
                for num in sector_data['numbers']:
                    probs[num] *= 1.3

        # Boost repeaters
        for rep in self.detect_repeaters():
            probs[rep['number']] *= (1 + rep['rate'])

        # Boost from dealer signature (if detected, favour common gap positions)
        if len(self.spin_history) >= 30:
            sig = self.dealer_signature_test()
            if sig['detected'] and self.spin_history:
                last_pos = NUMBER_TO_POSITION.get(self.spin_history[-1], 0)
                wheel_len = len(WHEEL_ORDER)
                for gap_info in sig['common_gaps']:
                    target_pos = (last_pos + gap_info['gap']) % wheel_len
                    target_num = WHEEL_ORDER[target_pos]
                    boost = 1.0 + min(0.3, (gap_info['ratio'] - 1.0) * 0.15)
                    probs[target_num] *= boost

        # Boost from sector autocorrelation (if sectors repeat, favour same sector)
        if len(self.spin_history) >= 20:
            sec_auto = self.sector_autocorrelation()
            if sec_auto['significant'] and sec_auto['correlation'] > 0:
                last_pos = NUMBER_TO_POSITION.get(self.spin_history[-1], 0)
                last_sector = last_pos // 5
                for num in range(TOTAL_NUMBERS):
                    pos = NUMBER_TO_POSITION.get(num, 0)
                    if pos // 5 == last_sector:
                        probs[num] *= 1.15

        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total

        return probs

    def get_pattern_strength(self):
        """Overall pattern strength score 0-100.

        Multi-factor: combines heuristic pattern scores with statistical
        test significance for a robust strength measure.
        """
        if len(self.spin_history) < MIN_SPINS_FOR_PATTERNS:
            return 0.0

        scores = []

        # ── Heuristic scores ──

        # Color streak strength
        color = self.analyze_color_patterns()
        if color['streak'] >= 4:
            scores.append(min(30, color['streak'] * 6))

        # Dozen dominance
        dozen = self.analyze_dozen_patterns()
        if dozen['distribution']:
            max_pct = max(dozen['distribution'].values())
            if max_pct > 0.4:
                scores.append((max_pct - 0.333) * 200)

        # Sector heat
        sectors = self.detect_sector_patterns()
        hot_sectors = sum(1 for s in sectors.values() if s.get('hot'))
        if hot_sectors:
            scores.append(hot_sectors * 15)

        # Repeater strength
        repeaters = self.detect_repeaters()
        if repeaters:
            scores.append(len(repeaters) * 10)

        # ── Statistical test scores ──
        # Each significant test adds to strength (max ~20 per test)
        if len(self.spin_history) >= 20:
            runs = self.runs_test()
            if runs['significant']:
                import math
                neg_log_p = -math.log10(max(runs['p_value'], 1e-10))
                scores.append(min(25, neg_log_p * 8))

            serial = self.serial_correlation_test()
            if serial['significant']:
                scores.append(min(20, abs(serial['correlation']) * 50))

            sec_auto = self.sector_autocorrelation()
            if sec_auto['significant']:
                scores.append(min(20, abs(sec_auto['correlation']) * 50))

            sig = self.dealer_signature_test()
            if sig['detected']:
                scores.append(min(25, sig['cluster_strength'] * 0.25))

        if not scores:
            return 0.0

        return round(min(100, sum(scores) / max(1, len(scores)) * 2), 1)

    def get_bet_suggestions(self):
        """Generate specific bet type suggestions based on patterns."""
        suggestions = []

        if len(self.spin_history) < MIN_SPINS_FOR_PATTERNS:
            return suggestions

        # Color-based suggestion
        color = self.analyze_color_patterns()
        if color['streak'] >= 3:
            # Bet against the streak (mean reversion)
            opposite = 'black' if color['streak_color'] == 'red' else 'red'
            suggestions.append({
                'type': 'red_black',
                'value': opposite,
                'reason': f"{color['streak_color']} streak of {color['streak']} - mean reversion",
                'confidence': min(70, 40 + color['streak'] * 5)
            })

        # Dozen-based suggestion
        dozen = self.analyze_dozen_patterns()
        if dozen.get('missing_dozen'):
            suggestions.append({
                'type': 'dozen',
                'value': dozen['missing_dozen'][0],
                'reason': f"{dozen['missing_dozen'][0]} dozen hasn't appeared recently",
                'confidence': 45
            })
        if dozen.get('dominant') and dozen['dominant'] != 'none':
            dist = dozen.get('distribution', {})
            if dist.get(dozen['dominant'], 0) > 0.4:
                suggestions.append({
                    'type': 'dozen',
                    'value': dozen['dominant'],
                    'reason': f"{dozen['dominant']} dozen is dominant ({dist[dozen['dominant']]:.0%})",
                    'confidence': 55
                })

        # Sector-based suggestion
        sectors = self.detect_sector_patterns()
        hot_sectors = [(name, data) for name, data in sectors.items() if data.get('hot')]
        for name, data in hot_sectors[:2]:
            suggestions.append({
                'type': 'sector',
                'value': name,
                'numbers': data['numbers'],
                'reason': f"Sector {name} is hot (ratio: {data['ratio']})",
                'confidence': min(65, 40 + int(data['ratio'] * 10))
            })

        return suggestions

    def get_summary(self):
        return {
            'color': self.analyze_color_patterns(),
            'dozen': self.analyze_dozen_patterns(),
            'column': self.analyze_column_patterns(),
            'high_low': self.analyze_high_low(),
            'odd_even': self.analyze_odd_even(),
            'repeaters': self.detect_repeaters(),
            'sectors': self.detect_sector_patterns(),
            'pattern_strength': self.get_pattern_strength(),
            'bet_suggestions': self.get_bet_suggestions(),
            'statistical_tests': self.get_statistical_tests(),
        }
