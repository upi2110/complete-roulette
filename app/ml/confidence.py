"""
Confidence Scoring Engine — Multi-factor confidence assessment
with Bayesian calibration.

Combines model agreement, historical accuracy, pattern strength,
sample size, and streak momentum.  Then applies isotonic calibration
so that reported confidence matches actual hit rate over time.
"""

import numpy as np
from collections import deque

import sys
sys.path.insert(0, '.')
from config import (
    TOTAL_NUMBERS,
    WEIGHT_MODEL_AGREEMENT, WEIGHT_HISTORICAL_ACCURACY,
    WEIGHT_PATTERN_STRENGTH, WEIGHT_SAMPLE_SIZE, WEIGHT_STREAK_MOMENTUM,
    WEIGHT_RECENT_HIT_RATE,
    CONFIDENCE_BET_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD,
    CALIBRATION_BINS, CALIBRATION_MIN_SAMPLES,
)


class ConfidenceCalibrator:
    """Isotonic-style calibration that maps raw confidence → calibrated confidence.

    Tracks actual hit rate (top-12 number hit) per confidence bin.
    After enough data, the calibrated confidence reflects REALITY —
    if the system says 60% confident but only hits 25% of the time
    at that level, the calibrated score moves toward 25%.
    """

    def __init__(self, n_bins=CALIBRATION_BINS, min_samples=CALIBRATION_MIN_SAMPLES):
        self.n_bins = n_bins
        self.min_samples = min_samples
        # Each bin stores (raw_confidence, did_hit) tuples
        self.bin_edges = np.linspace(0, 100, n_bins + 1)
        self.bin_data = [[] for _ in range(n_bins)]

    def record(self, raw_confidence, did_hit):
        """Record an observation: what raw confidence was, and whether we hit."""
        bin_idx = self._get_bin(raw_confidence)
        self.bin_data[bin_idx].append(1.0 if did_hit else 0.0)
        # Keep bins from growing too large
        if len(self.bin_data[bin_idx]) > 200:
            self.bin_data[bin_idx] = self.bin_data[bin_idx][-200:]

    def calibrate(self, raw_confidence):
        """Map raw confidence → calibrated confidence.

        If a bin has enough samples, uses the actual hit rate.
        Otherwise, blends with the raw score.
        """
        bin_idx = self._get_bin(raw_confidence)
        data = self.bin_data[bin_idx]

        if len(data) < self.min_samples:
            # Not enough data — return raw
            return raw_confidence

        actual_rate = sum(data) / len(data) * 100  # as percentage

        # Blend: weight shifts toward actual_rate as data accumulates
        # At min_samples: 50/50 blend.  At 4× min_samples: 90/10 actual.
        blend = min(0.9, len(data) / (self.min_samples * 4))
        calibrated = actual_rate * blend + raw_confidence * (1 - blend)

        return round(max(0, min(100, calibrated)), 1)

    def _get_bin(self, confidence):
        """Map confidence value to bin index."""
        idx = int(confidence / (100.0 / self.n_bins))
        return max(0, min(self.n_bins - 1, idx))

    def get_state(self):
        """Serializable state."""
        return {
            'bin_data': [list(b) for b in self.bin_data],
            'n_bins': self.n_bins,
        }

    def load_state(self, state):
        """Restore from serialized state."""
        if not state:
            return
        saved_bins = state.get('bin_data', [])
        saved_n = state.get('n_bins', self.n_bins)
        if saved_n == self.n_bins and len(saved_bins) == self.n_bins:
            self.bin_data = [list(b) for b in saved_bins]


class ConfidenceEngine:
    def __init__(self):
        self.prediction_history = deque(maxlen=100)
        self.accuracy_window = deque(maxlen=50)
        self.category_accuracy = {
            'color': deque(maxlen=50),
            'dozen': deque(maxlen=50),
            'column': deque(maxlen=50),
            'high_low': deque(maxlen=50),
            'odd_even': deque(maxlen=50),
            'number': deque(maxlen=50)
        }
        self.calibrator = ConfidenceCalibrator()

    def record_prediction(self, prediction, actual):
        """Record a prediction result for accuracy tracking."""
        record = {
            'predicted_numbers': prediction.get('numbers', []),
            'predicted_color': prediction.get('color'),
            'predicted_dozen': prediction.get('dozen'),
            'predicted_high_low': prediction.get('high_low'),
            'predicted_odd_even': prediction.get('odd_even'),
            'actual': actual,
            'correct_number': actual in prediction.get('numbers', []),
            'correct_color': prediction.get('color') == prediction.get('actual_color'),
            'correct_dozen': prediction.get('dozen') == prediction.get('actual_dozen'),
        }
        self.prediction_history.append(record)

        # Track category accuracies
        if prediction.get('color'):
            self.category_accuracy['color'].append(
                1 if prediction.get('color') == prediction.get('actual_color') else 0
            )
        if prediction.get('dozen'):
            self.category_accuracy['dozen'].append(
                1 if prediction.get('dozen') == prediction.get('actual_dozen') else 0
            )
        if prediction.get('numbers'):
            hit = actual in prediction['numbers']
            self.category_accuracy['number'].append(1 if hit else 0)

            # Feed calibrator with the raw confidence and whether we hit
            raw_conf = prediction.get('_raw_confidence', 50.0)
            self.calibrator.record(raw_conf, hit)

    def calculate_model_agreement(self, probability_distributions):
        """How much do different models agree on the prediction?
        Higher agreement = higher confidence.

        Scoring: random baseline overlap ≈ 27% → score 50 (neutral).
        Below baseline → 0-50, above → 50-100.
        This ensures model_agreement doesn't drag confidence down unfairly.
        """
        if len(probability_distributions) < 2:
            return 50.0

        top_n = 10
        top_sets = []
        for dist in probability_distributions:
            top_indices = set(np.argsort(dist)[-top_n:])
            top_sets.append(top_indices)

        if len(top_sets) < 2:
            return 50.0

        overlaps = []
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                overlap = len(top_sets[i] & top_sets[j]) / top_n
                overlaps.append(overlap)

        avg_overlap = np.mean(overlaps)
        # Random baseline: 10/37 ≈ 0.27 overlap → score 50
        # 0% overlap → 0, 27% → 50, 100% → 100
        baseline = 10.0 / 37.0  # ≈ 0.27
        if avg_overlap <= baseline:
            score = (avg_overlap / baseline) * 50.0
        else:
            score = 50.0 + ((avg_overlap - baseline) / (1.0 - baseline)) * 50.0
        return round(max(0, min(100, score)), 1)

    def calculate_historical_accuracy(self):
        """How accurate have recent predictions been?"""
        if len(self.prediction_history) < 5:
            return 50.0

        weights = {'number': 3.0, 'color': 1.0, 'dozen': 1.5}
        total_weight = 0
        weighted_score = 0

        for cat, weight in weights.items():
            acc_list = list(self.category_accuracy.get(cat, []))
            if len(acc_list) >= 3:
                recent = acc_list[-20:]
                acc = sum(recent) / len(recent)
                weighted_score += acc * weight * 100
                total_weight += weight

        if total_weight == 0:
            return 50.0

        score = weighted_score / total_weight
        return round(max(0, min(100, score)), 1)

    def calculate_sample_size_factor(self, total_spins):
        """More data = more confident in predictions.
        Smooth logarithmic curve instead of step function.
        """
        if total_spins < 5:
            return 10.0
        # Log curve: starts fast, plateaus around 200+ spins
        # log2(5) ≈ 2.3, log2(200) ≈ 7.6, log2(2000) ≈ 11
        import math
        raw = math.log2(max(5, total_spins))
        # Scale: log2(5)=2.3→20, log2(50)=5.6→60, log2(200)=7.6→80, log2(1000)=10→97
        score = 20 + (raw - 2.3) / 7.7 * 77
        return round(max(10, min(97, score)), 1)

    def calculate_streak_momentum(self, spin_history, prediction_results=None):
        """Current momentum based on recent NUMBER prediction success.
        Only counts actual number hits (top-12) — not color/dozen which
        have high random baselines and would inflate the score.
        """
        if not self.prediction_history:
            return 50.0

        recent = list(self.prediction_history)[-10:]
        if not recent:
            return 50.0

        # Count only number hits — these are what matter for straight bets
        number_hits = sum(1 for r in recent if r.get('correct_number'))
        hit_rate = number_hits / len(recent)

        # Scale relative to random baseline (12/37 ≈ 32.4%)
        baseline = 12.0 / 37.0
        if hit_rate <= baseline:
            momentum = (hit_rate / baseline) * 50.0
        else:
            momentum = 50.0 + ((hit_rate - baseline) / (1.0 - baseline)) * 50.0

        return round(max(0, min(100, momentum)), 1)

    def calculate_recent_hit_rate(self):
        """What fraction of recent predictions contained the actual number?
        This is the most direct measure of prediction quality.
        Uses the 'number' category accuracy which tracks top-12 hits.
        """
        number_acc = list(self.category_accuracy.get('number', deque()))
        if len(number_acc) < 3:
            return 50.0  # Neutral default before enough data

        recent = number_acc[-20:]
        hit_rate = sum(recent) / len(recent)

        # Scale to 0-100
        # Baseline for 12/37 coverage ≈ 32.4% random hit rate
        baseline = 12.0 / 37.0
        if hit_rate <= baseline:
            score = (hit_rate / baseline) * 50.0
        else:
            score = 50.0 + ((hit_rate - baseline) / (1.0 - baseline)) * 50.0

        return round(max(0, min(100, score)), 1)

    def calculate_confidence(self, probability_distributions, total_spins,
                             pattern_strength, spin_history=None):
        """Master confidence score combining 6 factors."""
        model_agreement = self.calculate_model_agreement(probability_distributions)
        historical_accuracy = self.calculate_historical_accuracy()
        sample_size = self.calculate_sample_size_factor(total_spins)
        streak_momentum = self.calculate_streak_momentum(spin_history or [])
        recent_hit_rate = self.calculate_recent_hit_rate()

        raw_confidence = (
            model_agreement * WEIGHT_MODEL_AGREEMENT +
            historical_accuracy * WEIGHT_HISTORICAL_ACCURACY +
            pattern_strength * WEIGHT_PATTERN_STRENGTH +
            sample_size * WEIGHT_SAMPLE_SIZE +
            streak_momentum * WEIGHT_STREAK_MOMENTUM +
            recent_hit_rate * WEIGHT_RECENT_HIT_RATE
        )

        raw_confidence = round(max(0, min(100, raw_confidence)), 1)

        # Store raw for calibration feedback
        self._last_raw_confidence = raw_confidence

        # Apply Bayesian calibration
        calibrated = self.calibrator.calibrate(raw_confidence)

        return calibrated

    def get_mode(self, confidence):
        """Determine BET or WAIT mode based on confidence."""
        if confidence >= CONFIDENCE_HIGH_THRESHOLD:
            return 'BET_HIGH'
        elif confidence >= CONFIDENCE_BET_THRESHOLD:
            return 'BET'
        else:
            return 'WAIT'

    def get_breakdown(self, probability_distributions, total_spins,
                      pattern_strength, spin_history=None):
        """Detailed confidence breakdown for UI display."""
        model_agreement = self.calculate_model_agreement(probability_distributions)
        historical_accuracy = self.calculate_historical_accuracy()
        sample_size = self.calculate_sample_size_factor(total_spins)
        streak_momentum = self.calculate_streak_momentum(spin_history or [])
        recent_hit_rate = self.calculate_recent_hit_rate()
        confidence = self.calculate_confidence(
            probability_distributions, total_spins, pattern_strength, spin_history
        )
        raw = getattr(self, '_last_raw_confidence', confidence)

        # Cast all values to native Python types for JSON serialization
        # (numpy.float64 / numpy.bool_ are NOT JSON-serializable)
        return {
            'overall': float(confidence),
            'raw_confidence': float(raw),
            'calibrated': bool(confidence != raw),
            'mode': self.get_mode(confidence),
            'factors': {
                'model_agreement': {
                    'score': float(model_agreement),
                    'weight': float(WEIGHT_MODEL_AGREEMENT),
                    'contribution': float(round(model_agreement * WEIGHT_MODEL_AGREEMENT, 1))
                },
                'historical_accuracy': {
                    'score': float(historical_accuracy),
                    'weight': float(WEIGHT_HISTORICAL_ACCURACY),
                    'contribution': float(round(historical_accuracy * WEIGHT_HISTORICAL_ACCURACY, 1))
                },
                'pattern_strength': {
                    'score': float(pattern_strength),
                    'weight': float(WEIGHT_PATTERN_STRENGTH),
                    'contribution': float(round(pattern_strength * WEIGHT_PATTERN_STRENGTH, 1))
                },
                'sample_size': {
                    'score': float(sample_size),
                    'weight': float(WEIGHT_SAMPLE_SIZE),
                    'contribution': float(round(sample_size * WEIGHT_SAMPLE_SIZE, 1))
                },
                'streak_momentum': {
                    'score': float(streak_momentum),
                    'weight': float(WEIGHT_STREAK_MOMENTUM),
                    'contribution': float(round(streak_momentum * WEIGHT_STREAK_MOMENTUM, 1))
                },
                'recent_hit_rate': {
                    'score': float(recent_hit_rate),
                    'weight': float(WEIGHT_RECENT_HIT_RATE),
                    'contribution': float(round(recent_hit_rate * WEIGHT_RECENT_HIT_RATE, 1))
                }
            },
            'total_predictions': len(self.prediction_history),
            'thresholds': {
                'bet': float(CONFIDENCE_BET_THRESHOLD),
                'high': float(CONFIDENCE_HIGH_THRESHOLD)
            }
        }

    # ─── State persistence for calibration ────────────────────────────

    def get_calibration_state(self):
        """Serialize calibration data for persistence."""
        return self.calibrator.get_state()

    def load_calibration_state(self, state):
        """Restore calibration data from persistence."""
        self.calibrator.load_state(state)
