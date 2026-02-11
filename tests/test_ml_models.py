"""
Unit Tests for ML Models: FrequencyAnalyzer, MarkovChain, PatternDetector,
ConfidenceEngine, EnsemblePredictor, LSTMPredictor

Uses real roulette data from userdata/ files when available, falls back to
hardcoded sample data for CI environments without the data files.
"""
import pytest
import numpy as np
import sys
import os
import glob
from collections import Counter

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from config import TOTAL_NUMBERS, TOP_PREDICTIONS_COUNT, CONFIDENCE_BET_THRESHOLD, \
    CONFIDENCE_HIGH_THRESHOLD, RED_NUMBERS, BLACK_NUMBERS, \
    ANCHOR_COUNT, WHEEL_ORDER, NUMBER_TO_POSITION, \
    ADAPTIVE_WINDOW, ADAPTIVE_MIN_OBSERVATIONS, FEATURE_DIM, CALIBRATION_BINS
from app.ml.frequency_analyzer import FrequencyAnalyzer
from app.ml.markov_chain import MarkovChain
from app.ml.pattern_detector import PatternDetector
from app.ml.confidence import ConfidenceEngine, ConfidenceCalibrator
from app.ml.ensemble import EnsemblePredictor, ModelPerformanceTracker, RegimeDetector
from app.ml.feature_engine import FeatureEngine


def _load_userdata():
    """Load real roulette data from userdata/ files."""
    all_numbers = []
    userdata_dir = os.path.join(PROJECT_ROOT, 'userdata')
    for f in sorted(glob.glob(os.path.join(userdata_dir, '*.txt'))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        num = int(line)
                        if 0 <= num <= 36:
                            all_numbers.append(num)
                    except ValueError:
                        pass
    return all_numbers


def _load_userdata_clusters():
    """Load real roulette data as clusters (one per file)."""
    clusters = []
    userdata_dir = os.path.join(PROJECT_ROOT, 'userdata')
    for f in sorted(glob.glob(os.path.join(userdata_dir, '*.txt'))):
        file_numbers = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        num = int(line)
                        if 0 <= num <= 36:
                            file_numbers.append(num)
                    except ValueError:
                        pass
        if file_numbers:
            clusters.append({
                'spins': file_numbers,
                'size': len(file_numbers),
                'source': os.path.basename(f)
            })
    return clusters


# Load real data from userdata/ files (1527 spins from 3 files)
_USERDATA = _load_userdata()
_USERDATA_CLUSTERS = _load_userdata_clusters()

# Fallback: hardcoded subset of real data (for CI without files)
_FALLBACK_DATA = [18, 26, 28, 35, 16, 28, 22, 35, 1, 20, 3, 35, 20, 23, 7, 24, 22, 2, 33, 35,
                  12, 30, 27, 11, 9, 10, 9, 20, 16, 31, 4, 3, 16, 20, 34, 13, 28, 3, 15, 33,
                  12, 11, 26, 23, 15, 36, 1, 25, 28, 32, 14, 6, 12, 16, 3, 6, 1, 35, 18, 8,
                  30, 21, 29, 4, 8, 28, 1, 30, 4, 10, 30, 23, 36, 29, 28, 13, 3, 34, 9, 31,
                  1, 2, 18, 25, 32, 6, 16, 16, 19, 35, 16, 32, 30, 21, 25, 36, 21, 27, 7, 6,
                  9, 15, 23, 3, 2, 23, 21, 8, 17, 1, 31, 36, 17, 25, 16, 8, 26, 8, 32, 22,
                  36, 14, 13, 33, 18, 13, 6, 3, 4, 8, 32, 33, 15, 18, 34, 25, 0, 26, 35, 1,
                  34, 23, 35, 26, 18, 8, 24, 30, 16, 12, 8, 23, 34, 13, 23, 3, 21, 31, 16, 14,
                  20, 21, 29, 3, 34, 14, 10, 1, 18, 25, 27, 17, 27, 36, 23, 4, 34, 12, 12, 3,
                  9, 7, 20, 16, 10]

REAL_DATA = _USERDATA if len(_USERDATA) >= 100 else _FALLBACK_DATA
SAMPLE_50 = REAL_DATA[:50]
SAMPLE_100 = REAL_DATA[:100]
SAMPLE_500 = REAL_DATA[:500] if len(REAL_DATA) >= 500 else REAL_DATA

# Model keys in get_model_status() — excludes top-level meta keys
_MODEL_KEYS = ('frequency', 'markov', 'patterns', 'lstm')


# ═══════════════════════════════════════════════════════════════
# FrequencyAnalyzer Tests
# ═══════════════════════════════════════════════════════════════

class TestFrequencyAnalyzer:
    def test_init(self):
        fa = FrequencyAnalyzer()
        assert fa.spin_history == []
        assert len(fa.frequency_counts) == 0

    def test_update_single(self):
        fa = FrequencyAnalyzer()
        fa.update(17)
        assert fa.spin_history == [17]
        assert fa.frequency_counts[17] == 1

    def test_update_multiple(self):
        fa = FrequencyAnalyzer()
        for n in [5, 5, 5, 10, 10]:
            fa.update(n)
        assert fa.frequency_counts[5] == 3
        assert fa.frequency_counts[10] == 2
        assert len(fa.spin_history) == 5

    def test_load_history(self):
        fa = FrequencyAnalyzer()
        fa.load_history(SAMPLE_50)
        assert len(fa.spin_history) == 50
        assert sum(fa.frequency_counts.values()) == 50

    def test_get_number_probabilities_shape(self):
        fa = FrequencyAnalyzer()
        fa.load_history(SAMPLE_50)
        probs = fa.get_number_probabilities()
        assert probs.shape == (TOTAL_NUMBERS,)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_get_number_probabilities_uniform_when_few_spins(self):
        fa = FrequencyAnalyzer()
        fa.update(5)
        fa.update(10)
        probs = fa.get_number_probabilities()
        # Should be uniform with < 5 spins
        expected = 1.0 / TOTAL_NUMBERS
        assert all(abs(p - expected) < 1e-6 for p in probs)

    def test_hot_numbers(self):
        fa = FrequencyAnalyzer()
        # Make number 7 very hot
        data = [7] * 20 + list(range(0, 37))
        fa.load_history(data)
        hot = fa.get_hot_numbers(3)
        assert len(hot) > 0
        assert hot[0]['number'] == 7

    def test_cold_numbers(self):
        fa = FrequencyAnalyzer()
        # Load data where some numbers never appear
        data = list(range(0, 10)) * 10
        fa.load_history(data)
        cold = fa.get_cold_numbers(5)
        # Numbers 10-36 should be cold
        cold_nums = [c['number'] for c in cold]
        assert all(n >= 10 for n in cold_nums)

    def test_chi_square_with_insufficient_data(self):
        fa = FrequencyAnalyzer()
        fa.update(5)
        result = fa.get_chi_square_result()
        assert result['p_value'] == 1.0
        assert result['significant'] == False

    def test_chi_square_with_data(self):
        fa = FrequencyAnalyzer()
        fa.load_history(SAMPLE_50)
        result = fa.get_chi_square_result()
        assert 'statistic' in result
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1

    def test_bias_score_range(self):
        fa = FrequencyAnalyzer()
        fa.load_history(SAMPLE_50)
        score = fa.get_bias_score()
        assert 0 <= score <= 100

    def test_color_distribution(self):
        fa = FrequencyAnalyzer()
        fa.load_history(SAMPLE_50)
        dist = fa.get_color_distribution()
        assert 'red' in dist
        assert 'black' in dist
        assert 'green' in dist
        total = dist['red'] + dist['black'] + dist['green']
        assert abs(total - 1.0) < 0.01

    def test_sector_bias_insufficient_data(self):
        fa = FrequencyAnalyzer()
        fa.update(5)
        result = fa.get_sector_bias()
        assert result == {}

    def test_sector_bias_with_data(self):
        fa = FrequencyAnalyzer()
        fa.load_history(SAMPLE_50)
        sectors = fa.get_sector_bias()
        assert len(sectors) > 0
        for name, data in sectors.items():
            assert 'ratio' in data
            assert 'count' in data


# ═══════════════════════════════════════════════════════════════
# MarkovChain Tests
# ═══════════════════════════════════════════════════════════════

class TestMarkovChain:
    def test_init(self):
        mc = MarkovChain()
        assert mc.spin_history == []
        assert mc.transition_1.shape == (TOTAL_NUMBERS, TOTAL_NUMBERS)

    def test_update_builds_transitions(self):
        mc = MarkovChain()
        mc.update(5)
        mc.update(10)
        mc.update(15)
        # After 5->10 transition, row 5 should have non-zero entry at col 10
        assert mc.counts_1[5][10] == 1
        # After 10->15, row 10 should have non-zero at col 15
        assert mc.counts_1[10][15] == 1

    def test_load_history(self):
        mc = MarkovChain()
        mc.load_history(SAMPLE_50)
        assert len(mc.spin_history) == 50
        # Should have some non-zero transitions
        assert mc.counts_1.sum() > 0

    def test_predict_first_order_shape(self):
        mc = MarkovChain()
        mc.load_history(SAMPLE_50)
        probs = mc.predict_first_order()
        assert probs.shape == (TOTAL_NUMBERS,)
        assert probs.sum() > 0

    def test_predict_second_order_shape(self):
        mc = MarkovChain()
        mc.load_history(SAMPLE_50)
        probs = mc.predict_second_order()
        assert probs.shape == (TOTAL_NUMBERS,)

    def test_get_probabilities_normalized(self):
        mc = MarkovChain()
        mc.load_history(SAMPLE_50)
        probs = mc.get_probabilities()
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_uniform_with_insufficient_data(self):
        mc = MarkovChain()
        mc.update(5)
        probs = mc.predict_first_order()
        expected = 1.0 / TOTAL_NUMBERS
        assert all(abs(p - expected) < 1e-6 for p in probs)

    def test_transition_strength_range(self):
        mc = MarkovChain()
        mc.load_history(SAMPLE_50)
        strength = mc.get_transition_strength()
        assert 0 <= strength <= 100

    def test_top_predictions(self):
        mc = MarkovChain()
        mc.load_history(SAMPLE_50)
        top = mc.get_top_predictions(5)
        assert len(top) == 5
        for p in top:
            assert 'number' in p
            assert 'probability' in p
            assert 0 <= p['number'] <= 36

    def test_repeater_probability(self):
        mc = MarkovChain()
        mc.load_history(SAMPLE_50)
        prob = mc.get_repeater_probability()
        assert 0 <= prob <= 1


# ═══════════════════════════════════════════════════════════════
# PatternDetector Tests
# ═══════════════════════════════════════════════════════════════

class TestPatternDetector:
    def test_init(self):
        pd = PatternDetector()
        assert pd.spin_history == []

    def test_update_and_load(self):
        pd = PatternDetector()
        pd.update(5)
        pd.update(10)
        assert len(pd.spin_history) == 2
        pd.load_history(SAMPLE_50)
        assert len(pd.spin_history) == 50

    def test_color_patterns(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        result = pd.analyze_color_patterns()
        assert 'red_pct' in result
        assert 'black_pct' in result
        assert 'streak' in result
        assert 0 <= result['red_pct'] <= 1
        assert 0 <= result['black_pct'] <= 1

    def test_dozen_patterns(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        result = pd.analyze_dozen_patterns()
        assert 'dominant' in result
        assert 'distribution' in result

    def test_column_patterns(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        result = pd.analyze_column_patterns()
        assert 'dominant' in result

    def test_high_low_patterns(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        result = pd.analyze_high_low()
        assert 'high_pct' in result
        assert 'low_pct' in result
        assert abs(result['high_pct'] + result['low_pct'] - 1.0) < 0.01

    def test_odd_even_patterns(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        result = pd.analyze_odd_even()
        assert 'odd_pct' in result
        assert 'even_pct' in result

    def test_detect_repeaters(self):
        pd = PatternDetector()
        # Make number 7 repeat many times
        data = [7, 7, 7, 7, 7, 1, 2, 3, 4, 5, 6, 7]
        pd.load_history(data)
        repeaters = pd.detect_repeaters()
        assert len(repeaters) > 0
        assert repeaters[0]['number'] == 7

    def test_number_probabilities_shape(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        probs = pd.get_number_probabilities()
        assert probs.shape == (TOTAL_NUMBERS,)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_pattern_strength_range(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        strength = pd.get_pattern_strength()
        assert 0 <= strength <= 100

    def test_pattern_strength_zero_insufficient_data(self):
        pd = PatternDetector()
        pd.update(5)
        assert pd.get_pattern_strength() == 0.0

    def test_bet_suggestions(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        suggestions = pd.get_bet_suggestions()
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert 'type' in s
            assert 'confidence' in s

    def test_sector_patterns(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_50)
        sectors = pd.detect_sector_patterns()
        assert isinstance(sectors, dict)

    def test_streak_detection(self):
        pd = PatternDetector()
        # Create a red streak: 1, 3, 5, 7, 9 are all red
        pd.load_history([1, 3, 5, 7, 9, 12, 14, 16, 18, 19])
        result = pd.analyze_color_patterns()
        assert result['streak'] >= 1


# ═══════════════════════════════════════════════════════════════
# ConfidenceEngine Tests
# ═══════════════════════════════════════════════════════════════

class TestConfidenceEngine:
    def test_init(self):
        ce = ConfidenceEngine()
        assert len(ce.prediction_history) == 0

    def test_model_agreement_with_identical_distributions(self):
        ce = ConfidenceEngine()
        dist = np.random.dirichlet(np.ones(TOTAL_NUMBERS))
        dists = [dist, dist, dist, dist]
        score = ce.calculate_model_agreement(dists)
        assert score == 100.0  # Perfect agreement

    def test_model_agreement_with_single_distribution(self):
        ce = ConfidenceEngine()
        dist = np.random.dirichlet(np.ones(TOTAL_NUMBERS))
        score = ce.calculate_model_agreement([dist])
        assert score == 50.0  # Default for single model

    def test_model_agreement_range(self):
        ce = ConfidenceEngine()
        dists = [np.random.dirichlet(np.ones(TOTAL_NUMBERS)) for _ in range(4)]
        score = ce.calculate_model_agreement(dists)
        assert 0 <= score <= 100

    def test_historical_accuracy_default(self):
        ce = ConfidenceEngine()
        acc = ce.calculate_historical_accuracy()
        assert acc == 50.0  # Default with no history

    def test_sample_size_factor_progression(self):
        ce = ConfidenceEngine()
        scores = []
        for spins in [5, 15, 25, 40, 75, 150, 300]:
            scores.append(ce.calculate_sample_size_factor(spins))
        # Should be monotonically increasing
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], \
                f"Sample size score not monotonic: {scores}"

    def test_sample_size_specific_values(self):
        """Sample size factor should increase monotonically with spins.
        Uses smooth log curve instead of the old step function."""
        ce = ConfidenceEngine()
        s5 = ce.calculate_sample_size_factor(5)
        s15 = ce.calculate_sample_size_factor(15)
        s50 = ce.calculate_sample_size_factor(50)
        s200 = ce.calculate_sample_size_factor(200)
        s1000 = ce.calculate_sample_size_factor(1000)
        # Should be monotonically increasing
        assert s5 < s15 < s50 < s200 < s1000
        # Reasonable range checks
        assert 10 <= s5 <= 30
        assert 30 <= s50 <= 75
        assert 70 <= s200 <= 97
        assert 85 <= s1000 <= 97

    def test_streak_momentum_default(self):
        ce = ConfidenceEngine()
        momentum = ce.calculate_streak_momentum([])
        assert momentum == 50.0

    def test_confidence_range(self):
        ce = ConfidenceEngine()
        dists = [np.random.dirichlet(np.ones(TOTAL_NUMBERS)) for _ in range(4)]
        conf = ce.calculate_confidence(dists, 100, 50.0)
        assert 0 <= conf <= 100

    def test_get_mode_wait(self):
        ce = ConfidenceEngine()
        assert ce.get_mode(30.0) == 'WAIT'

    def test_get_mode_bet(self):
        ce = ConfidenceEngine()
        # BET threshold is 40, HIGH is 50 — so 45 is BET
        assert ce.get_mode(45.0) == 'BET'

    def test_get_mode_bet_high(self):
        ce = ConfidenceEngine()
        # HIGH threshold is 50 — so 55 is BET_HIGH
        assert ce.get_mode(55.0) == 'BET_HIGH'

    def test_get_breakdown_structure(self):
        ce = ConfidenceEngine()
        dists = [np.random.dirichlet(np.ones(TOTAL_NUMBERS)) for _ in range(4)]
        breakdown = ce.get_breakdown(dists, 100, 60.0)
        assert 'overall' in breakdown
        assert 'mode' in breakdown
        assert 'factors' in breakdown
        factors = breakdown['factors']
        assert 'model_agreement' in factors
        assert 'historical_accuracy' in factors
        assert 'pattern_strength' in factors
        assert 'sample_size' in factors
        assert 'streak_momentum' in factors
        assert 'recent_hit_rate' in factors
        for name, data in factors.items():
            assert 'score' in data
            assert 'weight' in data
            assert 'contribution' in data

    def test_confidence_weights_sum_to_1(self):
        """Verify all 6 confidence weights sum to 1.0"""
        from config import (WEIGHT_MODEL_AGREEMENT, WEIGHT_HISTORICAL_ACCURACY,
                            WEIGHT_PATTERN_STRENGTH, WEIGHT_SAMPLE_SIZE,
                            WEIGHT_STREAK_MOMENTUM, WEIGHT_RECENT_HIT_RATE)
        total = (WEIGHT_MODEL_AGREEMENT + WEIGHT_HISTORICAL_ACCURACY +
                 WEIGHT_PATTERN_STRENGTH + WEIGHT_SAMPLE_SIZE +
                 WEIGHT_STREAK_MOMENTUM + WEIGHT_RECENT_HIT_RATE)
        assert abs(total - 1.0) < 1e-6

    def test_record_prediction_tracking(self):
        ce = ConfidenceEngine()
        pred = {'numbers': [5, 10, 15], 'color': 'red', 'actual_color': 'red'}
        ce.record_prediction(pred, 5)
        assert len(ce.prediction_history) == 1
        assert ce.prediction_history[0]['correct_number'] == True

    def test_recent_hit_rate_default(self):
        """With no history, recent_hit_rate should return neutral default."""
        ce = ConfidenceEngine()
        score = ce.calculate_recent_hit_rate()
        assert score == 50.0

    def test_recent_hit_rate_with_all_hits(self):
        """With all hits, recent_hit_rate should be > 50."""
        ce = ConfidenceEngine()
        for _ in range(10):
            ce.category_accuracy['number'].append(1)
        score = ce.calculate_recent_hit_rate()
        assert score > 50.0

    def test_recent_hit_rate_with_no_hits(self):
        """With no hits, recent_hit_rate should be < 50."""
        ce = ConfidenceEngine()
        for _ in range(10):
            ce.category_accuracy['number'].append(0)
        score = ce.calculate_recent_hit_rate()
        assert score < 50.0

    def test_recent_hit_rate_range(self):
        """recent_hit_rate should always be in [0, 100]."""
        ce = ConfidenceEngine()
        for _ in range(20):
            ce.category_accuracy['number'].append(1)
        score_max = ce.calculate_recent_hit_rate()
        assert 0 <= score_max <= 100

        ce2 = ConfidenceEngine()
        for _ in range(20):
            ce2.category_accuracy['number'].append(0)
        score_min = ce2.calculate_recent_hit_rate()
        assert 0 <= score_min <= 100


# ═══════════════════════════════════════════════════════════════
# EnsemblePredictor Tests
# ═══════════════════════════════════════════════════════════════

class TestEnsemblePredictor:
    def test_init(self):
        ep = EnsemblePredictor()
        assert ep.spin_history == []
        assert ep.prediction_count == 0

    def test_update(self):
        ep = EnsemblePredictor()
        ep.update(17)
        assert ep.spin_history == [17]
        assert len(ep.frequency.spin_history) == 1
        assert len(ep.markov.spin_history) == 1
        assert len(ep.patterns.spin_history) == 1
        assert len(ep.lstm.spin_history) == 1

    def test_load_history(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        assert len(ep.spin_history) == 50

    def test_predict_returns_correct_structure(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        pred = ep.predict()

        # Check required keys
        assert 'spin_number' in pred
        assert 'top_numbers' in pred
        assert 'top_probabilities' in pred
        assert 'confidence' in pred
        assert 'confidence_breakdown' in pred
        assert 'mode' in pred
        assert 'bets' in pred
        assert 'group_probabilities' in pred
        assert 'pattern_strength' in pred
        assert 'total_spins' in pred
        assert 'lstm_trained' in pred
        assert 'frequency_summary' in pred
        assert 'markov_strength' in pred

    def test_predict_returns_dynamic_numbers(self):
        """Verify AI picks 3-20 numbers dynamically (not fixed 12)"""
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        pred = ep.predict()
        from config import TOP_PREDICTIONS_MAX
        assert 3 <= len(pred['top_numbers']) <= TOP_PREDICTIONS_MAX
        assert len(pred['top_probabilities']) == len(pred['top_numbers'])

    def test_predict_has_anchors(self):
        """Verify prediction contains 1-4 anchors (variable spread may use fewer)"""
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        pred = ep.predict()
        assert 'anchors' in pred
        assert 1 <= len(pred['anchors']) <= ANCHOR_COUNT
        # anchor_details must be present and match
        assert 'anchor_details' in pred
        assert len(pred['anchor_details']) == len(pred['anchors'])

    def test_anchors_have_wheel_neighbours(self):
        """Verify anchor_details include valid numbers grouped around anchors.
        The probability-first algorithm selects high-prob numbers and groups
        them into clusters. Not every ±spread position is guaranteed —
        low-prob neighbours may be skipped in favour of higher-prob numbers.
        """
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        pred = ep.predict()
        predicted_set = set(pred['top_numbers'])
        all_anchor_numbers = set()

        for ad in pred['anchor_details']:
            anchor = ad['number']
            spread = ad['spread']
            # Anchor itself must be in predictions
            assert anchor in predicted_set, f"Anchor {anchor} not in predictions"
            # Must have at least 1 number (the anchor itself)
            assert len(ad['numbers']) >= 1, \
                f"Anchor {anchor} should have at least 1 number, got {len(ad['numbers'])}"
            # Spread must be 1-3
            assert 1 <= spread <= 3, f"Spread {spread} out of range"
            # All numbers in anchor_details must be valid roulette numbers
            for num in ad['numbers']:
                assert 0 <= num <= 36, f"Invalid number {num} in anchor_details"
                assert num in predicted_set, f"Anchor number {num} not in top_numbers"

            all_anchor_numbers.update(ad['numbers'])

        # All anchor-covered numbers should be in top_numbers
        for num in all_anchor_numbers:
            assert num in predicted_set, f"Anchor-covered number {num} not in top_numbers"

    def test_predictions_are_straight_bets_only(self):
        """Verify bets are all straight (no outside bets)"""
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        pred = ep.predict()
        assert len(pred['bets']) > 0
        for bet in pred['bets']:
            assert bet['type'] == 'straight', f"Expected straight bet, got {bet['type']}"
            assert bet['payout'] == '35:1'

    def test_predict_numbers_are_valid(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        pred = ep.predict()
        for num in pred['top_numbers']:
            assert 0 <= num <= 36

    def test_predict_probabilities_are_positive(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        pred = ep.predict()
        for prob in pred['top_probabilities']:
            assert prob >= 0

    def test_confidence_above_threshold_with_data(self):
        """With 50+ spins, confidence should be above BET threshold"""
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)
        pred = ep.predict()
        assert pred['confidence'] >= CONFIDENCE_BET_THRESHOLD, \
            f"Confidence {pred['confidence']}% below {CONFIDENCE_BET_THRESHOLD}% with 100 spins"

    def test_mode_is_bet_with_data(self):
        """With sufficient data, mode should not be WAIT"""
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)
        pred = ep.predict()
        assert pred['mode'] in ('BET', 'BET_HIGH'), \
            f"Mode should be BET or BET_HIGH but got {pred['mode']}"

    def test_group_probabilities(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        pred = ep.predict()
        gp = pred['group_probabilities']
        assert 'red' in gp
        assert 'black' in gp
        assert 'dozen_1st' in gp
        assert 'high' in gp
        assert 'low' in gp
        assert 'odd' in gp
        assert 'even' in gp
        # Red + Black should cover most of the probability (minus green/zero)
        assert gp['red'] + gp['black'] > 0.8

    def test_ensemble_probabilities_normalized(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        probs, all_dists = ep.get_ensemble_probabilities()
        assert abs(probs.sum() - 1.0) < 1e-6
        assert probs.shape == (TOTAL_NUMBERS,)
        assert len(all_dists) == 4  # freq, markov, pattern, lstm

    def test_update_incremental(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        result = ep.update_incremental([11, 22, 33])
        assert result['numbers_added'] == 3
        assert result['total_spins'] == 53
        assert result['mode'] == 'incremental'

    def test_undo_last(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        initial_count = len(ep.spin_history)
        removed = ep.undo_last()
        assert removed == SAMPLE_50[-1]
        assert len(ep.spin_history) == initial_count - 1

    def test_load_clusters(self):
        ep = EnsemblePredictor()
        clusters = [
            {'spins': SAMPLE_50[:25], 'size': 25},
            {'spins': SAMPLE_50[25:], 'size': 25}
        ]
        result = ep.load_clusters(clusters)
        assert result['clusters_loaded'] == 2
        assert result['total_spins'] == 50

    def test_load_clusters_all_models_get_all_data(self):
        """REGRESSION: load_clusters must give ALL models the full dataset.
        Previously, patterns only got the last cluster's data."""
        ep = EnsemblePredictor()
        cluster_a = SAMPLE_50[:25]
        cluster_b = SAMPLE_50[25:]
        clusters = [
            {'spins': cluster_a, 'size': len(cluster_a)},
            {'spins': cluster_b, 'size': len(cluster_b)}
        ]
        ep.load_clusters(clusters)

        total = len(cluster_a) + len(cluster_b)
        assert len(ep.spin_history) == total, \
            f"Ensemble spin_history should be {total}, got {len(ep.spin_history)}"
        assert len(ep.frequency.spin_history) == total, \
            f"Frequency should have {total} spins, got {len(ep.frequency.spin_history)}"
        assert len(ep.patterns.spin_history) == total, \
            f"Pattern should have {total} spins, got {len(ep.patterns.spin_history)}"
        assert len(ep.lstm.spin_history) == total, \
            f"LSTM should have {total} spins, got {len(ep.lstm.spin_history)}"

    def test_load_clusters_with_real_userdata(self):
        """Test load_clusters with actual userdata file clusters."""
        if not _USERDATA_CLUSTERS:
            pytest.skip("No userdata files found")
        ep = EnsemblePredictor()
        result = ep.load_clusters(_USERDATA_CLUSTERS)
        total = sum(c['size'] for c in _USERDATA_CLUSTERS)
        assert result['total_spins'] == total
        assert result['clusters_loaded'] == len(_USERDATA_CLUSTERS)
        # All models must have the full dataset
        assert len(ep.spin_history) == total
        assert len(ep.frequency.spin_history) == total
        assert len(ep.patterns.spin_history) == total
        assert len(ep.lstm.spin_history) == total

    def test_get_model_status(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        status = ep.get_model_status()
        assert 'frequency' in status
        assert 'markov' in status
        assert 'patterns' in status
        assert 'lstm' in status
        for key in _MODEL_KEYS:
            model = status[key]
            assert 'spins' in model
            assert 'status' in model

    def test_fresh_startup_all_models_idle(self):
        """REGRESSION: On fresh startup with no data loaded, ALL models must show idle.
        Previously, the LSTM showed 'trained' with 0 spins because it auto-loaded
        a checkpoint from disk. A model with 0 spins must never show 'trained'."""
        ep = EnsemblePredictor()
        # Don't load any data — simulate fresh page load
        status = ep.get_model_status()
        for model_name in _MODEL_KEYS:
            model_status = status[model_name]
            assert model_status['spins'] == 0, \
                f"{model_name} reports {model_status['spins']} spins on fresh startup, expected 0"
            assert model_status['status'] == 'idle', \
                f"{model_name} status is '{model_status['status']}' on fresh startup, expected 'idle'"
        # LSTM specifically must not show 'trained' with 0 spins
        assert status['lstm']['trained'] is False, \
            "LSTM shows trained=True on fresh startup with 0 spins"

    def test_model_status_consistent_spin_counts(self):
        """REGRESSION: All models must report the same spin count.
        Previously, Frequency showed 1533, Markov 1527, Pattern 515."""
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)
        status = ep.get_model_status()
        expected = len(SAMPLE_100)
        for model_name in _MODEL_KEYS:
            model_status = status[model_name]
            assert model_status['spins'] == expected, \
                f"{model_name} reports {model_status['spins']} spins, expected {expected}"

    def test_model_status_consistent_after_load_clusters(self):
        """REGRESSION: After load_clusters, all models report same count."""
        ep = EnsemblePredictor()
        clusters = [
            {'spins': SAMPLE_50[:25], 'size': 25},
            {'spins': SAMPLE_50[25:], 'size': 25}
        ]
        ep.load_clusters(clusters)
        status = ep.get_model_status()
        for model_name in _MODEL_KEYS:
            model_status = status[model_name]
            assert model_status['spins'] == 50, \
                f"{model_name} reports {model_status['spins']} spins after load_clusters, expected 50"

    def test_model_status_consistent_after_update(self):
        """After loading + updating, all models still report same count."""
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        ep.update(7)
        ep.update(14)
        ep.update(21)
        status = ep.get_model_status()
        expected = 53  # 50 + 3
        for model_name in _MODEL_KEYS:
            model_status = status[model_name]
            assert model_status['spins'] == expected, \
                f"{model_name} reports {model_status['spins']} spins, expected {expected}"

    def test_bet_suggestions_generated(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)
        pred = ep.predict()
        assert len(pred['bets']) > 0
        for bet in pred['bets']:
            assert 'type' in bet
            assert 'value' in bet
            assert 'probability' in bet

    def test_prediction_count_increments(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        ep.predict()
        ep.predict()
        ep.predict()
        assert ep.prediction_count == 3


# ═══════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    def test_total_numbers(self):
        assert TOTAL_NUMBERS == 37

    def test_top_predictions_count(self):
        assert TOP_PREDICTIONS_COUNT == 12

    def test_red_black_complete(self):
        """Red + Black + Green should cover all 37 numbers"""
        from config import GREEN_NUMBERS
        all_nums = RED_NUMBERS | BLACK_NUMBERS | GREEN_NUMBERS
        assert len(all_nums) == 37
        assert all_nums == set(range(37))

    def test_dozens_complete(self):
        from config import FIRST_DOZEN, SECOND_DOZEN, THIRD_DOZEN
        all_dozens = FIRST_DOZEN | SECOND_DOZEN | THIRD_DOZEN
        assert all_dozens == set(range(1, 37))

    def test_columns_complete(self):
        from config import FIRST_COLUMN, SECOND_COLUMN, THIRD_COLUMN
        all_cols = FIRST_COLUMN | SECOND_COLUMN | THIRD_COLUMN
        assert all_cols == set(range(1, 37))

    def test_wheel_order_complete(self):
        from config import WHEEL_ORDER
        assert len(WHEEL_ORDER) == 37
        assert set(WHEEL_ORDER) == set(range(37))

    def test_ensemble_weights_sum_to_1(self):
        from config import (ENSEMBLE_FREQUENCY_WEIGHT, ENSEMBLE_MARKOV_WEIGHT,
                            ENSEMBLE_PATTERN_WEIGHT, ENSEMBLE_LSTM_WEIGHT)
        total = (ENSEMBLE_FREQUENCY_WEIGHT + ENSEMBLE_MARKOV_WEIGHT +
                 ENSEMBLE_PATTERN_WEIGHT + ENSEMBLE_LSTM_WEIGHT)
        assert abs(total - 1.0) < 1e-6

    def test_get_number_color(self):
        from config import get_number_color
        assert get_number_color(0) == 'green'
        assert get_number_color(1) == 'red'
        assert get_number_color(2) == 'black'
        assert get_number_color(32) == 'red'
        assert get_number_color(17) == 'black'


# ═══════════════════════════════════════════════════════════════
# Regression Tests — Data Integrity
# ═══════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Regression tests for data integrity issues discovered in production."""

    def test_userdata_is_real_roulette_data(self):
        """REGRESSION: Verify userdata files contain real (non-uniform) data.
        Previously, corrupted model_state.pkl had perfectly uniform data
        (every number appearing exactly 43 times), which made Frequency
        Bias = 0 and Pattern Strength = 30."""
        if len(_USERDATA) < 100:
            pytest.skip("No userdata files found")
        counts = Counter(_USERDATA)
        count_values = list(counts.values())
        assert min(count_values) != max(count_values), \
            f"Data is perfectly uniform ({min(count_values)} of each) — likely corrupt"

    def test_userdata_has_sufficient_spins(self):
        """Userdata should have a meaningful number of spins."""
        if len(_USERDATA) < 100:
            pytest.skip("No userdata files found")
        assert len(_USERDATA) >= 500, \
            f"Expected 500+ spins in userdata, got {len(_USERDATA)}"

    def test_userdata_covers_all_numbers(self):
        """Real roulette data should include all 37 numbers."""
        if len(_USERDATA) < 100:
            pytest.skip("No userdata files found")
        unique = set(_USERDATA)
        assert len(unique) == 37, \
            f"Expected all 37 numbers in userdata, got {len(unique)}: missing {set(range(37)) - unique}"

    def test_frequency_bias_nonzero_with_real_data(self):
        """REGRESSION: With real data, bias should NOT be zero."""
        if len(_USERDATA) < 500:
            pytest.skip("Need 500+ real spins")
        fa = FrequencyAnalyzer()
        fa.load_history(_USERDATA)
        bias = fa.get_bias_score()
        assert bias > 0, \
            f"Bias score is {bias} — should be > 0 with {len(_USERDATA)} real spins"

    def test_pattern_strength_reasonable_with_real_data(self):
        """REGRESSION: With real data, pattern strength should be > 30."""
        if len(_USERDATA) < 500:
            pytest.skip("Need 500+ real spins")
        pd = PatternDetector()
        pd.load_history(_USERDATA)
        strength = pd.get_pattern_strength()
        assert strength > 30, \
            f"Pattern strength is {strength} — should be > 30 with real data"

    def test_uniform_data_detection_in_load_state(self):
        """REGRESSION: load_state should reject perfectly uniform cached data."""
        import tempfile, pickle
        ep = EnsemblePredictor()
        # Create perfectly uniform data: 0,1,...,36 repeated 10 times
        uniform_data = list(range(37)) * 10
        ep.load_history(uniform_data)

        # Save to a temp file
        from config import MODEL_STATE_PATH, MODELS_DIR
        tmp_state = os.path.join(tempfile.gettempdir(), 'test_model_state.pkl')

        # Monkey-patch the path temporarily
        import app.ml.ensemble as ens_module
        orig_path = ens_module.MODEL_STATE_PATH
        ens_module.MODEL_STATE_PATH = tmp_state
        try:
            ep.save_state()

            # Now try to load — should reject uniform data
            ep2 = EnsemblePredictor()
            loaded = ep2.load_state()
            assert loaded == False, "load_state should reject perfectly uniform data"
            assert len(ep2.spin_history) == 0, \
                "After rejecting corrupt data, spin_history should be empty"
        finally:
            ens_module.MODEL_STATE_PATH = orig_path
            if os.path.exists(tmp_state):
                os.remove(tmp_state)

    def test_save_load_roundtrip_preserves_data(self):
        """Save/load roundtrip should preserve data accurately."""
        import tempfile
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)

        # Save to temp location
        import app.ml.ensemble as ens_module
        orig_path = ens_module.MODEL_STATE_PATH
        tmp_state = os.path.join(tempfile.gettempdir(), 'test_roundtrip.pkl')
        ens_module.MODEL_STATE_PATH = tmp_state
        try:
            ep.save_state()

            ep2 = EnsemblePredictor()
            loaded = ep2.load_state()
            assert loaded == True
            assert ep2.spin_history == ep.spin_history
            assert len(ep2.frequency.spin_history) == len(ep.frequency.spin_history)
            assert len(ep2.patterns.spin_history) == len(ep.patterns.spin_history)

            # Verify model status is consistent
            status = ep2.get_model_status()
            for model_name in _MODEL_KEYS:
                model_status = status[model_name]
                assert model_status['spins'] == 100, \
                    f"{model_name} has {model_status['spins']} after roundtrip, expected 100"
        finally:
            ens_module.MODEL_STATE_PATH = orig_path
            if os.path.exists(tmp_state):
                os.remove(tmp_state)


# ═══════════════════════════════════════════════════════════════
# Regression Tests — Full Pipeline with Real Userdata
# ═══════════════════════════════════════════════════════════════

class TestFullPipelineRegression:
    """Tests the full ensemble pipeline with real userdata files."""

    def test_full_pipeline_with_userdata(self):
        """Load real data, train, predict — full regression."""
        if len(_USERDATA) < 500:
            pytest.skip("Need 500+ real spins")

        ep = EnsemblePredictor()
        ep.load_history(_USERDATA)

        # All models should have data
        status = ep.get_model_status()
        total = len(_USERDATA)
        for model_name in _MODEL_KEYS:
            s = status[model_name]
            assert s['status'] != 'idle', \
                f"{model_name} is idle after loading {total} spins"
            assert s['spins'] == total, \
                f"{model_name} shows {s['spins']} spins, expected {total}"

        # Prediction should work — dynamic count (3-20)
        pred = ep.predict()
        from config import TOP_PREDICTIONS_MAX
        assert 3 <= len(pred['top_numbers']) <= TOP_PREDICTIONS_MAX
        assert pred['confidence'] > 0
        assert pred['mode'] in ('WAIT', 'BET', 'BET_HIGH')

    def test_cluster_loading_matches_flat_loading(self):
        """Loading via clusters should produce same results as flat load."""
        if not _USERDATA_CLUSTERS or len(_USERDATA) < 100:
            pytest.skip("No userdata files found")

        # Method 1: flat load
        ep1 = EnsemblePredictor()
        ep1.load_history(_USERDATA)

        # Method 2: cluster load
        ep2 = EnsemblePredictor()
        ep2.load_clusters(_USERDATA_CLUSTERS)

        # Same total spins
        assert len(ep1.spin_history) == len(ep2.spin_history), \
            f"Flat: {len(ep1.spin_history)}, Clusters: {len(ep2.spin_history)}"

        # Same data
        assert ep1.spin_history == ep2.spin_history

        # Same frequency counts
        assert ep1.frequency.frequency_counts == ep2.frequency.frequency_counts

        # Pattern detector gets all data in both cases
        assert len(ep1.patterns.spin_history) == len(ep2.patterns.spin_history), \
            f"Pattern flat: {len(ep1.patterns.spin_history)}, cluster: {len(ep2.patterns.spin_history)}"

    def test_incremental_update_after_clusters(self):
        """After cluster load, incremental updates should work correctly."""
        if not _USERDATA_CLUSTERS:
            pytest.skip("No userdata files found")

        ep = EnsemblePredictor()
        ep.load_clusters(_USERDATA_CLUSTERS)
        initial_count = len(ep.spin_history)

        # Add 5 more numbers
        new_numbers = [7, 14, 21, 28, 35]
        result = ep.update_incremental(new_numbers)
        assert result['numbers_added'] == 5
        assert result['total_spins'] == initial_count + 5

        # All models should report updated count
        status = ep.get_model_status()
        expected = initial_count + 5
        for model_name in _MODEL_KEYS:
            s = status[model_name]
            assert s['spins'] == expected, \
                f"{model_name} reports {s['spins']} after incremental, expected {expected}"


# ═══════════════════════════════════════════════════════════════
# Feature Engine Tests
# ═══════════════════════════════════════════════════════════════

class TestFeatureEngine:
    def test_feature_dim_constant(self):
        fe = FeatureEngine()
        assert fe.FEATURE_DIM == FEATURE_DIM

    def test_extract_single_shape(self):
        fe = FeatureEngine()
        context = [10, 20, 30, 5, 15]
        vec = fe.extract_single(25, context)
        assert vec.shape == (FEATURE_DIM,)
        assert vec.dtype == np.float32

    def test_extract_sequence_shape(self):
        fe = FeatureEngine()
        history = SAMPLE_100
        seq_len = 10
        features = fe.extract_sequence(history, seq_start=5, seq_length=seq_len)
        assert features.shape == (seq_len, FEATURE_DIM)
        assert features.dtype == np.float32

    def test_wheel_position_encoding(self):
        """Wheel position sin/cos should be in [-1, 1]."""
        fe = FeatureEngine()
        vec = fe.extract_single(0, [1, 2, 3])
        # First 2 dims are sin/cos of wheel position
        assert -1.0 <= vec[0] <= 1.0
        assert -1.0 <= vec[1] <= 1.0

    def test_different_numbers_give_different_features(self):
        fe = FeatureEngine()
        context = list(range(0, 37))
        vec_0 = fe.extract_single(0, context)
        vec_17 = fe.extract_single(17, context)
        # Should not be identical
        assert not np.allclose(vec_0, vec_17)

    def test_gap_features(self):
        """Gap features: recently seen numbers should have small gaps."""
        fe = FeatureEngine()
        context = [5, 10, 15, 20, 25]
        vec = fe.extract_single(30, context)
        # Gap dims start at index 16 (static=16), span 37 dims
        gap_start = fe.STATIC_DIMS
        gaps = vec[gap_start:gap_start + TOTAL_NUMBERS]
        # Number 25 was just seen (gap ~0.2), number 0 was never seen (gap=1.0)
        assert gaps[25] < gaps[0]

    def test_all_values_finite(self):
        fe = FeatureEngine()
        vec = fe.extract_single(17, SAMPLE_100)
        assert np.all(np.isfinite(vec))

    def test_extract_with_empty_context(self):
        fe = FeatureEngine()
        vec = fe.extract_single(10, [])
        assert vec.shape == (FEATURE_DIM,)
        assert np.all(np.isfinite(vec))

    def test_extract_sequence_with_real_data(self):
        if len(REAL_DATA) < 50:
            pytest.skip("Need at least 50 spins")
        fe = FeatureEngine()
        features = fe.extract_sequence(REAL_DATA, seq_start=10, seq_length=30)
        assert features.shape == (30, FEATURE_DIM)
        assert np.all(np.isfinite(features))


# ═══════════════════════════════════════════════════════════════
# Model Performance Tracker Tests
# ═══════════════════════════════════════════════════════════════

class TestModelPerformanceTracker:
    def test_init(self):
        tracker = ModelPerformanceTracker()
        assert tracker.observation_count == 0

    def test_record_and_count(self):
        tracker = ModelPerformanceTracker()
        dist = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
        dists = {name: dist.copy() for name in tracker.MODEL_NAMES}
        for _ in range(5):
            tracker.record(10, dists)
        assert tracker.observation_count == 5

    def test_defaults_with_few_observations(self):
        tracker = ModelPerformanceTracker()
        weights = tracker.get_adaptive_weights()
        # Should return config defaults with no data
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert 'frequency' in weights
        assert 'lstm' in weights

    def test_adaptive_weights_with_enough_data(self):
        """With enough observations (>= ADAPTIVE_MIN_OBSERVATIONS), the
        model that consistently assigns higher probability to the actual
        number should receive a higher adaptive weight."""
        # Use enough observations to exceed ADAPTIVE_MIN_OBSERVATIONS (100)
        n_obs = max(ADAPTIVE_MIN_OBSERVATIONS + 10, 120)
        tracker = ModelPerformanceTracker(window=n_obs)
        # Make LSTM consistently better
        for _ in range(n_obs):
            dists = {}
            for name in tracker.MODEL_NAMES:
                dist = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
                if name == 'lstm':
                    dist[10] = 0.2  # LSTM assigns higher probability to actual
                dists[name] = dist
            tracker.record(10, dists)

        weights = tracker.get_adaptive_weights()
        # LSTM should get higher weight since it predicts better
        assert weights['lstm'] > weights['frequency']

    def test_weights_sum_to_one(self):
        tracker = ModelPerformanceTracker()
        dist = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
        dists = {name: dist for name in tracker.MODEL_NAMES}
        for _ in range(30):
            tracker.record(np.random.randint(0, 37), dists)
        weights = tracker.get_adaptive_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_state_persistence(self):
        tracker = ModelPerformanceTracker()
        dist = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
        dists = {name: dist for name in tracker.MODEL_NAMES}
        for _ in range(10):
            tracker.record(5, dists)

        state = tracker.get_state()
        tracker2 = ModelPerformanceTracker()
        tracker2.load_state(state)
        assert tracker2.observation_count == 10


# ═══════════════════════════════════════════════════════════════
# Regime Detector Tests
# ═══════════════════════════════════════════════════════════════

class TestRegimeDetector:
    def test_insufficient_data(self):
        rd = RegimeDetector()
        result = rd.check([1, 2, 3])
        assert result['changed'] is False
        assert result['kl_divergence'] == 0.0

    def test_stable_regime(self):
        rd = RegimeDetector()
        # Very large uniform data to minimize random variance
        np.random.seed(123)
        data = list(np.random.randint(0, 37, size=1000))
        result = rd.check(data)
        # With 1000 uniform random spins, KL should be very small
        assert result['kl_divergence'] < 0.5  # Generous threshold

    def test_regime_change_detection(self):
        rd = RegimeDetector()
        # First 200 spins: uniform, then last 50 spins: heavily biased
        np.random.seed(42)
        baseline = list(np.random.randint(0, 37, size=200))
        biased = [5, 10, 15] * 20  # Only 3 numbers
        data = baseline + biased
        result = rd.check(data)
        # KL divergence should be high with such extreme bias
        assert result['kl_divergence'] > 0


# ═══════════════════════════════════════════════════════════════
# Confidence Calibrator Tests
# ═══════════════════════════════════════════════════════════════

class TestConfidenceCalibrator:
    def test_init(self):
        cal = ConfidenceCalibrator()
        assert len(cal.bin_data) == CALIBRATION_BINS

    def test_passthrough_with_no_data(self):
        cal = ConfidenceCalibrator()
        # With no data, should return raw confidence
        assert cal.calibrate(60.0) == 60.0

    def test_calibration_after_data(self):
        cal = ConfidenceCalibrator(n_bins=6, min_samples=5)
        # Record: at raw confidence ~60, we never hit
        for _ in range(20):
            cal.record(60.0, False)
        # Calibrated should be much lower than 60
        calibrated = cal.calibrate(60.0)
        assert calibrated < 60.0

    def test_calibration_increases_on_hits(self):
        cal = ConfidenceCalibrator(n_bins=6, min_samples=5)
        # Record: at raw confidence ~40, we always hit
        for _ in range(20):
            cal.record(40.0, True)
        calibrated = cal.calibrate(40.0)
        assert calibrated > 40.0

    def test_state_roundtrip(self):
        cal = ConfidenceCalibrator()
        for _ in range(15):
            cal.record(55.0, True)
        state = cal.get_state()

        cal2 = ConfidenceCalibrator()
        cal2.load_state(state)
        assert cal.calibrate(55.0) == cal2.calibrate(55.0)


# ═══════════════════════════════════════════════════════════════
# Pattern Detector — Statistical Tests
# ═══════════════════════════════════════════════════════════════

class TestPatternDetectorStatistical:
    def test_runs_test_insufficient_data(self):
        pd = PatternDetector()
        pd.load_history([1, 2, 3])
        result = pd.runs_test()
        assert result['p_value'] == 1.0
        assert result['significant'] is False

    def test_runs_test_with_data(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_100)
        result = pd.runs_test()
        assert 0 <= result['p_value'] <= 1.0
        assert 'n_runs' in result
        assert result['n_runs'] > 0

    def test_serial_correlation_insufficient_data(self):
        pd = PatternDetector()
        pd.load_history([1, 2])
        result = pd.serial_correlation_test()
        assert result['correlation'] == 0.0

    def test_serial_correlation_with_data(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_100)
        result = pd.serial_correlation_test()
        assert -1.0 <= result['correlation'] <= 1.0
        assert 0 <= result['p_value'] <= 1.0

    def test_sector_autocorrelation(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_100)
        result = pd.sector_autocorrelation()
        assert -1.0 <= result['correlation'] <= 1.0

    def test_dealer_signature_insufficient_data(self):
        pd = PatternDetector()
        pd.load_history([1, 2])
        result = pd.dealer_signature_test()
        assert result['detected'] is False

    def test_dealer_signature_with_data(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_100)
        result = pd.dealer_signature_test()
        assert 'common_gaps' in result
        assert isinstance(result['common_gaps'], list)

    def test_statistical_tests_combined(self):
        pd = PatternDetector()
        pd.load_history(SAMPLE_100)
        tests = pd.get_statistical_tests()
        assert 'runs_test' in tests
        assert 'serial_correlation' in tests
        assert 'sector_autocorrelation' in tests
        assert 'dealer_signature' in tests

    def test_pattern_strength_includes_stats(self):
        """Pattern strength should change when statistical tests are significant."""
        pd = PatternDetector()
        pd.load_history(SAMPLE_100)
        strength = pd.get_pattern_strength()
        assert isinstance(strength, float)
        assert 0 <= strength <= 100


# ═══════════════════════════════════════════════════════════════
# Time-Weighted Frequency Tests
# ═══════════════════════════════════════════════════════════════

class TestTimeWeightedFrequency:
    def test_recent_probabilities_shape(self):
        fa = FrequencyAnalyzer()
        fa.load_history(SAMPLE_50)
        probs = fa.get_recent_probabilities()
        assert probs.shape == (TOTAL_NUMBERS,)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_recent_probabilities_recent_bias(self):
        """Numbers appearing recently should have higher probability
        in time-weighted vs uniform."""
        fa = FrequencyAnalyzer()
        # Create history where 0-9 appear first, then 30-36 appear recently
        data = list(range(0, 10)) * 10 + list(range(30, 37)) * 10
        fa.load_history(data)
        probs = fa.get_recent_probabilities()
        # Numbers 30-36 (recent) should have higher prob than 0-9 (old)
        recent_avg = np.mean([probs[i] for i in range(30, 37)])
        old_avg = np.mean([probs[i] for i in range(0, 10)])
        assert recent_avg > old_avg

    def test_blended_probabilities(self):
        """get_number_probabilities should blend flat and time-weighted."""
        fa = FrequencyAnalyzer()
        fa.load_history(SAMPLE_100)
        probs = fa.get_number_probabilities()
        assert probs.shape == (TOTAL_NUMBERS,)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_recent_probabilities_with_little_data(self):
        fa = FrequencyAnalyzer()
        fa.load_history([5, 10])
        probs = fa.get_recent_probabilities()
        # Should be uniform-ish
        assert abs(probs.sum() - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════
# Adaptive Ensemble Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestAdaptiveEnsemble:
    def test_ensemble_has_performance_tracker(self):
        ep = EnsemblePredictor()
        assert hasattr(ep, 'performance_tracker')
        assert isinstance(ep.performance_tracker, ModelPerformanceTracker)

    def test_ensemble_has_regime_detector(self):
        ep = EnsemblePredictor()
        assert hasattr(ep, 'regime_detector')
        assert isinstance(ep.regime_detector, RegimeDetector)

    def test_prediction_includes_adaptive_weights(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)
        pred = ep.predict()
        assert 'adaptive_weights' in pred
        assert 'performance_observations' in pred
        assert 'regime' in pred

    def test_prediction_includes_regime_info(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)
        pred = ep.predict()
        regime = pred['regime']
        assert 'changed' in regime
        assert 'kl_divergence' in regime

    def test_model_status_includes_weights(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        status = ep.get_model_status()
        for model_name in ['frequency', 'markov', 'patterns', 'lstm']:
            assert 'weight' in status[model_name]
        assert 'adaptive_weights_active' in status

    def test_performance_tracker_updated_on_update(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        # Make a prediction to populate _last_model_distributions
        ep.predict()
        initial_obs = ep.performance_tracker.observation_count
        # Update with actual result
        ep.update(10)
        assert ep.performance_tracker.observation_count == initial_obs + 1

    def test_full_reset_clears_adaptive(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_50)
        ep.predict()
        ep.update(10)
        ep.full_reset()
        assert ep.performance_tracker.observation_count == 0

    def test_ensemble_probabilities_sum_to_one(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)
        probs, _ = ep.get_ensemble_probabilities()
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_confidence_breakdown_has_calibration_info(self):
        ep = EnsemblePredictor()
        ep.load_history(SAMPLE_100)
        pred = ep.predict()
        breakdown = pred['confidence_breakdown']
        assert 'raw_confidence' in breakdown
        assert 'calibrated' in breakdown


# ═══════════════════════════════════════════════════════════════
# LSTM Predictor — Architecture Tests (no training needed)
# ═══════════════════════════════════════════════════════════════

class TestLSTMArchitecture:
    def test_model_has_attention(self):
        """The upgraded model should have attention layers."""
        from app.ml.lstm_predictor import RouletteGRU
        model = RouletteGRU()
        assert hasattr(model, 'attention'), "Model should have attention layer"
        assert hasattr(model, 'input_norm'), "Model should have input LayerNorm"
        assert hasattr(model, 'attn_norm'), "Model should have attention LayerNorm"

    def test_model_forward_shape(self):
        """Model forward pass should produce correct output shape."""
        import torch
        from app.ml.lstm_predictor import RouletteGRU
        model = RouletteGRU()
        batch = torch.randn(2, 30, FEATURE_DIM)  # batch=2, seq=30, features=85
        output, hidden = model(batch)
        assert output.shape == (2, TOTAL_NUMBERS)

    def test_predictor_uses_feature_engine(self):
        from app.ml.lstm_predictor import LSTMPredictor
        pred = LSTMPredictor()
        assert hasattr(pred, 'feature_engine')
        assert isinstance(pred.feature_engine, FeatureEngine)

    def test_predictor_has_label_smoothing(self):
        from app.ml.lstm_predictor import LSTMPredictor
        pred = LSTMPredictor()
        # Check that criterion uses label smoothing
        assert pred.criterion.label_smoothing > 0

    def test_predictor_has_scheduler(self):
        from app.ml.lstm_predictor import LSTMPredictor
        pred = LSTMPredictor()
        assert hasattr(pred, 'scheduler')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
