"""
Ensemble Predictor - Master orchestrator combining all ML models.
Generates weighted predictions, bet suggestions, and WAIT/BET decisions.

Industrial-level features:
  - Adaptive model weights (learn which sub-model to trust)
  - Regime detection (detect when data-generating process changes)
  - Performance-tracked online learning
"""

import math
import numpy as np
import pickle
import os
from collections import defaultdict, deque, Counter

try:
    import eventlet
    _sleep = eventlet.sleep
except ImportError:
    _sleep = lambda x: None

import sys
sys.path.insert(0, '.')
from config import (
    TOTAL_NUMBERS, RED_NUMBERS, BLACK_NUMBERS,
    FIRST_DOZEN, SECOND_DOZEN, THIRD_DOZEN,
    FIRST_COLUMN, SECOND_COLUMN, THIRD_COLUMN,
    LOW_NUMBERS, HIGH_NUMBERS, ODD_NUMBERS, EVEN_NUMBERS,
    ENSEMBLE_FREQUENCY_WEIGHT, ENSEMBLE_MARKOV_WEIGHT,
    ENSEMBLE_PATTERN_WEIGHT, ENSEMBLE_LSTM_WEIGHT,
    CONFIDENCE_STRAIGHT_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_BET_THRESHOLD, RETRAIN_INTERVAL,
    TOP_PREDICTIONS_COUNT, TOP_PREDICTIONS_MAX, NEIGHBOURS_PER_ANCHOR,
    PREDICTION_CONFIDENCE_FACTOR,
    WHEEL_ORDER, NUMBER_TO_POSITION,
    MODEL_STATE_PATH, MODELS_DIR, SESSIONS_DIR, LSTM_MODEL_PATH,
    ADAPTIVE_WINDOW, ADAPTIVE_TEMPERATURE, ADAPTIVE_MIN_WEIGHT,
    ADAPTIVE_MIN_OBSERVATIONS,
    REGIME_RECENT_WINDOW, REGIME_BASELINE_WINDOW, REGIME_KL_THRESHOLD,
    CONSECUTIVE_MISS_THRESHOLD, EXPLORATION_BOOST_FACTOR, SECTOR_COOLDOWN_WINDOW,
    get_number_color
)

from app.ml.frequency_analyzer import FrequencyAnalyzer
from app.ml.markov_chain import MarkovChain
from app.ml.pattern_detector import PatternDetector
from app.ml.lstm_predictor import LSTMPredictor
from app.ml.confidence import ConfidenceEngine


# ─── Model Performance Tracker ────────────────────────────────────────

class ModelPerformanceTracker:
    """Tracks per-model prediction accuracy over a sliding window.
    After each spin, records the probability each model assigned to the
    actual outcome.  Models that consistently assign higher probability
    to actual outcomes get higher adaptive weights.
    """

    MODEL_NAMES = ('frequency', 'markov', 'patterns', 'lstm')

    def __init__(self, window=ADAPTIVE_WINDOW):
        self.model_scores = {
            name: deque(maxlen=window) for name in self.MODEL_NAMES
        }

    def record(self, actual_number, model_distributions):
        """Record how well each model predicted the actual number.

        Args:
            actual_number: int, the spin that actually occurred
            model_distributions: dict name → np.array of shape (37,)
        """
        for name in self.MODEL_NAMES:
            dist = model_distributions.get(name)
            if dist is not None and len(dist) == TOTAL_NUMBERS:
                score = float(dist[actual_number])
                self.model_scores[name].append(score)

    def get_adaptive_weights(self):
        """Softmax over mean scores → weights that sum to 1.

        Falls back to config defaults if insufficient data.
        Returns: dict name → float weight
        """
        defaults = {
            'frequency': ENSEMBLE_FREQUENCY_WEIGHT,
            'markov': ENSEMBLE_MARKOV_WEIGHT,
            'patterns': ENSEMBLE_PATTERN_WEIGHT,
            'lstm': ENSEMBLE_LSTM_WEIGHT,
        }

        # Need minimum observations before switching to adaptive
        min_obs = min(len(self.model_scores[n]) for n in self.MODEL_NAMES)
        if min_obs < ADAPTIVE_MIN_OBSERVATIONS:
            return defaults

        # Compute mean score per model
        means = {}
        for name in self.MODEL_NAMES:
            scores = list(self.model_scores[name])
            means[name] = sum(scores) / len(scores) if scores else 0.0

        # Softmax with temperature
        max_mean = max(means.values())
        exp_scores = {}
        for name in self.MODEL_NAMES:
            exp_scores[name] = math.exp(
                (means[name] - max_mean) / ADAPTIVE_TEMPERATURE
            )
        total_exp = sum(exp_scores.values())

        weights = {}
        for name in self.MODEL_NAMES:
            raw = exp_scores[name] / total_exp if total_exp > 0 else 0.25
            weights[name] = max(raw, ADAPTIVE_MIN_WEIGHT)

        # Renormalize to sum to 1
        total_w = sum(weights.values())
        for name in weights:
            weights[name] /= total_w

        return weights

    @property
    def observation_count(self):
        return min(len(self.model_scores[n]) for n in self.MODEL_NAMES)

    def get_state(self):
        """Serializable state for persistence."""
        return {name: list(scores)
                for name, scores in self.model_scores.items()}

    def load_state(self, state):
        """Restore from serialized state."""
        for name in self.MODEL_NAMES:
            if name in state:
                self.model_scores[name] = deque(
                    state[name], maxlen=ADAPTIVE_WINDOW
                )


# ─── Regime Detector ──────────────────────────────────────────────────

class RegimeDetector:
    """Detects when the data-generating process changes (dealer change,
    wheel maintenance) using KL divergence between recent and baseline
    distributions.
    """

    def __init__(self):
        self.regime_changed = False
        self.last_kl = 0.0

    def check(self, spin_history):
        """Check if regime has changed.

        Args:
            spin_history: full list of spin results

        Returns:
            dict with 'changed', 'kl_divergence', 'message'
        """
        n = len(spin_history)
        if n < REGIME_RECENT_WINDOW + 20:
            self.regime_changed = False
            self.last_kl = 0.0
            return {
                'changed': False,
                'kl_divergence': 0.0,
                'message': 'Insufficient data for regime detection'
            }

        recent = spin_history[-REGIME_RECENT_WINDOW:]
        baseline_end = max(0, n - REGIME_RECENT_WINDOW)
        baseline_start = max(0, baseline_end - REGIME_BASELINE_WINDOW)
        baseline = spin_history[baseline_start:baseline_end]

        if len(baseline) < 20:
            self.regime_changed = False
            self.last_kl = 0.0
            return {
                'changed': False,
                'kl_divergence': 0.0,
                'message': 'Insufficient baseline data'
            }

        # Compute distributions with Laplace smoothing
        recent_counts = Counter(recent)
        baseline_counts = Counter(baseline)

        recent_dist = np.array([
            (recent_counts.get(i, 0) + 1) / (len(recent) + TOTAL_NUMBERS)
            for i in range(TOTAL_NUMBERS)
        ])
        baseline_dist = np.array([
            (baseline_counts.get(i, 0) + 1) / (len(baseline) + TOTAL_NUMBERS)
            for i in range(TOTAL_NUMBERS)
        ])

        # KL divergence: D_KL(recent || baseline)
        kl = float(np.sum(recent_dist * np.log(recent_dist / baseline_dist)))
        self.last_kl = kl
        self.regime_changed = kl > REGIME_KL_THRESHOLD

        if self.regime_changed:
            msg = f'Regime change detected (KL={kl:.3f} > {REGIME_KL_THRESHOLD})'
        else:
            msg = f'Stable regime (KL={kl:.3f})'

        return {
            'changed': self.regime_changed,
            'kl_divergence': round(kl, 4),
            'message': msg
        }


class EnsemblePredictor:
    def __init__(self):
        self.frequency = FrequencyAnalyzer()
        self.markov = MarkovChain()
        self.patterns = PatternDetector()
        self.lstm = LSTMPredictor()
        self.confidence_engine = ConfidenceEngine()

        # Adaptive components
        self.performance_tracker = ModelPerformanceTracker()
        self.regime_detector = RegimeDetector()

        self.spin_history = []
        self.prediction_count = 0
        self.last_prediction = None
        self._last_model_distributions = None  # Cache for performance tracking

        # Loss-reactive strategy state
        self.consecutive_misses = 0
        self.sector_miss_counts = {}

    def full_reset(self):
        """Completely reset all models to fresh state — no training data, no history."""
        import os
        self.frequency = FrequencyAnalyzer()
        self.markov = MarkovChain()
        self.patterns = PatternDetector()
        self.lstm = LSTMPredictor()
        self.confidence_engine = ConfidenceEngine()
        self.performance_tracker = ModelPerformanceTracker()
        self.regime_detector = RegimeDetector()
        self.spin_history = []
        self.prediction_count = 0
        self.last_prediction = None
        self._last_model_distributions = None
        self.consecutive_misses = 0
        self.sector_miss_counts = {}

        # Reset LSTM weights to random (undo any loaded checkpoint)
        for module in self.lstm.model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.lstm.is_trained = False
        self.lstm.training_loss_history = []
        self.lstm.validation_loss_history = []
        self.lstm.spin_history = []

        # Delete saved model files so they don't reload on next init
        if os.path.exists(LSTM_MODEL_PATH):
            os.remove(LSTM_MODEL_PATH)
        if os.path.exists(MODEL_STATE_PATH):
            os.remove(MODEL_STATE_PATH)

        print("[RESET] All models cleared — fresh state")

    def load_history(self, history):
        """Load historical data into all models."""
        self.spin_history = list(history)
        self.frequency.load_history(history)
        self.markov.load_history(history)
        self.patterns.load_history(history)
        self.lstm.load_history(history)

    def update(self, number):
        """Feed a new spin result to all models.
        NOTE: LSTM auto-retrain is handled by the socketio handler as a
        non-blocking background task (every RETRAIN_INTERVAL spins).
        """
        self.spin_history.append(number)
        self.frequency.update(number)
        self.markov.update(number)
        self.patterns.update(number)
        self.lstm.update(number)

        # Record per-model prediction accuracy for adaptive weighting
        if self._last_model_distributions is not None:
            self.performance_tracker.record(number, self._last_model_distributions)

        # Record prediction accuracy if we had a prediction
        if self.last_prediction:
            self._record_accuracy(number)

            # Loss-reactive strategy: track consecutive misses
            predicted_numbers = self.last_prediction.get('top_numbers', [])
            hit = number in predicted_numbers

            if hit:
                self.consecutive_misses = 0
                # Decay sector miss counts on hit
                for sector in list(self.sector_miss_counts):
                    self.sector_miss_counts[sector] = max(0, self.sector_miss_counts[sector] - 1)
                    if self.sector_miss_counts[sector] == 0:
                        del self.sector_miss_counts[sector]
            else:
                self.consecutive_misses += 1
                # Track which sectors we predicted but missed
                for pred_num in predicted_numbers:
                    pred_pos = NUMBER_TO_POSITION.get(pred_num, 0)
                    pred_sector = pred_pos // 5
                    self.sector_miss_counts[pred_sector] = self.sector_miss_counts.get(pred_sector, 0) + 1

    def _record_accuracy(self, actual):
        """Record how accurate the last prediction was."""
        pred = self.last_prediction
        pred_record = {
            'numbers': pred.get('top_numbers', []),
            'color': pred.get('suggested_color'),
            'dozen': pred.get('suggested_dozen'),
            'high_low': pred.get('suggested_high_low'),
            'odd_even': pred.get('suggested_odd_even'),
            'actual_color': get_number_color(actual),
            'actual_dozen': self._get_dozen(actual),
        }
        self.confidence_engine.record_prediction(pred_record, actual)

    def _get_dozen(self, n):
        if n in FIRST_DOZEN: return '1st'
        if n in SECOND_DOZEN: return '2nd'
        if n in THIRD_DOZEN: return '3rd'
        return 'zero'

    def get_ensemble_probabilities(self):
        """Weighted combination of all model probability distributions.

        Uses adaptive weights (learned from per-model accuracy) when enough
        observations are available, otherwise falls back to config defaults.
        """
        freq_probs = self.frequency.get_number_probabilities()
        markov_probs = self.markov.get_probabilities()
        pattern_probs = self.patterns.get_number_probabilities()
        lstm_probs = self.lstm.predict()

        # Cache individual distributions for performance tracking on next update
        self._last_model_distributions = {
            'frequency': freq_probs,
            'markov': markov_probs,
            'patterns': pattern_probs,
            'lstm': lstm_probs,
        }

        # Get weights — adaptive if enough data, else config defaults
        weights = self.performance_tracker.get_adaptive_weights()
        freq_w = weights['frequency']
        markov_w = weights['markov']
        pattern_w = weights['patterns']
        lstm_weight = weights['lstm']

        # If LSTM isn't trained, redistribute its weight among others
        if not self.lstm.is_trained:
            redistrib = lstm_weight
            lstm_weight = 0
            total_other = freq_w + markov_w + pattern_w
            if total_other > 0:
                freq_w += redistrib * (freq_w / total_other)
                markov_w += redistrib * (markov_w / total_other)
                pattern_w += redistrib * (pattern_w / total_other)

        ensemble = (
            freq_probs * freq_w +
            markov_probs * markov_w +
            pattern_probs * pattern_w +
            lstm_probs * lstm_weight
        )

        # Normalize
        total = ensemble.sum()
        if total > 0:
            ensemble /= total

        # Loss-reactive strategy: diversify after consecutive misses
        if self.consecutive_misses >= CONSECUTIVE_MISS_THRESHOLD:
            uniform = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

            # Suppress sectors we keep predicting but missing, boost unexplored
            cold_boost = np.ones(TOTAL_NUMBERS)
            if self.sector_miss_counts:
                max_misses = max(self.sector_miss_counts.values())
                if max_misses > 0:
                    for num in range(TOTAL_NUMBERS):
                        pos = NUMBER_TO_POSITION.get(num, 0)
                        sector = pos // 5
                        misses = self.sector_miss_counts.get(sector, 0)
                        cold_boost[num] = 1.0 - (misses / max_misses) * 0.5
                cold_boost /= cold_boost.sum() / TOTAL_NUMBERS  # Normalize to mean 1.0

            # Exploration scales with consecutive misses (30% at threshold, +10%/miss, max 60%)
            extra_misses = self.consecutive_misses - CONSECUTIVE_MISS_THRESHOLD
            exploration = min(0.6, EXPLORATION_BOOST_FACTOR + extra_misses * 0.1)

            ensemble = (1 - exploration) * ensemble * cold_boost + exploration * uniform
            ensemble /= ensemble.sum()

        return ensemble, [freq_probs, markov_probs, pattern_probs, lstm_probs]

    def _get_group_probability(self, probs, number_set):
        """Sum probabilities for a group of numbers."""
        return sum(probs[n] for n in number_set if n < len(probs))

    def _smart_anchor_selection(self, ensemble_probs, all_sorted, wheel_len):
        """Dynamic AI-driven prediction selection.

        The AI decides how many numbers to predict based on probability strength:
          1. Select all numbers above the confidence threshold (≥ 1.2× uniform)
          2. For each selected number, include its ±2 wheel neighbours
          3. Group into natural wheel clusters for display
          4. No fixed anchor count — could be 2 anchors or 6, AI decides

        Returns: (anchors, top_numbers, top_probs, anchor_details)
          anchor_details = [{number, spread, numbers_covered}, ...]
        """
        neighbours_per_side = NEIGHBOURS_PER_ANCHOR  # 2
        max_budget = TOP_PREDICTIONS_MAX  # 20 safety cap
        uniform = 1.0 / TOTAL_NUMBERS
        threshold = uniform * PREDICTION_CONFIDENCE_FACTOR

        # Step 1: Find all numbers the AI is confident about
        confident_nums = []
        for idx in all_sorted:
            num = int(idx)
            if ensemble_probs[num] >= threshold:
                confident_nums.append(num)
            else:
                break  # sorted descending, so stop once below threshold

        # Ensure at least 3 numbers, cap at max_budget
        if len(confident_nums) < 3:
            confident_nums = [int(all_sorted[i]) for i in range(min(3, len(all_sorted)))]
        if len(confident_nums) > max_budget:
            confident_nums = confident_nums[:max_budget]

        # Step 2: Build anchor groups from confident numbers
        # Each confident number becomes a potential anchor — merge if they're
        # within ±(neighbours_per_side) on the wheel
        selected = set()
        anchors = []
        anchor_details = []
        top_numbers = []
        top_probs = []

        # Sort confident nums by probability (highest first)
        confident_nums.sort(key=lambda n: ensemble_probs[n], reverse=True)

        for num in confident_nums:
            if num in selected:
                continue
            if len(top_numbers) >= max_budget:
                break

            pos = NUMBER_TO_POSITION.get(num, -1)
            if pos < 0:
                continue

            # Check if this number is already covered by an existing anchor's neighbours
            already_covered = False
            for ad in anchor_details:
                if num in ad['numbers']:
                    already_covered = True
                    break
            if already_covered:
                continue

            # This is a new anchor
            group = [num]
            selected.add(num)

            # Add ±neighbours_per_side wheel neighbours
            for direction in [1, -1]:
                for reach in range(1, neighbours_per_side + 1):
                    if len(top_numbers) + len(group) >= max_budget:
                        break
                    nb_pos = (pos + direction * reach) % wheel_len
                    nb_num = WHEEL_ORDER[nb_pos]
                    if nb_num not in selected:
                        group.append(nb_num)
                        selected.add(nb_num)

            anchors.append(num)
            for n in group:
                top_numbers.append(n)
                top_probs.append(round(float(ensemble_probs[n]), 4))

            anchor_details.append({
                'number': num,
                'spread': neighbours_per_side,
                'numbers': list(group)
            })

        return anchors, top_numbers, top_probs, anchor_details

    def predict(self):
        """Generate full prediction with bet suggestions."""
        self.prediction_count += 1

        ensemble_probs, all_distributions = self.get_ensemble_probabilities()

        # Smart Anchor Selection: maximise wheel coverage with minimal overlap.
        # Instead of naively picking top 4, detect clusters of adjacent high-prob
        # numbers on the wheel and place anchors to cover them efficiently.
        all_sorted = np.argsort(ensemble_probs)[::-1]  # descending probability
        wheel_len = len(WHEEL_ORDER)

        anchors, top_numbers, top_probs, anchor_details = self._smart_anchor_selection(
            ensemble_probs, all_sorted, wheel_len
        )

        # Group probabilities
        red_prob = self._get_group_probability(ensemble_probs, RED_NUMBERS)
        black_prob = self._get_group_probability(ensemble_probs, BLACK_NUMBERS)
        d1_prob = self._get_group_probability(ensemble_probs, FIRST_DOZEN)
        d2_prob = self._get_group_probability(ensemble_probs, SECOND_DOZEN)
        d3_prob = self._get_group_probability(ensemble_probs, THIRD_DOZEN)
        c1_prob = self._get_group_probability(ensemble_probs, FIRST_COLUMN)
        c2_prob = self._get_group_probability(ensemble_probs, SECOND_COLUMN)
        c3_prob = self._get_group_probability(ensemble_probs, THIRD_COLUMN)
        high_prob = self._get_group_probability(ensemble_probs, HIGH_NUMBERS)
        low_prob = self._get_group_probability(ensemble_probs, LOW_NUMBERS)
        odd_prob = self._get_group_probability(ensemble_probs, ODD_NUMBERS)
        even_prob = self._get_group_probability(ensemble_probs, EVEN_NUMBERS)

        # Pattern strength from pattern detector
        pattern_strength = self.patterns.get_pattern_strength()

        # Confidence calculation
        confidence_data = self.confidence_engine.get_breakdown(
            all_distributions,
            len(self.spin_history),
            pattern_strength,
            self.spin_history
        )
        confidence = float(confidence_data['overall'])
        mode = confidence_data['mode']

        # Generate bet suggestions based on confidence
        bets = self._generate_bet_suggestions(
            ensemble_probs, confidence, top_numbers,
            red_prob, black_prob, d1_prob, d2_prob, d3_prob,
            c1_prob, c2_prob, c3_prob, high_prob, low_prob,
            odd_prob, even_prob
        )

        # Determine suggested color/dozen/etc for accuracy tracking
        suggested_color = 'red' if red_prob > black_prob else 'black'
        dozens = {'1st': d1_prob, '2nd': d2_prob, '3rd': d3_prob}
        suggested_dozen = max(dozens, key=dozens.get)

        # Regime detection
        regime_info = self.regime_detector.check(self.spin_history)

        # Adaptive weight info
        adaptive_weights = self.performance_tracker.get_adaptive_weights()

        prediction = {
            'spin_number': self.prediction_count,
            'top_numbers': top_numbers,
            'top_probabilities': top_probs,
            'anchors': anchors,
            'anchor_details': anchor_details,
            'confidence': confidence,
            'confidence_breakdown': confidence_data,
            'mode': mode,
            'bets': bets,
            'group_probabilities': {
                'red': float(round(red_prob, 4)),
                'black': float(round(black_prob, 4)),
                'dozen_1st': float(round(d1_prob, 4)),
                'dozen_2nd': float(round(d2_prob, 4)),
                'dozen_3rd': float(round(d3_prob, 4)),
                'column_1st': float(round(c1_prob, 4)),
                'column_2nd': float(round(c2_prob, 4)),
                'column_3rd': float(round(c3_prob, 4)),
                'high': float(round(high_prob, 4)),
                'low': float(round(low_prob, 4)),
                'odd': float(round(odd_prob, 4)),
                'even': float(round(even_prob, 4))
            },
            'pattern_strength': float(pattern_strength),
            'suggested_color': suggested_color,
            'suggested_dozen': suggested_dozen,
            'suggested_high_low': 'high' if high_prob > low_prob else 'low',
            'suggested_odd_even': 'odd' if odd_prob > even_prob else 'even',
            'lstm_trained': self.lstm.is_trained,
            'total_spins': len(self.spin_history),
            'frequency_summary': {
                'hot': self.frequency.get_hot_numbers(3),
                'cold': self.frequency.get_cold_numbers(3),
                'bias_score': self.frequency.get_bias_score()
            },
            'markov_strength': self.markov.get_transition_strength(),
            'regime': regime_info,
            'adaptive_weights': {
                k: round(v, 3) for k, v in adaptive_weights.items()
            },
            'performance_observations': self.performance_tracker.observation_count,
            'consecutive_misses': self.consecutive_misses,
            'exploration_active': self.consecutive_misses >= CONSECUTIVE_MISS_THRESHOLD,
        }

        self.last_prediction = prediction
        return prediction

    def _generate_bet_suggestions(self, probs, confidence, top_numbers,
                                   red_prob, black_prob, d1, d2, d3,
                                   c1, c2, c3, high, low, odd, even):
        """Generate straight number bet suggestions for all predicted numbers."""
        bets = []

        # Straight bets on all predicted numbers (anchors + neighbours)
        for num in top_numbers:
            bets.append({
                'type': 'straight',
                'value': str(num),
                'probability': round(float(probs[num]), 4),
                'payout': '35:1',
                'risk': 'high',
                'color': get_number_color(num)
            })

        return bets

    def train_lstm(self):
        """Explicitly trigger LSTM training."""
        return self.lstm.train()

    def load_clusters(self, clusters):
        """Train models on multiple independent data clusters (FULL RESET).
        Used only at session start to bootstrap from historical data.
        Each cluster is treated as a separate sequence - NOT concatenated.
        Models learn patterns from each cluster independently.

        Args:
            clusters: list of dicts with 'spins' key containing number lists
        """
        # For frequency analysis: all data contributes (number frequency is global)
        all_spins = []
        for cluster in clusters:
            all_spins.extend(cluster['spins'])
        self.frequency.load_history(all_spins)

        # For Markov chains: train on each cluster separately to avoid
        # false transitions between end of one cluster and start of another
        self.markov = MarkovChain()
        for i, cluster in enumerate(clusters):
            if len(cluster['spins']) >= 3:
                temp_markov = MarkovChain()
                temp_markov.load_history(cluster['spins'])
                # Accumulate transition counts
                self.markov.counts_1 += temp_markov.counts_1
                self.markov.counts_2.update(temp_markov.counts_2)
            # Yield to event loop periodically to keep WebSocket heartbeats alive
            if i % 5 == 0:
                _sleep(0)
        # Rebuild transition matrices from accumulated counts
        for i in range(TOTAL_NUMBERS):
            row_sum = self.markov.counts_1[i].sum()
            if row_sum > 0:
                self.markov.transition_1[i] = self.markov.counts_1[i] / row_sum
        for key, counts in self.markov.counts_2.items():
            row_sum = counts.sum()
            if row_sum > 0:
                self.markov.transition_2[key] = counts / row_sum
        # Set the last cluster's spins as current context for Markov prediction
        if clusters:
            last_cluster = clusters[-1]['spins']
            self.markov.spin_history = list(last_cluster)
        # Track total spins for accurate status reporting
        self.markov._total_spins_loaded = len(all_spins)

        # For patterns: use ALL data (pattern detection benefits from full history)
        self.patterns.load_history(all_spins)

        # For LSTM: train on all data (it learns sequences within its window)
        self.lstm.load_history(all_spins)

        # Keep combined history for reference
        self.spin_history = all_spins

        return {
            'clusters_loaded': len(clusters),
            'total_spins': len(all_spins),
            'cluster_sizes': [c['size'] for c in clusters]
        }

    def update_incremental(self, numbers):
        """INCREMENTALLY add new data to existing trained models.
        Does NOT reset any model state - applies new numbers on top of
        what models have already learned. Used for data imports during
        an active session.

        NOTE: Does NOT auto-retrain LSTM. The caller (import handler)
        decides whether to train based on user's button choice.

        Args:
            numbers: list of new spin numbers to feed into models
        Returns:
            dict with update stats
        """
        added = 0
        for num in numbers:
            self.spin_history.append(num)
            self.frequency.update(num)
            self.markov.update(num)
            self.patterns.update(num)
            self.lstm.update(num)
            added += 1
            # Yield to event loop periodically to keep WebSocket heartbeats alive
            if added % 20 == 0:
                _sleep(0)

        return {
            'numbers_added': added,
            'total_spins': len(self.spin_history),
            'mode': 'incremental'
        }

    def undo_last(self):
        """Remove the last spin from all models and regenerate state."""
        if not self.spin_history:
            return None

        removed = self.spin_history.pop()

        # Reload all models from remaining history
        self.frequency.load_history(self.spin_history)
        self.markov.load_history(self.spin_history)
        self.patterns.load_history(self.spin_history)
        self.lstm.load_history(self.spin_history)

        self.last_prediction = None
        return removed

    def run_test(self, numbers):
        """Test mode: run predictions against known data and return accuracy report.

        Takes a list of numbers, uses first N-1 to predict, compares against actual.
        Returns detailed accuracy metrics.

        Args:
            numbers: list of spin numbers to test against
        """
        if len(numbers) < 5:
            return {'error': 'Need at least 5 numbers for test mode'}

        # Reset models for clean test
        test_predictor = EnsemblePredictor()
        results = []
        correct_number = 0
        correct_color = 0
        correct_dozen = 0
        correct_half = 0
        correct_parity = 0
        total_predictions = 0
        correct_in_top5 = 0

        # Feed numbers one at a time, predict before each
        for i in range(len(numbers)):
            actual = numbers[i]

            if i >= 3:  # Need at least 3 spins before predicting
                prediction = test_predictor.predict()
                total_predictions += 1

                # Check accuracy
                top5 = prediction.get('top_numbers', [])[:5]
                hit_number = actual in top5
                if hit_number:
                    correct_in_top5 += 1

                pred_color = prediction.get('suggested_color', '')
                actual_color = get_number_color(actual)
                hit_color = pred_color == actual_color
                if hit_color:
                    correct_color += 1

                pred_dozen = prediction.get('suggested_dozen', '')
                actual_dozen = self._get_dozen(actual)
                hit_dozen = pred_dozen == actual_dozen
                if hit_dozen:
                    correct_dozen += 1

                pred_hl = prediction.get('suggested_high_low', '')
                actual_hl = 'high' if actual >= 19 and actual != 0 else 'low' if actual != 0 else 'zero'
                hit_hl = pred_hl == actual_hl
                if hit_hl:
                    correct_half += 1

                pred_oe = prediction.get('suggested_odd_even', '')
                actual_oe = 'odd' if actual % 2 == 1 and actual != 0 else 'even' if actual != 0 else 'zero'
                hit_oe = pred_oe == actual_oe
                if hit_oe:
                    correct_parity += 1

                results.append({
                    'spin': i + 1,
                    'actual': actual,
                    'actual_color': actual_color,
                    'predicted_top5': top5,
                    'confidence': prediction.get('confidence', 0),
                    'mode': prediction.get('mode', 'WAIT'),
                    'hit_top5': hit_number,
                    'hit_color': hit_color,
                    'hit_dozen': hit_dozen,
                    'suggested_color': pred_color,
                    'suggested_dozen': pred_dozen,
                })

            # Feed the actual number to update models
            test_predictor.update(actual)

        # Calculate accuracy rates
        report = {
            'total_spins': len(numbers),
            'total_predictions': total_predictions,
            'accuracy': {
                'top5_number': round(correct_in_top5 / total_predictions * 100, 1) if total_predictions else 0,
                'color': round(correct_color / total_predictions * 100, 1) if total_predictions else 0,
                'dozen': round(correct_dozen / total_predictions * 100, 1) if total_predictions else 0,
                'high_low': round(correct_half / total_predictions * 100, 1) if total_predictions else 0,
                'odd_even': round(correct_parity / total_predictions * 100, 1) if total_predictions else 0,
            },
            'expected_random': {
                'top5_number': round(5 / 37 * 100, 1),  # 13.5%
                'color': round(18 / 37 * 100, 1),        # 48.6%
                'dozen': round(12 / 37 * 100, 1),        # 32.4%
                'high_low': round(18 / 37 * 100, 1),     # 48.6%
                'odd_even': round(18 / 37 * 100, 1),     # 48.6%
            },
            'details': results[-20:],  # Last 20 for UI display
            'total_correct_top5': correct_in_top5,
            'total_correct_color': correct_color,
            'total_correct_dozen': correct_dozen,
        }

        return report

    def get_model_status(self):
        """Status of all models for the dashboard.
        Uses ensemble spin_history as single source of truth for spin counts
        so all models report the same number consistently.
        """
        # Single source of truth: ensemble spin_history
        total_spins = len(self.spin_history)

        # Individual model "active" checks use their own data
        freq_spins = len(self.frequency.spin_history)
        markov_has_data = (len(self.markov.spin_history) >= 3 or
                          getattr(self.markov, '_total_spins_loaded', 0) >= 3)
        pattern_spins = len(self.patterns.spin_history)
        lstm_spins = len(self.lstm.spin_history)

        # Adaptive weight info
        weights = self.performance_tracker.get_adaptive_weights()
        using_adaptive = self.performance_tracker.observation_count >= ADAPTIVE_MIN_OBSERVATIONS

        return {
            'frequency': {
                'spins': total_spins,
                'bias_score': self.frequency.get_bias_score(),
                'status': 'active' if freq_spins > 0 else 'idle',
                'weight': round(weights['frequency'], 3),
            },
            'markov': {
                'spins': total_spins,
                'strength': self.markov.get_transition_strength(),
                'status': 'active' if markov_has_data else 'idle',
                'weight': round(weights['markov'], 3),
            },
            'patterns': {
                'spins': total_spins,
                'strength': self.patterns.get_pattern_strength(),
                'status': 'active' if pattern_spins >= 10 else 'idle',
                'weight': round(weights['patterns'], 3),
            },
            'lstm': {
                'spins': total_spins,
                'trained': self.lstm.is_trained and total_spins > 0,
                'device': str(self.lstm.device),
                'confidence': self.lstm.get_confidence_score(),
                'status': ('trained' if self.lstm.is_trained and total_spins > 0
                           else 'collecting_data' if lstm_spins > 0
                           else 'idle'),
                'weight': round(weights['lstm'], 3),
            },
            'adaptive_weights_active': using_adaptive,
            'performance_observations': self.performance_tracker.observation_count,
        }

    # ─── State Persistence ──────────────────────────────────────────────

    def save_state(self):
        """Persist all non-LSTM model state to disk for fast session restart.
        LSTM weights are already persisted separately via PyTorch checkpoints.
        """
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            state = {
                'version': 1,
                # FrequencyAnalyzer
                'frequency': {
                    'frequency_counts': dict(self.frequency.frequency_counts),
                    'spin_history': list(self.frequency.spin_history),
                },
                # MarkovChain
                'markov': {
                    'counts_1': self.markov.counts_1,
                    'counts_2': {k: v for k, v in self.markov.counts_2.items()},
                    'transition_1': self.markov.transition_1,
                    'transition_2': dict(self.markov.transition_2),
                    'spin_history': list(self.markov.spin_history[-50:]),
                    # Use ensemble spin_history length as ground truth for total
                    # (markov.spin_history is truncated to last 50 for context)
                    'total_spins': max(
                        getattr(self.markov, '_total_spins_loaded', 0),
                        len(self.markov.spin_history),
                        len(self.spin_history)
                    ),
                },
                # PatternDetector
                'patterns': {
                    'spin_history': list(self.patterns.spin_history),
                },
                # ConfidenceEngine
                'confidence': {
                    'prediction_history': list(self.confidence_engine.prediction_history),
                    'accuracy_window': list(self.confidence_engine.accuracy_window),
                    'category_accuracy': {
                        k: list(v) for k, v in self.confidence_engine.category_accuracy.items()
                    },
                },
                # LSTM trained flag (weights saved separately, but flag needs to persist)
                'lstm': {
                    'is_trained': self.lstm.is_trained,
                    'training_loss_history': self.lstm.training_loss_history[-20:],
                    'validation_loss_history': self.lstm.validation_loss_history[-20:],
                },
                # EnsemblePredictor
                'ensemble': {
                    'spin_history': list(self.spin_history),
                    'prediction_count': self.prediction_count,
                },
                # Adaptive model performance tracker
                'performance_tracker': self.performance_tracker.get_state(),
                # Confidence calibration data
                'confidence_calibration': (
                    self.confidence_engine.get_calibration_state()
                    if hasattr(self.confidence_engine, 'get_calibration_state')
                    else {}
                ),
                # Loss-reactive strategy state
                'consecutive_misses': self.consecutive_misses,
                'sector_miss_counts': dict(self.sector_miss_counts),
            }

            tmp_path = MODEL_STATE_PATH + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, MODEL_STATE_PATH)

            print(f"[State] Saved model state ({len(self.spin_history)} spins)")
            return True
        except Exception as e:
            print(f"[State] Failed to save: {e}")
            return False

    def load_state(self):
        """Load persisted non-LSTM model state from disk.
        Returns True if loaded successfully, False otherwise.
        LSTM weights are loaded separately in LSTMPredictor.__init__().
        """
        if not os.path.exists(MODEL_STATE_PATH):
            print("[State] No cached model state found")
            return False

        try:
            with open(MODEL_STATE_PATH, 'rb') as f:
                state = pickle.load(f)

            if state.get('version') != 1:
                print(f"[State] Unknown state version {state.get('version')}, ignoring")
                return False

            # Restore FrequencyAnalyzer
            freq = state['frequency']
            self.frequency.frequency_counts = Counter(freq['frequency_counts'])
            self.frequency.spin_history = freq['spin_history']

            # Restore MarkovChain
            m = state['markov']
            self.markov.counts_1 = m['counts_1']
            self.markov.counts_2 = defaultdict(lambda: np.zeros(TOTAL_NUMBERS))
            self.markov.counts_2.update(m['counts_2'])
            self.markov.transition_1 = m['transition_1']
            self.markov.transition_2 = m['transition_2']
            self.markov.spin_history = m['spin_history']
            # Store actual total for accurate reporting (spin_history is truncated to last 50)
            self.markov._total_spins_loaded = m.get('total_spins', len(m['spin_history']))

            # Restore PatternDetector
            self.patterns.spin_history = state['patterns']['spin_history']

            # Restore ConfidenceEngine
            c = state['confidence']
            self.confidence_engine.prediction_history = deque(
                c['prediction_history'], maxlen=100
            )
            self.confidence_engine.accuracy_window = deque(
                c['accuracy_window'], maxlen=50
            )
            for cat_key, cat_data in c['category_accuracy'].items():
                self.confidence_engine.category_accuracy[cat_key] = deque(
                    cat_data, maxlen=50
                )

            # Restore EnsemblePredictor
            ens = state['ensemble']
            self.spin_history = ens['spin_history']
            self.prediction_count = ens['prediction_count']

            # Restore LSTM state
            self.lstm.spin_history = list(self.spin_history)
            # Load LSTM weights from checkpoint (not auto-loaded on __init__ anymore)
            self.lstm.load_checkpoint()
            if 'lstm' in state:
                self.lstm.is_trained = state['lstm'].get('is_trained', self.lstm.is_trained)
                saved_loss = state['lstm'].get('training_loss_history', [])
                if saved_loss:
                    self.lstm.training_loss_history = saved_loss
                saved_val_loss = state['lstm'].get('validation_loss_history', [])
                if saved_val_loss:
                    self.lstm.validation_loss_history = saved_val_loss

            # Restore performance tracker
            if 'performance_tracker' in state:
                self.performance_tracker.load_state(state['performance_tracker'])

            # Restore confidence calibration
            if ('confidence_calibration' in state and
                    hasattr(self.confidence_engine, 'load_calibration_state')):
                self.confidence_engine.load_calibration_state(
                    state['confidence_calibration']
                )

            # Restore loss-reactive strategy state
            self.consecutive_misses = state.get('consecutive_misses', 0)
            self.sector_miss_counts = state.get('sector_miss_counts', {})

            # ── Data integrity check ─────────────────────────────────
            # If every number 0-36 appears exactly the same count, the
            # cached data is almost certainly corrupt (e.g. sequential
            # 0,1,2,...,36 repeated).  Real roulette data always has
            # some natural variance.  Reject the cache and force a
            # rebuild from the source files.
            if len(self.spin_history) >= TOTAL_NUMBERS * 2:
                counts = Counter(self.spin_history)
                if len(counts) == TOTAL_NUMBERS:
                    count_values = list(counts.values())
                    if min(count_values) == max(count_values):
                        print(f"[State] WARNING: Cached data is perfectly uniform "
                              f"({min(count_values)} of each number in "
                              f"{len(self.spin_history)} spins) — likely corrupt. "
                              f"Discarding cache.")
                        os.remove(MODEL_STATE_PATH)
                        self.__init__()  # Reset to fresh state
                        return False

            print(f"[State] Loaded cached state: {len(self.spin_history)} spins, LSTM trained={self.lstm.is_trained}")
            return True

        except Exception as e:
            print(f"[State] Failed to load: {e}")
            try:
                os.remove(MODEL_STATE_PATH)
                print("[State] Removed corrupt state file")
            except OSError:
                pass
            return False

    @staticmethod
    def is_state_fresh():
        """Check if cached model state is newer than all session files.
        Returns True if cache is usable without rebuilding.
        """
        if not os.path.exists(MODEL_STATE_PATH):
            return False

        cache_mtime = os.path.getmtime(MODEL_STATE_PATH)

        if not os.path.exists(SESSIONS_DIR):
            return True

        for filename in os.listdir(SESSIONS_DIR):
            if filename.startswith('session_') and filename.endswith('.json'):
                filepath = os.path.join(SESSIONS_DIR, filename)
                if os.path.getmtime(filepath) > cache_mtime:
                    return False

        return True
