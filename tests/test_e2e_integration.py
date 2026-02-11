"""
Comprehensive E2E, Integration, Serialization, Regression, and UI Contract Tests.

Tests cover:
  1. JSON Serialization — ALL SocketIO emit paths must produce valid JSON
  2. E2E Flow — startup → train → session → spin → predict → undo → end
  3. Edge Cases — empty data, invalid inputs, boundary conditions
  4. State Persistence — save → reload → verify consistency
  5. UI Contract — response shapes match what frontend JavaScript expects
  6. Regression — all previously-fixed bugs remain fixed
"""

import json
import os
import sys
import math
import pickle
import tempfile
import shutil
import numpy as np
import pytest
from collections import Counter, deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    TOTAL_NUMBERS, TOP_PREDICTIONS_COUNT, RETRAIN_INTERVAL,
    RED_NUMBERS, BLACK_NUMBERS, FIRST_DOZEN, SECOND_DOZEN, THIRD_DOZEN,
    WHEEL_ORDER, NUMBER_TO_POSITION, MODEL_STATE_PATH, LSTM_MODEL_PATH,
    CONFIDENCE_BET_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD,
    get_number_color,
)
from app.ml.frequency_analyzer import FrequencyAnalyzer
from app.ml.markov_chain import MarkovChain
from app.ml.pattern_detector import PatternDetector
from app.ml.lstm_predictor import LSTMPredictor
from app.ml.confidence import ConfidenceEngine, ConfidenceCalibrator
from app.ml.ensemble import EnsemblePredictor, ModelPerformanceTracker, RegimeDetector
from app.ml.feature_engine import FeatureEngine
from app.money.bankroll_manager import BankrollManager
from app.session.session_manager import SessionManager

# ─── Test Data ─────────────────────────────────────────────────────────
# Real-ish roulette data for testing
SAMPLE_SPINS = [
    17, 25, 2, 21, 4, 19, 15, 3, 26, 0,
    32, 14, 35, 22, 9, 18, 29, 7, 28, 12,
    8, 30, 11, 36, 13, 27, 6, 34, 10, 33,
    1, 20, 16, 5, 24, 23, 31, 17, 25, 2,
    21, 4, 19, 15, 3, 26, 0, 32, 14, 35,
]

LARGE_SAMPLE = SAMPLE_SPINS * 10  # 500 spins

_MODEL_KEYS = ('frequency', 'markov', 'patterns', 'lstm')


# ═══════════════════════════════════════════════════════════════════════
#  1. JSON SERIALIZATION TESTS
#     Every SocketIO emit must produce valid JSON.
# ═══════════════════════════════════════════════════════════════════════

class TestJSONSerialization:
    """Verify all data structures sent via SocketIO are JSON-serializable.
    This was the root cause of the 'bool_ not serializable' bug.
    """

    @pytest.fixture
    def trained_predictor(self):
        """An EnsemblePredictor with 500 spins of training data."""
        p = EnsemblePredictor()
        for num in LARGE_SAMPLE:
            p.update(num)
        return p

    @pytest.fixture
    def bankroll(self):
        return BankrollManager()

    def _assert_json_serializable(self, obj, label="object"):
        """Assert that obj can be JSON-serialized without errors."""
        try:
            result = json.dumps(obj)
            assert isinstance(result, str)
            assert len(result) > 0
        except TypeError as e:
            # Find the offending value for a better error message
            bad_paths = []
            self._find_non_serializable(obj, "", bad_paths)
            pytest.fail(
                f"{label} is NOT JSON-serializable: {e}\n"
                f"Non-serializable paths: {bad_paths}"
            )

    def _find_non_serializable(self, obj, path, results):
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._find_non_serializable(v, f"{path}.{k}", results)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._find_non_serializable(v, f"{path}[{i}]", results)
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            results.append(f"{path}: {type(obj).__name__} = {obj}")

    def test_predict_response_serializable(self, trained_predictor, bankroll):
        """The full prediction_result emit payload must serialize."""
        prediction = trained_predictor.predict()
        should_bet, reason = bankroll.should_bet(prediction['confidence'], prediction['mode'])
        bet_per = bankroll.calculate_bet_amount(prediction['confidence'], 'straight')
        total_bet = round(bet_per * len(prediction['top_numbers']), 2)

        recommended_bet = None
        if should_bet and prediction['top_numbers']:
            recommended_bet = {
                'type': 'straight',
                'value': str(prediction['anchors'][0]) if prediction.get('anchors') else str(prediction['top_numbers'][0]),
                'numbers': prediction['top_numbers'],
                'amount': bet_per,
                'total_bet': total_bet,
                'payout': '35:1',
                'risk': 'high',
                'probability': round(len(prediction['top_numbers']) / 37, 4)
            }

        money_advice = bankroll.get_advice(
            ai_confidence=prediction['confidence'],
            ai_mode=prediction['mode'],
            prediction=prediction,
        )

        response = {
            'prediction': prediction,
            'should_bet': should_bet,
            'bet_reason': reason,
            'recommended_bet': recommended_bet,
            'bet_amount': bet_per,
            'total_bet': total_bet,
            'bankroll': bankroll.get_status(),
            'money_advice': money_advice,
        }

        self._assert_json_serializable(response, "prediction_result")

    def test_spin_processed_response_serializable(self, trained_predictor, bankroll):
        """The full spin_processed emit payload must serialize."""
        # Simulate submitting a spin
        number = 17
        trained_predictor.update(number)

        next_pred = trained_predictor.predict()
        should_bet, reason = bankroll.should_bet(next_pred['confidence'], next_pred['mode'])
        next_bet_per = bankroll.calculate_bet_amount(next_pred['confidence'], 'straight')
        next_total = round(next_bet_per * len(next_pred['top_numbers']), 2)

        money_advice = bankroll.get_advice(
            ai_confidence=next_pred['confidence'],
            ai_mode=next_pred['mode'],
            prediction=next_pred,
        )

        response = {
            'spin_result': {
                'number': number,
                'color': get_number_color(number),
                'bet_result': None,
            },
            'next_prediction': next_pred,
            'should_bet': should_bet,
            'bet_reason': reason,
            'recommended_bet': None,
            'bet_amount': next_bet_per,
            'total_bet': next_total,
            'bankroll': bankroll.get_status(),
            'model_status': trained_predictor.get_model_status(),
            'money_advice': money_advice,
        }

        self._assert_json_serializable(response, "spin_processed")

    def test_model_status_serializable(self, trained_predictor):
        """get_model_status must serialize (used in many emits)."""
        status = trained_predictor.get_model_status()
        self._assert_json_serializable(status, "model_status")

    def test_confidence_breakdown_serializable(self, trained_predictor):
        """confidence_breakdown with calibration flag must serialize."""
        pred = trained_predictor.predict()
        breakdown = pred['confidence_breakdown']

        # Specifically check the calibrated field type
        assert isinstance(breakdown['calibrated'], bool), \
            f"calibrated should be Python bool, got {type(breakdown['calibrated']).__name__}"

        self._assert_json_serializable(breakdown, "confidence_breakdown")

    def test_group_probabilities_serializable(self, trained_predictor):
        """group_probabilities should all be native Python floats."""
        pred = trained_predictor.predict()
        gp = pred['group_probabilities']

        for key, val in gp.items():
            assert isinstance(val, float), \
                f"group_probabilities['{key}'] should be float, got {type(val).__name__}"

        self._assert_json_serializable(gp, "group_probabilities")

    def test_regime_info_serializable(self, trained_predictor):
        """regime detection output must serialize."""
        pred = trained_predictor.predict()
        regime = pred['regime']
        self._assert_json_serializable(regime, "regime_info")

    def test_adaptive_weights_serializable(self, trained_predictor):
        """adaptive_weights must contain only native Python types."""
        pred = trained_predictor.predict()
        aw = pred['adaptive_weights']
        for key, val in aw.items():
            assert isinstance(val, float), \
                f"adaptive_weights['{key}'] should be float, got {type(val).__name__}"
        self._assert_json_serializable(aw, "adaptive_weights")

    def test_bankroll_status_serializable(self, bankroll):
        """bankroll.get_status() must serialize."""
        status = bankroll.get_status()
        self._assert_json_serializable(status, "bankroll_status")

    def test_money_advice_serializable(self, trained_predictor, bankroll):
        """bankroll.get_advice() must serialize."""
        pred = trained_predictor.predict()
        advice = bankroll.get_advice(
            ai_confidence=pred['confidence'],
            ai_mode=pred['mode'],
            prediction=pred,
        )
        self._assert_json_serializable(advice, "money_advice")

    def test_training_complete_response_serializable(self, trained_predictor):
        """The training_complete emit payload structure must serialize."""
        # Simulate what handle_train_model builds
        status = trained_predictor.get_model_status()
        response = {
            'status': 'trained',
            'message': 'Test message',
            'files': [{'filename': 'data1.txt', 'numbers': 100, 'skipped': 0, 'error': None}],
            'summary': {
                'total_files': 1,
                'total_spins': 500,
                'unique_numbers': 37,
                'training_time': 1.5,
                'lstm_result': {'status': 'trained', 'final_loss': '3.610', 'epochs': 15, 'dataset_size': 460},
                'hot_numbers': [{'number': 17, 'count': 20, 'ratio': 1.48}],
                'cold_numbers': [{'number': 36, 'count': 5, 'ratio': 0.37}],
                'color_distribution': {'red': 240, 'red_pct': 48.0, 'black': 245, 'black_pct': 49.0, 'green': 15, 'green_pct': 3.0},
            },
            'last_numbers': [17, 25, 2, 21, 4, 19, 15, 3, 26, 0],
            'model_status': status,
        }
        self._assert_json_serializable(response, "training_complete")

    def test_no_numpy_types_in_prediction(self, trained_predictor):
        """Exhaustive check: no numpy types anywhere in prediction dict."""
        pred = trained_predictor.predict()
        bad_paths = []
        self._find_non_serializable(pred, "prediction", bad_paths)
        assert bad_paths == [], f"Found numpy types in prediction:\n" + "\n".join(bad_paths)

    def test_serialization_after_multiple_predictions(self, trained_predictor, bankroll):
        """Serialization should work even after many predictions (calibrator active)."""
        for i in range(50):
            trained_predictor.update(i % 37)
            pred = trained_predictor.predict()

        # Build full response
        pred = trained_predictor.predict()
        response = {
            'prediction': pred,
            'should_bet': True,
            'bankroll': bankroll.get_status(),
            'model_status': trained_predictor.get_model_status(),
        }
        self._assert_json_serializable(response, "after_50_predictions")


# ═══════════════════════════════════════════════════════════════════════
#  2. E2E FLOW TESTS
#     Full lifecycle: train → session → spins → predictions → undo → end
# ═══════════════════════════════════════════════════════════════════════

class TestE2EFlow:
    """Simulate the full user journey through the application."""

    def test_full_lifecycle(self):
        """Train → predict → submit spins → predict again → undo → end."""
        predictor = EnsemblePredictor()
        bankroll = BankrollManager()

        # Phase 1: Load data (like handle_train_model)
        for num in SAMPLE_SPINS:
            predictor.update(num)

        assert len(predictor.spin_history) == len(SAMPLE_SPINS)

        # Phase 2: Get prediction (like handle_get_prediction)
        pred = predictor.predict()
        assert 'top_numbers' in pred
        assert 'confidence' in pred
        assert 'mode' in pred
        # Dynamic prediction count: AI picks 3-20 numbers based on confidence
        from config import TOP_PREDICTIONS_MAX
        assert 3 <= len(pred['top_numbers']) <= TOP_PREDICTIONS_MAX

        should_bet, reason = bankroll.should_bet(pred['confidence'], pred['mode'])
        assert isinstance(should_bet, bool)
        assert isinstance(reason, str)

        # Phase 3: Submit several spins (like handle_submit_spin)
        new_spins = [5, 17, 23, 0, 12]
        for num in new_spins:
            predictor.update(num)
            next_pred = predictor.predict()
            assert 'top_numbers' in next_pred
            assert next_pred['total_spins'] == len(predictor.spin_history)

        assert len(predictor.spin_history) == len(SAMPLE_SPINS) + len(new_spins)

        # Phase 4: Undo last spin (like handle_undo_spin)
        removed = predictor.undo_last()
        assert removed == 12
        assert len(predictor.spin_history) == len(SAMPLE_SPINS) + len(new_spins) - 1

        # Phase 5: Prediction still works after undo
        post_undo_pred = predictor.predict()
        assert 'top_numbers' in post_undo_pred

    def test_cluster_loading_e2e(self):
        """Load clusters like session_start does, then predict."""
        predictor = EnsemblePredictor()

        clusters = [
            {'spins': SAMPLE_SPINS[:25], 'size': 25, 'source': 'test1.txt'},
            {'spins': SAMPLE_SPINS[25:], 'size': 25, 'source': 'test2.txt'},
        ]

        result = predictor.load_clusters(clusters)
        assert result['clusters_loaded'] == 2
        assert result['total_spins'] == 50

        pred = predictor.predict()
        assert pred['total_spins'] == 50
        assert len(pred['top_numbers']) > 0

    def test_incremental_update_e2e(self):
        """Incremental import on top of existing data."""
        predictor = EnsemblePredictor()

        # Initial load
        for num in SAMPLE_SPINS[:30]:
            predictor.update(num)
        assert len(predictor.spin_history) == 30

        # Incremental import (like handle_import_data)
        result = predictor.update_incremental(SAMPLE_SPINS[30:])
        assert result['numbers_added'] == 20
        assert result['total_spins'] == 50
        assert len(predictor.spin_history) == 50

        # Predictions still work
        pred = predictor.predict()
        assert pred['total_spins'] == 50

    def test_reset_e2e(self):
        """Full reset clears everything."""
        predictor = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            predictor.update(num)

        assert len(predictor.spin_history) > 0

        predictor.full_reset()

        assert len(predictor.spin_history) == 0
        assert predictor.lstm.is_trained is False
        assert predictor.prediction_count == 0
        assert predictor.performance_tracker.observation_count == 0

    def test_bankroll_bet_flow(self):
        """Bankroll management through win/loss cycles."""
        bankroll = BankrollManager()
        initial = bankroll.bankroll

        # Simulate a loss (payout_amount, not payout)
        bankroll.process_result(bet_amount=12, won=False, payout_amount=0)
        assert bankroll.bankroll < initial
        assert bankroll.consecutive_losses == 1
        assert bankroll.total_losses == 1

        # Simulate a win
        bankroll.process_result(bet_amount=12, won=True, payout_amount=35)
        assert bankroll.consecutive_losses == 0
        assert bankroll.total_wins == 1

    def test_run_test_mode(self):
        """Test mode: evaluate predictions against known data."""
        predictor = EnsemblePredictor()
        # Need some training data first
        for num in SAMPLE_SPINS:
            predictor.update(num)

        test_data = [5, 17, 23, 0, 12, 7, 31, 22, 14, 8]
        report = predictor.run_test(test_data)

        assert 'accuracy' in report
        assert 'total_predictions' in report
        assert report['total_spins'] == 10
        assert report['total_predictions'] > 0
        assert 0 <= report['accuracy']['color'] <= 100


# ═══════════════════════════════════════════════════════════════════════
#  3. EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_predict_with_no_data(self):
        """Prediction with zero spins should not crash."""
        p = EnsemblePredictor()
        pred = p.predict()
        assert 'top_numbers' in pred
        assert pred['total_spins'] == 0

    def test_predict_with_one_spin(self):
        """Prediction with just 1 spin."""
        p = EnsemblePredictor()
        p.update(17)
        pred = p.predict()
        assert pred['total_spins'] == 1
        assert isinstance(pred['confidence'], float)

    def test_predict_with_three_spins(self):
        """Minimum spins for Markov second-order."""
        p = EnsemblePredictor()
        for n in [5, 17, 23]:
            p.update(n)
        pred = p.predict()
        assert pred['total_spins'] == 3

    def test_all_same_number(self):
        """What happens when all spins are the same number."""
        p = EnsemblePredictor()
        for _ in range(100):
            p.update(17)

        pred = p.predict()
        assert 'top_numbers' in pred
        # Number 17 should be prominent
        assert 17 in pred['top_numbers']

    def test_only_zero(self):
        """All zeros — the green edge case."""
        p = EnsemblePredictor()
        for _ in range(50):
            p.update(0)
        pred = p.predict()
        assert 0 in pred['top_numbers']

    def test_sequential_numbers(self):
        """Sequential 0-36 repeated — uniform distribution."""
        p = EnsemblePredictor()
        for _ in range(5):
            for n in range(37):
                p.update(n)
        pred = p.predict()
        assert pred['total_spins'] == 185

    def test_undo_with_no_history(self):
        """Undo with empty history returns None."""
        p = EnsemblePredictor()
        removed = p.undo_last()
        assert removed is None

    def test_undo_restores_correct_state(self):
        """After undo, history length decreases by 1."""
        p = EnsemblePredictor()
        for n in [5, 17, 23]:
            p.update(n)
        p.undo_last()
        assert len(p.spin_history) == 2
        assert p.spin_history[-1] == 17

    def test_boundary_numbers(self):
        """Numbers 0 and 36 (boundaries) should work."""
        p = EnsemblePredictor()
        p.update(0)
        p.update(36)
        pred = p.predict()
        assert pred['total_spins'] == 2

    def test_pattern_detector_empty(self):
        """Pattern detector with no data."""
        pd = PatternDetector()
        assert pd.get_pattern_strength() == 0.0
        probs = pd.get_number_probabilities()
        assert len(probs) == TOTAL_NUMBERS

    def test_frequency_analyzer_empty(self):
        """Frequency analyzer with no data."""
        fa = FrequencyAnalyzer()
        assert fa.get_bias_score() == 0.0
        probs = fa.get_number_probabilities()
        assert len(probs) == TOTAL_NUMBERS
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_markov_chain_empty(self):
        """Markov chain with no data."""
        mc = MarkovChain()
        probs = mc.get_probabilities()
        assert len(probs) == TOTAL_NUMBERS

    def test_feature_engine_empty_context(self):
        """Feature engine with empty context."""
        fe = FeatureEngine()
        features = fe.extract_single(17, [])
        assert len(features) > 0
        assert np.all(np.isfinite(features))

    def test_confidence_engine_no_history(self):
        """Confidence engine with no prediction history."""
        ce = ConfidenceEngine()
        dists = [np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS) for _ in range(4)]
        breakdown = ce.get_breakdown(dists, 0, 0.0)
        assert 'overall' in breakdown
        assert isinstance(breakdown['overall'], float)
        assert isinstance(breakdown['calibrated'], bool)

    def test_large_spin_count(self):
        """2000+ spins should not crash or slow down dramatically."""
        p = EnsemblePredictor()
        import random
        random.seed(42)
        for _ in range(2000):
            p.update(random.randint(0, 36))
        pred = p.predict()
        assert pred['total_spins'] == 2000
        # Dynamic prediction count: AI picks 3-20 numbers
        from config import TOP_PREDICTIONS_MAX
        assert 3 <= len(pred['top_numbers']) <= TOP_PREDICTIONS_MAX


# ═══════════════════════════════════════════════════════════════════════
#  4. STATE PERSISTENCE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestStatePersistence:

    def test_save_and_load_state(self, tmp_path):
        """Save state, create new predictor, load, verify identical."""
        import app.ml.ensemble as ens_mod
        original_path = ens_mod.MODEL_STATE_PATH
        test_state_path = str(tmp_path / "test_model_state.pkl")
        ens_mod.MODEL_STATE_PATH = test_state_path

        try:
            # Train and save
            p1 = EnsemblePredictor()
            for num in SAMPLE_SPINS:
                p1.update(num)
            p1.predict()  # Generate a prediction so we have cache
            saved = p1.save_state()
            assert saved is True

            # Load into fresh predictor
            p2 = EnsemblePredictor()
            loaded = p2.load_state()
            assert loaded is True

            # Verify
            assert len(p2.spin_history) == len(p1.spin_history)
            assert p2.spin_history == p1.spin_history
            assert p2.prediction_count == p1.prediction_count

            # Both should produce similar predictions
            pred1 = p1.predict()
            pred2 = p2.predict()
            assert pred1['total_spins'] == pred2['total_spins']
        finally:
            ens_mod.MODEL_STATE_PATH = original_path

    def test_corrupted_state_handled_gracefully(self, tmp_path):
        """Corrupt state file should not crash, should return False."""
        # load_state() uses the module-level MODEL_STATE_PATH imported in ensemble.py
        # We need to patch it at the module level where it's used
        import app.ml.ensemble as ens_mod
        original_path = ens_mod.MODEL_STATE_PATH
        test_state_path = str(tmp_path / "corrupt_state.pkl")
        ens_mod.MODEL_STATE_PATH = test_state_path

        try:
            # Write garbage
            with open(test_state_path, 'wb') as f:
                f.write(b"THIS IS NOT PICKLE DATA")

            p = EnsemblePredictor()
            loaded = p.load_state()
            assert loaded is False
            assert len(p.spin_history) == 0
        finally:
            ens_mod.MODEL_STATE_PATH = original_path

    def test_missing_state_file(self, tmp_path):
        """Missing state file should return False gracefully."""
        import app.ml.ensemble as ens_mod
        original_path = ens_mod.MODEL_STATE_PATH
        ens_mod.MODEL_STATE_PATH = str(tmp_path / "nonexistent.pkl")

        try:
            p = EnsemblePredictor()
            loaded = p.load_state()
            assert loaded is False
        finally:
            ens_mod.MODEL_STATE_PATH = original_path

    def test_performance_tracker_persists(self, tmp_path):
        """Adaptive performance tracker state survives save/load."""
        import app.ml.ensemble as ens_mod
        original_path = ens_mod.MODEL_STATE_PATH
        test_state_path = str(tmp_path / "tracker_state.pkl")
        ens_mod.MODEL_STATE_PATH = test_state_path

        try:
            p1 = EnsemblePredictor()
            for num in LARGE_SAMPLE:
                p1.update(num)
                if len(p1.spin_history) % 10 == 0:
                    p1.predict()

            obs_before = p1.performance_tracker.observation_count
            assert obs_before > 0

            p1.save_state()

            p2 = EnsemblePredictor()
            p2.load_state()

            assert p2.performance_tracker.observation_count == obs_before
        finally:
            ens_mod.MODEL_STATE_PATH = original_path

    def test_confidence_calibrator_persists(self, tmp_path):
        """Confidence calibration data survives save/load."""
        import app.ml.ensemble as ens_mod
        original_path = ens_mod.MODEL_STATE_PATH
        test_state_path = str(tmp_path / "calib_state.pkl")
        ens_mod.MODEL_STATE_PATH = test_state_path

        try:
            p1 = EnsemblePredictor()
            for num in LARGE_SAMPLE:
                p1.update(num)

            # Force some calibration data
            p1.confidence_engine.calibrator.record(50.0, True)
            p1.confidence_engine.calibrator.record(50.0, False)
            p1.confidence_engine.calibrator.record(50.0, True)

            p1.save_state()

            p2 = EnsemblePredictor()
            p2.load_state()

            # Calibrator should have data
            bin_data = p2.confidence_engine.calibrator.bin_data
            total_records = sum(len(b) for b in bin_data)
            assert total_records >= 3
        finally:
            ens_mod.MODEL_STATE_PATH = original_path


# ═══════════════════════════════════════════════════════════════════════
#  5. UI CONTRACT TESTS
#     Verify response shapes match what frontend JavaScript expects.
# ═══════════════════════════════════════════════════════════════════════

class TestUIContract:
    """Ensure backend response shapes match frontend JavaScript expectations.
    Each test maps to specific frontend code in app.js.
    """

    @pytest.fixture
    def prediction_response(self):
        """Build a realistic prediction_result response."""
        p = EnsemblePredictor()
        for num in LARGE_SAMPLE:
            p.update(num)
        prediction = p.predict()
        b = BankrollManager()
        should_bet, reason = b.should_bet(prediction['confidence'], prediction['mode'])
        bet_per = b.calculate_bet_amount(prediction['confidence'], 'straight')
        total_bet = round(bet_per * len(prediction['top_numbers']), 2)

        recommended_bet = None
        if should_bet and prediction['top_numbers']:
            recommended_bet = {
                'type': 'straight',
                'value': str(prediction['anchors'][0]) if prediction.get('anchors') else str(prediction['top_numbers'][0]),
                'numbers': prediction['top_numbers'],
                'amount': bet_per,
                'total_bet': total_bet,
                'payout': '35:1',
                'risk': 'high',
                'probability': round(len(prediction['top_numbers']) / 37, 4)
            }

        advice = b.get_advice(
            ai_confidence=prediction['confidence'],
            ai_mode=prediction['mode'],
            prediction=prediction,
        )

        return {
            'prediction': prediction,
            'should_bet': should_bet,
            'bet_reason': reason,
            'recommended_bet': recommended_bet,
            'bet_amount': bet_per,
            'total_bet': total_bet,
            'bankroll': b.get_status(),
            'money_advice': advice,
        }

    # ── app.js line 1036-1201: updatePrediction(data) ──

    def test_prediction_has_confidence(self, prediction_response):
        """app.js line 1040: const confidence = pred.confidence"""
        pred = prediction_response['prediction']
        assert 'confidence' in pred
        assert isinstance(pred['confidence'], (int, float))

    def test_prediction_has_mode(self, prediction_response):
        """app.js line 1062: pred.mode"""
        pred = prediction_response['prediction']
        assert 'mode' in pred
        assert pred['mode'] in ('WAIT', 'BET', 'BET_HIGH')

    def test_prediction_has_top_numbers(self, prediction_response):
        """app.js line 1091: pred.top_numbers"""
        pred = prediction_response['prediction']
        assert 'top_numbers' in pred
        assert isinstance(pred['top_numbers'], list)
        assert all(isinstance(n, int) for n in pred['top_numbers'])

    def test_prediction_has_top_probabilities(self, prediction_response):
        """app.js line 1109: pred.top_probabilities[idx]"""
        pred = prediction_response['prediction']
        assert 'top_probabilities' in pred
        assert len(pred['top_probabilities']) == len(pred['top_numbers'])
        assert all(isinstance(p, float) for p in pred['top_probabilities'])

    def test_prediction_has_anchors(self, prediction_response):
        """app.js line 1080: pred.anchors"""
        pred = prediction_response['prediction']
        assert 'anchors' in pred
        assert isinstance(pred['anchors'], list)

    def test_prediction_has_anchor_details(self, prediction_response):
        """app.js line 1079: pred.anchor_details"""
        pred = prediction_response['prediction']
        assert 'anchor_details' in pred
        for ad in pred['anchor_details']:
            assert 'number' in ad
            assert 'spread' in ad
            assert 'numbers' in ad
            assert isinstance(ad['numbers'], list)

    def test_prediction_has_confidence_breakdown(self, prediction_response):
        """app.js line 1148: pred.confidence_breakdown.factors (6 factors)"""
        pred = prediction_response['prediction']
        assert 'confidence_breakdown' in pred
        cb = pred['confidence_breakdown']
        assert 'factors' in cb
        for factor_name in ('model_agreement', 'historical_accuracy', 'pattern_strength',
                            'sample_size', 'streak_momentum', 'recent_hit_rate'):
            assert factor_name in cb['factors'], f"Missing factor: {factor_name}"
            factor = cb['factors'][factor_name]
            assert 'score' in factor
            assert isinstance(factor['score'], float)

    def test_prediction_has_group_probabilities(self, prediction_response):
        """app.js line 1189: pred.group_probabilities"""
        pred = prediction_response['prediction']
        assert 'group_probabilities' in pred
        gp = pred['group_probabilities']
        expected_keys = ['red', 'black', 'dozen_1st', 'dozen_2nd', 'dozen_3rd',
                         'high', 'low', 'odd', 'even']
        for key in expected_keys:
            assert key in gp, f"Missing group_probabilities key: {key}"
            assert isinstance(gp[key], float), f"gp['{key}'] should be float"

    def test_response_has_should_bet(self, prediction_response):
        """app.js line 1059: data.should_bet"""
        assert 'should_bet' in prediction_response
        assert isinstance(prediction_response['should_bet'], bool)

    def test_response_has_bet_reason(self, prediction_response):
        """app.js line 1074: data.bet_reason"""
        assert 'bet_reason' in prediction_response
        assert isinstance(prediction_response['bet_reason'], str)

    def test_recommended_bet_shape(self, prediction_response):
        """app.js line 1063-1068: data.recommended_bet"""
        bet = prediction_response['recommended_bet']
        if bet is not None:
            assert 'amount' in bet
            assert 'total_bet' in bet
            assert 'numbers' in bet
            assert isinstance(bet['numbers'], list)

    # ── app.js line 1210-1324: updateMoneyAdvice(advice) ──

    def test_money_advice_shape(self, prediction_response):
        """app.js line 1215: advice.strategy, action, risk_level etc."""
        advice = prediction_response['money_advice']
        assert 'strategy' in advice
        assert 'action' in advice
        assert 'action_label' in advice
        assert 'reason' in advice
        assert 'risk_level' in advice
        assert 'drawdown' in advice
        assert 'momentum' in advice
        assert 'distance_to_target' in advice
        assert 'distance_to_stop_loss' in advice

    def test_money_advice_bet_sizing(self, prediction_response):
        """app.js line 1254: advice.bet_sizing.straight"""
        advice = prediction_response['money_advice']
        if advice['action'] == 'BET':
            assert 'bet_sizing' in advice
            assert 'straight' in advice['bet_sizing']

    # ── app.js line 1326-1356: updateBankroll(status) ──

    def test_bankroll_status_shape(self, prediction_response):
        """app.js line 1329-1341: status fields."""
        status = prediction_response['bankroll']
        required = ['bankroll', 'profit_loss', 'session_target', 'target_progress',
                     'win_rate', 'total_bets', 'total_wins', 'total_losses',
                     'consecutive_losses']
        for key in required:
            assert key in status, f"Missing bankroll key: {key}"

    # ── app.js line 1358-1392: updateModelStatus(models) ──

    def test_model_status_shape(self):
        """app.js line 1367: models.frequency.status, .bias_score, .spins"""
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            p.update(num)
        status = p.get_model_status()

        for key in _MODEL_KEYS:
            assert key in status, f"Missing model: {key}"
            assert 'spins' in status[key]
            assert 'status' in status[key]
            assert 'weight' in status[key]

        assert 'bias_score' in status['frequency']
        assert 'strength' in status['markov']
        assert 'strength' in status['patterns']
        assert 'trained' in status['lstm']
        assert 'device' in status['lstm']

    def test_model_status_extra_keys(self):
        """Model status has extra top-level keys that are NOT model dicts."""
        p = EnsemblePredictor()
        status = p.get_model_status()
        assert 'adaptive_weights_active' in status
        assert isinstance(status['adaptive_weights_active'], bool)
        assert 'performance_observations' in status
        assert isinstance(status['performance_observations'], int)


# ═══════════════════════════════════════════════════════════════════════
#  6. REGRESSION TESTS
#     Verify all previously-fixed bugs remain fixed.
# ═══════════════════════════════════════════════════════════════════════

class TestRegression:
    """Tests for previously-identified and fixed bugs."""

    def test_all_models_report_same_spin_count(self):
        """BUG FIX: All 4 models must show identical spin count.
        Previously, models had inconsistent spin counts.
        """
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            p.update(num)
        status = p.get_model_status()

        counts = set()
        for key in _MODEL_KEYS:
            counts.add(status[key]['spins'])

        assert len(counts) == 1, f"Models report different spin counts: {counts}"
        assert counts.pop() == len(SAMPLE_SPINS)

    def test_spin_count_after_cluster_load(self):
        """BUG FIX: Spin count consistent after cluster loading."""
        p = EnsemblePredictor()
        clusters = [
            {'spins': SAMPLE_SPINS[:25], 'size': 25, 'source': 'a.txt'},
            {'spins': SAMPLE_SPINS[25:], 'size': 25, 'source': 'b.txt'},
        ]
        p.load_clusters(clusters)
        status = p.get_model_status()

        for key in _MODEL_KEYS:
            assert status[key]['spins'] == 50, \
                f"{key} reports {status[key]['spins']} spins, expected 50"

    def test_bias_score_not_stuck_low(self):
        """BUG FIX: Bias score should be meaningful with 500+ spins.
        Previously, bias score was stuck at ~34 due to single-factor scoring.
        """
        p = EnsemblePredictor()
        for num in LARGE_SAMPLE:
            p.update(num)
        bias = p.frequency.get_bias_score()
        # With repeated data, bias should be significant (>40)
        assert bias > 40, f"Bias score {bias} too low — multi-factor scoring may be broken"

    def test_pattern_detector_gets_all_data(self):
        """BUG FIX: Pattern detector receives full training data."""
        p = EnsemblePredictor()
        clusters = [
            {'spins': SAMPLE_SPINS, 'size': len(SAMPLE_SPINS), 'source': 'test.txt'},
        ]
        p.load_clusters(clusters)
        assert len(p.patterns.spin_history) == len(SAMPLE_SPINS)

    def test_numpy_bool_not_in_confidence_breakdown(self):
        """BUG FIX: The 'calibrated' field in confidence breakdown must be
        a native Python bool, NOT numpy.bool_.
        This was the root cause of the 'bool_ is not JSON serializable' error
        that prevented spin submission and predictions from working.
        """
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            p.update(num)
        pred = p.predict()

        calibrated = pred['confidence_breakdown']['calibrated']
        assert type(calibrated) is bool, \
            f"calibrated is {type(calibrated).__name__}, must be Python bool"

    def test_numpy_float64_not_in_group_probabilities(self):
        """BUG FIX: group_probabilities must be native Python floats."""
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            p.update(num)
        pred = p.predict()

        for key, val in pred['group_probabilities'].items():
            assert type(val) is float, \
                f"group_probabilities['{key}'] is {type(val).__name__}, must be float"

    def test_confidence_is_native_float(self):
        """BUG FIX: prediction.confidence must be native Python float."""
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            p.update(num)
        pred = p.predict()
        assert type(pred['confidence']) is float, \
            f"confidence is {type(pred['confidence']).__name__}, must be float"

    def test_uniform_data_rejected_on_load(self):
        """BUG FIX: Perfectly uniform cached data is detected and rejected."""
        import app.ml.ensemble as ens_mod
        original_path = ens_mod.MODEL_STATE_PATH

        # Use a temp path to avoid touching real state
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        ens_mod.MODEL_STATE_PATH = temp_path

        try:
            # Build a predictor with perfectly uniform data
            p1 = EnsemblePredictor()
            # 37 * 4 = 148 spins, each number appears exactly 4 times
            uniform_data = list(range(37)) * 4
            p1.spin_history = uniform_data
            p1.frequency.load_history(uniform_data)
            p1.markov.load_history(uniform_data)
            p1.patterns.load_history(uniform_data)
            p1.lstm.load_history(uniform_data)
            p1.save_state()

            # Try loading — should reject
            p2 = EnsemblePredictor()
            loaded = p2.load_state()
            assert loaded is False, "Perfectly uniform data should be rejected as corrupt"
        finally:
            ens_mod.MODEL_STATE_PATH = original_path
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_model_status_iteration_safe(self):
        """BUG FIX: Iterating model_status must handle extra top-level keys.
        Previously, code iterated all items and tried to access ['spins']
        on boolean/int values (adaptive_weights_active, performance_observations).
        """
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            p.update(num)
        status = p.get_model_status()

        # Simulate what the old buggy code did (should NOT crash)
        for model_name in _MODEL_KEYS:
            s = status[model_name]
            assert 'spins' in s
            assert isinstance(s['spins'], int)

        # Extra keys exist and are not model dicts
        assert 'adaptive_weights_active' in status
        assert 'performance_observations' in status

    def test_sample_size_factor_monotonic(self):
        """BUG FIX: Sample size factor must increase monotonically."""
        ce = ConfidenceEngine()
        prev = 0
        for spins in [5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
            score = ce.calculate_sample_size_factor(spins)
            assert score >= prev, \
                f"Sample size factor decreased: {prev} → {score} at {spins} spins"
            prev = score

    def test_old_lstm_checkpoint_handled(self):
        """BUG FIX: Old LSTM checkpoint with incompatible architecture
        should not crash — should start fresh with a warning.
        """
        lstm = LSTMPredictor()
        # If there's a checkpoint that doesn't match, init should still work
        assert lstm.model is not None
        probs = lstm.predict()
        assert len(probs) == TOTAL_NUMBERS

    def test_anchor_no_overlap(self):
        """BUG FIX: No number should appear in more than one anchor group.
        Previously, nums_covered.append() was outside the 'if not selected'
        check, causing duplicate numbers across anchor groups.
        """
        p = EnsemblePredictor()
        for num in LARGE_SAMPLE:
            p.update(num)
        pred = p.predict()

        anchor_details = pred.get('anchor_details', [])
        seen = set()
        for ad in anchor_details:
            for num in ad['numbers']:
                assert num not in seen, \
                    f"Number {num} appears in multiple anchor groups — overlap bug!"
                seen.add(num)

    def test_prediction_has_exploration_fields(self):
        """Prediction must include consecutive_misses and exploration_active."""
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            p.update(num)
        pred = p.predict()

        assert 'consecutive_misses' in pred, \
            "prediction should include consecutive_misses"
        assert 'exploration_active' in pred, \
            "prediction should include exploration_active"
        assert isinstance(pred['consecutive_misses'], int)
        assert isinstance(pred['exploration_active'], bool)

    def test_loss_reactive_strategy_activates(self):
        """After N consecutive misses, exploration should activate and
        predictions should shift to different numbers.
        """
        from config import CONSECUTIVE_MISS_THRESHOLD

        p = EnsemblePredictor()
        for num in LARGE_SAMPLE:
            p.update(num)

        # Get initial prediction
        initial_pred = p.predict()
        initial_nums = set(initial_pred['top_numbers'])

        # Simulate consecutive misses by feeding numbers NOT in predictions
        all_numbers = set(range(37))
        for _ in range(CONSECUTIVE_MISS_THRESHOLD + 1):
            miss_candidates = list(all_numbers - set(p.predict()['top_numbers']))
            if miss_candidates:
                p.update(miss_candidates[0])

        # After enough misses, exploration should be active
        post_miss_pred = p.predict()
        assert post_miss_pred['consecutive_misses'] >= CONSECUTIVE_MISS_THRESHOLD, \
            f"consecutive_misses ({post_miss_pred['consecutive_misses']}) should be >= threshold ({CONSECUTIVE_MISS_THRESHOLD})"
        assert post_miss_pred['exploration_active'] is True, \
            "exploration_active should be True after consecutive misses"

    def test_loss_reactive_resets_on_hit(self):
        """After a hit, consecutive_misses should reset to 0."""
        p = EnsemblePredictor()
        for num in LARGE_SAMPLE:
            p.update(num)

        # Get prediction and feed one of the predicted numbers (a hit)
        pred = p.predict()
        if pred['top_numbers']:
            hit_number = pred['top_numbers'][0]
            p.update(hit_number)
            post_hit_pred = p.predict()
            assert post_hit_pred['consecutive_misses'] == 0, \
                f"consecutive_misses should reset to 0 after a hit, got {post_hit_pred['consecutive_misses']}"

    def test_recent_hit_rate_in_confidence_breakdown(self):
        """Confidence breakdown must include the recent_hit_rate factor."""
        p = EnsemblePredictor()
        for num in LARGE_SAMPLE:
            p.update(num)
        pred = p.predict()

        bd = pred['confidence_breakdown']
        assert 'factors' in bd
        assert 'recent_hit_rate' in bd['factors'], \
            "confidence breakdown missing recent_hit_rate factor"
        rhr = bd['factors']['recent_hit_rate']
        assert 'score' in rhr
        assert 'weight' in rhr
        assert 'contribution' in rhr
        assert isinstance(rhr['score'], float)


# ═══════════════════════════════════════════════════════════════════════
#  7. MODEL-SPECIFIC INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestModelIntegration:

    def test_frequency_analyzer_integration(self):
        """Frequency analyzer produces valid probabilities."""
        fa = FrequencyAnalyzer()
        for num in SAMPLE_SPINS:
            fa.update(num)

        probs = fa.get_number_probabilities()
        assert len(probs) == TOTAL_NUMBERS
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs)

        # Time-weighted probs
        recent = fa.get_recent_probabilities()
        assert len(recent) == TOTAL_NUMBERS
        assert abs(sum(recent) - 1.0) < 1e-6

    def test_markov_chain_integration(self):
        """Markov chain produces valid probabilities."""
        mc = MarkovChain()
        for num in SAMPLE_SPINS:
            mc.update(num)

        probs = mc.get_probabilities()
        assert len(probs) == TOTAL_NUMBERS
        assert all(p >= 0 for p in probs)

        strength = mc.get_transition_strength()
        assert 0 <= strength <= 100

    def test_pattern_detector_integration(self):
        """Pattern detector produces valid analysis."""
        pd = PatternDetector()
        for num in LARGE_SAMPLE:
            pd.update(num)

        probs = pd.get_number_probabilities()
        assert len(probs) == TOTAL_NUMBERS
        assert abs(sum(probs) - 1.0) < 1e-6

        strength = pd.get_pattern_strength()
        assert 0 <= strength <= 100

        summary = pd.get_summary()
        assert 'statistical_tests' in summary
        assert 'runs_test' in summary['statistical_tests']

    def test_lstm_predictor_integration(self):
        """LSTM predictor produces valid probabilities."""
        lstm = LSTMPredictor()
        for num in SAMPLE_SPINS:
            lstm.update(num)

        probs = lstm.predict()
        assert len(probs) == TOTAL_NUMBERS
        assert all(p >= 0 for p in probs)

    def test_feature_engine_integration(self):
        """Feature engine produces consistent features."""
        fe = FeatureEngine()
        context = SAMPLE_SPINS[:20]

        # extract_sequence(spin_history, seq_start, seq_length)
        seq = fe.extract_sequence(context, seq_start=10, seq_length=10)
        assert seq.shape[0] == 10
        assert seq.shape[1] == fe.FEATURE_DIM
        assert np.all(np.isfinite(seq))

    def test_confidence_engine_integration(self):
        """Confidence engine produces valid scores with calibration."""
        ce = ConfidenceEngine()

        # Generate some prediction records
        for i in range(20):
            record = {
                'numbers': [i % 37, (i + 1) % 37, (i + 2) % 37],
                'color': 'red',
                'dozen': '1st',
                'actual_color': 'red' if i % 2 == 0 else 'black',
                'actual_dozen': '1st' if i % 3 == 0 else '2nd',
                '_raw_confidence': 50.0,
            }
            ce.record_prediction(record, i % 37)

        dists = [np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS) for _ in range(4)]
        breakdown = ce.get_breakdown(dists, 500, 50.0, SAMPLE_SPINS)

        assert isinstance(breakdown['overall'], float)
        assert isinstance(breakdown['calibrated'], bool)
        assert 0 <= breakdown['overall'] <= 100

    def test_ensemble_weights_sum_to_one(self):
        """Adaptive weights must sum to approximately 1."""
        p = EnsemblePredictor()
        for num in LARGE_SAMPLE:
            p.update(num)
            if len(p.spin_history) % 5 == 0:
                p.predict()

        weights = p.performance_tracker.get_adaptive_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"

    def test_regime_detector_with_real_data(self):
        """Regime detector doesn't crash with real data."""
        rd = RegimeDetector()
        result = rd.check(LARGE_SAMPLE)
        assert 'changed' in result
        assert 'kl_divergence' in result
        assert isinstance(result['changed'], bool)


# ═══════════════════════════════════════════════════════════════════════
#  8. CONCURRENT / MULTI-OPERATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestConcurrentOperations:

    def test_predict_during_update(self):
        """Predict interleaved with updates should not crash."""
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS:
            p.update(num)
            if len(p.spin_history) >= 3 and len(p.spin_history) % 5 == 0:
                pred = p.predict()
                assert 'top_numbers' in pred

    def test_rapid_undo_redo(self):
        """Rapid undo and re-add should maintain consistency."""
        p = EnsemblePredictor()
        for num in SAMPLE_SPINS[:10]:
            p.update(num)

        # Undo 5, re-add 5 different
        for _ in range(5):
            p.undo_last()
        assert len(p.spin_history) == 5

        for num in [1, 2, 3, 4, 5]:
            p.update(num)
        assert len(p.spin_history) == 10

        pred = p.predict()
        assert pred['total_spins'] == 10

    def test_save_during_predictions(self):
        """Save state while predictions are happening."""
        import app.ml.ensemble as ens_mod
        original_path = ens_mod.MODEL_STATE_PATH

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        ens_mod.MODEL_STATE_PATH = temp_path

        try:
            p = EnsemblePredictor()
            for num in SAMPLE_SPINS:
                p.update(num)
                if len(p.spin_history) % 10 == 0:
                    p.predict()
                    p.save_state()

            # Final save and verify
            p.save_state()
            p2 = EnsemblePredictor()
            loaded = p2.load_state()
            assert loaded
            assert len(p2.spin_history) == len(SAMPLE_SPINS)
        finally:
            ens_mod.MODEL_STATE_PATH = original_path
            if os.path.exists(temp_path):
                os.remove(temp_path)
