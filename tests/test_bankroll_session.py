"""
Unit Tests for BankrollManager and SessionManager
"""
import pytest
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (INITIAL_BANKROLL, BASE_BET, SESSION_TARGET,
                    STOP_LOSS_THRESHOLD, MAX_CONSECUTIVE_LOSSES,
                    FORCED_WAIT_SPINS, CONFIDENCE_BET_THRESHOLD,
                    CONFIDENCE_HIGH_THRESHOLD, MIN_BET, BET_SEQUENCE,
                    SESSION_TARGET_ENABLED, TOP_PREDICTIONS_COUNT)
from app.money.bankroll_manager import BankrollManager


# ═══════════════════════════════════════════════════════════════
# BankrollManager Tests
# ═══════════════════════════════════════════════════════════════

class TestBankrollManager:
    def test_init_defaults(self):
        bm = BankrollManager()
        assert bm.bankroll == INITIAL_BANKROLL
        assert bm.base_bet == BASE_BET
        assert bm.consecutive_losses == 0
        assert bm.total_wins == 0
        assert bm.total_losses == 0
        assert bm.in_recovery_mode == False
        assert bm.in_protection_mode == False

    def test_init_custom_bankroll(self):
        bm = BankrollManager(initial_bankroll=5000)
        assert bm.bankroll == 5000

    def test_profit_loss_initial(self):
        bm = BankrollManager()
        assert bm.profit_loss == 0

    def test_win_rate_no_bets(self):
        bm = BankrollManager()
        assert bm.win_rate == 0.0

    def test_process_win(self):
        """Straight bet on N numbers: $1 each = $N total.
        One number hits: payout $35 + $1 stake back - $N total = net profit."""
        bm = BankrollManager()
        initial = bm.bankroll
        n = TOP_PREDICTIONS_COUNT  # 14 numbers
        # Simulate: $1/number × N numbers = $N total, payout 35:1 = $35
        bm.process_result(bet_amount=n, won=True, payout_amount=35)
        # Net = $35 + $1 (stake) - $N (total) = 36 - N
        assert bm.bankroll == initial + (36 - n)
        assert bm.total_wins == 1
        assert bm.consecutive_losses == 0
        assert bm.consecutive_wins == 1

    def test_process_loss(self):
        bm = BankrollManager()
        initial = bm.bankroll
        bm.process_result(bet_amount=2, won=False)
        assert bm.bankroll == initial - 2
        assert bm.total_losses == 1
        assert bm.consecutive_losses == 1
        assert bm.consecutive_wins == 0

    def test_consecutive_losses_trigger_recovery_mode(self):
        bm = BankrollManager()
        for _ in range(MAX_CONSECUTIVE_LOSSES):
            bm.process_result(bet_amount=2, won=False)
        assert bm.in_recovery_mode == True

    def test_loss_streak_bets_with_high_confidence(self):
        """After losses, still bets if confidence is high enough."""
        bm = BankrollManager()
        for _ in range(5):
            bm.process_result(bet_amount=2, won=False)
        # With conf >= 65, should still bet (no cooldown, no forced wait)
        can_bet, reason = bm.should_bet(70, 'BET_HIGH')
        assert can_bet == True
        assert reason == 'BET'

    def test_tick_wait_decrements(self):
        bm = BankrollManager()
        bm.forced_wait_remaining = 3
        bm.tick_wait()
        assert bm.forced_wait_remaining == 2
        bm.tick_wait()
        assert bm.forced_wait_remaining == 1
        bm.tick_wait()
        assert bm.forced_wait_remaining == 0

    def test_should_bet_low_confidence(self):
        bm = BankrollManager()
        can_bet, reason = bm.should_bet(20, 'WAIT')
        assert can_bet == False
        assert reason == 'LOW_CONFIDENCE'

    def test_should_bet_sufficient_confidence(self):
        bm = BankrollManager()
        can_bet, reason = bm.should_bet(70, 'BET_HIGH')
        assert can_bet == True
        assert reason == 'BET'

    def test_session_target_disabled(self):
        """With SESSION_TARGET_ENABLED=False, target is never reached (no profit cap)."""
        bm = BankrollManager()
        bm.bankroll = INITIAL_BANKROLL + SESSION_TARGET + 1
        assert bm.session_target_reached == False  # Always False when disabled
        can_bet, reason = bm.should_bet(70, 'BET_HIGH')
        assert can_bet == True
        assert reason == 'BET'

    def test_stop_loss_hit(self):
        """Stop-loss at $3500 = $500 loss from $4000 bankroll."""
        bm = BankrollManager()
        bm.bankroll = STOP_LOSS_THRESHOLD - 1  # Below $3500
        assert bm.stop_loss_hit == True
        can_bet, reason = bm.should_bet(80, 'BET_HIGH')
        assert can_bet == False
        assert reason == 'STOP_LOSS_HIT'

    def test_protection_mode_activation(self):
        bm = BankrollManager()
        bm.bankroll = STOP_LOSS_THRESHOLD + 5
        bm.process_result(bet_amount=10, won=False)
        # Bankroll should now be below stop loss
        if bm.bankroll < STOP_LOSS_THRESHOLD:
            assert bm.in_protection_mode == True

    def test_sequence_starts_at_step_0(self):
        """Initial sequence position is 0, bet = BASE_BET × 1."""
        bm = BankrollManager()
        assert bm.seq_idx == 0
        assert bm.bet_sequence == list(BET_SEQUENCE)
        assert bm.unit_size == BASE_BET

    def test_calculate_bet_amount_step_0(self):
        """Step 0: bet = $7.50 × 1 = $7.50 per number."""
        bm = BankrollManager()
        bet = bm.calculate_bet_amount(70, 'straight')
        assert bet == BASE_BET * BET_SEQUENCE[0]  # $7.50 × 1

    def test_sequence_advance_on_win(self):
        """After a win, sequence advances to next step."""
        bm = BankrollManager()
        assert bm.seq_idx == 0
        bm.process_result(bet_amount=TOP_PREDICTIONS_COUNT * BASE_BET, won=True, payout_amount=35 * BASE_BET)
        assert bm.seq_idx == 1  # Advanced to step 1 (x6)

    def test_sequence_reset_on_loss(self):
        """After a loss, sequence resets to step 0."""
        bm = BankrollManager()
        # First win to advance
        bm.process_result(bet_amount=TOP_PREDICTIONS_COUNT * BASE_BET, won=True, payout_amount=35 * BASE_BET)
        assert bm.seq_idx == 1
        # Then loss to reset
        bm.process_result(bet_amount=TOP_PREDICTIONS_COUNT * 45, won=False)
        assert bm.seq_idx == 0

    def test_bet_amount_at_each_step(self):
        """Verify dollar amounts: $7.50, $45, $97.50, $112.50, $67.50."""
        bm = BankrollManager()
        expected = [7.50, 45.0, 97.50, 112.50, 67.50]
        for i, exp in enumerate(expected):
            bm.seq_idx = i
            bet = bm.calculate_bet_amount(70, 'straight', 14)
            assert bet == exp, f"Step {i}: expected ${exp}, got ${bet}"

    def test_sequence_wraps_after_full_cycle(self):
        """After 5 consecutive wins, sequence wraps back to step 0."""
        bm = BankrollManager()
        for i in range(5):
            bm.process_result(bet_amount=10, won=True, payout_amount=35)
        assert bm.seq_idx == 0  # Wrapped back

    def test_sequence_full_cycle(self):
        """Win-win-loss pattern: advance-advance-reset."""
        bm = BankrollManager()
        bm.process_result(10, True, 35)   # idx 0→1
        bm.process_result(10, True, 35)   # idx 1→2
        assert bm.seq_idx == 2
        bm.process_result(10, False)       # idx→0
        assert bm.seq_idx == 0
        bm.process_result(10, True, 35)   # idx 0→1
        assert bm.seq_idx == 1

    def test_total_bet_is_per_number_times_count(self):
        bm = BankrollManager()
        per_number = bm.calculate_bet_amount(70, 'straight', TOP_PREDICTIONS_COUNT)
        total = per_number * TOP_PREDICTIONS_COUNT
        assert total == BASE_BET * BET_SEQUENCE[0] * TOP_PREDICTIONS_COUNT

    def test_bet_floor_at_min_bet(self):
        """Bet should never go below MIN_BET."""
        bm = BankrollManager()
        bet = bm.calculate_bet_amount(70, 'straight', 14)
        assert bet >= MIN_BET

    def test_undo_rebuilds_sequence_position(self):
        """Undo should correctly rebuild seq_idx from history."""
        bm = BankrollManager()
        bm.process_result(10, True, 35)   # idx→1
        bm.process_result(10, True, 35)   # idx→2
        bm.process_result(10, False)       # idx→0
        bm.process_result(10, True, 35)   # idx→1
        assert bm.seq_idx == 1
        bm.undo_last_bet()                 # Remove last win, should be at idx=0
        assert bm.seq_idx == 0
        bm.undo_last_bet()                 # Remove the loss, should be at idx=2
        assert bm.seq_idx == 2

    def test_drawdown_calculation(self):
        bm = BankrollManager()
        bm.peak_bankroll = 4100
        bm.bankroll = 4000
        dd = bm.drawdown
        expected = (4100 - 4000) / 4100 * 100
        assert abs(dd - expected) < 0.1

    def test_risk_level_low(self):
        bm = BankrollManager()
        assert bm.risk_level == 'low'

    def test_risk_level_medium(self):
        bm = BankrollManager()
        bm.consecutive_losses = 1
        assert bm.risk_level == 'medium'

    def test_risk_level_critical(self):
        bm = BankrollManager()
        bm.bankroll = STOP_LOSS_THRESHOLD - 100  # Below $3500
        assert bm.risk_level == 'critical'

    def test_momentum_tracking(self):
        bm = BankrollManager()
        for _ in range(3):
            bm.process_result(2, True, 2)
        assert bm.hot_streak == True
        assert bm.cold_streak == False

    def test_cold_streak(self):
        bm = BankrollManager()
        for _ in range(3):
            bm.process_result(2, False)
        assert bm.cold_streak == True
        assert bm.hot_streak == False

    def test_get_status_structure(self):
        bm = BankrollManager()
        status = bm.get_status()
        expected_keys = ['bankroll', 'initial_bankroll', 'profit_loss',
                         'session_target', 'target_progress', 'win_rate',
                         'total_wins', 'total_losses', 'total_bets',
                         'consecutive_losses', 'forced_wait_remaining',
                         'in_recovery_mode', 'in_protection_mode',
                         'risk_level', 'drawdown', 'momentum',
                         'seq_position', 'seq_step_label', 'seq_multiplier']
        for key in expected_keys:
            assert key in status, f"Missing key: {key}"

    def test_get_advice_structure(self):
        bm = BankrollManager()
        advice = bm.get_advice(ai_confidence=70, ai_mode='BET_HIGH')
        assert 'action' in advice
        assert 'reason' in advice
        assert 'strategy' in advice
        assert 'risk_level' in advice
        assert 'warnings' in advice
        assert 'tips' in advice
        assert 'momentum' in advice
        assert 'distance_to_target' in advice

    def test_get_advice_bet_action(self):
        bm = BankrollManager()
        advice = bm.get_advice(ai_confidence=70, ai_mode='BET_HIGH')
        assert advice['action'] == 'BET'
        assert advice['bet_amount'] > 0

    def test_get_advice_wait_action(self):
        bm = BankrollManager()
        advice = bm.get_advice(ai_confidence=20, ai_mode='WAIT')
        assert advice['action'] == 'WAIT'

    def test_full_reset(self):
        bm = BankrollManager()
        bm.process_result(2, True, 2)
        bm.process_result(2, False)
        bm.full_reset()
        assert bm.bankroll == INITIAL_BANKROLL
        assert bm.total_wins == 0
        assert bm.total_losses == 0

    def test_bankroll_history_tracking(self):
        bm = BankrollManager()
        bm.process_result(2, True, 2)
        bm.process_result(2, False)
        assert len(bm.bankroll_history) == 3  # Initial + 2 results

    def test_win_rate_calculation(self):
        bm = BankrollManager()
        bm.process_result(2, True, 2)
        bm.process_result(2, True, 2)
        bm.process_result(2, False)
        assert bm.win_rate == pytest.approx(66.7, abs=0.1)

    def test_recommended_bet_types_by_confidence(self):
        bm = BankrollManager()
        types_low = bm.get_recommended_bet_type(50)
        types_high = bm.get_recommended_bet_type(90)
        assert 'red_black' in types_low
        assert 'straight' in types_high

    def test_insufficient_bankroll(self):
        bm = BankrollManager()
        bm.bankroll = 0.5  # Less than base bet AND below stop loss ($3500)
        can_bet, reason = bm.should_bet(80, 'BET_HIGH')
        assert can_bet == False
        # Stop loss check fires first since bankroll < STOP_LOSS_THRESHOLD ($3500)
        assert reason == 'STOP_LOSS_HIT'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
