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
                    CONFIDENCE_HIGH_THRESHOLD, MIN_BET, BET_INCREMENT,
                    BET_DECREMENT, LOSSES_BEFORE_INCREMENT,
                    WINS_BEFORE_DECREMENT, TOP_PREDICTIONS_COUNT)
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
        """Straight bet on 12 numbers: $1 each = $12 total.
        One number hits: payout $35 + $1 stake back - $12 total = +$24 net profit."""
        bm = BankrollManager()
        initial = bm.bankroll
        # Simulate: $1/number × 12 numbers = $12 total, payout 35:1 = $35
        bm.process_result(bet_amount=12, won=True, payout_amount=35)
        # Net = $35 + $1 (stake) - $12 (total) = $24
        assert bm.bankroll == initial + 24
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

    def test_loss_streak_requires_high_confidence(self):
        """After MAX_CONSECUTIVE_LOSSES, AI requires higher confidence to bet."""
        bm = BankrollManager()
        for _ in range(MAX_CONSECUTIVE_LOSSES):
            bm.process_result(bet_amount=2, won=False)
        # Low confidence (35%) should be blocked after loss streak
        can_bet, reason = bm.should_bet(35, 'BET')
        assert can_bet == False
        assert 'LOSS_STREAK' in reason
        # High confidence (CONFIDENCE_HIGH_THRESHOLD+) should still allow betting
        can_bet_high, reason_high = bm.should_bet(CONFIDENCE_HIGH_THRESHOLD + 5, 'BET')
        assert can_bet_high == True

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
        can_bet, reason = bm.should_bet(60, 'BET')
        assert can_bet == True
        assert reason == 'BET'

    def test_session_target_reached(self):
        bm = BankrollManager()
        # Simulate enough wins to reach target
        bm.bankroll = INITIAL_BANKROLL + SESSION_TARGET + 1
        assert bm.session_target_reached == True
        can_bet, reason = bm.should_bet(80, 'BET_HIGH')
        assert can_bet == False
        assert reason == 'SESSION_TARGET_REACHED'

    def test_stop_loss_hit(self):
        bm = BankrollManager()
        bm.bankroll = STOP_LOSS_THRESHOLD - 1
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

    def test_base_bet_is_1_dollar(self):
        bm = BankrollManager()
        assert bm.current_bet_level == MIN_BET
        assert MIN_BET == 1.0

    def test_calculate_bet_amount_returns_current_level(self):
        bm = BankrollManager()
        bet = bm.calculate_bet_amount(60, 'straight')
        assert bet == MIN_BET

    def test_bet_increment_after_3_losses(self):
        """After LOSSES_BEFORE_INCREMENT successive losses, bet should increase by BET_INCREMENT"""
        bm = BankrollManager()
        assert bm.current_bet_level == MIN_BET
        # Simulate 3 successive losses (loss_streak_for_increment counts across forced waits)
        for _ in range(LOSSES_BEFORE_INCREMENT):
            bm.process_result(bet_amount=12, won=False)
        assert bm.current_bet_level == MIN_BET + BET_INCREMENT

    def test_bet_decrement_after_2_wins(self):
        """After WINS_BEFORE_DECREMENT consecutive wins, bet should decrease by BET_DECREMENT"""
        bm = BankrollManager()
        # First increase the bet level
        bm.current_bet_level = 3.0
        # Simulate 2 consecutive wins
        for _ in range(WINS_BEFORE_DECREMENT):
            bm.process_result(bet_amount=12, won=True, payout_amount=35)
        assert bm.current_bet_level == 3.0 - BET_DECREMENT

    def test_bet_floor_at_1_dollar(self):
        """Bet should never go below MIN_BET"""
        bm = BankrollManager()
        assert bm.current_bet_level == MIN_BET
        # Simulate 2 consecutive wins at minimum level — should stay at MIN_BET
        for _ in range(WINS_BEFORE_DECREMENT):
            bm.process_result(bet_amount=12, won=True, payout_amount=35)
        assert bm.current_bet_level == MIN_BET

    def test_increment_decrement_cycle(self):
        """Full cycle: 3 successive losses → +$1 → 3 more → +$1 → 2 wins → -$1"""
        bm = BankrollManager()
        assert bm.current_bet_level == 1.0
        # 3 successive losses → bet goes to 2.0
        # (Note: forced wait triggers at loss 2, but loss_streak_for_increment
        #  keeps counting across forced waits)
        for _ in range(LOSSES_BEFORE_INCREMENT):
            bm.process_result(bet_amount=12, won=False)
        assert bm.current_bet_level == 2.0
        # 3 more successive losses → bet goes to 3.0
        for _ in range(LOSSES_BEFORE_INCREMENT):
            bm.process_result(bet_amount=24, won=False)
        assert bm.current_bet_level == 3.0
        # 2 consecutive wins → bet goes back to 2.0
        for _ in range(WINS_BEFORE_DECREMENT):
            bm.process_result(bet_amount=36, won=True, payout_amount=35)
        assert bm.current_bet_level == 2.0

    def test_total_bet_is_per_number_times_12(self):
        bm = BankrollManager()
        per_number = bm.calculate_bet_amount()
        total = per_number * TOP_PREDICTIONS_COUNT
        assert total == MIN_BET * 12

    def test_bet_amount_capped_at_5_percent(self):
        bm = BankrollManager()
        bet = bm.calculate_bet_amount(99, 'straight')
        total = bet * TOP_PREDICTIONS_COUNT
        assert total <= bm.bankroll * 0.05 + 0.01  # small rounding tolerance

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
        bm.bankroll = STOP_LOSS_THRESHOLD - 100
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
                         'risk_level', 'drawdown', 'momentum']
        for key in expected_keys:
            assert key in status, f"Missing key: {key}"

    def test_get_advice_structure(self):
        bm = BankrollManager()
        advice = bm.get_advice(ai_confidence=60, ai_mode='BET')
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
        bm.bankroll = 0.5  # Less than base bet AND below stop loss
        can_bet, reason = bm.should_bet(80, 'BET_HIGH')
        assert can_bet == False
        # Stop loss check fires first since bankroll < STOP_LOSS_THRESHOLD
        assert reason in ('STOP_LOSS_HIT', 'INSUFFICIENT_BANKROLL')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
