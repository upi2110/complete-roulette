"""
Bankroll Manager - THE ONE Strategy (1-3-2-6 Sequence Variant).

Money Management Strategy:
- Sequence betting: [1,6,13,15,9] multipliers × $7.50 unit
- On WIN: advance to next sequence step (wraps at end)
- On LOSS: reset to step 0
- Confidence gate: only bet when AI >= 65%
- Stop-loss: $500 loss ($3,500 bankroll threshold)
- No profit cap: let winners run
- Marathon proven: 3.36M strategies, 1.57B sessions, P/R 0.152
"""

import sys
sys.path.insert(0, '.')
from config import (
    INITIAL_BANKROLL, BASE_BET, SESSION_TARGET,
    STOP_LOSS_THRESHOLD, MAX_CONSECUTIVE_LOSSES,
    KELLY_FRACTION, MAX_BET_MULTIPLIER, RECOVERY_BET_FACTOR,
    FORCED_WAIT_SPINS, CONFIDENCE_BET_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD, WIN_PROBABILITIES, PAYOUTS,
    BET_SEQUENCE, MIN_BET, SESSION_TARGET_ENABLED,
    TOP_PREDICTIONS_COUNT, WAIT_AFTER_LOSS_SPINS
)


class BankrollManager:
    def __init__(self, initial_bankroll=INITIAL_BANKROLL):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.base_bet = BASE_BET
        self.session_target = SESSION_TARGET

        # Loss tracking
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_bets_placed = 0
        self.total_skips = 0

        # Sequence betting state (THE ONE strategy)
        self.bet_sequence = list(BET_SEQUENCE)  # [1, 6, 13, 15, 9]
        self.seq_idx = 0                         # Current position in sequence
        self.unit_size = BASE_BET                # $7.50 per unit

        # State machine
        self.forced_wait_remaining = 0
        self.loss_cooldown_remaining = 0   # Wait N spins after every loss
        self.in_recovery_mode = False
        self.in_protection_mode = False

        # Peak/trough tracking
        self.peak_bankroll = initial_bankroll
        self.trough_bankroll = initial_bankroll

        # Momentum tracking (for streak-based adjustments)
        self.recent_results = []  # Last 10 results: True=win, False=loss
        self.hot_streak = False
        self.cold_streak = False

        # History
        self.bankroll_history = [initial_bankroll]
        self.bet_history = []
        self.pnl_history = []

    @property
    def profit_loss(self):
        return round(self.bankroll - self.initial_bankroll, 2)

    @property
    def win_rate(self):
        total = self.total_wins + self.total_losses
        if total == 0:
            return 0.0
        return round(self.total_wins / total * 100, 1)

    @property
    def session_target_reached(self):
        if not SESSION_TARGET_ENABLED:
            return False  # No profit cap — let winners run
        return self.profit_loss >= self.session_target

    @property
    def stop_loss_hit(self):
        return self.bankroll <= STOP_LOSS_THRESHOLD

    @property
    def drawdown(self):
        """Current drawdown from peak."""
        if self.peak_bankroll == 0:
            return 0
        dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100
        return round(max(0, dd), 1)

    @property
    def risk_level(self):
        """Overall risk assessment: low, medium, high, critical."""
        if self.stop_loss_hit:
            return 'critical'
        if self.in_protection_mode or self.drawdown > 15:
            return 'high'
        if self.in_recovery_mode or self.consecutive_losses >= 1:
            return 'medium'
        return 'low'

    def _update_momentum(self, won):
        """Track momentum from recent results."""
        self.recent_results.append(won)
        if len(self.recent_results) > 10:
            self.recent_results.pop(0)

        if len(self.recent_results) >= 3:
            last3 = self.recent_results[-3:]
            self.hot_streak = all(last3)
            self.cold_streak = not any(last3)
        else:
            self.hot_streak = False
            self.cold_streak = False

    def _get_momentum_score(self):
        """Calculate momentum from recent results (-100 to +100)."""
        if not self.recent_results:
            return 0
        wins = sum(1 for r in self.recent_results if r)
        total = len(self.recent_results)
        # Center around 0: all wins = +100, all losses = -100
        return round((wins / total - 0.5) * 200)

    def should_bet(self, ai_confidence, ai_mode):
        """Determine if we should place a bet considering all factors.

        Strategy: After every loss, WAIT 2 spins before betting again.
        User observed that WAIT periods correlate with better outcomes.
        """
        # Session target reached - stop
        if self.session_target_reached:
            return False, 'SESSION_TARGET_REACHED'

        # Stop loss hit - stop
        if self.stop_loss_hit:
            return False, 'STOP_LOSS_HIT'

        # Loss cooldown: wait N spins after each loss
        if self.loss_cooldown_remaining > 0:
            return False, f'LOSS_COOLDOWN ({self.loss_cooldown_remaining} spins remaining)'

        # Protection mode - only bet on very high confidence
        if self.in_protection_mode and ai_confidence < CONFIDENCE_HIGH_THRESHOLD:
            return False, 'PROTECTION_MODE'

        # After MAX_CONSECUTIVE_LOSSES, require higher confidence
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            required = CONFIDENCE_HIGH_THRESHOLD
            if ai_confidence < required:
                return False, f'LOSS_STREAK_CAUTION ({self.consecutive_losses} losses — need {required}% confidence)'

        # AI says WAIT
        if ai_mode == 'WAIT':
            return False, 'LOW_CONFIDENCE'

        # Can't afford minimum bet
        if self.bankroll < self.base_bet:
            return False, 'INSUFFICIENT_BANKROLL'

        return True, 'BET'

    def calculate_bet_amount(self, confidence=0, bet_type='straight', num_predictions=None):
        """Calculate bet amount per number using sequence strategy (THE ONE).

        Returns unit_size × sequence[seq_idx] as the per-number bet.
        The sequence [1,6,13,15,9] applied to $7.50 unit gives:
          Step 0: $7.50, Step 1: $45, Step 2: $97.50, Step 3: $112.50, Step 4: $67.50

        Args:
            num_predictions: actual number of predictions (dynamic). Falls back to TOP_PREDICTIONS_COUNT.
        """
        n_preds = num_predictions or TOP_PREDICTIONS_COUNT
        self._last_num_predictions = n_preds  # Store for process_result

        # Sequence-based bet: unit_size × current multiplier
        multiplier = self.bet_sequence[self.seq_idx % len(self.bet_sequence)]
        per_number = self.unit_size * multiplier

        # Enforce max: BASE_BET × MAX_BET_MULTIPLIER ($7.50 × 15 = $112.50)
        max_bet = BASE_BET * MAX_BET_MULTIPLIER
        per_number = min(per_number, max_bet)

        # Floor
        per_number = max(MIN_BET, per_number)

        # Safety: can't bet more than bankroll allows
        per_number = min(per_number, self.bankroll / max(1, n_preds))

        return round(per_number, 2)

    def get_recommended_bet_type(self, confidence):
        """Recommend the best bet type based on confidence level.
        Higher confidence → riskier bets with bigger payouts.
        Lower confidence → safer even-money bets."""
        if confidence >= 85:
            return ['straight', 'dozen', 'red_black']
        elif confidence >= 80:
            return ['dozen', 'column', 'red_black']
        elif confidence >= 70:
            return ['dozen', 'red_black', 'high_low']
        elif confidence >= 65:
            return ['red_black', 'high_low', 'odd_even']
        else:
            return ['red_black', 'odd_even']

    def get_advice(self, ai_confidence=0, ai_mode='WAIT', prediction=None):
        """Generate comprehensive money management advice.

        Returns a detailed advice dict with:
        - action: 'BET', 'WAIT', 'STOP'
        - reason: human-readable reason for the advice
        - bet_amount: recommended bet size (0 if waiting)
        - bet_types: recommended bet types in order of preference
        - risk_level: current risk assessment
        - strategy: current strategy description
        - warnings: list of active warnings
        - tips: contextual tips based on current state
        """
        warnings = []
        tips = []

        # Determine primary action
        should_bet_result, bet_reason = self.should_bet(ai_confidence, ai_mode)

        # Build detailed action
        if self.session_target_reached:
            action = 'STOP'
            action_label = 'TARGET REACHED'
            reason = f'Session target of +${self.session_target} reached! Lock in your profit of +${self.profit_loss}.'
            strategy = 'Take Profit'
            tips.append('Consider ending this session and starting fresh.')
            tips.append(f'You are up ${self.profit_loss} - well done!')

        elif self.stop_loss_hit:
            action = 'STOP'
            action_label = 'STOP LOSS'
            reason = f'Bankroll dropped to ${self.bankroll:.2f}, below ${STOP_LOSS_THRESHOLD} safety limit. Stop betting to preserve capital.'
            strategy = 'Capital Preservation'
            warnings.append('Stop-loss threshold reached. End session recommended.')
            tips.append('The house edge makes recovery from large losses very difficult.')

        elif self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES and ai_confidence < CONFIDENCE_HIGH_THRESHOLD:
            action = 'WAIT'
            action_label = f'CAUTION ({self.consecutive_losses} losses)'
            reason = f'{self.consecutive_losses} consecutive losses. AI waiting for high confidence ({CONFIDENCE_HIGH_THRESHOLD}%+) before next bet. Current: {ai_confidence}%.'
            strategy = 'Loss Recovery'
            tips.append('After a losing streak, the AI requires stronger signals before risking more capital.')
            deficit = CONFIDENCE_HIGH_THRESHOLD - ai_confidence
            tips.append(f'Need {deficit:.0f}% more confidence. Keep entering spins.')

        elif self.in_protection_mode and ai_confidence < CONFIDENCE_HIGH_THRESHOLD:
            action = 'WAIT'
            action_label = 'PROTECTION MODE'
            amt_below = STOP_LOSS_THRESHOLD - self.bankroll
            reason = f'Protection mode active (bankroll near stop-loss). Only betting on high confidence ({CONFIDENCE_HIGH_THRESHOLD}%+). Current AI: {ai_confidence}%.'
            strategy = 'Defensive'
            warnings.append(f'Bankroll is ${abs(amt_below):.0f} above stop-loss. Playing very cautiously.')
            tips.append(f'AI confidence needs to reach {CONFIDENCE_HIGH_THRESHOLD}% before we bet in protection mode.')

        elif self.loss_cooldown_remaining > 0:
            action = 'WAIT'
            action_label = f'COOLDOWN ({self.loss_cooldown_remaining} spins)'
            reason = f'Cooling down after loss. Wait {self.loss_cooldown_remaining} more spin{"s" if self.loss_cooldown_remaining > 1 else ""} before betting again.'
            strategy = 'Loss Cooldown'
            tips.append('Waiting after a loss helps avoid chasing losses.')
            tips.append('Keep entering spins — the AI is still learning from each result.')

        elif ai_mode == 'WAIT' or ai_confidence < CONFIDENCE_BET_THRESHOLD:
            action = 'WAIT'
            action_label = 'WAIT'
            deficit = CONFIDENCE_BET_THRESHOLD - ai_confidence
            reason = f'AI confidence is {ai_confidence}%, below {CONFIDENCE_BET_THRESHOLD}% threshold. Wait for a stronger signal.'
            strategy = 'Selective Betting'
            tips.append(f'Need {deficit:.0f}% more confidence to trigger a bet. Keep entering spins to improve AI accuracy.')
            if ai_confidence >= 55:
                tips.append('Getting close to the threshold. The AI is learning patterns.')

        elif should_bet_result:
            action = 'BET'
            # Calculate amounts for different bet types
            recommended_types = self.get_recommended_bet_type(ai_confidence)
            primary_type = recommended_types[0]
            bet_amount = self.calculate_bet_amount(ai_confidence, primary_type)

            multiplier = self.bet_sequence[self.seq_idx % len(self.bet_sequence)]
            if ai_confidence >= CONFIDENCE_HIGH_THRESHOLD:
                action_label = 'BET (HIGH CONFIDENCE)'
                reason = f'AI confidence {ai_confidence}% meets {CONFIDENCE_BET_THRESHOLD}% gate. Sequence step {self.seq_idx + 1}/{len(self.bet_sequence)} (x{multiplier}).'
                strategy = f'THE ONE - Step {self.seq_idx + 1}'
                if self.hot_streak:
                    tips.append('Hot streak detected! Sequence advancing — let it ride.')
            else:
                action_label = 'BET'
                reason = f'AI confidence {ai_confidence}% meets {CONFIDENCE_BET_THRESHOLD}% gate. Sequence step {self.seq_idx + 1}/{len(self.bet_sequence)} (x{multiplier}).'
                strategy = f'THE ONE - Step {self.seq_idx + 1}'

            if self.in_recovery_mode:
                strategy = 'Recovery'
                reason += ' Recovery mode: using reduced bet sizes.'
                tips.append('In recovery mode - betting conservatively until we rebuild.')
        else:
            action = 'WAIT'
            action_label = 'WAIT'
            reason = bet_reason
            strategy = 'Selective'

        # Calculate bet sizing breakdown — straight bets only
        bet_sizing = {}
        if action == 'BET':
            amt = self.calculate_bet_amount(ai_confidence, 'straight')
            bet_sizing['straight'] = amt

        # Only straight bets allowed
        allowed_types = ['straight'] if action == 'BET' else []

        # Build warning list
        if self.consecutive_losses >= 1 and self.consecutive_losses < MAX_CONSECUTIVE_LOSSES:
            warnings.append(f'{self.consecutive_losses} consecutive loss(es). AI monitoring confidence closely.')
        if self.in_recovery_mode:
            warnings.append('Recovery mode active. Bet sizes reduced by 25%.')
        if self.drawdown > 10:
            warnings.append(f'Drawdown: {self.drawdown}% from peak. Be cautious.')
        if self.cold_streak:
            warnings.append('Cold streak detected (3+ losses in last results). Consider reducing exposure.')

        # Momentum-based tips
        momentum = self._get_momentum_score()
        if momentum > 50:
            tips.append('Strong positive momentum. This is a good time to be in the game.')
        elif momentum < -50:
            tips.append('Negative momentum. The AI will wait for better opportunities.')

        # Distance to target (only when target is enabled)
        if SESSION_TARGET_ENABLED:
            remaining = self.session_target - self.profit_loss
            if remaining > 0 and action == 'BET':
                bets_to_target = remaining / self.base_bet if self.base_bet > 0 else 0
                tips.append(f'${remaining:.2f} away from session target. ~{int(bets_to_target)} base bets remaining.')

        # Build primary bet amount
        primary_bet_amount = 0
        if action == 'BET':
            primary_type = allowed_types[0] if allowed_types else 'red_black'
            primary_bet_amount = self.calculate_bet_amount(ai_confidence, primary_type)

        return {
            'action': action,
            'action_label': action_label,
            'reason': reason,
            'strategy': strategy,
            'risk_level': self.risk_level,
            'bet_amount': primary_bet_amount,
            'bet_sizing': bet_sizing,
            'allowed_bet_types': allowed_types,
            'warnings': warnings,
            'tips': tips,
            'momentum': momentum,
            'drawdown': self.drawdown,
            'hot_streak': self.hot_streak,
            'cold_streak': self.cold_streak,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'distance_to_target': round(self.session_target - self.profit_loss, 2),
            'distance_to_stop_loss': round(self.bankroll - STOP_LOSS_THRESHOLD, 2),
            'kelly_fraction': KELLY_FRACTION,
            'base_bet': self.base_bet,
            'max_bet': round(self.base_bet * max(self.bet_sequence), 2),
            'bankroll_pct_at_risk': round(primary_bet_amount / self.bankroll * 100, 2) if self.bankroll > 0 and primary_bet_amount > 0 else 0
        }

    def process_result(self, bet_amount, won, payout_amount=0):
        """Process bet result and update state machine with incremental betting.

        For straight bets on N numbers ($1 each = $N total):
        - WIN:  One number hits → payout $35 + $1 stake back = $36 return.
                Net profit = $36 - $N.
        - LOSS: No number hits → lose $N.

        Args:
            bet_amount: Total amount wagered across all numbers (e.g. $14)
            won: Whether any predicted number hit
            payout_amount: Raw payout on winning number (e.g. 35 for 35:1)
        """
        self.total_bets_placed += 1

        if won:
            # Net profit = winnings (35:1) + stake back - total cost of all bets
            # e.g. $35 + $1 - $14 = $22 if betting $1 on each of 14 numbers
            per_number_stake = bet_amount / max(1, self._last_num_predictions) if hasattr(self, '_last_num_predictions') else bet_amount / max(1, TOP_PREDICTIONS_COUNT)
            net = payout_amount + per_number_stake - bet_amount
            self.bankroll += net
            self.total_wins += 1
            self.consecutive_losses = 0
            self.consecutive_wins += 1
            self.in_recovery_mode = False
            self.loss_cooldown_remaining = 0   # Clear cooldown on win

            # Sequence strategy: advance on win, wrap at end
            self.seq_idx = (self.seq_idx + 1) % len(self.bet_sequence)

            # Exit protection mode after 2 consecutive wins
            if self.in_protection_mode and self.total_wins >= 2:
                recent_results = [h['won'] for h in self.bet_history[-2:]]
                if all(recent_results):
                    self.in_protection_mode = False
        else:
            self.bankroll -= bet_amount
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

            # Sequence strategy: reset to step 0 on loss
            self.seq_idx = 0

            # After MAX_CONSECUTIVE_LOSSES, enter recovery mode (higher confidence required)
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.in_recovery_mode = True

            # Enter protection mode
            if self.bankroll < STOP_LOSS_THRESHOLD:
                self.in_protection_mode = True

            # Activate loss cooldown: wait N spins before betting again
            self.loss_cooldown_remaining = WAIT_AFTER_LOSS_SPINS

            net = -bet_amount

        # Update momentum
        self._update_momentum(won)

        # Update peak/trough
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        self.trough_bankroll = min(self.trough_bankroll, self.bankroll)

        # Record history
        self.bankroll_history.append(round(self.bankroll, 2))
        self.bet_history.append({
            'bet': bet_amount,
            'won': won,
            'net': round(net, 2),
            'bankroll_after': round(self.bankroll, 2)
        })
        self.pnl_history.append(round(self.profit_loss, 2))

    def tick_wait(self):
        """Call each spin when in forced wait to count down.
        Also ticks down loss cooldown (wait-after-loss timer)."""
        if self.forced_wait_remaining > 0:
            self.forced_wait_remaining -= 1
        if self.loss_cooldown_remaining > 0:
            self.loss_cooldown_remaining -= 1

    def get_status(self):
        """Full bankroll status for the dashboard."""
        return {
            'bankroll': round(self.bankroll, 2),
            'initial_bankroll': self.initial_bankroll,
            'profit_loss': self.profit_loss,
            'session_target': self.session_target,
            'target_progress': round(min(100, max(0, self.profit_loss / self.session_target * 100)), 1) if SESSION_TARGET_ENABLED else 0,
            'win_rate': self.win_rate,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'total_bets': self.total_bets_placed,
            'total_skips': self.total_skips,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'forced_wait_remaining': self.forced_wait_remaining,
            'loss_cooldown_remaining': self.loss_cooldown_remaining,
            'in_recovery_mode': self.in_recovery_mode,
            'in_protection_mode': self.in_protection_mode,
            'session_target_reached': self.session_target_reached,
            'stop_loss_hit': self.stop_loss_hit,
            'base_bet': self.base_bet,
            'current_bet_level': round(self.unit_size * self.bet_sequence[self.seq_idx % len(self.bet_sequence)], 2),
            'seq_position': self.seq_idx,
            'seq_step_label': f"Step {self.seq_idx + 1}/{len(self.bet_sequence)}",
            'seq_multiplier': self.bet_sequence[self.seq_idx % len(self.bet_sequence)],
            'risk_level': self.risk_level,
            'drawdown': self.drawdown,
            'peak_bankroll': round(self.peak_bankroll, 2),
            'momentum': self._get_momentum_score(),
            'hot_streak': self.hot_streak,
            'cold_streak': self.cold_streak,
            'bankroll_history': self.bankroll_history[-50:],
            'pnl_history': self.pnl_history[-50:]
        }

    def reset_session(self):
        """Reset for a new session but keep bankroll."""
        current_bankroll = self.bankroll
        self.__init__(initial_bankroll=current_bankroll)
        self.initial_bankroll = current_bankroll

    def undo_last_bet(self):
        """Reverse the last process_result() call — used when user undoes a spin.

        Restores bankroll, win/loss counters, streak counters, bet level,
        momentum, peak/trough, and history to the state before the last bet.
        Returns the removed bet_history entry, or None if no bets to undo.
        """
        if not self.bet_history:
            return None

        last_bet = self.bet_history.pop()
        self.pnl_history.pop() if self.pnl_history else None
        self.bankroll_history.pop() if len(self.bankroll_history) > 1 else None

        # Reverse bankroll change
        net = last_bet['net']
        self.bankroll = round(self.bankroll - net, 2)

        # Reverse counters
        self.total_bets_placed = max(0, self.total_bets_placed - 1)

        if last_bet['won']:
            self.total_wins = max(0, self.total_wins - 1)
        else:
            self.total_losses = max(0, self.total_losses - 1)

        # Rebuild streak counters from remaining bet_history
        # This is the safest approach — recalculate from the tail of history
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        for entry in reversed(self.bet_history):
            if entry['won']:
                self.consecutive_wins += 1
                if self.consecutive_losses > 0:
                    break  # Stop once we hit the opposite result
            else:
                self.consecutive_losses += 1
                if self.consecutive_wins > 0:
                    break
            # Both are 0 still — first entry sets the direction
            if self.consecutive_wins == 0 and self.consecutive_losses == 0:
                if entry['won']:
                    self.consecutive_wins = 1
                else:
                    self.consecutive_losses = 1

        # Recalculate max_consecutive_losses from full history
        max_cl = 0
        current_cl = 0
        for entry in self.bet_history:
            if not entry['won']:
                current_cl += 1
                max_cl = max(max_cl, current_cl)
            else:
                current_cl = 0
        self.max_consecutive_losses = max_cl

        # Rebuild sequence position from bet history
        # Rule: on win advance (wrapping), on loss reset to 0
        self.seq_idx = 0
        for entry in self.bet_history:
            if entry['won']:
                self.seq_idx = (self.seq_idx + 1) % len(self.bet_sequence)
            else:
                self.seq_idx = 0

        # Rebuild momentum (recent_results)
        self.recent_results = []
        for entry in self.bet_history[-10:]:
            self.recent_results.append(entry['won'])
        if len(self.recent_results) >= 3:
            last3 = self.recent_results[-3:]
            self.hot_streak = all(last3)
            self.cold_streak = not any(last3)
        else:
            self.hot_streak = False
            self.cold_streak = False

        # Recalculate peak/trough from bankroll_history
        if self.bankroll_history:
            self.peak_bankroll = max(self.bankroll_history)
            self.trough_bankroll = min(self.bankroll_history)
        else:
            self.peak_bankroll = self.bankroll
            self.trough_bankroll = self.bankroll

        # Recalculate mode flags
        self.in_protection_mode = self.bankroll < STOP_LOSS_THRESHOLD
        self.in_recovery_mode = self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES

        # Recalculate loss cooldown: if last bet was a loss, set cooldown
        if self.bet_history and not self.bet_history[-1]['won']:
            self.loss_cooldown_remaining = WAIT_AFTER_LOSS_SPINS
        else:
            self.loss_cooldown_remaining = 0

        return last_bet

    def full_reset(self):
        """Full reset to initial state."""
        self.__init__(initial_bankroll=INITIAL_BANKROLL)
