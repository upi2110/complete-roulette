#!/usr/bin/env python3
"""
Session Optimizer — Find the configuration that reliably produces +$100/session.

Goal: $100 profit from $4,000 bankroll per session.
Approach:
  1. Test multiple model configs (from backtest winners)
  2. Test multiple bet-sizing strategies (flat, progressive, Kelly-inspired)
  3. Test multiple session lengths (100, 200, 300, 500 spins)
  4. Simulate many sessions from real data to measure reliability
  5. Find the config that most consistently hits $100+ profit

Uses ONLY real userdata (2,528 spins), walk-forward validation.
"""

import sys
import os
import json
import numpy as np
from collections import Counter, deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import WHEEL_ORDER, NUMBER_TO_POSITION, TOTAL_NUMBERS


# ─── Load real data ─────────────────────────────────────────────────
def load_userdata():
    userdata_dir = os.path.join(os.path.dirname(__file__), '..', 'userdata')
    all_spins = []
    files = sorted(f for f in os.listdir(userdata_dir) if f.endswith('.txt'))
    for fname in files:
        path = os.path.join(userdata_dir, fname)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    all_spins.append(int(line))
    return all_spins


# ─── Model implementations (standalone for fast simulation) ─────────

class FreqModel:
    """Laplace-smoothed frequency with optional time decay."""
    def __init__(self, decay=1.0, flat_w=1.0, recent_w=0.0):
        self.decay = decay
        self.flat_w = flat_w
        self.recent_w = recent_w
        self.counts = np.ones(TOTAL_NUMBERS)  # Laplace
        self.total = TOTAL_NUMBERS
        self.history = []

    def update(self, num):
        self.counts[num] += 1
        self.total += 1
        self.history.append(num)

    def load_history(self, h):
        self.history = list(h)
        self.counts = np.ones(TOTAL_NUMBERS)
        self.total = TOTAL_NUMBERS
        for n in h:
            self.counts[n] += 1
            self.total += 1

    def probabilities(self):
        flat = self.counts / self.total
        if self.recent_w == 0 or self.decay >= 1.0 or len(self.history) < 10:
            return flat
        # Time-weighted
        weights = np.zeros(TOTAL_NUMBERS)
        n = len(self.history)
        for i, num in enumerate(self.history):
            w = self.decay ** (n - 1 - i)
            weights[num] += w
        total_w = weights.sum()
        if total_w > 0:
            recent = weights / total_w
        else:
            recent = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
        blend = self.flat_w * flat + self.recent_w * recent
        blend /= blend.sum()
        return blend


class MarkovModel:
    """Order-1 + Order-2 Markov chain."""
    def __init__(self, o1_w=0.6, o2_w=0.4, smoothing=0.5):
        self.o1_w = o1_w
        self.o2_w = o2_w
        self.smoothing = smoothing
        self.trans1 = np.full((TOTAL_NUMBERS, TOTAL_NUMBERS), smoothing)
        self.trans2 = {}
        self.history = []

    def update(self, num):
        if len(self.history) >= 1:
            prev = self.history[-1]
            self.trans1[prev][num] += 1
        if len(self.history) >= 2:
            key = (self.history[-2], self.history[-1])
            if key not in self.trans2:
                self.trans2[key] = np.full(TOTAL_NUMBERS, self.smoothing)
            self.trans2[key][num] += 1
        self.history.append(num)

    def load_history(self, h):
        self.history = []
        self.trans1 = np.full((TOTAL_NUMBERS, TOTAL_NUMBERS), self.smoothing)
        self.trans2 = {}
        for n in h:
            self.update(n)

    def probabilities(self):
        if len(self.history) < 2:
            return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
        prev = self.history[-1]
        row1 = self.trans1[prev]
        p1 = row1 / row1.sum()

        key = (self.history[-2], self.history[-1])
        if key in self.trans2:
            row2 = self.trans2[key]
            p2 = row2 / row2.sum()
        else:
            p2 = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        blend = self.o1_w * p1 + self.o2_w * p2
        blend /= blend.sum()
        return blend


class PatternModel:
    """Sector-biased pattern detector with repeater and neighbour boosts."""
    def __init__(self, lookback=50, sector_boost=1.3, repeater_boost=1.5,
                 hot_threshold=1.5, neighbour_boost=1.15):
        self.lookback = lookback
        self.sector_boost = sector_boost
        self.repeater_boost = repeater_boost
        self.hot_threshold = hot_threshold
        self.neighbour_boost = neighbour_boost
        self.history = []

    def update(self, num):
        self.history.append(num)

    def load_history(self, h):
        self.history = list(h)

    def probabilities(self):
        probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
        if len(self.history) < 10:
            return probs

        recent = self.history[-self.lookback:]

        # Sector bias: boost sectors that appear more often recently
        sector_counts = Counter()
        for n in recent:
            pos = NUMBER_TO_POSITION.get(n, 0)
            sector = pos // 5
            sector_counts[sector] += 1

        expected_per_sector = len(recent) / (len(WHEEL_ORDER) / 5)
        for num in range(TOTAL_NUMBERS):
            pos = NUMBER_TO_POSITION.get(num, 0)
            sector = pos // 5
            if sector_counts[sector] > expected_per_sector * 1.2:
                probs[num] *= self.sector_boost

        # Repeater boost: numbers that appeared recently
        recent_short = self.history[-5:]
        for n in recent_short:
            if 0 <= n < TOTAL_NUMBERS:
                probs[n] *= self.repeater_boost

        # Hot number boost
        counts = Counter(recent)
        expected = len(recent) / TOTAL_NUMBERS
        for num, cnt in counts.items():
            if cnt > expected * self.hot_threshold:
                probs[num] *= 1.2

        # Neighbour boost: numbers adjacent on wheel to recent hits
        for n in recent_short:
            pos = NUMBER_TO_POSITION.get(n, -1)
            if pos >= 0:
                for offset in [-2, -1, 1, 2]:
                    nb_pos = (pos + offset) % len(WHEEL_ORDER)
                    nb_num = WHEEL_ORDER[nb_pos]
                    probs[nb_num] *= self.neighbour_boost

        probs /= probs.sum()
        return probs


class EnsembleModel:
    """Combines Frequency, Markov, Pattern models with configurable weights."""
    def __init__(self, freq_cfg, markov_cfg, pattern_cfg, weights,
                 diversify_after=3, exploration_factor=0.3):
        self.freq = FreqModel(**freq_cfg)
        self.markov = MarkovModel(**markov_cfg)
        self.pattern = PatternModel(**pattern_cfg)
        self.weights = weights  # {'freq': 0.3, 'markov': 0.3, 'pattern': 0.4}
        self.diversify_after = diversify_after
        self.exploration_factor = exploration_factor
        self.consecutive_misses = 0
        self.history = []

    def load_history(self, h):
        self.history = list(h)
        self.freq.load_history(h)
        self.markov.load_history(h)
        self.pattern.load_history(h)

    def update(self, num, predicted_nums=None):
        self.history.append(num)
        self.freq.update(num)
        self.markov.update(num)
        self.pattern.update(num)
        if predicted_nums is not None:
            if num in predicted_nums:
                self.consecutive_misses = 0
            else:
                self.consecutive_misses += 1

    def probabilities(self):
        fp = self.freq.probabilities()
        mp = self.markov.probabilities()
        pp = self.pattern.probabilities()

        w = self.weights
        ensemble = w['freq'] * fp + w['markov'] * mp + w['pattern'] * pp
        ensemble /= ensemble.sum()

        # Loss-reactive diversification
        if self.consecutive_misses >= self.diversify_after:
            uniform = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
            extra = self.consecutive_misses - self.diversify_after
            exploration = min(0.6, self.exploration_factor + extra * 0.1)
            ensemble = (1 - exploration) * ensemble + exploration * uniform
            ensemble /= ensemble.sum()

        return ensemble

    def get_top_n(self, n):
        """Get top-N numbers by probability."""
        probs = self.probabilities()
        top_indices = np.argsort(probs)[::-1][:n]
        return list(top_indices), [float(probs[i]) for i in top_indices]

    def get_dynamic(self, factor):
        """Get numbers above threshold, capped at 18."""
        probs = self.probabilities()
        threshold = (1.0 / TOTAL_NUMBERS) * factor
        selected = [(i, probs[i]) for i in range(TOTAL_NUMBERS) if probs[i] >= threshold]
        selected.sort(key=lambda x: x[1], reverse=True)
        selected = selected[:18]
        if len(selected) < 3:
            top = np.argsort(probs)[::-1][:3]
            selected = [(int(i), float(probs[i])) for i in top]
        return [s[0] for s in selected], [s[1] for s in selected]


# ─── Bet Sizing Strategies ──────────────────────────────────────────

class FlatBet:
    """Fixed $X per number every spin."""
    def __init__(self, bet_per_number):
        self.bet = bet_per_number

    def get_bet(self, bankroll, consecutive_losses, consecutive_wins, spin_num):
        return self.bet

    def name(self):
        return f"Flat${self.bet}"


class ProgressiveBet:
    """Increase bet after losses, decrease after wins."""
    def __init__(self, base=1.0, increment=1.0, decrement=1.0,
                 loss_trigger=3, win_trigger=2, max_bet=20.0, min_bet=1.0):
        self.base = base
        self.increment = increment
        self.decrement = decrement
        self.loss_trigger = loss_trigger
        self.win_trigger = win_trigger
        self.max_bet = max_bet
        self.min_bet = min_bet
        self.current_bet = base

    def get_bet(self, bankroll, consecutive_losses, consecutive_wins, spin_num):
        if consecutive_losses >= self.loss_trigger:
            self.current_bet = min(self.max_bet, self.current_bet + self.increment)
        elif consecutive_wins >= self.win_trigger:
            self.current_bet = max(self.min_bet, self.current_bet - self.decrement)
        return self.current_bet

    def name(self):
        return f"Prog(b={self.base},+{self.increment}L{self.loss_trigger},-{self.decrement}W{self.win_trigger},max={self.max_bet})"


class PercentBet:
    """Bet a percentage of bankroll per number."""
    def __init__(self, pct=0.002, min_bet=1.0, max_bet=20.0):
        self.pct = pct
        self.min_bet = min_bet
        self.max_bet = max_bet

    def get_bet(self, bankroll, consecutive_losses, consecutive_wins, spin_num):
        bet = bankroll * self.pct
        return max(self.min_bet, min(self.max_bet, bet))

    def name(self):
        return f"Pct({self.pct*100:.1f}%,max={self.max_bet})"


class TargetBet:
    """Adjust bet to reach target profit within session. Smarter approach."""
    def __init__(self, target=100, session_length=300, min_bet=1.0, max_bet=15.0,
                 num_count=10, edge_estimate=0.015):
        self.target = target
        self.session_length = session_length
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.num_count = num_count
        self.edge_estimate = edge_estimate

    def get_bet(self, bankroll, consecutive_losses, consecutive_wins, spin_num):
        remaining_spins = max(1, self.session_length - spin_num)
        current_profit = bankroll - 4000
        remaining_target = self.target - current_profit

        if remaining_target <= 0:
            # Already hit target, bet minimum to protect
            return self.min_bet

        # Expected profit per spin = bet * num_count * (hit_rate * 35 - 1)
        # where hit_rate ~ (num_count/37) * (1 + edge)
        hit_rate = (self.num_count / 37.0) * (1.0 + self.edge_estimate)
        ev_per_dollar = self.num_count * (hit_rate * 35 - 1)
        if ev_per_dollar <= 0:
            ev_per_dollar = 0.01  # prevent division by zero

        ideal_bet = remaining_target / (remaining_spins * ev_per_dollar)
        bet = max(self.min_bet, min(self.max_bet, ideal_bet))

        # Safety: don't bet more than 1% of bankroll per number
        safety_cap = bankroll * 0.01
        bet = min(bet, safety_cap)

        return round(bet, 1)

    def name(self):
        return f"Target($100,ses={self.session_length},max={self.max_bet})"


# ─── Session Simulator ──────────────────────────────────────────────

def simulate_session(all_spins, start_idx, session_length, model_cfg, num_strategy,
                     bet_strategy, warmup=100, bankroll=4000.0, stop_loss=3000.0,
                     target_profit=100.0):
    """Simulate a single session.

    Returns dict with profit, hit_rate, max_drawdown, spins_played, etc.
    """
    # Build model and warm up
    model = EnsembleModel(**model_cfg)
    warmup_data = all_spins[max(0, start_idx - warmup):start_idx]
    if warmup_data:
        model.load_history(warmup_data)

    end_idx = min(start_idx + session_length, len(all_spins))
    if end_idx <= start_idx:
        return None

    initial_bankroll = bankroll
    max_bankroll = bankroll
    max_drawdown = 0
    hits = 0
    total_bets = 0
    total_wagered = 0
    consecutive_losses = 0
    consecutive_wins = 0
    target_hit_spin = None
    spins_played = 0

    for i in range(start_idx, end_idx):
        spin_num = i - start_idx
        actual = all_spins[i]

        # Get prediction
        if num_strategy[0] == 'top':
            predicted, probs = model.get_top_n(num_strategy[1])
        elif num_strategy[0] == 'dynamic':
            predicted, probs = model.get_dynamic(num_strategy[1])
        else:
            predicted, probs = model.get_top_n(10)

        num_count = len(predicted)

        # Get bet size per number
        bet_per_num = bet_strategy.get_bet(bankroll, consecutive_losses,
                                            consecutive_wins, spin_num)

        total_cost = bet_per_num * num_count

        # Stop loss check
        if bankroll - total_cost < stop_loss and total_cost > bankroll * 0.05:
            # Reduce to minimum bet
            bet_per_num = bet_strategy.min_bet if hasattr(bet_strategy, 'min_bet') else 1.0
            total_cost = bet_per_num * num_count

        if bankroll < total_cost:
            break  # Busted

        # Place bets
        bankroll -= total_cost
        total_wagered += total_cost
        total_bets += 1
        spins_played += 1

        # Check result
        if actual in predicted:
            payout = bet_per_num * 36  # 35:1 + return of bet
            bankroll += payout
            hits += 1
            consecutive_wins += 1
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            consecutive_wins = 0

        # Track max drawdown
        max_bankroll = max(max_bankroll, bankroll)
        drawdown = max_bankroll - bankroll
        max_drawdown = max(max_drawdown, drawdown)

        # Track when target was first hit
        if target_hit_spin is None and (bankroll - initial_bankroll) >= target_profit:
            target_hit_spin = spin_num

        # Update model
        model.update(actual, set(predicted))

    profit = bankroll - initial_bankroll
    hit_rate = hits / total_bets if total_bets > 0 else 0
    baseline = (num_count / 37.0) if num_count else 0
    edge = hit_rate - baseline

    return {
        'profit': round(profit, 2),
        'final_bankroll': round(bankroll, 2),
        'hit_rate': round(hit_rate * 100, 2),
        'baseline': round(baseline * 100, 2),
        'edge': round(edge * 100, 2),
        'hits': hits,
        'total_bets': total_bets,
        'spins_played': spins_played,
        'max_drawdown': round(max_drawdown, 2),
        'total_wagered': round(total_wagered, 2),
        'roi': round(profit / total_wagered * 100, 2) if total_wagered > 0 else 0,
        'target_hit': target_hit_spin is not None,
        'target_hit_spin': target_hit_spin,
        'num_count': num_count,
    }


def run_multi_session(all_spins, session_length, model_cfg, num_strategy,
                      bet_strategy, warmup=100, overlap_step=50):
    """Run multiple overlapping sessions through the entire dataset.

    Slide a window of session_length through all_spins with overlap_step.
    """
    results = []
    start = warmup
    while start + session_length <= len(all_spins):
        # Reset bet strategy state for each session
        if isinstance(bet_strategy, ProgressiveBet):
            bet_strategy.current_bet = bet_strategy.base
        result = simulate_session(
            all_spins, start, session_length, model_cfg, num_strategy,
            bet_strategy, warmup=warmup
        )
        if result:
            results.append(result)
        start += overlap_step
    return results


# ─── Main Optimization ──────────────────────────────────────────────

def main():
    print("=" * 90)
    print("SESSION OPTIMIZER — Finding config for +$100/session from $4,000 bankroll")
    print("=" * 90)

    all_spins = load_userdata()
    print(f"\nData: {len(all_spins)} spins from userdata/")

    import time
    t0 = time.time()

    # ─── Model configurations to test (from backtest winners) ────────
    model_configs = {
        'PatternHeavy': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.15},
            'weights': {'freq': 0.25, 'markov': 0.15, 'pattern': 0.60},
            'diversify_after': 3, 'exploration_factor': 0.3,
        },
        'PatternDominant': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.4, 'repeater_boost': 1.8,
                           'hot_threshold': 1.3, 'neighbour_boost': 1.2},
            'weights': {'freq': 0.15, 'markov': 0.10, 'pattern': 0.75},
            'diversify_after': 4, 'exploration_factor': 0.25,
        },
        'TunedEnsemble': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.15},
            'weights': {'freq': 0.60, 'markov': 0.30, 'pattern': 0.10},
            'diversify_after': 3, 'exploration_factor': 0.3,
        },
        'FreqFlat': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.6, 'o2_w': 0.4, 'smoothing': 0.5},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.15},
            'weights': {'freq': 0.80, 'markov': 0.10, 'pattern': 0.10},
            'diversify_after': 3, 'exploration_factor': 0.3,
        },
        'BalancedFP': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.15},
            'weights': {'freq': 0.45, 'markov': 0.10, 'pattern': 0.45},
            'diversify_after': 3, 'exploration_factor': 0.3,
        },
        'AggressivePattern': {
            'freq_cfg': {'decay': 0.995, 'flat_w': 0.5, 'recent_w': 0.5},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 40, 'sector_boost': 1.5, 'repeater_boost': 2.0,
                           'hot_threshold': 1.2, 'neighbour_boost': 1.25},
            'weights': {'freq': 0.20, 'markov': 0.10, 'pattern': 0.70},
            'diversify_after': 5, 'exploration_factor': 0.2,
        },
        'SectorFocused': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.6, 'repeater_boost': 1.3,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.3},
            'weights': {'freq': 0.20, 'markov': 0.15, 'pattern': 0.65},
            'diversify_after': 4, 'exploration_factor': 0.3,
        },
        'NoDiversify': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.15},
            'weights': {'freq': 0.25, 'markov': 0.15, 'pattern': 0.60},
            'diversify_after': 999, 'exploration_factor': 0.0,  # Never diversify
        },
    }

    # ─── Number selection strategies ─────────────────────────────────
    num_strategies = {
        'top-8':  ('top', 8),
        'top-10': ('top', 10),
        'top-12': ('top', 12),
        'top-14': ('top', 14),
        'dyn-1.05': ('dynamic', 1.05),
        'dyn-1.10': ('dynamic', 1.10),
    }

    # ─── Bet sizing strategies ───────────────────────────────────────
    bet_strategies = {
        'Flat$1': FlatBet(1.0),
        'Flat$2': FlatBet(2.0),
        'Flat$3': FlatBet(3.0),
        'Flat$5': FlatBet(5.0),
        'Prog-mild': ProgressiveBet(base=2.0, increment=1.0, decrement=1.0,
                                     loss_trigger=3, win_trigger=2, max_bet=10.0),
        'Prog-aggr': ProgressiveBet(base=2.0, increment=2.0, decrement=1.0,
                                     loss_trigger=3, win_trigger=2, max_bet=15.0),
        'Prog-steep': ProgressiveBet(base=3.0, increment=2.0, decrement=1.0,
                                      loss_trigger=2, win_trigger=2, max_bet=20.0),
        'Pct-0.1%': PercentBet(pct=0.001, max_bet=10.0),
        'Pct-0.2%': PercentBet(pct=0.002, max_bet=15.0),
        'Pct-0.3%': PercentBet(pct=0.003, max_bet=20.0),
        'Target-300': TargetBet(target=100, session_length=300, max_bet=15.0,
                                 num_count=10, edge_estimate=0.015),
        'Target-200': TargetBet(target=100, session_length=200, max_bet=15.0,
                                 num_count=10, edge_estimate=0.015),
    }

    # ─── Session lengths to test ─────────────────────────────────────
    session_lengths = [100, 200, 300]

    # ─── Run all combinations ────────────────────────────────────────
    all_results = []
    total_combos = len(model_configs) * len(num_strategies) * len(bet_strategies) * len(session_lengths)
    tested = 0

    print(f"\nTesting {total_combos} strategy combinations...")
    print(f"  {len(model_configs)} model configs × {len(num_strategies)} num strategies "
          f"× {len(bet_strategies)} bet strategies × {len(session_lengths)} session lengths\n")

    for model_name, model_cfg in model_configs.items():
        for num_name, num_strat in num_strategies.items():
            for bet_name, bet_strat in bet_strategies.items():
                for ses_len in session_lengths:
                    tested += 1
                    if tested % 200 == 0:
                        print(f"  Progress: {tested}/{total_combos} ({tested*100//total_combos}%)")

                    sessions = run_multi_session(
                        all_spins, ses_len, model_cfg, num_strat, bet_strat,
                        warmup=100, overlap_step=ses_len // 2  # 50% overlap
                    )

                    if not sessions:
                        continue

                    profits = [s['profit'] for s in sessions]
                    hit_rates = [s['hit_rate'] for s in sessions]
                    drawdowns = [s['max_drawdown'] for s in sessions]
                    target_hits = [1 if s['target_hit'] else 0 for s in sessions]

                    avg_profit = np.mean(profits)
                    median_profit = np.median(profits)
                    min_profit = min(profits)
                    max_profit = max(profits)
                    pct_profitable = sum(1 for p in profits if p > 0) / len(profits) * 100
                    pct_target = sum(target_hits) / len(target_hits) * 100
                    avg_drawdown = np.mean(drawdowns)
                    max_drawdown = max(drawdowns)
                    avg_hit_rate = np.mean(hit_rates)
                    worst_session = min(profits)
                    std_profit = np.std(profits)

                    # Risk-adjusted score: we want high avg profit with low variance
                    # and the session should not bust (min > -1000)
                    sharpe = avg_profit / std_profit if std_profit > 0 else 0
                    score = avg_profit * (pct_target / 100) - max_drawdown * 0.1

                    all_results.append({
                        'model': model_name,
                        'nums': num_name,
                        'bet': bet_name,
                        'session_len': ses_len,
                        'n_sessions': len(sessions),
                        'avg_profit': round(avg_profit, 2),
                        'median_profit': round(median_profit, 2),
                        'min_profit': round(min_profit, 2),
                        'max_profit': round(max_profit, 2),
                        'std_profit': round(std_profit, 2),
                        'pct_profitable': round(pct_profitable, 1),
                        'pct_target_hit': round(pct_target, 1),
                        'avg_hit_rate': round(avg_hit_rate, 2),
                        'avg_drawdown': round(avg_drawdown, 2),
                        'max_drawdown': round(max_drawdown, 2),
                        'sharpe': round(sharpe, 3),
                        'score': round(score, 2),
                    })

    # ─── Sort by score and print results ─────────────────────────────
    all_results.sort(key=lambda x: x['score'], reverse=True)

    elapsed = time.time() - t0
    print(f"\nOptimization time: {elapsed:.0f}s")
    print(f"Total strategy combinations: {total_combos}")
    print(f"Strategies with sessions: {len(all_results)}")

    # ─── TOP 30 by score ─────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("TOP 30 STRATEGIES BY SCORE (avg_profit × target_hit_rate - drawdown_penalty)")
    print("=" * 130)
    header = f"{'#':>3} {'Model':<18} {'Nums':<10} {'Bet':<16} {'SLen':>4} {'#Ses':>4} " \
             f"{'AvgP':>8} {'MedP':>8} {'MinP':>8} {'MaxP':>8} {'StdP':>7} " \
             f"{'%Prof':>6} {'%Tgt':>5} {'AvgHR':>6} {'MaxDD':>7} {'Sharpe':>7} {'Score':>7}"
    print(header)
    print("-" * 130)

    for i, r in enumerate(all_results[:30]):
        print(f"{i+1:>3} {r['model']:<18} {r['nums']:<10} {r['bet']:<16} {r['session_len']:>4} "
              f"{r['n_sessions']:>4} {r['avg_profit']:>8.1f} {r['median_profit']:>8.1f} "
              f"{r['min_profit']:>8.1f} {r['max_profit']:>8.1f} {r['std_profit']:>7.1f} "
              f"{r['pct_profitable']:>5.1f}% {r['pct_target_hit']:>4.1f}% "
              f"{r['avg_hit_rate']:>5.1f}% {r['max_drawdown']:>7.1f} "
              f"{r['sharpe']:>7.3f} {r['score']:>7.1f}")

    # ─── TOP 30 by % sessions hitting $100 target ────────────────────
    all_results.sort(key=lambda x: (x['pct_target_hit'], x['avg_profit']), reverse=True)
    print("\n" + "=" * 130)
    print("TOP 30 STRATEGIES BY % OF SESSIONS HITTING $100+ PROFIT TARGET")
    print("=" * 130)
    print(header)
    print("-" * 130)

    for i, r in enumerate(all_results[:30]):
        print(f"{i+1:>3} {r['model']:<18} {r['nums']:<10} {r['bet']:<16} {r['session_len']:>4} "
              f"{r['n_sessions']:>4} {r['avg_profit']:>8.1f} {r['median_profit']:>8.1f} "
              f"{r['min_profit']:>8.1f} {r['max_profit']:>8.1f} {r['std_profit']:>7.1f} "
              f"{r['pct_profitable']:>5.1f}% {r['pct_target_hit']:>4.1f}% "
              f"{r['avg_hit_rate']:>5.1f}% {r['max_drawdown']:>7.1f} "
              f"{r['sharpe']:>7.3f} {r['score']:>7.1f}")

    # ─── TOP 30 by highest average profit ────────────────────────────
    all_results.sort(key=lambda x: x['avg_profit'], reverse=True)
    print("\n" + "=" * 130)
    print("TOP 30 STRATEGIES BY AVERAGE PROFIT PER SESSION")
    print("=" * 130)
    print(header)
    print("-" * 130)

    for i, r in enumerate(all_results[:30]):
        print(f"{i+1:>3} {r['model']:<18} {r['nums']:<10} {r['bet']:<16} {r['session_len']:>4} "
              f"{r['n_sessions']:>4} {r['avg_profit']:>8.1f} {r['median_profit']:>8.1f} "
              f"{r['min_profit']:>8.1f} {r['max_profit']:>8.1f} {r['std_profit']:>7.1f} "
              f"{r['pct_profitable']:>5.1f}% {r['pct_target_hit']:>4.1f}% "
              f"{r['avg_hit_rate']:>5.1f}% {r['max_drawdown']:>7.1f} "
              f"{r['sharpe']:>7.3f} {r['score']:>7.1f}")

    # ─── SAFEST strategies (lowest max drawdown among profitable ones) ─
    safe = [r for r in all_results if r['avg_profit'] > 0]
    safe.sort(key=lambda x: x['max_drawdown'])
    print("\n" + "=" * 130)
    print("TOP 20 SAFEST PROFITABLE STRATEGIES (lowest max drawdown)")
    print("=" * 130)
    print(header)
    print("-" * 130)

    for i, r in enumerate(safe[:20]):
        print(f"{i+1:>3} {r['model']:<18} {r['nums']:<10} {r['bet']:<16} {r['session_len']:>4} "
              f"{r['n_sessions']:>4} {r['avg_profit']:>8.1f} {r['median_profit']:>8.1f} "
              f"{r['min_profit']:>8.1f} {r['max_profit']:>8.1f} {r['std_profit']:>7.1f} "
              f"{r['pct_profitable']:>5.1f}% {r['pct_target_hit']:>4.1f}% "
              f"{r['avg_hit_rate']:>5.1f}% {r['max_drawdown']:>7.1f} "
              f"{r['sharpe']:>7.3f} {r['score']:>7.1f}")

    # ─── BEST overall recommendation ─────────────────────────────────
    # Score: highest (avg_profit >= 100) and (pct_target >= 30%) and (max_drawdown < 2000)
    candidates = [r for r in all_results
                  if r['avg_profit'] >= 50 and r['max_drawdown'] < 2000]
    if candidates:
        candidates.sort(key=lambda x: x['pct_target_hit'] * x['avg_profit'] / max(1, x['max_drawdown']),
                       reverse=True)
        best = candidates[0]
    else:
        all_results.sort(key=lambda x: x['avg_profit'], reverse=True)
        best = all_results[0]

    print("\n" + "=" * 90)
    print("★ RECOMMENDED CONFIGURATION ★")
    print("=" * 90)
    print(f"  Model:          {best['model']}")
    print(f"  Numbers:        {best['nums']}")
    print(f"  Bet Strategy:   {best['bet']}")
    print(f"  Session Length:  {best['session_len']} spins")
    print(f"  ---")
    print(f"  Avg Profit:     ${best['avg_profit']:.2f}")
    print(f"  Median Profit:  ${best['median_profit']:.2f}")
    print(f"  Min Profit:     ${best['min_profit']:.2f} (worst session)")
    print(f"  Max Profit:     ${best['max_profit']:.2f} (best session)")
    print(f"  % Profitable:   {best['pct_profitable']:.1f}%")
    print(f"  % Hit $100:     {best['pct_target_hit']:.1f}%")
    print(f"  Avg Hit Rate:   {best['avg_hit_rate']:.1f}%")
    print(f"  Max Drawdown:   ${best['max_drawdown']:.2f}")
    print(f"  Sharpe Ratio:   {best['sharpe']:.3f}")

    # Save all results
    out_path = os.path.join(os.path.dirname(__file__), 'session_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == '__main__':
    main()
