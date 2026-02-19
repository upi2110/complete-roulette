#!/usr/bin/env python3
"""
Bet Strategy MEGA Sweep — 10,000+ Money Management Strategies.

Tests 20 strategy types across ALL data files to find strategies
that hit +$100 EVERY session (100% hit rate target).

STRATEGY TYPES:
  V1: Escalation (increase during losses, decrease during wins)
  A:  Recovery After Hit (flat during losses, bump after hit)
  B:  Streak Rider (increase during win streaks)
  C:  Loss Counter Recovery (bump after hit based on loss count)
  D:  Patience Recovery (only bump after long cold streaks)
  E:  Progressive Win Rider (progressive increase during win streaks)
  F:  Hybrid (recovery + streak combined)
  G:  Percentage Recovery (recover X% of losses after hit)
  H:  Fibonacci (follow Fib sequence on losses, drop levels on win)
  I:  D'Alembert (+step on loss, -step on win)
  J:  Anti-Martingale (multiply on win, reset on loss)
  K:  Labouchere (sequence-based: bet=first+last, cross off on win)
  L:  Oscar's Grind (goal: +1 unit per cycle)
  M:  Paroli (double up to N wins then reset)
  N:  1-3-2-6 System (fixed progression on consecutive wins)
  O:  Plateau (flat for N spins then step up)
  P:  Martingale Lite (double on loss with low cap)
  Q:  Reverse Recovery (bet high when hot, low when cold)
  R:  Pure Grind (flat bet always — baseline)
  S:  Session Profit Aware (adjust based on current P&L)
  T:  Time-Based Escalation (increase every N spins)

Total: 100,000+ strategy combinations.
Tests across ALL data files (~5,000+ numbers).
"""

import sys
import os
import time
import math
import numpy as np
from collections import defaultdict, deque

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from config import (
    TOTAL_NUMBERS, TOP_PREDICTIONS_COUNT,
    SESSION_TARGET, INITIAL_BANKROLL,
)
from app.ml.ensemble import EnsemblePredictor

NUM_COUNT = TOP_PREDICTIONS_COUNT  # 14 numbers per spin
MAX_SESSION_SPINS = 60  # Hard cap: 60 spins per session max
FIB_SEQ = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


def load_data_file(filepath):
    """Load numbers from a data file."""
    numbers = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                n = int(line)
                if 0 <= n <= 36:
                    numbers.append(n)
    return numbers


def precompute_predictions(all_numbers):
    """Pre-compute predictions for every position (done ONCE per data file)."""
    predictor = EnsemblePredictor()
    predictions = []

    for i, number in enumerate(all_numbers):
        if i >= 3:
            pred = predictor.predict()
            top_nums = pred.get('top_numbers', [])[:NUM_COUNT]
            predictions.append((set(top_nums), top_nums))
        else:
            predictions.append((set(), []))
        predictor.update(number)

    return predictions


# ═══════════════════════════════════════════════════════════════════════
# SESSION RUNNERS — One per strategy type
# ═══════════════════════════════════════════════════════════════════════

def _session_setup(all_numbers, start_idx):
    """Common session setup. Returns remaining count or None if too short."""
    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None
    return min(remaining, MAX_SESSION_SPINS)


def _session_result(spins_played, total_wins, total_losses, session_profit, max_drawdown):
    """Build standard result dict."""
    if spins_played == 0:
        return None
    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1),
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= SESSION_TARGET,
        'max_drawdown': round(max_drawdown, 2),
    }


# ─── TYPE V1: Original Escalation ─────────────────────────────────────
def run_session_v1(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    increment = params['increment']
    decrement = params['decrement']
    losses_before_inc = params['losses_before_inc']
    wins_before_dec = params['wins_before_dec']
    max_bet = params['max_bet']

    bet = base_bet
    consec_losses = 0
    consec_wins = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            consec_wins += 1
            consec_losses = 0
            if consec_wins >= wins_before_dec:
                bet = max(1.0, bet - decrement)
                consec_wins = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses += 1
            consec_losses += 1
            consec_wins = 0
            if consec_losses >= losses_before_inc:
                bet = min(max_bet, bet + increment)
                consec_losses = 0

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses, profit, dd)


# ─── TYPE A: Recovery After Hit ───────────────────────────────────────
def run_session_a(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    divisor = params['divisor']
    recovery_wins = params['recovery_wins']
    max_bet = params['max_bet']

    bet = base_bet
    consec_losses = 0
    recovery_left = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            if recovery_left > 0:
                recovery_left -= 1
                if recovery_left == 0:
                    bet = base_bet
            else:
                if consec_losses > 0:
                    recovery_amount = math.ceil(consec_losses / divisor)
                    bet = min(max_bet, base_bet + recovery_amount)
                    recovery_left = recovery_wins - 1
                    if recovery_left <= 0:
                        bet = base_bet
            consec_losses = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_losses += 1
            if recovery_left > 0:
                recovery_left = 0
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE B: Streak Rider ─────────────────────────────────────────────
def run_session_b(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    increment = params['increment']
    max_bet = params['max_bet']

    bet = base_bet
    consec_wins = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            consec_wins += 1
            bet = min(max_bet, base_bet + consec_wins * increment)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_wins = 0
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE C: Loss Counter Recovery ────────────────────────────────────
def run_session_c(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    divisor = params['divisor']
    step = params['step']
    hold_wins = params['hold_wins']
    max_bet = params['max_bet']

    bet = base_bet
    consec_losses = 0
    elevated_left = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            if elevated_left > 0:
                elevated_left -= 1
                if elevated_left == 0:
                    bet = base_bet
            else:
                if consec_losses > 0:
                    bump = math.floor(consec_losses / divisor) * step
                    bet = min(max_bet, base_bet + bump)
                    elevated_left = hold_wins - 1
                    if elevated_left <= 0:
                        bet = base_bet
            consec_losses = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_losses += 1
            if elevated_left > 0:
                elevated_left = 0
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE D: Patience Recovery ────────────────────────────────────────
def run_session_d(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    min_losses = params['min_losses']
    multiplier = params['multiplier']
    hold_wins = params['hold_wins']
    max_bet = params['max_bet']

    bet = base_bet
    consec_losses = 0
    elevated_left = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            if elevated_left > 0:
                elevated_left -= 1
                if elevated_left == 0:
                    bet = base_bet
            else:
                if consec_losses >= min_losses:
                    bet = min(max_bet, base_bet * multiplier)
                    elevated_left = hold_wins - 1
                    if elevated_left <= 0:
                        bet = base_bet
            consec_losses = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_losses += 1
            if elevated_left > 0:
                elevated_left = 0
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE E: Progressive Win Rider ────────────────────────────────────
def run_session_e(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    step = params['step']
    wins_before_boost = params['wins_before_boost']
    max_bet = params['max_bet']

    bet = base_bet
    consec_wins = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            consec_wins += 1
            if consec_wins >= wins_before_boost:
                boosts = consec_wins - wins_before_boost + 1
                bet = min(max_bet, base_bet + boosts * step)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_wins = 0
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE F: Hybrid Recovery + Streak ─────────────────────────────────
def run_session_f(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    divisor = params['divisor']
    win_step = params['win_step']
    max_bet = params['max_bet']

    bet = base_bet
    consec_losses = 0
    consec_wins = 0
    in_recovery = False
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            consec_wins += 1
            if consec_losses > 0 and not in_recovery:
                recovery_amount = math.ceil(consec_losses / divisor)
                bet = min(max_bet, base_bet + recovery_amount)
                in_recovery = True
            elif consec_wins >= 2:
                bet = min(max_bet, bet + win_step)
            consec_losses = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_losses += 1
            consec_wins = 0
            in_recovery = False
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE G: Percentage Recovery ──────────────────────────────────────
def run_session_g(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    recovery_pct = params['recovery_pct']
    max_bet = params['max_bet']
    payout_per_unit = 36 - NUM_COUNT

    bet = base_bet
    cumulative_loss = 0.0
    in_recovery = False
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            if in_recovery:
                in_recovery = False
                cumulative_loss = 0.0
                bet = base_bet
            elif cumulative_loss > 0:
                target_recovery = cumulative_loss * recovery_pct
                recovery_bet = base_bet + math.ceil(target_recovery / max(payout_per_unit, 1))
                bet = min(max_bet, recovery_bet)
                in_recovery = True
                cumulative_loss = 0.0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            cumulative_loss += total_bet
            if in_recovery:
                in_recovery = False
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE H: Fibonacci ────────────────────────────────────────────────
def run_session_h(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    max_fib_level = params['max_fib_level']
    drop_on_win = params['drop_on_win']
    max_bet = params['max_bet']

    fib_level = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        bet = min(max_bet, base_bet * FIB_SEQ[fib_level])
        total_bet = bet * NUM_COUNT

        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            fib_level = max(0, fib_level - drop_on_win)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            fib_level = min(max_fib_level, fib_level + 1)

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE I: D'Alembert ───────────────────────────────────────────────
def run_session_i(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    step = params['step']
    max_bet = params['max_bet']

    bet = base_bet
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            bet = max(1.0, bet - step)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            bet = min(max_bet, bet + step)

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE J: Anti-Martingale ──────────────────────────────────────────
def run_session_j(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    multiplier = params['multiplier']
    max_doubles = params['max_doubles']
    max_bet = params['max_bet']

    bet = base_bet
    consec_wins = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            consec_wins += 1
            if consec_wins < max_doubles:
                bet = min(max_bet, bet * multiplier)
            else:
                bet = base_bet
                consec_wins = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_wins = 0
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE K: Labouchere ───────────────────────────────────────────────
def run_session_k(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    initial_seq = list(params['initial_sequence'])
    max_bet = params['max_bet']

    sequence = list(initial_seq)
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        if not sequence:
            sequence = list(initial_seq)

        if len(sequence) >= 2:
            bet_units = sequence[0] + sequence[-1]
        else:
            bet_units = sequence[0]

        bet = min(max_bet, base_bet * bet_units)
        total_bet = bet * NUM_COUNT

        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            if len(sequence) >= 2:
                sequence.pop(0)
                if sequence:
                    sequence.pop(-1)
            else:
                sequence.pop(0)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            sequence.append(bet_units)

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE L: Oscar's Grind ────────────────────────────────────────────
def run_session_l(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    unit_size = params['unit_size']
    max_bet = params['max_bet']

    bet_units = 1
    cycle_profit = 0.0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        bet = min(max_bet, unit_size * bet_units)
        total_bet = bet * NUM_COUNT

        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            cycle_profit += net
            wins += 1
            if cycle_profit >= unit_size * 22:  # one unit profit from a hit
                bet_units = 1
                cycle_profit = 0.0
            elif cycle_profit < 0:
                max_units = max(1, int(max_bet / unit_size))
                bet_units = min(max_units, bet_units + 1)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            cycle_profit += net
            losses_count += 1
            # Oscar's Grind: keep same bet on loss

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE M: Paroli ───────────────────────────────────────────────────
def run_session_m(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    max_paroli_wins = params['max_paroli_wins']
    multiplier = params['multiplier']
    max_bet = params['max_bet']

    bet = base_bet
    consec_wins = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            consec_wins += 1
            if consec_wins >= max_paroli_wins:
                bet = base_bet
                consec_wins = 0
            else:
                bet = min(max_bet, bet * multiplier)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_wins = 0
            bet = base_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE N: 1-3-2-6 System ──────────────────────────────────────────
def run_session_n(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    unit_size = params['unit_size']
    seq = params['sequence']
    max_bet = params['max_bet']

    seq_pos = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        bet = min(max_bet, unit_size * seq[seq_pos])
        total_bet = bet * NUM_COUNT

        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            seq_pos += 1
            if seq_pos >= len(seq):
                seq_pos = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            seq_pos = 0

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE O: Plateau ──────────────────────────────────────────────────
def run_session_o(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    step_up = params['step_up']
    plateau_length = params['plateau_length']
    max_bet = params['max_bet']

    bet = base_bet
    spins_in_plateau = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1

        spins += 1
        spins_in_plateau += 1
        if spins_in_plateau >= plateau_length:
            bet = min(max_bet, bet + step_up)
            spins_in_plateau = 0

        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE P: Martingale Lite ──────────────────────────────────────────
def run_session_p(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    multiplier = params['multiplier']
    max_bet = params['max_bet']
    reset_to_base = params['reset_to_base']

    bet = base_bet
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            bet = base_bet if reset_to_base else 1.0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            bet = min(max_bet, bet * multiplier)

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE Q: Reverse Recovery (Ride Hot) ──────────────────────────────
def run_session_q(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    hot_bet = params['hot_bet']
    cold_bet = params['cold_bet']
    lookback = params['lookback']
    hot_threshold = params['hot_threshold']

    recent = deque(maxlen=lookback)
    bet = cold_bet
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            recent.append(1)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            recent.append(0)

        # Decide next bet
        if len(recent) >= lookback:
            win_frac = sum(recent) / len(recent)
            bet = hot_bet if win_frac >= hot_threshold else cold_bet
        else:
            bet = cold_bet

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE R: Pure Grind ───────────────────────────────────────────────
def run_session_r(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    bet = params['flat_bet']
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE S: Session Profit Aware ─────────────────────────────────────
def run_session_s(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    aggressive_bet = params['aggressive_bet']
    conservative_bet = params['conservative_bet']
    pace_target = params['pace_target']
    aggression_threshold = params['aggression_threshold']

    bet = base_bet
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        # Decide bet based on session P&L
        if profit >= pace_target:
            bet = conservative_bet
        elif profit < aggression_threshold:
            bet = aggressive_bet
        else:
            bet = base_bet

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1

        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ─── TYPE T: Time-Based Escalation ────────────────────────────────────
def run_session_t(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None

    base_bet = params['base_bet']
    step = params['step_per_interval']
    interval = params['interval']
    max_bet = params['max_bet']

    bet = base_bet
    spins_since_step = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    dd = 0.0

    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]
        if not pred_set:
            spins += 1
            continue

        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1

        spins += 1
        spins_since_step += 1
        if spins_since_step >= interval:
            bet = min(max_bet, bet + step)
            spins_since_step = 0

        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break

    return _session_result(spins, wins, losses_count, profit, dd)


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY GENERATOR — Target 100,000+
# ═══════════════════════════════════════════════════════════════════════

def generate_all_strategies():
    """Generate 100,000+ strategy combinations across 20 types."""
    strategies = []

    base_bets_xl = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    base_bets_lg = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    base_bets_med = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    base_bets_sm = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    caps_xl = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 200.0]
    caps_lg = [5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 200.0]
    caps_med = [6.0, 8.0, 10.0, 15.0, 200.0]
    caps_sm = [8.0, 10.0, 15.0, 200.0]
    cap_label = lambda c: f'${c:.0f}' if c < 200 else 'NoCap'

    # ─── V1: Escalation ─── (~18,000)
    for base in [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for inc in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            for dec in [0.25, 0.5, 1.0, 1.5, 2.0]:
                for li in [1, 2, 3, 4, 5]:
                    for wi in [1, 2, 3, 4, 5]:
                        for cap in [6.0, 8.0, 10.0, 15.0, 200.0]:
                            if cap < base:
                                continue
                            strategies.append({
                                'type': 'V1', 'runner': run_session_v1,
                                'label': f"V1 ${base} +${inc}/{li}L -${dec}/{wi}W Cap={cap_label(cap)}",
                                'params': {'base_bet': base, 'increment': inc, 'decrement': dec,
                                           'losses_before_inc': li, 'wins_before_dec': wi, 'max_bet': cap}
                            })

    # ─── A: Recovery After Hit ─── (~3,500)
    for base in base_bets_lg:
        for div in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            for rw in [1, 2, 3, 4, 5]:
                for cap in caps_lg:
                    if cap < base:
                        continue
                    strategies.append({
                        'type': 'A', 'runner': run_session_a,
                        'label': f"A ${base} L/{div} hold{rw}W Cap={cap_label(cap)}",
                        'params': {'base_bet': base, 'divisor': div, 'recovery_wins': rw, 'max_bet': cap}
                    })

    # ─── B: Streak Rider ─── (~1,200)
    for base in base_bets_xl:
        for inc in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            for cap in caps_lg:
                if cap < base:
                    continue
                strategies.append({
                    'type': 'B', 'runner': run_session_b,
                    'label': f"B ${base} +${inc}/win Cap={cap_label(cap)}",
                    'params': {'base_bet': base, 'increment': inc, 'max_bet': cap}
                })

    # ─── C: Loss Counter Recovery ─── (~7,000)
    for base in base_bets_lg:
        for div in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]:
            for step in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
                for hw in [1, 2, 3, 4, 5]:
                    for cap in caps_med:
                        if cap < base:
                            continue
                        strategies.append({
                            'type': 'C', 'runner': run_session_c,
                            'label': f"C ${base} L/{div}*${step} hold{hw}W Cap={cap_label(cap)}",
                            'params': {'base_bet': base, 'divisor': div, 'step': step, 'hold_wins': hw, 'max_bet': cap}
                        })

    # ─── D: Patience Recovery ─── (~6,000)
    for base in base_bets_lg:
        for ml in [2, 3, 4, 5, 6, 8, 10]:
            for mult in [1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
                for hw in [1, 2, 3, 4, 5]:
                    for cap in [8.0, 10.0, 15.0, 200.0]:
                        if cap < base:
                            continue
                        strategies.append({
                            'type': 'D', 'runner': run_session_d,
                            'label': f"D ${base} {ml}L->x{mult} hold{hw}W Cap={cap_label(cap)}",
                            'params': {'base_bet': base, 'min_losses': ml, 'multiplier': mult,
                                       'hold_wins': hw, 'max_bet': cap}
                        })

    # ─── E: Progressive Win Rider ─── (~3,500)
    for base in base_bets_lg:
        for step in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for wb in [1, 2, 3, 4, 5]:
                for cap in caps_lg:
                    if cap < base:
                        continue
                    strategies.append({
                        'type': 'E', 'runner': run_session_e,
                        'label': f"E ${base} +${step}/win>={wb} Cap={cap_label(cap)}",
                        'params': {'base_bet': base, 'step': step, 'wins_before_boost': wb, 'max_bet': cap}
                    })

    # ─── F: Hybrid ─── (~3,800)
    for base in base_bets_lg:
        for div in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            for ws in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
                for cap in caps_med:
                    if cap < base:
                        continue
                    strategies.append({
                        'type': 'F', 'runner': run_session_f,
                        'label': f"F ${base} L/{div}+${ws}/win Cap={cap_label(cap)}",
                        'params': {'base_bet': base, 'divisor': div, 'win_step': ws, 'max_bet': cap}
                    })

    # ─── G: Percentage Recovery ─── (~2,000)
    for base in base_bets_xl:
        for pct in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 0.85, 1.0]:
            for cap in caps_lg:
                if cap < base:
                    continue
                strategies.append({
                    'type': 'G', 'runner': run_session_g,
                    'label': f"G ${base} recov={int(pct*100)}% Cap={cap_label(cap)}",
                    'params': {'base_bet': base, 'recovery_pct': pct, 'max_bet': cap}
                })

    # ─── H: Fibonacci ─── (~3,000)
    for base in base_bets_xl:
        for max_lvl in [3, 4, 5, 6, 7, 8, 9, 10]:
            for drop in [1, 2, 3]:
                for cap in caps_lg:
                    if cap < base:
                        continue
                    strategies.append({
                        'type': 'H', 'runner': run_session_h,
                        'label': f"H ${base} lvl{max_lvl} drop{drop} Cap={cap_label(cap)}",
                        'params': {'base_bet': base, 'max_fib_level': max_lvl, 'drop_on_win': drop, 'max_bet': cap}
                    })

    # ─── I: D'Alembert ─── (~1,300)
    for base in base_bets_xl:
        for step in [0.10, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
            for cap in caps_lg:
                if cap < base:
                    continue
                strategies.append({
                    'type': 'I', 'runner': run_session_i,
                    'label': f"I ${base} step=${step} Cap={cap_label(cap)}",
                    'params': {'base_bet': base, 'step': step, 'max_bet': cap}
                })

    # ─── J: Anti-Martingale ─── (~2,800)
    for base in base_bets_lg:
        for mult in [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
            for md in [2, 3, 4, 5, 6, 7, 8]:
                for cap in caps_med:
                    if cap < base:
                        continue
                    strategies.append({
                        'type': 'J', 'runner': run_session_j,
                        'label': f"J ${base} x{mult} max{md}W Cap={cap_label(cap)}",
                        'params': {'base_bet': base, 'multiplier': mult, 'max_doubles': md, 'max_bet': cap}
                    })

    # ─── K: Labouchere ─── (~2,000)
    seq_configs = [(1, 2, 3), (1, 1, 2, 2), (1, 2, 3, 4), (1, 1, 1, 1, 1),
                   (2, 3, 4), (1, 2, 3, 4, 5), (1, 1, 1), (2, 2, 3, 3),
                   (1, 3, 5), (1, 2, 2, 3), (1, 1, 2, 3, 5), (2, 4, 6),
                   (1, 2), (3, 4, 5), (1, 1, 3, 3, 5), (1, 3, 3, 5),
                   (2, 2, 4, 4), (1, 2, 4), (1, 1, 2, 4), (3, 3, 5)]
    for base in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        for seq in seq_configs:
            for cap in caps_lg:
                if cap < base:
                    continue
                seq_str = ','.join(str(s) for s in seq)
                strategies.append({
                    'type': 'K', 'runner': run_session_k,
                    'label': f"K ${base} seq=[{seq_str}] Cap={cap_label(cap)}",
                    'params': {'base_bet': base, 'initial_sequence': seq, 'max_bet': cap}
                })

    # ─── L: Oscar's Grind ─── (~180)
    for unit in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        for cap in caps_lg:
            if cap < unit:
                continue
            strategies.append({
                'type': 'L', 'runner': run_session_l,
                'label': f"L unit=${unit} Cap={cap_label(cap)}",
                'params': {'unit_size': unit, 'max_bet': cap}
            })

    # ─── M: Paroli ─── (~2,100)
    for base in base_bets_lg:
        for mpw in [2, 3, 4, 5, 6, 7]:
            for mult in [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
                for cap in caps_med:
                    if cap < base:
                        continue
                    strategies.append({
                        'type': 'M', 'runner': run_session_m,
                        'label': f"M ${base} x{mult} max{mpw}W Cap={cap_label(cap)}",
                        'params': {'base_bet': base, 'max_paroli_wins': mpw, 'multiplier': mult, 'max_bet': cap}
                    })

    # ─── N: 1-3-2-6 System ─── (~1,500)
    n_sequences = [[1, 3, 2, 6], [1, 2, 3, 4], [1, 2, 4, 8], [1, 1, 2, 3],
                   [1, 2, 3, 6], [1, 1, 3, 5], [1, 2, 2, 4], [2, 3, 4, 6],
                   [1, 3, 3, 6], [1, 2, 4, 4], [1, 1, 1, 2], [2, 2, 3, 6],
                   [1, 3, 5, 7], [1, 2, 2, 6], [1, 1, 4, 8]]
    for unit in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        for seq in n_sequences:
            for cap in caps_med:
                if cap < unit:
                    continue
                seq_str = '-'.join(str(s) for s in seq)
                strategies.append({
                    'type': 'N', 'runner': run_session_n,
                    'label': f"N ${unit} seq={seq_str} Cap={cap_label(cap)}",
                    'params': {'unit_size': unit, 'sequence': seq, 'max_bet': cap}
                })

    # ─── O: Plateau ─── (~3,500)
    for base in base_bets_lg:
        for step_up in [0.10, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            for pl in [3, 5, 8, 10, 12, 15, 20, 25, 30]:
                for cap in caps_lg:
                    if cap < base:
                        continue
                    strategies.append({
                        'type': 'O', 'runner': run_session_o,
                        'label': f"O ${base} +${step_up}/{pl}sp Cap={cap_label(cap)}",
                        'params': {'base_bet': base, 'step_up': step_up, 'plateau_length': pl, 'max_bet': cap}
                    })

    # ─── P: Martingale Lite ─── (~1,200)
    for base in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        for mult in [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
            for cap in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]:
                if cap < base:
                    continue
                for rtb in [True, False]:
                    strategies.append({
                        'type': 'P', 'runner': run_session_p,
                        'label': f"P ${base} x{mult} Cap=${cap:.0f} rst={'base' if rtb else '$1'}",
                        'params': {'base_bet': base, 'multiplier': mult, 'max_bet': cap, 'reset_to_base': rtb}
                    })

    # ─── Q: Reverse Recovery ─── (~3,000)
    for hb in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]:
        for cb in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            for lb in [3, 5, 8, 10, 15]:
                for ht in [0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.6]:
                    strategies.append({
                        'type': 'Q', 'runner': run_session_q,
                        'label': f"Q hot=${hb} cold=${cb} look{lb} thr={ht}",
                        'params': {'hot_bet': hb, 'cold_bet': cb, 'lookback': lb, 'hot_threshold': ht}
                    })

    # ─── R: Pure Grind ─── (~20)
    for fb in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]:
        strategies.append({
            'type': 'R', 'runner': run_session_r,
            'label': f"R flat=${fb}",
            'params': {'flat_bet': fb}
        })

    # ─── S: Session Profit Aware ─── (~3,000)
    for base in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        for agg in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            for con in [0.25, 0.5, 1.0, 1.5, 2.0]:
                for pt in [25.0, 50.0, 75.0, 100.0]:
                    for at in [-20.0, -30.0, -50.0, -75.0, -100.0, -150.0]:
                        strategies.append({
                            'type': 'S', 'runner': run_session_s,
                            'label': f"S ${base} agg=${agg} con=${con} pace={pt} thr={at}",
                            'params': {'base_bet': base, 'aggressive_bet': agg, 'conservative_bet': con,
                                       'pace_target': pt, 'aggression_threshold': at}
                        })

    # ─── T: Time-Based Escalation ─── (~4,000)
    for base in base_bets_lg:
        for step in [0.10, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            for intv in [2, 3, 5, 7, 8, 10, 12, 15, 20, 25, 30]:
                for cap in caps_lg:
                    if cap < base:
                        continue
                    strategies.append({
                        'type': 'T', 'runner': run_session_t,
                        'label': f"T ${base} +${step}/{intv}sp Cap={cap_label(cap)}",
                        'params': {'base_bet': base, 'step_per_interval': step, 'interval': intv, 'max_bet': cap}
                    })

    return strategies


# ═══════════════════════════════════════════════════════════════════════
# TEST & AGGREGATION
# ═══════════════════════════════════════════════════════════════════════

def run_strategy_test(datasets, strategy):
    """Test one strategy across all data files and starting positions."""
    runner = strategy['runner']
    params = strategy['params']
    results = []

    for all_numbers, predictions, sample_positions in datasets:
        for start_idx in sample_positions:
            result = runner(all_numbers, predictions, start_idx, params)
            if result:
                results.append(result)

    if not results:
        return None

    targets_hit = sum(1 for r in results if r['target_reached'])
    profits = [r['final_profit'] for r in results]
    spins = [r['spins_played'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    win_rates = [r['win_rate'] for r in results]

    return {
        'total_sessions': len(results),
        'targets_hit': targets_hit,
        'hit_rate': round(targets_hit / len(results) * 100, 1),
        'avg_profit': round(np.mean(profits), 2),
        'median_profit': round(float(np.median(profits)), 2),
        'worst_profit': round(min(profits), 2),
        'best_profit': round(max(profits), 2),
        'avg_spins': round(np.mean(spins), 1),
        'avg_drawdown': round(np.mean(drawdowns), 2),
        'worst_drawdown': round(min(drawdowns), 2),
        'avg_win_rate': round(np.mean(win_rates), 1),
    }


def print_table(title, ranked, count=50):
    print(f"\n{'─'*130}")
    print(f"  {title}")
    print(f"{'─'*130}")
    print(f"{'Rank':<5} {'Type':<3} {'Strategy':<55} {'Hit%':>6} {'AvgP':>8} {'MedP':>8} "
          f"{'WorstP':>8} {'BestP':>8} {'AvgSp':>6} {'WrstDd':>8} {'WinR%':>6}")
    print(f"{'─'*130}")

    for i, r in enumerate(ranked[:count], 1):
        res = r['result']
        print(f"{i:<5} {r['type']:<3} {r['label']:<55} "
              f"{res['hit_rate']:>5.1f}% "
              f"${res['avg_profit']:>7.0f} "
              f"${res['median_profit']:>7.0f} "
              f"${res['worst_profit']:>7.0f} "
              f"${res['best_profit']:>7.0f} "
              f"{res['avg_spins']:>5.0f} "
              f"${res['worst_drawdown']:>7.0f} "
              f"{res['avg_win_rate']:>5.1f}%")


def main():
    print(f"\n{'='*130}")
    print(f"BET STRATEGY MEGA SWEEP — 100,000+ Money Management Strategies")
    print(f"{'='*130}")

    # ─── Load ALL data files ───
    datasets = []
    total_numbers = 0

    # Test data
    test_path = os.path.join(os.path.dirname(__file__), 'test_data', 'test_data1.txt')
    if os.path.exists(test_path):
        nums = load_data_file(test_path)
        if nums:
            print(f"  Loading test_data1.txt ... {len(nums)} numbers")
            preds = precompute_predictions(nums)
            positions = list(range(30, len(nums) - 30, 10))
            datasets.append((nums, preds, positions))
            total_numbers += len(nums)

    # Userdata files
    userdata_dir = os.path.join(PROJECT_ROOT, 'userdata')
    if os.path.exists(userdata_dir):
        for i in range(1, 11):
            path = os.path.join(userdata_dir, f'data{i}.txt')
            if os.path.exists(path):
                nums = load_data_file(path)
                if len(nums) >= 60:  # Need at least 60 for meaningful sessions
                    print(f"  Loading data{i}.txt ... {len(nums)} numbers")
                    preds = precompute_predictions(nums)
                    positions = list(range(30, len(nums) - 30, 10))
                    datasets.append((nums, preds, positions))
                    total_numbers += len(nums)

    total_positions = sum(len(d[2]) for d in datasets)
    print(f"\n  Total: {len(datasets)} data files, {total_numbers} numbers, {total_positions} starting positions")

    print(f"\n  Numbers per spin: {NUM_COUNT}")
    print(f"  Max spins/session: {MAX_SESSION_SPINS}")
    print(f"  Target: +${SESSION_TARGET} per session")
    print(f"  Bankroll: ${INITIAL_BANKROLL}")

    # ─── Generate strategies ───
    all_strategies = generate_all_strategies()
    total_combos = len(all_strategies)

    type_counts = defaultdict(int)
    for s in all_strategies:
        type_counts[s['type']] += 1

    print(f"\n  Total strategy combinations: {total_combos}")
    for t in sorted(type_counts.keys()):
        print(f"    Type {t}: {type_counts[t]}")
    print(f"{'='*130}\n")

    # ─── Run sweep ───
    start_time = time.time()
    all_results = []
    tested = 0

    for strategy in all_strategies:
        result = run_strategy_test(datasets, strategy)
        tested += 1

        if result:
            all_results.append({
                'label': strategy['label'],
                'type': strategy['type'],
                'params': strategy['params'],
                'result': result,
            })

        if tested % 500 == 0:
            elapsed = time.time() - start_time
            pct = tested / total_combos * 100
            print(f"  ... {tested}/{total_combos} ({pct:.0f}%) tested in {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"\n{'='*130}")
    print(f"SWEEP COMPLETE — {tested} strategies tested in {elapsed:.0f} seconds")
    print(f"Sessions per strategy: ~{total_positions}")
    print(f"Total session simulations: ~{tested * total_positions:,}")
    print(f"{'='*130}\n")

    if not all_results:
        print("No valid results!")
        return

    # ─── Hit Rate Distribution ───
    hit_rates = [r['result']['hit_rate'] for r in all_results]
    print(f"HIT RATE DISTRIBUTION:")
    print(f"  100% hit rate:  {sum(1 for h in hit_rates if h >= 100.0)} strategies")
    print(f"  95%+ hit rate:  {sum(1 for h in hit_rates if h >= 95.0)} strategies")
    print(f"  90%+ hit rate:  {sum(1 for h in hit_rates if h >= 90.0)} strategies")
    print(f"  85%+ hit rate:  {sum(1 for h in hit_rates if h >= 85.0)} strategies")
    print(f"  80%+ hit rate:  {sum(1 for h in hit_rates if h >= 80.0)} strategies")
    print(f"  70%+ hit rate:  {sum(1 for h in hit_rates if h >= 70.0)} strategies")
    print(f"  <70% hit rate:  {sum(1 for h in hit_rates if h < 70.0)} strategies")

    # ─── 100% Hit Rate Strategies ───
    perfect = [r for r in all_results if r['result']['hit_rate'] >= 100.0]
    if perfect:
        print(f"\n{'='*130}")
        print(f"  *** 100% HIT RATE STRATEGIES — These hit +$100 EVERY session ***")
        print(f"{'='*130}")
        perfect = sorted(perfect, key=lambda x: (-x['result']['median_profit'], x['result']['avg_spins']))
        for i, r in enumerate(perfect, 1):
            res = r['result']
            print(f"\n  #{i} [{r['type']}] {r['label']}")
            print(f"      Hit: 100% ({res['total_sessions']} sessions) | "
                  f"Median: ${res['median_profit']} | Avg: ${res['avg_profit']} | "
                  f"Worst DD: ${res['worst_drawdown']} | Avg Spins: {res['avg_spins']}")
    else:
        print(f"\n  No strategy achieved 100% hit rate.")
        # Show 95%+ if no 100%
        near_perfect = [r for r in all_results if r['result']['hit_rate'] >= 95.0]
        if near_perfect:
            print(f"  {len(near_perfect)} strategies achieved 95%+ hit rate.")

    # ─── Combined Score ───
    max_median = max(r['result']['median_profit'] for r in all_results)
    min_drawdown = min(r['result']['worst_drawdown'] for r in all_results)
    for r in all_results:
        hr = r['result']['hit_rate']
        mp = r['result']['median_profit']
        wd = r['result']['worst_drawdown']
        norm_median = (mp / max_median * 100) if max_median > 0 else 0
        norm_drawdown = (1 - wd / min_drawdown) * 100 if min_drawdown < 0 else 100
        r['combined_score'] = hr * 0.5 + norm_median * 0.25 + norm_drawdown * 0.25

    # ─── Sort ───
    by_hit_rate = sorted(all_results, key=lambda x: (-x['result']['hit_rate'], -x['result']['median_profit']))
    by_combined = sorted(all_results, key=lambda x: -x['combined_score'])
    by_safest_70 = sorted([r for r in all_results if r['result']['hit_rate'] >= 70.0],
                          key=lambda x: x['result']['worst_drawdown'], reverse=True)

    # ─── Print Tables ───
    print_table("TOP 50 — HIGHEST HIT RATE", by_hit_rate, 50)
    print_table("TOP 50 — BEST OVERALL (combined score)", by_combined, 50)
    if by_safest_70:
        print_table("TOP 30 — SAFEST (smallest drawdown, 70%+ hit rate)", by_safest_70, 30)

    # ─── Best per type ───
    print(f"\n{'='*130}")
    print(f"  BEST STRATEGY PER TYPE (by combined score)")
    print(f"{'='*130}")
    types_seen = set()
    for r in by_combined:
        t = r['type']
        if t not in types_seen:
            types_seen.add(t)
            res = r['result']
            print(f"\n  [{t}] {r['label']}")
            print(f"      Hit: {res['hit_rate']}% | Avg: ${res['avg_profit']} | "
                  f"Median: ${res['median_profit']} | Worst DD: ${res['worst_drawdown']} | "
                  f"Avg Spins: {res['avg_spins']} | Score: {r['combined_score']:.1f}")

    # ─── The Winner ───
    winner = by_combined[0]
    wr = winner['result']
    print(f"\n{'='*130}")
    print(f"RECOMMENDED STRATEGY")
    print(f"{'='*130}")
    print(f"  Type: {winner['type']}")
    print(f"  Strategy: {winner['label']}")
    print(f"  Combined Score: {winner['combined_score']:.1f}")
    print(f"")
    print(f"  Results:")
    print(f"    Target hit rate:    {wr['hit_rate']}% of sessions reach +$100")
    print(f"    Average profit:     ${wr['avg_profit']}")
    print(f"    Median profit:      ${wr['median_profit']}")
    print(f"    Best session:       ${wr['best_profit']}")
    print(f"    Worst session:      ${wr['worst_profit']}")
    print(f"    Avg spins/session:  {wr['avg_spins']}")
    print(f"    Avg win rate:       {wr['avg_win_rate']}%")
    print(f"    Worst drawdown:     ${wr['worst_drawdown']}")
    print(f"{'='*130}")


if __name__ == '__main__':
    main()
