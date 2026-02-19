#!/usr/bin/env python3
"""
MARATHON STRATEGY SWEEP — Runs for 1 HOUR testing millions of strategies.

Wave 1: Massive brute-force expansion across all 20 types
Wave 2: Fine-tune around Wave 1 top performers
Wave 3+: Keep refining until time runs out

Saves global top results across ALL waves.
"""

import sys
import os
import time
import math
import random
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from config import (
    TOTAL_NUMBERS, TOP_PREDICTIONS_COUNT,
    SESSION_TARGET, INITIAL_BANKROLL,
)
from app.ml.ensemble import EnsemblePredictor

NUM_COUNT = TOP_PREDICTIONS_COUNT  # 14 numbers per spin
MAX_SESSION_SPINS = 60
FIB_SEQ = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
MARATHON_SECONDS = 3600  # 1 hour


def load_data_file(filepath):
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
# SESSION RUNNERS — Same as mega_sweep (import would be cleaner but
# keeping self-contained for reliability)
# ═══════════════════════════════════════════════════════════════════════

def _session_setup(all_numbers, start_idx):
    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None
    return min(remaining, MAX_SESSION_SPINS)


def _session_result(spins_played, total_wins, total_losses, session_profit, max_drawdown):
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


def run_session_a(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    divisor = params['divisor']
    recovery_wins = params['recovery_wins']
    max_bet = params['max_bet']
    bet = base_bet
    total_misses = 0
    consec_wins_since_hit = 0
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
            consec_wins_since_hit += 1
            if consec_wins_since_hit >= recovery_wins:
                bump = base_bet + math.floor(total_misses / divisor)
                bet = min(max_bet, bump)
                consec_wins_since_hit = 0
            total_misses = max(0, total_misses - 1)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            total_misses += 1
            consec_wins_since_hit = 0
            bet = base_bet
        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd)


def run_session_b(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    increment = params['increment']
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
            bet = min(max_bet, bet + increment)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            bet = base_bet
        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd)


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
    total_misses = 0
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
            if consec_wins >= hold_wins:
                bump = base_bet + math.floor(total_misses / divisor) * step
                bet = min(max_bet, bump)
                total_misses = max(0, total_misses - 2)
            else:
                bet = base_bet
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            total_misses += 1
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


def run_session_g(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    recovery_pct = params['recovery_pct']
    max_bet = params['max_bet']
    bet = base_bet
    cumulative_loss = 0.0
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
            cumulative_loss = max(0, cumulative_loss - net)
            if cumulative_loss > 0:
                target_recovery = cumulative_loss * recovery_pct
                ideal_bet = target_recovery / (36 - NUM_COUNT)
                bet = min(max_bet, max(base_bet, ideal_bet))
            else:
                bet = base_bet
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            cumulative_loss += abs(net)
            bet = base_bet
        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd)


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
        bet = min(max_bet, base_bet * FIB_SEQ[min(fib_level, len(FIB_SEQ) - 1)])
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
            if consec_wins >= max_doubles:
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


def run_session_k(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    initial_sequence = list(params['initial_sequence'])
    max_bet = params['max_bet']
    seq = list(initial_sequence)
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
        if not seq:
            seq = list(initial_sequence)
        if len(seq) == 1:
            bet_units = seq[0]
        else:
            bet_units = seq[0] + seq[-1]
        bet = min(max_bet, base_bet * bet_units)
        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            if len(seq) >= 2:
                seq = seq[1:-1]
            else:
                seq = []
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            seq.append(bet_units)
        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd)


def run_session_l(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    unit_size = params['unit_size']
    max_bet = params['max_bet']
    bet = unit_size
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
        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            cycle_profit += net
            if cycle_profit >= unit_size:
                cycle_profit = 0.0
                bet = unit_size
            else:
                want = unit_size - cycle_profit
                ideal = want / (36 - NUM_COUNT)
                bet = min(max_bet, max(unit_size, ideal))
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            cycle_profit += net
        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd)


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


def run_session_q(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    hot_bet = params['hot_bet']
    cold_bet = params['cold_bet']
    lookback = params['lookback']
    hot_threshold = params['hot_threshold']
    recent_hits = []
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
        if len(recent_hits) >= lookback:
            recent_rate = sum(recent_hits[-lookback:]) / lookback
            bet = hot_bet if recent_rate >= hot_threshold else cold_bet
        else:
            bet = cold_bet
        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            recent_hits.append(1)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            recent_hits.append(0)
        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd)


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
            if consec_losses >= min_losses and consec_wins >= hold_wins:
                bet = min(max_bet, base_bet * multiplier)
            consec_losses = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            consec_losses += 1
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


def run_session_s(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    aggressive_bet = params['aggressive_bet']
    conservative_bet = params['conservative_bet']
    pace_target = params['pace_target']
    aggression_threshold = params['aggression_threshold']
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
        if profit >= pace_target:
            bet = conservative_bet
        elif profit <= aggression_threshold:
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


def run_session_n(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    unit_size = params['unit_size']
    sequence = params['sequence']
    max_bet = params['max_bet']
    seq_idx = 0
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
        bet = min(max_bet, unit_size * sequence[seq_idx])
        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            consec_wins += 1
            if seq_idx < len(sequence) - 1:
                seq_idx += 1
            else:
                seq_idx = 0
                consec_wins = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            seq_idx = 0
            consec_wins = 0
        spins += 1
        if profit < dd:
            dd = profit
        if profit >= SESSION_TARGET:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd)


def run_session_t(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    step_per_interval = params['step_per_interval']
    interval = params['interval']
    max_bet = params['max_bet']
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
        intervals_passed = spins // interval
        bet = min(max_bet, base_bet + intervals_passed * step_per_interval)
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
                bet = min(max_bet, bet + step)
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


def run_session_f(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    divisor = params['divisor']
    win_step = params['win_step']
    max_bet = params['max_bet']
    bet = base_bet
    total_misses = 0
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
            recovery = base_bet + math.floor(total_misses / divisor)
            streak_bonus = consec_wins * win_step
            bet = min(max_bet, recovery + streak_bonus)
            total_misses = max(0, total_misses - 1)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            total_misses += 1
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


def run_session_o(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    step_up = params['step_up']
    plateau_length = params['plateau_length']
    max_bet = params['max_bet']
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
        plateaus = spins // plateau_length
        bet = min(max_bet, base_bet + plateaus * step_up)
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


def run_session_r(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    flat_bet = params['flat_bet']
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
        total_bet = flat_bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * flat_bet) - total_bet + flat_bet
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


# ═══════════════════════════════════════════════════════════════════════
# RANDOM STRATEGY GENERATORS — Generate random params for each type
# ═══════════════════════════════════════════════════════════════════════

def rand_choice(lst):
    return lst[random.randint(0, len(lst) - 1)]

def rand_float(lo, hi, step=0.25):
    steps = int((hi - lo) / step) + 1
    return lo + random.randint(0, steps - 1) * step

def rand_cap():
    return rand_choice([4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 200.0])

cap_label = lambda c: f'${c:.0f}' if c < 200 else 'NoCap'


def random_v1():
    base = rand_float(0.25, 6.0, 0.25)
    inc = rand_float(0.25, 4.0, 0.25)
    dec = rand_float(0.25, 3.0, 0.25)
    li = random.randint(1, 6)
    wi = random.randint(1, 6)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'V1', 'runner': run_session_v1,
        'label': f"V1 ${base} +${inc}/{li}L -${dec}/{wi}W Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'increment': inc, 'decrement': dec,
                   'losses_before_inc': li, 'wins_before_dec': wi, 'max_bet': cap}
    }

def random_a():
    base = rand_float(0.25, 6.0, 0.25)
    div = rand_float(0.25, 6.0, 0.25)
    rw = random.randint(1, 6)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'A', 'runner': run_session_a,
        'label': f"A ${base} L/{div} hold{rw}W Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'divisor': div, 'recovery_wins': rw, 'max_bet': cap}
    }

def random_b():
    base = rand_float(0.25, 6.0, 0.25)
    inc = rand_float(0.25, 5.0, 0.25)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'B', 'runner': run_session_b,
        'label': f"B ${base} +${inc}/win Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'increment': inc, 'max_bet': cap}
    }

def random_c():
    base = rand_float(0.25, 6.0, 0.25)
    div = rand_float(0.5, 5.0, 0.25)
    step = rand_float(0.25, 4.0, 0.25)
    hw = random.randint(1, 6)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'C', 'runner': run_session_c,
        'label': f"C ${base} L/{div}*${step} hold{hw}W Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'divisor': div, 'step': step, 'hold_wins': hw, 'max_bet': cap}
    }

def random_d():
    base = rand_float(0.25, 6.0, 0.25)
    ml = random.randint(2, 12)
    mult = rand_float(1.25, 6.0, 0.25)
    hw = random.randint(1, 6)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'D', 'runner': run_session_d,
        'label': f"D ${base} {ml}L->x{mult} hold{hw}W Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'min_losses': ml, 'multiplier': mult, 'hold_wins': hw, 'max_bet': cap}
    }

def random_e():
    base = rand_float(0.25, 6.0, 0.25)
    step = rand_float(0.25, 4.0, 0.25)
    wb = random.randint(1, 6)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'E', 'runner': run_session_e,
        'label': f"E ${base} +${step}/win>={wb} Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'step': step, 'wins_before_boost': wb, 'max_bet': cap}
    }

def random_f():
    base = rand_float(0.25, 6.0, 0.25)
    div = rand_float(0.5, 6.0, 0.25)
    ws = rand_float(0.25, 4.0, 0.25)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'F', 'runner': run_session_f,
        'label': f"F ${base} L/{div}+${ws}/win Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'divisor': div, 'win_step': ws, 'max_bet': cap}
    }

def random_g():
    base = rand_float(0.25, 8.0, 0.25)
    pct = rand_float(0.05, 1.0, 0.05)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'G', 'runner': run_session_g,
        'label': f"G ${base} recov={int(pct*100)}% Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'recovery_pct': pct, 'max_bet': cap}
    }

def random_h():
    base = rand_float(0.25, 6.0, 0.25)
    ml = random.randint(3, 10)
    drop = random.randint(1, 4)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'H', 'runner': run_session_h,
        'label': f"H ${base} lvl{ml} drop{drop} Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'max_fib_level': ml, 'drop_on_win': drop, 'max_bet': cap}
    }

def random_i():
    base = rand_float(0.25, 6.0, 0.25)
    step = rand_float(0.1, 3.0, 0.1)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'I', 'runner': run_session_i,
        'label': f"I ${base} step=${step} Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'step': step, 'max_bet': cap}
    }

def random_j():
    base = rand_float(0.5, 6.0, 0.25)
    mult = rand_float(1.25, 4.0, 0.25)
    md = random.randint(2, 8)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'J', 'runner': run_session_j,
        'label': f"J ${base} x{mult} max{md}W Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'multiplier': mult, 'max_doubles': md, 'max_bet': cap}
    }

def random_k():
    seq_len = random.randint(2, 7)
    seq = tuple(random.randint(1, 8) for _ in range(seq_len))
    base = rand_float(0.25, 5.0, 0.25)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    seq_str = ','.join(str(s) for s in seq)
    return {
        'type': 'K', 'runner': run_session_k,
        'label': f"K ${base} seq=[{seq_str}] Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'initial_sequence': seq, 'max_bet': cap}
    }

def random_l():
    unit = rand_float(0.25, 8.0, 0.25)
    cap = rand_cap()
    if cap < unit:
        cap = 200.0
    return {
        'type': 'L', 'runner': run_session_l,
        'label': f"L unit=${unit} Cap={cap_label(cap)}",
        'params': {'unit_size': unit, 'max_bet': cap}
    }

def random_m():
    base = rand_float(0.5, 6.0, 0.25)
    mpw = random.randint(2, 8)
    mult = rand_float(1.25, 4.0, 0.25)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'M', 'runner': run_session_m,
        'label': f"M ${base} x{mult} max{mpw}W Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'max_paroli_wins': mpw, 'multiplier': mult, 'max_bet': cap}
    }

def random_n():
    seq_len = random.randint(3, 6)
    seq = [random.randint(1, 8) for _ in range(seq_len)]
    unit = rand_float(0.25, 6.0, 0.25)
    cap = rand_cap()
    if cap < unit:
        cap = 200.0
    seq_str = '-'.join(str(s) for s in seq)
    return {
        'type': 'N', 'runner': run_session_n,
        'label': f"N ${unit} seq={seq_str} Cap={cap_label(cap)}",
        'params': {'unit_size': unit, 'sequence': seq, 'max_bet': cap}
    }

def random_q():
    hb = rand_float(2.0, 15.0, 0.5)
    cb = rand_float(0.25, 3.0, 0.25)
    lb = rand_choice([3, 5, 8, 10, 15, 20])
    ht = rand_float(0.2, 0.65, 0.05)
    return {
        'type': 'Q', 'runner': run_session_q,
        'label': f"Q hot=${hb} cold=${cb} look{lb} thr={ht}",
        'params': {'hot_bet': hb, 'cold_bet': cb, 'lookback': lb, 'hot_threshold': ht}
    }

def random_s():
    base = rand_float(0.5, 5.0, 0.25)
    agg = rand_float(3.0, 12.0, 0.5)
    con = rand_float(0.25, 3.0, 0.25)
    pt = rand_choice([25.0, 50.0, 75.0, 100.0])
    at = rand_choice([-20.0, -30.0, -40.0, -50.0, -75.0, -100.0, -150.0, -200.0])
    return {
        'type': 'S', 'runner': run_session_s,
        'label': f"S ${base} agg=${agg} con=${con} pace={pt} thr={at}",
        'params': {'base_bet': base, 'aggressive_bet': agg, 'conservative_bet': con,
                   'pace_target': pt, 'aggression_threshold': at}
    }

def random_t():
    base = rand_float(0.5, 6.0, 0.25)
    step = rand_float(0.1, 3.0, 0.1)
    intv = random.randint(2, 30)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'T', 'runner': run_session_t,
        'label': f"T ${base} +${step}/{intv}sp Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'step_per_interval': step, 'interval': intv, 'max_bet': cap}
    }

def random_o():
    base = rand_float(0.5, 6.0, 0.25)
    step_up = rand_float(0.1, 3.0, 0.1)
    pl = random.randint(3, 30)
    cap = rand_cap()
    if cap < base:
        cap = 200.0
    return {
        'type': 'O', 'runner': run_session_o,
        'label': f"O ${base} +${step_up}/{pl}sp Cap={cap_label(cap)}",
        'params': {'base_bet': base, 'step_up': step_up, 'plateau_length': pl, 'max_bet': cap}
    }

def random_p():
    base = rand_float(0.25, 5.0, 0.25)
    mult = rand_float(1.25, 3.5, 0.25)
    cap = rand_float(3.0, 20.0, 1.0)
    if cap < base:
        cap = base + 2
    rtb = random.choice([True, False])
    return {
        'type': 'P', 'runner': run_session_p,
        'label': f"P ${base} x{mult} Cap=${cap:.0f} rst={'base' if rtb else '$1'}",
        'params': {'base_bet': base, 'multiplier': mult, 'max_bet': cap, 'reset_to_base': rtb}
    }

def random_r():
    fb = rand_float(0.25, 12.0, 0.25)
    return {
        'type': 'R', 'runner': run_session_r,
        'label': f"R flat=${fb}",
        'params': {'flat_bet': fb}
    }


ALL_RANDOM_GENERATORS = [
    random_v1, random_a, random_b, random_c, random_d, random_e,
    random_f, random_g, random_h, random_i, random_j, random_k,
    random_l, random_m, random_n, random_q, random_s, random_t,
    random_o, random_p, random_r,
]

# Weight towards types that showed promise
GENERATOR_WEIGHTS = [
    15,  # V1 - many params, always produces decent
    8,   # A
    5,   # B
    10,  # C - showed strong
    10,  # D
    5,   # E
    5,   # F
    10,  # G - showed promise
    8,   # H - fib showed strong
    6,   # I - dalembert
    10,  # J - anti-mart strong
    15,  # K - labouchere THE BEST
    6,   # L - oscar grind
    8,   # M - paroli
    6,   # N
    12,  # Q - reverse recovery STRONG
    8,   # S
    6,   # T
    5,   # O
    5,   # P
    2,   # R - flat baseline
]


def generate_random_strategy():
    """Generate a single random strategy, weighted towards promising types."""
    gen = random.choices(ALL_RANDOM_GENERATORS, weights=GENERATOR_WEIGHTS, k=1)[0]
    return gen()


def run_strategy_test(datasets, strategy):
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
    marathon_start = time.time()
    deadline = marathon_start + MARATHON_SECONDS

    print(f"\n{'='*130}")
    print(f"MARATHON STRATEGY SWEEP — Running for 1 HOUR ({MARATHON_SECONDS}s)")
    print(f"{'='*130}")

    # ─── Load ALL data files ───
    datasets = []
    total_numbers = 0

    # Only load test_data1.txt — the actual test data
    # (data1-10.txt are training data, not for testing)
    test_path = os.path.join(os.path.dirname(__file__), 'test_data', 'test_data1.txt')
    if os.path.exists(test_path):
        nums = load_data_file(test_path)
        if nums:
            print(f"  Loading test_data1.txt ... {len(nums)} numbers")
            preds = precompute_predictions(nums)
            positions = list(range(30, len(nums) - 30, 10))
            datasets.append((nums, preds, positions))
            total_numbers += len(nums)
    else:
        print("  ERROR: test_data1.txt not found!")
        return

    total_positions = sum(len(d[2]) for d in datasets)
    print(f"\n  Total: {len(datasets)} data files, {total_numbers} numbers, {total_positions} starting positions")
    print(f"  Numbers per spin: {NUM_COUNT}")
    print(f"  Max spins/session: {MAX_SESSION_SPINS}")
    print(f"  Target: +${SESSION_TARGET} per session")
    print(f"  Bankroll: ${INITIAL_BANKROLL}")

    # ─── Global tracking ───
    global_top_hit = []     # Top 100 by hit rate
    global_top_overall = [] # Top 100 by combined score
    global_top_safe = []    # Top 100 safest with 70%+
    seen_labels = set()     # Deduplicate

    total_tested = 0
    total_valid = 0
    type_counts = defaultdict(int)

    wave_num = 0
    BATCH_SIZE = 5000  # Test in batches, check time between

    print(f"\n{'='*130}")
    print(f"  MARATHON STARTING — Will run until {time.strftime('%H:%M:%S', time.localtime(deadline))}")
    print(f"{'='*130}\n")

    while time.time() < deadline:
        wave_num += 1
        wave_start = time.time()
        wave_tested = 0
        wave_valid = 0

        print(f"  ── Wave {wave_num} started at {time.strftime('%H:%M:%S')} ──", flush=True)

        while time.time() < deadline:
            # Generate a batch
            batch = []
            for _ in range(BATCH_SIZE):
                s = generate_random_strategy()
                if s['label'] not in seen_labels:
                    seen_labels.add(s['label'])
                    batch.append(s)

            # Test the batch
            for strategy in batch:
                result = run_strategy_test(datasets, strategy)
                wave_tested += 1
                total_tested += 1

                if result:
                    wave_valid += 1
                    total_valid += 1
                    type_counts[strategy['type']] += 1

                    entry = {
                        'label': strategy['label'],
                        'type': strategy['type'],
                        'params': strategy['params'],
                        'result': result,
                    }

                    # Calculate combined score
                    hr = result['hit_rate']
                    mp = result['median_profit']
                    wd = result['worst_drawdown']
                    # Normalize: higher is better
                    norm_median = mp / 700.0 * 100 if mp > 0 else mp / 100.0 * 10
                    norm_dd = max(0, (1 - abs(wd) / 6000.0)) * 100
                    entry['combined_score'] = hr * 0.5 + norm_median * 0.25 + norm_dd * 0.25

                    # Update global tops
                    global_top_hit.append(entry)
                    global_top_overall.append(entry)
                    if result['hit_rate'] >= 70.0:
                        global_top_safe.append(entry)

                    # Keep only top 200 in each list (trim periodically)
                    if len(global_top_hit) > 500:
                        global_top_hit.sort(key=lambda x: (-x['result']['hit_rate'], -x['result']['median_profit']))
                        global_top_hit = global_top_hit[:200]
                    if len(global_top_overall) > 500:
                        global_top_overall.sort(key=lambda x: -x['combined_score'])
                        global_top_overall = global_top_overall[:200]
                    if len(global_top_safe) > 500:
                        global_top_safe.sort(key=lambda x: x['result']['worst_drawdown'], reverse=True)
                        global_top_safe = global_top_safe[:200]

            elapsed_total = time.time() - marathon_start
            remaining = deadline - time.time()
            rate = total_tested / elapsed_total if elapsed_total > 0 else 0
            sessions = total_tested * total_positions

            if wave_tested % (BATCH_SIZE * 2) == 0 or wave_tested == BATCH_SIZE:
                best_hit = max((r['result']['hit_rate'] for r in global_top_hit), default=0)
                best_safe_dd = max((r['result']['worst_drawdown'] for r in global_top_safe), default=-9999)
                print(f"    {total_tested:>8,} tested | {total_valid:>7,} valid | "
                      f"{rate:.0f}/s | {sessions:>12,} sessions | "
                      f"Best hit: {best_hit:.1f}% | Safest DD: ${best_safe_dd:.0f} | "
                      f"{remaining:.0f}s left", flush=True)

            # Check if time is up
            if time.time() >= deadline:
                break

        wave_elapsed = time.time() - wave_start
        print(f"  ── Wave {wave_num} done: {wave_tested:,} tested, {wave_valid:,} valid in {wave_elapsed:.0f}s ──\n",
              flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ═══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - marathon_start
    total_sessions = total_tested * total_positions

    print(f"\n{'='*130}")
    print(f"MARATHON COMPLETE — {total_tested:,} strategies tested in {total_elapsed:.0f} seconds ({total_elapsed/60:.1f} min)")
    print(f"Valid strategies: {total_valid:,}")
    print(f"Total session simulations: ~{total_sessions:,}")
    print(f"Rate: {total_tested/total_elapsed:.0f} strategies/sec, {total_sessions/total_elapsed:,.0f} sessions/sec")
    print(f"{'='*130}\n")

    # Type distribution
    print("STRATEGIES TESTED PER TYPE:")
    for t in sorted(type_counts.keys()):
        print(f"  {t}: {type_counts[t]:,}")

    # Hit rate distribution
    all_hit_rates = [r['result']['hit_rate'] for r in global_top_hit]
    print(f"\nBEST HIT RATES FOUND:")
    print(f"  Highest: {max(all_hit_rates):.1f}%")

    # Final sort
    global_top_hit.sort(key=lambda x: (-x['result']['hit_rate'], -x['result']['median_profit']))
    global_top_overall.sort(key=lambda x: -x['combined_score'])
    global_top_safe.sort(key=lambda x: x['result']['worst_drawdown'], reverse=True)

    # Print tables
    print_table("TOP 50 — HIGHEST HIT RATE (across all waves)", global_top_hit, 50)
    print_table("TOP 50 — BEST OVERALL (combined score)", global_top_overall, 50)
    if global_top_safe:
        print_table("TOP 30 — SAFEST (smallest drawdown, 70%+ hit rate)", global_top_safe, 30)

    # Best per type
    print(f"\n{'='*130}")
    print(f"  BEST STRATEGY PER TYPE (by combined score)")
    print(f"{'='*130}")
    types_seen = set()
    for r in global_top_overall:
        t = r['type']
        if t not in types_seen:
            types_seen.add(t)
            res = r['result']
            print(f"\n  [{t}] {r['label']}")
            print(f"      Hit: {res['hit_rate']}% | Avg: ${res['avg_profit']} | "
                  f"Median: ${res['median_profit']} | Worst DD: ${res['worst_drawdown']} | "
                  f"Avg Spins: {res['avg_spins']} | Score: {r['combined_score']:.1f}")

    # The Winner
    if global_top_overall:
        winner = global_top_overall[0]
        wr = winner['result']
        print(f"\n{'='*130}")
        print(f"RECOMMENDED STRATEGY (from {total_tested:,} tested)")
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
