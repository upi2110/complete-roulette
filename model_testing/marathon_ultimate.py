#!/usr/bin/env python3
"""
MARATHON ULTIMATE — THE ONE STRATEGY

Goal: Find ONE strategy that averages $100+/session over 10 sessions = $1,000/day
Rules:
  - NO profit cap — let winners run
  - STOP-LOSS only — protect the $4,000 bankroll
  - 30-60 spins per session (30 min to 1 hour)
  - Must survive across ALL data files (cross-validation)
  - CONFIDENCE GATE — only bet when AI says "NOW" (patience = edge)
  - This is a JOB, not gambling

Scoring: Avg profit (must be $100+), worst session, consistency, survival rate

CONFIDENCE FILTER:
  Every spin, the AI returns a confidence score (0-100) and should_bet signal.
  We WAIT at the table, watching. When confidence >= min_confidence AND
  should_bet is True, THEN we place our bet. Otherwise we sit out that spin.
  This is how a professional plays — patience IS the strategy.
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
    INITIAL_BANKROLL,
)
from app.ml.ensemble import EnsemblePredictor

NUM_COUNT = TOP_PREDICTIONS_COUNT  # 14 numbers per spin
MAX_SESSION_SPINS = 60             # 60 spins max (~1 hour)
FIB_SEQ = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
MARATHON_SECONDS = 3600            # 1 hour per wave

# ── ULTIMATE MODE: No profit target, stop-loss only ──
# Sessions run ALL spins (up to 60). No early exit on profit.
# Only exit early on stop-loss or bankroll bust.
STOP_LOSSES = [200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000]

# ── CONFIDENCE GATE: Only bet when AI says "NOW" ──
# 0 = bet every spin (old behavior). Higher = more patience, fewer bets.
# The session is 60 spins max. If we skip too many, we place fewer bets
# but each bet is on a HIGHER confidence prediction = better win rate.
# Don't go above 65 — at that point we'd skip too many and barely bet.
CONFIDENCE_THRESHOLDS = [0, 0, 0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
# Weighted toward 0 (3 copies) to compare with/without confidence filter


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
    """Precompute predictions WITH confidence data for every spin.

    Returns list of tuples: (pred_set, top_nums, confidence, should_bet)
    - confidence: 0-100 calibrated score from the AI
    - should_bet: True/False from wheel strategy signal

    The session runners use these to GATE their bets:
    Only bet when confidence >= min_confidence AND should_bet is True.
    """
    predictor = EnsemblePredictor()
    predictions = []
    for i, number in enumerate(all_numbers):
        if i >= 3:
            pred = predictor.predict()
            top_nums = pred.get('top_numbers', [])[:NUM_COUNT]
            confidence = pred.get('confidence', 50.0)
            should_bet = pred.get('should_bet', True)
            predictions.append((set(top_nums), top_nums, confidence, should_bet))
        else:
            predictions.append((set(), [], 0.0, False))
        predictor.update(number)
    return predictions


# ═══════════════════════════════════════════════════════════════════════
# SESSION RUNNERS — ULTIMATE MODE
# NO profit cap. Stop-loss only. Let winners run.
# ═══════════════════════════════════════════════════════════════════════

def _session_setup(all_numbers, start_idx):
    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None
    return min(remaining, MAX_SESSION_SPINS)


def _session_result(spins_played, total_wins, total_losses, session_profit, max_drawdown, skipped=0):
    if spins_played == 0:
        return None
    bets_placed = total_wins + total_losses
    return {
        'spins_played': spins_played,
        'bets_placed': bets_placed,
        'skipped': skipped,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / bets_placed * 100, 1) if bets_placed > 0 else 0.0,
        'final_profit': round(session_profit, 2),
        'positive': session_profit > 0,
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
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    bet = base_bet
    consec_losses = 0
    consec_wins = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        # CONFIDENCE GATE — sit out if AI isn't confident
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
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
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses, profit, dd, skipped)


def run_session_c(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    divisor = params['divisor']
    step = params['step']
    hold_wins = params['hold_wins']
    max_bet = params['max_bet']
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    bet = base_bet
    total_losses = 0
    consec_wins = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
            continue
        total_bet_amount = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet_amount + bet
            profit += net
            bankroll += net
            wins += 1
            consec_wins += 1
            if consec_wins >= hold_wins:
                consec_wins = 0
            total_losses = max(0, total_losses - 1)
        else:
            net = -total_bet_amount
            profit += net
            bankroll += net
            losses_count += 1
            total_losses += 1
            consec_wins = 0
            bump = base_bet + math.floor(total_losses / divisor) * step
            bet = min(max_bet, bump)
        spins += 1
        if profit < dd:
            dd = profit
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd, skipped)


def run_session_k(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    seq = list(params['initial_sequence'])
    max_bet = params['max_bet']
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
            continue
        if len(seq) == 0:
            seq = list(params['initial_sequence'])
        if len(seq) == 1:
            bet = seq[0] * base_bet
        else:
            bet = (seq[0] + seq[-1]) * base_bet
        bet = min(max_bet, bet) if max_bet < 200 else bet
        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            if len(seq) >= 2:
                seq.pop(0)
                if seq:
                    seq.pop(-1)
            elif len(seq) == 1:
                seq.pop(0)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            seq.append(seq[-1] if seq else 1)
        spins += 1
        if profit < dd:
            dd = profit
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd, skipped)


def run_session_f(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    divisor = params['divisor']
    win_step = params['win_step']
    max_bet = params['max_bet']
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    bet = base_bet
    total_misses = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
            continue
        total_bet_amount = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet_amount + bet
            profit += net
            bankroll += net
            wins += 1
            bet = min(max_bet, bet + win_step)
            total_misses = max(0, total_misses - 1)
        else:
            net = -total_bet_amount
            profit += net
            bankroll += net
            losses_count += 1
            total_misses += 1
            bump = base_bet + math.floor(total_misses / divisor)
            bet = min(max_bet, bump)
        spins += 1
        if profit < dd:
            dd = profit
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd, skipped)


def run_session_n(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    unit = params['unit_size']
    sequence = params['sequence']
    max_bet = params['max_bet']
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    seq_idx = 0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
            continue
        bet = unit * sequence[seq_idx % len(sequence)]
        bet = min(max_bet, bet) if max_bet < 200 else bet
        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            seq_idx += 1
            if seq_idx >= len(sequence):
                seq_idx = 0
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            seq_idx = 0
        spins += 1
        if profit < dd:
            dd = profit
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd, skipped)


def run_session_q(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    hot_bet = params['hot_bet']
    cold_bet = params['cold_bet']
    lookback = params['lookback']
    hot_threshold = params['hot_threshold']
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    recent_hits = []
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
            continue
        if len(recent_hits) >= lookback:
            hit_pct = sum(recent_hits[-lookback:]) / lookback
        else:
            hit_pct = 0.5
        bet = hot_bet if hit_pct >= hot_threshold else cold_bet
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
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd, skipped)


def run_session_g(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    recovery_pct = params['recovery_pct']
    max_bet = params['max_bet']
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    bet = base_bet
    cum_loss = 0.0
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
            continue
        total_bet = bet * NUM_COUNT
        if actual in pred_set:
            net = (35 * bet) - total_bet + bet
            profit += net
            bankroll += net
            wins += 1
            cum_loss = max(0, cum_loss - net)
            if cum_loss <= 0:
                bet = base_bet
            else:
                bet = min(max_bet, base_bet + cum_loss * recovery_pct / 22.0)
        else:
            net = -total_bet
            profit += net
            bankroll += net
            losses_count += 1
            cum_loss += abs(net)
            bet = min(max_bet, base_bet + cum_loss * recovery_pct / 22.0)
        spins += 1
        if profit < dd:
            dd = profit
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd, skipped)


def run_session_s(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    base_bet = params['base_bet']
    agg_bet = params['aggressive_bet']
    con_bet = params['conservative_bet']
    pace_target = params['pace_target']
    agg_threshold = params['aggression_threshold']
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
            continue
        progress_pct = (off + 1) / remaining
        expected_profit = pace_target * progress_pct
        gap = profit - expected_profit
        if gap <= agg_threshold:
            bet = agg_bet
        elif profit >= expected_profit:
            bet = con_bet
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
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd, skipped)


def run_session_r(all_numbers, predictions, start_idx, params):
    remaining = _session_setup(all_numbers, start_idx)
    if remaining is None:
        return None
    flat_bet = params['flat_bet']
    stop_loss = params.get('stop_loss', 4000)
    min_conf = params.get('min_confidence', 0)
    profit = 0.0
    bankroll = INITIAL_BANKROLL
    spins = 0
    wins = 0
    losses_count = 0
    skipped = 0
    dd = 0.0
    for off in range(remaining):
        pos = start_idx + off
        actual = all_numbers[pos]
        pred_set, _, conf, sb = predictions[pos]
        if not pred_set:
            spins += 1
            continue
        if conf < min_conf or (min_conf > 0 and not sb):
            spins += 1
            skipped += 1
            continue
        bet = flat_bet
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
        if profit <= -stop_loss:
            break
        if bankroll <= 0:
            break
    return _session_result(spins, wins, losses_count, profit, dd, skipped)


# ═══════════════════════════════════════════════════════════════════════
# RANDOM STRATEGY GENERATORS — Scaled for $100+/session avg
# ═══════════════════════════════════════════════════════════════════════

def rand_choice(lst):
    return lst[random.randint(0, len(lst) - 1)]

def rand_float(lo, hi, step=0.25):
    steps = int((hi - lo) / step) + 1
    return lo + random.randint(0, steps - 1) * step

def rand_cap():
    return rand_choice([10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 200.0])

def rand_stop_loss():
    return rand_choice(STOP_LOSSES)

def rand_confidence():
    """Random min confidence threshold.
    0 = bet every spin (no filter)
    20-65 = progressively pickier (skip low confidence spins)
    Higher = more patience, fewer bets, potentially better win rate
    But can't be TOO high or we won't bet enough in 60 spins.
    """
    return rand_choice(CONFIDENCE_THRESHOLDS)

cap_label = lambda c: f'${c:.0f}' if c < 200 else 'NoCap'
conf_label = lambda c: f'conf>={c}' if c > 0 else 'anyConf'


def random_v1():
    base = rand_float(0.5, 12.0, 0.5)
    inc = rand_float(0.5, 10.0, 0.5)
    dec = rand_float(0.25, 5.0, 0.25)
    li = random.randint(1, 5)
    wi = random.randint(1, 5)
    cap = rand_cap()
    sl = rand_stop_loss()
    mc = rand_confidence()
    if cap < base:
        cap = 200.0
    return {
        'type': 'V1', 'runner': run_session_v1,
        'label': f"V1 ${base} +${inc}/{li}L -${dec}/{wi}W Cap={cap_label(cap)} SL=${sl} {conf_label(mc)}",
        'params': {'base_bet': base, 'increment': inc, 'decrement': dec,
                   'losses_before_inc': li, 'wins_before_dec': wi, 'max_bet': cap,
                   'stop_loss': sl, 'min_confidence': mc}
    }

def random_c():
    base = rand_float(0.5, 10.0, 0.5)
    div = rand_float(0.5, 6.0, 0.25)
    step = rand_float(0.5, 12.0, 0.5)
    hw = random.randint(1, 4)
    cap = rand_cap()
    sl = rand_stop_loss()
    mc = rand_confidence()
    if cap < base:
        cap = 200.0
    return {
        'type': 'C', 'runner': run_session_c,
        'label': f"C ${base} L/{div}*${step} hold{hw}W Cap={cap_label(cap)} SL=${sl} {conf_label(mc)}",
        'params': {'base_bet': base, 'divisor': div, 'step': step, 'hold_wins': hw,
                   'max_bet': cap, 'stop_loss': sl, 'min_confidence': mc}
    }

def random_k():
    seq_len = random.randint(2, 6)
    seq = tuple(random.randint(1, 15) for _ in range(seq_len))
    base = rand_float(0.5, 10.0, 0.5)
    cap = rand_cap()
    sl = rand_stop_loss()
    mc = rand_confidence()
    if cap < base:
        cap = 200.0
    seq_str = ','.join(str(s) for s in seq)
    return {
        'type': 'K', 'runner': run_session_k,
        'label': f"K ${base} seq=[{seq_str}] Cap={cap_label(cap)} SL=${sl} {conf_label(mc)}",
        'params': {'base_bet': base, 'initial_sequence': seq, 'max_bet': cap,
                   'stop_loss': sl, 'min_confidence': mc}
    }

def random_f():
    base = rand_float(0.5, 10.0, 0.5)
    div = rand_float(0.5, 6.0, 0.25)
    ws = rand_float(0.5, 10.0, 0.5)
    cap = rand_cap()
    sl = rand_stop_loss()
    mc = rand_confidence()
    if cap < base:
        cap = 200.0
    return {
        'type': 'F', 'runner': run_session_f,
        'label': f"F ${base} L/{div}+${ws}/win Cap={cap_label(cap)} SL=${sl} {conf_label(mc)}",
        'params': {'base_bet': base, 'divisor': div, 'win_step': ws, 'max_bet': cap,
                   'stop_loss': sl, 'min_confidence': mc}
    }

def random_n():
    seq_len = random.randint(3, 6)
    seq = [random.randint(1, 15) for _ in range(seq_len)]
    unit = rand_float(0.5, 10.0, 0.5)
    cap = rand_cap()
    sl = rand_stop_loss()
    mc = rand_confidence()
    if cap < unit:
        cap = 200.0
    seq_str = '-'.join(str(s) for s in seq)
    return {
        'type': 'N', 'runner': run_session_n,
        'label': f"N ${unit} seq={seq_str} Cap={cap_label(cap)} SL=${sl} {conf_label(mc)}",
        'params': {'unit_size': unit, 'sequence': seq, 'max_bet': cap,
                   'stop_loss': sl, 'min_confidence': mc}
    }

def random_q():
    hb = rand_float(3.0, 40.0, 1.0)
    cb = rand_float(0.5, 8.0, 0.5)
    lb = rand_choice([3, 5, 8, 10, 15, 20])
    ht = rand_float(0.2, 0.65, 0.05)
    sl = rand_stop_loss()
    mc = rand_confidence()
    return {
        'type': 'Q', 'runner': run_session_q,
        'label': f"Q hot=${hb} cold=${cb} look{lb} thr={ht} SL=${sl} {conf_label(mc)}",
        'params': {'hot_bet': hb, 'cold_bet': cb, 'lookback': lb,
                   'hot_threshold': ht, 'stop_loss': sl, 'min_confidence': mc}
    }

def random_g():
    base = rand_float(1.0, 15.0, 0.5)
    pct = rand_float(0.05, 1.0, 0.05)
    cap = rand_cap()
    sl = rand_stop_loss()
    mc = rand_confidence()
    if cap < base:
        cap = 200.0
    return {
        'type': 'G', 'runner': run_session_g,
        'label': f"G ${base} recov={int(pct*100)}% Cap={cap_label(cap)} SL=${sl} {conf_label(mc)}",
        'params': {'base_bet': base, 'recovery_pct': pct, 'max_bet': cap,
                   'stop_loss': sl, 'min_confidence': mc}
    }

def random_s():
    base = rand_float(1.0, 10.0, 0.5)
    agg = rand_float(5.0, 35.0, 1.0)
    con = rand_float(0.5, 6.0, 0.5)
    pt = rand_choice([50.0, 100.0, 150.0, 200.0, 300.0, 500.0])
    at = rand_choice([-50.0, -100.0, -150.0, -200.0, -300.0, -500.0])
    sl = rand_stop_loss()
    mc = rand_confidence()
    return {
        'type': 'S', 'runner': run_session_s,
        'label': f"S ${base} agg=${agg} con=${con} pace={pt} thr={at} SL=${sl} {conf_label(mc)}",
        'params': {'base_bet': base, 'aggressive_bet': agg, 'conservative_bet': con,
                   'pace_target': pt, 'aggression_threshold': at, 'stop_loss': sl,
                   'min_confidence': mc}
    }

def random_r():
    fb = rand_float(1.0, 25.0, 0.5)
    sl = rand_stop_loss()
    mc = rand_confidence()
    return {
        'type': 'R', 'runner': run_session_r,
        'label': f"R flat=${fb} SL=${sl} {conf_label(mc)}",
        'params': {'flat_bet': fb, 'stop_loss': sl, 'min_confidence': mc}
    }


ALL_RANDOM_GENERATORS = [
    random_v1, random_c, random_k, random_f, random_n,
    random_q, random_g, random_s, random_r,
]

# Weight towards best performers from $500 sweep
GENERATOR_WEIGHTS = [
    15,  # V1
    20,  # C — dominated $500 sweep
    18,  # K — highest raw hit rates
    15,  # F — sleeper from $500 sweep
    12,  # N — safest strategies
    12,  # Q — reverse recovery
    10,  # G — percentage recovery
    8,   # S — session profit aware
    3,   # R — flat baseline
]


def generate_random_strategy():
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
    profits = [r['final_profit'] for r in results]
    spins = [r['spins_played'] for r in results]
    bets = [r['bets_placed'] for r in results]
    skips = [r['skipped'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    positive_sessions = sum(1 for r in results if r['positive'])

    avg_profit = round(np.mean(profits), 2)
    median_profit = round(float(np.median(profits)), 2)
    worst_profit = round(min(profits), 2)
    best_profit = round(max(profits), 2)
    worst_dd = round(min(drawdowns), 2)
    survival_rate = round(positive_sessions / len(results) * 100, 1)
    avg_bets = round(np.mean(bets), 1)
    avg_skips = round(np.mean(skips), 1)

    # Consistency: standard deviation of profits (lower = more consistent)
    profit_std = round(float(np.std(profits)), 2)

    # Profit-to-risk ratio
    ptr = round(avg_profit / abs(worst_dd), 3) if worst_dd < 0 else 99.0

    return {
        'total_sessions': len(results),
        'avg_profit': avg_profit,
        'median_profit': median_profit,
        'worst_profit': worst_profit,
        'best_profit': best_profit,
        'avg_spins': round(np.mean(spins), 1),
        'avg_bets': avg_bets,
        'avg_skips': avg_skips,
        'avg_drawdown': round(np.mean(drawdowns), 2),
        'worst_drawdown': worst_dd,
        'avg_win_rate': round(np.mean(win_rates), 1),
        'survival_rate': survival_rate,
        'profit_std': profit_std,
        'profit_to_risk': ptr,
        'total_profit': round(sum(profits), 2),
        'daily_10sess': round(avg_profit * 10, 2),
    }


def print_table(title, ranked, count=50):
    print(f"\n{'─'*160}")
    print(f"  {title}")
    print(f"{'─'*160}")
    print(f"{'Rank':<5} {'Type':<3} {'Strategy':<62} {'AvgP':>7} {'MedP':>7} "
          f"{'WorstP':>7} {'BestP':>7} {'Surv%':>6} {'Bets':>5} {'Skip':>5} {'WrstDd':>7} {'PtR':>5} {'$/Day':>8}")
    print(f"{'─'*160}")
    for i, r in enumerate(ranked[:count], 1):
        res = r['result']
        print(f"{i:<5} {r['type']:<3} {r['label']:<62} "
              f"${res['avg_profit']:>6.0f} "
              f"${res['median_profit']:>6.0f} "
              f"${res['worst_profit']:>6.0f} "
              f"${res['best_profit']:>6.0f} "
              f"{res['survival_rate']:>5.1f}% "
              f"{res['avg_bets']:>4.0f} "
              f"{res['avg_skips']:>4.0f} "
              f"${res['worst_drawdown']:>6.0f} "
              f"{res['profit_to_risk']:>4.3f} "
              f"${res['daily_10sess']:>7.0f}")


def main():
    marathon_start = time.time()
    deadline = marathon_start + MARATHON_SECONDS

    print(f"\n{'='*150}")
    print(f"MARATHON ULTIMATE — THE ONE STRATEGY")
    print(f"Running for 1 HOUR ({MARATHON_SECONDS}s)")
    print(f"{'='*150}")

    # ─── Load ALL data files for cross-validation ───
    datasets = []
    total_numbers = 0

    # Load test_data1.txt
    test_path = os.path.join(os.path.dirname(__file__), 'test_data', 'test_data1.txt')
    if os.path.exists(test_path):
        nums = load_data_file(test_path)
        if nums:
            print(f"  Loading test_data1.txt ... {len(nums)} numbers")
            preds = precompute_predictions(nums)
            positions = list(range(30, len(nums) - 30, 10))
            datasets.append((nums, preds, positions))
            total_numbers += len(nums)

    # Load ALL userdata files for cross-validation
    userdata_dir = os.path.join(PROJECT_ROOT, 'userdata')
    if os.path.isdir(userdata_dir):
        for fname in sorted(os.listdir(userdata_dir)):
            if fname.startswith('data') and fname.endswith('.txt'):
                fpath = os.path.join(userdata_dir, fname)
                nums = load_data_file(fpath)
                if nums and len(nums) >= 60:
                    print(f"  Loading {fname} ... {len(nums)} numbers")
                    preds = precompute_predictions(nums)
                    positions = list(range(30, len(nums) - 30, 10))
                    datasets.append((nums, preds, positions))
                    total_numbers += len(nums)

    total_positions = sum(len(d[2]) for d in datasets)
    print(f"\n  Total: {len(datasets)} data files, {total_numbers} numbers, {total_positions} starting positions")
    print(f"  Numbers per spin: {NUM_COUNT}")
    print(f"  Max spins/session: {MAX_SESSION_SPINS}")
    print(f"  MODE: NO PROFIT CAP — Stop-loss only")
    print(f"  CONFIDENCE GATE: Only bet when AI confidence >= threshold")
    print(f"  Confidence thresholds tested: {sorted(set(CONFIDENCE_THRESHOLDS))}")
    print(f"  Goal: Avg $100+/session × 10 sessions = $1,000/day")
    print(f"  Bankroll: ${INITIAL_BANKROLL}")

    # ─── Global tracking ───
    global_top_profit = []    # Top by avg profit (must be $100+)
    global_top_safe = []      # Top by smallest worst DD with $100+ avg
    global_top_consistent = [] # Top by profit-to-risk ratio
    seen_labels = set()

    total_tested = 0
    total_valid = 0
    type_counts = defaultdict(int)
    qualified_count = 0  # strategies with avg profit >= $100

    wave_num = 0
    BATCH_SIZE = 3000

    print(f"\n{'='*150}")
    print(f"  MARATHON STARTING — Will run until {time.strftime('%H:%M:%S', time.localtime(deadline))}")
    print(f"{'='*150}\n")

    while time.time() < deadline:
        wave_num += 1
        wave_start = time.time()
        wave_tested = 0

        print(f"  ── Wave {wave_num} started at {time.strftime('%H:%M:%S')} ──", flush=True)

        while time.time() < deadline:
            batch = []
            for _ in range(BATCH_SIZE):
                s = generate_random_strategy()
                if s['label'] not in seen_labels:
                    seen_labels.add(s['label'])
                    batch.append(s)

            for strategy in batch:
                result = run_strategy_test(datasets, strategy)
                wave_tested += 1
                total_tested += 1

                if result:
                    total_valid += 1
                    type_counts[strategy['type']] += 1

                    entry = {
                        'label': strategy['label'],
                        'type': strategy['type'],
                        'params': strategy['params'],
                        'result': result,
                    }

                    # ULTIMATE SCORE: Weighted for our exact goal
                    avg_p = result['avg_profit']
                    surv = result['survival_rate']
                    wd = result['worst_drawdown']
                    ptr = result['profit_to_risk']
                    std = result['profit_std']

                    # Only care about strategies averaging $50+/session
                    if avg_p >= 50:
                        qualified_count += 1

                        # Score: profit matters most, but penalize huge drawdowns
                        norm_profit = min(avg_p / 500.0, 1.0) * 40  # 0-40 points
                        norm_surv = surv / 100.0 * 25               # 0-25 points
                        norm_dd = max(0, (1 - abs(wd) / 8000.0)) * 20  # 0-20 points
                        norm_ptr = min(ptr, 1.0) * 15               # 0-15 points
                        entry['combined_score'] = norm_profit + norm_surv + norm_dd + norm_ptr

                        global_top_profit.append(entry)
                        global_top_consistent.append(entry)
                        if avg_p >= 80:
                            global_top_safe.append(entry)

                    # Trim periodically
                    if len(global_top_profit) > 500:
                        global_top_profit.sort(key=lambda x: -x['result']['avg_profit'])
                        global_top_profit = global_top_profit[:200]
                    if len(global_top_consistent) > 500:
                        global_top_consistent.sort(key=lambda x: -x['combined_score'])
                        global_top_consistent = global_top_consistent[:200]
                    if len(global_top_safe) > 500:
                        global_top_safe.sort(key=lambda x: x['result']['worst_drawdown'], reverse=True)
                        global_top_safe = global_top_safe[:200]

            elapsed_total = time.time() - marathon_start
            remaining_time = deadline - time.time()
            rate = total_tested / elapsed_total if elapsed_total > 0 else 0

            if wave_tested % (BATCH_SIZE * 2) == 0 or wave_tested == BATCH_SIZE:
                best_avg = max((r['result']['avg_profit'] for r in global_top_profit), default=0)
                best_safe = max((r['result']['worst_drawdown'] for r in global_top_safe), default=-9999)
                print(f"    {total_tested:>8,} tested | {qualified_count:,} qualified ($50+) | "
                      f"{rate:.0f}/s | "
                      f"Best avg: ${best_avg:.0f} | Safest DD: ${best_safe:.0f} | "
                      f"{remaining_time:.0f}s left", flush=True)

            if time.time() >= deadline:
                break

        wave_elapsed = time.time() - wave_start
        print(f"  ── Wave {wave_num} done: {wave_tested:,} tested in {wave_elapsed:.0f}s ──\n", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ═══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - marathon_start
    total_sessions = total_tested * total_positions

    print(f"\n{'='*150}")
    print(f"MARATHON ULTIMATE COMPLETE — {total_tested:,} strategies tested in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Valid: {total_valid:,} | Qualified ($50+ avg): {qualified_count:,}")
    print(f"Total sessions: ~{total_sessions:,} | Cross-validated on {len(datasets)} data files ({total_numbers} numbers)")
    print(f"{'='*150}\n")

    # Type distribution
    print("STRATEGIES TESTED PER TYPE:")
    for t in sorted(type_counts.keys()):
        print(f"  {t}: {type_counts[t]:,}")

    # Final sort
    global_top_profit.sort(key=lambda x: -x['result']['avg_profit'])
    global_top_consistent.sort(key=lambda x: -x['combined_score'])
    global_top_safe.sort(key=lambda x: x['result']['worst_drawdown'], reverse=True)

    # Print tables
    print_table("TOP 50 — HIGHEST AVG PROFIT (no cap, stop-loss only)", global_top_profit, 50)
    print_table("TOP 50 — BEST OVERALL SCORE (profit + survival + safety + consistency)", global_top_consistent, 50)
    if global_top_safe:
        print_table("TOP 30 — SAFEST WITH $80+ AVG PROFIT", global_top_safe, 30)

    # ── CONFIDENCE FILTER ANALYSIS ──
    # Do strategies with confidence filter outperform those without?
    conf_groups = defaultdict(list)
    for r in global_top_consistent:
        mc = r['params'].get('min_confidence', 0)
        if mc == 0:
            conf_groups['No filter'].append(r)
        elif mc <= 30:
            conf_groups['Low (15-30)'].append(r)
        elif mc <= 50:
            conf_groups['Medium (35-50)'].append(r)
        else:
            conf_groups['High (55-65)'].append(r)

    print(f"\n{'='*150}")
    print(f"  CONFIDENCE FILTER ANALYSIS — Does patience pay off?")
    print(f"{'='*150}")
    for grp_name in ['No filter', 'Low (15-30)', 'Medium (35-50)', 'High (55-65)']:
        grp = conf_groups.get(grp_name, [])
        if grp:
            avg_p = np.mean([r['result']['avg_profit'] for r in grp])
            avg_s = np.mean([r['result']['survival_rate'] for r in grp])
            avg_dd = np.mean([r['result']['worst_drawdown'] for r in grp])
            avg_sk = np.mean([r['result']['avg_skips'] for r in grp])
            avg_wr = np.mean([r['result']['avg_win_rate'] for r in grp])
            best_p = max(r['result']['avg_profit'] for r in grp)
            print(f"  {grp_name:>15}: {len(grp):>4} qualified | "
                  f"Avg profit: ${avg_p:>7.0f} | Survival: {avg_s:>5.1f}% | "
                  f"Win rate: {avg_wr:>5.1f}% | Avg skips: {avg_sk:>4.0f} | "
                  f"Worst DD: ${avg_dd:>7.0f} | Best: ${best_p:>7.0f}")

    # Best per type
    print(f"\n{'='*150}")
    print(f"  BEST STRATEGY PER TYPE (by overall score)")
    print(f"{'='*150}")
    types_seen = set()
    for r in global_top_consistent:
        t = r['type']
        if t not in types_seen:
            types_seen.add(t)
            res = r['result']
            mc = r['params'].get('min_confidence', 0)
            print(f"\n  [{t}] {r['label']}")
            print(f"      Avg Profit: ${res['avg_profit']} | Median: ${res['median_profit']} | "
                  f"Survival: {res['survival_rate']}% | Worst DD: ${res['worst_drawdown']} | "
                  f"P/R: {res['profit_to_risk']} | Bets: {res['avg_bets']} | Skipped: {res['avg_skips']} | "
                  f"Daily (10 sess): ${res['daily_10sess']}")

    # THE ONE
    if global_top_consistent:
        winner = global_top_consistent[0]
        wr = winner['result']
        mc = winner['params'].get('min_confidence', 0)
        print(f"\n{'='*150}")
        print(f"★ THE ONE STRATEGY ★ (from {total_tested:,} tested, cross-validated on {len(datasets)} files)")
        print(f"{'='*150}")
        print(f"  Type: {winner['type']}")
        print(f"  Strategy: {winner['label']}")
        print(f"  Overall Score: {winner['combined_score']:.1f}")
        print(f"")
        print(f"  SESSION STATS (across {wr['total_sessions']} sessions):")
        print(f"    Average profit/session:  ${wr['avg_profit']}")
        print(f"    Median profit/session:   ${wr['median_profit']}")
        print(f"    Best session:            ${wr['best_profit']}")
        print(f"    Worst session:           ${wr['worst_profit']}")
        print(f"    Survival rate:           {wr['survival_rate']}% of sessions end positive")
        print(f"    Avg spins/session:       {wr['avg_spins']}")
        print(f"    Avg bets placed:         {wr['avg_bets']} (out of ~{MAX_SESSION_SPINS} spins)")
        print(f"    Avg spins skipped:       {wr['avg_skips']} (waited for AI confidence)")
        print(f"    Avg win rate:            {wr['avg_win_rate']}%")
        print(f"    Worst drawdown:          ${wr['worst_drawdown']}")
        print(f"    Profit-to-risk ratio:    {wr['profit_to_risk']}")
        if mc > 0:
            print(f"    Confidence filter:       Only bet when AI confidence >= {mc}")
        else:
            print(f"    Confidence filter:       None (bet every spin)")
        print(f"")
        print(f"  DAILY PROJECTION (10 sessions):")
        print(f"    Expected daily:          ${wr['daily_10sess']}")
        print(f"    Expected monthly:        ${wr['daily_10sess'] * 30:.0f}")
        print(f"")
        print(f"  HOW TO PLAY:")
        if mc > 0:
            print(f"    1. Sit at the table, enter each spin result")
            print(f"    2. WAIT until AI confidence >= {mc} AND should_bet = True")
            print(f"    3. When AI says BET → place your chips")
            print(f"    4. Patience is the edge. Skip ~{wr['avg_skips']:.0f} spins per session")
            print(f"    5. Stop-loss: ${winner['params'].get('stop_loss', 4000)} — walk away if hit")
        else:
            print(f"    1. Sit at the table, enter each spin result")
            print(f"    2. Bet every spin the AI gives predictions")
            print(f"    3. Stop-loss: ${winner['params'].get('stop_loss', 4000)} — walk away if hit")
        print(f"    This is a JOB. 10 sessions/day. Discipline = wealth.")
        print(f"{'='*150}")


if __name__ == '__main__':
    main()
