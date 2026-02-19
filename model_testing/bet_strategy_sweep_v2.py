#!/usr/bin/env python3
"""
Bet Strategy Sweep V2 — Recovery-After-Hit & Creative Betting Strategies.

Tests SMART betting strategies that don't escalate during losses:

STRATEGY TYPE A: "Recovery After Hit"
  - During losses: keep bet FLAT at base
  - After a HIT: look back at consecutive losses, divide by factor, round up
  - Use that as recovery bet for next N wins, then drop back to base
  - If miss after recovery bet, drop back to base

STRATEGY TYPE B: "Streak Rider"
  - Start at base bet
  - After each consecutive win, increase bet by increment
  - After any loss, drop back to base immediately
  - Ride the winning streaks, cut losses fast

STRATEGY TYPE C: "Loss Counter Recovery"
  - During losses: keep bet flat at base
  - After a HIT: bet = base + (total_consecutive_losses_before_hit / divisor)
  - Keep elevated bet for N wins or until a miss
  - Different divisors: /2, /3, /4

STRATEGY TYPE D: "Patience Recovery"
  - During losses: keep flat
  - After a HIT following 3+ misses: elevated bet = base × multiplier
  - After a HIT following <3 misses: stay at base
  - Only go big when recovering from a real cold streak

STRATEGY TYPE E: "Progressive Win Rider"
  - Start at base
  - Win 1: stay at base
  - Win 2 in a row: go to base + $1
  - Win 3 in a row: go to base + $2
  - Any loss: back to base
  - Capitalize on hot streaks

Total: ~3,000+ strategy combinations tested.
"""

import sys
import os
import time
import math
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from config import (
    TOTAL_NUMBERS, TOP_PREDICTIONS_COUNT,
    SESSION_TARGET, INITIAL_BANKROLL,
)
from app.ml.ensemble import EnsemblePredictor


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
    """Pre-compute predictions for every position (done ONCE)."""
    print("Pre-computing predictions for all positions...", flush=True)
    predictor = EnsemblePredictor()
    predictions = []

    for i, number in enumerate(all_numbers):
        if i >= 3:
            pred = predictor.predict()
            top_nums = pred.get('top_numbers', [])[:TOP_PREDICTIONS_COUNT]
            predictions.append((set(top_nums), top_nums))
        else:
            predictions.append((set(), []))

        predictor.update(number)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(all_numbers)} positions computed", flush=True)

    print(f"  Done: {len(predictions)} predictions cached\n", flush=True)
    return predictions


# ═══════════════════════════════════════════════════════════════════════
# SESSION RUNNERS — One per strategy type
# ═══════════════════════════════════════════════════════════════════════

def run_session_type_a(all_numbers, predictions, start_idx, params):
    """TYPE A: Recovery After Hit.

    During losses: flat at base_bet.
    After HIT: recovery_bet = base + ceil(consecutive_losses / divisor)
    Keep recovery bet for `recovery_wins` wins, then back to base.
    If miss during recovery: back to base.
    """
    base_bet = params['base_bet']
    divisor = params['divisor']
    recovery_wins_needed = params['recovery_wins']
    max_bet = params['max_bet']
    min_bet = 1.0
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL

    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None

    bet_per_number = base_bet
    consecutive_losses = 0
    recovery_wins_left = 0  # How many more wins at elevated bet
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining):
        pos = start_idx + spin_offset
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]

        if not pred_set:
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual in pred_set

        if hit:
            net = (35 * bet_per_number) - total_bet + bet_per_number
            session_profit += net
            bankroll += net
            total_wins += 1

            if recovery_wins_left > 0:
                recovery_wins_left -= 1
                if recovery_wins_left == 0:
                    bet_per_number = base_bet  # Recovery complete, back to base
            else:
                # Just hit — check how many losses we had before this
                if consecutive_losses > 0:
                    # Calculate recovery bet
                    recovery_amount = math.ceil(consecutive_losses / divisor)
                    bet_per_number = min(max_bet, base_bet + recovery_amount)
                    recovery_wins_left = recovery_wins_needed - 1  # -1 because this win counts
                    if recovery_wins_left <= 0:
                        bet_per_number = base_bet  # Only needed 1 win
                # else: was already winning, stay at base

            consecutive_losses = 0

        else:
            net = -total_bet
            session_profit += net
            bankroll += net
            total_losses += 1
            consecutive_losses += 1

            # If we were in recovery mode and missed, drop to base
            if recovery_wins_left > 0:
                recovery_wins_left = 0
                bet_per_number = base_bet

            # During losses: stay flat at base
            bet_per_number = base_bet

        spins_played += 1
        if session_profit < max_drawdown:
            max_drawdown = session_profit
        if session_profit >= target:
            break
        if bankroll <= 0:
            break

    if spins_played == 0:
        return None

    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1) if spins_played else 0,
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= target,
        'max_drawdown': round(max_drawdown, 2),
    }


def run_session_type_b(all_numbers, predictions, start_idx, params):
    """TYPE B: Streak Rider.

    After each consecutive win: bet += increment
    After any loss: bet = base_bet (instant reset)
    """
    base_bet = params['base_bet']
    increment = params['increment']
    max_bet = params['max_bet']
    min_bet = 1.0
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL

    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None

    bet_per_number = base_bet
    consecutive_wins = 0
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining):
        pos = start_idx + spin_offset
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]

        if not pred_set:
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual in pred_set

        if hit:
            net = (35 * bet_per_number) - total_bet + bet_per_number
            session_profit += net
            bankroll += net
            total_wins += 1
            consecutive_wins += 1
            # Increase bet for next spin (riding the streak)
            bet_per_number = min(max_bet, base_bet + consecutive_wins * increment)
        else:
            net = -total_bet
            session_profit += net
            bankroll += net
            total_losses += 1
            consecutive_wins = 0
            bet_per_number = base_bet  # Instant reset

        spins_played += 1
        if session_profit < max_drawdown:
            max_drawdown = session_profit
        if session_profit >= target:
            break
        if bankroll <= 0:
            break

    if spins_played == 0:
        return None

    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1) if spins_played else 0,
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= target,
        'max_drawdown': round(max_drawdown, 2),
    }


def run_session_type_c(all_numbers, predictions, start_idx, params):
    """TYPE C: Loss Counter Recovery.

    During losses: flat at base.
    After HIT: bet = base + floor(total_consec_losses / divisor) * step
    Keep elevated for `hold_wins` wins or until miss.
    """
    base_bet = params['base_bet']
    divisor = params['divisor']
    step = params['step']
    hold_wins = params['hold_wins']
    max_bet = params['max_bet']
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL

    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None

    bet_per_number = base_bet
    consecutive_losses = 0
    elevated_wins_left = 0
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining):
        pos = start_idx + spin_offset
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]

        if not pred_set:
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual in pred_set

        if hit:
            net = (35 * bet_per_number) - total_bet + bet_per_number
            session_profit += net
            bankroll += net
            total_wins += 1

            if elevated_wins_left > 0:
                elevated_wins_left -= 1
                if elevated_wins_left == 0:
                    bet_per_number = base_bet
            else:
                if consecutive_losses > 0:
                    bump = math.floor(consecutive_losses / divisor) * step
                    bet_per_number = min(max_bet, base_bet + bump)
                    elevated_wins_left = hold_wins - 1
                    if elevated_wins_left <= 0:
                        bet_per_number = base_bet

            consecutive_losses = 0

        else:
            net = -total_bet
            session_profit += net
            bankroll += net
            total_losses += 1
            consecutive_losses += 1

            # Miss during elevated: drop back
            if elevated_wins_left > 0:
                elevated_wins_left = 0
            bet_per_number = base_bet

        spins_played += 1
        if session_profit < max_drawdown:
            max_drawdown = session_profit
        if session_profit >= target:
            break
        if bankroll <= 0:
            break

    if spins_played == 0:
        return None

    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1) if spins_played else 0,
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= target,
        'max_drawdown': round(max_drawdown, 2),
    }


def run_session_type_d(all_numbers, predictions, start_idx, params):
    """TYPE D: Patience Recovery.

    Flat bet during losses.
    After HIT following `min_losses`+ misses: bet = base × multiplier
    After HIT following <min_losses misses: stay at base
    Keep elevated for `hold_wins` wins or until miss.
    """
    base_bet = params['base_bet']
    min_losses = params['min_losses']
    multiplier = params['multiplier']
    hold_wins = params['hold_wins']
    max_bet = params['max_bet']
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL

    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None

    bet_per_number = base_bet
    consecutive_losses = 0
    elevated_wins_left = 0
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining):
        pos = start_idx + spin_offset
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]

        if not pred_set:
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual in pred_set

        if hit:
            net = (35 * bet_per_number) - total_bet + bet_per_number
            session_profit += net
            bankroll += net
            total_wins += 1

            if elevated_wins_left > 0:
                elevated_wins_left -= 1
                if elevated_wins_left == 0:
                    bet_per_number = base_bet
            else:
                if consecutive_losses >= min_losses:
                    bet_per_number = min(max_bet, base_bet * multiplier)
                    elevated_wins_left = hold_wins - 1
                    if elevated_wins_left <= 0:
                        bet_per_number = base_bet

            consecutive_losses = 0

        else:
            net = -total_bet
            session_profit += net
            bankroll += net
            total_losses += 1
            consecutive_losses += 1

            if elevated_wins_left > 0:
                elevated_wins_left = 0
            bet_per_number = base_bet

        spins_played += 1
        if session_profit < max_drawdown:
            max_drawdown = session_profit
        if session_profit >= target:
            break
        if bankroll <= 0:
            break

    if spins_played == 0:
        return None

    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1) if spins_played else 0,
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= target,
        'max_drawdown': round(max_drawdown, 2),
    }


def run_session_type_e(all_numbers, predictions, start_idx, params):
    """TYPE E: Progressive Win Rider.

    Win 1: stay at base
    Win 2 in a row: base + 1×step
    Win 3 in a row: base + 2×step
    Win N in a row: base + (N-1)×step
    Any loss: back to base
    """
    base_bet = params['base_bet']
    step = params['step']
    wins_before_boost = params['wins_before_boost']
    max_bet = params['max_bet']
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL

    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None

    bet_per_number = base_bet
    consecutive_wins = 0
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining):
        pos = start_idx + spin_offset
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]

        if not pred_set:
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual in pred_set

        if hit:
            net = (35 * bet_per_number) - total_bet + bet_per_number
            session_profit += net
            bankroll += net
            total_wins += 1
            consecutive_wins += 1

            # Progressive increase after wins_before_boost consecutive wins
            if consecutive_wins >= wins_before_boost:
                boosts = consecutive_wins - wins_before_boost + 1
                bet_per_number = min(max_bet, base_bet + boosts * step)
        else:
            net = -total_bet
            session_profit += net
            bankroll += net
            total_losses += 1
            consecutive_wins = 0
            bet_per_number = base_bet

        spins_played += 1
        if session_profit < max_drawdown:
            max_drawdown = session_profit
        if session_profit >= target:
            break
        if bankroll <= 0:
            break

    if spins_played == 0:
        return None

    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1) if spins_played else 0,
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= target,
        'max_drawdown': round(max_drawdown, 2),
    }


# Also include original V1 style for comparison
def run_session_original(all_numbers, predictions, start_idx, params):
    """Original V1: Escalate during losses, decrease during wins."""
    base_bet = params['base_bet']
    increment = params['increment']
    decrement = params['decrement']
    losses_before_inc = params['losses_before_inc']
    wins_before_dec = params['wins_before_dec']
    max_bet = params['max_bet']
    min_bet = 1.0
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL

    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None

    bet_per_number = base_bet
    consecutive_losses = 0
    consecutive_wins = 0
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining):
        pos = start_idx + spin_offset
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]

        if not pred_set:
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual in pred_set

        if hit:
            net = (35 * bet_per_number) - total_bet + bet_per_number
            session_profit += net
            bankroll += net
            total_wins += 1
            consecutive_wins += 1
            consecutive_losses = 0

            if consecutive_wins >= wins_before_dec:
                bet_per_number = max(min_bet, bet_per_number - decrement)
                consecutive_wins = 0
        else:
            net = -total_bet
            session_profit += net
            bankroll += net
            total_losses += 1
            consecutive_losses += 1
            consecutive_wins = 0

            if consecutive_losses >= losses_before_inc:
                bet_per_number = min(max_bet, bet_per_number + increment)
                consecutive_losses = 0

        spins_played += 1
        if session_profit < max_drawdown:
            max_drawdown = session_profit
        if session_profit >= target:
            break
        if bankroll <= 0:
            break

    if spins_played == 0:
        return None

    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1) if spins_played else 0,
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= target,
        'max_drawdown': round(max_drawdown, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# TYPE F: Hybrid — Recovery After Hit + Streak Rider combined
# ═══════════════════════════════════════════════════════════════════════

def run_session_type_f(all_numbers, predictions, start_idx, params):
    """TYPE F: Hybrid Recovery + Streak.

    During losses: flat at base.
    After HIT following losses: recovery bet = base + ceil(losses/divisor)
    During consecutive wins: keep increasing by win_step each win
    After miss: back to base.
    Combines recovery with streak riding.
    """
    base_bet = params['base_bet']
    divisor = params['divisor']
    win_step = params['win_step']
    max_bet = params['max_bet']
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL

    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None

    bet_per_number = base_bet
    consecutive_losses = 0
    consecutive_wins = 0
    in_recovery = False
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining):
        pos = start_idx + spin_offset
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]

        if not pred_set:
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual in pred_set

        if hit:
            net = (35 * bet_per_number) - total_bet + bet_per_number
            session_profit += net
            bankroll += net
            total_wins += 1
            consecutive_wins += 1

            if consecutive_losses > 0 and not in_recovery:
                # First win after losses — set recovery bet
                recovery_amount = math.ceil(consecutive_losses / divisor)
                bet_per_number = min(max_bet, base_bet + recovery_amount)
                in_recovery = True
            elif consecutive_wins >= 2:
                # Streak riding — keep increasing
                bet_per_number = min(max_bet, bet_per_number + win_step)

            consecutive_losses = 0
        else:
            net = -total_bet
            session_profit += net
            bankroll += net
            total_losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            in_recovery = False
            bet_per_number = base_bet

        spins_played += 1
        if session_profit < max_drawdown:
            max_drawdown = session_profit
        if session_profit >= target:
            break
        if bankroll <= 0:
            break

    if spins_played == 0:
        return None

    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1) if spins_played else 0,
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= target,
        'max_drawdown': round(max_drawdown, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# TYPE G: Percentage Recovery — bet a % of losses to recover
# ═══════════════════════════════════════════════════════════════════════

def run_session_type_g(all_numbers, predictions, start_idx, params):
    """TYPE G: Percentage Recovery.

    During losses: flat at base.
    Track cumulative loss amount during cold streak.
    After HIT: bet = base + (total_loss_amount × recovery_pct / payout)
    This tries to recover a percentage of losses in one hit.
    After hit or miss at elevated bet: back to base, reset loss tracker.
    """
    base_bet = params['base_bet']
    recovery_pct = params['recovery_pct']  # e.g., 0.25 = try to recover 25% of losses
    max_bet = params['max_bet']
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL
    payout_per_unit = 36 - num_count  # net per $1 bet per number on a hit

    remaining = len(all_numbers) - start_idx
    if remaining < 30:
        return None

    bet_per_number = base_bet
    cumulative_loss = 0.0
    in_recovery = False
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining):
        pos = start_idx + spin_offset
        actual = all_numbers[pos]
        pred_set, _ = predictions[pos]

        if not pred_set:
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual in pred_set

        if hit:
            net = (35 * bet_per_number) - total_bet + bet_per_number
            session_profit += net
            bankroll += net
            total_wins += 1

            if in_recovery:
                # Recovery complete, back to base
                in_recovery = False
                cumulative_loss = 0.0
                bet_per_number = base_bet
            elif cumulative_loss > 0:
                # Just won after losses — set recovery bet for next spin
                target_recovery = cumulative_loss * recovery_pct
                recovery_bet = base_bet + math.ceil(target_recovery / max(payout_per_unit, 1))
                bet_per_number = min(max_bet, recovery_bet)
                in_recovery = True
                cumulative_loss = 0.0
            # else: winning streak, stay at base

        else:
            net = -total_bet
            session_profit += net
            bankroll += net
            total_losses += 1
            cumulative_loss += total_bet

            if in_recovery:
                in_recovery = False
            bet_per_number = base_bet

        spins_played += 1
        if session_profit < max_drawdown:
            max_drawdown = session_profit
        if session_profit >= target:
            break
        if bankroll <= 0:
            break

    if spins_played == 0:
        return None

    return {
        'spins_played': spins_played,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / spins_played * 100, 1) if spins_played else 0,
        'final_profit': round(session_profit, 2),
        'target_reached': session_profit >= target,
        'max_drawdown': round(max_drawdown, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY GENERATOR — Target ~1000 strategies
# ═══════════════════════════════════════════════════════════════════════

def generate_all_strategies():
    """Generate ~1000 diverse strategy combinations to test."""
    strategies = []

    base_bets = [1.0, 2.0, 3.0]
    # Use fewer cap options — many caps give identical results
    max_bets = [8.0, 10.0, 15.0, 200.0]
    max_bet_labels = {8.0: '$8', 10.0: '$10', 15.0: '$15', 200.0: 'NoCap'}

    # ─── TYPE A: Recovery After Hit (your exact idea) ───
    # ~120 strategies
    for base in base_bets:
        for divisor in [1.5, 2, 3, 4]:
            for recovery_wins in [1, 2, 3]:
                for max_bet in max_bets:
                    if max_bet < base:
                        continue
                    label = f"A:Recov ${base:.0f} L/{divisor} hold{recovery_wins}W Cap={max_bet_labels[max_bet]}"
                    strategies.append({
                        'type': 'A',
                        'label': label,
                        'runner': run_session_type_a,
                        'params': {
                            'base_bet': base,
                            'divisor': divisor,
                            'recovery_wins': recovery_wins,
                            'max_bet': max_bet,
                        }
                    })

    # ─── TYPE B: Streak Rider ───
    # ~48 strategies
    for base in base_bets:
        for increment in [0.5, 1.0, 1.5, 2.0]:
            for max_bet in max_bets:
                if max_bet < base:
                    continue
                label = f"B:Streak ${base:.0f} +${increment}/win Cap={max_bet_labels[max_bet]}"
                strategies.append({
                    'type': 'B',
                    'label': label,
                    'runner': run_session_type_b,
                    'params': {
                        'base_bet': base,
                        'increment': increment,
                        'max_bet': max_bet,
                    }
                })

    # ─── TYPE C: Loss Counter Recovery ───
    # ~108 strategies
    for base in base_bets:
        for divisor in [2, 3, 4]:
            for step in [1.0, 2.0, 3.0]:
                for hold_wins in [1, 2, 3]:
                    for max_bet in [10.0, 200.0]:
                        if max_bet < base:
                            continue
                        label = f"C:LossCount ${base:.0f} L/{divisor}*${step:.0f} hold{hold_wins}W Cap={max_bet_labels[max_bet]}"
                        strategies.append({
                            'type': 'C',
                            'label': label,
                            'runner': run_session_type_c,
                            'params': {
                                'base_bet': base,
                                'divisor': divisor,
                                'step': step,
                                'hold_wins': hold_wins,
                                'max_bet': max_bet,
                            }
                        })

    # ─── TYPE D: Patience Recovery ───
    # ~192 strategies
    for base in base_bets:
        for min_losses in [2, 3, 4, 5]:
            for multiplier in [1.5, 2.0, 2.5, 3.0]:
                for hold_wins in [1, 2, 3, 4]:
                    for max_bet in [10.0, 200.0]:
                        if max_bet < base:
                            continue
                        label = f"D:Patient ${base:.0f} {min_losses}L->x{multiplier} hold{hold_wins}W Cap={max_bet_labels[max_bet]}"
                        strategies.append({
                            'type': 'D',
                            'label': label,
                            'runner': run_session_type_d,
                            'params': {
                                'base_bet': base,
                                'divisor': min_losses,
                                'min_losses': min_losses,
                                'multiplier': multiplier,
                                'hold_wins': hold_wins,
                                'max_bet': max_bet,
                            }
                        })

    # ─── TYPE E: Progressive Win Rider ───
    # ~108 strategies
    for base in base_bets:
        for step in [0.5, 1.0, 2.0]:
            for wins_before in [1, 2, 3]:
                for max_bet in max_bets:
                    if max_bet < base:
                        continue
                    label = f"E:ProgWin ${base:.0f} +${step}/win>={wins_before} Cap={max_bet_labels[max_bet]}"
                    strategies.append({
                        'type': 'E',
                        'label': label,
                        'runner': run_session_type_e,
                        'params': {
                            'base_bet': base,
                            'step': step,
                            'wins_before_boost': wins_before,
                            'max_bet': max_bet,
                        }
                    })

    # ─── TYPE F: Hybrid Recovery + Streak Rider ───
    # ~144 strategies
    for base in base_bets:
        for divisor in [1.5, 2, 3, 4]:
            for win_step in [0.5, 1.0, 2.0]:
                for max_bet in max_bets:
                    if max_bet < base:
                        continue
                    label = f"F:Hybrid ${base:.0f} L/{divisor}+${win_step}/win Cap={max_bet_labels[max_bet]}"
                    strategies.append({
                        'type': 'F',
                        'label': label,
                        'runner': run_session_type_f,
                        'params': {
                            'base_bet': base,
                            'divisor': divisor,
                            'win_step': win_step,
                            'max_bet': max_bet,
                        }
                    })

    # ─── TYPE G: Percentage Recovery ───
    # ~72 strategies
    for base in base_bets:
        for recovery_pct in [0.15, 0.25, 0.35, 0.50, 0.75, 1.0]:
            for max_bet in max_bets:
                if max_bet < base:
                    continue
                pct_label = f"{int(recovery_pct*100)}%"
                label = f"G:PctRecov ${base:.0f} recov={pct_label} Cap={max_bet_labels[max_bet]}"
                strategies.append({
                    'type': 'G',
                    'label': label,
                    'runner': run_session_type_g,
                    'params': {
                        'base_bet': base,
                        'recovery_pct': recovery_pct,
                        'max_bet': max_bet,
                    }
                })

    # ─── TYPE V1: Original Escalation (expanded for comparison) ───
    # ~200 strategies
    for base in base_bets:
        for losses_inc in [1, 2, 3, 4]:
            for wins_dec in [1, 2, 3]:
                for inc in [1.0, 2.0]:
                    for dec in [1.0]:
                        for max_bet in [10.0, 15.0, 200.0]:
                            if max_bet < base:
                                continue
                            label = f"V1:Escal ${base:.0f} +${inc:.0f}/{losses_inc}L -${dec:.0f}/{wins_dec}W Cap={max_bet_labels[max_bet]}"
                            strategies.append({
                                'type': 'V1',
                                'label': label,
                                'runner': run_session_original,
                                'params': {
                                    'base_bet': base,
                                    'increment': inc,
                                    'decrement': dec,
                                    'losses_before_inc': losses_inc,
                                    'wins_before_dec': wins_dec,
                                    'max_bet': max_bet,
                                }
                            })

    # ─── BASELINE markers ───
    strategies.append({
        'type': 'BASE',
        'label': "CURRENT: $2 +$1/2L -$1/2W Cap=$10",
        'runner': run_session_original,
        'params': {
            'base_bet': 2.0, 'increment': 1.0, 'decrement': 1.0,
            'losses_before_inc': 2, 'wins_before_dec': 2, 'max_bet': 10.0,
        }
    })
    strategies.append({
        'type': 'BASE',
        'label': "V1-WINNER: $3 +$2/1L -$1/2W NoCap",
        'runner': run_session_original,
        'params': {
            'base_bet': 3.0, 'increment': 2.0, 'decrement': 1.0,
            'losses_before_inc': 1, 'wins_before_dec': 2, 'max_bet': 200.0,
        }
    })

    return strategies


def run_strategy_test(all_numbers, predictions, strategy, sample_positions):
    """Test a single strategy across multiple starting positions."""
    runner = strategy['runner']
    params = strategy['params']
    results = []

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
        'median_profit': round(np.median(profits), 2),
        'worst_profit': round(min(profits), 2),
        'best_profit': round(max(profits), 2),
        'avg_spins': round(np.mean(spins), 1),
        'avg_drawdown': round(np.mean(drawdowns), 2),
        'worst_drawdown': round(min(drawdowns), 2),
        'avg_win_rate': round(np.mean(win_rates), 1),
        'total_profit': round(sum(profits), 2),
    }


def print_table(title, ranked, count=25):
    print(f"\n{'─'*120}")
    print(f"  {title}")
    print(f"{'─'*120}")
    print(f"{'Rank':<5} {'Type':<4} {'Strategy':<60} {'Hit%':>6} {'AvgP':>8} {'MedP':>8} "
          f"{'WorstP':>8} {'BestP':>8} {'AvgSp':>6} {'WrstDd':>8} {'WinR%':>6}")
    print(f"{'─'*120}")

    for i, r in enumerate(ranked[:count], 1):
        res = r['result']
        print(f"{i:<5} {r['type']:<4} {r['label']:<60} "
              f"{res['hit_rate']:>5.1f}% "
              f"${res['avg_profit']:>7.0f} "
              f"${res['median_profit']:>7.0f} "
              f"${res['worst_profit']:>7.0f} "
              f"${res['best_profit']:>7.0f} "
              f"{res['avg_spins']:>5.0f} "
              f"${res['worst_drawdown']:>7.0f} "
              f"{res['avg_win_rate']:>5.1f}%")


def main():
    # Find data file
    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
    else:
        test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        import glob
        files = sorted(glob.glob(os.path.join(test_data_dir, '*.txt')))
        if not files:
            print("Usage: python model_testing/bet_strategy_sweep_v2.py <data_file.txt>")
            sys.exit(1)
        filepath = files[0]

    all_numbers = load_data_file(filepath)
    total_numbers = len(all_numbers)
    filename = os.path.basename(filepath)

    print(f"\n{'='*120}")
    print(f"BET STRATEGY SWEEP V2 — Recovery & Creative Betting Strategies")
    print(f"{'='*120}")
    print(f"Data file: {filename} ({total_numbers} numbers)")
    print(f"Model: gap(35%) + tab_streak(22%) + frequency(20%) + wheel(13%) + hot(5%) + pattern(5%)")
    print(f"Numbers per spin: {TOP_PREDICTIONS_COUNT}")
    print(f"Target: +${SESSION_TARGET} per session")
    print(f"Bankroll: ${INITIAL_BANKROLL}")
    print(f"{'='*120}")
    print()
    print("Strategy Types:")
    print("  A: Recovery After Hit — flat during losses, bump after hit based on losses/divisor")
    print("  B: Streak Rider — increase during win streaks, reset on loss")
    print("  C: Loss Counter Recovery — similar to A with step-based bump")
    print("  D: Patience Recovery — only bump after long cold streaks (3+ losses)")
    print("  E: Progressive Win Rider — progressive increase during win streaks")
    print("  F: Hybrid — recovery after hit + keep riding if wins continue")
    print("  G: Percentage Recovery — bet to recover X% of total losses after hit")
    print("  V1: Original Escalation — increase during losses (expanded)")
    print("  BASE: Current config & V1 winner for comparison")
    print()

    # Pre-compute predictions
    predictions = precompute_predictions(all_numbers)

    # Sample positions
    min_start = 30
    min_remaining = 30
    max_start = total_numbers - min_remaining
    sample_positions = list(range(min_start, max_start, 5))
    print(f"Testing {len(sample_positions)} starting positions per strategy\n")

    # Generate all strategies
    all_strategies = generate_all_strategies()
    total_combos = len(all_strategies)

    # Count by type
    type_counts = defaultdict(int)
    for s in all_strategies:
        type_counts[s['type']] += 1

    print(f"Total strategy combinations: {total_combos}")
    for t in sorted(type_counts.keys()):
        print(f"  Type {t}: {type_counts[t]}")
    print()

    start_time = time.time()
    all_results = []
    tested = 0

    for strategy in all_strategies:
        result = run_strategy_test(all_numbers, predictions, strategy, sample_positions)
        tested += 1

        if result:
            all_results.append({
                'label': strategy['label'],
                'type': strategy['type'],
                'strategy': strategy['params'],
                'result': result,
            })

        if tested % 200 == 0:
            elapsed = time.time() - start_time
            pct = tested / total_combos * 100
            print(f"  ... {tested}/{total_combos} ({pct:.0f}%) tested in {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"\n{'='*120}")
    print(f"SWEEP COMPLETE — {tested} strategies tested in {elapsed:.0f} seconds")
    print(f"{'='*120}\n")

    if not all_results:
        print("No valid results!")
        return

    # ─── Score and rank ───

    # Calculate combined score
    max_median = max(r['result']['median_profit'] for r in all_results)
    min_drawdown = min(r['result']['worst_drawdown'] for r in all_results)

    for r in all_results:
        hr = r['result']['hit_rate']
        mp = r['result']['median_profit']
        wd = r['result']['worst_drawdown']
        norm_median = (mp / max_median * 100) if max_median > 0 else 0
        norm_drawdown = (1 - wd / min_drawdown) * 100 if min_drawdown < 0 else 100
        r['combined_score'] = hr * 0.5 + norm_median * 0.25 + norm_drawdown * 0.25

    by_hit_rate = sorted(all_results, key=lambda x: x['result']['hit_rate'], reverse=True)
    by_median = sorted(all_results, key=lambda x: x['result']['median_profit'], reverse=True)
    by_avg = sorted(all_results, key=lambda x: x['result']['avg_profit'], reverse=True)
    by_safest = sorted(all_results, key=lambda x: x['result']['worst_drawdown'], reverse=True)
    by_combined = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)

    # ─── Print by strategy type first ───
    print_table("TOP 25 — HIGHEST HIT RATE (all types)", by_hit_rate, 25)
    print_table("TOP 25 — HIGHEST MEDIAN PROFIT", by_median, 25)
    print_table("TOP 25 — SAFEST (smallest worst drawdown)", by_safest, 25)
    print_table("TOP 25 — BEST OVERALL (combined score)", by_combined, 25)

    # ─── Best per type ───
    print(f"\n{'='*120}")
    print(f"  BEST STRATEGY PER TYPE (by combined score)")
    print(f"{'='*120}")
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
    ws = winner['strategy']

    print(f"\n{'='*120}")
    print(f"RECOMMENDED STRATEGY")
    print(f"{'='*120}")
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
    print(f"{'='*120}")

    # ─── Comparison: Recovery types vs V1 types ───
    print(f"\n{'='*120}")
    print(f"  RECOVERY STYLES (A,C,D) vs STREAK STYLES (B,E) vs V1 ESCALATION")
    print(f"{'='*120}")

    for group_name, group_types in [
        ("Recovery (flat during loss, bump after hit)", ['A', 'C', 'D', 'G']),
        ("Streak Riders (increase during wins)", ['B', 'E']),
        ("Hybrid (recovery + streak)", ['F']),
        ("V1 Escalation (increase during losses)", ['V1', 'BASE']),
    ]:
        group_results = [r for r in all_results if r['type'] in group_types]
        if not group_results:
            continue
        best = max(group_results, key=lambda x: x['combined_score'])
        avg_hit = np.mean([r['result']['hit_rate'] for r in group_results])
        avg_drawdown = np.mean([r['result']['worst_drawdown'] for r in group_results])
        res = best['result']
        print(f"\n  {group_name}:")
        print(f"    Strategies tested: {len(group_results)}")
        print(f"    Avg hit rate across all: {avg_hit:.1f}%")
        print(f"    Avg worst drawdown: ${avg_drawdown:.0f}")
        print(f"    BEST: {best['label']}")
        print(f"      Hit: {res['hit_rate']}% | Median: ${res['median_profit']} | "
              f"Worst DD: ${res['worst_drawdown']} | Score: {best['combined_score']:.1f}")


if __name__ == '__main__':
    main()
