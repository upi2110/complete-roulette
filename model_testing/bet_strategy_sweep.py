#!/usr/bin/env python3
"""
Bet Strategy Sweep — Test ALL betting management combinations.

Tests every combination of:
  - Losses before increment: 1, 2, 3, 4, 5
  - Wins before decrement: 1, 2, 3, 4, 5
  - Bet increment: $1, $2
  - Bet decrement: $1, $2
  - Max bet cap: $6, $8, $10, $15, $20, No cap
  - Starting bet: $1, $2, $3

Total combinations: 5 × 5 × 2 × 2 × 6 × 3 = 1,800 strategies

Each strategy is tested across ALL starting positions in the data file
(sliding window sessions), targeting +$100 per session.

Output: Ranked analysis of all strategies.
"""

import sys
import os
import time
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from config import (
    TOTAL_NUMBERS, TOP_PREDICTIONS_COUNT,
    WHEEL_ORDER, WHEEL_TABLE_0, WHEEL_TABLE_19,
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
    """Pre-compute predictions for every position in the data.

    This is the KEY optimization — instead of rebuilding the model
    for every session × every strategy, we compute predictions ONCE
    and reuse them across all strategy tests.

    Returns:
        list of (predicted_numbers_set, predicted_numbers_list) for each position
    """
    print("Pre-computing predictions for all positions...", flush=True)
    predictor = EnsemblePredictor()
    predictions = []

    for i, number in enumerate(all_numbers):
        if i >= 3:  # Need at least 3 spins before predicting
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


def run_session_fast(all_numbers, predictions, start_idx, strategy):
    """Run a single session using pre-computed predictions.

    Much faster than rebuilding the model each time.
    """
    base_bet = strategy['base_bet']
    increment = strategy['increment']
    decrement = strategy['decrement']
    losses_before_inc = strategy['losses_before_inc']
    wins_before_dec = strategy['wins_before_dec']
    max_bet = strategy['max_bet']
    min_bet = 1.0
    num_count = TOP_PREDICTIONS_COUNT
    target = SESSION_TARGET
    bankroll = INITIAL_BANKROLL

    remaining_count = len(all_numbers) - start_idx
    if remaining_count < 30:
        return None

    bet_per_number = base_bet
    consecutive_losses = 0
    consecutive_wins = 0
    session_profit = 0.0
    spins_played = 0
    total_wins = 0
    total_losses = 0
    max_drawdown = 0.0

    for spin_offset in range(remaining_count):
        pos = start_idx + spin_offset
        actual_number = all_numbers[pos]

        # Get pre-computed prediction
        pred_set, _ = predictions[pos]

        if not pred_set:
            # No prediction available (too early in data)
            spins_played += 1
            continue

        total_bet = bet_per_number * num_count
        hit = actual_number in pred_set

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

        # Track max drawdown
        if session_profit < max_drawdown:
            max_drawdown = session_profit

        # Target reached
        if session_profit >= target:
            break

        # Bankroll bust
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


def run_strategy_test(all_numbers, predictions, strategy, sample_positions):
    """Test a single strategy across multiple starting positions."""
    results = []

    for start_idx in sample_positions:
        result = run_session_fast(all_numbers, predictions, start_idx, strategy)
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


def main():
    # Find data file
    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
    else:
        test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        import glob
        files = sorted(glob.glob(os.path.join(test_data_dir, '*.txt')))
        if not files:
            print("Usage: python model_testing/bet_strategy_sweep.py <data_file.txt>")
            sys.exit(1)
        filepath = files[0]

    all_numbers = load_data_file(filepath)
    total_numbers = len(all_numbers)
    filename = os.path.basename(filepath)

    print(f"\n{'='*100}")
    print(f"BET STRATEGY SWEEP — Testing ALL Betting Management Combinations")
    print(f"{'='*100}")
    print(f"Data file: {filename} ({total_numbers} numbers)")
    print(f"Model: gap(35%) + tab_streak(22%) + frequency(20%) + wheel(13%) + hot(5%) + pattern(5%)")
    print(f"Numbers per spin: {TOP_PREDICTIONS_COUNT}")
    print(f"Target: +${SESSION_TARGET} per session")
    print(f"Bankroll: ${INITIAL_BANKROLL}")
    print(f"{'='*100}\n")

    # Pre-compute predictions (this is done ONCE)
    predictions = precompute_predictions(all_numbers)

    # Sample positions to test (every 5th position for speed, skip first 30 for history)
    min_start = 30
    min_remaining = 30
    max_start = total_numbers - min_remaining
    sample_positions = list(range(min_start, max_start, 5))
    print(f"Testing {len(sample_positions)} starting positions per strategy\n")

    # Define all parameter combinations
    losses_before_inc_options = [1, 2, 3, 4, 5]
    wins_before_dec_options = [1, 2, 3, 4, 5]
    increment_options = [1.0, 2.0]
    decrement_options = [1.0, 2.0]
    max_bet_options = [6.0, 8.0, 10.0, 15.0, 20.0, 200.0]  # 200 = effectively no cap
    base_bet_options = [1.0, 2.0, 3.0]

    max_bet_labels = {6.0: '$6', 8.0: '$8', 10.0: '$10', 15.0: '$15', 20.0: '$20', 200.0: 'No cap'}

    total_combos = (len(losses_before_inc_options) * len(wins_before_dec_options) *
                    len(increment_options) * len(decrement_options) *
                    len(max_bet_options) * len(base_bet_options))

    print(f"Total strategy combinations: {total_combos}")
    print(f"Estimated time: ~{total_combos * len(sample_positions) * 0.0001:.0f} seconds\n")

    start_time = time.time()
    all_results = []
    tested = 0

    for base_bet in base_bet_options:
        for losses_before_inc in losses_before_inc_options:
            for wins_before_dec in wins_before_dec_options:
                for increment in increment_options:
                    for decrement in decrement_options:
                        for max_bet in max_bet_options:
                            # Skip invalid: max_bet must be >= base_bet
                            if max_bet < base_bet:
                                tested += 1
                                continue

                            strategy = {
                                'base_bet': base_bet,
                                'increment': increment,
                                'decrement': decrement,
                                'losses_before_inc': losses_before_inc,
                                'wins_before_dec': wins_before_dec,
                                'max_bet': max_bet,
                            }

                            result = run_strategy_test(all_numbers, predictions, strategy, sample_positions)
                            tested += 1

                            if result:
                                strategy_label = (
                                    f"Start=${base_bet:.0f} "
                                    f"+${increment:.0f}/{losses_before_inc}L "
                                    f"-${decrement:.0f}/{wins_before_dec}W "
                                    f"Cap={max_bet_labels[max_bet]}"
                                )
                                all_results.append({
                                    'label': strategy_label,
                                    'strategy': strategy.copy(),
                                    'result': result,
                                })

                            if tested % 100 == 0:
                                elapsed = time.time() - start_time
                                pct = tested / total_combos * 100
                                print(f"  ... {tested}/{total_combos} ({pct:.0f}%) "
                                      f"tested in {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"\n{'='*100}")
    print(f"SWEEP COMPLETE — {tested} strategies tested in {elapsed:.0f} seconds")
    print(f"{'='*100}\n")

    if not all_results:
        print("No valid results!")
        return

    # ─── Sort by different criteria ───

    # 1. Best by target hit rate
    by_hit_rate = sorted(all_results, key=lambda x: x['result']['hit_rate'], reverse=True)

    # 2. Best by median profit
    by_median = sorted(all_results, key=lambda x: x['result']['median_profit'], reverse=True)

    # 3. Best by average profit
    by_avg = sorted(all_results, key=lambda x: x['result']['avg_profit'], reverse=True)

    # 4. Best combined score (hit_rate * 0.6 + normalized_median * 0.2 + normalized_drawdown * 0.2)
    max_median = max(r['result']['median_profit'] for r in all_results)
    min_drawdown = min(r['result']['worst_drawdown'] for r in all_results)
    for r in all_results:
        hr = r['result']['hit_rate']
        mp = r['result']['median_profit']
        wd = r['result']['worst_drawdown']
        # Normalize: higher is better for all
        norm_median = (mp / max_median * 100) if max_median > 0 else 0
        norm_drawdown = (1 - wd / min_drawdown) * 100 if min_drawdown < 0 else 100
        r['combined_score'] = hr * 0.5 + norm_median * 0.25 + norm_drawdown * 0.25

    by_combined = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)

    # 5. Safest (smallest worst drawdown)
    by_safest = sorted(all_results, key=lambda x: x['result']['worst_drawdown'], reverse=True)

    # ─── Print Results ───

    def print_table(title, ranked, count=20):
        print(f"\n{'─'*100}")
        print(f"  {title}")
        print(f"{'─'*100}")
        print(f"{'Rank':<5} {'Strategy':<45} {'Hit%':>6} {'AvgP':>8} {'MedP':>8} "
              f"{'WorstP':>8} {'BestP':>8} {'AvgSp':>6} {'WrstDd':>8} {'WinR%':>6}")
        print(f"{'─'*100}")

        for i, r in enumerate(ranked[:count], 1):
            res = r['result']
            print(f"{i:<5} {r['label']:<45} "
                  f"{res['hit_rate']:>5.1f}% "
                  f"${res['avg_profit']:>7.0f} "
                  f"${res['median_profit']:>7.0f} "
                  f"${res['worst_profit']:>7.0f} "
                  f"${res['best_profit']:>7.0f} "
                  f"{res['avg_spins']:>5.0f} "
                  f"${res['worst_drawdown']:>7.0f} "
                  f"{res['avg_win_rate']:>5.1f}%")

    print_table("🏆 TOP 20 — HIGHEST TARGET HIT RATE (most sessions reach +$100)", by_hit_rate, 20)
    print_table("💰 TOP 20 — HIGHEST MEDIAN PROFIT", by_median, 20)
    print_table("📊 TOP 20 — HIGHEST AVERAGE PROFIT", by_avg, 20)
    print_table("🛡️  TOP 20 — SAFEST (smallest worst drawdown)", by_safest, 20)
    print_table("⭐ TOP 20 — BEST OVERALL (combined: hit rate + profit + safety)", by_combined, 20)

    # ─── The Winner ───
    winner = by_combined[0]
    wr = winner['result']
    ws = winner['strategy']

    print(f"\n{'='*100}")
    print(f"🏆 RECOMMENDED STRATEGY")
    print(f"{'='*100}")
    print(f"  Strategy: {winner['label']}")
    print(f"  Combined Score: {winner['combined_score']:.1f}")
    print(f"")
    print(f"  Settings:")
    print(f"    Starting bet:       ${ws['base_bet']:.0f} per number")
    print(f"    Numbers per spin:   {TOP_PREDICTIONS_COUNT}")
    print(f"    Increase bet:       +${ws['increment']:.0f} after {ws['losses_before_inc']} consecutive losses")
    print(f"    Decrease bet:       -${ws['decrement']:.0f} after {ws['wins_before_dec']} consecutive wins")
    print(f"    Maximum bet:        ${ws['max_bet']:.0f} per number" if ws['max_bet'] < 200 else
          f"    Maximum bet:        No cap")
    print(f"    Minimum bet:        $1 per number")
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
    print(f"{'='*100}")

    # Also show runner-ups
    print(f"\n{'─'*100}")
    print(f"  RUNNER-UPS (Top 5 Overall)")
    print(f"{'─'*100}")
    for i, r in enumerate(by_combined[:5], 1):
        res = r['result']
        print(f"  {i}. {r['label']}")
        print(f"     Hit: {res['hit_rate']}% | Avg: ${res['avg_profit']} | "
              f"Median: ${res['median_profit']} | Worst: ${res['worst_drawdown']} | "
              f"Score: {r['combined_score']:.1f}")


if __name__ == '__main__':
    main()
