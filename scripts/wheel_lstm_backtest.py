#!/usr/bin/env python3
"""
Wheel Strategy + LSTM Backtest — Proper walk-forward with LSTM training.

Unlike other models that just count/compute, LSTM needs actual training.
This script:
  1. LSTM solo (trains every 50 spins, predicts from trained model)
  2. WheelStrategy solo (for comparison)
  3. Frequency solo (for comparison)
  4. WS + LSTM at various weight ratios
  5. WS + LSTM + Frequency triple combo
  6. Signal gate on WS+LSTM combos
  7. All 6 models: WS + LSTM + Freq + HotNumber + Pattern + Markov

Walk-forward: train on data[:i], predict data[i], advance.
LSTM retrained every RETRAIN_INTERVAL spins (50).
"""

import sys
import os
import time
import numpy as np
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    TOTAL_NUMBERS, WHEEL_ORDER, NUMBER_TO_POSITION,
    WHEEL_TABLE_0, WHEEL_TABLE_19,
    WHEEL_POSITIVE, WHEEL_NEGATIVE,
    WHEEL_SET_1, WHEEL_SET_2, WHEEL_SET_3,
    WHEEL_SECTORS, USERDATA_DIR,
    FREQUENCY_RECENT_WINDOW,
    HOT_NUMBER_WINDOW, HOT_NUMBER_BOOST_FACTOR,
    LSTM_SEQUENCE_LENGTH, RETRAIN_INTERVAL,
)

# Import the actual production models
from app.ml.lstm_predictor import LSTMPredictor
from app.ml.wheel_strategy import WheelStrategyAnalyzer
from app.ml.frequency_analyzer import FrequencyAnalyzer
from app.ml.hot_number import HotNumberAnalyzer
from app.ml.markov_chain import MarkovChain
from app.ml.pattern_detector import PatternDetector

TOP_N = 12
WARMUP = 100
BET_PER_NUM = 2.0
PAYOUT = 35


# ─── Data Loading ──────────────────────────────────────────────────────
def load_all_userdata():
    all_spins = []
    for fname in sorted(os.listdir(USERDATA_DIR)):
        if fname.endswith('.txt'):
            fpath = os.path.join(USERDATA_DIR, fname)
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        n = int(line)
                        if 0 <= n <= 36:
                            all_spins.append(n)
    return all_spins


# ─── Wheel signal strength (for signal gate) ──────────────────────────
def wheel_signal_strength(history, window=50):
    if len(history) < 5:
        return 0.0
    recent = history[-window:]
    total = len(recent)
    scores = []

    t0 = sum(1 for n in recent if n in WHEEL_TABLE_0)
    t0_dev = abs(t0/total - 19/37)
    scores.append(min(100, t0_dev / 0.25 * 100))

    pos = sum(1 for n in recent if n in WHEEL_POSITIVE)
    pos_dev = abs(pos/total - 19/37)
    scores.append(min(100, pos_dev / 0.25 * 100))

    s_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        if n in WHEEL_SET_1: s_counts[1] += 1
        elif n in WHEEL_SET_2: s_counts[2] += 1
        else: s_counts[3] += 1
    set_pcts = [s_counts[k]/total for k in [1,2,3]]
    max_dev = max(abs(p - 1/3) for p in set_pcts)
    scores.append(min(100, max_dev / 0.20 * 100))

    return sum(scores) / len(scores)


# ─── Walk-forward with real LSTM training ──────────────────────────────
def walk_forward_with_lstm(data, warmup=WARMUP):
    """
    Walk-forward backtest using the ACTUAL production models.
    LSTM is trained every RETRAIN_INTERVAL spins.

    Returns per-spin predictions from each model and the ensemble combos.
    """
    n = len(data)

    # Initialize production models
    lstm = LSTMPredictor()
    wheel = WheelStrategyAnalyzer()
    freq = FrequencyAnalyzer()
    hot = HotNumberAnalyzer()
    markov = MarkovChain()
    pattern = PatternDetector()

    # Feed warmup data
    warmup_data = data[:warmup]
    freq.load_history(warmup_data)
    markov.load_history(warmup_data)
    pattern.load_history(warmup_data)
    wheel.load_history(warmup_data)
    hot.load_history(warmup_data)
    lstm.load_history(warmup_data)

    # Train LSTM on warmup
    print(f"  [LSTM] Initial training on {warmup} spins...", end='', flush=True)
    train_result = lstm.train(epochs=50)
    print(f" done. Status: {train_result.get('status')}, "
          f"epochs: {train_result.get('epochs', 0)}, "
          f"loss: {train_result.get('final_loss', 'N/A')}")

    spins_since_train = 0

    # Storage for results
    results = {
        'lstm_solo': [],
        'wheel_solo': [],
        'freq_solo': [],
        'hot_solo': [],
        'markov_solo': [],
        'pattern_solo': [],
        'actuals': [],
        'signals': [],
        # Store raw probabilities for combo computation
        'lstm_probs_list': [],
        'wheel_probs_list': [],
        'freq_probs_list': [],
        'hot_probs_list': [],
        'markov_probs_list': [],
        'pattern_probs_list': [],
    }

    retrain_count = 0

    for i in range(warmup, n):
        actual = data[i]
        history = data[:i]

        # Get probabilities from each model
        lstm_probs = lstm.predict()
        wheel_probs = wheel.get_number_probabilities()
        freq_probs = freq.get_number_probabilities()
        hot_probs = hot.get_number_probabilities()
        markov_probs_arr = markov.get_probabilities()
        pattern_probs = pattern.get_number_probabilities()

        # Signal strength
        sig = wheel.get_strategy_strength()

        # Top-12 for each model
        lstm_top = set(np.argsort(lstm_probs)[::-1][:TOP_N])
        wheel_top = set(np.argsort(wheel_probs)[::-1][:TOP_N])
        freq_top = set(np.argsort(freq_probs)[::-1][:TOP_N])
        hot_top = set(np.argsort(hot_probs)[::-1][:TOP_N])
        markov_top = set(np.argsort(markov_probs_arr)[::-1][:TOP_N])
        pattern_top = set(np.argsort(pattern_probs)[::-1][:TOP_N])

        # Record hits
        results['lstm_solo'].append(1 if actual in lstm_top else 0)
        results['wheel_solo'].append(1 if actual in wheel_top else 0)
        results['freq_solo'].append(1 if actual in freq_top else 0)
        results['hot_solo'].append(1 if actual in hot_top else 0)
        results['markov_solo'].append(1 if actual in markov_top else 0)
        results['pattern_solo'].append(1 if actual in pattern_top else 0)
        results['actuals'].append(actual)
        results['signals'].append(sig)

        # Store probability arrays for combo computation
        results['lstm_probs_list'].append(lstm_probs.copy())
        results['wheel_probs_list'].append(wheel_probs.copy())
        results['freq_probs_list'].append(freq_probs.copy())
        results['hot_probs_list'].append(hot_probs.copy())
        results['markov_probs_list'].append(markov_probs_arr.copy())
        results['pattern_probs_list'].append(pattern_probs.copy())

        # Update all models with actual number
        freq.update(actual)
        markov.update(actual)
        pattern.update(actual)
        wheel.update(actual)
        hot.update(actual)
        lstm.update(actual)
        spins_since_train += 1

        # Retrain LSTM periodically
        if spins_since_train >= RETRAIN_INTERVAL and lstm.can_train():
            retrain_count += 1
            if retrain_count % 5 == 0:
                print(f"  [LSTM] Retrain #{retrain_count} at spin {i} "
                      f"({len(lstm.spin_history)} total)...", end='', flush=True)
            train_result = lstm.train(epochs=50)
            spins_since_train = 0
            if retrain_count % 5 == 0:
                print(f" loss={train_result.get('final_loss', 'N/A')}")

        # Progress
        if (i - warmup) % 500 == 0 and i > warmup:
            pct = (i - warmup) / (n - warmup) * 100
            lstm_hr = sum(results['lstm_solo']) / len(results['lstm_solo']) * 100
            print(f"  Progress: {pct:.0f}% ({i-warmup}/{n-warmup}) "
                  f"LSTM hit: {lstm_hr:.1f}%  retrains: {retrain_count}")

    print(f"  Walk-forward complete. Total retrains: {retrain_count}")
    return results


def compute_combo_hits(results, model_weights, signal_gate=0):
    """
    Compute hit stats for a weighted combo from stored probability arrays.

    model_weights: dict of model_name -> weight (e.g. {'lstm': 0.5, 'wheel': 0.5})
    """
    model_probs_keys = {
        'lstm': 'lstm_probs_list',
        'wheel': 'wheel_probs_list',
        'freq': 'freq_probs_list',
        'hot': 'hot_probs_list',
        'markov': 'markov_probs_list',
        'pattern': 'pattern_probs_list',
    }

    hits = 0
    total_bets = 0
    skipped = 0
    bankroll = 4000.0
    peak = 4000.0
    max_dd = 0
    hit_log = []

    n_spins = len(results['actuals'])
    for idx in range(n_spins):
        # Signal gate
        if signal_gate > 0:
            if results['signals'][idx] < signal_gate:
                skipped += 1
                continue

        # Weighted ensemble
        ensemble = np.zeros(TOTAL_NUMBERS)
        for model_name, weight in model_weights.items():
            if weight > 0:
                probs = results[model_probs_keys[model_name]][idx]
                ensemble += weight * probs

        total = ensemble.sum()
        if total > 0:
            ensemble /= total

        top = set(np.argsort(ensemble)[::-1][:TOP_N])
        actual = results['actuals'][idx]

        hit = 1 if actual in top else 0
        hits += hit
        total_bets += 1
        hit_log.append(hit)

        cost = TOP_N * BET_PER_NUM
        if hit:
            bankroll += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
        else:
            bankroll -= cost

        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

    if total_bets == 0:
        return None

    hit_rate = hits / total_bets * 100
    baseline = TOP_N / 37 * 100
    edge = hit_rate - baseline
    total_cost = total_bets * TOP_N * BET_PER_NUM
    total_win = hits * (PAYOUT + 1) * BET_PER_NUM
    profit = total_win - total_cost

    # Streaks
    max_win = 0; max_loss = 0; cur = 0
    for h in hit_log:
        if h:
            cur = cur + 1 if cur > 0 else 1
            max_win = max(max_win, cur)
        else:
            cur = cur - 1 if cur < 0 else -1
            max_loss = max(max_loss, abs(cur))

    bet_pct = total_bets / n_spins * 100

    return {
        'bets': total_bets,
        'skipped': skipped,
        'bet_pct': round(bet_pct, 1),
        'hits': hits,
        'hit_rate': round(hit_rate, 2),
        'baseline': round(baseline, 2),
        'edge': round(edge, 2),
        'profit': round(profit, 2),
        'final_bankroll': round(bankroll, 2),
        'max_drawdown': round(max_dd, 2),
        'max_win_streak': max_win,
        'max_loss_streak': max_loss,
    }


def print_result(label, r):
    if r is None:
        print(f"  {label:<45} — NO DATA")
        return
    print(f"  {label:<45} {r['hit_rate']:>6.2f}% {r['edge']:>+6.2f}% "
          f"${r['profit']:>+8.0f}  final=${r['final_bankroll']:>7.0f}  "
          f"DD=${r['max_drawdown']:>6.0f}  W{r['max_win_streak']} L{r['max_loss_streak']}")


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 100)
    print("WHEEL STRATEGY + LSTM BACKTEST (with actual LSTM training)")
    print(f"Top-{TOP_N} numbers | ${BET_PER_NUM}/number | {PAYOUT}:1 payout")
    print("=" * 100)

    data = load_all_userdata()
    n_preds = len(data) - WARMUP
    print(f"\nData: {len(data)} spins | Warmup: {WARMUP} | Predictions: {n_preds}")
    print(f"Baseline: {TOP_N}/37 = {TOP_N/37*100:.2f}% | Breakeven: {TOP_N}/36 = {TOP_N/36*100:.2f}%")
    print(f"LSTM retrain every {RETRAIN_INTERVAL} spins, 50 epochs per train\n")

    start = time.time()

    # ═══ RUN WALK-FORWARD ═══
    print("Running walk-forward (this takes a while due to LSTM training)...")
    results = walk_forward_with_lstm(data)

    elapsed_wf = time.time() - start
    print(f"\nWalk-forward completed in {elapsed_wf:.0f}s\n")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: EACH MODEL SOLO
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("SECTION 1: EACH MODEL SOLO (top-12)")
    print("=" * 100)
    print(f"  {'Model':<45} {'Hit%':>7} {'Edge':>7} {'Profit':>9}  {'Final$':>9}  "
          f"{'MaxDD':>8}  {'Streaks'}")
    print("-" * 100)

    solo_models = {
        'LSTM (trained)': {'lstm': 1.0},
        'WheelStrategy': {'wheel': 1.0},
        'Frequency': {'freq': 1.0},
        'HotNumber': {'hot': 1.0},
        'Markov': {'markov': 1.0},
        'Pattern': {'pattern': 1.0},
    }

    for name, weights in solo_models.items():
        r = compute_combo_hits(results, weights)
        print_result(name, r)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: WS + LSTM weight sweep
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 2: WHEEL STRATEGY + LSTM (weight sweep)")
    print("=" * 100)
    print(f"  {'Combo':<45} {'Hit%':>7} {'Edge':>7} {'Profit':>9}  {'Final$':>9}  "
          f"{'MaxDD':>8}  {'Streaks'}")
    print("-" * 100)

    for ws_w in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        lstm_w = 100 - ws_w
        weights = {'wheel': ws_w/100, 'lstm': lstm_w/100}
        r = compute_combo_hits(results, weights)
        label = f"WS({ws_w}%) + LSTM({lstm_w}%)"
        print_result(label, r)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: WS + LSTM + Frequency triple
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 3: WS + LSTM + FREQUENCY (triple combo)")
    print("=" * 100)
    print(f"  {'Combo':<45} {'Hit%':>7} {'Edge':>7} {'Profit':>9}  {'Final$':>9}  "
          f"{'MaxDD':>8}  {'Streaks'}")
    print("-" * 100)

    for ws_w in range(10, 70, 10):
        for lstm_w in range(10, 80 - ws_w, 10):
            freq_w = 100 - ws_w - lstm_w
            if freq_w < 10:
                continue
            weights = {'wheel': ws_w/100, 'lstm': lstm_w/100, 'freq': freq_w/100}
            r = compute_combo_hits(results, weights)
            label = f"WS({ws_w}%)+LSTM({lstm_w}%)+Freq({freq_w}%)"
            print_result(label, r)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: WS + LSTM + each other model
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 4: WS + LSTM + each other model (best weight search)")
    print("=" * 100)

    other_models = ['freq', 'hot', 'markov', 'pattern']
    other_labels = {'freq': 'Frequency', 'hot': 'HotNumber', 'markov': 'Markov', 'pattern': 'Pattern'}

    for other in other_models:
        print(f"\n  --- WS + LSTM + {other_labels[other]} ---")
        print(f"  {'Combo':<45} {'Hit%':>7} {'Edge':>7} {'Profit':>9}  {'Final$':>9}")
        print("  " + "-" * 90)

        best_edge = -999
        best_label = ""

        for ws_w in range(10, 70, 10):
            for lstm_w in range(10, 80 - ws_w, 10):
                other_w = 100 - ws_w - lstm_w
                if other_w < 10:
                    continue
                weights = {'wheel': ws_w/100, 'lstm': lstm_w/100, other: other_w/100}
                r = compute_combo_hits(results, weights)
                if r and r['edge'] > best_edge:
                    best_edge = r['edge']
                    best_r = r
                    best_label = f"WS({ws_w}%)+LSTM({lstm_w}%)+{other_labels[other]}({other_w}%)"

        if best_edge > -999:
            print_result(f"BEST: {best_label}", best_r)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: ALL 6 MODELS ensemble sweep
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 5: FULL 6-MODEL ENSEMBLE (selected weight combos)")
    print("=" * 100)
    print(f"  {'Combo':<45} {'Hit%':>7} {'Edge':>7} {'Profit':>9}  {'Final$':>9}  "
          f"{'MaxDD':>8}")
    print("-" * 100)

    # Test the current production weights
    production_weights = {
        'freq': 0.562,
        'wheel': 0.306,
        'hot': 0.08,
        'pattern': 0.052,
        'markov': 0.0,
        'lstm': 0.0,
    }
    r = compute_combo_hits(results, production_weights)
    print_result("CURRENT PRODUCTION (no LSTM)", r)

    # Production with LSTM enabled at various weights
    for lstm_pct in [5, 10, 15, 20, 30]:
        lstm_w = lstm_pct / 100
        scale = 1.0 - lstm_w
        w = {
            'freq': 0.562 * scale,
            'wheel': 0.306 * scale,
            'hot': 0.08 * scale,
            'pattern': 0.052 * scale,
            'markov': 0.0,
            'lstm': lstm_w,
        }
        r = compute_combo_hits(results, w)
        print_result(f"Production + LSTM({lstm_pct}%)", r)

    # Optimized combos — test a few promising configurations
    test_configs = [
        ("Freq(90%)+WS(10%)", {'freq': 0.90, 'wheel': 0.10}),
        ("Freq(80%)+WS(10%)+LSTM(10%)", {'freq': 0.80, 'wheel': 0.10, 'lstm': 0.10}),
        ("Freq(70%)+WS(10%)+LSTM(20%)", {'freq': 0.70, 'wheel': 0.10, 'lstm': 0.20}),
        ("Freq(60%)+WS(10%)+LSTM(30%)", {'freq': 0.60, 'wheel': 0.10, 'lstm': 0.30}),
        ("Freq(50%)+WS(10%)+LSTM(40%)", {'freq': 0.50, 'wheel': 0.10, 'lstm': 0.40}),
        ("Freq(70%)+LSTM(30%)", {'freq': 0.70, 'lstm': 0.30}),
        ("Freq(50%)+LSTM(50%)", {'freq': 0.50, 'lstm': 0.50}),
        ("LSTM(50%)+WS(30%)+Freq(20%)", {'lstm': 0.50, 'wheel': 0.30, 'freq': 0.20}),
        ("Freq(60%)+WS(20%)+LSTM(10%)+Hot(10%)", {'freq': 0.60, 'wheel': 0.20, 'lstm': 0.10, 'hot': 0.10}),
        ("Freq(50%)+WS(20%)+LSTM(20%)+Hot(10%)", {'freq': 0.50, 'wheel': 0.20, 'lstm': 0.20, 'hot': 0.10}),
    ]

    print()
    for label, w in test_configs:
        r = compute_combo_hits(results, w)
        print_result(label, r)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 6: SIGNAL GATE on WS+LSTM combos
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 6: SIGNAL GATE on WS+LSTM combos")
    print("Only bet when wheel signal >= threshold")
    print("=" * 100)

    gate_combos = [
        ("WS(10%)+LSTM(90%)", {'wheel': 0.10, 'lstm': 0.90}),
        ("WS(30%)+LSTM(70%)", {'wheel': 0.30, 'lstm': 0.70}),
        ("WS(50%)+LSTM(50%)", {'wheel': 0.50, 'lstm': 0.50}),
        ("WS(10%)+LSTM(10%)+Freq(80%)", {'wheel': 0.10, 'lstm': 0.10, 'freq': 0.80}),
        ("Freq(90%)+WS(10%) [no LSTM]", {'freq': 0.90, 'wheel': 0.10}),
    ]

    for combo_label, weights in gate_combos:
        print(f"\n  --- {combo_label} ---")
        print(f"  {'Gate':>5} {'Bets':>5} {'Bet%':>6} {'Hit%':>7} {'Edge':>7} "
              f"{'Profit':>9} {'$/Spin':>7}")
        print("  " + "-" * 60)

        for gate in [0, 15, 20, 25, 30, 35, 40]:
            r = compute_combo_hits(results, weights, signal_gate=gate)
            if r and r['bets'] > 30:
                ps = r['profit'] / r['bets'] if r['bets'] > 0 else 0
                print(f"  {gate:>5} {r['bets']:>5} {r['bet_pct']:>6.1f} {r['hit_rate']:>7.2f} "
                      f"{r['edge']:>+7.2f} {r['profit']:>+9.0f} {ps:>+7.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start

    print("\n" + "=" * 100)
    print("GRAND SUMMARY — Does LSTM help or hurt?")
    print("=" * 100)

    # Compare key configs
    key_configs = [
        ("LSTM solo", {'lstm': 1.0}),
        ("WheelStrategy solo", {'wheel': 1.0}),
        ("Frequency solo", {'freq': 1.0}),
        ("WS(10%)+Freq(90%) [BEST no-LSTM]", {'freq': 0.90, 'wheel': 0.10}),
        ("WS(10%)+LSTM(90%)", {'wheel': 0.10, 'lstm': 0.90}),
        ("WS(50%)+LSTM(50%)", {'wheel': 0.50, 'lstm': 0.50}),
        ("Freq(80%)+WS(10%)+LSTM(10%)", {'freq': 0.80, 'wheel': 0.10, 'lstm': 0.10}),
        ("Freq(70%)+WS(10%)+LSTM(20%)", {'freq': 0.70, 'wheel': 0.10, 'lstm': 0.20}),
        ("CURRENT PRODUCTION", production_weights),
    ]

    print(f"\n  {'Config':<45} {'Hit%':>7} {'Edge':>7} {'Profit':>9}")
    print("  " + "-" * 75)

    ranked = []
    for label, w in key_configs:
        r = compute_combo_hits(results, w)
        if r:
            ranked.append((label, r))
            print(f"  {label:<45} {r['hit_rate']:>7.2f} {r['edge']:>+7.2f} {r['profit']:>+9.0f}")

    ranked.sort(key=lambda x: x[1]['edge'], reverse=True)

    print(f"\n  RANKED BY EDGE:")
    for i, (label, r) in enumerate(ranked):
        marker = " ◀ WINNER" if i == 0 else ""
        print(f"    {i+1}. {label:<43} edge={r['edge']:>+.2f}%  "
              f"profit=${r['profit']:>+.0f}{marker}")

    # Does LSTM help?
    no_lstm = compute_combo_hits(results, {'freq': 0.90, 'wheel': 0.10})
    with_lstm = compute_combo_hits(results, {'freq': 0.80, 'wheel': 0.10, 'lstm': 0.10})

    print(f"\n  LSTM IMPACT TEST:")
    print(f"    Without LSTM (Freq90+WS10):   {no_lstm['hit_rate']:.2f}% hit, "
          f"edge={no_lstm['edge']:+.2f}%, profit=${no_lstm['profit']:+.0f}")
    print(f"    With LSTM 10% (Freq80+WS10+LSTM10): {with_lstm['hit_rate']:.2f}% hit, "
          f"edge={with_lstm['edge']:+.2f}%, profit=${with_lstm['profit']:+.0f}")

    diff = with_lstm['edge'] - no_lstm['edge']
    profit_diff = with_lstm['profit'] - no_lstm['profit']
    if diff > 0:
        print(f"    → LSTM HELPS: +{diff:.2f}% edge, +${profit_diff:.0f} profit")
    elif diff < 0:
        print(f"    → LSTM HURTS: {diff:.2f}% edge, ${profit_diff:.0f} profit")
    else:
        print(f"    → LSTM NO EFFECT")

    print(f"\nTotal backtest time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
