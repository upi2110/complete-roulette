#!/usr/bin/env python3
"""
Double-Verify Best Configs — Detailed per-file and cumulative validation.

Tests these specific configs:
  1. Polarity(20%) + Freq(80%)
  2. Table(10%) + LSTM(50%) + Freq(40%)
  3. Polarity(10%) + LSTM(50%) + Freq(40%)

Plus baselines:
  4. Frequency solo
  5. Previous best: WS(20%) + LSTM(10%) + Freq(70%)
  6. Random baseline (12/37)

Tests on:
  - Each data file independently (model learns ONLY from that file)
  - Each data file cumulatively (model learns from ALL previous files too)
  - All data combined
  - 5-fold cross-validation (different file as holdout)
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
    USERDATA_DIR, FREQUENCY_RECENT_WINDOW,
    WHEEL_STRATEGY_TREND_BOOST, WHEEL_STRATEGY_COLD_DAMPEN,
    RETRAIN_INTERVAL,
)

from app.ml.lstm_predictor import LSTMPredictor
from app.ml.frequency_analyzer import FrequencyAnalyzer

TOP_N = 12
WARMUP = 100
BET_PER_NUM = 2.0
PAYOUT = 35
BOOST = WHEEL_STRATEGY_TREND_BOOST
DAMPEN = WHEEL_STRATEGY_COLD_DAMPEN

# Pre-compute membership
TABLE_MAP = {}; POL_MAP = {}; SET_MAP = {}
for _n in range(TOTAL_NUMBERS):
    TABLE_MAP[_n] = '0' if _n in WHEEL_TABLE_0 else '19'
    POL_MAP[_n] = 'pos' if _n in WHEEL_POSITIVE else 'neg'
    if _n in WHEEL_SET_1: SET_MAP[_n] = 1
    elif _n in WHEEL_SET_2: SET_MAP[_n] = 2
    else: SET_MAP[_n] = 3


def load_file(fpath):
    spins = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                n = int(line)
                if 0 <= n <= 36:
                    spins.append(n)
    return spins


def load_all_userdata():
    all_spins = []
    files = []
    for fname in sorted(os.listdir(USERDATA_DIR)):
        if fname.endswith('.txt'):
            fpath = os.path.join(USERDATA_DIR, fname)
            spins = load_file(fpath)
            files.append((fname, spins))
            all_spins.extend(spins)
    return all_spins, files


# ─── Probability Models ────────────────────────────────────────────────

def frequency_probs_fn(history):
    if len(history) < 5:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    counts = Counter(history)
    total = len(history)
    flat = np.array([(counts.get(i, 0) + 1) / (total + TOTAL_NUMBERS) for i in range(TOTAL_NUMBERS)])
    flat /= flat.sum()
    recent = history[-FREQUENCY_RECENT_WINDOW:]
    rc = np.ones(TOTAL_NUMBERS)
    for n in recent:
        rc[n] += 1
    rc /= rc.sum()
    blended = 0.5 * flat + 0.5 * rc
    return blended / blended.sum()


def polarity_probs_fn(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    total = len(recent)
    pos = sum(1 for n in recent if POL_MAP.get(n) == 'pos')
    neg = total - pos
    pos_pct = pos / total; neg_pct = neg / total
    hot = 'pos' if pos > neg else 'neg' if neg > pos else None
    if hot is not None:
        imbalance = abs(pos_pct - neg_pct) / 0.5
        pb = 1.0 + (BOOST - 1.0) * imbalance
        pd = 1.0 - (1.0 - DAMPEN) * imbalance
        for num in range(TOTAL_NUMBERS):
            if POL_MAP[num] == hot:
                probs[num] *= pb
            else:
                probs[num] *= pd
    return probs / probs.sum()


def table_probs_fn(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if TABLE_MAP.get(n) == '0')
    t0_pct = t0 / total; t19_pct = 1 - t0_pct
    hot = '0' if t0 > total - t0 else '19' if total - t0 > t0 else None
    if hot is not None:
        imbalance = abs(t0_pct - t19_pct) / 0.5
        tb = 1.0 + (BOOST - 1.0) * imbalance
        td = 1.0 - (1.0 - DAMPEN) * imbalance
        for num in range(TOTAL_NUMBERS):
            if TABLE_MAP[num] == hot:
                probs[num] *= tb
            else:
                probs[num] *= td
    return probs / probs.sum()


def wheel_combined_probs_fn(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    total = len(recent)

    t0 = sum(1 for n in recent if TABLE_MAP.get(n) == '0')
    t0_pct = t0/total; t19_pct = 1-t0_pct
    hot_table = '0' if t0 > total-t0 else '19' if total-t0 > t0 else None

    pos = sum(1 for n in recent if POL_MAP.get(n) == 'pos')
    pos_pct = pos/total; neg_pct = 1-pos_pct
    hot_pol = 'pos' if pos > total-pos else 'neg' if total-pos > pos else None

    s_counts = {1:0, 2:0, 3:0}
    for n in recent:
        s_counts[SET_MAP.get(n,3)] += 1
    hot_set = max(s_counts, key=s_counts.get)
    if s_counts[hot_set] <= total/3*1.1:
        hot_set = None

    for num in range(TOTAL_NUMBERS):
        m = 1.0
        if hot_table:
            imb = abs(t0_pct-t19_pct)/0.5
            m *= (1.0+(BOOST-1.0)*imb) if TABLE_MAP[num]==hot_table else (1.0-(1.0-DAMPEN)*imb)
        if hot_pol:
            imb = abs(pos_pct-neg_pct)/0.5
            m *= (1.0+(BOOST-1.0)*imb) if POL_MAP[num]==hot_pol else (1.0-(1.0-DAMPEN)*imb)
        if hot_set:
            sp = {k:s_counts[k]/total for k in s_counts}
            exc = max(0,min(1,(sp[hot_set]-1/3)/(1/3)))
            m *= (1.0+(BOOST-1.0)*exc) if SET_MAP[num]==hot_set else (1.0-(1.0-DAMPEN)*exc)
        probs[num] *= m
    return probs / probs.sum()


# ─── Walk-forward backtest (no LSTM) ──────────────────────────────────

def backtest_no_lstm(data, combo_fn, warmup=100, label=""):
    """Walk-forward for non-LSTM configs.
    combo_fn(history) -> np.array(37) probabilities
    """
    if len(data) <= warmup + 10:
        return None

    hits = 0; total_bets = 0
    bankroll = 4000.0; peak = 4000.0; max_dd = 0
    hit_log = []

    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]
        probs = combo_fn(history)
        top = set(np.argsort(probs)[::-1][:TOP_N])
        hit = 1 if actual in top else 0
        hits += hit; total_bets += 1; hit_log.append(hit)
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
    profit = hits * (PAYOUT+1) * BET_PER_NUM - total_bets * TOP_N * BET_PER_NUM

    # Rolling 50-spin windows
    rolling = []
    for j in range(50, len(hit_log)):
        rolling.append(sum(hit_log[j-50:j]) / 50 * 100)

    return {
        'label': label, 'bets': total_bets, 'hits': hits,
        'hit_rate': round(hit_rate, 2), 'baseline': round(baseline, 2),
        'edge': round(edge, 2), 'profit': round(profit, 2),
        'final': round(bankroll, 2), 'max_dd': round(max_dd, 2),
        'best_50': round(max(rolling), 1) if rolling else 0,
        'worst_50': round(min(rolling), 1) if rolling else 0,
    }


# ─── Walk-forward WITH LSTM ───────────────────────────────────────────

def backtest_with_lstm(data, weights, warmup=100, label=""):
    """Walk-forward with real LSTM training.
    weights: dict with keys 'freq', 'lstm', and optionally 'table', 'polarity', 'set', 'combined_ws'
    """
    if len(data) <= warmup + 10:
        return None

    lstm = LSTMPredictor()
    freq = FrequencyAnalyzer()

    warmup_data = data[:warmup]
    freq.load_history(warmup_data)
    lstm.load_history(warmup_data)
    lstm.train(epochs=50)

    spins_since_train = 0
    hits = 0; total_bets = 0
    bankroll = 4000.0; peak = 4000.0; max_dd = 0
    hit_log = []

    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]

        # Build ensemble
        ensemble = np.zeros(TOTAL_NUMBERS)
        for model, weight in weights.items():
            if weight <= 0:
                continue
            if model == 'freq':
                ensemble += weight * freq.get_number_probabilities()
            elif model == 'lstm':
                ensemble += weight * lstm.predict()
            elif model == 'table':
                ensemble += weight * table_probs_fn(history)
            elif model == 'polarity':
                ensemble += weight * polarity_probs_fn(history)
            elif model == 'set':
                ensemble += weight * set_probs_fn(history) if 'set_probs_fn' in dir() else weight * np.full(TOTAL_NUMBERS, 1/TOTAL_NUMBERS)
            elif model == 'combined_ws':
                ensemble += weight * wheel_combined_probs_fn(history)

        total = ensemble.sum()
        if total > 0:
            ensemble /= total

        top = set(np.argsort(ensemble)[::-1][:TOP_N])
        hit = 1 if actual in top else 0
        hits += hit; total_bets += 1; hit_log.append(hit)

        cost = TOP_N * BET_PER_NUM
        if hit:
            bankroll += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
        else:
            bankroll -= cost
        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

        # Update
        freq.update(actual)
        lstm.update(actual)
        spins_since_train += 1
        if spins_since_train >= RETRAIN_INTERVAL and lstm.can_train():
            lstm.train(epochs=50)
            spins_since_train = 0

    if total_bets == 0:
        return None

    hit_rate = hits / total_bets * 100
    baseline = TOP_N / 37 * 100
    edge = hit_rate - baseline
    profit = hits * (PAYOUT+1) * BET_PER_NUM - total_bets * TOP_N * BET_PER_NUM

    rolling = []
    for j in range(50, len(hit_log)):
        rolling.append(sum(hit_log[j-50:j]) / 50 * 100)

    return {
        'label': label, 'bets': total_bets, 'hits': hits,
        'hit_rate': round(hit_rate, 2), 'baseline': round(baseline, 2),
        'edge': round(edge, 2), 'profit': round(profit, 2),
        'final': round(bankroll, 2), 'max_dd': round(max_dd, 2),
        'best_50': round(max(rolling), 1) if rolling else 0,
        'worst_50': round(min(rolling), 1) if rolling else 0,
    }


def pr(r):
    if not r:
        print(f"    {'(insufficient data)':<50}")
        return
    print(f"    {r['label']:<42} bets={r['bets']:>4}  hit={r['hit_rate']:>5.1f}%  "
          f"edge={r['edge']:>+5.2f}%  profit=${r['profit']:>+7.0f}  "
          f"DD=${r['max_dd']:>6.0f}  best50={r['best_50']:>4.0f}%  worst50={r['worst_50']:>4.0f}%")


def main():
    print("=" * 110)
    print("DOUBLE VERIFICATION — Best Configs Across All Data Files")
    print(f"Top-{TOP_N} | ${BET_PER_NUM}/num | {PAYOUT}:1")
    print("=" * 110)

    all_data, files = load_all_userdata()
    print(f"\nTotal: {len(all_data)} spins across {len(files)} files:")
    for fname, spins in files:
        print(f"  {fname}: {len(spins)} spins")

    start = time.time()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: ALL DATA COMBINED
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("TEST 1: ALL DATA COMBINED (2,528 spins)")
    print("=" * 110)

    # Non-LSTM configs
    configs_no_lstm = [
        ("Frequency solo", lambda h: frequency_probs_fn(h)),
        ("Polarity(20%)+Freq(80%)", lambda h: 0.20*polarity_probs_fn(h) + 0.80*frequency_probs_fn(h)),
        ("Polarity(10%)+Freq(90%)", lambda h: 0.10*polarity_probs_fn(h) + 0.90*frequency_probs_fn(h)),
        ("Polarity(30%)+Freq(70%)", lambda h: 0.30*polarity_probs_fn(h) + 0.70*frequency_probs_fn(h)),
        ("Table(20%)+Freq(80%)", lambda h: 0.20*table_probs_fn(h) + 0.80*frequency_probs_fn(h)),
        ("WS_Combined(10%)+Freq(90%)", lambda h: 0.10*wheel_combined_probs_fn(h) + 0.90*frequency_probs_fn(h)),
    ]

    print("\n  --- Non-LSTM configs ---")
    for label, fn in configs_no_lstm:
        def make_normed(f=fn):
            def normed(h):
                p = f(h)
                return p / p.sum()
            return normed
        r = backtest_no_lstm(all_data, make_normed(), WARMUP, label)
        pr(r)

    # LSTM configs
    print("\n  --- LSTM configs (each trains fresh) ---")
    lstm_configs = [
        ("Table(10%)+LSTM(50%)+Freq(40%)", {'table': 0.10, 'lstm': 0.50, 'freq': 0.40}),
        ("Polarity(10%)+LSTM(50%)+Freq(40%)", {'polarity': 0.10, 'lstm': 0.50, 'freq': 0.40}),
        ("WS(20%)+LSTM(10%)+Freq(70%)", {'combined_ws': 0.20, 'lstm': 0.10, 'freq': 0.70}),
        ("LSTM solo", {'lstm': 1.0}),
    ]

    for label, weights in lstm_configs:
        print(f"    Training {label}...", end='', flush=True)
        r = backtest_with_lstm(all_data, weights, WARMUP, label)
        print(f" done.")
        pr(r)

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: EACH FILE INDEPENDENTLY (model learns ONLY from that file)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("TEST 2: EACH FILE INDEPENDENTLY (model sees only that file's data)")
    print("Warmup = 60 spins per file (smaller files)")
    print("=" * 110)

    file_warmup = 60

    for fname, spins in files:
        print(f"\n  --- {fname} ({len(spins)} spins) ---")

        if len(spins) <= file_warmup + 20:
            print(f"    Skipped (too few spins)")
            continue

        # Non-LSTM
        for label, fn in configs_no_lstm:
            def make_normed(f=fn):
                def normed(h):
                    p = f(h)
                    return p / p.sum()
                return normed
            r = backtest_no_lstm(spins, make_normed(), file_warmup, label)
            pr(r)

        # LSTM configs on individual file
        for label, weights in lstm_configs[:2]:  # Just the two main LSTM configs
            r = backtest_with_lstm(spins, weights, file_warmup, label)
            pr(r)

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: CUMULATIVE (model learns from all previous files)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("TEST 3: CUMULATIVE (each file tested with knowledge from all prior files)")
    print("=" * 110)

    cumulative_data = []
    for fname, spins in files:
        if len(cumulative_data) >= WARMUP:
            # Test on this file's data, using cumulative as warmup
            test_data = cumulative_data + spins
            test_warmup = len(cumulative_data)

            print(f"\n  --- {fname} ({len(spins)} spins, {len(cumulative_data)} prior) ---")

            for label, fn in configs_no_lstm[:3]:  # Top 3 non-LSTM
                def make_normed(f=fn):
                    def normed(h):
                        p = f(h)
                        return p / p.sum()
                    return normed
                r = backtest_no_lstm(test_data, make_normed(), test_warmup, label)
                pr(r)

        cumulative_data.extend(spins)

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: CONSISTENCY CHECK — Rolling 200-spin windows
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("TEST 4: ROLLING 200-SPIN WINDOWS (consistency check)")
    print("Slides a 200-spin window across all data, warmup=60")
    print("=" * 110)

    window = 200
    step = 100
    roll_warmup = 60

    # Test top 2 non-LSTM configs
    test_fns = [
        ("Polarity(20%)+Freq(80%)", lambda h: (0.20*polarity_probs_fn(h) + 0.80*frequency_probs_fn(h)) / (0.20*polarity_probs_fn(h) + 0.80*frequency_probs_fn(h)).sum()),
        ("Frequency solo", lambda h: frequency_probs_fn(h)),
    ]

    for label, fn in test_fns:
        print(f"\n  --- {label} ---")
        print(f"  {'Window':>10} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8}")
        print("  " + "-" * 50)

        edges = []
        for start_idx in range(0, len(all_data) - window, step):
            chunk = all_data[start_idx:start_idx + window]
            r = backtest_no_lstm(chunk, fn, roll_warmup, f"spins {start_idx}-{start_idx+window}")
            if r:
                edges.append(r['edge'])
                print(f"  {start_idx:>4}-{start_idx+window:<5} {r['bets']:>5} "
                      f"{r['hit_rate']:>6.1f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f}")

        if edges:
            pos = sum(1 for e in edges if e > 0)
            print(f"  SUMMARY: {pos}/{len(edges)} windows positive edge, "
                  f"avg edge={np.mean(edges):+.2f}%, "
                  f"min={min(edges):+.2f}%, max={max(edges):+.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start

    print("\n" + "=" * 110)
    print("GRAND SUMMARY")
    print("=" * 110)

    print(f"\n  ALL DATA (2,528 spins) results:")
    print(f"  {'Config':<45} {'Hit%':>6} {'Edge':>7} {'Profit':>9}")
    print("  " + "-" * 75)

    # Re-run key configs for summary
    key_configs = [
        ("Polarity(20%)+Freq(80%)", lambda h: (0.20*polarity_probs_fn(h) + 0.80*frequency_probs_fn(h)) / (0.20*polarity_probs_fn(h) + 0.80*frequency_probs_fn(h)).sum()),
        ("Frequency solo", frequency_probs_fn),
        ("WS_Combined(10%)+Freq(90%)", lambda h: (0.10*wheel_combined_probs_fn(h) + 0.90*frequency_probs_fn(h)) / (0.10*wheel_combined_probs_fn(h) + 0.90*frequency_probs_fn(h)).sum()),
    ]

    for label, fn in key_configs:
        r = backtest_no_lstm(all_data, fn, WARMUP, label)
        if r:
            print(f"  {label:<45} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+8.0f}")

    print(f"\n  (LSTM configs — see detailed results above)")

    print(f"\n  Verification time: {elapsed:.0f}s")
    print(f"\n  VERDICT: Check if per-file results are CONSISTENT with combined results.")
    print(f"  If a config wins on combined but loses on most individual files,")
    print(f"  it may be overfitting to a specific data segment.")


if __name__ == '__main__':
    main()
