#!/usr/bin/env python3
"""
SOLO MODEL TEST — Each model at 100% weight, no combinations.

Tests every model independently:
  1. Frequency (100%)
  2. Polarity (100%)
  3. Table (100%)
  4. Set (100%)
  5. Hot Number (100%)
  6. Sector (100%)
  7. Neighbor (100%)
  8. Repeater (100%)
  9. Gap (100%)

Each tested with:
  - Different top-N (10, 11, 12, 13, 14)
  - Different window sizes
  - With/without signal gates
  - On all data combined + per file
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
    USERDATA_DIR,
    WHEEL_STRATEGY_TREND_BOOST, WHEEL_STRATEGY_COLD_DAMPEN,
)

BET_PER_NUM = 2.0
PAYOUT = 35
BOOST = WHEEL_STRATEGY_TREND_BOOST
DAMPEN = WHEEL_STRATEGY_COLD_DAMPEN

TABLE_MAP = {}; POL_MAP = {}; SET_MAP = {}
for _n in range(TOTAL_NUMBERS):
    TABLE_MAP[_n] = '0' if _n in WHEEL_TABLE_0 else '19'
    POL_MAP[_n] = 'pos' if _n in WHEEL_POSITIVE else 'neg'
    if _n in WHEEL_SET_1: SET_MAP[_n] = 1
    elif _n in WHEEL_SET_2: SET_MAP[_n] = 2
    else: SET_MAP[_n] = 3

SECTOR_SIZE = 6
POS_TO_NUM = {v: k for k, v in NUMBER_TO_POSITION.items()}


def load_all_data():
    all_spins = []
    files = []
    for fname in sorted(os.listdir(USERDATA_DIR)):
        if fname.endswith('.txt'):
            fpath = os.path.join(USERDATA_DIR, fname)
            spins = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        n = int(line)
                        if 0 <= n <= 36:
                            spins.append(n)
            if spins:
                files.append((fname, spins))
                all_spins.extend(spins)
    return all_spins, files


# ═══════════════════════════════════════════════════════════════════════════
# SOLO MODELS — each returns np.array(37) probabilities
# ═══════════════════════════════════════════════════════════════════════════

def model_frequency(history, window=30):
    """Pure frequency counting — Laplace-smoothed full + recent window blend."""
    if len(history) < 5:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    counts = Counter(history)
    total = len(history)
    flat = np.array([(counts.get(i, 0) + 1) / (total + TOTAL_NUMBERS) for i in range(TOTAL_NUMBERS)])
    flat /= flat.sum()
    recent = history[-window:]
    rc = np.ones(TOTAL_NUMBERS)
    for n in recent:
        rc[n] += 1
    rc /= rc.sum()
    blended = 0.5 * flat + 0.5 * rc
    return blended / blended.sum()


def model_polarity(history, window=50):
    """Boost numbers in the hot polarity group (pos/neg)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    total = len(recent)
    pos = sum(1 for n in recent if POL_MAP[n] == 'pos')
    neg = total - pos
    hot = 'pos' if pos > neg else 'neg' if neg > pos else None
    if hot:
        imb = abs(pos - neg) / total / 0.5
        pb = 1.0 + (BOOST - 1.0) * imb
        pd = 1.0 - (1.0 - DAMPEN) * imb
        for num in range(TOTAL_NUMBERS):
            probs[num] *= pb if POL_MAP[num] == hot else pd
    return probs / probs.sum()


def model_table(history, window=50):
    """Boost numbers in the hot table half (0/19)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if TABLE_MAP[n] == '0')
    t19 = total - t0
    hot = '0' if t0 > t19 else '19' if t19 > t0 else None
    if hot:
        imb = abs(t0 - t19) / total / 0.5
        tb = 1.0 + (BOOST - 1.0) * imb
        td = 1.0 - (1.0 - DAMPEN) * imb
        for num in range(TOTAL_NUMBERS):
            probs[num] *= tb if TABLE_MAP[num] == hot else td
    return probs / probs.sum()


def model_set(history, window=50):
    """Boost numbers in the hot set (1/2/3)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    total = len(recent)
    sc = {1: 0, 2: 0, 3: 0}
    for n in recent:
        sc[SET_MAP[n]] += 1
    hot_set = max(sc, key=sc.get)
    if sc[hot_set] <= total / 3 * 1.05:
        return probs
    for num in range(TOTAL_NUMBERS):
        sp = sc[SET_MAP[num]] / total
        expected = 1 / 3
        if sp > expected:
            exc = min(1, (sp - expected) / expected)
            probs[num] *= 1.0 + (BOOST - 1.0) * exc
        else:
            exc = min(1, (expected - sp) / expected)
            probs[num] *= 1.0 - (1.0 - DAMPEN) * exc
    return probs / probs.sum()


def model_hot_number(history, window=15):
    """Boost numbers that appeared frequently in recent short window."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window:
        return probs
    recent = history[-window:]
    rc = Counter(recent)
    expected = window / TOTAL_NUMBERS
    for num in range(TOTAL_NUMBERS):
        if rc.get(num, 0) > expected * 1.5:
            probs[num] *= 2.0
        elif rc.get(num, 0) > expected:
            probs[num] *= 1.3
    return probs / probs.sum()


def model_sector(history, window=30):
    """Boost numbers in hot wheel sectors."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10:
        return probs
    recent = history[-window:]
    sector_counts = defaultdict(int)
    num_sectors = (TOTAL_NUMBERS + SECTOR_SIZE - 1) // SECTOR_SIZE
    for n in recent:
        pos = NUMBER_TO_POSITION.get(n, 0)
        sector = pos // SECTOR_SIZE
        sector_counts[sector] += 1
    avg = len(recent) / num_sectors
    for num in range(TOTAL_NUMBERS):
        pos = NUMBER_TO_POSITION.get(num, 0)
        sector = pos // SECTOR_SIZE
        count = sector_counts.get(sector, 0)
        if count > avg * 1.3:
            probs[num] *= 1.5
        elif count < avg * 0.7:
            probs[num] *= 0.7
    return probs / probs.sum()


def model_neighbor(history, window=20, neighbor_range=2):
    """Boost wheel neighbors of recent numbers."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    for n in recent:
        pos = NUMBER_TO_POSITION.get(n, 0)
        for offset in range(-neighbor_range, neighbor_range + 1):
            neighbor_pos = (pos + offset) % len(WHEEL_ORDER)
            neighbor_num = WHEEL_ORDER[neighbor_pos]
            probs[neighbor_num] *= 1.1
    return probs / probs.sum()


def model_repeater(history, window=10):
    """Boost numbers seen in recent very short window (repeater bias)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window:
        return probs
    recent = history[-window:]
    seen = Counter(recent)
    for num, count in seen.items():
        probs[num] *= 1.0 + count * 0.8
    return probs / probs.sum()


def model_gap(history):
    """Boost numbers with large gaps since last appearance (due numbers)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 50:
        return probs
    last_seen = {}
    for i, n in enumerate(history):
        last_seen[n] = i
    max_idx = len(history)
    for num in range(TOTAL_NUMBERS):
        gap = max_idx - last_seen.get(num, 0)
        expected_gap = TOTAL_NUMBERS
        if gap > expected_gap * 2:
            probs[num] *= 1.5
        elif gap > expected_gap * 1.5:
            probs[num] *= 1.2
    return probs / probs.sum()


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def backtest(data, model_fn, top_n=12, warmup=80, model_kwargs=None):
    if len(data) <= warmup + 10:
        return None

    kwargs = model_kwargs or {}
    hits = 0; total_bets = 0
    bankroll = 4000.0; peak = 4000.0; max_dd = 0
    hit_log = []

    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]
        probs = model_fn(history, **kwargs) if kwargs else model_fn(history)
        top = set(np.argsort(probs)[::-1][:top_n])

        hit = 1 if actual in top else 0
        hits += hit; total_bets += 1; hit_log.append(hit)

        cost = top_n * BET_PER_NUM
        if hit:
            bankroll += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
        else:
            bankroll -= cost
        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

    if total_bets == 0:
        return None

    hit_rate = hits / total_bets * 100
    baseline = top_n / 37 * 100
    edge = hit_rate - baseline
    profit = hits * (PAYOUT + 1) * BET_PER_NUM - total_bets * top_n * BET_PER_NUM

    # Rolling window
    rolling = []
    ws = min(50, total_bets // 3) if total_bets > 50 else total_bets
    if ws > 0:
        for j in range(ws, len(hit_log)):
            rolling.append(sum(hit_log[j-ws:j]) / ws * 100)

    return {
        'bets': total_bets, 'hits': hits,
        'hit_rate': round(hit_rate, 2), 'baseline': round(baseline, 2),
        'edge': round(edge, 2), 'profit': round(profit, 2),
        'max_dd': round(max_dd, 2), 'bankroll': round(bankroll, 2),
        'best_roll': round(max(rolling), 1) if rolling else 0,
        'worst_roll': round(min(rolling), 1) if rolling else 0,
    }


def main():
    print("=" * 120)
    print("SOLO MODEL TEST — Each model at 100%, NO combinations")
    print("=" * 120)

    all_data, files = load_all_data()
    print(f"\nTotal: {len(all_data)} spins from {len(files)} files:")
    for fname, spins in files:
        print(f"  {fname}: {len(spins)} spins")

    start = time.time()

    MODELS = [
        ("Frequency",   model_frequency, {}),
        ("Polarity",    model_polarity,  {}),
        ("Table",       model_table,     {}),
        ("Set",         model_set,       {}),
        ("Hot Number",  model_hot_number, {}),
        ("Sector",      model_sector,    {}),
        ("Neighbor",    model_neighbor,  {}),
        ("Repeater",    model_repeater,  {}),
        ("Gap (Due)",   model_gap,       {}),
    ]

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: ALL DATA COMBINED — Each model solo, top-12
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("TEST 1: ALL DATA COMBINED — Each model SOLO at 100%, Top-12")
    print(f"{'='*120}")
    print(f"  {'Model':<20} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'MaxDD':>7} {'Best50':>7} {'Worst50':>8}")
    print("  " + "-" * 80)

    solo_results = []
    for name, fn, kw in MODELS:
        r = backtest(all_data, fn, 12, 80, kw)
        if r:
            solo_results.append((name, r))
            marker = " ✓" if r['edge'] > 0 else " ✗"
            print(f"  {name:<20} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} ${r['max_dd']:>6.0f} {r['best_roll']:>6.0f}% {r['worst_roll']:>7.0f}%{marker}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: EACH MODEL SOLO WITH DIFFERENT TOP-N
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("TEST 2: EACH MODEL SOLO — Different Top-N (10, 11, 12, 13, 14)")
    print(f"{'='*120}")

    for name, fn, kw in MODELS:
        print(f"\n  --- {name} ---")
        print(f"  {'N':>4} {'Bets':>5} {'Hit%':>6} {'BrkEvn':>7} {'Edge':>7} {'Profit':>8} {'MaxDD':>7}")
        print("  " + "-" * 55)
        for top_n in [10, 11, 12, 13, 14]:
            r = backtest(all_data, fn, top_n, 80, kw)
            if r:
                be = top_n / (PAYOUT + 1) * 100
                marker = " ✓" if r['hit_rate'] > be else ""
                print(f"  {top_n:>4} {r['bets']:>5} {r['hit_rate']:>6.2f} {be:>7.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} ${r['max_dd']:>6.0f}{marker}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: EACH MODEL SOLO WITH DIFFERENT WINDOW SIZES
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("TEST 3: WINDOW SIZE SWEEP — Each model solo, Top-12")
    print(f"{'='*120}")

    window_models = [
        ("Frequency",   model_frequency, 'window', [10, 15, 20, 25, 30, 40, 50, 75, 100]),
        ("Polarity",    model_polarity,  'window', [20, 30, 40, 50, 75, 100]),
        ("Table",       model_table,     'window', [20, 30, 40, 50, 75, 100]),
        ("Set",         model_set,       'window', [20, 30, 40, 50, 75, 100]),
        ("Hot Number",  model_hot_number, 'window', [5, 8, 10, 12, 15, 20, 25, 30]),
        ("Sector",      model_sector,    'window', [10, 15, 20, 25, 30, 40, 50]),
        ("Neighbor",    model_neighbor,  'window', [5, 10, 15, 20, 25, 30, 40]),
        ("Repeater",    model_repeater,  'window', [3, 5, 7, 10, 15, 20]),
    ]

    for name, fn, param, windows in window_models:
        print(f"\n  --- {name} (varying {param}) ---")
        print(f"  {'Window':>7} {'Hit%':>6} {'Edge':>7} {'Profit':>8}")
        print("  " + "-" * 35)
        best_w = None; best_edge = -999
        for w in windows:
            r = backtest(all_data, fn, 12, 80, {param: w})
            if r:
                if r['edge'] > best_edge:
                    best_edge = r['edge']; best_w = w
                marker = " ◀ BEST" if r['edge'] == best_edge else ""
                print(f"  {w:>7} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f}{marker}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: PER-FILE — Each model solo
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("TEST 4: PER-FILE CONSISTENCY — Each model solo, Top-12")
    print(f"{'='*120}")

    print(f"\n  {'Model':<15}", end='')
    for fname, _ in files:
        print(f" {fname:>10}", end='')
    print(f" {'COMBINED':>10}  {'Positive':>8}")
    print("  " + "-" * (15 + 11 * (len(files) + 1) + 10))

    for name, fn, kw in MODELS:
        print(f"  {name:<15}", end='')
        pos_count = 0
        for fname, fdata in files:
            r = backtest(fdata, fn, 12, min(60, len(fdata) // 5), kw)
            if r:
                marker = "✓" if r['edge'] > 0 else "✗"
                if r['edge'] > 0:
                    pos_count += 1
                print(f" {r['hit_rate']:>5.1f}%{marker:>3}", end='')
            else:
                print(f" {'N/A':>9}", end='')
        r_all = backtest(all_data, fn, 12, 80, kw)
        if r_all:
            marker = "✓" if r_all['edge'] > 0 else "✗"
            print(f" {r_all['hit_rate']:>5.1f}%{marker:>3}", end='')
        print(f"  {pos_count}/{len(files)}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 5: BEST SOLO WITH N=13 PER FILE
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("TEST 5: PER-FILE CONSISTENCY — Each model solo, Top-13")
    print(f"{'='*120}")

    print(f"\n  {'Model':<15}", end='')
    for fname, _ in files:
        print(f" {fname:>10}", end='')
    print(f" {'COMBINED':>10}  {'Positive':>8}")
    print("  " + "-" * (15 + 11 * (len(files) + 1) + 10))

    for name, fn, kw in MODELS:
        print(f"  {name:<15}", end='')
        pos_count = 0
        for fname, fdata in files:
            r = backtest(fdata, fn, 13, min(60, len(fdata) // 5), kw)
            if r:
                be = 13 / 37 * 100  # 35.14%
                marker = "✓" if r['hit_rate'] > be else "✗"
                if r['hit_rate'] > be:
                    pos_count += 1
                print(f" {r['hit_rate']:>5.1f}%{marker:>3}", end='')
            else:
                print(f" {'N/A':>9}", end='')
        r_all = backtest(all_data, fn, 13, 80, kw)
        if r_all:
            be = 13 / 37 * 100
            marker = "✓" if r_all['hit_rate'] > be else "✗"
            print(f" {r_all['hit_rate']:>5.1f}%{marker:>3}", end='')
        print(f"  {pos_count}/{len(files)}")

    # ═══════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start

    print(f"\n{'='*120}")
    print("GRAND SUMMARY — SOLO MODELS RANKED")
    print(f"{'='*120}")

    # Rank by combined hit rate
    solo_results.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n  RANKING (All data combined, Top-12):")
    print(f"  {'#':>3} {'Model':<20} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'MaxDD':>7}")
    print("  " + "-" * 55)
    for i, (name, r) in enumerate(solo_results):
        target = "  ← TARGET ZONE" if r['hit_rate'] >= 35 else ""
        print(f"  {i+1:>3} {name:<20} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} ${r['max_dd']:>6.0f}{target}")

    print(f"\n  Baseline (12/37): 32.43%")
    print(f"  Break-even (12 nums): 33.33%")
    print(f"  TARGET: 35-36%")
    print(f"\n  Time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
