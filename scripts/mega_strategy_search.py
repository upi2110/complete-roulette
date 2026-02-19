#!/usr/bin/env python3
"""
MEGA STRATEGY SEARCH — Exhaustive search for 35-36% sustained hit rate.

Tests EVERYTHING:
  Phase 1: Weight sweeps (all model combos, 5% step)
  Phase 2: Top-N variations (8-16 numbers)
  Phase 3: Trigger strategies (polarity, table, set triggers + combinations)
  Phase 4: Window size variations (20, 30, 40, 50, 75, 100)
  Phase 5: Signal gate sweep (10-50 strength threshold)
  Phase 6: Hybrid combos (best weights + best trigger + best gate)
  Phase 7: Streak-based strategies (bet after N wins, skip after N losses)
  Phase 8: Hot/cold number overlay strategies
  Phase 9: Sector-based filtering
  Phase 10: Cross-validation on all data files

Runs on ALL data in userdata/ folder.
Target: 35-36% hit rate sustained across multiple files.
"""

import sys
import os
import time
import numpy as np
from collections import Counter, defaultdict
from itertools import product

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    TOTAL_NUMBERS, WHEEL_ORDER, NUMBER_TO_POSITION,
    WHEEL_TABLE_0, WHEEL_TABLE_19,
    WHEEL_POSITIVE, WHEEL_NEGATIVE,
    WHEEL_SET_1, WHEEL_SET_2, WHEEL_SET_3,
    USERDATA_DIR, FREQUENCY_RECENT_WINDOW,
    WHEEL_STRATEGY_TREND_BOOST, WHEEL_STRATEGY_COLD_DAMPEN,
)

TOP_N_DEFAULT = 12
WARMUP_DEFAULT = 80
BET_PER_NUM = 2.0
PAYOUT = 35
BOOST = WHEEL_STRATEGY_TREND_BOOST
DAMPEN = WHEEL_STRATEGY_COLD_DAMPEN
BASELINE = TOP_N_DEFAULT / 37 * 100  # 32.43%

# Pre-compute membership maps
TABLE_MAP = {}; POL_MAP = {}; SET_MAP = {}
for _n in range(TOTAL_NUMBERS):
    TABLE_MAP[_n] = '0' if _n in WHEEL_TABLE_0 else '19'
    POL_MAP[_n] = 'pos' if _n in WHEEL_POSITIVE else 'neg'
    if _n in WHEEL_SET_1: SET_MAP[_n] = 1
    elif _n in WHEEL_SET_2: SET_MAP[_n] = 2
    else: SET_MAP[_n] = 3

POS_NUMS = sorted(WHEEL_POSITIVE)
NEG_NUMS = sorted(WHEEL_NEGATIVE)
TABLE_0_NUMS = sorted(WHEEL_TABLE_0)
TABLE_19_NUMS = sorted(WHEEL_TABLE_19)
SET_1_NUMS = sorted(WHEEL_SET_1)
SET_2_NUMS = sorted(WHEEL_SET_2)
SET_3_NUMS = sorted(WHEEL_SET_3)

# Sector map (wheel position)
SECTOR_SIZE = 6  # ~6 numbers per sector
NUM_SECTORS = (TOTAL_NUMBERS + SECTOR_SIZE - 1) // SECTOR_SIZE
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
# PROBABILITY MODELS
# ═══════════════════════════════════════════════════════════════════════════

def freq_probs(history, window=None):
    if len(history) < 5:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    counts = Counter(history)
    total = len(history)
    flat = np.array([(counts.get(i, 0) + 1) / (total + TOTAL_NUMBERS) for i in range(TOTAL_NUMBERS)])
    flat /= flat.sum()
    w = window or FREQUENCY_RECENT_WINDOW
    recent = history[-w:]
    rc = np.ones(TOTAL_NUMBERS)
    for n in recent:
        rc[n] += 1
    rc /= rc.sum()
    blended = 0.5 * flat + 0.5 * rc
    return blended / blended.sum()


def polarity_probs(history, window=50):
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


def table_probs(history, window=50):
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


def set_probs(history, window=50):
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


def hot_number_probs(history, window=15):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window:
        return probs
    recent = history[-window:]
    rc = Counter(recent)
    expected = window / TOTAL_NUMBERS
    for num in range(TOTAL_NUMBERS):
        if rc.get(num, 0) > expected * 1.5:
            probs[num] *= 2.0
    return probs / probs.sum()


def sector_probs(history, window=30):
    """Boost numbers in wheel sectors that have been hot."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10:
        return probs
    recent = history[-window:]
    # Count hits per wheel sector (groups of adjacent wheel positions)
    sector_counts = defaultdict(int)
    for n in recent:
        pos = NUMBER_TO_POSITION.get(n, 0)
        sector = pos // SECTOR_SIZE
        sector_counts[sector] += 1
    # Find hot sectors
    avg = len(recent) / NUM_SECTORS
    for num in range(TOTAL_NUMBERS):
        pos = NUMBER_TO_POSITION.get(num, 0)
        sector = pos // SECTOR_SIZE
        count = sector_counts.get(sector, 0)
        if count > avg * 1.3:
            probs[num] *= 1.5
        elif count < avg * 0.7:
            probs[num] *= 0.7
    return probs / probs.sum()


def neighbor_probs(history, window=20, neighbor_range=2):
    """Boost neighbors of recently hit numbers on the wheel."""
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


def repeater_probs(history, window=10):
    """Boost numbers that appeared in recent short window (repeater bias)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window:
        return probs
    recent = history[-window:]
    seen = set(recent)
    for num in seen:
        probs[num] *= 1.8
    return probs / probs.sum()


def gap_probs(history):
    """Boost numbers with large gaps (due numbers)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 50:
        return probs
    last_seen = {}
    for i, n in enumerate(history):
        last_seen[n] = i
    max_gap = len(history)
    for num in range(TOTAL_NUMBERS):
        gap = max_gap - last_seen.get(num, 0)
        expected_gap = TOTAL_NUMBERS
        if gap > expected_gap * 2:
            probs[num] *= 1.5
        elif gap > expected_gap * 1.5:
            probs[num] *= 1.2
    return probs / probs.sum()


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

MODEL_FNS = {
    'freq': freq_probs,
    'polarity': polarity_probs,
    'table': table_probs,
    'set': set_probs,
    'hot': hot_number_probs,
    'sector': sector_probs,
    'neighbor': neighbor_probs,
    'repeater': repeater_probs,
    'gap': gap_probs,
}


def build_ensemble(history, weights, freq_window=None):
    ensemble = np.zeros(TOTAL_NUMBERS)
    for model, weight in weights.items():
        if weight <= 0:
            continue
        if model == 'freq' and freq_window:
            ensemble += weight * freq_probs(history, window=freq_window)
        else:
            fn = MODEL_FNS.get(model)
            if fn:
                ensemble += weight * fn(history)
    total = ensemble.sum()
    if total > 0:
        ensemble /= total
    return ensemble


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL STRENGTH
# ═══════════════════════════════════════════════════════════════════════════

def polarity_signal(history, window=50):
    if len(history) < 10:
        return 0
    recent = history[-window:]
    total = len(recent)
    pos = sum(1 for n in recent if POL_MAP[n] == 'pos')
    return abs(pos / total - 0.5) * 200  # 0-100 scale


def table_signal(history, window=50):
    if len(history) < 10:
        return 0
    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if TABLE_MAP[n] == '0')
    return abs(t0 / total - 0.5) * 200


def combined_signal(history, window=50):
    return max(polarity_signal(history, window), table_signal(history, window))


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def backtest(data, weights, top_n=12, warmup=80, freq_window=None,
             trigger=None, gate_threshold=0, gate_fn=None,
             streak_filter=None, filter_group=None):
    """
    Universal backtest engine.

    trigger: None, 'pol_follow', 'pol_oppose', 'pol_double_follow',
             'pol_double_oppose', 'table_follow', 'table_double_follow',
             'table_oppose', 'table_double_oppose',
             'set_follow', 'set_double_follow',
             'alternating' (bet only when polarity alternates)
    gate_threshold: minimum signal strength to bet
    gate_fn: function(history) -> signal strength 0-100
    streak_filter: ('after_win', N) or ('after_loss', N) or ('skip_after_loss', N)
    filter_group: 'polarity', 'table', 'set' — restricts picks to triggered group
    """
    if len(data) <= warmup + 10:
        return None

    hits = 0; total_bets = 0; skipped = 0
    bankroll = 4000.0; peak = 4000.0; max_dd = 0
    hit_log = []; consec_w = 0; consec_l = 0

    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]

        # ── Gate check ──
        if gate_threshold > 0 and gate_fn:
            sig = gate_fn(history)
            if sig < gate_threshold:
                skipped += 1
                continue

        # ── Trigger check ──
        should_bet = True
        restricted_nums = None

        if trigger == 'pol_follow':
            last_pol = POL_MAP[data[i-1]]
            restricted_nums = POS_NUMS if last_pol == 'pos' else NEG_NUMS

        elif trigger == 'pol_oppose':
            last_pol = POL_MAP[data[i-1]]
            restricted_nums = NEG_NUMS if last_pol == 'pos' else POS_NUMS

        elif trigger == 'pol_double_follow':
            if i < warmup + 2:
                skipped += 1; continue
            p1, p2 = POL_MAP[data[i-1]], POL_MAP[data[i-2]]
            if p1 != p2:
                skipped += 1; continue
            restricted_nums = POS_NUMS if p1 == 'pos' else NEG_NUMS

        elif trigger == 'pol_double_oppose':
            if i < warmup + 2:
                skipped += 1; continue
            p1, p2 = POL_MAP[data[i-1]], POL_MAP[data[i-2]]
            if p1 != p2:
                skipped += 1; continue
            restricted_nums = NEG_NUMS if p1 == 'pos' else POS_NUMS

        elif trigger == 'table_follow':
            last_t = TABLE_MAP[data[i-1]]
            restricted_nums = TABLE_0_NUMS if last_t == '0' else TABLE_19_NUMS

        elif trigger == 'table_double_follow':
            if i < warmup + 2:
                skipped += 1; continue
            t1, t2 = TABLE_MAP[data[i-1]], TABLE_MAP[data[i-2]]
            if t1 != t2:
                skipped += 1; continue
            restricted_nums = TABLE_0_NUMS if t1 == '0' else TABLE_19_NUMS

        elif trigger == 'table_oppose':
            last_t = TABLE_MAP[data[i-1]]
            restricted_nums = TABLE_19_NUMS if last_t == '0' else TABLE_0_NUMS

        elif trigger == 'table_double_oppose':
            if i < warmup + 2:
                skipped += 1; continue
            t1, t2 = TABLE_MAP[data[i-1]], TABLE_MAP[data[i-2]]
            if t1 != t2:
                skipped += 1; continue
            restricted_nums = TABLE_19_NUMS if t1 == '0' else TABLE_0_NUMS

        elif trigger == 'alternating':
            if i < warmup + 2:
                skipped += 1; continue
            p1, p2 = POL_MAP[data[i-1]], POL_MAP[data[i-2]]
            if p1 == p2:
                skipped += 1; continue
            # After alternation, bet on the one that just appeared
            restricted_nums = POS_NUMS if p1 == 'pos' else NEG_NUMS

        elif trigger == 'triple_follow':
            if i < warmup + 3:
                skipped += 1; continue
            p1 = POL_MAP[data[i-1]]
            p2 = POL_MAP[data[i-2]]
            p3 = POL_MAP[data[i-3]]
            if not (p1 == p2 == p3):
                skipped += 1; continue
            restricted_nums = POS_NUMS if p1 == 'pos' else NEG_NUMS

        elif trigger == 'triple_oppose':
            if i < warmup + 3:
                skipped += 1; continue
            p1 = POL_MAP[data[i-1]]
            p2 = POL_MAP[data[i-2]]
            p3 = POL_MAP[data[i-3]]
            if not (p1 == p2 == p3):
                skipped += 1; continue
            restricted_nums = NEG_NUMS if p1 == 'pos' else POS_NUMS

        # ── Streak filter ──
        if streak_filter:
            mode, n = streak_filter
            if mode == 'after_win' and consec_w < n:
                skipped += 1; continue
            elif mode == 'after_loss' and consec_l < n:
                skipped += 1; continue
            elif mode == 'skip_after_loss' and consec_l >= n:
                skipped += 1; continue

        # ── Build probabilities and pick numbers ──
        probs = build_ensemble(history, weights, freq_window)

        if restricted_nums is not None:
            # Pick top-N from restricted group only
            group_probs = [(n, probs[n]) for n in restricted_nums]
            group_probs.sort(key=lambda x: x[1], reverse=True)
            top = set(n for n, p in group_probs[:top_n])
        else:
            top = set(np.argsort(probs)[::-1][:top_n])

        # ── Score ──
        hit = 1 if actual in top else 0
        hits += hit; total_bets += 1; hit_log.append(hit)

        cost = top_n * BET_PER_NUM
        if hit:
            bankroll += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
            consec_w += 1; consec_l = 0
        else:
            bankroll -= cost
            consec_l += 1; consec_w = 0
        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

    if total_bets == 0:
        return None

    hit_rate = hits / total_bets * 100
    edge = hit_rate - (top_n / 37 * 100)
    profit = hits * (PAYOUT + 1) * BET_PER_NUM - total_bets * top_n * BET_PER_NUM
    bet_pct = total_bets / (len(data) - warmup) * 100

    return {
        'bets': total_bets, 'hits': hits, 'hit_rate': round(hit_rate, 2),
        'edge': round(edge, 2), 'profit': round(profit, 2),
        'max_dd': round(max_dd, 2), 'bet_pct': round(bet_pct, 1),
        'bankroll': round(bankroll, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def generate_weight_combos():
    """Generate weight combinations in 10% steps for up to 4 models."""
    combos = []
    models = ['freq', 'polarity', 'table', 'set', 'hot', 'sector', 'neighbor', 'repeater', 'gap']

    # 2-model combos (freq + each other) in 5% steps
    for m in models[1:]:
        for w in range(5, 100, 5):
            combos.append({
                'freq': (100 - w) / 100,
                m: w / 100,
            })

    # 3-model combos (freq + 2 others) in 10% steps
    for i, m1 in enumerate(models[1:]):
        for m2 in models[i+2:]:
            for w1 in range(10, 50, 10):
                for w2 in range(10, 50, 10):
                    wf = 100 - w1 - w2
                    if wf >= 20:
                        combos.append({
                            'freq': wf / 100,
                            m1: w1 / 100,
                            m2: w2 / 100,
                        })

    # Key combos from previous analysis
    combos.extend([
        {'freq': 0.80, 'polarity': 0.20},
        {'freq': 0.85, 'polarity': 0.15},
        {'freq': 0.75, 'polarity': 0.25},
        {'freq': 0.70, 'polarity': 0.20, 'table': 0.10},
        {'freq': 0.70, 'polarity': 0.20, 'hot': 0.10},
        {'freq': 0.60, 'polarity': 0.20, 'hot': 0.10, 'sector': 0.10},
        {'freq': 0.60, 'polarity': 0.20, 'neighbor': 0.10, 'repeater': 0.10},
        {'freq': 0.50, 'polarity': 0.20, 'hot': 0.10, 'repeater': 0.10, 'gap': 0.10},
        {'freq': 0.70, 'polarity': 0.10, 'repeater': 0.20},
        {'freq': 0.60, 'polarity': 0.10, 'hot': 0.15, 'repeater': 0.15},
        {'freq': 0.70, 'hot': 0.15, 'repeater': 0.15},
        {'freq': 0.80, 'repeater': 0.20},
        {'freq': 0.80, 'hot': 0.20},
        {'freq': 0.70, 'neighbor': 0.30},
        {'freq': 0.80, 'gap': 0.20},
        {'freq': 0.60, 'gap': 0.20, 'hot': 0.20},
        {'freq': 0.90, 'polarity': 0.10},
        {'freq': 0.90, 'table': 0.10},
    ])

    return combos


def weights_label(w):
    parts = []
    for k, v in sorted(w.items(), key=lambda x: -x[1]):
        if v > 0:
            parts.append(f"{k.capitalize()[:3]}({int(v*100)}%)")
    return "+".join(parts)


def main():
    print("=" * 110)
    print("MEGA STRATEGY SEARCH — Finding 35-36% sustained hit rate")
    print("=" * 110)

    all_data, files = load_all_data()
    print(f"\nLoaded {len(all_data)} total spins from {len(files)} files:")
    for fname, spins in files:
        print(f"  {fname}: {len(spins)} spins")

    start_time = time.time()
    results = []  # (label, combined_result, per_file_results, config)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: WEIGHT SWEEP
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PHASE 1: WEIGHT SWEEP — All model combinations")
    print(f"{'='*110}")

    weight_combos = generate_weight_combos()
    print(f"Testing {len(weight_combos)} weight combinations...")

    phase1_results = []
    for wi, weights in enumerate(weight_combos):
        r = backtest(all_data, weights, TOP_N_DEFAULT, WARMUP_DEFAULT)
        if r and r['edge'] > 0:
            label = weights_label(weights)
            phase1_results.append((label, r, weights))

        if (wi + 1) % 100 == 0:
            print(f"  ... {wi+1}/{len(weight_combos)} tested", flush=True)

    phase1_results.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n  Top 20 weight combos (combined data):")
    print(f"  {'#':>3} {'Config':<55} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'Bets':>5}")
    print("  " + "-" * 90)
    for i, (label, r, w) in enumerate(phase1_results[:20]):
        print(f"  {i+1:>3} {label:<55} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} {r['bets']:>5}")

    # Save top-50 for later phases
    top50_weights = [(label, w) for label, r, w in phase1_results[:50]]

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: TOP-N VARIATIONS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PHASE 2: TOP-N VARIATIONS (8-16 numbers)")
    print(f"{'='*110}")

    # Test top 5 weight combos with different N values
    phase2_results = []
    for top_n in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
        for label, weights in top50_weights[:5]:
            r = backtest(all_data, weights, top_n, WARMUP_DEFAULT)
            if r:
                baseline_n = top_n / 37 * 100
                # Profit calculation: need to recalculate for different N
                profit_per_hit = (PAYOUT + 1) * BET_PER_NUM - top_n * BET_PER_NUM
                profit_per_miss = -top_n * BET_PER_NUM
                actual_profit = r['hits'] * (PAYOUT + 1) * BET_PER_NUM - r['bets'] * top_n * BET_PER_NUM
                # Break-even hit rate for this N
                break_even = (top_n * BET_PER_NUM) / ((PAYOUT + 1) * BET_PER_NUM) * 100
                phase2_results.append((f"N={top_n} {label}", r, top_n, break_even, actual_profit))

    phase2_results.sort(key=lambda x: -x[4])  # Sort by actual profit
    print(f"\n  Top results by profit:")
    print(f"  {'Config':<60} {'N':>3} {'Hit%':>6} {'BrkEvn':>7} {'Profit':>8}")
    print("  " + "-" * 90)
    for label, r, n, be, profit in phase2_results[:15]:
        marker = " ✓" if r['hit_rate'] > be else ""
        print(f"  {label:<60} {n:>3} {r['hit_rate']:>6.2f} {be:>7.2f} ${profit:>+7.0f}{marker}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: TRIGGER STRATEGIES
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PHASE 3: TRIGGER STRATEGIES")
    print(f"{'='*110}")

    triggers = [
        None, 'pol_follow', 'pol_oppose', 'pol_double_follow', 'pol_double_oppose',
        'table_follow', 'table_double_follow', 'table_oppose', 'table_double_oppose',
        'alternating', 'triple_follow', 'triple_oppose',
    ]
    trigger_names = {
        None: 'No trigger (bet every spin)',
        'pol_follow': 'Polarity follow (last→same)',
        'pol_oppose': 'Polarity oppose (last→opposite)',
        'pol_double_follow': 'Polarity 2x follow (2 same→same)',
        'pol_double_oppose': 'Polarity 2x oppose (2 same→opposite)',
        'table_follow': 'Table follow (last→same)',
        'table_double_follow': 'Table 2x follow (2 same→same)',
        'table_oppose': 'Table oppose (last→opposite)',
        'table_double_oppose': 'Table 2x oppose (2 same→opposite)',
        'alternating': 'Alternating (pol switches→follow new)',
        'triple_follow': 'Triple follow (3 same pol→same)',
        'triple_oppose': 'Triple oppose (3 same pol→opposite)',
    }

    phase3_results = []
    # Test top 10 weight combos × all triggers
    for label, weights in top50_weights[:10]:
        for trigger in triggers:
            r = backtest(all_data, weights, TOP_N_DEFAULT, WARMUP_DEFAULT, trigger=trigger)
            if r and r['bets'] > 50:
                tname = trigger_names.get(trigger, trigger or 'none')
                phase3_results.append((f"{label} | {tname}", r, weights, trigger))

    phase3_results.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n  Top 25 trigger combos:")
    print(f"  {'Config':<75} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'Bet%':>5}")
    print("  " + "-" * 110)
    for i, (label, r, w, t) in enumerate(phase3_results[:25]):
        print(f"  {label:<75} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} {r['bet_pct']:>4.0f}%")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: WINDOW SIZE VARIATIONS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PHASE 4: FREQUENCY WINDOW SIZE VARIATIONS")
    print(f"{'='*110}")

    phase4_results = []
    for fw in [10, 15, 20, 25, 30, 40, 50, 75, 100, 150]:
        for label, weights in top50_weights[:5]:
            r = backtest(all_data, weights, TOP_N_DEFAULT, WARMUP_DEFAULT, freq_window=fw)
            if r:
                phase4_results.append((f"FW={fw} {label}", r, fw))

    phase4_results.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n  Top 15 window variations:")
    print(f"  {'Config':<65} {'Hit%':>6} {'Edge':>7} {'Profit':>8}")
    print("  " + "-" * 90)
    for label, r, fw in phase4_results[:15]:
        print(f"  {label:<65} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: SIGNAL GATE SWEEP
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PHASE 5: SIGNAL GATE SWEEP")
    print(f"{'='*110}")

    gate_fns = [
        ('Polarity gate', polarity_signal),
        ('Table gate', table_signal),
        ('Combined gate', combined_signal),
    ]

    phase5_results = []
    for gate_name, gate_fn in gate_fns:
        for threshold in range(5, 55, 5):
            for label, weights in top50_weights[:5]:
                r = backtest(all_data, weights, TOP_N_DEFAULT, WARMUP_DEFAULT,
                           gate_threshold=threshold, gate_fn=gate_fn)
                if r and r['bets'] > 50:
                    phase5_results.append((
                        f"{gate_name}≥{threshold} {label}", r, gate_fn, threshold))

    phase5_results.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n  Top 20 gate combos:")
    print(f"  {'Config':<70} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'Bet%':>5}")
    print("  " + "-" * 105)
    for i, (label, r, gf, gt) in enumerate(phase5_results[:20]):
        print(f"  {label:<70} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} {r['bet_pct']:>4.0f}%")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 6: STREAK FILTERS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PHASE 6: STREAK-BASED FILTERS")
    print(f"{'='*110}")

    streak_filters = [
        ('After 1 win', ('after_win', 1)),
        ('After 2 wins', ('after_win', 2)),
        ('After 3 wins', ('after_win', 3)),
        ('After 1 loss', ('after_loss', 1)),
        ('After 2 losses', ('after_loss', 2)),
        ('After 3 losses', ('after_loss', 3)),
        ('Skip after 2 losses', ('skip_after_loss', 2)),
        ('Skip after 3 losses', ('skip_after_loss', 3)),
        ('Skip after 4 losses', ('skip_after_loss', 4)),
    ]

    phase6_results = []
    for sf_name, sf in streak_filters:
        for label, weights in top50_weights[:5]:
            r = backtest(all_data, weights, TOP_N_DEFAULT, WARMUP_DEFAULT, streak_filter=sf)
            if r and r['bets'] > 50:
                phase6_results.append((f"{sf_name} | {label}", r))

    phase6_results.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n  Top 15 streak filters:")
    print(f"  {'Config':<70} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'Bet%':>5}")
    print("  " + "-" * 105)
    for label, r in phase6_results[:15]:
        print(f"  {label:<70} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} {r['bet_pct']:>4.0f}%")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 7: BEST COMBOS — HYBRID (gate + trigger + weights)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PHASE 7: HYBRID STRATEGIES (combining best gate + trigger + weights)")
    print(f"{'='*110}")

    # Collect best elements from each phase
    best_weights_list = [(w, l) for l, w in top50_weights[:10]]
    best_triggers = [None, 'pol_double_follow', 'pol_double_oppose', 'table_double_follow',
                     'triple_follow', 'triple_oppose', 'alternating']
    best_gates = [(None, 0), (polarity_signal, 10), (polarity_signal, 20),
                  (table_signal, 10), (table_signal, 20), (combined_signal, 15)]
    best_streaks = [None, ('after_win', 1), ('skip_after_loss', 3)]
    best_top_ns = [11, 12, 13]

    phase7_results = []
    total_combos = 0
    for weights, wlabel in best_weights_list[:5]:
        for trigger in best_triggers:
            for gate_fn, gate_thresh in best_gates:
                for sf in best_streaks:
                    for top_n in best_top_ns:
                        total_combos += 1
                        r = backtest(all_data, weights, top_n, WARMUP_DEFAULT,
                                   trigger=trigger, gate_threshold=gate_thresh,
                                   gate_fn=gate_fn, streak_filter=sf)
                        if r and r['bets'] > 30 and r['hit_rate'] > 33:
                            tname = trigger or 'none'
                            gname = f"gate≥{gate_thresh}" if gate_thresh > 0 else 'no-gate'
                            sname = f"{sf[0]}_{sf[1]}" if sf else 'no-streak'
                            label = f"N={top_n} {wlabel} | {tname} | {gname} | {sname}"
                            phase7_results.append((label, r))

    phase7_results.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n  Tested {total_combos} hybrid combinations")
    print(f"  Top 30 hybrids (min 30 bets, min 33% hit):")
    print(f"  {'#':>3} {'Config':<85} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'Bet%':>5}")
    print("  " + "-" * 120)
    for i, (label, r) in enumerate(phase7_results[:30]):
        print(f"  {i+1:>3} {label:<85} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} {r['bet_pct']:>4.0f}%")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 8: CROSS-VALIDATION — Best strategies on each file
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("PHASE 8: CROSS-VALIDATION — Top 10 strategies tested per file")
    print(f"{'='*110}")

    # Collect top strategies to validate
    top_strategies = []

    # From phase 1 (weight combos)
    for label, r, w in phase1_results[:5]:
        top_strategies.append((f"P1: {label}", w, None, 0, None, None, TOP_N_DEFAULT))

    # From phase 7 (hybrids) — need to decode, just use top results with standard params
    # We'll re-test the top weight combos with default params on each file
    for label, r, w in phase1_results[:3]:
        top_strategies.append((f"P1+Gate20: {label}", w, None, 20, polarity_signal, None, TOP_N_DEFAULT))

    # Always include our verified winner
    top_strategies.append(("VERIFIED: Pol(20%)+Freq(80%)",
                          {'freq': 0.80, 'polarity': 0.20}, None, 0, None, None, TOP_N_DEFAULT))

    print(f"\n  {'Strategy':<55}", end='')
    for fname, _ in files:
        print(f" {fname:>10}", end='')
    print(f" {'COMBINED':>10}")
    print("  " + "-" * (55 + 11 * (len(files) + 1)))

    for label, weights, trigger, gate_t, gate_fn, sf, top_n in top_strategies:
        print(f"  {label:<55}", end='')
        for fname, fdata in files:
            r = backtest(fdata, weights, top_n, min(60, len(fdata)//5),
                        trigger=trigger, gate_threshold=gate_t, gate_fn=gate_fn,
                        streak_filter=sf)
            if r:
                marker = "✓" if r['edge'] > 0 else "✗"
                print(f" {r['hit_rate']:>5.1f}%{marker:>3}", end='')
            else:
                print(f" {'N/A':>9}", end='')

        r_all = backtest(all_data, weights, top_n, WARMUP_DEFAULT,
                        trigger=trigger, gate_threshold=gate_t, gate_fn=gate_fn,
                        streak_filter=sf)
        if r_all:
            print(f" {r_all['hit_rate']:>5.1f}%{'✓' if r_all['edge'] > 0 else '✗':>3}")
        else:
            print(f" {'N/A':>9}")

    # ═══════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    print(f"\n{'='*110}")
    print("GRAND SUMMARY — BEST STRATEGIES FOUND")
    print(f"{'='*110}")

    # Collect ALL results and rank
    all_results = []
    for label, r, w in phase1_results:
        all_results.append((label, r, 'weight'))
    for label, r, w, t in phase3_results:
        all_results.append((label, r, 'trigger'))
    for label, r, gf, gt in phase5_results:
        all_results.append((label, r, 'gate'))
    for label, r in phase6_results:
        all_results.append((label, r, 'streak'))
    for label, r in phase7_results:
        all_results.append((label, r, 'hybrid'))

    # Rank by hit rate (min 100 bets for reliability)
    reliable = [(l, r, t) for l, r, t in all_results if r['bets'] >= 100]
    reliable.sort(key=lambda x: -x[1]['hit_rate'])

    print(f"\n  TOP 20 STRATEGIES (min 100 bets for reliability):")
    print(f"  {'#':>3} {'Type':<8} {'Config':<70} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8}")
    print("  " + "-" * 110)
    for i, (label, r, typ) in enumerate(reliable[:20]):
        print(f"  {i+1:>3} {typ:<8} {label:<70} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f}")

    # High hit rate (any bet count)
    all_results.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n  HIGHEST HIT RATES (any bet count, min 30 bets):")
    shown = set()
    count = 0
    for label, r, typ in all_results:
        if r['bets'] >= 30 and label not in shown:
            shown.add(label)
            print(f"  {count+1:>3} {typ:<8} {label:<70} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f}")
            count += 1
            if count >= 15:
                break

    # Best profit
    profitable = [(l, r, t) for l, r, t in all_results if r['profit'] > 0]
    profitable.sort(key=lambda x: -x[1]['profit'])
    print(f"\n  MOST PROFITABLE (sorted by $):")
    shown = set()
    count = 0
    for label, r, typ in profitable:
        if label not in shown:
            shown.add(label)
            print(f"  {count+1:>3} {typ:<8} {label:<70} {r['bets']:>5} {r['hit_rate']:>6.2f} ${r['profit']:>+7.0f}")
            count += 1
            if count >= 10:
                break

    print(f"\n  Search time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Total spins analyzed: {len(all_data)}")
    print(f"  Files: {[f[0] for f in files]}")
    print(f"\n  TARGET: 35-36% sustained = +$100/session")
    print(f"  Baseline: {BASELINE:.2f}%")
    print(f"  Break-even (12 numbers): 33.33%")


if __name__ == '__main__':
    main()
