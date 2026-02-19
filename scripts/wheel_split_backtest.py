#!/usr/bin/env python3
"""
Split Wheel Strategy Backtest — Tests Table(0/19), Polarity(Pos/Neg), Set(1/2/3)
as 3 INDEPENDENT models instead of one combined Wheel Strategy.

Tests:
  1. Each wheel sub-model solo
  2. Each sub-model + Frequency
  3. Each sub-model + LSTM
  4. All sub-model pairs (Table+Polarity, Table+Set, Polarity+Set)
  5. Sub-models + Frequency + LSTM combos
  6. Signal gate per sub-model
  7. Compare vs combined WS and vs previous best WS(20%)+LSTM(10%)+Freq(70%)

Uses stored LSTM probabilities from production model (real training, walk-forward).
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
    WHEEL_STRATEGY_TREND_BOOST, WHEEL_STRATEGY_COLD_DAMPEN,
    LSTM_SEQUENCE_LENGTH, RETRAIN_INTERVAL,
)

from app.ml.lstm_predictor import LSTMPredictor
from app.ml.frequency_analyzer import FrequencyAnalyzer
from app.ml.hot_number import HotNumberAnalyzer
from app.ml.markov_chain import MarkovChain
from app.ml.pattern_detector import PatternDetector

TOP_N = 12
WARMUP = 100
BET_PER_NUM = 2.0
PAYOUT = 35
BOOST = WHEEL_STRATEGY_TREND_BOOST     # 1.30
DAMPEN = WHEEL_STRATEGY_COLD_DAMPEN    # 0.70

# Pre-compute membership maps
TABLE_MAP = {}
POL_MAP = {}
SET_MAP = {}
for _n in range(TOTAL_NUMBERS):
    TABLE_MAP[_n] = '0' if _n in WHEEL_TABLE_0 else '19'
    POL_MAP[_n] = 'pos' if _n in WHEEL_POSITIVE else 'neg'
    if _n in WHEEL_SET_1: SET_MAP[_n] = 1
    elif _n in WHEEL_SET_2: SET_MAP[_n] = 2
    else: SET_MAP[_n] = 3


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


# ─── SPLIT Wheel Sub-Models ───────────────────────────────────────────

def table_probs(history, window=50):
    """Table 0 vs Table 19 — only this one grouping system."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs

    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if TABLE_MAP.get(n) == '0')
    t19 = total - t0
    t0_pct = t0 / total
    t19_pct = t19 / total
    hot = '0' if t0 > t19 else '19' if t19 > t0 else None

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


def polarity_probs(history, window=50):
    """Positive vs Negative — only this one grouping system."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs

    recent = history[-window:]
    total = len(recent)
    pos = sum(1 for n in recent if POL_MAP.get(n) == 'pos')
    neg = total - pos
    pos_pct = pos / total
    neg_pct = neg / total
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


def set_probs(history, window=50):
    """Set 1 vs Set 2 vs Set 3 — only this one grouping system."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs

    recent = history[-window:]
    total = len(recent)
    s_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        s_counts[SET_MAP.get(n, 3)] += 1

    hot = max(s_counts, key=s_counts.get)
    if s_counts[hot] <= total / 3 * 1.1:
        hot = None

    if hot is not None:
        set_pcts = {k: s_counts[k] / total for k in s_counts}
        excess = (set_pcts[hot] - 1/3) / (1/3)
        excess = max(0, min(1, excess))
        sb = 1.0 + (BOOST - 1.0) * excess
        sd = 1.0 - (1.0 - DAMPEN) * excess
        for num in range(TOTAL_NUMBERS):
            if SET_MAP[num] == hot:
                probs[num] *= sb
            else:
                probs[num] *= sd

    return probs / probs.sum()


def wheel_combined_probs(history, window=50):
    """All 3 combined (original WheelStrategy behavior)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs

    recent = history[-window:]
    total = len(recent)

    # Table
    t0 = sum(1 for n in recent if TABLE_MAP.get(n) == '0')
    t0_pct = t0 / total; t19_pct = 1 - t0_pct
    hot_table = '0' if t0 > total - t0 else '19' if total - t0 > t0 else None

    # Polarity
    pos = sum(1 for n in recent if POL_MAP.get(n) == 'pos')
    pos_pct = pos / total; neg_pct = 1 - pos_pct
    hot_pol = 'pos' if pos > total - pos else 'neg' if total - pos > pos else None

    # Set
    s_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        s_counts[SET_MAP.get(n, 3)] += 1
    hot_set = max(s_counts, key=s_counts.get)
    if s_counts[hot_set] <= total / 3 * 1.1:
        hot_set = None

    for num in range(TOTAL_NUMBERS):
        m = 1.0
        if hot_table is not None:
            imb = abs(t0_pct - t19_pct) / 0.5
            tb = 1.0 + (BOOST - 1.0) * imb
            td = 1.0 - (1.0 - DAMPEN) * imb
            m *= tb if TABLE_MAP[num] == hot_table else td

        if hot_pol is not None:
            imb = abs(pos_pct - neg_pct) / 0.5
            pb = 1.0 + (BOOST - 1.0) * imb
            pd = 1.0 - (1.0 - DAMPEN) * imb
            m *= pb if POL_MAP[num] == hot_pol else pd

        if hot_set is not None:
            sp = {k: s_counts[k]/total for k in s_counts}
            exc = max(0, min(1, (sp[hot_set] - 1/3) / (1/3)))
            sb = 1.0 + (BOOST - 1.0) * exc
            sd = 1.0 - (1.0 - DAMPEN) * exc
            m *= sb if SET_MAP[num] == hot_set else sd

        probs[num] *= m

    return probs / probs.sum()


def frequency_probs_fn(history):
    """Frequency: 50% flat Laplace + 50% last-30 window."""
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


# ─── Signal strength per sub-model ─────────────────────────────────────

def table_signal(history, window=50):
    if len(history) < 5:
        return 0.0
    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if n in WHEEL_TABLE_0)
    t0_dev = abs(t0/total - 19/37)
    return min(100, t0_dev / 0.25 * 100)


def polarity_signal(history, window=50):
    if len(history) < 5:
        return 0.0
    recent = history[-window:]
    total = len(recent)
    pos = sum(1 for n in recent if n in WHEEL_POSITIVE)
    pos_dev = abs(pos/total - 19/37)
    return min(100, pos_dev / 0.25 * 100)


def set_signal(history, window=50):
    if len(history) < 5:
        return 0.0
    recent = history[-window:]
    total = len(recent)
    s_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        if n in WHEEL_SET_1: s_counts[1] += 1
        elif n in WHEEL_SET_2: s_counts[2] += 1
        else: s_counts[3] += 1
    set_pcts = [s_counts[k]/total for k in [1,2,3]]
    max_dev = max(abs(p - 1/3) for p in set_pcts)
    return min(100, max_dev / 0.20 * 100)


def combined_signal(history, window=50):
    return (table_signal(history, window) + polarity_signal(history, window) + set_signal(history, window)) / 3


# ─── Walk-forward with LSTM ────────────────────────────────────────────

def walk_forward(data):
    """Walk-forward storing per-spin probabilities from all models including LSTM."""
    n = len(data)

    lstm = LSTMPredictor()
    freq = FrequencyAnalyzer()

    # Feed warmup
    warmup_data = data[:WARMUP]
    freq.load_history(warmup_data)
    lstm.load_history(warmup_data)

    print(f"  [LSTM] Training on {WARMUP} warmup spins...", end='', flush=True)
    tr = lstm.train(epochs=50)
    print(f" done ({tr.get('status')}, epochs={tr.get('epochs',0)})")

    spins_since_train = 0
    retrain_count = 0

    results = {
        'actuals': [],
        'histories': [],  # Store history reference for signal computation
        'lstm_probs': [],
        'freq_probs': [],
        'table_probs': [],
        'polarity_probs': [],
        'set_probs': [],
        'combined_ws_probs': [],
        'table_signals': [],
        'polarity_signals': [],
        'set_signals': [],
        'combined_signals': [],
    }

    for i in range(WARMUP, n):
        actual = data[i]
        history = data[:i]

        # Get probabilities
        lp = lstm.predict()
        fp = freq.get_number_probabilities()
        tp = table_probs(history)
        pp = polarity_probs(history)
        sp = set_probs(history)
        cp = wheel_combined_probs(history)

        results['actuals'].append(actual)
        results['lstm_probs'].append(lp.copy())
        results['freq_probs'].append(fp.copy())
        results['table_probs'].append(tp.copy())
        results['polarity_probs'].append(pp.copy())
        results['set_probs'].append(sp.copy())
        results['combined_ws_probs'].append(cp.copy())
        results['table_signals'].append(table_signal(history))
        results['polarity_signals'].append(polarity_signal(history))
        results['set_signals'].append(set_signal(history))
        results['combined_signals'].append(combined_signal(history))

        # Update
        freq.update(actual)
        lstm.update(actual)
        spins_since_train += 1

        if spins_since_train >= RETRAIN_INTERVAL and lstm.can_train():
            retrain_count += 1
            lstm.train(epochs=50)
            spins_since_train = 0

        if (i - WARMUP) % 500 == 0 and i > WARMUP:
            pct = (i - WARMUP) / (n - WARMUP) * 100
            print(f"  Progress: {pct:.0f}% retrains: {retrain_count}")

    print(f"  Complete. {retrain_count} LSTM retrains.")
    return results


def compute_hits(results, model_weights, signal_type=None, signal_gate=0):
    """
    model_weights: dict of probs_key -> weight
      Keys: 'lstm', 'freq', 'table', 'polarity', 'set', 'combined_ws'
    signal_type: 'table', 'polarity', 'set', 'combined', or None
    signal_gate: threshold (0=always bet)
    """
    key_map = {
        'lstm': 'lstm_probs',
        'freq': 'freq_probs',
        'table': 'table_probs',
        'polarity': 'polarity_probs',
        'set': 'set_probs',
        'combined_ws': 'combined_ws_probs',
    }
    signal_key_map = {
        'table': 'table_signals',
        'polarity': 'polarity_signals',
        'set': 'set_signals',
        'combined': 'combined_signals',
    }

    hits = 0; total_bets = 0; skipped = 0
    bankroll = 4000.0; peak = 4000.0; max_dd = 0
    hit_log = []

    n_spins = len(results['actuals'])
    for idx in range(n_spins):
        # Signal gate
        if signal_gate > 0 and signal_type:
            sig = results[signal_key_map[signal_type]][idx]
            if sig < signal_gate:
                skipped += 1
                continue

        ensemble = np.zeros(TOTAL_NUMBERS)
        for model, weight in model_weights.items():
            if weight > 0:
                ensemble += weight * results[key_map[model]][idx]

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

    max_win = 0; max_loss = 0; cur = 0
    for h in hit_log:
        if h:
            cur = cur + 1 if cur > 0 else 1
            max_win = max(max_win, cur)
        else:
            cur = cur - 1 if cur < 0 else -1
            max_loss = max(max_loss, abs(cur))

    return {
        'bets': total_bets, 'skipped': skipped,
        'bet_pct': round(total_bets / n_spins * 100, 1),
        'hits': hits, 'hit_rate': round(hit_rate, 2),
        'baseline': round(baseline, 2), 'edge': round(edge, 2),
        'profit': round(profit, 2), 'final': round(bankroll, 2),
        'max_dd': round(max_dd, 2),
        'max_win': max_win, 'max_loss': max_loss,
    }


def pr(label, r):
    if not r:
        print(f"  {label:<50} — no data")
        return
    print(f"  {label:<50} {r['hit_rate']:>6.2f}% {r['edge']:>+6.2f}% "
          f"${r['profit']:>+8.0f}  DD=${r['max_dd']:>6.0f}  W{r['max_win']} L{r['max_loss']}")


def pr_gate(gate, r):
    if not r or r['bets'] < 30:
        return
    ps = r['profit'] / r['bets'] if r['bets'] > 0 else 0
    print(f"  {gate:>5} {r['bets']:>5} {r['bet_pct']:>5.1f}% {r['hit_rate']:>6.2f}% "
          f"{r['edge']:>+6.2f}% ${r['profit']:>+8.0f}  ${ps:>+5.2f}/spin")


def main():
    print("=" * 105)
    print("SPLIT WHEEL STRATEGY BACKTEST")
    print("Table(0/19), Polarity(Pos/Neg), Set(1/2/3) tested as SEPARATE models")
    print(f"Top-{TOP_N} | ${BET_PER_NUM}/num | {PAYOUT}:1 | Boost={BOOST} Dampen={DAMPEN}")
    print("=" * 105)

    data = load_all_userdata()
    n_preds = len(data) - WARMUP
    print(f"\nData: {len(data)} spins | Warmup: {WARMUP} | Predictions: {n_preds}")
    print(f"Baseline: {TOP_N/37*100:.2f}% | Breakeven: {TOP_N/36*100:.2f}%")

    # Describe the groups
    print(f"\n  Table 0  ({len(WHEEL_TABLE_0)} nums): {sorted(WHEEL_TABLE_0)}")
    print(f"  Table 19 ({len(WHEEL_TABLE_19)} nums): {sorted(WHEEL_TABLE_19)}")
    print(f"  Positive ({len(WHEEL_POSITIVE)} nums): {sorted(WHEEL_POSITIVE)}")
    print(f"  Negative ({len(WHEEL_NEGATIVE)} nums): {sorted(WHEEL_NEGATIVE)}")
    print(f"  Set 1    ({len(WHEEL_SET_1)} nums): {sorted(WHEEL_SET_1)}")
    print(f"  Set 2    ({len(WHEEL_SET_2)} nums): {sorted(WHEEL_SET_2)}")
    print(f"  Set 3    ({len(WHEEL_SET_3)} nums): {sorted(WHEEL_SET_3)}")

    start = time.time()

    print(f"\nRunning walk-forward with LSTM training...")
    results = walk_forward(data)
    print(f"Walk-forward: {time.time()-start:.0f}s\n")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: EACH SUB-MODEL SOLO
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 105)
    print("SECTION 1: EACH MODEL SOLO")
    print("=" * 105)
    solos = [
        ("Table (0/19)", {'table': 1.0}),
        ("Polarity (Pos/Neg)", {'polarity': 1.0}),
        ("Set (1/2/3)", {'set': 1.0}),
        ("WheelStrategy Combined (all 3)", {'combined_ws': 1.0}),
        ("Frequency", {'freq': 1.0}),
        ("LSTM (trained)", {'lstm': 1.0}),
    ]
    for label, w in solos:
        pr(label, compute_hits(results, w))

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: EACH SUB-MODEL + FREQUENCY (weight sweep)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 105)
    print("SECTION 2: EACH WHEEL SUB-MODEL + FREQUENCY (weight sweep)")
    print("=" * 105)

    for sub_name, sub_key in [("Table", "table"), ("Polarity", "polarity"), ("Set", "set")]:
        print(f"\n  --- {sub_name} + Frequency ---")
        best_edge = -999; best_r = None; best_w = 0
        for sw in range(10, 91, 10):
            fw = 100 - sw
            w = {sub_key: sw/100, 'freq': fw/100}
            r = compute_hits(results, w)
            if r and r['edge'] > best_edge:
                best_edge = r['edge']; best_r = r; best_w = sw
            pr(f"  {sub_name}({sw}%)+Freq({fw}%)", r)
        if best_r:
            print(f"  → BEST: {sub_name}({best_w}%)+Freq({100-best_w}%) edge={best_edge:+.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: EACH SUB-MODEL + LSTM (weight sweep)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 105)
    print("SECTION 3: EACH WHEEL SUB-MODEL + LSTM (weight sweep)")
    print("=" * 105)

    for sub_name, sub_key in [("Table", "table"), ("Polarity", "polarity"), ("Set", "set")]:
        print(f"\n  --- {sub_name} + LSTM ---")
        best_edge = -999; best_r = None; best_w = 0
        for sw in range(10, 91, 10):
            lw = 100 - sw
            w = {sub_key: sw/100, 'lstm': lw/100}
            r = compute_hits(results, w)
            if r and r['edge'] > best_edge:
                best_edge = r['edge']; best_r = r; best_w = sw
            pr(f"  {sub_name}({sw}%)+LSTM({lw}%)", r)
        if best_r:
            print(f"  → BEST: {sub_name}({best_w}%)+LSTM({100-best_w}%) edge={best_edge:+.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: SUB-MODEL PAIRS (no Freq, no LSTM)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 105)
    print("SECTION 4: WHEEL SUB-MODEL PAIRS")
    print("=" * 105)

    pairs = [
        ("Table+Polarity", "table", "polarity"),
        ("Table+Set", "table", "set"),
        ("Polarity+Set", "polarity", "set"),
    ]
    for pair_name, k1, k2 in pairs:
        print(f"\n  --- {pair_name} ---")
        best_edge = -999; best_label = ""
        for w1 in range(10, 91, 10):
            w2 = 100 - w1
            w = {k1: w1/100, k2: w2/100}
            r = compute_hits(results, w)
            if r and r['edge'] > best_edge:
                best_edge = r['edge']; best_r = r
                best_label = f"{k1}({w1}%)+{k2}({w2}%)"
            pr(f"  {k1}({w1}%)+{k2}({w2}%)", r)
        print(f"  → BEST: {best_label} edge={best_edge:+.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: BEST TRIPLE COMBOS (sub-model + Freq + LSTM)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 105)
    print("SECTION 5: SUB-MODEL + FREQUENCY + LSTM (triple combos)")
    print("=" * 105)

    for sub_name, sub_key in [("Table", "table"), ("Polarity", "polarity"), ("Set", "set")]:
        print(f"\n  --- {sub_name} + Freq + LSTM ---")
        best_edge = -999; best_label = ""; best_r = None
        for sw in range(10, 60, 10):
            for lw in range(10, 70 - sw, 10):
                fw = 100 - sw - lw
                if fw < 10:
                    continue
                w = {sub_key: sw/100, 'lstm': lw/100, 'freq': fw/100}
                r = compute_hits(results, w)
                if r and r['edge'] > best_edge:
                    best_edge = r['edge']; best_r = r
                    best_label = f"{sub_name}({sw}%)+LSTM({lw}%)+Freq({fw}%)"
        if best_r:
            pr(f"BEST: {best_label}", best_r)

    # Also test all 3 sub-models + Freq + LSTM
    print(f"\n  --- All 3 sub-models + Freq + LSTM ---")
    best_edge = -999; best_r = None; best_label = ""
    for tw in range(10, 40, 5):
        for pw in range(5, 35, 5):
            for sw in range(5, 30, 5):
                for lw in range(5, 25, 5):
                    fw = 100 - tw - pw - sw - lw
                    if fw < 10:
                        continue
                    w = {'table': tw/100, 'polarity': pw/100, 'set': sw/100,
                         'lstm': lw/100, 'freq': fw/100}
                    r = compute_hits(results, w)
                    if r and r['edge'] > best_edge:
                        best_edge = r['edge']; best_r = r
                        best_label = f"T({tw}%)+P({pw}%)+S({sw}%)+LSTM({lw}%)+F({fw}%)"
    if best_r:
        pr(f"BEST 5-model: {best_label}", best_r)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 6: SIGNAL GATE per sub-model
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 105)
    print("SECTION 6: SIGNAL GATE — each sub-model's own signal")
    print("Using best sub-model+Freq combo, gated by that sub-model's signal")
    print("=" * 105)

    gate_configs = [
        ("Table(10%)+Freq(90%)", {'table': 0.10, 'freq': 0.90}, 'table'),
        ("Polarity(10%)+Freq(90%)", {'polarity': 0.10, 'freq': 0.90}, 'polarity'),
        ("Set(10%)+Freq(90%)", {'set': 0.10, 'freq': 0.90}, 'set'),
        ("Combined WS(10%)+Freq(90%)", {'combined_ws': 0.10, 'freq': 0.90}, 'combined'),
    ]

    for label, weights, sig_type in gate_configs:
        print(f"\n  --- {label} gated by {sig_type} signal ---")
        print(f"  {'Gate':>5} {'Bets':>5} {'Bet%':>6} {'Hit%':>7} {'Edge':>7} {'Profit':>9}  {'$/Spin':>7}")
        print("  " + "-" * 65)
        for gate in [0, 10, 15, 20, 25, 30, 35, 40, 50]:
            r = compute_hits(results, weights, signal_type=sig_type, signal_gate=gate)
            pr_gate(gate, r)

    # Also test best triple combo with signal gate
    print(f"\n  --- Previous best: WS(20%)+LSTM(10%)+Freq(70%) with combined gate ---")
    print(f"  {'Gate':>5} {'Bets':>5} {'Bet%':>6} {'Hit%':>7} {'Edge':>7} {'Profit':>9}  {'$/Spin':>7}")
    print("  " + "-" * 65)
    for gate in [0, 15, 20, 25, 30, 35, 40]:
        r = compute_hits(results, {'combined_ws': 0.20, 'lstm': 0.10, 'freq': 0.70},
                        signal_type='combined', signal_gate=gate)
        pr_gate(gate, r)

    # ═══════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start

    print("\n" + "=" * 105)
    print("GRAND SUMMARY — COMPARISON")
    print("=" * 105)

    summary_configs = [
        ("Table solo", {'table': 1.0}, None, 0),
        ("Polarity solo", {'polarity': 1.0}, None, 0),
        ("Set solo", {'set': 1.0}, None, 0),
        ("WS Combined solo", {'combined_ws': 1.0}, None, 0),
        ("Frequency solo", {'freq': 1.0}, None, 0),
        ("LSTM solo", {'lstm': 1.0}, None, 0),
        ("Prev best: WS(20%)+LSTM(10%)+Freq(70%)", {'combined_ws': 0.20, 'lstm': 0.10, 'freq': 0.70}, None, 0),
        ("Freq(90%)+WS(10%)", {'freq': 0.90, 'combined_ws': 0.10}, None, 0),
        ("Freq(90%)+Table(10%)", {'freq': 0.90, 'table': 0.10}, None, 0),
        ("Freq(90%)+Polarity(10%)", {'freq': 0.90, 'polarity': 0.10}, None, 0),
        ("Freq(90%)+Set(10%)", {'freq': 0.90, 'set': 0.10}, None, 0),
        ("Freq(80%)+Table(10%)+LSTM(10%)", {'freq': 0.80, 'table': 0.10, 'lstm': 0.10}, None, 0),
        ("Freq(80%)+Polarity(10%)+LSTM(10%)", {'freq': 0.80, 'polarity': 0.10, 'lstm': 0.10}, None, 0),
        ("Freq(80%)+Set(10%)+LSTM(10%)", {'freq': 0.80, 'set': 0.10, 'lstm': 0.10}, None, 0),
        # With signal gate
        ("Freq(90%)+Table(10%) gate=30", {'freq': 0.90, 'table': 0.10}, 'table', 30),
        ("Freq(90%)+Polarity(10%) gate=30", {'freq': 0.90, 'polarity': 0.10}, 'polarity', 30),
        ("Freq(90%)+Set(10%) gate=30", {'freq': 0.90, 'set': 0.10}, 'set', 30),
        ("Freq(90%)+WS(10%) gate=30", {'freq': 0.90, 'combined_ws': 0.10}, 'combined', 30),
        ("Freq(80%)+Table(10%)+LSTM(10%) gate=30", {'freq': 0.80, 'table': 0.10, 'lstm': 0.10}, 'table', 30),
    ]

    ranked = []
    for label, w, sig, gate in summary_configs:
        r = compute_hits(results, w, signal_type=sig, signal_gate=gate)
        if r:
            ranked.append((label, r))

    ranked.sort(key=lambda x: x[1]['edge'], reverse=True)

    print(f"\n  {'#':>3} {'Config':<55} {'Bets':>5} {'Hit%':>7} {'Edge':>7} {'Profit':>9}")
    print("  " + "-" * 95)
    for i, (label, r) in enumerate(ranked):
        marker = " ◀" if i == 0 else ""
        print(f"  {i+1:>3} {label:<55} {r['bets']:>5} {r['hit_rate']:>7.2f} "
              f"{r['edge']:>+7.2f} ${r['profit']:>+8.0f}{marker}")

    print(f"\n  Backtest time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
