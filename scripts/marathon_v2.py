#!/usr/bin/env python3
"""
MARATHON V2 — 12-hour time-boxed autonomous strategy search.

KEY IMPROVEMENT: Strict per-phase time budgets so ALL phases get tested.
Phase 1: 2-model pairs (coarse+finetune)         — 60 min budget
Phase 2: 3-model combos (top 8 models only)       — 90 min budget
Phase 3: Top-N sweep (8-16) with top 50 weights   — 30 min budget
Phase 4: Trigger strategies × top weights          — 30 min budget
Phase 5: Signal gates × top weights                — 45 min budget
Phase 6: Gate + Trigger hybrids                    — 45 min budget
Phase 7: Streak filters × top weights              — 30 min budget
Phase 8: 4-model combos (top 6 models, 10% step)  — 60 min budget
Phase 9: Window sweep (different history windows)  — 30 min budget
Phase 10: Ensemble of top strategies               — 30 min budget
Phase 11: Random perturbations (remaining time)    — all remaining
Phase 12: New data re-check                        — 15 min
FINAL: Cross-validation + report
"""

import sys
import os
import time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

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
POS_NUMS = sorted(WHEEL_POSITIVE)
NEG_NUMS = sorted(WHEEL_NEGATIVE)
TABLE_0_NUMS = sorted(WHEEL_TABLE_0)
TABLE_19_NUMS = sorted(WHEEL_TABLE_19)
RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data():
    all_spins = []
    files = []
    for fname in sorted(os.listdir(USERDATA_DIR)):
        if fname.endswith('.txt') and not fname.startswith('.'):
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

def m_freq(history, window=30):
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
    return (0.5 * flat + 0.5 * rc) / (0.5 * flat + 0.5 * rc).sum()

def m_freq_decay(history, decay=0.97):
    if len(history) < 5:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    probs = np.ones(TOTAL_NUMBERS) * 0.01
    for i, n in enumerate(history):
        weight = decay ** (len(history) - 1 - i)
        probs[n] += weight
    return probs / probs.sum()

def m_polarity(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5: return probs
    recent = history[-window:]
    total = len(recent)
    pos = sum(1 for n in recent if POL_MAP[n] == 'pos')
    neg = total - pos
    hot = 'pos' if pos > neg else 'neg' if neg > pos else None
    if hot:
        imb = abs(pos - neg) / total / 0.5
        for num in range(TOTAL_NUMBERS):
            probs[num] *= (1.0 + (BOOST-1)*imb) if POL_MAP[num] == hot else (1.0 - (1-DAMPEN)*imb)
    return probs / probs.sum()

def m_polarity_contrarian(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5: return probs
    recent = history[-window:]
    total = len(recent)
    pos = sum(1 for n in recent if POL_MAP[n] == 'pos')
    neg = total - pos
    cold = 'neg' if pos > neg else 'pos' if neg > pos else None
    if cold:
        imb = abs(pos - neg) / total / 0.5
        for num in range(TOTAL_NUMBERS):
            probs[num] *= (1.0 + (BOOST-1)*imb) if POL_MAP[num] == cold else (1.0 - (1-DAMPEN)*imb)
    return probs / probs.sum()

def m_table(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5: return probs
    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if TABLE_MAP[n] == '0')
    t19 = total - t0
    hot = '0' if t0 > t19 else '19' if t19 > t0 else None
    if hot:
        imb = abs(t0 - t19) / total / 0.5
        for num in range(TOTAL_NUMBERS):
            probs[num] *= (1.0 + (BOOST-1)*imb) if TABLE_MAP[num] == hot else (1.0 - (1-DAMPEN)*imb)
    return probs / probs.sum()

def m_table_contrarian(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5: return probs
    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if TABLE_MAP[n] == '0')
    t19 = total - t0
    cold = '19' if t0 > t19 else '0' if t19 > t0 else None
    if cold:
        imb = abs(t0 - t19) / total / 0.5
        for num in range(TOTAL_NUMBERS):
            probs[num] *= (1.0 + (BOOST-1)*imb) if TABLE_MAP[num] == cold else (1.0 - (1-DAMPEN)*imb)
    return probs / probs.sum()

def m_set(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5: return probs
    recent = history[-window:]
    total = len(recent)
    sc = {1:0, 2:0, 3:0}
    for n in recent: sc[SET_MAP[n]] += 1
    hot_set = max(sc, key=sc.get)
    if sc[hot_set] <= total / 3 * 1.05: return probs
    for num in range(TOTAL_NUMBERS):
        sp = sc[SET_MAP[num]] / total
        exp = 1/3
        exc = min(1, abs(sp - exp) / exp)
        if sp > exp: probs[num] *= 1.0 + (BOOST-1) * exc
        else: probs[num] *= 1.0 - (1-DAMPEN) * exc
    return probs / probs.sum()

def m_hot(history, window=20):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window: return probs
    recent = history[-window:]
    rc = Counter(recent)
    exp = window / TOTAL_NUMBERS
    for num in range(TOTAL_NUMBERS):
        c = rc.get(num, 0)
        if c > exp * 1.5: probs[num] *= 2.0
        elif c > exp: probs[num] *= 1.3
    return probs / probs.sum()

def m_cold(history, window=20):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window: return probs
    recent = history[-window:]
    seen = set(recent)
    for num in range(TOTAL_NUMBERS):
        if num not in seen: probs[num] *= 1.5
    return probs / probs.sum()

def m_sector(history, window=30):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    sc = defaultdict(int)
    ns = (TOTAL_NUMBERS + SECTOR_SIZE - 1) // SECTOR_SIZE
    for n in recent: sc[NUMBER_TO_POSITION.get(n, 0) // SECTOR_SIZE] += 1
    avg = len(recent) / ns
    for num in range(TOTAL_NUMBERS):
        s = NUMBER_TO_POSITION.get(num, 0) // SECTOR_SIZE
        c = sc.get(s, 0)
        if c > avg * 1.3: probs[num] *= 1.5
        elif c < avg * 0.7: probs[num] *= 0.7
    return probs / probs.sum()

def m_neighbor(history, window=25, nr=2):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5: return probs
    recent = history[-window:]
    for n in recent:
        pos = NUMBER_TO_POSITION.get(n, 0)
        for off in range(-nr, nr + 1):
            npos = (pos + off) % len(WHEEL_ORDER)
            probs[WHEEL_ORDER[npos]] *= 1.1
    return probs / probs.sum()

def m_repeater(history, window=10):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window: return probs
    recent = history[-window:]
    seen = Counter(recent)
    for num, count in seen.items():
        probs[num] *= 1.0 + count * 0.8
    return probs / probs.sum()

def m_gap(history):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 50: return probs
    last_seen = {}
    for i, n in enumerate(history): last_seen[n] = i
    mx = len(history)
    for num in range(TOTAL_NUMBERS):
        gap = mx - last_seen.get(num, 0)
        if gap > TOTAL_NUMBERS * 2: probs[num] *= 1.5
        elif gap > TOTAL_NUMBERS * 1.5: probs[num] *= 1.2
    return probs / probs.sum()

def m_markov(history):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 50: return probs
    trans = np.ones((TOTAL_NUMBERS, TOTAL_NUMBERS)) * 0.5
    for i in range(1, len(history)):
        trans[history[i-1]][history[i]] += 1
    last = history[-1]
    row = trans[last]
    return row / row.sum()

def m_markov2(history):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 100: return probs
    trans = defaultdict(lambda: np.ones(TOTAL_NUMBERS) * 0.5)
    for i in range(2, len(history)):
        key = (history[i-2], history[i-1])
        trans[key][history[i]] += 1
    if len(history) >= 2:
        key = (history[-2], history[-1])
        row = trans[key]
        return row / row.sum()
    return probs

def m_dozen(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    d = {0:0, 1:0, 2:0, 3:0}
    for n in recent:
        if n == 0: d[0] += 1
        elif n <= 12: d[1] += 1
        elif n <= 24: d[2] += 1
        else: d[3] += 1
    total = len(recent)
    hot_d = max([1,2,3], key=lambda x: d[x])
    if d[hot_d] > total / 3 * 1.1:
        for num in range(TOTAL_NUMBERS):
            if num == 0: continue
            nd = 1 if num <= 12 else 2 if num <= 24 else 3
            if nd == hot_d: probs[num] *= 1.3
            else: probs[num] *= 0.85
    return probs / probs.sum()

def m_column(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    c = {1:0, 2:0, 0:0}
    for n in recent:
        if n == 0: continue
        c[n % 3] += 1
    total = sum(c.values())
    if total == 0: return probs
    hot_c = max(c, key=c.get)
    if c[hot_c] > total / 3 * 1.1:
        for num in range(1, TOTAL_NUMBERS):
            if num % 3 == hot_c: probs[num] *= 1.3
            else: probs[num] *= 0.85
    return probs / probs.sum()

def m_red_black(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    r = sum(1 for n in recent if n in RED)
    b = len(recent) - r
    hot = 'red' if r > b else 'black' if b > r else None
    if hot:
        imb = abs(r - b) / len(recent) / 0.5
        for num in range(TOTAL_NUMBERS):
            is_red = num in RED
            if (hot == 'red' and is_red) or (hot == 'black' and not is_red and num != 0):
                probs[num] *= 1.0 + (BOOST-1) * imb
            else:
                probs[num] *= 1.0 - (1-DAMPEN) * imb
    return probs / probs.sum()

def m_odd_even(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    odd = sum(1 for n in recent if n > 0 and n % 2 == 1)
    even = sum(1 for n in recent if n > 0 and n % 2 == 0)
    hot = 'odd' if odd > even else 'even' if even > odd else None
    if hot:
        imb = abs(odd - even) / len(recent) / 0.5
        for num in range(1, TOTAL_NUMBERS):
            is_odd = num % 2 == 1
            if (hot == 'odd' and is_odd) or (hot == 'even' and not is_odd):
                probs[num] *= 1.0 + (BOOST-1) * imb
            else:
                probs[num] *= 1.0 - (1-DAMPEN) * imb
    return probs / probs.sum()

def m_high_low(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    lo = sum(1 for n in recent if 1 <= n <= 18)
    hi = sum(1 for n in recent if 19 <= n <= 36)
    hot = 'low' if lo > hi else 'high' if hi > lo else None
    if hot:
        imb = abs(lo - hi) / len(recent) / 0.5
        for num in range(1, TOTAL_NUMBERS):
            is_lo = 1 <= num <= 18
            if (hot == 'low' and is_lo) or (hot == 'high' and not is_lo):
                probs[num] *= 1.0 + (BOOST-1) * imb
            else:
                probs[num] *= 1.0 - (1-DAMPEN) * imb
    return probs / probs.sum()

# NEW: Quadrant model — divide wheel into 4 physical quadrants
def m_quadrant(history, window=30):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    quad_size = len(WHEEL_ORDER) // 4
    qc = {0:0, 1:0, 2:0, 3:0}
    for n in recent:
        pos = NUMBER_TO_POSITION.get(n, 0)
        qc[min(pos // quad_size, 3)] += 1
    total = len(recent)
    hot_q = max(qc, key=qc.get)
    if qc[hot_q] > total / 4 * 1.15:
        for num in range(TOTAL_NUMBERS):
            pos = NUMBER_TO_POSITION.get(num, 0)
            q = min(pos // quad_size, 3)
            ratio = qc[q] / total
            exp = 0.25
            if ratio > exp:
                probs[num] *= 1.0 + (ratio - exp) * 3
            else:
                probs[num] *= 1.0 - (exp - ratio) * 2
    return probs / probs.sum()

# NEW: Third-order pattern (last 3 numbers inform next)
def m_pattern3(history):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 200: return probs
    # Use polarity patterns instead of exact numbers (more data)
    trans = defaultdict(lambda: np.ones(TOTAL_NUMBERS) * 0.5)
    for i in range(3, len(history)):
        key = (POL_MAP[history[i-3]], POL_MAP[history[i-2]], POL_MAP[history[i-1]])
        trans[key][history[i]] += 1
    if len(history) >= 3:
        key = (POL_MAP[history[-3]], POL_MAP[history[-2]], POL_MAP[history[-1]])
        row = trans[key]
        return row / row.sum()
    return probs

# NEW: Velocity model — track acceleration of number frequency
def m_velocity(history, window1=15, window2=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window2: return probs
    recent = Counter(history[-window1:])
    older = Counter(history[-window2:-window1])
    for num in range(TOTAL_NUMBERS):
        r_freq = recent.get(num, 0) / window1
        o_freq = older.get(num, 0) / (window2 - window1) if (window2 - window1) > 0 else 0
        vel = r_freq - o_freq
        if vel > 0: probs[num] *= 1.0 + vel * 10
        elif vel < 0: probs[num] *= max(0.5, 1.0 + vel * 5)
    return probs / probs.sum()


ALL_MODELS = {
    'freq': m_freq, 'freq_decay': m_freq_decay,
    'polarity': m_polarity, 'pol_contra': m_polarity_contrarian,
    'table': m_table, 'tab_contra': m_table_contrarian,
    'set': m_set, 'hot': m_hot, 'cold': m_cold,
    'sector': m_sector, 'neighbor': m_neighbor,
    'repeater': m_repeater, 'gap': m_gap,
    'markov': m_markov, 'markov2': m_markov2,
    'dozen': m_dozen, 'column': m_column,
    'red_black': m_red_black, 'odd_even': m_odd_even,
    'high_low': m_high_low,
    'quadrant': m_quadrant, 'pattern3': m_pattern3,
    'velocity': m_velocity,
}


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def sig_polarity(history, window=50):
    if len(history) < 10: return 0
    recent = history[-window:]
    pos = sum(1 for n in recent if POL_MAP[n] == 'pos')
    return abs(pos / len(recent) - 0.5) * 200

def sig_table(history, window=50):
    if len(history) < 10: return 0
    recent = history[-window:]
    t0 = sum(1 for n in recent if TABLE_MAP[n] == '0')
    return abs(t0 / len(recent) - 0.5) * 200

def sig_combined(history, window=50):
    return max(sig_polarity(history, window), sig_table(history, window))

def sig_set(history, window=50):
    if len(history) < 10: return 0
    recent = history[-window:]
    sc = {1:0, 2:0, 3:0}
    for n in recent: sc[SET_MAP[n]] += 1
    total = len(recent)
    max_dev = max(abs(sc[k]/total - 1/3) for k in sc)
    return max_dev * 300


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def backtest(data, weights, top_n=12, warmup=80, trigger=None,
             gate_threshold=0, gate_fn=None, streak_filter=None,
             window_override=None):
    if len(data) <= warmup + 10:
        return None

    hits = 0; total_bets = 0; skipped = 0
    bankroll = 4000.0; peak = 4000.0; max_dd = 0
    hit_log = []; cw = 0; cl = 0

    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]

        if gate_threshold > 0 and gate_fn:
            if gate_fn(history) < gate_threshold:
                skipped += 1; continue

        restricted = None
        if trigger == 'pol_double_follow':
            if i < warmup+2: skipped += 1; continue
            if POL_MAP[data[i-1]] != POL_MAP[data[i-2]]: skipped += 1; continue
            restricted = POS_NUMS if POL_MAP[data[i-1]] == 'pos' else NEG_NUMS
        elif trigger == 'pol_double_oppose':
            if i < warmup+2: skipped += 1; continue
            if POL_MAP[data[i-1]] != POL_MAP[data[i-2]]: skipped += 1; continue
            restricted = NEG_NUMS if POL_MAP[data[i-1]] == 'pos' else POS_NUMS
        elif trigger == 'table_double_follow':
            if i < warmup+2: skipped += 1; continue
            if TABLE_MAP[data[i-1]] != TABLE_MAP[data[i-2]]: skipped += 1; continue
            restricted = TABLE_0_NUMS if TABLE_MAP[data[i-1]] == '0' else TABLE_19_NUMS
        elif trigger == 'table_double_oppose':
            if i < warmup+2: skipped += 1; continue
            if TABLE_MAP[data[i-1]] != TABLE_MAP[data[i-2]]: skipped += 1; continue
            restricted = TABLE_19_NUMS if TABLE_MAP[data[i-1]] == '0' else TABLE_0_NUMS
        elif trigger == 'alternating':
            if i < warmup+2: skipped += 1; continue
            if POL_MAP[data[i-1]] == POL_MAP[data[i-2]]: skipped += 1; continue
            restricted = POS_NUMS if POL_MAP[data[i-1]] == 'pos' else NEG_NUMS
        elif trigger == 'triple_follow':
            if i < warmup+3: skipped += 1; continue
            p = POL_MAP[data[i-1]]
            if POL_MAP[data[i-2]] != p or POL_MAP[data[i-3]] != p: skipped += 1; continue
            restricted = POS_NUMS if p == 'pos' else NEG_NUMS
        elif trigger == 'triple_oppose':
            if i < warmup+3: skipped += 1; continue
            p = POL_MAP[data[i-1]]
            if POL_MAP[data[i-2]] != p or POL_MAP[data[i-3]] != p: skipped += 1; continue
            restricted = NEG_NUMS if p == 'pos' else POS_NUMS
        elif trigger == 'tab_alternating':
            if i < warmup+2: skipped += 1; continue
            if TABLE_MAP[data[i-1]] == TABLE_MAP[data[i-2]]: skipped += 1; continue
            restricted = TABLE_0_NUMS if TABLE_MAP[data[i-1]] == '0' else TABLE_19_NUMS

        if streak_filter:
            mode, n = streak_filter
            if mode == 'skip_loss' and cl >= n: skipped += 1; continue
            if mode == 'after_win' and cw < n: skipped += 1; continue
            if mode == 'skip_win' and cw >= n: skipped += 1; continue

        ensemble = np.zeros(TOTAL_NUMBERS)
        for model, weight in weights.items():
            if weight <= 0: continue
            fn = ALL_MODELS.get(model)
            if fn: ensemble += weight * fn(history)
        s = ensemble.sum()
        if s > 0: ensemble /= s

        if restricted:
            gp = [(n, ensemble[n]) for n in restricted]
            gp.sort(key=lambda x: x[1], reverse=True)
            top = set(n for n, p in gp[:top_n])
        else:
            top = set(np.argsort(ensemble)[::-1][:top_n])

        hit = 1 if actual in top else 0
        hits += hit; total_bets += 1; hit_log.append(hit)
        cost = top_n * BET_PER_NUM
        if hit:
            bankroll += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
            cw += 1; cl = 0
        else:
            bankroll -= cost
            cl += 1; cw = 0
        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

    if total_bets == 0: return None
    hr = hits / total_bets * 100
    bl = top_n / 37 * 100
    profit = hits * (PAYOUT+1) * BET_PER_NUM - total_bets * top_n * BET_PER_NUM
    return {
        'bets': total_bets, 'hits': hits, 'hit_rate': round(hr, 2),
        'edge': round(hr - bl, 2), 'profit': round(profit, 2),
        'max_dd': round(max_dd, 2), 'bankroll': round(bankroll, 2),
        'bet_pct': round(total_bets / (len(data) - warmup) * 100, 1),
    }


def wlabel(w):
    parts = []
    for k, v in sorted(w.items(), key=lambda x: -x[1]):
        if v > 0:
            parts.append(f"{k}({int(v*100)}%)")
    return "+".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════

LEADERBOARD = []
MAX_LEADERS = 300

def add_to_leaderboard(label, result, config=None):
    global LEADERBOARD
    if result and result['bets'] >= 30:
        LEADERBOARD.append((label, result, config))
        LEADERBOARD.sort(key=lambda x: -x[1]['hit_rate'])
        if len(LEADERBOARD) > MAX_LEADERS:
            LEADERBOARD = LEADERBOARD[:MAX_LEADERS]

def print_leaderboard(title="CURRENT LEADERBOARD", top=20, min_bets=50):
    filtered = [(l, r, c) for l, r, c in LEADERBOARD if r['bets'] >= min_bets]
    print(f"\n  {'='*110}")
    print(f"  {title} (min {min_bets} bets)")
    print(f"  {'='*110}")
    print(f"  {'#':>3} {'Config':<70} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'Bet%':>5}")
    print(f"  {'-'*105}")
    for i, (l, r, c) in enumerate(filtered[:top]):
        star = " *" if r['hit_rate'] >= 35 else ""
        print(f"  {i+1:>3} {l:<70} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} {r['bet_pct']:>4.0f}%{star}")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN MARATHON V2
# ═══════════════════════════════════════════════════════════════════════════

def main():
    START_TIME = time.time()
    MAX_HOURS = 12
    MAX_SECONDS = MAX_HOURS * 3600

    print("=" * 120)
    print(f"MARATHON V2 — {MAX_HOURS}-hour time-boxed strategy search")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"IMPROVEMENT: Strict per-phase time budgets — ALL phases will run")
    print("=" * 120)

    all_data, files = load_all_data()
    print(f"\nLoaded {len(all_data)} spins from {len(files)} files:")
    for f, s in files:
        print(f"  {f}: {len(s)} spins")
    sys.stdout.flush()

    phase_num = 0
    total_tested = 0

    def elapsed():
        return time.time() - START_TIME

    def remaining():
        return MAX_SECONDS - elapsed()

    def phase_header(name, budget_min):
        nonlocal phase_num
        phase_num += 1
        e = elapsed()
        print(f"\n{'='*120}")
        print(f"PHASE {phase_num}: {name}")
        print(f"Time: {e/60:.0f}min elapsed, {remaining()/60:.0f}min remaining | Budget: {budget_min}min | Tested: {total_tested}")
        print(f"{'='*120}")
        sys.stdout.flush()

    def phase_deadline(budget_min):
        """Returns the deadline timestamp for this phase."""
        return time.time() + budget_min * 60

    def quick_test(label, weights, top_n=12, trigger=None, gate_t=0, gate_fn=None, sf=None):
        nonlocal total_tested
        r = backtest(all_data, weights, top_n, 80, trigger, gate_t, gate_fn, sf)
        total_tested += 1
        if r and r['edge'] > -0.5:
            add_to_leaderboard(label, r, {'w': weights, 'n': top_n, 't': trigger, 'g': gate_t, 'gfn': gate_fn.__name__ if gate_fn else None})
        return r

    model_names = list(ALL_MODELS.keys())

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: 2-MODEL PAIRS (coarse + finetune) — 60 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("2-MODEL PAIRS (5% coarse + 1% finetune)", 60)
    deadline = phase_deadline(60)
    best_pairs = []

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            if time.time() > deadline: break
            best_edge = -99; best_w = None; best_r = None
            for w1 in range(5, 100, 5):
                w2 = 100 - w1
                weights = {m1: w1/100, m2: w2/100}
                r = quick_test(f"{m1}({w1}%)+{m2}({w2}%)", weights)
                if r and r['edge'] > best_edge:
                    best_edge = r['edge']; best_w = w1; best_r = r
            if best_w:
                for w1 in range(max(1, best_w-4), min(100, best_w+5)):
                    w2 = 100 - w1
                    weights = {m1: w1/100, m2: w2/100}
                    r = quick_test(f"{m1}({w1}%)+{m2}({w2}%)", weights)
                    if r and r['edge'] > best_edge:
                        best_edge = r['edge']; best_w = w1; best_r = r
            if best_r and best_edge > 0:
                best_pairs.append((m1, m2, best_w, best_r))
                print(f"  {m1}({best_w}%)+{m2}({100-best_w}%) = {best_r['hit_rate']}% edge={best_r['edge']:+.2f}%")
                sys.stdout.flush()
        if time.time() > deadline: break

    best_pairs.sort(key=lambda x: -x[3]['hit_rate'])
    print(f"\n  Top 10 pairs:")
    for m1, m2, w, r in best_pairs[:10]:
        print(f"    {m1}({w}%)+{m2}({100-w}%) = {r['hit_rate']}% ${r['profit']:+.0f}")
    print_leaderboard("AFTER PHASE 1", 15, 100)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: 3-MODEL COMBOS — top 10 models only — 90 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("3-MODEL COMBOS (top 10 models, 10% steps)", 90)
    deadline = phase_deadline(90)

    # Find top 10 models by how often they appear in top pairs
    model_score = defaultdict(float)
    for l, r, c in LEADERBOARD[:50]:
        if c and 'w' in c:
            for m, w in c['w'].items():
                model_score[m] += r['edge'] * w
    top_models = sorted(model_score.keys(), key=lambda x: -model_score[x])[:10]
    print(f"  Top 10 models for 3-model combos: {top_models}")
    sys.stdout.flush()

    count = 0
    for m1, m2, m3 in combinations(top_models, 3):
        if time.time() > deadline: break
        for w1 in range(10, 80, 10):
            for w2 in range(10, 90 - w1, 10):
                w3 = 100 - w1 - w2
                if w3 < 10: continue
                weights = {m1: w1/100, m2: w2/100, m3: w3/100}
                quick_test(f"{m1}({w1})+{m2}({w2})+{m3}({w3})", weights)
                count += 1
                if count % 2000 == 0:
                    print(f"  ... {count} tested, {elapsed()/60:.0f}min", flush=True)
        if time.time() > deadline:
            print(f"  Budget hit at {count} combos")
            break

    # Also try 5% steps around best 3-model combos
    top3_configs = [(l, c) for l, r, c in LEADERBOARD[:20] if c and len(c.get('w', {})) == 3]
    for label, cfg in top3_configs[:10]:
        if time.time() > deadline: break
        w = cfg['w']
        models = list(w.keys())
        if len(models) != 3: continue
        base_w = [int(w[m]*100) for m in models]
        for d1 in range(-10, 11, 5):
            for d2 in range(-10, 11, 5):
                w1 = base_w[0] + d1
                w2 = base_w[1] + d2
                w3 = 100 - w1 - w2
                if w1 < 5 or w2 < 5 or w3 < 5: continue
                weights = {models[0]: w1/100, models[1]: w2/100, models[2]: w3/100}
                quick_test(f"{models[0]}({w1})+{models[1]}({w2})+{models[2]}({w3})", weights)

    print_leaderboard("AFTER PHASE 2", 15, 100)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: TOP-N SWEEP — 30 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("TOP-N SWEEP (8-16) with top 40 weight combos", 30)
    deadline = phase_deadline(30)

    top_configs = [(l, c) for l, r, c in LEADERBOARD[:40] if c]
    for top_n in [8, 9, 10, 11, 13, 14, 15, 16]:
        if time.time() > deadline: break
        for label, cfg in top_configs:
            if time.time() > deadline: break
            quick_test(f"N={top_n} {wlabel(cfg['w'])}", cfg['w'], top_n)

    print_leaderboard("AFTER PHASE 3", 20, 50)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: TRIGGER STRATEGIES — 30 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("TRIGGER STRATEGIES with top weights", 30)
    deadline = phase_deadline(30)

    triggers = ['pol_double_follow', 'pol_double_oppose',
                'table_double_follow', 'table_double_oppose',
                'alternating', 'tab_alternating',
                'triple_follow', 'triple_oppose']

    top_configs = [(l, c) for l, r, c in LEADERBOARD[:30] if c]
    for label, cfg in top_configs[:25]:
        if time.time() > deadline: break
        for trig in triggers:
            for top_n in [12, 13, 14]:
                quick_test(f"N={top_n} {wlabel(cfg['w'])}|{trig}", cfg['w'], top_n, trigger=trig)

    print_leaderboard("AFTER PHASE 4", 20, 30)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: SIGNAL GATES — 45 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("SIGNAL GATE SWEEP", 45)
    deadline = phase_deadline(45)

    gates = [(sig_polarity, 'pol'), (sig_table, 'tab'), (sig_combined, 'comb'), (sig_set, 'set')]

    top_configs = [(l, c) for l, r, c in LEADERBOARD[:25] if c]
    for label, cfg in top_configs:
        if time.time() > deadline: break
        for gfn, gname in gates:
            if time.time() > deadline: break
            for gt in [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]:
                for top_n in [12, 13, 14]:
                    quick_test(f"N={top_n} {wlabel(cfg['w'])}|{gname}>={gt}",
                             cfg['w'], top_n, gate_t=gt, gate_fn=gfn)

    print_leaderboard("AFTER PHASE 5", 20, 30)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 6: GATE + TRIGGER HYBRIDS — 45 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("GATE + TRIGGER HYBRIDS", 45)
    deadline = phase_deadline(45)

    top_configs = [(l, c) for l, r, c in LEADERBOARD[:15] if c]
    for label, cfg in top_configs:
        if time.time() > deadline: break
        for trig in triggers:
            if time.time() > deadline: break
            for gfn, gname in gates:
                if time.time() > deadline: break
                for gt in [10, 15, 20, 25, 30]:
                    for top_n in [12, 13, 14]:
                        quick_test(f"N={top_n} {wlabel(cfg['w'])}|{trig}|{gname}>={gt}",
                                 cfg['w'], top_n, trigger=trig,
                                 gate_t=gt, gate_fn=gfn)

    print_leaderboard("AFTER PHASE 6", 20, 20)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 7: STREAK FILTERS — 30 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("STREAK-BASED FILTERS", 30)
    deadline = phase_deadline(30)

    streaks = [('after_win', 1), ('after_win', 2), ('after_win', 3),
               ('skip_loss', 2), ('skip_loss', 3), ('skip_loss', 4), ('skip_loss', 5),
               ('skip_win', 3), ('skip_win', 4)]

    top_configs = [(l, c) for l, r, c in LEADERBOARD[:15] if c]
    for label, cfg in top_configs:
        if time.time() > deadline: break
        for sf in streaks:
            sfn = f"{sf[0]}_{sf[1]}"
            for top_n in [12, 13]:
                quick_test(f"N={top_n} {wlabel(cfg['w'])}|{sfn}", cfg['w'], top_n, sf=sf)
                # Also with top 2 gates
                for gfn, gname in gates[:2]:
                    for gt in [10, 15, 20]:
                        quick_test(f"N={top_n} {wlabel(cfg['w'])}|{sfn}|{gname}>={gt}",
                                 cfg['w'], top_n, gate_t=gt,
                                 gate_fn=gfn, sf=sf)

    print_leaderboard("AFTER PHASE 7", 20, 20)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 8: 4-MODEL COMBOS — top 8 models, 60 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("4-MODEL COMBOS (top 8 models, 10% steps)", 60)
    deadline = phase_deadline(60)

    top8 = top_models[:8] if len(top_models) >= 8 else top_models
    print(f"  Using top 8 models: {top8}")
    sys.stdout.flush()

    count = 0
    for m1, m2, m3, m4 in combinations(top8, 4):
        if time.time() > deadline: break
        for w1 in range(10, 60, 10):
            if time.time() > deadline: break
            for w2 in range(10, 70 - w1, 10):
                for w3 in range(10, 80 - w1 - w2, 10):
                    w4 = 100 - w1 - w2 - w3
                    if w4 < 10: continue
                    weights = {m1: w1/100, m2: w2/100, m3: w3/100, m4: w4/100}
                    quick_test(f"{m1}({w1})+{m2}({w2})+{m3}({w3})+{m4}({w4})", weights)
                    count += 1
                    if count % 5000 == 0:
                        print(f"  ... {count} tested, {elapsed()/60:.0f}min", flush=True)

    # Also try N=13 with top 4-model combos
    top4_configs = [(l, c) for l, r, c in LEADERBOARD[:20] if c and len(c.get('w', {})) == 4]
    for label, cfg in top4_configs[:10]:
        if time.time() > deadline: break
        quick_test(f"N=13 {wlabel(cfg['w'])}", cfg['w'], 13)
        quick_test(f"N=14 {wlabel(cfg['w'])}", cfg['w'], 14)

    print_leaderboard("AFTER PHASE 8", 20, 50)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 9: WINDOW SWEEP — 30 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("WINDOW SWEEP — different warmup/history sizes", 30)
    deadline = phase_deadline(30)

    top_configs = [(l, c) for l, r, c in LEADERBOARD[:15] if c]
    for warmup in [40, 60, 100, 120, 150]:
        if time.time() > deadline: break
        for label, cfg in top_configs:
            if time.time() > deadline: break
            r = backtest(all_data, cfg['w'], cfg.get('n', 12), warmup,
                        cfg.get('t'), cfg.get('g', 0),
                        sig_combined if cfg.get('g', 0) > 0 else None)
            total_tested += 1
            if r:
                lbl = f"warmup={warmup} {wlabel(cfg['w'])}"
                add_to_leaderboard(lbl, r, {**cfg, 'warmup': warmup})
                if r['hit_rate'] > 34:
                    print(f"  {lbl} = {r['hit_rate']}%", flush=True)

    print_leaderboard("AFTER PHASE 9", 15, 100)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 10: MEGA COMBO — best of everything — 30 min budget
    # ═══════════════════════════════════════════════════════════════════
    phase_header("MEGA COMBOS — combine top configs with gates + triggers + N + streaks", 30)
    deadline = phase_deadline(30)

    # Take the top 10 weight configs and try every combination of gate/trigger/N/streak
    top_weights = []
    seen_w = set()
    for l, r, c in LEADERBOARD:
        if c and 'w' in c:
            key = tuple(sorted(c['w'].items()))
            if key not in seen_w:
                seen_w.add(key)
                top_weights.append(c['w'])
                if len(top_weights) >= 10: break

    for w in top_weights:
        if time.time() > deadline: break
        wl = wlabel(w)
        for top_n in [12, 13, 14]:
            if time.time() > deadline: break
            for trig in [None] + triggers:
                if time.time() > deadline: break
                tname = trig or 'none'
                for gfn, gname in [(None, 'none')] + gates:
                    if time.time() > deadline: break
                    for gt in ([0] if gfn is None else [10, 15, 20, 25]):
                        if time.time() > deadline: break
                        for sf in [None, ('skip_loss', 3), ('after_win', 1)]:
                            sfn = f"|{sf[0]}_{sf[1]}" if sf else ""
                            gn = f"|{gname}>={gt}" if gt > 0 else ""
                            tn = f"|{tname}" if trig else ""
                            lbl = f"N={top_n} {wl}{tn}{gn}{sfn}"
                            quick_test(lbl, w, top_n, trig, gt, gfn, sf)

    print_leaderboard("AFTER PHASE 10 — MEGA COMBOS", 25, 20)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 11: RANDOM PERTURBATIONS (remaining time minus 30 min for final)
    # ═══════════════════════════════════════════════════════════════════
    phase_header("RANDOM PERTURBATIONS of top configs", max(30, int(remaining()/60) - 30))
    deadline = phase_deadline(max(1, remaining() - 30*60))

    perturbation_count = 0
    while time.time() < deadline and remaining() > 30*60:
        if LEADERBOARD:
            idx = np.random.randint(0, min(30, len(LEADERBOARD)))
            _, _, cfg = LEADERBOARD[idx]
            if cfg and 'w' in cfg:
                new_w = {}
                for k, v in cfg['w'].items():
                    nv = v + np.random.uniform(-0.15, 0.15)
                    if nv > 0.01: new_w[k] = nv
                # Sometimes add a random model
                if np.random.random() < 0.3:
                    rm = np.random.choice(model_names)
                    new_w[rm] = np.random.uniform(0.05, 0.4)
                # Sometimes drop a model
                if np.random.random() < 0.2 and len(new_w) > 2:
                    drop = min(new_w, key=new_w.get)
                    del new_w[drop]
                total = sum(new_w.values())
                new_w = {k: v/total for k, v in new_w.items()}
                top_n = np.random.choice([11, 12, 13, 14])
                # Sometimes add a gate
                gt = 0; gfn = None
                if np.random.random() < 0.3:
                    gt = np.random.choice([10, 15, 20, 25])
                    gfn = np.random.choice([sig_polarity, sig_table, sig_combined, sig_set])
                # Sometimes add a trigger
                trig = None
                if np.random.random() < 0.2:
                    trig = np.random.choice(triggers)
                quick_test(f"N={top_n} PERT {wlabel(new_w)}", new_w, top_n,
                         trigger=trig, gate_t=gt, gate_fn=gfn)
                perturbation_count += 1
                if perturbation_count % 1000 == 0:
                    print(f"  ... {perturbation_count} perturbations, {elapsed()/60:.0f}min", flush=True)

    print(f"  Total perturbations: {perturbation_count}")
    print_leaderboard("AFTER PHASE 11", 20, 50)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 12: NEW DATA CHECK + RE-TEST
    # ═══════════════════════════════════════════════════════════════════
    new_data, new_files = load_all_data()
    if len(new_data) != len(all_data):
        phase_header(f"NEW DATA: {len(new_data)} spins (was {len(all_data)})", 15)
        all_data = new_data
        files = new_files
        print(f"  Re-testing top 30 configs on new data...")
        sys.stdout.flush()
        for label, r, cfg in list(LEADERBOARD[:30]):
            if cfg and 'w' in cfg:
                gfn = None
                if cfg.get('gfn') == 'sig_polarity': gfn = sig_polarity
                elif cfg.get('gfn') == 'sig_table': gfn = sig_table
                elif cfg.get('gfn') == 'sig_combined': gfn = sig_combined
                elif cfg.get('gfn') == 'sig_set': gfn = sig_set
                nr = backtest(all_data, cfg['w'], cfg.get('n', 12), 80,
                            cfg.get('t'), cfg.get('g', 0), gfn)
                if nr:
                    add_to_leaderboard(f"[NEWDATA] {wlabel(cfg['w'])}", nr, cfg)
        print_leaderboard("AFTER NEW DATA RETEST", 15, 100)

    # ═══════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════
    finalize(files, all_data)


def finalize(files, all_data):
    runtime = time.time()
    print(f"\n{'='*120}")
    print(f"{'='*120}")
    print(f"FINAL REPORT — Marathon V2 Complete")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total strategies tested: {sum(1 for _ in LEADERBOARD)} in leaderboard")
    print(f"{'='*120}")
    print(f"{'='*120}")

    print(f"\nData: {len(all_data)} spins from {len(files)} files")
    for f, s in files:
        print(f"  {f}: {len(s)} spins")

    # ── TOP 50 OVERALL ──
    print(f"\n{'='*120}")
    print("TOP 50 STRATEGIES — OVERALL RANKING")
    print(f"{'='*120}")
    print(f"  {'#':>3} {'Config':<85} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'Bet%':>5}")
    print(f"  {'-'*120}")
    seen = set()
    count = 0
    for l, r, c in LEADERBOARD:
        if r['bets'] >= 30 and l not in seen:
            seen.add(l)
            count += 1
            star = " ★" if r['hit_rate'] >= 35 else ""
            print(f"  {count:>3} {l:<85} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f} {r['bet_pct']:>4.0f}%{star}")
            if count >= 50: break

    # ── TOP 20 RELIABLE (min 200 bets) ──
    reliable = [(l, r, c) for l, r, c in LEADERBOARD if r['bets'] >= 200]
    reliable.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n{'='*120}")
    print("TOP 20 RELIABLE STRATEGIES (min 200 bets — needed for confidence)")
    print(f"{'='*120}")
    print(f"  {'#':>3} {'Config':<85} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8}")
    print(f"  {'-'*115}")
    seen = set(); count = 0
    for l, r, c in reliable:
        if l not in seen:
            seen.add(l)
            count += 1
            print(f"  {count:>3} {l:<85} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f}")
            if count >= 20: break

    # ── TOP 20 RELIABLE (min 500 bets) ──
    very_reliable = [(l, r, c) for l, r, c in LEADERBOARD if r['bets'] >= 500]
    very_reliable.sort(key=lambda x: -x[1]['hit_rate'])
    print(f"\n{'='*120}")
    print("TOP 20 VERY RELIABLE STRATEGIES (min 500 bets — high confidence)")
    print(f"{'='*120}")
    print(f"  {'#':>3} {'Config':<85} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8}")
    print(f"  {'-'*115}")
    seen = set(); count = 0
    for l, r, c in very_reliable:
        if l not in seen:
            seen.add(l)
            count += 1
            print(f"  {count:>3} {l:<85} {r['bets']:>5} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} ${r['profit']:>+7.0f}")
            if count >= 20: break

    # ── TOP 15 MOST PROFITABLE ──
    profitable = [(l, r, c) for l, r, c in LEADERBOARD if r['profit'] > 0]
    profitable.sort(key=lambda x: -x[1]['profit'])
    print(f"\n{'='*120}")
    print("TOP 15 MOST PROFITABLE")
    print(f"{'='*120}")
    seen = set(); count = 0
    for l, r, c in profitable:
        if l not in seen:
            seen.add(l)
            count += 1
            print(f"  {count:>3} {l:<85} ${r['profit']:>+8.0f} hit={r['hit_rate']}% bets={r['bets']}")
            if count >= 15: break

    # ── TOP 10 HIGH HIT RATE with min 100 bets ──
    high_hr = [(l, r, c) for l, r, c in LEADERBOARD if r['bets'] >= 100 and r['hit_rate'] >= 35]
    high_hr.sort(key=lambda x: -x[1]['hit_rate'])
    if high_hr:
        print(f"\n{'='*120}")
        print("★ STRATEGIES HITTING 35%+ TARGET (min 100 bets) ★")
        print(f"{'='*120}")
        seen = set(); count = 0
        for l, r, c in high_hr:
            if l not in seen:
                seen.add(l)
                count += 1
                print(f"  {count:>3} {l:<85} {r['hit_rate']:>6.2f}% ${r['profit']:>+7.0f} bets={r['bets']}")
                if count >= 10: break

    # ── CROSS-VALIDATION TOP 10 ──
    print(f"\n{'='*120}")
    print("CROSS-VALIDATION — Top 10 strategies on each individual file")
    print(f"{'='*120}")

    top_cv = []
    seen = set()
    for l, r, c in LEADERBOARD:
        if c and l not in seen and r['bets'] >= 100:
            seen.add(l); top_cv.append((l, c))
            if len(top_cv) >= 10: break

    print(f"\n  {'Strategy':<65}", end='')
    for f, _ in files:
        fn = f.replace('.txt','')[:8]
        print(f" {fn:>8}", end='')
    print(f" {'COMBINED':>9} {'Files+':>6}")
    print(f"  {'-'*(65 + 9*(len(files)+1) + 7)}")

    for label, cfg in top_cv:
        wl = wlabel(cfg['w'])[:63]
        print(f"  {wl:<65}", end='')
        files_positive = 0
        for fname, fdata in files:
            r = backtest(fdata, cfg['w'], cfg.get('n', 12),
                        min(60, len(fdata)//5), cfg.get('t'))
            if r:
                m = "+" if r['edge'] > 0 else "-"
                print(f" {r['hit_rate']:>5.1f}%{m}", end='')
                if r['edge'] > 0: files_positive += 1
            else:
                print(f" {'N/A':>7}", end='')
        # Combined
        gfn = None
        if cfg.get('gfn') == 'sig_polarity': gfn = sig_polarity
        elif cfg.get('gfn') == 'sig_table': gfn = sig_table
        elif cfg.get('gfn') == 'sig_combined': gfn = sig_combined
        elif cfg.get('gfn') == 'sig_set': gfn = sig_set
        r = backtest(all_data, cfg['w'], cfg.get('n', 12), 80, cfg.get('t'),
                    cfg.get('g', 0), gfn)
        if r:
            print(f" {r['hit_rate']:>5.1f}%{'+'if r['edge']>0 else'-'}", end='')
        else:
            print(f" {'N/A':>8}", end='')
        print(f" {files_positive:>3}/{len(files)}")

    # ── FINAL VERDICT ──
    print(f"\n{'='*120}")
    print("★ ★ ★  FINAL VERDICT — BEST MODELS  ★ ★ ★")
    print(f"{'='*120}")

    if LEADERBOARD:
        best_l, best_r, best_c = LEADERBOARD[0]
        print(f"\n  HIGHEST HIT RATE:")
        print(f"     {best_l}")
        print(f"     Hit: {best_r['hit_rate']}% | Edge: {best_r['edge']:+.2f}% | Profit: ${best_r['profit']:+.0f}")
        print(f"     Bets: {best_r['bets']} ({best_r['bet_pct']}% of spins)")

        if reliable:
            rl, rr, rc = reliable[0]
            print(f"\n  BEST RELIABLE (200+ bets):")
            print(f"     {rl}")
            print(f"     Hit: {rr['hit_rate']}% | Edge: {rr['edge']:+.2f}% | Profit: ${rr['profit']:+.0f}")
            print(f"     Bets: {rr['bets']} ({rr['bet_pct']}% of spins)")

        if very_reliable:
            vl, vr, vc = very_reliable[0]
            print(f"\n  BEST VERY RELIABLE (500+ bets):")
            print(f"     {vl}")
            print(f"     Hit: {vr['hit_rate']}% | Edge: {vr['edge']:+.2f}% | Profit: ${vr['profit']:+.0f}")
            print(f"     Bets: {vr['bets']} ({vr['bet_pct']}% of spins)")

        if profitable:
            pl, pr_, pc = profitable[0]
            print(f"\n  MOST PROFITABLE:")
            print(f"     {pl}")
            print(f"     Profit: ${pr_['profit']:+.0f} | Hit: {pr_['hit_rate']}% | Bets: {pr_['bets']}")

        if high_hr:
            hl, hr_, hc = high_hr[0]
            print(f"\n  HIGHEST HIT RATE (100+ bets, 35%+ target):")
            print(f"     {hl}")
            print(f"     Hit: {hr_['hit_rate']}% | Profit: ${hr_['profit']:+.0f} | Bets: {hr_['bets']}")

    print(f"\n  Baseline: 32.43% (12/37)")
    print(f"  Break-even N=12: 33.33%")
    print(f"  Break-even N=13: 36.11%")
    print(f"  TARGET: 35-36% sustained for +$100/session")

    # ── RECOMMENDATION ──
    print(f"\n{'='*120}")
    print("RECOMMENDATION FOR LIVE PLAY")
    print(f"{'='*120}")

    # Find best reliable with high bet%
    usable = [(l, r, c) for l, r, c in LEADERBOARD
              if r['bets'] >= 500 and r['hit_rate'] > 33.5 and r['bet_pct'] > 50]
    usable.sort(key=lambda x: -x[1]['profit'])
    if usable:
        ul, ur, uc = usable[0]
        print(f"\n  BEST FOR DAILY USE (500+ bets, >50% bet frequency):")
        print(f"     {ul}")
        print(f"     Hit: {ur['hit_rate']}% | Edge: {ur['edge']:+.2f}% | Profit: ${ur['profit']:+.0f}")
        print(f"     Bets: {ur['bets']} ({ur['bet_pct']}% of spins)")

    selective = [(l, r, c) for l, r, c in LEADERBOARD
                 if r['bets'] >= 100 and r['hit_rate'] >= 35]
    selective.sort(key=lambda x: -x[1]['profit'])
    if selective:
        sl, sr, sc = selective[0]
        print(f"\n  BEST SELECTIVE (35%+ hit rate, 100+ bets):")
        print(f"     {sl}")
        print(f"     Hit: {sr['hit_rate']}% | Edge: {sr['edge']:+.2f}% | Profit: ${sr['profit']:+.0f}")
        print(f"     Bets: {sr['bets']} ({sr['bet_pct']}% of spins)")

    print(f"\n{'='*120}")
    print(f"Marathon V2 finished. Total tested: {len(LEADERBOARD)} strategies in leaderboard.")
    print(f"{'='*120}\n")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
