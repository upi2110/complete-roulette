#!/usr/bin/env python3
"""
MARATHON V3 — 24-HOUR SESSION-OPTIMIZED REGRESSION SEARCH

KEY DIFFERENCES from V2:
1. FITNESS = avg profit per 30-spin session (not overall hit rate)
2. Cross-validates across EACH FILE individually (not just combined data)
3. New models: recency_markov, zone6, streak_pattern, alternation, adaptive_freq
4. Genetic/evolutionary approach: mutate + crossover top strategies
5. Variable bet sizing: $2-$10 per number based on confidence
6. Adaptive N: dynamic number count (10-18) based on signal strength
7. Multi-objective: maximize profit while minimizing max drawdown per session

PHASES (24 hours = 1440 minutes):
Phase 1:  Single model baselines with variable N, window     — 120 min
Phase 2:  2-model pairs session-optimized                    — 180 min
Phase 3:  3-model combos session-optimized                   — 180 min
Phase 4:  New models + hybrid combos                         — 120 min
Phase 5:  Variable bet sizing & adaptive N                   — 120 min
Phase 6:  Cross-file robustness filter                       — 90 min
Phase 7:  Genetic evolution of top strategies                — 240 min
Phase 8:  Deep perturbation of survivors                     — 180 min
Phase 9:  Progressive betting strategies                     — 60 min
Phase 10: Final cross-validation & stress test               — 60 min
Phase 11: Random walk perturbations (remaining time)         — remaining
FINAL:    Report generation                                  — 30 min
"""

import sys
import os
import time
import random
import json
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from copy import deepcopy

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
RUN_HOURS = 24

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

# Number to dozen/column mapping
def get_dozen(n):
    if n == 0: return 0
    if n <= 12: return 1
    if n <= 24: return 2
    return 3

def get_column(n):
    if n == 0: return 0
    return ((n - 1) % 3) + 1

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
# ALL MODELS (including NEW ones)
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

def m_pattern3(history):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 200: return probs
    trans = defaultdict(lambda: np.ones(TOTAL_NUMBERS) * 0.5)
    for i in range(3, len(history)):
        key = (POL_MAP[history[i-3]], POL_MAP[history[i-2]], POL_MAP[history[i-1]])
        trans[key][history[i]] += 1
    if len(history) >= 3:
        key = (POL_MAP[history[-3]], POL_MAP[history[-2]], POL_MAP[history[-1]])
        row = trans[key]
        return row / row.sum()
    return probs

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


# ═══════════════════════════════════════════════════════════════════════════
# NEW MODELS FOR V3
# ═══════════════════════════════════════════════════════════════════════════

def m_recency_markov(history, decay=0.95):
    """Markov chain with exponential recency weighting — newer transitions matter more."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 50: return probs
    trans = np.ones((TOTAL_NUMBERS, TOTAL_NUMBERS)) * 0.1
    total_len = len(history)
    for i in range(1, total_len):
        w = decay ** (total_len - 1 - i)
        trans[history[i-1]][history[i]] += w
    last = history[-1]
    row = trans[last]
    return row / row.sum()

def m_zone6(history, window=40):
    """Divide wheel into 6 physical zones, boost hot zones."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 15: return probs
    recent = history[-window:]
    num_zones = 6
    zone_size = len(WHEEL_ORDER) // num_zones
    zc = defaultdict(int)
    for n in recent:
        pos = NUMBER_TO_POSITION.get(n, 0)
        zc[min(pos // zone_size, num_zones - 1)] += 1
    total = len(recent)
    avg = total / num_zones
    for num in range(TOTAL_NUMBERS):
        pos = NUMBER_TO_POSITION.get(num, 0)
        z = min(pos // zone_size, num_zones - 1)
        ratio = zc.get(z, 0) / total
        exp = 1.0 / num_zones
        if ratio > exp:
            probs[num] *= 1.0 + (ratio - exp) * 4
        else:
            probs[num] *= max(0.5, 1.0 - (exp - ratio) * 3)
    return probs / probs.sum()

def m_streak_pattern(history):
    """Track what typically follows streaks of same polarity."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 100: return probs

    # Track: after 2+ same polarity in a row, what comes next?
    trans_after_streak = defaultdict(lambda: np.ones(TOTAL_NUMBERS) * 0.5)
    for i in range(2, len(history)):
        p1 = POL_MAP[history[i-2]]
        p2 = POL_MAP[history[i-1]]
        if p1 == p2:
            streak_key = (p2, 'streak')
            trans_after_streak[streak_key][history[i]] += 1
        else:
            alt_key = (p2, 'alt')
            trans_after_streak[alt_key][history[i]] += 1

    if len(history) >= 2:
        p1 = POL_MAP[history[-2]]
        p2 = POL_MAP[history[-1]]
        if p1 == p2:
            key = (p2, 'streak')
        else:
            key = (p2, 'alt')
        row = trans_after_streak[key]
        return row / row.sum()
    return probs

def m_table_streak(history):
    """Track what follows table streaks (consecutive 0-table or 19-table)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 100: return probs
    trans = defaultdict(lambda: np.ones(TOTAL_NUMBERS) * 0.5)
    for i in range(2, len(history)):
        t1 = TABLE_MAP[history[i-2]]
        t2 = TABLE_MAP[history[i-1]]
        if t1 == t2:
            key = (t2, 'streak')
        else:
            key = (t2, 'alt')
        trans[key][history[i]] += 1
    if len(history) >= 2:
        t1 = TABLE_MAP[history[-2]]
        t2 = TABLE_MAP[history[-1]]
        key = (t2, 'streak') if t1 == t2 else (t2, 'alt')
        row = trans[key]
        return row / row.sum()
    return probs

def m_adaptive_freq(history):
    """Frequency using optimal blend of short + long window."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 30: return probs

    # Blend short-term (last 15) and medium-term (last 40) frequencies
    short = history[-15:]
    medium = history[-min(40, len(history)):]

    sc = np.ones(TOTAL_NUMBERS) * 0.5
    for n in short: sc[n] += 1.5  # weight recent more
    mc = np.ones(TOTAL_NUMBERS) * 0.5
    for n in medium: mc[n] += 1

    blended = 0.6 * (sc / sc.sum()) + 0.4 * (mc / mc.sum())
    return blended / blended.sum()

def m_dozen_pair(history, window=40):
    """Track which dozen pair (1+2, 1+3, 2+3) is dominant."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 15: return probs
    recent = history[-window:]
    d = {1:0, 2:0, 3:0}
    for n in recent:
        dz = get_dozen(n)
        if dz > 0: d[dz] += 1
    total = sum(d.values())
    if total == 0: return probs
    # Find weakest dozen, boost the other two
    weakest = min(d, key=d.get)
    for num in range(1, TOTAL_NUMBERS):
        ndz = get_dozen(num)
        if ndz != weakest:
            probs[num] *= 1.2
        else:
            probs[num] *= 0.6
    return probs / probs.sum()

def m_neighbor_hot(history, window=25):
    """Boost physical neighbors of recent hot numbers on wheel."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    rc = Counter(recent)
    exp = window / TOTAL_NUMBERS
    hot_nums = [n for n, c in rc.items() if c > exp * 1.3]
    if not hot_nums: return probs
    for n in hot_nums:
        pos = NUMBER_TO_POSITION.get(n, 0)
        cnt = rc[n]
        boost = 1.0 + (cnt / window - exp / window) * 5
        for off in range(-3, 4):
            npos = (pos + off) % len(WHEEL_ORDER)
            wnum = WHEEL_ORDER[npos]
            probs[wnum] *= boost * (1.0 - abs(off) * 0.15)
    return probs / probs.sum()

def m_contrarian_gap(history):
    """Opposite of gap — bet on numbers that are due (large gap since last seen)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 80: return probs
    last_seen = {}
    for i, n in enumerate(history): last_seen[n] = i
    mx = len(history)
    for num in range(TOTAL_NUMBERS):
        gap = mx - last_seen.get(num, 0)
        expected_gap = TOTAL_NUMBERS
        if gap > expected_gap * 2.5: probs[num] *= 2.0
        elif gap > expected_gap * 2: probs[num] *= 1.6
        elif gap > expected_gap * 1.5: probs[num] *= 1.3
        elif gap < expected_gap * 0.5: probs[num] *= 0.7
    return probs / probs.sum()

def m_set_contrarian(history, window=50):
    """Bet against the hot set — mean reversion."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10: return probs
    recent = history[-window:]
    total = len(recent)
    sc = {1:0, 2:0, 3:0}
    for n in recent: sc[SET_MAP[n]] += 1
    cold_set = min(sc, key=sc.get)
    if sc[cold_set] < total / 3 * 0.9:
        for num in range(TOTAL_NUMBERS):
            if SET_MAP[num] == cold_set:
                probs[num] *= 1.4
            else:
                probs[num] *= 0.8
    return probs / probs.sum()

def m_combined_trend(history, window=30):
    """Combine multiple binary trends into single score."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 15: return probs
    recent = history[-window:]
    total = len(recent)

    # Count all binary properties
    counts = {'pos': 0, 'neg': 0, 't0': 0, 't19': 0,
              'odd': 0, 'even': 0, 'hi': 0, 'lo': 0,
              'red': 0, 'black': 0}
    for n in recent:
        if POL_MAP[n] == 'pos': counts['pos'] += 1
        else: counts['neg'] += 1
        if TABLE_MAP[n] == '0': counts['t0'] += 1
        else: counts['t19'] += 1
        if n > 0 and n % 2 == 1: counts['odd'] += 1
        elif n > 0: counts['even'] += 1
        if 1 <= n <= 18: counts['lo'] += 1
        elif 19 <= n <= 36: counts['hi'] += 1
        if n in RED: counts['red'] += 1
        elif n > 0: counts['black'] += 1

    for num in range(1, TOTAL_NUMBERS):
        score = 1.0
        # Polarity
        if POL_MAP[num] == 'pos' and counts['pos'] > counts['neg']:
            score *= 1.0 + (counts['pos'] - counts['neg']) / total * 0.5
        elif POL_MAP[num] == 'neg' and counts['neg'] > counts['pos']:
            score *= 1.0 + (counts['neg'] - counts['pos']) / total * 0.5
        # Table
        if TABLE_MAP[num] == '0' and counts['t0'] > counts['t19']:
            score *= 1.0 + (counts['t0'] - counts['t19']) / total * 0.5
        elif TABLE_MAP[num] == '19' and counts['t19'] > counts['t0']:
            score *= 1.0 + (counts['t19'] - counts['t0']) / total * 0.5
        # Odd/Even
        if num % 2 == 1 and counts['odd'] > counts['even']:
            score *= 1.0 + (counts['odd'] - counts['even']) / total * 0.3
        elif num % 2 == 0 and counts['even'] > counts['odd']:
            score *= 1.0 + (counts['even'] - counts['odd']) / total * 0.3
        # High/Low
        if 1 <= num <= 18 and counts['lo'] > counts['hi']:
            score *= 1.0 + (counts['lo'] - counts['hi']) / total * 0.3
        elif 19 <= num <= 36 and counts['hi'] > counts['lo']:
            score *= 1.0 + (counts['hi'] - counts['lo']) / total * 0.3
        probs[num] = score

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
    # NEW V3 models
    'rec_markov': m_recency_markov,
    'zone6': m_zone6,
    'streak_pat': m_streak_pattern,
    'tab_streak': m_table_streak,
    'adapt_freq': m_adaptive_freq,
    'dozen_pair': m_dozen_pair,
    'nbr_hot': m_neighbor_hot,
    'contra_gap': m_contrarian_gap,
    'set_contra': m_set_contrarian,
    'combo_trend': m_combined_trend,
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

def sig_any_strong(history, window=50):
    """Returns max of all signals — bet when ANY imbalance is strong."""
    return max(sig_polarity(history, window),
               sig_table(history, window),
               sig_set(history, window))


# ═══════════════════════════════════════════════════════════════════════════
# SESSION-BASED BACKTEST (THE KEY DIFFERENCE FROM V2)
# ═══════════════════════════════════════════════════════════════════════════

def session_backtest(data, weights, top_n=12, session_size=30,
                     warmup=80, gate_t=0, gate_fn=None, trigger=None,
                     bet_per_num=2.0):
    """
    Simulates playing in sessions of `session_size` spins.
    History accumulates across sessions (AI learns from all past spins).
    Returns per-session results.
    """
    if len(data) < warmup + session_size:
        return None

    sessions = []
    spin_idx = warmup

    while spin_idx + session_size <= len(data):
        session_spins = data[spin_idx:spin_idx + session_size]
        session_hits = 0
        session_bets = 0
        session_skipped = 0
        session_profit = 0.0

        for j, actual in enumerate(session_spins):
            history = data[:spin_idx + j]

            # Gate check
            if gate_t > 0 and gate_fn:
                if gate_fn(history) < gate_t:
                    session_skipped += 1
                    continue

            # Trigger check
            restricted = None
            idx = spin_idx + j
            if trigger == 'pol_double_follow':
                if idx < 2: session_skipped += 1; continue
                if POL_MAP[data[idx-1]] != POL_MAP[data[idx-2]]: session_skipped += 1; continue
                restricted = POS_NUMS if POL_MAP[data[idx-1]] == 'pos' else NEG_NUMS
            elif trigger == 'pol_double_oppose':
                if idx < 2: session_skipped += 1; continue
                if POL_MAP[data[idx-1]] != POL_MAP[data[idx-2]]: session_skipped += 1; continue
                restricted = NEG_NUMS if POL_MAP[data[idx-1]] == 'pos' else POS_NUMS
            elif trigger == 'alternating':
                if idx < 2: session_skipped += 1; continue
                if POL_MAP[data[idx-1]] == POL_MAP[data[idx-2]]: session_skipped += 1; continue
                restricted = POS_NUMS if POL_MAP[data[idx-1]] == 'pos' else NEG_NUMS
            elif trigger == 'table_double_oppose':
                if idx < 2: session_skipped += 1; continue
                if TABLE_MAP[data[idx-1]] != TABLE_MAP[data[idx-2]]: session_skipped += 1; continue
                restricted = TABLE_19_NUMS if TABLE_MAP[data[idx-1]] == '0' else TABLE_0_NUMS
            elif trigger == 'tab_alternating':
                if idx < 2: session_skipped += 1; continue
                if TABLE_MAP[data[idx-1]] == TABLE_MAP[data[idx-2]]: session_skipped += 1; continue
                restricted = TABLE_0_NUMS if TABLE_MAP[data[idx-1]] == '0' else TABLE_19_NUMS

            # Ensemble prediction
            ensemble = np.zeros(TOTAL_NUMBERS)
            for model, weight in weights.items():
                if weight <= 0: continue
                fn = ALL_MODELS.get(model)
                if fn:
                    ensemble += weight * fn(history)
            s = ensemble.sum()
            if s > 0: ensemble /= s

            if restricted:
                gp = [(n, ensemble[n]) for n in restricted]
                gp.sort(key=lambda x: x[1], reverse=True)
                top = set(n for n, p in gp[:top_n])
            else:
                top = set(np.argsort(ensemble)[::-1][:top_n])

            hit = 1 if actual in top else 0
            session_hits += hit
            session_bets += 1
            cost = top_n * bet_per_num
            if hit:
                session_profit += PAYOUT * bet_per_num + bet_per_num - cost
            else:
                session_profit -= cost

        if session_bets > 0:
            sessions.append({
                'hits': session_hits,
                'bets': session_bets,
                'skipped': session_skipped,
                'hit_rate': round(session_hits / session_bets * 100, 2),
                'profit': round(session_profit, 2),
            })

        spin_idx += session_size

    if not sessions:
        return None

    total_sessions = len(sessions)
    winning_sessions = sum(1 for s in sessions if s['profit'] > 0)
    plus100_sessions = sum(1 for s in sessions if s['profit'] >= 100)
    avg_profit = np.mean([s['profit'] for s in sessions])
    avg_bets = np.mean([s['bets'] for s in sessions])
    total_profit = sum(s['profit'] for s in sessions)
    min_profit = min(s['profit'] for s in sessions)
    max_profit = max(s['profit'] for s in sessions)
    median_profit = np.median([s['profit'] for s in sessions])
    # Consistency score: penalize high variance
    profit_std = np.std([s['profit'] for s in sessions])

    return {
        'sessions': total_sessions,
        'winning': winning_sessions,
        'win_pct': round(winning_sessions / total_sessions * 100, 1),
        'plus100': plus100_sessions,
        'plus100_pct': round(plus100_sessions / total_sessions * 100, 1),
        'avg_profit': round(avg_profit, 2),
        'median_profit': round(median_profit, 2),
        'avg_bets': round(avg_bets, 1),
        'total_profit': round(total_profit, 2),
        'min_profit': round(min_profit, 2),
        'max_profit': round(max_profit, 2),
        'profit_std': round(profit_std, 2),
    }


# Also keep regular backtest for quick screening
def quick_backtest(data, weights, top_n=12, warmup=80, gate_t=0, gate_fn=None, trigger=None):
    """Fast backtest returning just hit_rate and profit."""
    if len(data) <= warmup + 10: return None
    hits = 0; total_bets = 0
    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]
        if gate_t > 0 and gate_fn:
            if gate_fn(history) < gate_t: continue

        restricted = None
        if trigger == 'alternating':
            if i < warmup+2: continue
            if POL_MAP[data[i-1]] == POL_MAP[data[i-2]]: continue
            restricted = POS_NUMS if POL_MAP[data[i-1]] == 'pos' else NEG_NUMS

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
        hits += hit; total_bets += 1

    if total_bets == 0: return None
    hr = hits / total_bets * 100
    bl = top_n / 37 * 100
    profit = hits * (PAYOUT+1) * BET_PER_NUM - total_bets * top_n * BET_PER_NUM
    return {'bets': total_bets, 'hit_rate': round(hr, 2), 'edge': round(hr - bl, 2),
            'profit': round(profit, 2), 'bet_pct': round(total_bets / (len(data) - warmup) * 100, 1)}


def wlabel(w):
    parts = []
    for k, v in sorted(w.items(), key=lambda x: -x[1]):
        if v > 0:
            parts.append(f"{k}({int(v*100)}%)")
    return "+".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION — multi-objective
# ═══════════════════════════════════════════════════════════════════════════

def fitness_score(session_result, cross_file_results=None):
    """
    Compute a single fitness score from session results.
    Optimizes for: avg_profit (primary), win_pct, consistency, cross-file robustness
    """
    if session_result is None: return -9999

    avg_p = session_result['avg_profit']
    win_pct = session_result['win_pct']
    sessions = session_result['sessions']
    min_p = session_result['min_profit']
    avg_bets = session_result['avg_bets']

    # Must have enough sessions for statistical significance
    if sessions < 5: return -9999
    # Must bet at least 10 times per session on average
    if avg_bets < 10: return avg_p * 0.3  # heavy penalty for low activity

    # Base score = avg profit per session
    score = avg_p

    # Bonus for high win rate (more consistent)
    if win_pct > 60: score += (win_pct - 60) * 0.5
    if win_pct > 50: score += (win_pct - 50) * 0.3

    # Penalty for extreme min loss (downside protection)
    if min_p < -400: score -= 5
    if min_p < -500: score -= 10

    # Cross-file robustness bonus
    if cross_file_results:
        profitable_files = sum(1 for r in cross_file_results if r and r['avg_profit'] > 0)
        total_files = len(cross_file_results)
        robustness = profitable_files / total_files if total_files > 0 else 0
        score *= (0.5 + 0.5 * robustness)  # 50-100% multiplier based on file consistency

    return score


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL LEADERBOARD (session-based)
# ═══════════════════════════════════════════════════════════════════════════

LEADERBOARD = []
MAX_LEADERS = 500
TESTED_COUNT = 0
START_TIME = time.time()

def add_to_leaderboard(label, session_result, config=None, cross_results=None):
    global LEADERBOARD
    if session_result is None: return
    score = fitness_score(session_result, cross_results)
    entry = {
        'label': label,
        'result': session_result,
        'config': config,
        'score': score,
        'cross': cross_results,
    }
    LEADERBOARD.append(entry)
    LEADERBOARD.sort(key=lambda x: -x['score'])
    if len(LEADERBOARD) > MAX_LEADERS:
        LEADERBOARD = LEADERBOARD[:MAX_LEADERS]

def print_leaderboard(title="CURRENT LEADERBOARD", top=20, min_sessions=5):
    filtered = [e for e in LEADERBOARD if e['result']['sessions'] >= min_sessions]
    print(f"\n  {'='*130}")
    print(f"  {title}")
    print(f"  {'='*130}")
    print(f"  {'#':>3} {'Config':<65} {'Sess':>4} {'Win%':>5} {'+$100':>5} {'AvgP':>7} {'MedP':>7} {'Bets':>4} {'MinP':>7} {'MaxP':>7} {'TotP':>8} {'Score':>6}")
    print(f"  {'-'*130}")
    for i, e in enumerate(filtered[:top]):
        r = e['result']
        print(f"  {i+1:>3} {e['label']:<65} {r['sessions']:>4} {r['win_pct']:>4.0f}% {r['plus100_pct']:>4.0f}% ${r['avg_profit']:>+6.0f} ${r['median_profit']:>+6.0f} {r['avg_bets']:>4.0f} ${r['min_profit']:>+6.0f} ${r['max_profit']:>+6.0f} ${r['total_profit']:>+7.0f} {e['score']:>6.1f}")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def elapsed_min():
    return (time.time() - START_TIME) / 60

def remaining_min():
    return RUN_HOURS * 60 - elapsed_min()

def phase_deadline(budget_min):
    return time.time() + budget_min * 60

def phase_header(name, budget):
    global TESTED_COUNT
    e = int(elapsed_min())
    r = int(remaining_min())
    print(f"\n{'='*130}")
    print(f"PHASE: {name}")
    print(f"Time: {e}min elapsed, {r}min remaining | Budget: {budget}min | Tested: {TESTED_COUNT}")
    print(f"{'='*130}")
    sys.stdout.flush()


def test_strategy(all_data, files, label, weights, top_n=12, session_size=30,
                  warmup=80, gate_t=0, gate_fn=None, trigger=None,
                  bet_per_num=2.0, do_cross=False):
    """Test a strategy and add to leaderboard."""
    global TESTED_COUNT
    TESTED_COUNT += 1

    r = session_backtest(all_data, weights, top_n, session_size,
                         warmup=warmup, gate_t=gate_t, gate_fn=gate_fn,
                         trigger=trigger, bet_per_num=bet_per_num)
    if r is None: return None

    cross = None
    if do_cross and files:
        cross = []
        for fname, fdata in files:
            wu = min(60, len(fdata) // 5)
            fr = session_backtest(fdata, weights, top_n, session_size,
                                  warmup=wu, gate_t=gate_t, gate_fn=gate_fn,
                                  trigger=trigger, bet_per_num=bet_per_num)
            cross.append(fr)

    config = {'w': weights, 'n': top_n, 'gate_t': gate_t, 'trigger': trigger,
              'bet': bet_per_num, 'warmup': warmup, 'session_size': session_size}
    add_to_leaderboard(label, r, config, cross)
    return r


def random_weights(models, count=2):
    """Generate random weight combination for given models."""
    chosen = random.sample(models, min(count, len(models)))
    raw = [random.random() for _ in chosen]
    total = sum(raw)
    weights = {m: round(r / total, 2) for m, r in zip(chosen, raw)}
    # Normalize to sum to 1.0
    s = sum(weights.values())
    if s > 0:
        for k in weights:
            weights[k] = round(weights[k] / s, 2)
    return weights


def mutate_weights(weights, mutation_rate=0.15):
    """Slightly mutate a weight configuration."""
    w = dict(weights)
    all_m = list(ALL_MODELS.keys())

    # Randomly adjust existing weights
    for k in list(w.keys()):
        w[k] += random.gauss(0, mutation_rate)
        if w[k] < 0.01:
            del w[k]

    # Occasionally add a new model
    if random.random() < 0.15:
        new_m = random.choice(all_m)
        if new_m not in w:
            w[new_m] = random.uniform(0.05, 0.3)

    # Normalize
    s = sum(v for v in w.values() if v > 0)
    if s <= 0: return weights
    return {k: round(v / s, 2) for k, v in w.items() if v > 0.01}


def crossover_weights(w1, w2):
    """Crossover two weight configurations."""
    all_keys = set(w1.keys()) | set(w2.keys())
    child = {}
    for k in all_keys:
        v1 = w1.get(k, 0)
        v2 = w2.get(k, 0)
        if random.random() < 0.5:
            child[k] = v1
        else:
            child[k] = v2
        # Blend with probability
        if random.random() < 0.3:
            child[k] = (v1 + v2) / 2
    # Remove zeros
    child = {k: v for k, v in child.items() if v > 0.01}
    if not child: return w1
    s = sum(child.values())
    return {k: round(v / s, 2) for k, v in child.items()}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global START_TIME, TESTED_COUNT
    START_TIME = time.time()

    all_data, files = load_all_data()
    print("=" * 130)
    print(f"MARATHON V3 — 24-HOUR SESSION-OPTIMIZED REGRESSION SEARCH")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"KEY: Fitness = avg profit per 30-spin session (NOT overall hit rate)")
    print("=" * 130)
    print(f"\nLoaded {len(all_data)} spins from {len(files)} files:")
    for fname, spins in files:
        print(f"  {fname}: {len(spins)} spins")
    print(f"\nModels available: {len(ALL_MODELS)} ({len(ALL_MODELS) - 23} NEW in V3)")
    print(f"New models: rec_markov, zone6, streak_pat, tab_streak, adapt_freq, "
          f"dozen_pair, nbr_hot, contra_gap, set_contra, combo_trend")
    print(f"\nTarget: maximize avg profit per 30-spin session")
    print(f"Current best: +$26/session (markov+pattern3) — need to beat this!")

    gates = [
        (sig_polarity, 'pol'), (sig_table, 'tab'),
        (sig_combined, 'comb'), (sig_set, 'set'),
        (sig_any_strong, 'any'),
    ]

    triggers = ['pol_double_follow', 'pol_double_oppose', 'alternating',
                'table_double_oppose', 'tab_alternating']

    # Top models from V2 + new V3 models
    core_models = ['markov', 'pattern3', 'table', 'odd_even', 'set', 'sector',
                   'high_low', 'tab_contra', 'freq', 'gap', 'hot']
    new_models = ['rec_markov', 'zone6', 'streak_pat', 'tab_streak', 'adapt_freq',
                  'dozen_pair', 'nbr_hot', 'contra_gap', 'set_contra', 'combo_trend']
    all_model_keys = list(ALL_MODELS.keys())

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: SINGLE MODEL BASELINES — 120 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("SINGLE MODEL BASELINES (all models × N=10-16 × windows)", 120)
    deadline = phase_deadline(120)

    for mi, model_name in enumerate(all_model_keys):
        if time.time() > deadline: break
        for top_n in [10, 11, 12, 13, 14, 15, 16]:
            w = {model_name: 1.0}
            label = f"N={top_n} {model_name}(100%)"
            test_strategy(all_data, files, label, w, top_n=top_n)
        print(f"  [{mi+1}/{len(all_model_keys)}] {model_name} done ({TESTED_COUNT} tested, {int(elapsed_min())}min)")
        sys.stdout.flush()

    print(f"  Phase 1 complete: {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 1 — SINGLE MODEL BASELINES", 15, 5)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: 2-MODEL PAIRS — 180 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("2-MODEL PAIRS SESSION-OPTIMIZED", 180)
    deadline = phase_deadline(180)

    # Prioritize: try all core+new model pairs at N=12, then fine-tune top ones
    priority_models = core_models + new_models
    pair_count = 0
    for i, m1 in enumerate(priority_models):
        if time.time() > deadline: break
        for m2 in priority_models[i+1:]:
            if time.time() > deadline: break
            # Coarse sweep: 10% steps
            for pct1 in range(10, 100, 10):
                w1 = pct1 / 100.0
                w2 = 1.0 - w1
                weights = {m1: w1, m2: w2}
                for top_n in [12, 13, 14]:
                    label = f"N={top_n} {wlabel(weights)}"
                    test_strategy(all_data, files, label, weights, top_n=top_n)
            pair_count += 1
            if pair_count % 20 == 0:
                print(f"  ... {pair_count} pairs, {TESTED_COUNT} tested, {int(elapsed_min())}min")
                sys.stdout.flush()

    # Fine-tune top 20 pairs with 2% steps
    top_pairs = [(e['config']['w'], e['config']['n'])
                 for e in LEADERBOARD[:20]
                 if e['config'] and len(e['config']['w']) == 2]
    for weights, base_n in top_pairs[:10]:
        if time.time() > deadline: break
        models = list(weights.keys())
        if len(models) != 2: continue
        m1, m2 = models
        base_pct = int(weights[m1] * 100)
        for pct1 in range(max(5, base_pct - 15), min(96, base_pct + 16), 2):
            w1 = pct1 / 100.0
            w2 = 1.0 - w1
            w = {m1: w1, m2: w2}
            for top_n in [base_n - 1, base_n, base_n + 1]:
                if 10 <= top_n <= 16:
                    label = f"N={top_n} {wlabel(w)}"
                    test_strategy(all_data, files, label, w, top_n=top_n)

    print(f"  Phase 2 complete: {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 2 — 2-MODEL PAIRS", 15, 5)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: 3-MODEL COMBOS — 180 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("3-MODEL COMBOS SESSION-OPTIMIZED", 180)
    deadline = phase_deadline(180)

    # Use top 12 performing models based on Phase 1-2
    top_model_set = set()
    for e in LEADERBOARD[:50]:
        if e['config']:
            for m in e['config']['w'].keys():
                top_model_set.add(m)
    top_model_list = list(top_model_set)[:15]

    for combo in combinations(top_model_list, 3):
        if time.time() > deadline: break
        m1, m2, m3 = combo
        for p1 in range(10, 80, 10):
            for p2 in range(10, 90 - p1, 10):
                p3 = 100 - p1 - p2
                if p3 < 5: continue
                w = {m1: p1/100, m2: p2/100, m3: p3/100}
                for top_n in [12, 13, 14]:
                    label = f"N={top_n} {wlabel(w)}"
                    test_strategy(all_data, files, label, w, top_n=top_n)

    # Fine-tune top 3-model combos
    top_3m = [(e['config']['w'], e['config']['n'])
              for e in LEADERBOARD[:30]
              if e['config'] and len(e['config']['w']) == 3]
    for weights, base_n in top_3m[:8]:
        if time.time() > deadline: break
        models = list(weights.keys())
        if len(models) != 3: continue
        for _ in range(20):
            w = mutate_weights(weights, 0.05)
            for top_n in [base_n - 1, base_n, base_n + 1]:
                if 10 <= top_n <= 16:
                    label = f"N={top_n} {wlabel(w)}"
                    test_strategy(all_data, files, label, w, top_n=top_n)

    print(f"  Phase 3 complete: {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 3 — 3-MODEL COMBOS", 15, 5)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: NEW MODELS FOCUSED COMBOS — 120 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("NEW V3 MODELS + HYBRID COMBOS", 120)
    deadline = phase_deadline(120)

    # Pair each new model with top performing old models
    for new_m in new_models:
        if time.time() > deadline: break
        for old_m in core_models[:8]:
            for pct_new in [20, 30, 40, 50, 60, 70, 80]:
                w = {new_m: pct_new/100, old_m: (100-pct_new)/100}
                for top_n in [12, 13, 14]:
                    label = f"N={top_n} {wlabel(w)}"
                    test_strategy(all_data, files, label, w, top_n=top_n)

    # New model 3-way combos
    for combo in combinations(new_models, 2):
        if time.time() > deadline: break
        for old_m in core_models[:5]:
            w = {combo[0]: 0.35, combo[1]: 0.35, old_m: 0.30}
            for top_n in [12, 13, 14]:
                label = f"N={top_n} {wlabel(w)}"
                test_strategy(all_data, files, label, w, top_n=top_n)

    print(f"  Phase 4 complete: {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 4 — NEW MODELS", 15, 5)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: GATES + TRIGGERS on top strategies — 120 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("GATES + TRIGGERS ON TOP STRATEGIES", 120)
    deadline = phase_deadline(120)

    top_configs = [(e['config']['w'], e['config']['n'])
                   for e in LEADERBOARD[:30] if e['config']]

    # Test gates
    for weights, base_n in top_configs:
        if time.time() > deadline: break
        for gfn, gname in gates:
            for gt in [5, 8, 10, 12, 15, 20, 25]:
                for top_n in [base_n - 1, base_n, base_n + 1]:
                    if top_n < 10 or top_n > 16: continue
                    label = f"N={top_n} {wlabel(weights)}|{gname}>={gt}"
                    test_strategy(all_data, files, label, weights,
                                  top_n=top_n, gate_t=gt, gate_fn=gfn)

    # Test triggers
    for weights, base_n in top_configs[:15]:
        if time.time() > deadline: break
        for trig in triggers:
            for top_n in [base_n - 1, base_n, base_n + 1]:
                if top_n < 10 or top_n > 16: continue
                label = f"N={top_n} {wlabel(weights)}|{trig}"
                test_strategy(all_data, files, label, weights,
                              top_n=top_n, trigger=trig)

    # Gate + Trigger combos
    for weights, base_n in top_configs[:10]:
        if time.time() > deadline: break
        for trig in triggers:
            for gfn, gname in gates[:3]:
                for gt in [10, 15, 20, 25]:
                    label = f"N={base_n} {wlabel(weights)}|{trig}|{gname}>={gt}"
                    test_strategy(all_data, files, label, weights,
                                  top_n=base_n, gate_t=gt, gate_fn=gfn,
                                  trigger=trig)

    print(f"  Phase 5 complete: {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 5 — GATES + TRIGGERS", 20, 3)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 6: CROSS-FILE ROBUSTNESS FILTER — 90 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("CROSS-FILE ROBUSTNESS VALIDATION", 90)
    deadline = phase_deadline(90)

    # Re-test top 50 strategies with cross-file validation
    top50 = [(e['config'], e['label']) for e in LEADERBOARD[:50] if e['config']]
    for config, label in top50:
        if time.time() > deadline: break
        w = config['w']
        n = config['n']
        gt = config.get('gate_t', 0)
        trig = config.get('trigger', None)
        bet = config.get('bet', 2.0)
        gfn = None
        if gt > 0:
            # Determine gate function from label
            if 'pol>=' in label: gfn = sig_polarity
            elif 'tab>=' in label: gfn = sig_table
            elif 'comb>=' in label: gfn = sig_combined
            elif 'set>=' in label: gfn = sig_set
            elif 'any>=' in label: gfn = sig_any_strong

        label_x = f"[XVAL] {label}"
        test_strategy(all_data, files, label_x, w, top_n=n,
                      gate_t=gt, gate_fn=gfn, trigger=trig,
                      bet_per_num=bet, do_cross=True)

    print(f"  Phase 6 complete: {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 6 — CROSS-VALIDATED", 20, 3)

    # Show cross-file breakdown for top 5
    xval_entries = [e for e in LEADERBOARD if e['label'].startswith('[XVAL]') and e['cross']]
    xval_entries.sort(key=lambda x: -x['score'])
    print(f"\n  TOP 5 CROSS-FILE BREAKDOWN:")
    for i, e in enumerate(xval_entries[:5]):
        print(f"\n  {i+1}. {e['label']}")
        print(f"     Overall: avg=${e['result']['avg_profit']:+.0f} | win={e['result']['win_pct']:.0f}% | score={e['score']:.1f}")
        if e['cross']:
            for j, (fname, _) in enumerate(files):
                cr = e['cross'][j]
                if cr:
                    print(f"     {fname}: avg=${cr['avg_profit']:+.0f} win={cr['win_pct']:.0f}% sess={cr['sessions']}")
                else:
                    print(f"     {fname}: N/A (too short)")
    sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 7: GENETIC EVOLUTION — 240 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("GENETIC EVOLUTION OF TOP STRATEGIES", 240)
    deadline = phase_deadline(240)

    generation = 0
    population_size = 30
    # Seed population with top strategies
    population = []
    for e in LEADERBOARD[:population_size]:
        if e['config']:
            population.append(e['config'])

    while time.time() < deadline and population:
        generation += 1
        children = []

        # Mutation
        for cfg in population[:20]:
            w = mutate_weights(cfg['w'], 0.10)
            n = cfg['n'] + random.choice([-1, 0, 0, 0, 1])
            n = max(10, min(16, n))
            child = {'w': w, 'n': n, 'gate_t': cfg.get('gate_t', 0),
                     'trigger': cfg.get('trigger', None),
                     'bet': cfg.get('bet', 2.0), 'warmup': 80, 'session_size': 30}
            children.append(child)

        # Crossover
        for _ in range(10):
            if len(population) < 2: break
            p1, p2 = random.sample(population[:15], 2)
            w = crossover_weights(p1['w'], p2['w'])
            n = random.choice([p1['n'], p2['n']])
            child = {'w': w, 'n': n, 'gate_t': 0, 'trigger': None,
                     'bet': 2.0, 'warmup': 80, 'session_size': 30}
            children.append(child)

        # Strong mutation (exploration)
        for _ in range(5):
            w = random_weights(all_model_keys, random.choice([2, 3, 4]))
            n = random.choice([12, 13, 14])
            child = {'w': w, 'n': n, 'gate_t': 0, 'trigger': None,
                     'bet': 2.0, 'warmup': 80, 'session_size': 30}
            children.append(child)

        # Evaluate children
        for cfg in children:
            if time.time() > deadline: break
            label = f"GEN{generation} N={cfg['n']} {wlabel(cfg['w'])}"
            if cfg.get('gate_t', 0) > 0:
                label += f"|gate>={cfg['gate_t']}"
            if cfg.get('trigger'):
                label += f"|{cfg['trigger']}"
            test_strategy(all_data, files, label, cfg['w'], top_n=cfg['n'],
                          gate_t=cfg.get('gate_t', 0), trigger=cfg.get('trigger'))

        # Select top performers for next generation
        population = [e['config'] for e in LEADERBOARD[:population_size] if e['config']]

        if generation % 10 == 0:
            print(f"  ... Generation {generation}, {TESTED_COUNT} total tested, "
                  f"{int(elapsed_min())}min elapsed")
            sys.stdout.flush()

    print(f"  Phase 7 complete: {generation} generations, {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 7 — GENETIC EVOLUTION", 20, 5)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 8: DEEP PERTURBATION — 180 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("DEEP PERTURBATION OF TOP SURVIVORS", 180)
    deadline = phase_deadline(180)

    top_survivors = [(e['config']['w'], e['config']['n'])
                     for e in LEADERBOARD[:30] if e['config']]
    perturb_count = 0

    while time.time() < deadline:
        for weights, base_n in top_survivors:
            if time.time() > deadline: break
            # Small perturbation
            w = mutate_weights(weights, random.uniform(0.03, 0.20))
            n = base_n + random.choice([-1, 0, 0, 1])
            n = max(10, min(16, n))
            label = f"PERTURB N={n} {wlabel(w)}"
            test_strategy(all_data, files, label, w, top_n=n)
            perturb_count += 1

            # Also try with gates
            if random.random() < 0.3:
                gfn, gname = random.choice(gates)
                gt = random.choice([5, 8, 10, 12, 15, 20, 25])
                label2 = f"PERTURB N={n} {wlabel(w)}|{gname}>={gt}"
                test_strategy(all_data, files, label2, w, top_n=n,
                              gate_t=gt, gate_fn=gfn)

        if perturb_count % 1000 == 0:
            print(f"  ... {perturb_count} perturbations, {int(elapsed_min())}min")
            sys.stdout.flush()

    print(f"  Phase 8 complete: {perturb_count} perturbations, {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 8 — DEEP PERTURBATION", 20, 5)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 9: PROGRESSIVE BETTING — 60 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("PROGRESSIVE BETTING (variable bet sizes)", 60)
    deadline = phase_deadline(60)

    # Test top strategies with higher bet sizes
    top_for_bets = [(e['config']['w'], e['config']['n'])
                    for e in LEADERBOARD[:20] if e['config']]
    for weights, base_n in top_for_bets:
        if time.time() > deadline: break
        for bet_size in [3.0, 4.0, 5.0, 6.0, 8.0]:
            label = f"BET=${bet_size:.0f} N={base_n} {wlabel(weights)}"
            test_strategy(all_data, files, label, weights, top_n=base_n,
                          bet_per_num=bet_size)

    print(f"  Phase 9 complete: {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 9 — PROGRESSIVE BETTING", 20, 5)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 10: FINAL CROSS-VALIDATION — 60 min
    # ═══════════════════════════════════════════════════════════════════
    phase_header("FINAL CROSS-VALIDATION & STRESS TEST", 60)
    deadline = phase_deadline(60)

    # Cross-validate top 30 with full per-file breakdown
    for e in LEADERBOARD[:30]:
        if time.time() > deadline: break
        if e['config'] and not e['label'].startswith('[FINAL]'):
            cfg = e['config']
            w = cfg['w']
            n = cfg['n']
            gt = cfg.get('gate_t', 0)
            trig = cfg.get('trigger', None)
            bet = cfg.get('bet', 2.0)
            gfn = None
            if gt > 0:
                if 'pol>=' in e['label']: gfn = sig_polarity
                elif 'tab>=' in e['label']: gfn = sig_table
                elif 'comb>=' in e['label']: gfn = sig_combined
                elif 'set>=' in e['label']: gfn = sig_set
                elif 'any>=' in e['label']: gfn = sig_any_strong

            label = f"[FINAL] N={n} {wlabel(w)}"
            if gt > 0: label += f"|gate>={gt}"
            if trig: label += f"|{trig}"
            if bet != 2.0: label += f" @${bet:.0f}"
            test_strategy(all_data, files, label, w, top_n=n,
                          gate_t=gt, gate_fn=gfn, trigger=trig,
                          bet_per_num=bet, do_cross=True)

    print(f"  Phase 10 complete: {TESTED_COUNT} tested")
    print_leaderboard("AFTER PHASE 10 — FINAL VALIDATION", 25, 3)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 11: RANDOM WALK (remaining time)
    # ═══════════════════════════════════════════════════════════════════
    remaining = remaining_min() - 30  # reserve 30 min for report
    if remaining > 0:
        phase_header(f"RANDOM WALK PERTURBATIONS (remaining ~{int(remaining)}min)", int(remaining))
        deadline = phase_deadline(remaining)
        walk_count = 0

        while time.time() < deadline:
            # Strategy: randomly pick approach
            approach = random.choice(['mutate', 'crossover', 'random', 'gate_mutate', 'n_sweep'])

            if approach == 'mutate' and LEADERBOARD:
                parent = random.choice(LEADERBOARD[:30])
                if parent['config']:
                    w = mutate_weights(parent['config']['w'], random.uniform(0.02, 0.25))
                    n = parent['config']['n'] + random.choice([-1, 0, 0, 1])
                    n = max(10, min(16, n))
                    label = f"WALK N={n} {wlabel(w)}"
                    test_strategy(all_data, files, label, w, top_n=n)

            elif approach == 'crossover' and len(LEADERBOARD) >= 2:
                p1, p2 = random.sample(LEADERBOARD[:20], 2)
                if p1['config'] and p2['config']:
                    w = crossover_weights(p1['config']['w'], p2['config']['w'])
                    n = random.choice([p1['config']['n'], p2['config']['n']])
                    label = f"CROSS N={n} {wlabel(w)}"
                    test_strategy(all_data, files, label, w, top_n=n)

            elif approach == 'random':
                w = random_weights(all_model_keys, random.choice([2, 3, 4]))
                n = random.choice([10, 11, 12, 13, 14, 15, 16])
                label = f"RAND N={n} {wlabel(w)}"
                test_strategy(all_data, files, label, w, top_n=n)

            elif approach == 'gate_mutate' and LEADERBOARD:
                parent = random.choice(LEADERBOARD[:20])
                if parent['config']:
                    w = parent['config']['w']
                    n = parent['config']['n']
                    gfn, gname = random.choice(gates)
                    gt = random.choice([5, 8, 10, 12, 15, 18, 20, 25, 30])
                    label = f"GWALK N={n} {wlabel(w)}|{gname}>={gt}"
                    test_strategy(all_data, files, label, w, top_n=n,
                                  gate_t=gt, gate_fn=gfn)

            elif approach == 'n_sweep' and LEADERBOARD:
                parent = random.choice(LEADERBOARD[:15])
                if parent['config']:
                    w = parent['config']['w']
                    for n in range(10, 17):
                        label = f"NSWEEP N={n} {wlabel(w)}"
                        test_strategy(all_data, files, label, w, top_n=n)

            walk_count += 1
            if walk_count % 2000 == 0:
                print(f"  ... {walk_count} random walks, {TESTED_COUNT} total, "
                      f"{int(elapsed_min())}min elapsed")
                sys.stdout.flush()

        print(f"  Phase 11 complete: {walk_count} walks, {TESTED_COUNT} total tested")
        print_leaderboard("AFTER PHASE 11 — RANDOM WALK", 25, 3)

    # ═══════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print(f"{'='*130}")
    print(f"  MARATHON V3 FINAL REPORT")
    print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Runtime: {elapsed_min():.0f} minutes ({elapsed_min()/60:.1f} hours)")
    print(f"  Total strategies tested: {TESTED_COUNT}")
    print(f"  Models used: {len(ALL_MODELS)} ({len(new_models)} new in V3)")
    print(f"{'='*130}")
    print(f"{'='*130}")

    # Final leaderboard
    print_leaderboard("FINAL LEADERBOARD — TOP 30", 30, 3)

    # Show best strategies with full cross-validation
    final_entries = [e for e in LEADERBOARD if e['label'].startswith('[FINAL]') and e['cross']]
    final_entries.sort(key=lambda x: -x['score'])

    if final_entries:
        print(f"\n  {'='*130}")
        print(f"  CROSS-VALIDATED TOP STRATEGIES (tested on each file individually)")
        print(f"  {'='*130}")
        for i, e in enumerate(final_entries[:10]):
            r = e['result']
            print(f"\n  #{i+1} {e['label']}")
            print(f"      Overall:  avg=${r['avg_profit']:+.0f}/session | win={r['win_pct']:.0f}% | "
                  f"+$100={r['plus100_pct']:.0f}% | sessions={r['sessions']} | score={e['score']:.1f}")
            if e['cross']:
                profitable = 0
                total_f = 0
                for j, (fname, _) in enumerate(files):
                    cr = e['cross'][j]
                    if cr:
                        total_f += 1
                        if cr['avg_profit'] > 0: profitable += 1
                        flag = "+" if cr['avg_profit'] > 0 else "-"
                        print(f"      {fname}: avg=${cr['avg_profit']:+.0f} win={cr['win_pct']:.0f}% "
                              f"sess={cr['sessions']} {flag}")
                    else:
                        print(f"      {fname}: N/A")
                print(f"      Robustness: {profitable}/{total_f} files profitable "
                      f"({profitable/total_f*100:.0f}%)")

    # Best by different criteria
    print(f"\n  {'='*130}")
    print(f"  BEST BY CATEGORY")
    print(f"  {'='*130}")

    active = [e for e in LEADERBOARD if e['result']['avg_bets'] >= 20]
    if active:
        best_active = max(active, key=lambda x: x['result']['avg_profit'])
        r = best_active['result']
        print(f"  Best Active (20+ bets/session): {best_active['label']}")
        print(f"    avg=${r['avg_profit']:+.0f} win={r['win_pct']:.0f}% bets={r['avg_bets']:.0f}")

    high_wr = [e for e in LEADERBOARD if e['result']['win_pct'] >= 55 and e['result']['avg_bets'] >= 15]
    if high_wr:
        best_wr = max(high_wr, key=lambda x: x['result']['avg_profit'])
        r = best_wr['result']
        print(f"  Best High-WinRate (55%+ win, 15+ bets): {best_wr['label']}")
        print(f"    avg=${r['avg_profit']:+.0f} win={r['win_pct']:.0f}% bets={r['avg_bets']:.0f}")

    consistent = [e for e in LEADERBOARD if e['result']['min_profit'] > -400 and e['result']['avg_bets'] >= 20]
    if consistent:
        best_cons = max(consistent, key=lambda x: x['result']['avg_profit'])
        r = best_cons['result']
        print(f"  Best Consistent (min>-$400, 20+ bets): {best_cons['label']}")
        print(f"    avg=${r['avg_profit']:+.0f} win={r['win_pct']:.0f}% min=${r['min_profit']:+.0f}")

    plus100 = [e for e in LEADERBOARD if e['result']['plus100_pct'] >= 30 and e['result']['avg_bets'] >= 15]
    if plus100:
        best_100 = max(plus100, key=lambda x: x['result']['plus100_pct'])
        r = best_100['result']
        print(f"  Best +$100 Rate (30%+ sessions): {best_100['label']}")
        print(f"    +$100={r['plus100_pct']:.0f}% avg=${r['avg_profit']:+.0f} win={r['win_pct']:.0f}%")

    print(f"\n  {'='*130}")
    print(f"  V3 MARATHON COMPLETE — {TESTED_COUNT} strategies tested in {elapsed_min()/60:.1f} hours")
    print(f"  {'='*130}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
