#!/usr/bin/env python3
"""
SESSION SIMULATOR — Tests strategies in REALISTIC 30-spin sessions.

Key difference from marathon: instead of one long backtest across 3000+ spins,
this simulates actual casino sessions:
  - Each session = 30 spins (occasionally 60)
  - Bankroll starts at $4000 each session
  - AI uses warmup from previous sessions (cumulative learning)
  - Measures: profit per session, win rate per session, consistency

This answers: "Can I make +$100 per session with 30 spins?"
"""

import sys
import os
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
POS_NUMS = sorted(WHEEL_POSITIVE)
NEG_NUMS = sorted(WHEEL_NEGATIVE)
TABLE_0_NUMS = sorted(WHEEL_TABLE_0)
TABLE_19_NUMS = sorted(WHEEL_TABLE_19)
RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}


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
# SAME MODELS AS MARATHON V2
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

ALL_MODELS = {
    'freq': m_freq, 'polarity': m_polarity,
    'table': m_table, 'tab_contra': m_table_contrarian,
    'set': m_set, 'hot': m_hot, 'sector': m_sector,
    'gap': m_gap, 'markov': m_markov,
    'odd_even': m_odd_even, 'high_low': m_high_low,
    'pattern3': m_pattern3,
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


# ═══════════════════════════════════════════════════════════════════════════
# SESSION-BASED BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def session_backtest(all_data, weights, top_n=12, session_size=30,
                     warmup=80, gate_threshold=0, gate_fn=None, trigger=None):
    """
    Simulates playing in sessions of `session_size` spins.
    History accumulates across sessions (AI learns from all past spins).
    Returns per-session results.
    """
    if len(all_data) < warmup + session_size:
        return None

    sessions = []
    total_spins_used = 0

    # Start after warmup
    spin_idx = warmup

    while spin_idx + session_size <= len(all_data):
        # This session's spins
        session_spins = all_data[spin_idx:spin_idx + session_size]
        session_hits = 0
        session_bets = 0
        session_skipped = 0
        session_profit = 0.0

        for j, actual in enumerate(session_spins):
            history = all_data[:spin_idx + j]  # all history up to this point

            # Gate check
            if gate_threshold > 0 and gate_fn:
                if gate_fn(history) < gate_threshold:
                    session_skipped += 1
                    continue

            # Trigger check
            restricted = None
            idx = spin_idx + j
            if trigger == 'pol_double_follow':
                if idx < 2: session_skipped += 1; continue
                if POL_MAP[all_data[idx-1]] != POL_MAP[all_data[idx-2]]: session_skipped += 1; continue
                restricted = POS_NUMS if POL_MAP[all_data[idx-1]] == 'pos' else NEG_NUMS
            elif trigger == 'alternating':
                if idx < 2: session_skipped += 1; continue
                if POL_MAP[all_data[idx-1]] == POL_MAP[all_data[idx-2]]: session_skipped += 1; continue
                restricted = POS_NUMS if POL_MAP[all_data[idx-1]] == 'pos' else NEG_NUMS

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
            cost = top_n * BET_PER_NUM
            if hit:
                session_profit += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
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

    # Aggregate
    total_sessions = len(sessions)
    winning_sessions = sum(1 for s in sessions if s['profit'] > 0)
    plus100_sessions = sum(1 for s in sessions if s['profit'] >= 100)
    avg_profit = np.mean([s['profit'] for s in sessions])
    avg_hit_rate = np.mean([s['hit_rate'] for s in sessions])
    avg_bets = np.mean([s['bets'] for s in sessions])
    min_profit = min(s['profit'] for s in sessions)
    max_profit = max(s['profit'] for s in sessions)
    total_profit = sum(s['profit'] for s in sessions)

    return {
        'sessions': total_sessions,
        'winning': winning_sessions,
        'win_pct': round(winning_sessions / total_sessions * 100, 1),
        'plus100': plus100_sessions,
        'plus100_pct': round(plus100_sessions / total_sessions * 100, 1),
        'avg_profit': round(avg_profit, 2),
        'avg_hit_rate': round(avg_hit_rate, 2),
        'avg_bets': round(avg_bets, 1),
        'min_profit': round(min_profit, 2),
        'max_profit': round(max_profit, 2),
        'total_profit': round(total_profit, 2),
        'per_session': sessions,
    }


def wlabel(w):
    parts = []
    for k, v in sorted(w.items(), key=lambda x: -x[1]):
        if v > 0:
            parts.append(f"{k}({int(v*100)}%)")
    return "+".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Test top strategies from marathon in 30-spin sessions
# ═══════════════════════════════════════════════════════════════════════════

def main():
    all_data, files = load_all_data()
    print("=" * 120)
    print("SESSION SIMULATOR — Realistic 30-spin session testing")
    print("=" * 120)
    print(f"\nLoaded {len(all_data)} spins from {len(files)} files")
    print(f"Session size: 30 spins | Bankroll: $4000 | Bet: ${BET_PER_NUM}/number")
    print(f"Break-even N=12: 33.33% | Need ~35-36% for +$100/session")
    print(f"In 30 spins with N=12: need 11+ hits out of 30 for profit (36.67%)")
    print(f"In 30 spins with N=12: need 12+ hits for +$100 profit")

    # Top strategies from marathon v2 to test
    strategies = [
        # From Phase 1 — best 2-model pairs (bet every spin)
        ("table(82%)+odd_even(18%)", {'table': 0.82, 'odd_even': 0.18}, 12, None, 0, None),
        ("set(69%)+sector(31%)", {'set': 0.69, 'sector': 0.31}, 12, None, 0, None),
        ("set(20%)+odd_even(80%)", {'set': 0.20, 'odd_even': 0.80}, 12, None, 0, None),
        ("markov(30%)+pattern3(70%)", {'markov': 0.30, 'pattern3': 0.70}, 12, None, 0, None),
        ("freq(92%)+sector(8%)", {'freq': 0.92, 'sector': 0.08}, 12, None, 0, None),
        ("freq(80%)+polarity(20%)", {'freq': 0.80, 'polarity': 0.20}, 12, None, 0, None),
        ("tab_contra(65%)+high_low(35%)", {'tab_contra': 0.65, 'high_low': 0.35}, 12, None, 0, None),

        # From Phase 3 — N=16 (more numbers, lower break-even)
        ("N=16 high_low(60%)+set(30%)+tab_contra(10%)", {'high_low': 0.60, 'set': 0.30, 'tab_contra': 0.10}, 16, None, 0, None),

        # From Phase 5 — with gates (selective betting)
        ("N=14 set(65%)+sector(30%)+odd_even(5%)|tab>=25", {'set': 0.65, 'sector': 0.30, 'odd_even': 0.05}, 14, None, 25, sig_table),
        ("N=14 set(65%)+sector(30%)+odd_even(5%)|tab>=15", {'set': 0.65, 'sector': 0.30, 'odd_even': 0.05}, 14, None, 15, sig_table),
        ("N=14 set(65%)+sector(30%)+odd_even(5%)|tab>=10", {'set': 0.65, 'sector': 0.30, 'odd_even': 0.05}, 14, None, 10, sig_table),

        # From Phase 6 — gate + trigger
        ("N=13 set(50%)+sector(45%)+odd_even(5%)|alt|pol>=25", {'set': 0.50, 'sector': 0.45, 'odd_even': 0.05}, 13, 'alternating', 25, sig_polarity),

        # Different N values with best weights
        ("N=13 table(82%)+odd_even(18%)", {'table': 0.82, 'odd_even': 0.18}, 13, None, 0, None),
        ("N=14 table(82%)+odd_even(18%)", {'table': 0.82, 'odd_even': 0.18}, 14, None, 0, None),
        ("N=13 markov(30%)+pattern3(70%)", {'markov': 0.30, 'pattern3': 0.70}, 13, None, 0, None),
        ("N=14 markov(30%)+pattern3(70%)", {'markov': 0.30, 'pattern3': 0.70}, 14, None, 0, None),

        # Baseline — random / freq only
        ("BASELINE freq(100%)", {'freq': 1.0}, 12, None, 0, None),
    ]

    # Test each strategy in 30-spin sessions
    for session_size in [30, 60]:
        print(f"\n{'='*120}")
        print(f"SESSION SIZE: {session_size} spins per session")
        print(f"{'='*120}")
        print(f"\n  {'Strategy':<60} {'Sess':>4} {'Win%':>5} {'+$100':>5} {'AvgP':>7} {'AvgHR':>6} {'Bets':>4} {'MinP':>7} {'MaxP':>7} {'TotP':>8}")
        print(f"  {'-'*115}")

        results = []
        for name, weights, top_n, trigger, gate_t, gate_fn in strategies:
            r = session_backtest(all_data, weights, top_n, session_size,
                               warmup=80, gate_threshold=gate_t, gate_fn=gate_fn,
                               trigger=trigger)
            if r:
                results.append((name, r))
                star = " ***" if r['avg_profit'] >= 100 else " **" if r['avg_profit'] > 0 else ""
                print(f"  {name:<60} {r['sessions']:>4} {r['win_pct']:>4.0f}% {r['plus100_pct']:>4.0f}% ${r['avg_profit']:>+6.0f} {r['avg_hit_rate']:>5.1f}% {r['avg_bets']:>4.0f} ${r['min_profit']:>+6.0f} ${r['max_profit']:>+6.0f} ${r['total_profit']:>+7.0f}{star}")
            else:
                print(f"  {name:<60} N/A")

        # Show per-session breakdown for top 3
        results.sort(key=lambda x: -x[1]['avg_profit'])
        print(f"\n  {'─'*80}")
        print(f"  TOP 3 PER-SESSION BREAKDOWN (session_size={session_size}):")
        for name, r in results[:3]:
            print(f"\n  {name}:")
            print(f"    Sessions: {r['sessions']} | Win: {r['win_pct']}% | +$100: {r['plus100_pct']}%")
            print(f"    Avg profit: ${r['avg_profit']:+.0f} | Total: ${r['total_profit']:+.0f}")
            print(f"    Session-by-session: ", end='')
            for i, s in enumerate(r['per_session']):
                mark = "+" if s['profit'] > 0 else "-"
                print(f"${s['profit']:+.0f}({s['hit_rate']:.0f}%){mark}", end=' ')
                if (i+1) % 10 == 0:
                    print(f"\n{'':>23}", end='')
            print()

    # Per-file testing for best strategy
    print(f"\n{'='*120}")
    print("PER-FILE SESSION TESTS (30 spins per session)")
    print(f"{'='*120}")

    # Test top 5 strategies on each file individually
    results.sort(key=lambda x: -x[1]['avg_profit'])
    top5 = [(name, w, n, t, gt, gf) for name, w, n, t, gt, gf in strategies
            if any(name == r[0] for r in results[:5])]

    for fname, fdata in files:
        print(f"\n  === {fname} ({len(fdata)} spins) ===")
        for name, weights, top_n, trigger, gate_t, gate_fn in strategies[:7]:
            r = session_backtest(fdata, weights, top_n, 30,
                               warmup=min(60, len(fdata)//5),
                               gate_threshold=gate_t, gate_fn=gate_fn,
                               trigger=trigger)
            if r and r['sessions'] > 0:
                print(f"    {name:<55} sess={r['sessions']} win={r['win_pct']}% avg=${r['avg_profit']:+.0f} tot=${r['total_profit']:+.0f}")

    # Final summary
    print(f"\n{'='*120}")
    print("SUMMARY — Best strategies for 30-spin sessions")
    print(f"{'='*120}")
    print(f"\n  Target: +$100 per session consistently")
    print(f"  With N=12, need 12+ hits in 30 spins (40%) for +$100")
    print(f"  With N=13, need 12+ hits in 30 spins (40%) for +$100")
    print(f"  With N=14, need 14+ hits in 30 spins (46.7%) for +$100")
    print(f"\n  Reality check:")
    print(f"  - Random chance N=12: 32.43% → avg ${-24*30*(1-12/37) + 48*30*12/37:.0f} per 30 spins")
    print(f"  - Random chance N=13: 35.14% → need slight edge to break even")
    print(f"  - Random chance N=14: 37.84% → break even at 38.89%")

    results.sort(key=lambda x: -x[1]['avg_profit'])
    print(f"\n  RECOMMENDED (by avg profit per 30-spin session):")
    for i, (name, r) in enumerate(results[:5]):
        print(f"    {i+1}. {name}")
        print(f"       Avg: ${r['avg_profit']:+.0f}/session | Win: {r['win_pct']}% | +$100: {r['plus100_pct']}% | Bets/session: {r['avg_bets']:.0f}")
    print()


if __name__ == '__main__':
    main()
