#!/usr/bin/env python3
"""
Focused Optimization — Drill into the 3 biggest discoveries:

1. Hot Number Boost (+10% weight) → +1.34% edge, $1188 profit
2. Conditional Betting (thresh=25-30) → +2.11% to +3.48% edge
3. Anchor(2-4)+Nb(1) selection → +1.43% edge, $1404 profit
4. Picks=10 → better profit ($396 vs $-108)

This script tests combinations of these improvements together.
Also tests hot number model integration into the real ensemble.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import Counter

from config import (
    TOTAL_NUMBERS, WHEEL_ORDER, NUMBER_TO_POSITION,
    WHEEL_TABLE_0, WHEEL_TABLE_19,
    WHEEL_POSITIVE, WHEEL_NEGATIVE,
    WHEEL_SET_1, WHEEL_SET_2, WHEEL_SET_3,
)

# ─── Load all data ──────────────────────────────────────────────────────
def load_all_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'userdata')
    all_spins = []
    datasets = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.txt'):
            fpath = os.path.join(data_dir, fname)
            spins = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        n = int(line)
                        if 0 <= n <= 36:
                            spins.append(n)
            if spins:
                datasets.append((fname, spins))
                all_spins.extend(spins)
    return all_spins, datasets

# ─── Models ─────────────────────────────────────────────────────────────
_table_map = {}
_polarity_map = {}
_set_map = {}
for _n in range(TOTAL_NUMBERS):
    _table_map[_n] = '0' if _n in WHEEL_TABLE_0 else '19'
    _polarity_map[_n] = 'positive' if _n in WHEEL_POSITIVE else 'negative'
    if _n in WHEEL_SET_1: _set_map[_n] = 1
    elif _n in WHEEL_SET_2: _set_map[_n] = 2
    else: _set_map[_n] = 3


def frequency_probs(history, recent_window=30, flat_weight=0.5, decay=0.998):
    n = len(history)
    if n == 0:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    counts = np.zeros(TOTAL_NUMBERS)
    for i, num in enumerate(history):
        age = n - 1 - i
        counts[num] += decay ** age
    total = counts.sum()
    overall = counts / total if total > 0 else np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    recent = history[-recent_window:] if recent_window > 0 else history
    recent_counts = np.zeros(TOTAL_NUMBERS)
    for num in recent:
        recent_counts[num] += 1
    recent_total = recent_counts.sum()
    recent_dist = recent_counts / recent_total if recent_total > 0 else np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    blended = flat_weight * overall + (1.0 - flat_weight) * recent_dist
    blended /= blended.sum()
    return blended


def wheel_strategy_probs(history, window=50, boost=1.30, dampen=0.70):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if _table_map.get(n) == '0')
    t19 = total - t0
    t0_pct, t19_pct = t0/total, t19/total
    hot_table = '0' if t0 > t19 else '19' if t19 > t0 else None
    pos = sum(1 for n in recent if _polarity_map.get(n) == 'positive')
    neg = total - pos
    pos_pct, neg_pct = pos/total, neg/total
    hot_pol = 'positive' if pos > neg else 'negative' if neg > pos else None
    set_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        set_counts[_set_map.get(n, 3)] += 1
    hot_set_num = max(set_counts, key=set_counts.get)
    expected = total / 3
    hot_set = hot_set_num if set_counts[hot_set_num] > expected * 1.1 else None
    for num in range(TOTAL_NUMBERS):
        multiplier = 1.0
        if hot_table is not None:
            imbalance = abs(t0_pct - t19_pct) / 0.5
            tb = 1.0 + (boost - 1.0) * imbalance
            td = 1.0 - (1.0 - dampen) * imbalance
            multiplier *= tb if _table_map.get(num) == hot_table else td
        if hot_pol is not None:
            imbalance = abs(pos_pct - neg_pct) / 0.5
            pb = 1.0 + (boost - 1.0) * imbalance
            pd = 1.0 - (1.0 - dampen) * imbalance
            multiplier *= pb if _polarity_map.get(num) == hot_pol else pd
        if hot_set is not None:
            set_pcts = {s: set_counts[s]/total for s in [1,2,3]}
            hot_excess = max(0, min(1, (set_pcts[hot_set]-1/3)/(1/3)))
            sb = 1.0 + (boost - 1.0) * hot_excess
            sd = 1.0 - (1.0 - dampen) * hot_excess
            multiplier *= sb if _set_map.get(num, 3) == hot_set else sd
        probs[num] *= multiplier
    probs /= probs.sum()
    return probs


def pattern_probs(history):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10:
        return probs
    last_seen = {}
    for i, n in enumerate(history):
        last_seen[n] = i
    max_gap = len(history)
    for num in range(TOTAL_NUMBERS):
        if num in last_seen:
            gap = len(history) - 1 - last_seen[num]
            gap_factor = 1.0 + min(0.3, gap / max_gap)
            probs[num] *= gap_factor
        else:
            probs[num] *= 1.3
    probs /= probs.sum()
    return probs


def hot_number_probs(history, window=15, boost_factor=1.5):
    """Boost numbers that appeared frequently in recent window."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window:
        return probs
    recent = history[-window:]
    counts = Counter(recent)
    expected = window / TOTAL_NUMBERS
    for num in range(TOTAL_NUMBERS):
        c = counts.get(num, 0)
        if c > expected * 1.5:
            probs[num] *= boost_factor
        elif c == 0:
            probs[num] *= 0.8
    probs /= probs.sum()
    return probs


def wheel_signal_strength(history, window=50):
    if len(history) < 5:
        return 0.0
    recent = history[-window:]
    total = len(recent)
    t0 = sum(1 for n in recent if _table_map.get(n) == '0')
    table_dev = abs(t0/total - 19/37)
    table_score = min(100, table_dev / 0.25 * 100)
    pos = sum(1 for n in recent if _polarity_map.get(n) == 'positive')
    pol_dev = abs(pos/total - 19/37)
    pol_score = min(100, pol_dev / 0.25 * 100)
    set_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        set_counts[_set_map.get(n, 3)] += 1
    set_pcts = [set_counts[s]/total for s in [1,2,3]]
    set_dev = max(abs(p-1/3) for p in set_pcts)
    set_score = min(100, set_dev / 0.20 * 100)
    return (table_score + pol_score + set_score) / 3


def ensemble_blend(history, freq_w=0.611, wheel_w=0.333, pattern_w=0.056, hot_w=0.0,
                   freq_window=30, freq_flat=0.5,
                   wheel_window=50, wheel_boost=1.30, wheel_dampen=0.70,
                   hot_window=15, hot_boost=1.5):
    fp = frequency_probs(history, recent_window=freq_window, flat_weight=freq_flat)
    wp = wheel_strategy_probs(history, window=wheel_window, boost=wheel_boost, dampen=wheel_dampen)
    pp = pattern_probs(history)
    hp = hot_number_probs(history, window=hot_window, boost_factor=hot_boost)
    ensemble = fp * freq_w + wp * wheel_w + pp * pattern_w + hp * hot_w
    total = ensemble.sum()
    if total > 0:
        ensemble /= total
    return ensemble


def anchor_neighbour_select(probs, max_picks=12, neighbours=1):
    sorted_idx = np.argsort(probs)[::-1]
    selected = set()
    for anchor in sorted_idx:
        if len(selected) >= max_picks:
            break
        anchor = int(anchor)
        if anchor in selected:
            continue
        pos = NUMBER_TO_POSITION.get(anchor, -1)
        if pos < 0:
            continue
        already = False
        for s in selected:
            s_pos = NUMBER_TO_POSITION.get(s, -1)
            if s_pos >= 0:
                dist = min(abs(pos - s_pos), len(WHEEL_ORDER) - abs(pos - s_pos))
                if dist <= neighbours:
                    already = True
                    break
        if already:
            continue
        selected.add(anchor)
        for d in [1, -1]:
            for r in range(1, neighbours + 1):
                if len(selected) >= max_picks:
                    break
                nb_pos = (pos + d * r) % len(WHEEL_ORDER)
                selected.add(WHEEL_ORDER[nb_pos])
    return list(selected)[:max_picks]


# ─── Backtest Engine ────────────────────────────────────────────────────
def backtest(all_spins, prob_func, num_picks=12, warmup=50, label="",
             use_anchor=False, anchor_nb=1, conditional_func=None, cond_thresh=0):
    hits = 0
    total_bets = 0
    skipped = 0
    profit = 0.0
    bet_cost = 3.0
    payout = 35

    for i in range(warmup, len(all_spins)):
        history = all_spins[:i]
        actual = all_spins[i]

        if conditional_func:
            strength = conditional_func(history)
            if strength < cond_thresh:
                skipped += 1
                continue

        probs = prob_func(history)

        if use_anchor:
            selected = anchor_neighbour_select(probs, max_picks=num_picks, neighbours=anchor_nb)
        else:
            selected = list(np.argsort(probs)[::-1][:num_picks])

        total_bets += 1
        cost = bet_cost * len(selected)
        profit -= cost

        if actual in selected:
            hits += 1
            profit += bet_cost * (payout + 1)

    if total_bets == 0:
        return {'label': label, 'hit_rate': 0, 'edge': 0, 'profit': 0, 'bets': 0, 'skipped': skipped}

    hit_rate = hits / total_bets * 100
    expected_random = num_picks / TOTAL_NUMBERS * 100
    edge = hit_rate - expected_random

    return {
        'label': label,
        'hit_rate': round(hit_rate, 2),
        'edge': round(edge, 2),
        'profit': round(profit, 2),
        'bets': total_bets,
        'hits': hits,
        'skipped': skipped,
        'bet_pct': round(total_bets / (total_bets + skipped) * 100, 1) if (total_bets + skipped) > 0 else 100,
    }


def main():
    all_spins, datasets = load_all_data()
    print(f"Total spins: {len(all_spins)} from {len(datasets)} files\n")

    # ─── BASELINE ───────────────────────────────────────────────────────
    print("=" * 80)
    print("BASELINE (current config)")
    print("=" * 80)
    baseline = backtest(all_spins,
        lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056, hot_w=0),
        num_picks=12, label="Current (12 picks, pure top-N)")
    print(f"  {baseline['label']}: hit={baseline['hit_rate']}% edge={baseline['edge']}% profit=${baseline['profit']}")
    print()

    # ─── IMPROVEMENT 1: Add Hot Number Model ────────────────────────────
    print("=" * 80)
    print("IMPROVEMENT 1: ADD HOT NUMBER MODEL")
    print("=" * 80)

    # Fine-tune hot number parameters
    best = None
    for hot_w in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for hot_win in [10, 15, 20, 25]:
            for hot_boost in [1.3, 1.5, 1.8, 2.0]:
                # Rescale other weights proportionally
                remaining = 1.0 - hot_w
                fw = 0.611 * remaining / (0.611 + 0.333 + 0.056)
                ww = 0.333 * remaining / (0.611 + 0.333 + 0.056)
                pw = 0.056 * remaining / (0.611 + 0.333 + 0.056)

                result = backtest(all_spins,
                    lambda h, f=fw, w=ww, p=pw, hw=hot_w, hwi=hot_win, hb=hot_boost:
                        ensemble_blend(h, freq_w=f, wheel_w=w, pattern_w=p, hot_w=hw,
                                      hot_window=hwi, hot_boost=hb),
                    num_picks=12,
                    label=f"HotNum(w={hot_w},win={hot_win},b={hot_boost})")
                if best is None or result['edge'] > best['edge']:
                    best = result
    print(f"  BEST Hot Number: {best['label']}: hit={best['hit_rate']}% edge={best['edge']}% profit=${best['profit']}")
    best_hot = best
    print()

    # ─── IMPROVEMENT 2: Anchor+Neighbour Selection ──────────────────────
    print("=" * 80)
    print("IMPROVEMENT 2: ANCHOR+NEIGHBOUR SELECTION")
    print("=" * 80)

    best_an = None
    for nb in [1, 2]:
        for picks in [9, 10, 11, 12]:
            result = backtest(all_spins,
                lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056),
                num_picks=picks, use_anchor=True, anchor_nb=nb,
                label=f"Anchor+Nb({nb}), picks={picks}")
            marker = ""
            if best_an is None or result['edge'] > best_an['edge']:
                best_an = result
                marker = " *** BEST"
            print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  BEST Anchor: {best_an['label']}")
    print()

    # ─── IMPROVEMENT 3: Conditional Betting ─────────────────────────────
    print("=" * 80)
    print("IMPROVEMENT 3: CONDITIONAL BETTING FINE-TUNING")
    print("=" * 80)

    best_cond = None
    for thresh in [20, 22, 25, 27, 30, 33, 35]:
        result = backtest(all_spins,
            lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056),
            num_picks=12,
            conditional_func=lambda h: wheel_signal_strength(h, window=50),
            cond_thresh=thresh,
            label=f"Conditional(thresh={thresh})")
        marker = ""
        if best_cond is None or result['edge'] > best_cond['edge']:
            best_cond = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% bet_pct={result['bet_pct']}% profit=${result['profit']}{marker}")
    print(f"  BEST Conditional: {best_cond['label']}")
    print()

    # ─── IMPROVEMENT 4: Picks Count ─────────────────────────────────────
    print("=" * 80)
    print("IMPROVEMENT 4: OPTIMAL NUMBER COUNT")
    print("=" * 80)

    best_picks = None
    for picks in [8, 9, 10, 11, 12]:
        result = backtest(all_spins,
            lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056),
            num_picks=picks,
            label=f"Picks={picks}")
        marker = ""
        if best_picks is None or result['profit'] > best_picks['profit']:
            best_picks = result
            marker = " *** BEST PROFIT"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print()

    # ─── COMBINED: All improvements together ────────────────────────────
    print("=" * 80)
    print("COMBINED IMPROVEMENTS (testing all together)")
    print("=" * 80)

    # Parse best hot params
    # Extract from best_hot label
    import re
    hot_match = re.search(r'w=([\d.]+),win=(\d+),b=([\d.]+)', best_hot['label'])
    if hot_match:
        best_hot_w = float(hot_match.group(1))
        best_hot_win = int(hot_match.group(2))
        best_hot_boost = float(hot_match.group(3))
    else:
        best_hot_w, best_hot_win, best_hot_boost = 0.10, 15, 1.5

    remaining = 1.0 - best_hot_w
    fw = 0.611 * remaining / (0.611 + 0.333 + 0.056)
    ww = 0.333 * remaining / (0.611 + 0.333 + 0.056)
    pw = 0.056 * remaining / (0.611 + 0.333 + 0.056)

    combos = [
        # (label, freq_w, wheel_w, pat_w, hot_w, picks, use_anchor, anchor_nb, cond_func, cond_thresh)
        ("HotNum Only",
         fw, ww, pw, best_hot_w, 12, False, 0, None, 0),
        ("HotNum + Anchor(1)",
         fw, ww, pw, best_hot_w, 12, True, 1, None, 0),
        ("HotNum + Anchor(1) + Picks=10",
         fw, ww, pw, best_hot_w, 10, True, 1, None, 0),
        ("HotNum + Conditional(25)",
         fw, ww, pw, best_hot_w, 12, False, 0,
         lambda h: wheel_signal_strength(h, 50), 25),
        ("HotNum + Conditional(30)",
         fw, ww, pw, best_hot_w, 12, False, 0,
         lambda h: wheel_signal_strength(h, 50), 30),
        ("HotNum + Anchor(1) + Cond(25)",
         fw, ww, pw, best_hot_w, 12, True, 1,
         lambda h: wheel_signal_strength(h, 50), 25),
        ("HotNum + Anchor(1) + Cond(30)",
         fw, ww, pw, best_hot_w, 12, True, 1,
         lambda h: wheel_signal_strength(h, 50), 30),
        ("HotNum + Anchor(1) + Cond(25) + Picks=10",
         fw, ww, pw, best_hot_w, 10, True, 1,
         lambda h: wheel_signal_strength(h, 50), 25),
        ("HotNum + Anchor(1) + Cond(30) + Picks=10",
         fw, ww, pw, best_hot_w, 10, True, 1,
         lambda h: wheel_signal_strength(h, 50), 30),
        # Without hot num but with anchor + conditional
        ("NoHot + Anchor(1) + Cond(25)",
         0.611, 0.333, 0.056, 0, 12, True, 1,
         lambda h: wheel_signal_strength(h, 50), 25),
        ("NoHot + Anchor(1) + Cond(30)",
         0.611, 0.333, 0.056, 0, 12, True, 1,
         lambda h: wheel_signal_strength(h, 50), 30),
        ("NoHot + Anchor(1)",
         0.611, 0.333, 0.056, 0, 12, True, 1, None, 0),
        ("NoHot + Cond(25)",
         0.611, 0.333, 0.056, 0, 12, False, 0,
         lambda h: wheel_signal_strength(h, 50), 25),
        ("NoHot + Cond(30)",
         0.611, 0.333, 0.056, 0, 12, False, 0,
         lambda h: wheel_signal_strength(h, 50), 30),
    ]

    best_combined = None
    for label, f, w, p, hw, picks, use_an, an_nb, cf, ct in combos:
        result = backtest(all_spins,
            lambda h, f2=f, w2=w, p2=p, hw2=hw:
                ensemble_blend(h, freq_w=f2, wheel_w=w2, pattern_w=p2, hot_w=hw2,
                              hot_window=best_hot_win, hot_boost=best_hot_boost),
            num_picks=picks,
            use_anchor=use_an, anchor_nb=an_nb,
            conditional_func=cf, cond_thresh=ct,
            label=label)
        marker = ""
        if best_combined is None or result['profit'] > best_combined['profit']:
            best_combined = result
            marker = " *** BEST PROFIT"
        bp = result.get('bet_pct', 100)
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% bet={bp}% profit=${result['profit']}{marker}")
    print()

    # ─── Per-dataset validation ─────────────────────────────────────────
    print("=" * 80)
    print("PER-DATASET VALIDATION OF BEST STRATEGIES")
    print("=" * 80)

    strategies_to_test = [
        ("Current Config", lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056),
         12, False, 0, None, 0),
        ("+ HotNum", lambda h: ensemble_blend(h, freq_w=fw, wheel_w=ww, pattern_w=pw, hot_w=best_hot_w,
                                              hot_window=best_hot_win, hot_boost=best_hot_boost),
         12, False, 0, None, 0),
        ("+ Anchor(1)", lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056),
         12, True, 1, None, 0),
        ("+ Cond(25)", lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056),
         12, False, 0, lambda h: wheel_signal_strength(h, 50), 25),
        ("+ Cond(30)", lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056),
         12, False, 0, lambda h: wheel_signal_strength(h, 50), 30),
    ]

    for fname, spins in datasets:
        if len(spins) < 100:
            continue
        print(f"\n  {fname} ({len(spins)} spins):")
        for sname, pfunc, picks, use_an, an_nb, cf, ct in strategies_to_test:
            result = backtest(spins, pfunc, num_picks=picks, warmup=50,
                            use_anchor=use_an, anchor_nb=an_nb,
                            conditional_func=cf, cond_thresh=ct, label=sname)
            bp = result.get('bet_pct', 100)
            print(f"    {sname:20s}: hit={result['hit_rate']:6.2f}% edge={result['edge']:+6.2f}% bet={bp:5.1f}% profit=${result['profit']:8.2f}")
    print()

    # ─── FINAL RECOMMENDATION ───────────────────────────────────────────
    print("=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print(f"  Baseline:       hit={baseline['hit_rate']}% edge={baseline['edge']}% profit=${baseline['profit']}")
    print(f"  Best combined:  {best_combined['label']}")
    print(f"                  hit={best_combined['hit_rate']}% edge={best_combined['edge']}% profit=${best_combined['profit']}")
    print(f"                  bet_pct={best_combined.get('bet_pct', 100)}%")
    print()
    print(f"  Best hot num:   {best_hot['label']}")
    print(f"  Best anchor:    {best_an['label']}")
    print(f"  Best cond:      {best_cond['label']}")
    print(f"  Best picks:     {best_picks['label']}")
    print()

    imp = best_combined['profit'] - baseline['profit']
    print(f"  Profit improvement: ${imp:+.2f}")
    print(f"  Edge improvement:   {best_combined['edge'] - baseline['edge']:+.2f}%")


if __name__ == '__main__':
    main()
