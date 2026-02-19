#!/usr/bin/env python3
"""
Deep Accuracy Analysis — Find every possible improvement to prediction accuracy.

Tests:
  1. Frequency window sweep (10-100)
  2. Frequency blend ratio sweep (0.1-0.9 flat vs recent)
  3. Wheel strategy window sweep (20-100)
  4. Wheel strategy boost/dampen sweep
  5. Ensemble weight grid search (Freq/Wheel/Pattern combos)
  6. Neighbour expansion vs pure top-N selection
  7. Hot number boosting (repeat analysis)
  8. Sector momentum / autocorrelation
  9. Conditional betting (only bet when signal is strong)
  10. Combined best-of-all improvements

Walk-forward validation on all 5 userdata files.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import Counter
from itertools import product as iterproduct

from config import (
    TOTAL_NUMBERS, WHEEL_ORDER, NUMBER_TO_POSITION,
    WHEEL_TABLE_0, WHEEL_TABLE_19,
    WHEEL_POSITIVE, WHEEL_NEGATIVE,
    WHEEL_SET_1, WHEEL_SET_2, WHEEL_SET_3,
)

# ─── Load all data ──────────────────────────────────────────────────────
def load_all_data():
    """Load all userdata files."""
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


# ─── Frequency Model ────────────────────────────────────────────────────
def frequency_probs(history, recent_window=30, flat_weight=0.5, decay=0.998):
    """Compute frequency-based probabilities."""
    n = len(history)
    if n == 0:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

    # Overall decayed frequency
    counts = np.zeros(TOTAL_NUMBERS)
    for i, num in enumerate(history):
        age = n - 1 - i
        counts[num] += decay ** age
    total = counts.sum()
    if total > 0:
        overall = counts / total
    else:
        overall = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

    # Recent window frequency
    recent = history[-recent_window:] if recent_window > 0 else history
    recent_counts = np.zeros(TOTAL_NUMBERS)
    for num in recent:
        recent_counts[num] += 1
    recent_total = recent_counts.sum()
    if recent_total > 0:
        recent_dist = recent_counts / recent_total
    else:
        recent_dist = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

    # Blend
    blended = flat_weight * overall + (1.0 - flat_weight) * recent_dist
    blended /= blended.sum()
    return blended


# ─── Wheel Strategy Model ───────────────────────────────────────────────
# Pre-compute maps
_table_map = {}
_polarity_map = {}
_set_map = {}
for _n in range(TOTAL_NUMBERS):
    _table_map[_n] = '0' if _n in WHEEL_TABLE_0 else '19'
    _polarity_map[_n] = 'positive' if _n in WHEEL_POSITIVE else 'negative'
    if _n in WHEEL_SET_1:
        _set_map[_n] = 1
    elif _n in WHEEL_SET_2:
        _set_map[_n] = 2
    else:
        _set_map[_n] = 3


def wheel_strategy_probs(history, window=50, boost=1.30, dampen=0.70):
    """Compute wheel-strategy probabilities."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs

    recent = history[-window:]
    total = len(recent)

    # Table trend
    t0 = sum(1 for n in recent if _table_map.get(n) == '0')
    t19 = total - t0
    t0_pct = t0 / total
    t19_pct = t19 / total
    hot_table = '0' if t0 > t19 else '19' if t19 > t0 else None

    # Polarity trend
    pos = sum(1 for n in recent if _polarity_map.get(n) == 'positive')
    neg = total - pos
    pos_pct = pos / total
    neg_pct = neg / total
    hot_pol = 'positive' if pos > neg else 'negative' if neg > pos else None

    # Set trend
    set_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        s = _set_map.get(n, 3)
        set_counts[s] += 1
    hot_set_num = max(set_counts, key=set_counts.get)
    expected = total / 3
    hot_set = hot_set_num if set_counts[hot_set_num] > expected * 1.1 else None

    for num in range(TOTAL_NUMBERS):
        multiplier = 1.0

        # Table
        if hot_table is not None:
            imbalance = abs(t0_pct - t19_pct) / 0.5
            tb = 1.0 + (boost - 1.0) * imbalance
            td = 1.0 - (1.0 - dampen) * imbalance
            if _table_map.get(num) == hot_table:
                multiplier *= tb
            else:
                multiplier *= td

        # Polarity
        if hot_pol is not None:
            imbalance = abs(pos_pct - neg_pct) / 0.5
            pb = 1.0 + (boost - 1.0) * imbalance
            pd = 1.0 - (1.0 - dampen) * imbalance
            if _polarity_map.get(num) == hot_pol:
                multiplier *= pb
            else:
                multiplier *= pd

        # Set
        if hot_set is not None:
            set_pcts = {s: set_counts[s] / total for s in [1, 2, 3]}
            hot_excess = max(0, min(1, (set_pcts[hot_set] - 1/3) / (1/3)))
            sb = 1.0 + (boost - 1.0) * hot_excess
            sd = 1.0 - (1.0 - dampen) * hot_excess
            if _set_map.get(num, 3) == hot_set:
                multiplier *= sb
            else:
                multiplier *= sd

        probs[num] *= multiplier

    probs /= probs.sum()
    return probs


# ─── Pattern Model (simplified) ─────────────────────────────────────────
def pattern_probs(history):
    """Simple pattern-based probabilities."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 10:
        return probs

    # Gap analysis: numbers that haven't appeared for a while get slight boost
    last_seen = {}
    for i, n in enumerate(history):
        last_seen[n] = i

    max_gap = len(history)
    for num in range(TOTAL_NUMBERS):
        if num in last_seen:
            gap = len(history) - 1 - last_seen[num]
            # Numbers with larger gaps get slight boost (due-number effect)
            gap_factor = 1.0 + min(0.3, gap / max_gap)
            probs[num] *= gap_factor
        else:
            probs[num] *= 1.3  # Never seen = max boost

    probs /= probs.sum()
    return probs


# ─── Hot Number Boost ────────────────────────────────────────────────────
def hot_number_boost_probs(history, window=15, boost_factor=1.5):
    """Boost numbers that have appeared recently (hot numbers)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window:
        return probs

    recent = history[-window:]
    counts = Counter(recent)
    expected = window / TOTAL_NUMBERS

    for num in range(TOTAL_NUMBERS):
        if counts.get(num, 0) > expected * 1.5:  # Genuinely hot
            probs[num] *= boost_factor
        elif counts.get(num, 0) == 0:  # Cold
            probs[num] *= 0.8

    probs /= probs.sum()
    return probs


# ─── Sector Momentum ────────────────────────────────────────────────────
def sector_momentum_probs(history, sector_size=5, window=20, momentum_boost=1.4):
    """Boost sectors of the wheel that have been hit recently."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window:
        return probs

    recent = history[-window:]
    wheel_len = len(WHEEL_ORDER)
    num_sectors = (wheel_len + sector_size - 1) // sector_size

    # Count hits per sector
    sector_hits = [0] * num_sectors
    for n in recent:
        pos = NUMBER_TO_POSITION.get(n, 0)
        sector = pos // sector_size
        sector_hits[sector] += 1

    expected_per_sector = window * sector_size / TOTAL_NUMBERS

    for num in range(TOTAL_NUMBERS):
        pos = NUMBER_TO_POSITION.get(num, 0)
        sector = pos // sector_size
        ratio = sector_hits[sector] / max(expected_per_sector, 0.1)
        if ratio > 1.2:
            probs[num] *= 1.0 + (momentum_boost - 1.0) * min(1.0, (ratio - 1.0))
        elif ratio < 0.8:
            probs[num] *= 0.9

    probs /= probs.sum()
    return probs


# ─── Repeat Analysis ────────────────────────────────────────────────────
def repeat_boost_probs(history, lookback=5, repeat_boost=2.0):
    """Numbers that appeared in last N spins get boosted (repeats are common)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < lookback:
        return probs

    recent_set = set(history[-lookback:])
    for num in recent_set:
        probs[num] *= repeat_boost

    probs /= probs.sum()
    return probs


# ─── Transition Analysis ────────────────────────────────────────────────
def transition_probs(history, order=1):
    """First-order Markov transition probabilities."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < order + 1:
        return probs

    # Build transition matrix
    trans = np.ones((TOTAL_NUMBERS, TOTAL_NUMBERS))  # Laplace smoothing
    for i in range(order, len(history)):
        prev = history[i - order]
        curr = history[i]
        trans[prev][curr] += 1

    # Normalize rows
    for i in range(TOTAL_NUMBERS):
        row_sum = trans[i].sum()
        if row_sum > 0:
            trans[i] /= row_sum

    last = history[-order]
    probs = trans[last].copy()
    probs /= probs.sum()
    return probs


# ─── Anti-Trend Strategy ────────────────────────────────────────────────
def anti_trend_wheel_probs(history, window=50, boost=1.30, dampen=0.70):
    """REVERSE of wheel strategy — bet AGAINST the trend (mean reversion)."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs

    recent = history[-window:]
    total = len(recent)

    t0 = sum(1 for n in recent if _table_map.get(n) == '0')
    t19 = total - t0
    # COLD table (opposite of hot) gets boosted
    hot_table = '19' if t0 > t19 else '0' if t19 > t0 else None  # REVERSED

    pos = sum(1 for n in recent if _polarity_map.get(n) == 'positive')
    neg = total - pos
    hot_pol = 'negative' if pos > neg else 'positive' if neg > pos else None  # REVERSED

    set_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        set_counts[_set_map.get(n, 3)] += 1
    cold_set = min(set_counts, key=set_counts.get)
    expected = total / 3
    hot_set = cold_set if set_counts[max(set_counts, key=set_counts.get)] > expected * 1.1 else None  # REVERSED

    t0_pct = t0 / total
    t19_pct = t19 / total
    pos_pct = pos / total
    neg_pct = neg / total

    for num in range(TOTAL_NUMBERS):
        multiplier = 1.0
        if hot_table is not None:
            imbalance = abs(t0_pct - t19_pct) / 0.5
            tb = 1.0 + (boost - 1.0) * imbalance
            td = 1.0 - (1.0 - dampen) * imbalance
            if _table_map.get(num) == hot_table:
                multiplier *= tb
            else:
                multiplier *= td
        if hot_pol is not None:
            imbalance = abs(pos_pct - neg_pct) / 0.5
            pb = 1.0 + (boost - 1.0) * imbalance
            pd = 1.0 - (1.0 - dampen) * imbalance
            if _polarity_map.get(num) == hot_pol:
                multiplier *= pb
            else:
                multiplier *= pd
        if hot_set is not None:
            set_pcts = {s: set_counts[s] / total for s in [1, 2, 3]}
            hot_excess = max(0, min(1, (set_pcts[max(set_counts, key=set_counts.get)] - 1/3) / (1/3)))
            sb = 1.0 + (boost - 1.0) * hot_excess
            sd = 1.0 - (1.0 - dampen) * hot_excess
            if _set_map.get(num, 3) == hot_set:
                multiplier *= sb
            else:
                multiplier *= sd
        probs[num] *= multiplier

    probs /= probs.sum()
    return probs


# ─── Walk-Forward Backtest Engine ────────────────────────────────────────
def backtest(all_spins, prob_func, num_picks=12, warmup=50, label=""):
    """
    Walk-forward backtest.
    prob_func(history) → np.array(37,) probabilities
    Returns dict with hit_rate, edge, profit, etc.
    """
    hits = 0
    total_bets = 0
    profit = 0
    bet_cost = 3.0
    payout = 35

    for i in range(warmup, len(all_spins)):
        history = all_spins[:i]
        actual = all_spins[i]

        probs = prob_func(history)
        top_n = np.argsort(probs)[::-1][:num_picks]

        total_bets += 1
        cost = bet_cost * num_picks
        profit -= cost

        if actual in top_n:
            hits += 1
            profit += bet_cost * (payout + 1)  # Win = bet * 36

    if total_bets == 0:
        return {'label': label, 'hit_rate': 0, 'edge': 0, 'profit': 0, 'bets': 0}

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
        'expected': round(expected_random, 2),
    }


def backtest_conditional(all_spins, prob_func, strength_func, threshold, num_picks=12, warmup=50, label=""):
    """
    Conditional backtest — only bet when signal strength exceeds threshold.
    strength_func(history) → float (0-100)
    """
    hits = 0
    total_bets = 0
    skipped = 0
    profit = 0
    bet_cost = 3.0
    payout = 35

    for i in range(warmup, len(all_spins)):
        history = all_spins[:i]
        actual = all_spins[i]

        strength = strength_func(history)
        if strength < threshold:
            skipped += 1
            continue

        probs = prob_func(history)
        top_n = np.argsort(probs)[::-1][:num_picks]

        total_bets += 1
        cost = bet_cost * num_picks
        profit -= cost

        if actual in top_n:
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
        'bet_pct': round(total_bets / (total_bets + skipped) * 100, 1),
    }


# ─── Wheel Strategy Signal Strength ─────────────────────────────────────
def wheel_signal_strength(history, window=50):
    """Calculate wheel strategy signal strength (0-100)."""
    if len(history) < 5:
        return 0.0
    recent = history[-window:]
    total = len(recent)

    # Table imbalance
    t0 = sum(1 for n in recent if _table_map.get(n) == '0')
    t0_pct = t0 / total
    table_dev = abs(t0_pct - 19/37)
    table_score = min(100, table_dev / 0.25 * 100)

    # Polarity imbalance
    pos = sum(1 for n in recent if _polarity_map.get(n) == 'positive')
    pos_pct = pos / total
    pol_dev = abs(pos_pct - 19/37)
    pol_score = min(100, pol_dev / 0.25 * 100)

    # Set imbalance
    set_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        set_counts[_set_map.get(n, 3)] += 1
    set_pcts = [set_counts[s] / total for s in [1, 2, 3]]
    max_set_dev = max(abs(p - 1/3) for p in set_pcts)
    set_score = min(100, max_set_dev / 0.20 * 100)

    return (table_score + pol_score + set_score) / 3


# ─── Ensemble Blend ─────────────────────────────────────────────────────
def ensemble_blend(history, freq_w=0.611, wheel_w=0.333, pattern_w=0.056,
                   freq_window=30, freq_flat=0.5,
                   wheel_window=50, wheel_boost=1.30, wheel_dampen=0.70,
                   extra_models=None):
    """Ensemble blend of multiple models."""
    fp = frequency_probs(history, recent_window=freq_window, flat_weight=freq_flat)
    wp = wheel_strategy_probs(history, window=wheel_window, boost=wheel_boost, dampen=wheel_dampen)
    pp = pattern_probs(history)

    ensemble = fp * freq_w + wp * wheel_w + pp * pattern_w

    if extra_models:
        remaining_w = 1.0 - freq_w - wheel_w - pattern_w
        per_model = remaining_w / len(extra_models) if extra_models else 0
        for model_func, weight in extra_models:
            mp = model_func(history)
            ensemble += mp * weight

    total = ensemble.sum()
    if total > 0:
        ensemble /= total
    return ensemble


# ═════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def main():
    all_spins, datasets = load_all_data()
    print(f"Total spins: {len(all_spins)} from {len(datasets)} files")
    print(f"Files: {[d[0] for d in datasets]}")
    print()

    # Current baseline
    print("=" * 80)
    print("SECTION 1: CURRENT BASELINE (config values)")
    print("=" * 80)

    baseline = backtest(
        all_spins,
        lambda h: ensemble_blend(h, freq_w=0.611, wheel_w=0.333, pattern_w=0.056,
                                  freq_window=30, freq_flat=0.5,
                                  wheel_window=50, wheel_boost=1.30, wheel_dampen=0.70),
        label="Current Config"
    )
    print(f"  {baseline['label']}: hit={baseline['hit_rate']}% edge={baseline['edge']}% profit=${baseline['profit']}")
    print()

    # ─── SECTION 2: Frequency Window Sweep ──────────────────────────────
    print("=" * 80)
    print("SECTION 2: FREQUENCY RECENT WINDOW SWEEP")
    print("=" * 80)

    best_freq = None
    for window in [10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]:
        result = backtest(
            all_spins,
            lambda h, w=window: frequency_probs(h, recent_window=w, flat_weight=0.5),
            label=f"Freq(window={window})"
        )
        marker = ""
        if best_freq is None or result['edge'] > best_freq['edge']:
            best_freq = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> Best frequency window: {best_freq['label']}")
    print()

    # ─── SECTION 3: Frequency Flat/Recent Blend Sweep ───────────────────
    print("=" * 80)
    print("SECTION 3: FREQUENCY FLAT/RECENT BLEND SWEEP")
    print("=" * 80)

    best_blend = None
    for flat_w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        result = backtest(
            all_spins,
            lambda h, fw=flat_w: frequency_probs(h, recent_window=30, flat_weight=fw),
            label=f"Freq(flat={flat_w})"
        )
        marker = ""
        if best_blend is None or result['edge'] > best_blend['edge']:
            best_blend = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> Best blend: {best_blend['label']}")
    print()

    # ─── SECTION 4: Wheel Strategy Window Sweep ─────────────────────────
    print("=" * 80)
    print("SECTION 4: WHEEL STRATEGY WINDOW SWEEP")
    print("=" * 80)

    best_wheel = None
    for window in [20, 30, 40, 50, 60, 70, 80, 100]:
        result = backtest(
            all_spins,
            lambda h, w=window: wheel_strategy_probs(h, window=w, boost=1.30, dampen=0.70),
            label=f"Wheel(window={window})"
        )
        marker = ""
        if best_wheel is None or result['edge'] > best_wheel['edge']:
            best_wheel = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> Best wheel window: {best_wheel['label']}")
    print()

    # ─── SECTION 5: Wheel Boost/Dampen Sweep ────────────────────────────
    print("=" * 80)
    print("SECTION 5: WHEEL BOOST/DAMPEN SWEEP (window=50)")
    print("=" * 80)

    best_bd = None
    for boost in [1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.80, 2.00]:
        dampen = 2.0 - boost  # Mirror: boost=1.3 → dampen=0.7
        result = backtest(
            all_spins,
            lambda h, b=boost, d=dampen: wheel_strategy_probs(h, window=50, boost=b, dampen=d),
            label=f"Wheel(b={boost},d={round(dampen,2)})"
        )
        marker = ""
        if best_bd is None or result['edge'] > best_bd['edge']:
            best_bd = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> Best boost/dampen: {best_bd['label']}")
    print()

    # ─── SECTION 6: Ensemble Weight Grid Search ─────────────────────────
    print("=" * 80)
    print("SECTION 6: ENSEMBLE WEIGHT GRID SEARCH")
    print("=" * 80)

    best_ensemble = None
    for fw in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for ww in [0.1, 0.2, 0.3, 0.4]:
            pw = round(1.0 - fw - ww, 3)
            if pw < 0 or pw > 0.3:
                continue
            result = backtest(
                all_spins,
                lambda h, f=fw, w=ww, p=pw: ensemble_blend(h, freq_w=f, wheel_w=w, pattern_w=p),
                label=f"Ens(F={fw},W={ww},P={pw})"
            )
            marker = ""
            if best_ensemble is None or result['edge'] > best_ensemble['edge']:
                best_ensemble = result
                marker = " *** BEST"
            print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> Best ensemble: {best_ensemble['label']}")
    print()

    # ─── SECTION 7: Alternative Models (standalone) ─────────────────────
    print("=" * 80)
    print("SECTION 7: ALTERNATIVE MODELS (standalone)")
    print("=" * 80)

    alt_models = [
        ("Hot Number Boost", lambda h: hot_number_boost_probs(h, window=15, boost_factor=1.5)),
        ("Sector Momentum", lambda h: sector_momentum_probs(h, sector_size=5, window=20, momentum_boost=1.4)),
        ("Repeat Boost (5)", lambda h: repeat_boost_probs(h, lookback=5, repeat_boost=2.0)),
        ("Repeat Boost (3)", lambda h: repeat_boost_probs(h, lookback=3, repeat_boost=2.0)),
        ("Repeat Boost (10)", lambda h: repeat_boost_probs(h, lookback=10, repeat_boost=1.5)),
        ("Transition (order=1)", lambda h: transition_probs(h, order=1)),
        ("Anti-Trend Wheel", lambda h: anti_trend_wheel_probs(h, window=50)),
        ("Anti-Trend Wheel (w=30)", lambda h: anti_trend_wheel_probs(h, window=30)),
        ("Pattern (gap analysis)", lambda h: pattern_probs(h)),
    ]

    best_alt = None
    for name, func in alt_models:
        result = backtest(all_spins, func, label=name)
        marker = ""
        if best_alt is None or result['edge'] > best_alt['edge']:
            best_alt = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> Best alternative: {best_alt['label']}")
    print()

    # ─── SECTION 8: Ensemble + Extra Models ─────────────────────────────
    print("=" * 80)
    print("SECTION 8: ENSEMBLE + EXTRA MODELS")
    print("=" * 80)

    combos = [
        ("Base + HotNum(5%)", 0.58, 0.32, 0.05,
         [(lambda h: hot_number_boost_probs(h), 0.05)]),
        ("Base + Repeat3(5%)", 0.58, 0.32, 0.05,
         [(lambda h: repeat_boost_probs(h, lookback=3, repeat_boost=2.0), 0.05)]),
        ("Base + Repeat5(5%)", 0.58, 0.32, 0.05,
         [(lambda h: repeat_boost_probs(h, lookback=5, repeat_boost=2.0), 0.05)]),
        ("Base + SectorMom(5%)", 0.58, 0.32, 0.05,
         [(lambda h: sector_momentum_probs(h), 0.05)]),
        ("Base + Trans1(5%)", 0.58, 0.32, 0.05,
         [(lambda h: transition_probs(h, order=1), 0.05)]),
        ("Base + HotNum(10%)", 0.55, 0.30, 0.05,
         [(lambda h: hot_number_boost_probs(h), 0.10)]),
        ("Base + Repeat3(10%)", 0.55, 0.30, 0.05,
         [(lambda h: repeat_boost_probs(h, lookback=3, repeat_boost=2.0), 0.10)]),
    ]

    best_combo = None
    for name, fw, ww, pw, extras in combos:
        result = backtest(
            all_spins,
            lambda h, f=fw, w=ww, p=pw, e=extras: ensemble_blend(h, freq_w=f, wheel_w=w, pattern_w=p, extra_models=e),
            label=name
        )
        marker = ""
        if best_combo is None or result['edge'] > best_combo['edge']:
            best_combo = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> Best combo: {best_combo['label']}")
    print()

    # ─── SECTION 9: Conditional Betting (Signal Strength Gating) ────────
    print("=" * 80)
    print("SECTION 9: CONDITIONAL BETTING (bet only when signal is strong)")
    print("=" * 80)

    best_cond = None
    for threshold in [5, 10, 15, 20, 25, 30, 40]:
        result = backtest_conditional(
            all_spins,
            lambda h: ensemble_blend(h),
            lambda h: wheel_signal_strength(h, window=50),
            threshold=threshold,
            label=f"Conditional(thresh={threshold})"
        )
        marker = ""
        if best_cond is None or result['edge'] > best_cond['edge']:
            best_cond = result
            marker = " *** BEST"
        bet_pct = result.get('bet_pct', 'N/A')
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']} (bet {bet_pct}% of spins){marker}")
    print(f"  >> Best conditional: {best_cond['label']}")
    print()

    # ─── SECTION 10: Number Count Sweep (8-16 numbers) ──────────────────
    print("=" * 80)
    print("SECTION 10: NUMBER COUNT SWEEP")
    print("=" * 80)

    best_count = None
    for count in [6, 8, 10, 11, 12, 13, 14, 16, 18]:
        result = backtest(
            all_spins,
            lambda h: ensemble_blend(h),
            num_picks=count,
            label=f"Picks={count}"
        )
        marker = ""
        if best_count is None or result['edge'] > best_count['edge']:
            best_count = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> Best count: {best_count['label']}")
    print()

    # ─── SECTION 11: Pure Top-N vs Anchor+Neighbour ─────────────────────
    print("=" * 80)
    print("SECTION 11: PURE TOP-N vs ANCHOR+NEIGHBOUR SELECTION")
    print("=" * 80)

    def anchor_neighbour_select(probs, num_anchors=3, neighbours=2):
        """Select numbers using anchor + neighbours strategy."""
        sorted_idx = np.argsort(probs)[::-1]
        selected = set()
        for anchor in sorted_idx:
            if anchor in selected:
                continue
            if len(selected) >= 12:
                break
            pos = NUMBER_TO_POSITION.get(int(anchor), -1)
            if pos < 0:
                continue
            selected.add(int(anchor))
            for d in [1, -1]:
                for r in range(1, neighbours + 1):
                    if len(selected) >= 12:
                        break
                    nb_pos = (pos + d * r) % len(WHEEL_ORDER)
                    selected.add(WHEEL_ORDER[nb_pos])
        return list(selected)[:12]

    def backtest_selection(all_spins, prob_func, select_func, warmup=50, label=""):
        hits = 0
        total_bets = 0
        profit = 0
        bet_cost = 3.0
        payout = 35

        for i in range(warmup, len(all_spins)):
            history = all_spins[:i]
            actual = all_spins[i]

            probs = prob_func(history)
            selected = select_func(probs)

            total_bets += 1
            cost = bet_cost * len(selected)
            profit -= cost

            if actual in selected:
                hits += 1
                profit += bet_cost * (payout + 1)

        if total_bets == 0:
            return {'label': label, 'hit_rate': 0, 'edge': 0, 'profit': 0}

        hit_rate = hits / total_bets * 100
        expected_random = 12 / TOTAL_NUMBERS * 100
        edge = hit_rate - expected_random

        return {
            'label': label,
            'hit_rate': round(hit_rate, 2),
            'edge': round(edge, 2),
            'profit': round(profit, 2),
            'bets': total_bets,
            'hits': hits,
        }

    prob_func = lambda h: ensemble_blend(h)

    # Pure top-12
    result_topn = backtest(all_spins, prob_func, num_picks=12, label="Pure Top-12")
    print(f"  {result_topn['label']}: hit={result_topn['hit_rate']}% edge={result_topn['edge']}% profit=${result_topn['profit']}")

    # Anchor + 2 neighbours (current strategy)
    for anchors in [2, 3, 4]:
        for nb in [1, 2, 3]:
            result_an = backtest_selection(
                all_spins, prob_func,
                lambda p, a=anchors, n=nb: anchor_neighbour_select(p, num_anchors=a, neighbours=n),
                label=f"Anchor({anchors})+Nb({nb})"
            )
            print(f"  {result_an['label']}: hit={result_an['hit_rate']}% edge={result_an['edge']}% profit=${result_an['profit']}")
    print()

    # ─── SECTION 12: Autocorrelation Analysis ───────────────────────────
    print("=" * 80)
    print("SECTION 12: AUTOCORRELATION & REPEAT PATTERNS")
    print("=" * 80)

    # How often do numbers repeat within N spins?
    for gap in [1, 2, 3, 5, 10, 15, 20]:
        repeats = 0
        total_checks = 0
        for i in range(gap, len(all_spins)):
            total_checks += 1
            if all_spins[i] in all_spins[max(0, i-gap):i]:
                repeats += 1
        expected = 1 - ((TOTAL_NUMBERS - 1) / TOTAL_NUMBERS) ** gap
        actual_rate = repeats / total_checks if total_checks else 0
        print(f"  Repeat within {gap:2d} spins: actual={actual_rate:.3f} expected={expected:.3f} diff={actual_rate-expected:+.3f}")
    print()

    # ─── SECTION 13: Per-Dataset Breakdown ──────────────────────────────
    print("=" * 80)
    print("SECTION 13: PER-DATASET BREAKDOWN (current config)")
    print("=" * 80)

    offset = 0
    for fname, spins in datasets:
        if len(spins) < 100:
            print(f"  {fname}: too short ({len(spins)} spins)")
            offset += len(spins)
            continue
        result = backtest(
            spins,
            lambda h: ensemble_blend(h),
            warmup=50,
            label=fname
        )
        print(f"  {result['label']}: {len(spins)} spins, hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}")
        offset += len(spins)
    print()

    # ─── SECTION 14: Grand Optimized Combo ──────────────────────────────
    print("=" * 80)
    print("SECTION 14: GRAND OPTIMIZED COMBINATIONS")
    print("=" * 80)

    # Test the best parameters found above in combination
    grand_combos = [
        # (label, freq_w, wheel_w, pattern_w, freq_window, freq_flat, wheel_window, wheel_boost, wheel_dampen)
        ("Current Config", 0.611, 0.333, 0.056, 30, 0.5, 50, 1.30, 0.70),
        ("Freq=0.7 Wheel=0.2 Pat=0.1", 0.7, 0.2, 0.1, 30, 0.5, 50, 1.30, 0.70),
        ("Freq=0.6 Wheel=0.3 Pat=0.1", 0.6, 0.3, 0.1, 30, 0.5, 50, 1.30, 0.70),
        ("Freq=0.5 Wheel=0.4 Pat=0.1", 0.5, 0.4, 0.1, 30, 0.5, 50, 1.30, 0.70),
        ("Freq=0.8 Wheel=0.15 Pat=0.05", 0.8, 0.15, 0.05, 30, 0.5, 50, 1.30, 0.70),
        ("FreqW=40", 0.611, 0.333, 0.056, 40, 0.5, 50, 1.30, 0.70),
        ("FreqW=25", 0.611, 0.333, 0.056, 25, 0.5, 50, 1.30, 0.70),
        ("FreqFlat=0.4", 0.611, 0.333, 0.056, 30, 0.4, 50, 1.30, 0.70),
        ("FreqFlat=0.6", 0.611, 0.333, 0.056, 30, 0.6, 50, 1.30, 0.70),
        ("WheelW=60", 0.611, 0.333, 0.056, 30, 0.5, 60, 1.30, 0.70),
        ("WheelW=70", 0.611, 0.333, 0.056, 30, 0.5, 70, 1.30, 0.70),
        ("WheelB=1.20", 0.611, 0.333, 0.056, 30, 0.5, 50, 1.20, 0.80),
        ("WheelB=1.40", 0.611, 0.333, 0.056, 30, 0.5, 50, 1.40, 0.60),
        ("WheelB=1.50", 0.611, 0.333, 0.056, 30, 0.5, 50, 1.50, 0.50),
        # More aggressive combinations
        ("F=0.5 W=0.4 P=0.1 wB=1.20", 0.5, 0.4, 0.1, 30, 0.5, 50, 1.20, 0.80),
        ("F=0.6 W=0.3 P=0.1 wB=1.20", 0.6, 0.3, 0.1, 30, 0.5, 50, 1.20, 0.80),
        ("F=0.6 W=0.3 P=0.1 fW=25", 0.6, 0.3, 0.1, 25, 0.5, 50, 1.30, 0.70),
        ("F=0.6 W=0.3 P=0.1 fW=40", 0.6, 0.3, 0.1, 40, 0.5, 50, 1.30, 0.70),
        ("F=0.55 W=0.35 P=0.1 wW=60", 0.55, 0.35, 0.1, 30, 0.5, 60, 1.30, 0.70),
        ("F=0.65 W=0.25 P=0.1", 0.65, 0.25, 0.1, 30, 0.5, 50, 1.30, 0.70),
    ]

    best_grand = None
    for label, fw, ww, pw, freq_w, freq_f, wheel_w, wheel_b, wheel_d in grand_combos:
        result = backtest(
            all_spins,
            lambda h, f=fw, w=ww, p=pw, fwi=freq_w, ff=freq_f, ww2=wheel_w, wb=wheel_b, wd=wheel_d:
                ensemble_blend(h, freq_w=f, wheel_w=w, pattern_w=p,
                              freq_window=fwi, freq_flat=ff,
                              wheel_window=ww2, wheel_boost=wb, wheel_dampen=wd),
            label=label
        )
        marker = ""
        if best_grand is None or result['edge'] > best_grand['edge']:
            best_grand = result
            marker = " *** BEST"
        print(f"  {result['label']}: hit={result['hit_rate']}% edge={result['edge']}% profit=${result['profit']}{marker}")
    print(f"  >> GRAND BEST: {best_grand['label']}")
    print()

    # ─── FINAL SUMMARY ──────────────────────────────────────────────────
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  Current baseline:     hit={baseline['hit_rate']}% edge={baseline['edge']}% profit=${baseline['profit']}")
    print(f"  Best freq window:     {best_freq['label']}")
    print(f"  Best freq blend:      {best_blend['label']}")
    print(f"  Best wheel window:    {best_wheel['label']}")
    print(f"  Best wheel boost:     {best_bd['label']}")
    print(f"  Best ensemble:        {best_ensemble['label']}")
    print(f"  Best alt model:       {best_alt['label']}")
    print(f"  Best combo:           {best_combo['label']}")
    print(f"  Best conditional:     {best_cond['label']}")
    print(f"  Best num count:       {best_count['label']}")
    print(f"  GRAND BEST:           {best_grand['label']}")
    print()

    # Compare best grand vs baseline
    improvement = best_grand['edge'] - baseline['edge']
    print(f"  Edge improvement: {improvement:+.2f}%")
    print(f"  Profit improvement: ${best_grand['profit'] - baseline['profit']:+.2f}")


if __name__ == '__main__':
    main()
