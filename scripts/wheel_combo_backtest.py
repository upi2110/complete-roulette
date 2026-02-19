#!/usr/bin/env python3
"""
Wheel Strategy Combo Backtest — Tests Wheel Strategy paired with each model.

Walk-forward on real userdata (2,528 spins), top-12 straight numbers, $2/number.

Tests:
  1. Each model solo (6 models)
  2. Wheel Strategy + each other model (5 pairs, sweeping weight ratios)
  3. Best 2-model pair + 3rd model (triple combos)
  4. Signal gate analysis per combo
  5. Full weight grid for top pairs

Output: which combo extracts the most edge, and at what weight ratio.
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
)

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


# ─── Individual Model Probability Functions ─────────────────────────────

def frequency_probs(history):
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


def markov_probs(history):
    """Markov: 60% first-order + 40% second-order, Laplace smoothing=0.5."""
    if len(history) < 3:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    smooth = 0.5
    counts_1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
    counts_2 = defaultdict(lambda: np.zeros(TOTAL_NUMBERS))
    for j in range(1, len(history)):
        counts_1[history[j-1]][history[j]] += 1
        if j >= 2:
            counts_2[(history[j-2], history[j-1])][history[j]] += 1

    cur = history[-1]
    s1 = counts_1[cur] + smooth
    p1 = s1 / s1.sum()

    key = (history[-2], history[-1])
    if key in counts_2:
        s2 = counts_2[key] + smooth
        p2 = s2 / s2.sum()
    else:
        p2 = p1

    combined = 0.6 * p1 + 0.4 * p2
    return combined / combined.sum()


def pattern_probs(history):
    """Pattern: sector bias + repeater boost + dealer signature."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 20:
        return probs

    recent = history[-20:]
    # Sector boost
    for sector_nums in WHEEL_SECTORS.values():
        sc = sum(1 for n in recent if n in sector_nums)
        exp = len(recent) * len(sector_nums) / TOTAL_NUMBERS
        if exp > 0 and sc / exp >= 1.4:
            for n in sector_nums:
                probs[n] *= 1.3

    # Repeater boost
    rc = Counter(recent)
    for num, cnt in rc.items():
        if cnt >= 2:
            probs[num] *= (1 + cnt / len(recent))

    # Dealer signature (position gap clustering)
    if len(history) >= 50:
        diffs = []
        wl = len(WHEEL_ORDER)
        for i in range(max(1, len(history)-200), len(history)):
            p1 = NUMBER_TO_POSITION.get(history[i-1], 0)
            p2 = NUMBER_TO_POSITION.get(history[i], 0)
            diffs.append((p2 - p1) % wl)
        if diffs:
            gap_counts = Counter(diffs)
            top_gaps = gap_counts.most_common(3)
            last_pos = NUMBER_TO_POSITION.get(history[-1], 0)
            for gap, cnt in top_gaps:
                expected = len(diffs) / wl
                if cnt > expected * 1.5:
                    target_pos = (last_pos + gap) % wl
                    target_num = WHEEL_ORDER[target_pos]
                    probs[target_num] *= 1.15

    return probs / probs.sum()


def wheel_strategy_probs(history, window=50, boost=1.30, dampen=0.70):
    """Wheel Strategy: table/polarity/set trend boost/dampen."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs

    recent = history[-window:]

    # Pre-compute membership
    table_map = {}
    pol_map = {}
    set_map = {}
    for n in range(TOTAL_NUMBERS):
        table_map[n] = '0' if n in WHEEL_TABLE_0 else '19'
        pol_map[n] = 'pos' if n in WHEEL_POSITIVE else 'neg'
        if n in WHEEL_SET_1: set_map[n] = 1
        elif n in WHEEL_SET_2: set_map[n] = 2
        else: set_map[n] = 3

    # Table trend
    t0 = sum(1 for n in recent if table_map.get(n) == '0')
    t19 = len(recent) - t0
    total = len(recent)
    t0_pct = t0 / total
    t19_pct = t19 / total
    hot_table = '0' if t0 > t19 else '19' if t19 > t0 else None

    # Polarity trend
    pos = sum(1 for n in recent if pol_map.get(n) == 'pos')
    neg = len(recent) - pos
    pos_pct = pos / total
    neg_pct = neg / total
    hot_pol = 'pos' if pos > neg else 'neg' if neg > pos else None

    # Set trend
    s_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        s_counts[set_map.get(n, 3)] += 1
    hot_set = max(s_counts, key=s_counts.get)
    if s_counts[hot_set] <= total / 3 * 1.1:
        hot_set = None

    # Apply boosts
    for num in range(TOTAL_NUMBERS):
        multiplier = 1.0

        if hot_table is not None:
            imbalance = abs(t0_pct - t19_pct) / 0.5
            tb = 1.0 + (boost - 1.0) * imbalance
            td = 1.0 - (1.0 - dampen) * imbalance
            if table_map[num] == hot_table:
                multiplier *= tb
            else:
                multiplier *= td

        if hot_pol is not None:
            imbalance = abs(pos_pct - neg_pct) / 0.5
            pb = 1.0 + (boost - 1.0) * imbalance
            pd = 1.0 - (1.0 - dampen) * imbalance
            if pol_map[num] == hot_pol:
                multiplier *= pb
            else:
                multiplier *= pd

        if hot_set is not None:
            set_pcts = {k: s_counts[k] / total for k in s_counts}
            excess = (set_pcts[hot_set] - 1/3) / (1/3)
            excess = max(0, min(1, excess))
            sb = 1.0 + (boost - 1.0) * excess
            sd = 1.0 - (1.0 - dampen) * excess
            if set_map[num] == hot_set:
                multiplier *= sb
            else:
                multiplier *= sd

        probs[num] *= multiplier

    return probs / probs.sum()


def hot_number_probs(history, window=None, boost_factor=None):
    """Hot Number: boost numbers appearing frequently in last window."""
    if window is None:
        window = HOT_NUMBER_WINDOW
    if boost_factor is None:
        boost_factor = HOT_NUMBER_BOOST_FACTOR

    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < window:
        return probs

    recent = history[-window:]
    counts = Counter(recent)
    expected = window / TOTAL_NUMBERS

    for num in range(TOTAL_NUMBERS):
        c = counts.get(num, 0)
        if c > expected * 1.5:
            ratio = c / expected
            probs[num] *= 1.0 + (boost_factor - 1.0) * min(1.0, (ratio - 1.0))
        elif c == 0:
            probs[num] *= 0.8

    return probs / probs.sum()


def lstm_dummy_probs(history):
    """LSTM placeholder — returns uniform (since it's disabled/below random)."""
    return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)


# ─── Wheel Strategy Signal Strength ────────────────────────────────────
def wheel_signal_strength(history, window=50):
    """Calculate wheel strategy signal strength 0-100 (matches production code)."""
    if len(history) < 5:
        return 0.0

    recent = history[-window:]
    total = len(recent)
    scores = []

    # Table imbalance
    t0 = sum(1 for n in recent if n in WHEEL_TABLE_0)
    expected_t0 = 19 / 37
    t0_dev = abs(t0/total - expected_t0)
    scores.append(min(100, t0_dev / 0.25 * 100))

    # Polarity imbalance
    pos = sum(1 for n in recent if n in WHEEL_POSITIVE)
    expected_pos = 19 / 37
    pos_dev = abs(pos/total - expected_pos)
    scores.append(min(100, pos_dev / 0.25 * 100))

    # Set imbalance
    s_counts = {1: 0, 2: 0, 3: 0}
    for n in recent:
        if n in WHEEL_SET_1: s_counts[1] += 1
        elif n in WHEEL_SET_2: s_counts[2] += 1
        else: s_counts[3] += 1
    set_pcts = [s_counts[k]/total for k in [1,2,3]]
    max_dev = max(abs(p - 1/3) for p in set_pcts)
    scores.append(min(100, max_dev / 0.20 * 100))

    return sum(scores) / len(scores)


# ─── Backtest Engine ───────────────────────────────────────────────────
def backtest(data, strategy_fn, label="", signal_gate=0):
    """Walk-forward backtest.

    Args:
        data: full spin list
        strategy_fn: fn(history) -> np.array of 37 probabilities
        label: display label
        signal_gate: minimum wheel signal strength to bet (0=always)
    """
    hits = 0
    total_bets = 0
    total_skipped = 0
    bankroll = 4000.0
    peak = 4000.0
    low = 4000.0
    max_dd = 0
    hit_log = []

    for i in range(WARMUP, len(data)):
        history = data[:i]
        actual = data[i]

        # Signal gate check
        if signal_gate > 0:
            sig = wheel_signal_strength(history)
            if sig < signal_gate:
                total_skipped += 1
                continue

        probs = strategy_fn(history)
        top = list(np.argsort(probs)[::-1][:TOP_N])

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
        low = min(low, bankroll)
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

    # Rolling 100-spin
    rolling = []
    for j in range(100, len(hit_log)):
        chunk = hit_log[j-100:j]
        rolling.append(sum(chunk))

    bet_pct = total_bets / (len(data) - WARMUP) * 100

    return {
        'label': label,
        'bets': total_bets,
        'skipped': total_skipped,
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
        'best_100': max(rolling) if rolling else 0,
        'worst_100': min(rolling) if rolling else 0,
    }


# ─── Combo builder ─────────────────────────────────────────────────────
def make_combo(model_fns, weights):
    """Create weighted ensemble of model functions."""
    def combo(history):
        ensemble = np.zeros(TOTAL_NUMBERS)
        for fn, w in zip(model_fns, weights):
            if w > 0:
                ensemble += w * fn(history)
        total = ensemble.sum()
        if total > 0:
            ensemble /= total
        return ensemble
    return combo


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 100)
    print("WHEEL STRATEGY COMBO BACKTEST")
    print(f"Walk-forward on real userdata | Top-{TOP_N} numbers | ${BET_PER_NUM}/number | {PAYOUT}:1 payout")
    print("=" * 100)

    data = load_all_userdata()
    n_preds = len(data) - WARMUP
    print(f"\nData: {len(data)} spins | Warmup: {WARMUP} | Predictions: {n_preds}")
    print(f"Random baseline: {TOP_N}/37 = {TOP_N/37*100:.2f}%")
    print(f"Breakeven: {TOP_N}/36 = {TOP_N/36*100:.2f}%\n")

    start = time.time()

    MODELS = {
        'Frequency':     frequency_probs,
        'WheelStrategy': wheel_strategy_probs,
        'HotNumber':     hot_number_probs,
        'Pattern':       pattern_probs,
        'Markov':        markov_probs,
    }

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: EACH MODEL SOLO
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("SECTION 1: EACH MODEL SOLO (top-12)")
    print("=" * 100)
    print(f"{'Model':<20} {'Bets':>5} {'Hits':>5} {'Hit%':>6} {'Base%':>6} "
          f"{'Edge':>7} {'Profit':>9} {'Final$':>8} {'MaxDD':>7} {'WStr':>5} {'LStr':>5}")
    print("-" * 100)

    solo_results = {}
    for name, fn in MODELS.items():
        r = backtest(data, fn, name)
        solo_results[name] = r
        print(f"{name:<20} {r['bets']:>5} {r['hits']:>5} {r['hit_rate']:>6.2f} "
              f"{r['baseline']:>6.2f} {r['edge']:>+7.2f} {r['profit']:>+9.0f} "
              f"{r['final_bankroll']:>8.0f} {r['max_drawdown']:>7.0f} "
              f"{r['max_win_streak']:>5} {r['max_loss_streak']:>5}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: WHEEL STRATEGY + EACH MODEL (weight sweep)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 2: WHEEL STRATEGY + EACH MODEL (weight sweep)")
    print("WS = Wheel Strategy weight, Other = other model weight")
    print("=" * 100)

    other_models = {k: v for k, v in MODELS.items() if k != 'WheelStrategy'}
    weight_steps = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    pair_best = {}  # track best combo per pair

    for other_name, other_fn in other_models.items():
        print(f"\n--- WheelStrategy + {other_name} ---")
        print(f"{'WS_wt':>6} {'Oth_wt':>6} {'Hit%':>6} {'Edge':>7} {'Profit':>9} "
              f"{'Final$':>8} {'MaxDD':>7} {'Best100':>7} {'Wrst100':>7}")
        print("-" * 80)

        best_edge = -999
        best_r = None

        for ws_w in weight_steps:
            oth_w = round(1.0 - ws_w, 2)
            combo = make_combo(
                [wheel_strategy_probs, other_fn],
                [ws_w, oth_w]
            )
            label = f"WS({ws_w:.0%})+{other_name}({oth_w:.0%})"
            r = backtest(data, combo, label)
            if r:
                marker = " ◀ BEST" if r['edge'] > best_edge else ""
                if r['edge'] > best_edge:
                    best_edge = r['edge']
                    best_r = r
                    best_ws_w = ws_w
                print(f"{ws_w:>6.0%} {oth_w:>6.0%} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} "
                      f"{r['profit']:>+9.0f} {r['final_bankroll']:>8.0f} "
                      f"{r['max_drawdown']:>7.0f} {r['best_100']:>7} {r['worst_100']:>7}{marker}")

        if best_r:
            pair_best[f"WS+{other_name}"] = {
                'result': best_r,
                'ws_weight': best_ws_w,
                'other_weight': round(1.0 - best_ws_w, 2),
                'other_name': other_name,
                'other_fn': other_fn,
            }

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: BEST PAIRS SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 3: BEST PAIR SUMMARY (optimal weight per pair)")
    print("=" * 100)
    print(f"{'Pair':<30} {'WS_wt':>6} {'Oth_wt':>6} {'Hit%':>6} {'Edge':>7} "
          f"{'Profit':>9} {'Final$':>8} {'MaxDD':>7}")
    print("-" * 100)

    # Add solos for comparison
    ws_solo = solo_results['WheelStrategy']
    print(f"{'WheelStrategy SOLO':<30} {'100%':>6} {'0%':>6} {ws_solo['hit_rate']:>6.2f} "
          f"{ws_solo['edge']:>+7.2f} {ws_solo['profit']:>+9.0f} "
          f"{ws_solo['final_bankroll']:>8.0f} {ws_solo['max_drawdown']:>7.0f}")

    sorted_pairs = sorted(pair_best.items(), key=lambda x: x[1]['result']['edge'], reverse=True)
    for pair_name, info in sorted_pairs:
        r = info['result']
        print(f"{pair_name:<30} {info['ws_weight']:>5.0%} {info['other_weight']:>6.0%} "
              f"{r['hit_rate']:>6.2f} {r['edge']:>+7.2f} {r['profit']:>+9.0f} "
              f"{r['final_bankroll']:>8.0f} {r['max_drawdown']:>7.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: TRIPLE COMBOS (WS + best_pair_partner + 3rd model)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 4: TRIPLE COMBOS (Wheel Strategy + 2 other models)")
    print("Testing all 3-model combos with weight sweep")
    print("=" * 100)

    other_names = list(other_models.keys())
    triple_results = []

    for i in range(len(other_names)):
        for j in range(i+1, len(other_names)):
            name_a = other_names[i]
            name_b = other_names[j]
            fn_a = other_models[name_a]
            fn_b = other_models[name_b]

            print(f"\n--- WS + {name_a} + {name_b} ---")
            print(f"{'WS':>5} {'A':>5} {'B':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>9} {'Final$':>8}")
            print("-" * 60)

            best_edge = -999
            best_r = None

            # Sweep weights in increments of 10%
            for ws_w in range(10, 80, 10):
                for a_w in range(10, 90 - ws_w, 10):
                    b_w = 100 - ws_w - a_w
                    if b_w < 5:
                        continue
                    ww = ws_w / 100
                    aw = a_w / 100
                    bw = b_w / 100

                    combo = make_combo(
                        [wheel_strategy_probs, fn_a, fn_b],
                        [ww, aw, bw]
                    )
                    label = f"WS({ws_w}%)+{name_a}({a_w}%)+{name_b}({b_w}%)"
                    r = backtest(data, combo, label)
                    if r and r['edge'] > best_edge:
                        best_edge = r['edge']
                        best_r = r
                        best_weights = (ws_w, a_w, b_w)

            if best_r:
                w = best_weights
                print(f"{w[0]:>4}% {w[1]:>4}% {w[2]:>4}% {best_r['hit_rate']:>6.2f} "
                      f"{best_r['edge']:>+7.2f} {best_r['profit']:>+9.0f} "
                      f"{best_r['final_bankroll']:>8.0f}")
                triple_results.append({
                    'label': f"WS+{name_a}+{name_b}",
                    'weights': best_weights,
                    'result': best_r,
                })

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: SIGNAL GATE ANALYSIS (best combos with signal threshold)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SECTION 5: SIGNAL GATE ANALYSIS")
    print("Only bet when wheel strategy signal >= threshold")
    print("=" * 100)

    # Test the top 3 pairs with signal gate
    top_pairs = sorted_pairs[:3]
    gate_thresholds = [0, 10, 15, 20, 25, 30, 35, 40]

    for pair_name, info in top_pairs:
        r_info = info
        combo = make_combo(
            [wheel_strategy_probs, r_info['other_fn']],
            [r_info['ws_weight'], r_info['other_weight']]
        )

        print(f"\n--- {pair_name} (WS={r_info['ws_weight']:.0%}, Other={r_info['other_weight']:.0%}) ---")
        print(f"{'Gate':>5} {'Bets':>5} {'Bet%':>5} {'Hit%':>6} {'Edge':>7} "
              f"{'Profit':>9} {'$/Spin':>7} {'MaxDD':>7}")
        print("-" * 70)

        for gate in gate_thresholds:
            r = backtest(data, combo, f"{pair_name} gate={gate}", signal_gate=gate)
            if r and r['bets'] > 50:
                profit_per_spin = r['profit'] / r['bets'] if r['bets'] > 0 else 0
                print(f"{gate:>5} {r['bets']:>5} {r['bet_pct']:>5.1f} {r['hit_rate']:>6.2f} "
                      f"{r['edge']:>+7.2f} {r['profit']:>+9.0f} {profit_per_spin:>+7.2f} "
                      f"{r['max_drawdown']:>7.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 6: PER-FILE BREAKDOWN for top combo
    # ═══════════════════════════════════════════════════════════════════
    if sorted_pairs:
        best_pair_name, best_pair_info = sorted_pairs[0]
        best_combo = make_combo(
            [wheel_strategy_probs, best_pair_info['other_fn']],
            [best_pair_info['ws_weight'], best_pair_info['other_weight']]
        )

        print("\n" + "=" * 100)
        print(f"SECTION 6: PER-FILE BREAKDOWN — {best_pair_name}")
        print("=" * 100)

        data_files = sorted([f for f in os.listdir(USERDATA_DIR) if f.endswith('.txt')])
        for fname in data_files:
            fpath = os.path.join(USERDATA_DIR, fname)
            file_data = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        n = int(line)
                        if 0 <= n <= 36:
                            file_data.append(n)

            if len(file_data) > 60:
                r = backtest(file_data, best_combo, fname)
                if r:
                    print(f"  {fname:<12} spins={len(file_data):>4}  hit={r['hit_rate']:>5.1f}%  "
                          f"edge={r['edge']:>+5.1f}%  profit=${r['profit']:>+7.0f}  "
                          f"best100={r['best_100']:>3}  worst100={r['worst_100']:>3}")

    # ═══════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start

    print("\n" + "=" * 100)
    print("GRAND SUMMARY — ALL RESULTS RANKED BY EDGE")
    print("=" * 100)

    # Collect everything
    all_results = []
    for name, r in solo_results.items():
        all_results.append(r)
    for pair_name, info in pair_best.items():
        all_results.append(info['result'])
    for tr in triple_results:
        all_results.append(tr['result'])

    all_results.sort(key=lambda x: x['edge'], reverse=True)

    print(f"\n{'#':>3} {'Strategy':<45} {'Hit%':>6} {'Edge':>7} {'Profit':>9} "
          f"{'Final$':>8} {'MaxDD':>7}")
    print("-" * 100)
    for i, r in enumerate(all_results[:25]):
        print(f"{i+1:>3} {r['label']:<45} {r['hit_rate']:>6.2f} {r['edge']:>+7.2f} "
              f"{r['profit']:>+9.0f} {r['final_bankroll']:>8.0f} {r['max_drawdown']:>7.0f}")

    print(f"\nBacktest time: {elapsed:.1f}s")
    print(f"Total configurations tested: {len(all_results)}")

    # Key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    if all_results:
        best = all_results[0]
        print(f"  BEST OVERALL:  {best['label']}")
        print(f"                 Hit: {best['hit_rate']:.2f}%  Edge: {best['edge']:+.2f}%  "
              f"Profit: ${best['profit']:+.0f}")

    if sorted_pairs:
        bp_name, bp_info = sorted_pairs[0]
        print(f"\n  BEST WS PAIR:  {bp_name}")
        print(f"                 WS weight: {bp_info['ws_weight']:.0%}, "
              f"Other weight: {bp_info['other_weight']:.0%}")
        print(f"                 Hit: {bp_info['result']['hit_rate']:.2f}%  "
              f"Edge: {bp_info['result']['edge']:+.2f}%  "
              f"Profit: ${bp_info['result']['profit']:+.0f}")

    if triple_results:
        triple_results.sort(key=lambda x: x['result']['edge'], reverse=True)
        bt = triple_results[0]
        print(f"\n  BEST TRIPLE:   {bt['label']}")
        print(f"                 Weights: WS={bt['weights'][0]}%, "
              f"A={bt['weights'][1]}%, B={bt['weights'][2]}%")
        print(f"                 Hit: {bt['result']['hit_rate']:.2f}%  "
              f"Edge: {bt['result']['edge']:+.2f}%  "
              f"Profit: ${bt['result']['profit']:+.0f}")


if __name__ == '__main__':
    main()
