#!/usr/bin/env python3
"""
Full Backtest — STRAIGHT NUMBER PREDICTIONS ONLY, 3 to 18 numbers.
Walk-forward validation on real userdata ONLY.

Tests every model individually and in ensemble at each top-N count.
Includes profitability analysis, bankroll simulation, and streak analysis.
"""

import sys
import os
import time
import json
import numpy as np
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    TOTAL_NUMBERS, WHEEL_ORDER, NUMBER_TO_POSITION,
    RED_NUMBERS, BLACK_NUMBERS, WHEEL_SECTORS, USERDATA_DIR,
)


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


# ─── Model Functions ───────────────────────────────────────────────────
def frequency_probs(history):
    """Frequency model — Laplace-smoothed flat distribution."""
    counts = Counter(history)
    total = len(history)
    probs = np.array([
        (counts.get(i, 0) + 1) / (total + TOTAL_NUMBERS)
        for i in range(TOTAL_NUMBERS)
    ])
    return probs / probs.sum()


def frequency_probs_decay(history, decay=0.990):
    """Frequency with time-decay (recent spins weighted higher)."""
    n = len(history)
    wc = np.ones(TOTAL_NUMBERS, dtype=np.float64)
    for idx, num in enumerate(history):
        wc[num] += decay ** (n - 1 - idx)
    return wc / wc.sum()


def frequency_blend(history, decay=0.990, flat_w=0.5):
    """Blend of flat + time-weighted frequency."""
    fp = frequency_probs(history)
    rp = frequency_probs_decay(history, decay)
    blended = flat_w * fp + (1 - flat_w) * rp
    return blended / blended.sum()


def markov_probs(history, o1w=0.6, smooth=0.5):
    """Markov chain — first and second order blended."""
    if len(history) < 3:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

    counts_1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
    counts_2 = defaultdict(lambda: np.zeros(TOTAL_NUMBERS))
    for j in range(1, len(history)):
        counts_1[history[j - 1]][history[j]] += 1
        if j >= 2:
            counts_2[(history[j - 2], history[j - 1])][history[j]] += 1

    cur = history[-1]
    s1 = counts_1[cur] + smooth
    p1 = s1 / s1.sum()

    key = (history[-2], history[-1])
    if key in counts_2:
        s2 = counts_2[key] + smooth
        p2 = s2 / s2.sum()
    else:
        p2 = p1

    combined = o1w * p1 + (1 - o1w) * p2
    return combined / combined.sum()


def pattern_probs(history, lookback=50):
    """Pattern detector — sector bias + repeater boost."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 20:
        return probs

    recent = history[-lookback:]
    total_recent = len(recent)

    # Sector boost
    for sector_name, sector_nums in WHEEL_SECTORS.items():
        sc = sum(1 for n in recent if n in sector_nums)
        exp = total_recent * len(sector_nums) / TOTAL_NUMBERS
        if exp > 0 and sc / exp >= 1.5:
            for n in sector_nums:
                probs[n] *= 1.1

    # Repeater boost
    rc = Counter(recent)
    for num, cnt in rc.items():
        if cnt >= 2:
            probs[num] *= (1 + (cnt / total_recent) * 1.5)

    return probs / probs.sum()


def ensemble_current(history):
    """Current config: freq=0.30, markov=0.40, pattern=0.05."""
    if len(history) < 20:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    fp = frequency_blend(history, 0.998, 0.80)
    mp = markov_probs(history, 0.6, 0.5)
    pp = pattern_probs(history, 20)
    # Redistribute LSTM weight (0.25) among others proportionally
    # freq: 0.30+0.30*0.25/0.70=0.407, markov: 0.40+0.40*0.25/0.70=0.543, pattern: 0.05+0.05*0.25/0.70=0.050
    # Simplified: without LSTM, normalize 0.30+0.40+0.05=0.75 to 1.0
    ens = (0.30 / 0.75) * fp + (0.40 / 0.75) * mp + (0.05 / 0.75) * pp
    return ens / ens.sum()


def ensemble_tuned(history):
    """Tuned config: freq=0.60, markov=0.30, pattern=0.10."""
    if len(history) < 20:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    fp = frequency_probs(history)  # Best: pure flat
    mp = markov_probs(history, 0.7, 2.0)  # Best: o1=0.7, smooth=2.0
    pp = pattern_probs(history, 50)  # Best: lookback=50
    ens = 0.60 * fp + 0.30 * mp + 0.10 * pp
    return ens / ens.sum()


def sector_weighted(history, n_numbers, lookback=50):
    """Wheel-sector weighted: boost numbers in hot physical sectors."""
    if len(history) < 20:
        return list(range(n_numbers))

    recent = history[-lookback:]
    n = len(recent)
    pos_scores = np.zeros(len(WHEEL_ORDER))

    for i, num in enumerate(recent):
        pos = NUMBER_TO_POSITION.get(num, 0)
        weight = 0.98 ** (n - 1 - i)
        for offset in range(-2, 3):
            p = (pos + offset) % len(WHEEL_ORDER)
            boost = 1.0 if offset == 0 else 0.5
            pos_scores[p] += weight * boost

    top_positions = np.argsort(pos_scores)[::-1][:n_numbers]
    return [WHEEL_ORDER[p] for p in top_positions]


def anchor_expand(history, n_anchors, spread=2):
    """Pick top-N anchors from ensemble, expand ±spread on wheel."""
    if len(history) < 20:
        return list(range(n_anchors * (1 + 2 * spread)))
    probs = ensemble_tuned(history)
    top = list(np.argsort(probs)[::-1][:n_anchors])

    selected = set()
    for num in top:
        pos = NUMBER_TO_POSITION.get(num, 0)
        for offset in range(-spread, spread + 1):
            p = (pos + offset) % len(WHEEL_ORDER)
            selected.add(WHEEL_ORDER[p])
    return list(selected)


# ─── Walk-Forward Evaluation ──────────────────────────────────────────
def backtest(data, strategy_fn, warmup=100, label=""):
    """Walk-forward: at step i, predict from data[:i], check data[i].
    strategy_fn(history) -> list of predicted numbers
    """
    hits = 0
    total = 0
    hit_log = []
    all_num_counts = []

    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]
        predicted = strategy_fn(history)
        n_nums = len(predicted)
        all_num_counts.append(n_nums)

        hit = 1 if actual in predicted else 0
        hits += hit
        total += 1
        hit_log.append(hit)

    if total == 0:
        return None

    avg_nums = np.mean(all_num_counts)
    hit_rate = hits / total * 100
    baseline = avg_nums / 37 * 100
    edge = hit_rate - baseline

    # Straight bet profitability ($1 per number)
    bankroll = 4000.0
    peak = 4000.0
    low = 4000.0
    max_drawdown = 0

    for idx, hit in enumerate(hit_log):
        n = all_num_counts[idx]
        if hit:
            bankroll += 35 + 1 - n  # win: get 35+1 back, minus cost of n bets
        else:
            bankroll -= n
        peak = max(peak, bankroll)
        low = min(low, bankroll)
        max_drawdown = max(max_drawdown, peak - bankroll)

    total_cost = sum(all_num_counts)
    total_winnings = hits * 36  # each hit returns 36 (35 payout + 1 original)
    profit = total_winnings - total_cost
    roi = profit / total_cost * 100 if total_cost > 0 else 0

    # Streak analysis
    max_win = 0
    max_loss = 0
    cur = 0
    for h in hit_log:
        if h:
            cur = cur + 1 if cur > 0 else 1
            max_win = max(max_win, cur)
        else:
            cur = cur - 1 if cur < 0 else -1
            max_loss = max(max_loss, abs(cur))

    # Rolling 100-spin hit rate
    rolling_rates = []
    window = 100
    for j in range(window, len(hit_log)):
        chunk = hit_log[j - window:j]
        rolling_rates.append(sum(chunk) / window * 100)

    best_100 = max(rolling_rates) if rolling_rates else 0
    worst_100 = min(rolling_rates) if rolling_rates else 0

    return {
        'label': label,
        'avg_numbers': round(avg_nums, 1),
        'total_predictions': total,
        'hits': hits,
        'hit_rate': round(hit_rate, 2),
        'baseline': round(baseline, 2),
        'edge': round(edge, 2),
        'roi': round(roi, 2),
        'final_bankroll': round(bankroll, 2),
        'peak_bankroll': round(peak, 2),
        'low_bankroll': round(low, 2),
        'max_drawdown': round(max_drawdown, 2),
        'max_win_streak': max_win,
        'max_loss_streak': max_loss,
        'profit': round(profit, 2),
        'best_100_spin_rate': round(best_100, 1),
        'worst_100_spin_rate': round(worst_100, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print("FULL BACKTEST — STRAIGHT NUMBER PREDICTIONS, 3 to 18 numbers")
    print("Walk-Forward Validation on Real Userdata ONLY")
    print("$1 per number straight bet, 35:1 payout")
    print("=" * 90)

    data = load_all_userdata()
    warmup = 100
    n_preds = len(data) - warmup

    print(f"\nData: {len(data)} spins from userdata/")
    print(f"Warmup: {warmup} spins | Predictions: {n_preds}")
    print(f"Starting bankroll: $4,000")

    start = time.time()
    all_results = []

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: INDIVIDUAL MODELS AT EACH TOP-N
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 1: INDIVIDUAL MODELS — Top-N Straight Numbers")
    print("=" * 90)

    models = {
        'Freq (flat)': frequency_probs,
        'Freq (decay)': lambda h: frequency_probs_decay(h, 0.990),
        'Freq (blend)': lambda h: frequency_blend(h, 0.990, 0.5),
        'Markov (cur)': lambda h: markov_probs(h, 0.6, 0.5),
        'Markov (tuned)': lambda h: markov_probs(h, 0.7, 2.0),
        'Pattern (lb=20)': lambda h: pattern_probs(h, 20),
        'Pattern (lb=50)': lambda h: pattern_probs(h, 50),
        'Ens (current)': ensemble_current,
        'Ens (tuned)': ensemble_tuned,
    }

    top_ns = [3, 5, 6, 8, 9, 10, 12, 14, 16, 18]

    for top_n in top_ns:
        print(f"\n--- Top-{top_n} numbers (baseline: {top_n/37*100:.1f}%) ---")
        print(f"{'Model':<20} {'Hit%':>6} {'Base%':>6} {'Edge':>6} "
              f"{'ROI%':>7} {'Final$':>8} {'MaxDD':>7} {'Best100':>7} {'Worst100':>8} "
              f"{'WinStr':>6} {'LossStr':>7}")
        print("-" * 90)

        group = []
        for name, fn in models.items():
            def make_strat(mfn=fn, n=top_n):
                def strat(history):
                    if len(history) < 20:
                        return list(range(n))
                    probs = mfn(history)
                    return list(np.argsort(probs)[::-1][:n])
                return strat

            r = backtest(data, make_strat(), warmup, f"{name} top-{top_n}")
            if r:
                group.append(r)
                all_results.append(r)
                print(f"{name:<20} {r['hit_rate']:>6.1f} {r['baseline']:>6.1f} "
                      f"{r['edge']:>+6.1f} {r['roi']:>7.1f} {r['final_bankroll']:>8.0f} "
                      f"{r['max_drawdown']:>7.0f} {r['best_100_spin_rate']:>7.1f} "
                      f"{r['worst_100_spin_rate']:>8.1f} {r['max_win_streak']:>6} "
                      f"{r['max_loss_streak']:>7}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: SECTOR & ANCHOR-BASED STRATEGIES
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 2: SECTOR & ANCHOR STRATEGIES")
    print("=" * 90)

    print(f"\n{'Strategy':<30} {'#Nums':>5} {'Hit%':>6} {'Base%':>6} {'Edge':>6} "
          f"{'ROI%':>7} {'Final$':>8} {'MaxDD':>7}")
    print("-" * 90)

    for n_nums in [5, 8, 10, 12, 15, 18]:
        for lb in [30, 50, 100]:
            def make_sector(nn=n_nums, l=lb):
                return lambda h: sector_weighted(h, nn, l)

            r = backtest(data, make_sector(), warmup, f"Sector-{n_nums} lb={lb}")
            if r:
                all_results.append(r)
                print(f"{r['label']:<30} {r['avg_numbers']:>5.0f} {r['hit_rate']:>6.1f} "
                      f"{r['baseline']:>6.1f} {r['edge']:>+6.1f} {r['roi']:>7.1f} "
                      f"{r['final_bankroll']:>8.0f} {r['max_drawdown']:>7.0f}")

    # Anchor expansion
    print()
    for n_anch in [2, 3, 4, 5, 6]:
        for spread in [1, 2, 3]:
            def make_anch(na=n_anch, sp=spread):
                return lambda h: anchor_expand(h, na, sp)

            r = backtest(data, make_anch(), warmup, f"Anchor-{n_anch} ±{spread}")
            if r:
                all_results.append(r)
                print(f"{r['label']:<30} {r['avg_numbers']:>5.0f} {r['hit_rate']:>6.1f} "
                      f"{r['baseline']:>6.1f} {r['edge']:>+6.1f} {r['roi']:>7.1f} "
                      f"{r['final_bankroll']:>8.0f} {r['max_drawdown']:>7.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: CONFIDENCE-GATED (variable number count)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 3: DYNAMIC NUMBER COUNT (threshold-based selection)")
    print("AI picks numbers where probability > threshold × uniform")
    print("=" * 90)

    print(f"\n{'Factor':<8} {'AvgNums':>7} {'Hit%':>6} {'Base%':>6} {'Edge':>6} "
          f"{'ROI%':>7} {'Final$':>8} {'Profit':>8} {'MaxDD':>7}")
    print("-" * 90)

    for factor in [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        def make_dynamic(f=factor):
            def strat(history):
                if len(history) < 20:
                    return list(range(9))
                probs = ensemble_tuned(history)
                uniform = 1.0 / TOTAL_NUMBERS
                thresh = uniform * f
                selected = [j for j in range(TOTAL_NUMBERS) if probs[j] >= thresh]
                if len(selected) < 3:
                    selected = list(np.argsort(probs)[::-1][:3])
                if len(selected) > 18:
                    selected = list(np.argsort(probs)[::-1][:18])
                return selected
            return strat

        r = backtest(data, make_dynamic(), warmup, f"Dynamic f={factor}")
        if r:
            all_results.append(r)
            print(f"{factor:<8.2f} {r['avg_numbers']:>7.1f} {r['hit_rate']:>6.1f} "
                  f"{r['baseline']:>6.1f} {r['edge']:>+6.1f} {r['roi']:>7.1f} "
                  f"{r['final_bankroll']:>8.0f} {r['profit']:>+8.0f} "
                  f"{r['max_drawdown']:>7.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: PER-DATASET BREAKDOWN (each data file separately)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 4: PER-DATASET BREAKDOWN (Ens tuned top-12)")
    print("Shows how the best strategy performs on each data file independently")
    print("=" * 90)

    data_files = sorted([f for f in os.listdir(USERDATA_DIR) if f.endswith('.txt')])
    cumulative = []
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

        cumulative.extend(file_data)

        if len(file_data) > 50:
            def make_file_strat():
                return lambda h: list(np.argsort(ensemble_tuned(h))[::-1][:12]) \
                    if len(h) >= 20 else list(range(12))

            r = backtest(file_data, make_file_strat(), min(50, len(file_data) // 2),
                         f"{fname}")
            if r:
                print(f"  {fname:<15} spins={len(file_data):>4} hit={r['hit_rate']:>5.1f}% "
                      f"base={r['baseline']:>5.1f}% edge={r['edge']:>+5.1f}% "
                      f"profit=${r['profit']:>+7.0f} best100={r['best_100_spin_rate']:>5.1f}%")

    # Also test cumulative (building knowledge over all files)
    print(f"\n  Cumulative test (all {len(data)} spins, model learns across files):")
    r_cum = backtest(data, lambda h: list(np.argsort(ensemble_tuned(h))[::-1][:12])
                     if len(h) >= 20 else list(range(12)),
                     warmup, "Cumulative")
    if r_cum:
        print(f"  {'CUMULATIVE':<15} spins={len(data):>4} hit={r_cum['hit_rate']:>5.1f}% "
              f"base={r_cum['baseline']:>5.1f}% edge={r_cum['edge']:>+5.1f}% "
              f"profit=${r_cum['profit']:>+7.0f} best100={r_cum['best_100_spin_rate']:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start

    print("\n" + "=" * 90)
    print("GRAND SUMMARY — TOP 20 STRATEGIES BY EDGE OVER BASELINE")
    print("=" * 90)

    all_results.sort(key=lambda x: x['edge'], reverse=True)
    print(f"\n{'#':>3} {'Strategy':<30} {'#Nums':>5} {'Hit%':>6} {'Base%':>6} "
          f"{'Edge':>6} {'ROI%':>7} {'Profit':>8} {'Final$':>8}")
    print("-" * 90)
    for i, r in enumerate(all_results[:20]):
        print(f"{i+1:>3} {r['label']:<30} {r['avg_numbers']:>5.0f} {r['hit_rate']:>6.1f} "
              f"{r['baseline']:>6.1f} {r['edge']:>+6.1f} {r['roi']:>7.1f} "
              f"{r['profit']:>+8.0f} {r['final_bankroll']:>8.0f}")

    print(f"\n--- BOTTOM 10 (worst edge) ---")
    for r in all_results[-10:]:
        print(f"    {r['label']:<30} {r['avg_numbers']:>5.0f} {r['hit_rate']:>6.1f} "
              f"{r['baseline']:>6.1f} {r['edge']:>+6.1f}")

    print(f"\nBacktest time: {elapsed:.0f}s")
    print(f"Total strategies tested: {len(all_results)}")

    # Key insight
    print("\n" + "=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)
    profitable = [r for r in all_results if r['profit'] > 0]
    print(f"Profitable strategies: {len(profitable)} / {len(all_results)}")
    if profitable:
        best = profitable[0]
        print(f"Most profitable: {best['label']} — "
              f"${best['profit']:+.0f} profit, {best['hit_rate']:.1f}% hit rate, "
              f"{best['edge']:+.1f}% edge")

    positive_edge = [r for r in all_results if r['edge'] > 0]
    print(f"Positive edge strategies: {len(positive_edge)} / {len(all_results)}")

    # Break-even analysis
    print(f"\nBreak-even hit rates for straight bets ($1/number, 35:1 payout):")
    for n in [3, 5, 8, 9, 10, 12, 14, 16, 18]:
        breakeven = n / 36 * 100  # Need to win n/36 of the time to break even
        baseline = n / 37 * 100
        print(f"  {n:>2} numbers: need {breakeven:.1f}% (baseline {baseline:.1f}%, "
              f"gap={breakeven - baseline:.2f}%)")

    # Save
    out_path = os.path.join(BASE_DIR, 'scripts', 'backtest_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == '__main__':
    main()
