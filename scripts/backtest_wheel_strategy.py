#!/usr/bin/env python3
"""
Backtest — Wheel Strategy vs Old Ensemble vs Individual Models.

Walk-forward validation on real userdata (~2526 spins).
Tests the new Wheel Strategy (0/19 tables, Positive/Negative, Sets 1/2/3)
both as a standalone model and combined with Frequency in the new ensemble.

$3 per number straight bet, 35:1 payout, 12 numbers.
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
    RED_NUMBERS, BLACK_NUMBERS, WHEEL_SECTORS, USERDATA_DIR,
    WHEEL_TABLE_0, WHEEL_TABLE_19,
    WHEEL_POSITIVE, WHEEL_NEGATIVE,
    WHEEL_SET_1, WHEEL_SET_2, WHEEL_SET_3,
    FREQUENCY_RECENT_WINDOW,
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

def frequency_flat(history):
    """Laplace-smoothed flat frequency."""
    counts = Counter(history)
    total = len(history)
    probs = np.array([
        (counts.get(i, 0) + 1) / (total + TOTAL_NUMBERS)
        for i in range(TOTAL_NUMBERS)
    ])
    return probs / probs.sum()


def frequency_recent_window(history, window=30):
    """Recent window frequency (last N spins)."""
    recent = history[-window:] if len(history) >= window else history
    counts = np.ones(TOTAL_NUMBERS, dtype=np.float64)
    for num in recent:
        counts[num] += 1
    return counts / counts.sum()


def frequency_blend(history, flat_w=0.5, window=30):
    """50/50 blend of flat + recent window."""
    fp = frequency_flat(history)
    rp = frequency_recent_window(history, window)
    blended = flat_w * fp + (1 - flat_w) * rp
    return blended / blended.sum()


def markov_probs(history, o1w=0.6, smooth=0.5):
    """Markov chain — first and second order."""
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
    for sector_name, sector_nums in WHEEL_SECTORS.items():
        sc = sum(1 for n in recent if n in sector_nums)
        exp = total_recent * len(sector_nums) / TOTAL_NUMBERS
        if exp > 0 and sc / exp >= 1.5:
            for n in sector_nums:
                probs[n] *= 1.1
    rc = Counter(recent)
    for num, cnt in rc.items():
        if cnt >= 2:
            probs[num] *= (1 + (cnt / total_recent) * 1.5)
    return probs / probs.sum()


# ─── Wheel Strategy Models ──────────────────────────────────────────────

# Pre-compute mappings
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


def wheel_strategy_probs(history, window=20, boost=1.40, dampen=0.70):
    """Wheel strategy probability distribution."""
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs

    recent = history[-window:]

    # Table trend
    t0 = sum(1 for n in recent if _table_map.get(n) == '0')
    t19 = len(recent) - t0
    total = len(recent)
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
        set_counts[_set_map.get(n, 3)] += 1
    hot_set = max(set_counts, key=set_counts.get)
    if set_counts[hot_set] <= total / 3 * 1.1:
        hot_set = None

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
            if _set_map.get(num) == hot_set:
                multiplier *= sb
            else:
                multiplier *= sd

        probs[num] *= multiplier

    total_p = probs.sum()
    if total_p > 0:
        probs /= total_p
    return probs


# ─── Ensemble Combinations ────────────────────────────────────────────

def old_ensemble(history):
    """OLD: 90% freq + 10% pattern (no wheel strategy)."""
    if len(history) < 20:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    fp = frequency_blend(history, 0.5, 30)
    pp = pattern_probs(history, 50)
    ens = 0.90 * fp + 0.10 * pp
    return ens / ens.sum()


def new_ensemble(history):
    """NEW: 55% freq + 30% wheel strategy + 5% pattern (+ 10% redistributed from disabled)."""
    if len(history) < 5:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    fp = frequency_blend(history, 0.5, 30)
    wp = wheel_strategy_probs(history, 20, 1.40, 0.70)
    pp = pattern_probs(history, 50)
    # Markov + LSTM disabled (0%), redistribute among active:
    # freq=0.55, wheel=0.30, pattern=0.05 -> sum=0.90 -> normalize to 1.0
    # freq=0.611, wheel=0.333, pattern=0.056
    ens = 0.611 * fp + 0.333 * wp + 0.056 * pp
    return ens / ens.sum()


def wheel_only(history):
    """WHEEL STRATEGY ONLY (100%)."""
    return wheel_strategy_probs(history, 20, 1.40, 0.70)


def freq_wheel_5050(history):
    """50/50 Frequency + Wheel Strategy."""
    if len(history) < 5:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    fp = frequency_blend(history, 0.5, 30)
    wp = wheel_strategy_probs(history, 20, 1.40, 0.70)
    ens = 0.50 * fp + 0.50 * wp
    return ens / ens.sum()


def freq_wheel_7030(history):
    """70/30 Frequency + Wheel Strategy."""
    if len(history) < 5:
        return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    fp = frequency_blend(history, 0.5, 30)
    wp = wheel_strategy_probs(history, 20, 1.40, 0.70)
    ens = 0.70 * fp + 0.30 * wp
    return ens / ens.sum()


# ─── Walk-Forward Evaluation ──────────────────────────────────────────
def backtest(data, strategy_fn, warmup=100, label="", bet_per_number=3.0):
    """Walk-forward: at step i, predict from data[:i], check data[i]."""
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

    # Bankroll simulation ($3 per number)
    bankroll = 4000.0
    peak = 4000.0
    low = 4000.0
    max_drawdown = 0

    for idx, hit in enumerate(hit_log):
        n = all_num_counts[idx]
        total_bet = bet_per_number * n
        if hit:
            payout = bet_per_number * 35 + bet_per_number  # 35:1 + stake back
            bankroll += payout - total_bet
        else:
            bankroll -= total_bet
        peak = max(peak, bankroll)
        low = min(low, bankroll)
        max_drawdown = max(max_drawdown, peak - bankroll)

    total_cost = sum(n * bet_per_number for n in all_num_counts)
    total_winnings = hits * (bet_per_number * 36)
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

    # Rolling 50-spin hit rate
    rolling_50 = []
    for j in range(50, len(hit_log)):
        chunk = hit_log[j - 50:j]
        rolling_50.append(sum(chunk) / 50 * 100)
    best_50 = max(rolling_50) if rolling_50 else 0
    worst_50 = min(rolling_50) if rolling_50 else 0

    # Rolling 100-spin hit rate
    rolling_100 = []
    for j in range(100, len(hit_log)):
        chunk = hit_log[j - 100:j]
        rolling_100.append(sum(chunk) / 100 * 100)
    best_100 = max(rolling_100) if rolling_100 else 0
    worst_100 = min(rolling_100) if rolling_100 else 0

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
        'best_50_rate': round(best_50, 1),
        'worst_50_rate': round(worst_50, 1),
        'best_100_rate': round(best_100, 1),
        'worst_100_rate': round(worst_100, 1),
    }


# ─── Wheel Strategy Analysis ──────────────────────────────────────────
def analyze_wheel_groups(data, warmup=100):
    """Analyze how well each group (table/polarity/set) tracks trends."""
    print("\n" + "=" * 90)
    print("WHEEL STRATEGY GROUP ANALYSIS")
    print("=" * 90)

    window = 20
    recent = data[-200:]  # last 200 spins for analysis

    # Table distribution overall
    t0_total = sum(1 for n in data if n in WHEEL_TABLE_0)
    t19_total = len(data) - t0_total
    print(f"\n  Overall Table Distribution ({len(data)} spins):")
    print(f"    0-table:  {t0_total} ({t0_total/len(data)*100:.1f}%) — expected 51.4% (19/37)")
    print(f"    19-table: {t19_total} ({t19_total/len(data)*100:.1f}%) — expected 48.6% (18/37)")

    # Polarity distribution overall
    pos_total = sum(1 for n in data if n in WHEEL_POSITIVE)
    neg_total = len(data) - pos_total
    print(f"\n  Overall Polarity Distribution:")
    print(f"    Positive: {pos_total} ({pos_total/len(data)*100:.1f}%) — expected 51.4%")
    print(f"    Negative: {neg_total} ({neg_total/len(data)*100:.1f}%) — expected 48.6%")

    # Set distribution overall
    s1_total = sum(1 for n in data if n in WHEEL_SET_1)
    s2_total = sum(1 for n in data if n in WHEEL_SET_2)
    s3_total = sum(1 for n in data if n in WHEEL_SET_3)
    print(f"\n  Overall Set Distribution:")
    print(f"    Set 1: {s1_total} ({s1_total/len(data)*100:.1f}%) — expected 32.4% (12/37)")
    print(f"    Set 2: {s2_total} ({s2_total/len(data)*100:.1f}%) — expected 32.4% (12/37)")
    print(f"    Set 3: {s3_total} ({s3_total/len(data)*100:.1f}%) — expected 35.1% (13/37)")

    # Trend prediction accuracy
    # For each spin i > warmup+window, check if the hot group from last 20 spins
    # correctly predicts which group the next spin falls in
    table_correct = 0
    polarity_correct = 0
    set_correct = 0
    total_checks = 0

    for i in range(warmup, len(data)):
        if i < window:
            continue
        recent_w = data[i - window:i]
        actual = data[i]
        total_checks += 1

        # Table prediction
        t0 = sum(1 for n in recent_w if _table_map.get(n) == '0')
        t19 = len(recent_w) - t0
        if t0 != t19:
            hot = '0' if t0 > t19 else '19'
            if _table_map.get(actual) == hot:
                table_correct += 1

        # Polarity prediction
        pos = sum(1 for n in recent_w if _polarity_map.get(n) == 'positive')
        neg = len(recent_w) - pos
        if pos != neg:
            hot = 'positive' if pos > neg else 'negative'
            if _polarity_map.get(actual) == hot:
                polarity_correct += 1

        # Set prediction
        sc = {1: 0, 2: 0, 3: 0}
        for n in recent_w:
            sc[_set_map.get(n, 3)] += 1
        hot_s = max(sc, key=sc.get)
        if _set_map.get(actual) == hot_s:
            set_correct += 1

    if total_checks > 0:
        print(f"\n  Trend Prediction Accuracy (window={window}, {total_checks} predictions):")
        print(f"    Table (0/19):    {table_correct}/{total_checks} = {table_correct/total_checks*100:.1f}%  "
              f"(baseline: ~51.4%)")
        print(f"    Polarity (+/-):  {polarity_correct}/{total_checks} = {polarity_correct/total_checks*100:.1f}%  "
              f"(baseline: ~51.4%)")
        print(f"    Set (1/2/3):     {set_correct}/{total_checks} = {set_correct/total_checks*100:.1f}%  "
              f"(baseline: ~35.1%)")


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print("BACKTEST — WHEEL STRATEGY (0/19 Tables, Positive/Negative, Sets 1/2/3)")
    print("Walk-Forward Validation on Real Userdata")
    print("$3 per number straight bet, 35:1 payout")
    print("=" * 90)

    data = load_all_userdata()
    warmup = 100
    n_preds = len(data) - warmup
    bet = 3.0

    print(f"\nData: {len(data)} spins from userdata/")
    print(f"Warmup: {warmup} spins | Predictions: {n_preds}")
    print(f"Bet: ${bet:.0f}/number | Starting bankroll: $4,000")

    start = time.time()
    all_results = []

    # Analyze group distributions
    analyze_wheel_groups(data, warmup)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: HEAD-TO-HEAD — OLD vs NEW ENSEMBLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("SECTION 1: HEAD-TO-HEAD COMPARISON (12 numbers)")
    print("=" * 90)

    strategies_12 = {
        'OLD: Freq(90%)+Pat(10%)': old_ensemble,
        'NEW: Freq(55%)+Wheel(30%)+Pat(5%)': new_ensemble,
        'Frequency Only (blend)': lambda h: frequency_blend(h, 0.5, 30),
        'Wheel Strategy Only': wheel_only,
        'Freq(50%)+Wheel(50%)': freq_wheel_5050,
        'Freq(70%)+Wheel(30%)': freq_wheel_7030,
        'Markov Only': lambda h: markov_probs(h, 0.6, 0.5),
        'Pattern Only': lambda h: pattern_probs(h, 50),
    }

    print(f"\n{'Strategy':<35} {'Hit%':>6} {'Base%':>6} {'Edge':>7} "
          f"{'ROI%':>7} {'Final$':>8} {'Profit':>8} {'MaxDD':>7} "
          f"{'Best50':>6} {'Wrst50':>6} {'WStr':>4} {'LStr':>4}")
    print("-" * 120)

    for name, fn in strategies_12.items():
        def make_strat(mfn=fn, n=12):
            def strat(history):
                if len(history) < 5:
                    return list(range(n))
                probs = mfn(history)
                return list(np.argsort(probs)[::-1][:n])
            return strat

        r = backtest(data, make_strat(), warmup, name, bet)
        if r:
            all_results.append(r)
            marker = " <<<" if name.startswith('NEW') else ""
            print(f"{name:<35} {r['hit_rate']:>6.1f} {r['baseline']:>6.1f} "
                  f"{r['edge']:>+7.2f} {r['roi']:>7.1f} {r['final_bankroll']:>8.0f} "
                  f"{r['profit']:>+8.0f} {r['max_drawdown']:>7.0f} "
                  f"{r['best_50_rate']:>6.1f} {r['worst_50_rate']:>6.1f} "
                  f"{r['max_win_streak']:>4} {r['max_loss_streak']:>4}{marker}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: WHEEL STRATEGY PARAMETER SWEEP
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("SECTION 2: WHEEL STRATEGY PARAMETER SWEEP (12 numbers)")
    print("=" * 90)

    print(f"\n{'Window':>6} {'Boost':>6} {'Damp':>6} | "
          f"{'Hit%':>6} {'Base%':>6} {'Edge':>7} {'ROI%':>7} {'Profit':>8} {'Final$':>8}")
    print("-" * 90)

    for window in [10, 15, 20, 25, 30, 40, 50]:
        for boost in [1.20, 1.30, 1.40, 1.50, 1.60]:
            dampen = 1.0 - (boost - 1.0)  # dampen mirrors boost

            def make_ws(w=window, b=boost, d=dampen):
                def strat(history):
                    if len(history) < 5:
                        return list(range(12))
                    # Blend: 55% freq + 30% wheel + 5% pattern (normalized to 100%)
                    fp = frequency_blend(history, 0.5, 30)
                    wp = wheel_strategy_probs(history, w, b, d)
                    pp = pattern_probs(history, 50)
                    ens = 0.611 * fp + 0.333 * wp + 0.056 * pp
                    ens /= ens.sum()
                    return list(np.argsort(ens)[::-1][:12])
                return strat

            r = backtest(data, make_ws(), warmup, f"ws_w{window}_b{boost}_d{dampen:.2f}", bet)
            if r:
                all_results.append(r)
                print(f"{window:>6} {boost:>6.2f} {dampen:>6.2f} | "
                      f"{r['hit_rate']:>6.1f} {r['baseline']:>6.1f} {r['edge']:>+7.2f} "
                      f"{r['roi']:>7.1f} {r['profit']:>+8.0f} {r['final_bankroll']:>8.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: WEIGHT SWEEP — Freq vs Wheel vs Pattern
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("SECTION 3: WEIGHT SWEEP — Finding optimal Freq/Wheel/Pattern balance")
    print("=" * 90)

    print(f"\n{'Freq%':>5} {'Wheel%':>6} {'Pat%':>5} | "
          f"{'Hit%':>6} {'Base%':>6} {'Edge':>7} {'ROI%':>7} {'Profit':>8}")
    print("-" * 80)

    weight_results = []
    for fw in [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        for ww in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]:
            pw = 1.0 - fw - ww
            if pw < 0 or pw > 0.30:
                continue

            def make_weighted(f=fw, w=ww, p=pw):
                def strat(history):
                    if len(history) < 5:
                        return list(range(12))
                    fp = frequency_blend(history, 0.5, 30)
                    wp = wheel_strategy_probs(history, 20, 1.40, 0.70)
                    pp = pattern_probs(history, 50)
                    ens = f * fp + w * wp + p * pp
                    ens /= ens.sum()
                    return list(np.argsort(ens)[::-1][:12])
                return strat

            r = backtest(data, make_weighted(), warmup,
                         f"F{fw:.0%}+W{ww:.0%}+P{pw:.0%}", bet)
            if r:
                weight_results.append(r)
                all_results.append(r)
                print(f"{fw*100:>5.0f} {ww*100:>6.0f} {pw*100:>5.0f} | "
                      f"{r['hit_rate']:>6.1f} {r['baseline']:>6.1f} {r['edge']:>+7.2f} "
                      f"{r['roi']:>7.1f} {r['profit']:>+8.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: DIFFERENT NUMBER COUNTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("SECTION 4: NUMBER COUNT COMPARISON (New Ensemble)")
    print("=" * 90)

    print(f"\n{'#Nums':>5} {'Brkevn':>6} {'Hit%':>6} {'Base%':>6} {'Edge':>7} "
          f"{'ROI%':>7} {'Cost/Spin':>9} {'Profit':>8} {'Final$':>8}")
    print("-" * 80)

    for n_nums in [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]:
        breakeven = n_nums / 36 * 100

        def make_n(n=n_nums):
            def strat(history):
                if len(history) < 5:
                    return list(range(n))
                probs = new_ensemble(history)
                return list(np.argsort(probs)[::-1][:n])
            return strat

        r = backtest(data, make_n(), warmup, f"New-Ens top-{n_nums}", bet)
        if r:
            all_results.append(r)
            cost_per_spin = n_nums * bet
            print(f"{n_nums:>5} {breakeven:>6.1f} {r['hit_rate']:>6.1f} {r['baseline']:>6.1f} "
                  f"{r['edge']:>+7.2f} {r['roi']:>7.1f} ${cost_per_spin:>8.0f} "
                  f"{r['profit']:>+8.0f} {r['final_bankroll']:>8.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: PER-DATASET BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("SECTION 5: PER-DATASET BREAKDOWN (New Ensemble, 12 numbers)")
    print("=" * 90)

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

        if len(file_data) > 50:
            def make_file_strat_old():
                return lambda h: list(np.argsort(old_ensemble(h))[::-1][:12]) \
                    if len(h) >= 20 else list(range(12))

            def make_file_strat_new():
                return lambda h: list(np.argsort(new_ensemble(h))[::-1][:12]) \
                    if len(h) >= 5 else list(range(12))

            w = min(50, len(file_data) // 2)
            r_old = backtest(file_data, make_file_strat_old(), w, f"{fname} (OLD)", bet)
            r_new = backtest(file_data, make_file_strat_new(), w, f"{fname} (NEW)", bet)
            if r_old and r_new:
                improvement = r_new['edge'] - r_old['edge']
                print(f"  {fname:<12} ({len(file_data):>3} spins)")
                print(f"    OLD: hit={r_old['hit_rate']:>5.1f}% edge={r_old['edge']:>+5.2f}% profit=${r_old['profit']:>+7.0f}")
                print(f"    NEW: hit={r_new['hit_rate']:>5.1f}% edge={r_new['edge']:>+5.2f}% profit=${r_new['profit']:>+7.0f}  "
                      f"{'BETTER' if improvement > 0 else 'WORSE'} by {abs(improvement):.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start

    print("\n\n" + "=" * 90)
    print("GRAND SUMMARY — TOP 15 STRATEGIES BY EDGE")
    print("=" * 90)

    all_results.sort(key=lambda x: x['edge'], reverse=True)
    print(f"\n{'#':>3} {'Strategy':<40} {'#N':>3} {'Hit%':>6} {'Base%':>6} "
          f"{'Edge':>7} {'ROI%':>7} {'Profit':>8} {'Final$':>8}")
    print("-" * 100)
    for i, r in enumerate(all_results[:15]):
        print(f"{i+1:>3} {r['label']:<40} {r['avg_numbers']:>3.0f} {r['hit_rate']:>6.1f} "
              f"{r['baseline']:>6.1f} {r['edge']:>+7.2f} {r['roi']:>7.1f} "
              f"{r['profit']:>+8.0f} {r['final_bankroll']:>8.0f}")

    print(f"\n--- BOTTOM 5 ---")
    for r in all_results[-5:]:
        print(f"    {r['label']:<40} {r['avg_numbers']:>3.0f} hit={r['hit_rate']:>5.1f}% "
              f"edge={r['edge']:>+6.2f}%")

    print(f"\nBacktest time: {elapsed:.0f}s | Strategies tested: {len(all_results)}")

    # Key comparison
    old_r = next((r for r in all_results if 'OLD' in r['label'] and 'top' not in r['label']), None)
    new_r = next((r for r in all_results if 'NEW' in r['label'] and 'top' not in r['label']), None)
    if old_r and new_r:
        print(f"\n{'='*90}")
        print("KEY RESULT: OLD vs NEW ENSEMBLE (12 numbers, $3/number)")
        print(f"{'='*90}")
        print(f"  OLD (Freq 90% + Pattern 10%):")
        print(f"    Hit: {old_r['hit_rate']}%  Edge: {old_r['edge']:+.2f}%  "
              f"Profit: ${old_r['profit']:+.0f}  Final: ${old_r['final_bankroll']:.0f}")
        print(f"  NEW (Freq 55% + Wheel 30% + Pattern 5%):")
        print(f"    Hit: {new_r['hit_rate']}%  Edge: {new_r['edge']:+.2f}%  "
              f"Profit: ${new_r['profit']:+.0f}  Final: ${new_r['final_bankroll']:.0f}")
        diff = new_r['edge'] - old_r['edge']
        print(f"  DIFFERENCE: {diff:+.2f}% edge | ${new_r['profit'] - old_r['profit']:+.0f} profit")


if __name__ == '__main__':
    main()
