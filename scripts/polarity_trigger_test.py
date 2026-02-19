#!/usr/bin/env python3
"""
Polarity Trigger Strategies — Test on data6.txt

Three strategies tested:
  A) REGULAR: Polarity(20%)+Freq(80%), bet every spin, pick top-12
  B) SINGLE TRIGGER: Only bet when last spin was polarity X,
     then pick top-12 from ONLY that polarity group
  C) DOUBLE TRIGGER: Only bet when last 2 spins were SAME polarity,
     then pick top-12 from ONLY that polarity group

All use Polarity(20%)+Freq(80%) as the base probability engine.
"""

import sys
import os
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    TOTAL_NUMBERS, WHEEL_POSITIVE, WHEEL_NEGATIVE,
    FREQUENCY_RECENT_WINDOW,
    WHEEL_STRATEGY_TREND_BOOST, WHEEL_STRATEGY_COLD_DAMPEN,
)

TOP_N = 12
WARMUP = 60  # smaller warmup for single file
BET_PER_NUM = 2.0
PAYOUT = 35
BOOST = WHEEL_STRATEGY_TREND_BOOST
DAMPEN = WHEEL_STRATEGY_COLD_DAMPEN

# Polarity membership
POL_MAP = {}
for _n in range(TOTAL_NUMBERS):
    POL_MAP[_n] = 'pos' if _n in WHEEL_POSITIVE else 'neg'

POS_NUMS = sorted(WHEEL_POSITIVE)
NEG_NUMS = sorted(WHEEL_NEGATIVE)


def load_file(fpath):
    spins = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                n = int(line)
                if 0 <= n <= 36:
                    spins.append(n)
    return spins


def frequency_probs(history):
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


def polarity_probs(history, window=50):
    probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
    if len(history) < 5:
        return probs
    recent = history[-window:]
    total = len(recent)
    pos = sum(1 for n in recent if POL_MAP.get(n) == 'pos')
    neg = total - pos
    hot = 'pos' if pos > neg else 'neg' if neg > pos else None
    if hot is not None:
        imbalance = abs(pos / total - neg / total) / 0.5
        pb = 1.0 + (BOOST - 1.0) * imbalance
        pd = 1.0 - (1.0 - DAMPEN) * imbalance
        for num in range(TOTAL_NUMBERS):
            if POL_MAP[num] == hot:
                probs[num] *= pb
            else:
                probs[num] *= pd
    return probs / probs.sum()


def ensemble_probs(history):
    """Polarity(20%) + Freq(80%)"""
    p = 0.20 * polarity_probs(history) + 0.80 * frequency_probs(history)
    return p / p.sum()


def run_backtest(data, strategy, warmup=60, label=""):
    """
    strategy: 'regular', 'single_trigger', 'double_trigger'
    """
    if len(data) <= warmup + 10:
        return None

    hits = 0
    total_bets = 0
    skipped = 0
    bankroll = 4000.0
    peak = 4000.0
    max_dd = 0
    hit_log = []
    streak_hits = []  # track hits within betting windows

    # Track consecutive wins/losses for the strategy
    consec_wins = 0
    consec_losses = 0
    max_consec_wins = 0
    max_consec_losses = 0

    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]

        # ── Determine whether to bet and which numbers ──
        if strategy == 'regular':
            # Always bet, pick top-12 from full ensemble
            probs = ensemble_probs(history)
            top = set(np.argsort(probs)[::-1][:TOP_N])
            should_bet = True

        elif strategy == 'single_trigger':
            # Only bet if last spin was from a polarity group
            # Then bet on top-12 from THAT SAME polarity group
            last_spin = data[i - 1]
            last_pol = POL_MAP[last_spin]
            should_bet = True  # always triggers since every number has a polarity

            # Get full ensemble probs
            probs = ensemble_probs(history)

            # Filter: only pick from the same polarity as last spin
            group_nums = POS_NUMS if last_pol == 'pos' else NEG_NUMS
            # Get top-12 from within this group only
            group_probs = [(n, probs[n]) for n in group_nums]
            group_probs.sort(key=lambda x: x[1], reverse=True)
            top = set(n for n, p in group_probs[:TOP_N])

        elif strategy == 'double_trigger':
            # Only bet if last 2 spins are from SAME polarity
            if i < 2:
                skipped += 1
                continue
            last1 = POL_MAP[data[i - 1]]
            last2 = POL_MAP[data[i - 2]]

            if last1 != last2:
                skipped += 1
                continue

            should_bet = True
            trigger_pol = last1

            # Get full ensemble probs
            probs = ensemble_probs(history)

            # Filter: only pick from the triggered polarity group
            group_nums = POS_NUMS if trigger_pol == 'pos' else NEG_NUMS
            group_probs = [(n, probs[n]) for n in group_nums]
            group_probs.sort(key=lambda x: x[1], reverse=True)
            top = set(n for n, p in group_probs[:TOP_N])

        if not should_bet:
            skipped += 1
            continue

        # ── Score the bet ──
        hit = 1 if actual in top else 0
        hits += hit
        total_bets += 1
        hit_log.append(hit)

        cost = TOP_N * BET_PER_NUM
        if hit:
            bankroll += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
            consec_wins += 1
            consec_losses = 0
            max_consec_wins = max(max_consec_wins, consec_wins)
        else:
            bankroll -= cost
            consec_losses += 1
            consec_wins = 0
            max_consec_losses = max(max_consec_losses, consec_losses)

        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

    if total_bets == 0:
        return None

    hit_rate = hits / total_bets * 100
    baseline = TOP_N / 37 * 100
    edge = hit_rate - baseline
    profit = hits * (PAYOUT + 1) * BET_PER_NUM - total_bets * TOP_N * BET_PER_NUM

    # Rolling 30-spin windows (smaller for single file)
    rolling = []
    win_size = min(30, total_bets // 3) if total_bets > 30 else total_bets
    for j in range(win_size, len(hit_log)):
        rolling.append(sum(hit_log[j - win_size:j]) / win_size * 100)

    return {
        'label': label,
        'bets': total_bets,
        'skipped': skipped,
        'hits': hits,
        'hit_rate': round(hit_rate, 2),
        'baseline': round(baseline, 2),
        'edge': round(edge, 2),
        'profit': round(profit, 2),
        'final': round(bankroll, 2),
        'max_dd': round(max_dd, 2),
        'best_roll': round(max(rolling), 1) if rolling else 0,
        'worst_roll': round(min(rolling), 1) if rolling else 0,
        'max_consec_wins': max_consec_wins,
        'max_consec_losses': max_consec_losses,
        'bet_pct': round(total_bets / (len(data) - warmup) * 100, 1),
    }


def pr(r):
    if not r:
        print("    (insufficient data)")
        return
    print(f"    {r['label']}")
    print(f"      Bets: {r['bets']} / {r['bets'] + r['skipped']} spins "
          f"({r['bet_pct']}% of spins)")
    print(f"      Hits: {r['hits']} / {r['bets']} = {r['hit_rate']}%  "
          f"(baseline {r['baseline']}%)")
    print(f"      Edge: {r['edge']:+.2f}%")
    print(f"      Profit: ${r['profit']:+.0f}  |  Final bankroll: ${r['final']:,.0f}")
    print(f"      Max drawdown: ${r['max_dd']:,.0f}")
    print(f"      Max consecutive wins: {r['max_consec_wins']}  |  "
          f"Max consecutive losses: {r['max_consec_losses']}")
    print(f"      Rolling {30}-spin best: {r['best_roll']}%  |  "
          f"worst: {r['worst_roll']}%")
    print()


def main():
    data_file = os.path.join(BASE_DIR, 'userdata', 'data6.txt')
    data = load_file(data_file)

    print("=" * 90)
    print("POLARITY TRIGGER STRATEGIES — data6.txt")
    print(f"Total spins: {len(data)} | Warmup: {WARMUP} | Top-{TOP_N} | ${BET_PER_NUM}/num | {PAYOUT}:1")
    print("=" * 90)

    # Show polarity distribution in data6
    pos_count = sum(1 for n in data if POL_MAP[n] == 'pos')
    neg_count = len(data) - pos_count
    print(f"\nPolarity distribution in data6:")
    print(f"  Positive: {pos_count} ({pos_count/len(data)*100:.1f}%) — {len(WHEEL_POSITIVE)} numbers: {sorted(WHEEL_POSITIVE)}")
    print(f"  Negative: {neg_count} ({neg_count/len(data)*100:.1f}%) — {len(WHEEL_NEGATIVE)} numbers: {sorted(WHEEL_NEGATIVE)}")

    # Count double triggers
    same_pol_pairs = 0
    for i in range(1, len(data)):
        if POL_MAP[data[i-1]] == POL_MAP[data[i]]:
            same_pol_pairs += 1
    print(f"  Consecutive same-polarity pairs: {same_pol_pairs}/{len(data)-1} "
          f"({same_pol_pairs/(len(data)-1)*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY A: Regular
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("STRATEGY A: REGULAR — Polarity(20%)+Freq(80%), bet every spin, top-12")
    print("=" * 90)
    r_regular = run_backtest(data, 'regular', WARMUP, "Regular Pol(20%)+Freq(80%)")
    pr(r_regular)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY B: Single Trigger
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("STRATEGY B: SINGLE TRIGGER — Last spin was polarity X → bet top-12 from polarity X")
    print("  (Since every number belongs to pos or neg, this ALWAYS triggers but")
    print("   RESTRICTS picks to only the matching polarity group)")
    print("=" * 90)
    r_single = run_backtest(data, 'single_trigger', WARMUP, "Single trigger (follow last polarity)")
    pr(r_single)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY C: Double Trigger
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("STRATEGY C: DOUBLE TRIGGER — Last 2 spins SAME polarity → bet top-12 from that polarity")
    print("  (Only bets when 2 consecutive spins come from the same group)")
    print("=" * 90)
    r_double = run_backtest(data, 'double_trigger', WARMUP, "Double trigger (2 same polarity → follow)")
    pr(r_double)

    # ═══════════════════════════════════════════════════════════════════
    # Also test: OPPOSITE polarity triggers (bet AGAINST the streak)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("BONUS STRATEGIES — Bet on OPPOSITE polarity (contrarian)")
    print("=" * 90)

    # Single opposite: last spin polarity X → bet on opposite polarity
    # Need a custom backtest for this
    hits_opp_single = 0; bets_opp_single = 0; skipped_opp_s = 0
    bankroll_os = 4000.0; peak_os = 4000.0; max_dd_os = 0
    hit_log_os = []; cw_os = 0; cl_os = 0; mcw_os = 0; mcl_os = 0

    for i in range(WARMUP, len(data)):
        history = data[:i]
        actual = data[i]
        last_pol = POL_MAP[data[i - 1]]
        opp_pol = 'neg' if last_pol == 'pos' else 'pos'

        probs = ensemble_probs(history)
        group_nums = POS_NUMS if opp_pol == 'pos' else NEG_NUMS
        group_probs = [(n, probs[n]) for n in group_nums]
        group_probs.sort(key=lambda x: x[1], reverse=True)
        top = set(n for n, p in group_probs[:TOP_N])

        hit = 1 if actual in top else 0
        hits_opp_single += hit; bets_opp_single += 1; hit_log_os.append(hit)
        cost = TOP_N * BET_PER_NUM
        if hit:
            bankroll_os += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
            cw_os += 1; cl_os = 0; mcw_os = max(mcw_os, cw_os)
        else:
            bankroll_os -= cost
            cl_os += 1; cw_os = 0; mcl_os = max(mcl_os, cl_os)
        peak_os = max(peak_os, bankroll_os)
        max_dd_os = max(max_dd_os, peak_os - bankroll_os)

    if bets_opp_single > 0:
        hr_os = hits_opp_single / bets_opp_single * 100
        edge_os = hr_os - TOP_N / 37 * 100
        profit_os = hits_opp_single * (PAYOUT+1) * BET_PER_NUM - bets_opp_single * TOP_N * BET_PER_NUM
        print(f"\n    Single OPPOSITE trigger (last polarity X → bet opposite)")
        print(f"      Bets: {bets_opp_single}  |  Hit: {hr_os:.1f}%  |  Edge: {edge_os:+.2f}%")
        print(f"      Profit: ${profit_os:+.0f}  |  Max DD: ${max_dd_os:,.0f}")
        print(f"      Max consec wins: {mcw_os}  |  Max consec losses: {mcl_os}")

    # Double opposite: 2 same polarity → bet OPPOSITE
    hits_opp_double = 0; bets_opp_double = 0; skipped_od = 0
    bankroll_od = 4000.0; peak_od = 4000.0; max_dd_od = 0
    hit_log_od = []; cw_od = 0; cl_od = 0; mcw_od = 0; mcl_od = 0

    for i in range(WARMUP, len(data)):
        if i < 2:
            skipped_od += 1
            continue
        history = data[:i]
        actual = data[i]
        last1 = POL_MAP[data[i - 1]]
        last2 = POL_MAP[data[i - 2]]
        if last1 != last2:
            skipped_od += 1
            continue

        opp_pol = 'neg' if last1 == 'pos' else 'pos'
        probs = ensemble_probs(history)
        group_nums = POS_NUMS if opp_pol == 'pos' else NEG_NUMS
        group_probs = [(n, probs[n]) for n in group_nums]
        group_probs.sort(key=lambda x: x[1], reverse=True)
        top = set(n for n, p in group_probs[:TOP_N])

        hit = 1 if actual in top else 0
        hits_opp_double += hit; bets_opp_double += 1; hit_log_od.append(hit)
        cost = TOP_N * BET_PER_NUM
        if hit:
            bankroll_od += PAYOUT * BET_PER_NUM + BET_PER_NUM - cost
            cw_od += 1; cl_od = 0; mcw_od = max(mcw_od, cw_od)
        else:
            bankroll_od -= cost
            cl_od += 1; cw_od = 0; mcl_od = max(mcl_od, cl_od)
        peak_od = max(peak_od, bankroll_od)
        max_dd_od = max(max_dd_od, peak_od - bankroll_od)

    if bets_opp_double > 0:
        hr_od = hits_opp_double / bets_opp_double * 100
        edge_od = hr_od - TOP_N / 37 * 100
        profit_od = hits_opp_double * (PAYOUT+1) * BET_PER_NUM - bets_opp_double * TOP_N * BET_PER_NUM
        bet_pct_od = bets_opp_double / (len(data) - WARMUP) * 100
        print(f"\n    Double OPPOSITE trigger (2 same polarity → bet opposite)")
        print(f"      Bets: {bets_opp_double}/{bets_opp_double + skipped_od} spins ({bet_pct_od:.1f}%)")
        print(f"      Hit: {hr_od:.1f}%  |  Edge: {edge_od:+.2f}%")
        print(f"      Profit: ${profit_od:+.0f}  |  Max DD: ${max_dd_od:,.0f}")
        print(f"      Max consec wins: {mcw_od}  |  Max consec losses: {mcl_od}")

    # ═══════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'Strategy':<50} {'Bets':>5} {'Hit%':>6} {'Edge':>7} {'Profit':>8} {'MaxDD':>7}")
    print("  " + "-" * 85)

    rows = []
    if r_regular:
        rows.append((r_regular['label'], r_regular['bets'], r_regular['hit_rate'],
                      r_regular['edge'], r_regular['profit'], r_regular['max_dd']))
    if r_single:
        rows.append((r_single['label'], r_single['bets'], r_single['hit_rate'],
                      r_single['edge'], r_single['profit'], r_single['max_dd']))
    if r_double:
        rows.append((r_double['label'], r_double['bets'], r_double['hit_rate'],
                      r_double['edge'], r_double['profit'], r_double['max_dd']))
    if bets_opp_single > 0:
        rows.append(("Single OPPOSITE (bet against last)", bets_opp_single,
                      round(hr_os, 2), round(edge_os, 2), round(profit_os, 2), round(max_dd_os, 2)))
    if bets_opp_double > 0:
        rows.append(("Double OPPOSITE (2 same → bet against)", bets_opp_double,
                      round(hr_od, 2), round(edge_od, 2), round(profit_od, 2), round(max_dd_od, 2)))

    for label, bets, hr, edge, profit, dd in rows:
        marker = " ◀ BEST" if profit == max(r[4] for r in rows) else ""
        print(f"  {label:<50} {bets:>5} {hr:>6.1f} {edge:>+7.2f} ${profit:>+7.0f} ${dd:>6.0f}{marker}")

    print(f"\n  Baseline (random 12/37): {TOP_N/37*100:.2f}%")
    print(f"  Target: 50.00% hit rate")


if __name__ == '__main__':
    main()
