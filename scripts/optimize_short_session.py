#!/usr/bin/env python3
"""
Optimize for SHORT sessions: 100-200 spins (~30min to 1hr at table).

Goal: +$100 from $4,000 bankroll in 100-200 spins.
Key: higher bet sizes needed since fewer spins to accumulate edge.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from session_optimizer import (
    EnsembleModel, FlatBet, ProgressiveBet, PercentBet, TargetBet,
    simulate_session, load_userdata
)
from config import TOTAL_NUMBERS


def main():
    print("=" * 100)
    print("SHORT SESSION OPTIMIZER — 100-200 spins (~30min to 1hr)")
    print("Goal: +$100 from $4,000 bankroll")
    print("=" * 100)

    all_spins = load_userdata()
    print(f"Data: {len(all_spins)} spins")

    import time
    t0 = time.time()

    # The winning model config from optimizer
    best_model = {
        'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
        'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
        'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                       'hot_threshold': 1.5, 'neighbour_boost': 1.15},
        'weights': {'freq': 0.60, 'markov': 0.30, 'pattern': 0.10},
        'diversify_after': 3, 'exploration_factor': 0.3,
    }

    # For 100-200 spins, we need higher bets to reach $100 target
    # Math: 150 spins × 14 numbers × $X/num × 1.5% edge ≈ $100
    #   → $X ≈ $100 / (150 × 14 × 0.015) = ~$3.17/number
    #   But variance is high, so test range $2-$8

    warmup = 100

    session_lengths = [100, 120, 150, 180, 200]
    num_strategies = {
        'top-10': ('top', 10),
        'top-12': ('top', 12),
        'top-14': ('top', 14),
    }
    bet_strategies = {}

    # Build bet strategies for each session length
    for ses_len in session_lengths:
        bet_strategies[f'Flat$2_s{ses_len}'] = (FlatBet(2.0), ses_len)
        bet_strategies[f'Flat$3_s{ses_len}'] = (FlatBet(3.0), ses_len)
        bet_strategies[f'Flat$4_s{ses_len}'] = (FlatBet(4.0), ses_len)
        bet_strategies[f'Flat$5_s{ses_len}'] = (FlatBet(5.0), ses_len)
        bet_strategies[f'Flat$6_s{ses_len}'] = (FlatBet(6.0), ses_len)
        bet_strategies[f'Flat$8_s{ses_len}'] = (FlatBet(8.0), ses_len)
        bet_strategies[f'Target_s{ses_len}'] = (
            TargetBet(target=100, session_length=ses_len, max_bet=15.0,
                     num_count=14, edge_estimate=0.015), ses_len)
        bet_strategies[f'Target-hi_s{ses_len}'] = (
            TargetBet(target=100, session_length=ses_len, max_bet=20.0,
                     num_count=14, edge_estimate=0.015), ses_len)
        bet_strategies[f'Prog_s{ses_len}'] = (
            ProgressiveBet(base=3.0, increment=1.0, decrement=1.0,
                          loss_trigger=3, win_trigger=2, max_bet=12.0), ses_len)
        bet_strategies[f'Prog-hi_s{ses_len}'] = (
            ProgressiveBet(base=4.0, increment=2.0, decrement=1.0,
                          loss_trigger=3, win_trigger=2, max_bet=15.0), ses_len)
        bet_strategies[f'Pct-0.2_s{ses_len}'] = (PercentBet(pct=0.002, max_bet=15.0), ses_len)
        bet_strategies[f'Pct-0.3_s{ses_len}'] = (PercentBet(pct=0.003, max_bet=20.0), ses_len)

    results = []
    total = len(num_strategies) * len(bet_strategies)
    tested = 0

    for num_name, num_strat in num_strategies.items():
        for bet_name, (bet_strat_template, ses_len) in bet_strategies.items():
            tested += 1

            # Run non-overlapping sessions
            sessions = []
            start = warmup
            while start + ses_len <= len(all_spins):
                # Reset bet strategy state
                if isinstance(bet_strat_template, ProgressiveBet):
                    bet_strat = ProgressiveBet(
                        base=bet_strat_template.base,
                        increment=bet_strat_template.increment,
                        decrement=bet_strat_template.decrement,
                        loss_trigger=bet_strat_template.loss_trigger,
                        win_trigger=bet_strat_template.win_trigger,
                        max_bet=bet_strat_template.max_bet
                    )
                elif isinstance(bet_strat_template, TargetBet):
                    bet_strat = TargetBet(
                        target=bet_strat_template.target,
                        session_length=bet_strat_template.session_length,
                        max_bet=bet_strat_template.max_bet,
                        num_count=bet_strat_template.num_count,
                        edge_estimate=bet_strat_template.edge_estimate
                    )
                elif isinstance(bet_strat_template, PercentBet):
                    bet_strat = PercentBet(
                        pct=bet_strat_template.pct,
                        max_bet=bet_strat_template.max_bet
                    )
                else:
                    bet_strat = bet_strat_template

                result = simulate_session(
                    all_spins, start, ses_len, best_model,
                    num_strat, bet_strat, warmup=warmup
                )
                if result:
                    sessions.append(result)
                start += ses_len  # non-overlapping

            if not sessions:
                continue

            profits = [s['profit'] for s in sessions]
            drawdowns = [s['max_drawdown'] for s in sessions]
            targets = [s['target_hit'] for s in sessions]
            hit_rates = [s['hit_rate'] for s in sessions]

            results.append({
                'nums': num_name,
                'bet': bet_name.split('_s')[0],
                'session_len': ses_len,
                'n_sessions': len(sessions),
                'avg_profit': round(np.mean(profits), 1),
                'median_profit': round(np.median(profits), 1),
                'min_profit': round(min(profits), 1),
                'max_profit': round(max(profits), 1),
                'pct_profitable': round(sum(1 for p in profits if p > 0) / len(profits) * 100, 1),
                'pct_target': round(sum(1 for t in targets if t) / len(targets) * 100, 1),
                'avg_hit_rate': round(np.mean(hit_rates), 1),
                'max_drawdown': round(max(drawdowns), 1),
                'avg_wagered': round(np.mean([s['total_wagered'] for s in sessions]), 0),
            })

    # Sort by highest % target hit, then by avg profit
    results.sort(key=lambda x: (x['pct_target'], x['avg_profit']), reverse=True)

    elapsed = time.time() - t0
    print(f"\nOptimization time: {elapsed:.0f}s")
    print(f"Strategies tested: {len(results)}")

    # ─── Results by session length ───────────────────────────────────
    for ses_len in session_lengths:
        subset = [r for r in results if r['session_len'] == ses_len]
        if not subset:
            continue

        print(f"\n{'='*120}")
        print(f"SESSION LENGTH: {ses_len} spins (~{ses_len//3}-{ses_len//2} minutes)")
        print(f"{'='*120}")
        print(f"{'#':>3} {'Nums':<8} {'Bet':<12} {'#Ses':>4} {'AvgP':>8} {'MedP':>8} "
              f"{'MinP':>8} {'MaxP':>8} {'%Prof':>6} {'%$100':>6} {'AvgHR':>6} "
              f"{'MaxDD':>7} {'AvgWag':>8}")
        print("-" * 120)

        subset.sort(key=lambda x: (x['pct_target'], x['avg_profit']), reverse=True)
        for i, r in enumerate(subset[:20]):
            print(f"{i+1:>3} {r['nums']:<8} {r['bet']:<12} {r['n_sessions']:>4} "
                  f"{r['avg_profit']:>8.1f} {r['median_profit']:>8.1f} "
                  f"{r['min_profit']:>8.1f} {r['max_profit']:>8.1f} "
                  f"{r['pct_profitable']:>5.1f}% {r['pct_target']:>5.1f}% "
                  f"{r['avg_hit_rate']:>5.1f}% {r['max_drawdown']:>7.1f} "
                  f"{r['avg_wagered']:>8.0f}")

    # ─── Overall best for ~150 spins (30min session) ─────────────────
    target_len = 150
    best_150 = [r for r in results if r['session_len'] == target_len]
    if best_150:
        best_150.sort(key=lambda x: (x['pct_target'], x['avg_profit']), reverse=True)
        b = best_150[0]
        print(f"\n{'='*100}")
        print(f"★ BEST FOR 30-MINUTE SESSION ({target_len} spins)")
        print(f"{'='*100}")
        print(f"  Numbers: {b['nums']}")
        print(f"  Bet: {b['bet']}")
        print(f"  Avg profit: ${b['avg_profit']:.1f}")
        print(f"  % hit $100: {b['pct_target']:.1f}%")
        print(f"  % profitable: {b['pct_profitable']:.1f}%")
        print(f"  Max drawdown: ${b['max_drawdown']:.1f}")
        print(f"  Avg wagered: ${b['avg_wagered']:.0f}")

    # ─── Overall best for ~200 spins (1hr session) ───────────────────
    target_len = 200
    best_200 = [r for r in results if r['session_len'] == target_len]
    if best_200:
        best_200.sort(key=lambda x: (x['pct_target'], x['avg_profit']), reverse=True)
        b = best_200[0]
        print(f"\n{'='*100}")
        print(f"★ BEST FOR 1-HOUR SESSION ({target_len} spins)")
        print(f"{'='*100}")
        print(f"  Numbers: {b['nums']}")
        print(f"  Bet: {b['bet']}")
        print(f"  Avg profit: ${b['avg_profit']:.1f}")
        print(f"  % hit $100: {b['pct_target']:.1f}%")
        print(f"  % profitable: {b['pct_profitable']:.1f}%")
        print(f"  Max drawdown: ${b['max_drawdown']:.1f}")
        print(f"  Avg wagered: ${b['avg_wagered']:.0f}")

    # ─── Bet recommendation ──────────────────────────────────────────
    print(f"\n{'='*100}")
    print("BET SIZING RECOMMENDATION FOR $100 TARGET")
    print(f"{'='*100}")
    print("""
  For 100-200 spin sessions, the bet per number must be higher than $1.

  Math: To earn $100 in 150 spins with 14 numbers and 1.5% edge:
    - Expected profit per spin = bet × 14 × [(14/37)×(1.015)×36 - 14] / 14
    - Simplified: each $1 bet on 14 numbers earns ~$0.33/spin edge
    - Need ~$100/150 = $0.67/spin → bet ~$2/number
    - But with variance, higher bets hit target faster

  Recommended: $3-$5 per number (adjustable by user in UI)
    - $3/number × 14 numbers = $42/spin risk → ~$1/spin edge → $150 in 150 spins
    - $5/number × 14 numbers = $70/spin risk → ~$1.67/spin edge → $250 in 150 spins
    - Higher variance but faster to target
""")


if __name__ == '__main__':
    main()
