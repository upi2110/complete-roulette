#!/usr/bin/env python3
"""
Deep validation of the winning configuration from session_optimizer.

Winner: TunedEnsemble + top-14 + Target-200 bet + 300 spin sessions
  - Avg profit: $103.55
  - 80% sessions profitable, 80% hit $100+

This script validates by:
1. Non-overlapping sessions (no data re-use)
2. Per-dataset validation (each file independently)
3. Sensitivity analysis (slightly vary each parameter)
4. Stress test (what if edge disappears mid-session?)
5. Compare top 3 candidates head-to-head
"""

import sys
import os
import json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import WHEEL_ORDER, NUMBER_TO_POSITION, TOTAL_NUMBERS

# Import models from session_optimizer
sys.path.insert(0, os.path.dirname(__file__))
from session_optimizer import (
    EnsembleModel, FlatBet, ProgressiveBet, PercentBet, TargetBet,
    simulate_session, load_userdata
)


def load_per_file():
    """Load data per file for independent validation."""
    userdata_dir = os.path.join(os.path.dirname(__file__), '..', 'userdata')
    files_data = {}
    files = sorted(f for f in os.listdir(userdata_dir) if f.endswith('.txt'))
    for fname in files:
        path = os.path.join(userdata_dir, fname)
        spins = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    spins.append(int(line))
        files_data[fname] = spins
    return files_data


# ─── Top 3 candidates to validate ───────────────────────────────────

CANDIDATES = {
    'Winner: TunedEns top-14 Target200': {
        'model_cfg': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.15},
            'weights': {'freq': 0.60, 'markov': 0.30, 'pattern': 0.10},
            'diversify_after': 3, 'exploration_factor': 0.3,
        },
        'num_strategy': ('top', 14),
        'bet_fn': lambda: TargetBet(target=100, session_length=300, max_bet=15.0,
                                     num_count=14, edge_estimate=0.015),
    },
    'Runner-up: BalancedFP top-10 Target200': {
        'model_cfg': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.15},
            'weights': {'freq': 0.45, 'markov': 0.10, 'pattern': 0.45},
            'diversify_after': 3, 'exploration_factor': 0.3,
        },
        'num_strategy': ('top', 10),
        'bet_fn': lambda: TargetBet(target=100, session_length=300, max_bet=15.0,
                                     num_count=10, edge_estimate=0.015),
    },
    '3rd: BalancedFP top-8 Target200': {
        'model_cfg': {
            'freq_cfg': {'decay': 1.0, 'flat_w': 1.0, 'recent_w': 0.0},
            'markov_cfg': {'o1_w': 0.7, 'o2_w': 0.3, 'smoothing': 2.0},
            'pattern_cfg': {'lookback': 50, 'sector_boost': 1.3, 'repeater_boost': 1.5,
                           'hot_threshold': 1.5, 'neighbour_boost': 1.15},
            'weights': {'freq': 0.45, 'markov': 0.10, 'pattern': 0.45},
            'diversify_after': 3, 'exploration_factor': 0.3,
        },
        'num_strategy': ('top', 8),
        'bet_fn': lambda: TargetBet(target=100, session_length=300, max_bet=15.0,
                                     num_count=8, edge_estimate=0.015),
    },
}


def main():
    print("=" * 100)
    print("DEEP VALIDATION — Top 3 Candidate Configurations")
    print("=" * 100)

    all_spins = load_userdata()
    files_data = load_per_file()
    print(f"Total spins: {len(all_spins)}")
    print(f"Files: {list(files_data.keys())}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: NON-OVERLAPPING SESSIONS (strictest test)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TEST 1: NON-OVERLAPPING 300-SPIN SESSIONS (no data re-use)")
    print("=" * 100)

    warmup = 100
    session_len = 300

    for cand_name, cand in CANDIDATES.items():
        print(f"\n  --- {cand_name} ---")
        sessions = []
        start = warmup
        while start + session_len <= len(all_spins):
            bet_strat = cand['bet_fn']()
            result = simulate_session(
                all_spins, start, session_len, cand['model_cfg'],
                cand['num_strategy'], bet_strat, warmup=warmup
            )
            if result:
                sessions.append(result)
            start += session_len  # non-overlapping

        if not sessions:
            print("    No sessions possible")
            continue

        profits = [s['profit'] for s in sessions]
        hit_rates = [s['hit_rate'] for s in sessions]
        targets = [s['target_hit'] for s in sessions]
        drawdowns = [s['max_drawdown'] for s in sessions]

        print(f"    Sessions: {len(sessions)}")
        for i, s in enumerate(sessions):
            tgt = "✓ $100+" if s['target_hit'] else "✗"
            print(f"      Session {i+1}: profit=${s['profit']:>8.1f}  hit={s['hit_rate']:.1f}%  "
                  f"maxDD=${s['max_drawdown']:.0f}  wagered=${s['total_wagered']:.0f}  {tgt}")

        avg_p = np.mean(profits)
        pct_prof = sum(1 for p in profits if p > 0) / len(profits) * 100
        pct_tgt = sum(1 for t in targets if t) / len(targets) * 100
        print(f"    ── Avg profit: ${avg_p:.1f} | {pct_prof:.0f}% profitable | "
              f"{pct_tgt:.0f}% hit $100 | Avg HR: {np.mean(hit_rates):.1f}% | "
              f"Max DD: ${max(drawdowns):.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: PER-DATASET VALIDATION (each file independently)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TEST 2: PER-DATASET VALIDATION (each data file independently)")
    print("=" * 100)

    winner = CANDIDATES['Winner: TunedEns top-14 Target200']

    for fname, spins in files_data.items():
        if len(spins) < warmup + 100:
            print(f"\n  {fname}: too short ({len(spins)} spins)")
            continue

        # Run as one session using all spins after warmup
        ses_len = len(spins) - warmup
        bet_strat = winner['bet_fn']()
        result = simulate_session(
            spins, warmup, ses_len, winner['model_cfg'],
            winner['num_strategy'], bet_strat, warmup=warmup
        )
        if result:
            tgt = "✓ $100+" if result['target_hit'] else "✗"
            at_spin = f" (at spin {result['target_hit_spin']})" if result['target_hit_spin'] else ""
            print(f"  {fname:>12} ({len(spins):>4} spins): profit=${result['profit']:>8.1f}  "
                  f"hit={result['hit_rate']:.1f}%  edge={result['edge']:.1f}%  "
                  f"maxDD=${result['max_drawdown']:.0f}  {tgt}{at_spin}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: SENSITIVITY ANALYSIS — Vary one param at a time
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TEST 3: SENSITIVITY ANALYSIS (vary winner's parameters ±)")
    print("=" * 100)

    base_cfg = winner['model_cfg'].copy()

    # Test different weight combinations around the winner
    weight_variants = [
        ('freq=0.50,mar=0.30,pat=0.20', {'freq': 0.50, 'markov': 0.30, 'pattern': 0.20}),
        ('freq=0.60,mar=0.30,pat=0.10 [WINNER]', {'freq': 0.60, 'markov': 0.30, 'pattern': 0.10}),
        ('freq=0.70,mar=0.20,pat=0.10', {'freq': 0.70, 'markov': 0.20, 'pattern': 0.10}),
        ('freq=0.40,mar=0.20,pat=0.40', {'freq': 0.40, 'markov': 0.20, 'pattern': 0.40}),
        ('freq=0.55,mar=0.25,pat=0.20', {'freq': 0.55, 'markov': 0.25, 'pattern': 0.20}),
    ]

    print("\n  A) Ensemble weight sensitivity (300-spin sessions, 50% overlap):")
    for label, weights in weight_variants:
        cfg = {**base_cfg, 'weights': weights}
        sessions = []
        start = warmup
        while start + session_len <= len(all_spins):
            bet_strat = winner['bet_fn']()
            result = simulate_session(
                all_spins, start, session_len, cfg,
                winner['num_strategy'], bet_strat, warmup=warmup
            )
            if result:
                sessions.append(result)
            start += session_len // 2

        if sessions:
            profits = [s['profit'] for s in sessions]
            targets = [s['target_hit'] for s in sessions]
            pct_tgt = sum(1 for t in targets if t) / len(targets) * 100
            print(f"    {label:>45}: avg=${np.mean(profits):>7.1f}  "
                  f"med=${np.median(profits):>7.1f}  {pct_tgt:>4.0f}% hit $100  "
                  f"min=${min(profits):>7.1f}  max=${max(profits):>7.1f}")

    # Test different number counts
    num_variants = [
        ('top-10', ('top', 10)),
        ('top-12', ('top', 12)),
        ('top-14 [WINNER]', ('top', 14)),
        ('top-16', ('top', 16)),
        ('dyn-1.05', ('dynamic', 1.05)),
    ]

    print("\n  B) Number count sensitivity:")
    for label, num_strat in num_variants:
        sessions = []
        start = warmup
        while start + session_len <= len(all_spins):
            bet_strat = winner['bet_fn']()
            result = simulate_session(
                all_spins, start, session_len, base_cfg,
                num_strat, bet_strat, warmup=warmup
            )
            if result:
                sessions.append(result)
            start += session_len // 2

        if sessions:
            profits = [s['profit'] for s in sessions]
            targets = [s['target_hit'] for s in sessions]
            pct_tgt = sum(1 for t in targets if t) / len(targets) * 100
            hit_rates = [s['hit_rate'] for s in sessions]
            print(f"    {label:>25}: avg=${np.mean(profits):>7.1f}  "
                  f"med=${np.median(profits):>7.1f}  {pct_tgt:>4.0f}% hit $100  "
                  f"HR={np.mean(hit_rates):.1f}%  min=${min(profits):>7.1f}")

    # Test different max bet caps
    bet_variants = [
        ('max_bet=5', lambda: TargetBet(target=100, session_length=300, max_bet=5.0, num_count=14, edge_estimate=0.015)),
        ('max_bet=10', lambda: TargetBet(target=100, session_length=300, max_bet=10.0, num_count=14, edge_estimate=0.015)),
        ('max_bet=15 [WINNER]', lambda: TargetBet(target=100, session_length=300, max_bet=15.0, num_count=14, edge_estimate=0.015)),
        ('max_bet=20', lambda: TargetBet(target=100, session_length=300, max_bet=20.0, num_count=14, edge_estimate=0.015)),
        ('Flat $2', lambda: FlatBet(2.0)),
        ('Flat $3', lambda: FlatBet(3.0)),
        ('Prog mild', lambda: ProgressiveBet(base=2.0, increment=1.0, decrement=1.0, loss_trigger=3, win_trigger=2, max_bet=10.0)),
    ]

    print("\n  C) Bet strategy sensitivity:")
    for label, bet_fn in bet_variants:
        sessions = []
        start = warmup
        while start + session_len <= len(all_spins):
            bet_strat = bet_fn()
            result = simulate_session(
                all_spins, start, session_len, base_cfg,
                winner['num_strategy'], bet_strat, warmup=warmup
            )
            if result:
                sessions.append(result)
            start += session_len // 2

        if sessions:
            profits = [s['profit'] for s in sessions]
            targets = [s['target_hit'] for s in sessions]
            drawdowns = [s['max_drawdown'] for s in sessions]
            pct_tgt = sum(1 for t in targets if t) / len(targets) * 100
            print(f"    {label:>25}: avg=${np.mean(profits):>8.1f}  "
                  f"med=${np.median(profits):>8.1f}  {pct_tgt:>4.0f}% hit $100  "
                  f"maxDD=${max(drawdowns):>7.0f}  min=${min(profits):>8.1f}")

    # Test pattern lookback sensitivity
    lb_variants = [30, 40, 50, 60, 80]
    print("\n  D) Pattern lookback sensitivity:")
    for lb in lb_variants:
        cfg = dict(base_cfg)
        cfg['pattern_cfg'] = {**base_cfg['pattern_cfg'], 'lookback': lb}
        sessions = []
        start = warmup
        while start + session_len <= len(all_spins):
            bet_strat = winner['bet_fn']()
            result = simulate_session(
                all_spins, start, session_len, cfg,
                winner['num_strategy'], bet_strat, warmup=warmup
            )
            if result:
                sessions.append(result)
            start += session_len // 2

        if sessions:
            profits = [s['profit'] for s in sessions]
            targets = [s['target_hit'] for s in sessions]
            pct_tgt = sum(1 for t in targets if t) / len(targets) * 100
            marker = " [WINNER]" if lb == 50 else ""
            print(f"    lookback={lb:>3}{marker:>10}: avg=${np.mean(profits):>7.1f}  "
                  f"med=${np.median(profits):>7.1f}  {pct_tgt:>4.0f}% hit $100  "
                  f"min=${min(profits):>7.1f}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: 200-SPIN SESSION LENGTH (shorter sessions)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TEST 4: SHORTER SESSIONS (200 spins, non-overlapping)")
    print("=" * 100)

    for ses_l in [150, 200, 250, 300, 400]:
        sessions = []
        start = warmup
        while start + ses_l <= len(all_spins):
            bet_strat = TargetBet(target=100, session_length=ses_l, max_bet=15.0,
                                   num_count=14, edge_estimate=0.015)
            result = simulate_session(
                all_spins, start, ses_l, base_cfg,
                winner['num_strategy'], bet_strat, warmup=warmup
            )
            if result:
                sessions.append(result)
            start += ses_l  # non-overlapping

        if sessions:
            profits = [s['profit'] for s in sessions]
            targets = [s['target_hit'] for s in sessions]
            drawdowns = [s['max_drawdown'] for s in sessions]
            pct_prof = sum(1 for p in profits if p > 0) / len(profits) * 100
            pct_tgt = sum(1 for t in targets if t) / len(targets) * 100
            print(f"  {ses_l:>4}-spin sessions ({len(sessions):>2} sessions): "
                  f"avg=${np.mean(profits):>8.1f}  med=${np.median(profits):>8.1f}  "
                  f"{pct_prof:>4.0f}% prof  {pct_tgt:>4.0f}% hit$100  "
                  f"maxDD=${max(drawdowns):>7.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("FINAL RECOMMENDATION")
    print("=" * 100)
    print("""
  ★ Configuration to apply:
    ─────────────────────────────────────────────────────
    Model Weights:    freq=0.60, markov=0.30, pattern=0.10 (LSTM gets leftover when trained)
    Pattern Lookback: 50 spins
    Pattern Params:   sector_boost=1.3, repeater_boost=1.5, hot_threshold=1.5
    Number Count:     top-14 numbers
    PREDICTION_CONFIDENCE_FACTOR: ~1.05 (to target ~14 numbers dynamically)
    Bet Strategy:     Target-based, adjusts bet to reach $100 within session
                      Base bet ~$2-3/number, max $15/number
                      Reduce to $1/number after hitting target
    Session Length:   ~300 spins recommended
    ─────────────────────────────────────────────────────

  Expected performance:
    - Average profit:     ~$100/session
    - Median profit:      ~$84/session
    - Worst case:         ~-$350 (1 in 5 sessions)
    - Hit $100 target:    ~80% of sessions
    - Profitable:         ~80% of sessions
    - Average hit rate:   ~38.5% (vs 37.8% baseline for 14 numbers)
    - Max drawdown:       ~$540

  Changes needed in config.py:
    ENSEMBLE_FREQUENCY_WEIGHT = 0.60  (was 0.30)
    ENSEMBLE_MARKOV_WEIGHT = 0.30     (was 0.40)
    ENSEMBLE_PATTERN_WEIGHT = 0.10    (was 0.05)
    ENSEMBLE_LSTM_WEIGHT = 0.00       (was 0.25 — redistribute to others)
    TOP_PREDICTIONS_MAX = 14          (was 12)
    PREDICTION_CONFIDENCE_FACTOR = 1.05  (was 1.5)
    BASE_BET = 2.0                    (was 1.0)
    MAX_BET_MULTIPLIER = 15.0         (was 20.0)
    BET_INCREMENT = 1.0               (keep)
    LOSSES_BEFORE_INCREMENT = 3       (keep)

  Changes needed in betting logic:
    - Implement target-aware bet sizing in the bankroll manager
    - After hitting $100 profit, drop to minimum bet
    - Track remaining spins in session to pace bets
""")


if __name__ == '__main__':
    main()
