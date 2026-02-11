#!/usr/bin/env python3
"""
Model Weight Backtester — Test different weight combinations against real data.

This is a STANDALONE script. It does NOT modify any app code.
It loads your real spin data, runs each model independently, and shows
which model (and which weight combination) produces the best top-12 hit rate.

Usage:
    python backtest_models.py
"""

import sys
import os
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TOTAL_NUMBERS, WHEEL_ORDER, NUMBER_TO_POSITION,
    TOP_PREDICTIONS_COUNT,
    ENSEMBLE_FREQUENCY_WEIGHT, ENSEMBLE_MARKOV_WEIGHT,
    ENSEMBLE_PATTERN_WEIGHT, ENSEMBLE_LSTM_WEIGHT,
    USERDATA_DIR,
)
from app.ml.frequency_analyzer import FrequencyAnalyzer
from app.ml.markov_chain import MarkovChain
from app.ml.pattern_detector import PatternDetector


# ─── Weight Combos to Test ──────────────────────────────────────────
# Each entry: (name, freq_w, markov_w, pattern_w)
# Note: LSTM is excluded from backtest because it requires separate training
# and would be the same untrained model for all combos.
WEIGHT_COMBOS = [
    # Current config
    ("CURRENT (F35/M40/P5)",         0.35, 0.40, 0.05),
    # Pure models
    ("FREQUENCY ONLY",               1.00, 0.00, 0.00),
    ("MARKOV ONLY",                  0.00, 1.00, 0.00),
    ("PATTERN ONLY",                 0.00, 0.00, 1.00),
    # 2-model combos
    ("F50/M50",                      0.50, 0.50, 0.00),
    ("F60/M40",                      0.60, 0.40, 0.00),
    ("F40/M60",                      0.40, 0.60, 0.00),
    ("F70/M30",                      0.70, 0.30, 0.00),
    ("F30/M70",                      0.30, 0.70, 0.00),
    ("F80/M20",                      0.80, 0.20, 0.00),
    ("F20/M80",                      0.20, 0.80, 0.00),
    # 3-model combos
    ("F40/M40/P20",                  0.40, 0.40, 0.20),
    ("F35/M35/P30",                  0.35, 0.35, 0.30),
    ("F45/M45/P10",                  0.45, 0.45, 0.10),
    ("F50/M40/P10",                  0.50, 0.40, 0.10),
    ("F30/M50/P20",                  0.30, 0.50, 0.20),
    # Heavy single + support
    ("F70/M20/P10",                  0.70, 0.20, 0.10),
    ("F20/M70/P10",                  0.20, 0.70, 0.10),
    ("F50/M30/P20",                  0.50, 0.30, 0.20),
    ("F30/M30/P40",                  0.30, 0.30, 0.40),
]


def load_userdata():
    """Load all spins from userdata/*.txt files."""
    all_spins = []
    for filename in sorted(os.listdir(USERDATA_DIR)):
        if filename.endswith('.txt'):
            filepath = os.path.join(USERDATA_DIR, filename)
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            num = int(line)
                            if 0 <= num <= 36:
                                all_spins.append(num)
                        except ValueError:
                            pass
    return all_spins


def get_top_n(probs, n=TOP_PREDICTIONS_COUNT):
    """Get top N numbers by probability."""
    return set(np.argsort(probs)[-n:])


def run_backtest(spins, warmup=50):
    """Run backtesting on all weight combos.

    Args:
        spins: List of spin results (integers 0-36)
        warmup: Number of initial spins to feed models before tracking accuracy
                (models need some data before predictions are meaningful)
    """
    total = len(spins)
    print(f"\n{'='*70}")
    print(f"  MODEL WEIGHT BACKTESTER")
    print(f"  Total spins: {total} | Warmup: {warmup} | Evaluated: {total - warmup}")
    print(f"  Top-{TOP_PREDICTIONS_COUNT} predictions per spin")
    print(f"  Random baseline: {TOP_PREDICTIONS_COUNT}/{TOTAL_NUMBERS} = "
          f"{TOP_PREDICTIONS_COUNT/TOTAL_NUMBERS*100:.1f}%")
    print(f"{'='*70}\n")

    # Create independent model instances for each combo
    # (We only need ONE set of models since they all share the same state)
    freq = FrequencyAnalyzer()
    markov = MarkovChain()
    pattern = PatternDetector()

    # Track results per combo
    combo_hits = defaultdict(int)
    combo_total = defaultdict(int)

    # Track per-model individual performance
    model_hits = {'frequency': 0, 'markov': 0, 'pattern': 0}
    model_total = 0

    # Track rolling performance (last 100 spins)
    rolling_window = 100
    combo_rolling = {name: [] for name, _, _, _ in WEIGHT_COMBOS}
    model_rolling = {'frequency': [], 'markov': [], 'pattern': []}

    # Progress milestones
    milestones = set(range(200, total, 200))

    for i, actual in enumerate(spins):
        # Before updating models, get predictions (if past warmup)
        if i >= warmup:
            freq_probs = freq.get_number_probabilities()
            markov_probs = markov.get_probabilities()
            pattern_probs = pattern.get_number_probabilities()

            # Track individual model performance
            model_total += 1
            for model_name, probs in [('frequency', freq_probs),
                                       ('markov', markov_probs),
                                       ('pattern', pattern_probs)]:
                top = get_top_n(probs)
                hit = actual in top
                if hit:
                    model_hits[model_name] += 1
                model_rolling[model_name].append(1 if hit else 0)
                if len(model_rolling[model_name]) > rolling_window:
                    model_rolling[model_name].pop(0)

            # Test each weight combo
            for name, fw, mw, pw in WEIGHT_COMBOS:
                # Combine with weights (normalize)
                total_w = fw + mw + pw
                if total_w == 0:
                    continue
                ensemble = (freq_probs * fw + markov_probs * mw +
                           pattern_probs * pw) / total_w

                top = get_top_n(ensemble)
                hit = actual in top
                combo_total[name] += 1
                if hit:
                    combo_hits[name] += 1

                combo_rolling[name].append(1 if hit else 0)
                if len(combo_rolling[name]) > rolling_window:
                    combo_rolling[name].pop(0)

        # NOW update models with the actual result
        freq.update(actual)
        markov.update(actual)
        pattern.update(actual)

        # Print progress at milestones
        if i in milestones and i >= warmup:
            evaluated = i - warmup + 1
            print(f"  ... {i}/{total} spins processed ({evaluated} evaluated)")

    # ─── Results ────────────────────────────────────────────────────
    evaluated = model_total
    baseline_rate = TOP_PREDICTIONS_COUNT / TOTAL_NUMBERS * 100

    print(f"\n{'='*70}")
    print(f"  INDIVIDUAL MODEL RESULTS (top-{TOP_PREDICTIONS_COUNT} hit rate)")
    print(f"  Evaluated: {evaluated} spins | Random baseline: {baseline_rate:.1f}%")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'Hits':>6} {'Rate':>8} {'vs Random':>10} {'Last 100':>10}")
    print(f"  {'-'*54}")

    for model_name in ['frequency', 'markov', 'pattern']:
        hits = model_hits[model_name]
        rate = hits / max(1, evaluated) * 100
        vs_random = rate - baseline_rate
        last100 = model_rolling[model_name]
        last100_rate = sum(last100) / max(1, len(last100)) * 100
        marker = " ★" if rate == max(model_hits[m] / max(1, evaluated) * 100
                                      for m in model_hits) else ""
        print(f"  {model_name:<20} {hits:>6} {rate:>7.1f}% {vs_random:>+9.1f}% {last100_rate:>9.1f}%{marker}")

    print(f"\n{'='*70}")
    print(f"  WEIGHT COMBINATION RESULTS (top-{TOP_PREDICTIONS_COUNT} hit rate)")
    print(f"  Evaluated: {evaluated} spins | Random baseline: {baseline_rate:.1f}%")
    print(f"{'='*70}")
    print(f"  {'Combo':<30} {'Hits':>6} {'Rate':>8} {'vs Random':>10} {'Last 100':>10}")
    print(f"  {'-'*64}")

    # Sort by hit rate (best first)
    sorted_combos = sorted(WEIGHT_COMBOS,
                           key=lambda x: combo_hits[x[0]] / max(1, combo_total[x[0]]),
                           reverse=True)

    best_rate = 0
    for idx, (name, fw, mw, pw) in enumerate(sorted_combos):
        hits = combo_hits[name]
        total_eval = combo_total[name]
        rate = hits / max(1, total_eval) * 100
        vs_random = rate - baseline_rate
        last100 = combo_rolling[name]
        last100_rate = sum(last100) / max(1, len(last100)) * 100

        if idx == 0:
            best_rate = rate

        # Markers
        marker = ""
        if idx == 0:
            marker = " 🏆 BEST"
        elif rate == best_rate:
            marker = " 🏆 TIE"
        elif name.startswith("CURRENT"):
            marker = " ← current"

        print(f"  {name:<30} {hits:>6} {rate:>7.1f}% {vs_random:>+9.1f}% {last100_rate:>9.1f}%{marker}")

    # Summary recommendation
    best_name = sorted_combos[0][0]
    best_hits = combo_hits[best_name]
    best_total = combo_total[best_name]
    best_pct = best_hits / max(1, best_total) * 100

    current_name = "CURRENT (F35/M40/P5)"
    current_hits = combo_hits[current_name]
    current_pct = current_hits / max(1, combo_total[current_name]) * 100

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Best combo:    {best_name} — {best_pct:.1f}% hit rate")
    print(f"  Current combo: {current_name} — {current_pct:.1f}% hit rate")
    print(f"  Difference:    {best_pct - current_pct:+.1f}%")
    print(f"  Random:        {baseline_rate:.1f}%")
    print(f"{'='*70}\n")

    # ─── Per-Model Agreement Analysis ─────────────────────────────
    print(f"{'='*70}")
    print(f"  MODEL AGREEMENT ANALYSIS")
    print(f"  When do individual models agree/disagree?")
    print(f"{'='*70}")

    # Re-run for agreement analysis
    freq2 = FrequencyAnalyzer()
    markov2 = MarkovChain()
    pattern2 = PatternDetector()

    agree_hits = 0
    agree_total = 0
    disagree_hits = 0
    disagree_total = 0
    all3_agree_hits = 0
    all3_agree_total = 0

    for i, actual in enumerate(spins):
        if i >= warmup:
            fp = freq2.get_number_probabilities()
            mp = markov2.get_probabilities()
            pp = pattern2.get_number_probabilities()

            f_top = get_top_n(fp)
            m_top = get_top_n(mp)
            p_top = get_top_n(pp)

            # Numbers in ALL 3 model top-12
            in_all3 = f_top & m_top & p_top
            # Numbers in at least 2 model top-12
            in_2plus = (f_top & m_top) | (f_top & p_top) | (m_top & p_top)

            # When frequency and markov agree on a number
            fm_agree = f_top & m_top
            if actual in fm_agree:
                agree_hits += 1
            agree_total += 1

            if actual in (f_top | m_top | p_top) and actual not in in_2plus:
                disagree_total += 1
                # This spin, models disagreed but one got it
                disagree_hits += 1

            if in_all3:
                all3_agree_total += 1
                if actual in in_all3:
                    all3_agree_hits += 1

        freq2.update(actual)
        markov2.update(actual)
        pattern2.update(actual)

    print(f"  Freq+Markov agree (number in both top-12):")
    agree_rate = agree_hits / max(1, agree_total) * 100
    print(f"    Hit rate when agreed: {agree_hits}/{agree_total} = {agree_rate:.1f}%")

    if all3_agree_total > 0:
        all3_rate = all3_agree_hits / max(1, all3_agree_total) * 100
        print(f"  All 3 models agree:")
        print(f"    Spins where all 3 agreed: {all3_agree_total}")
        print(f"    Hit rate when all agreed: {all3_agree_hits}/{all3_agree_total} = {all3_rate:.1f}%")
    print()


if __name__ == '__main__':
    spins = load_userdata()
    if not spins:
        print("ERROR: No spin data found in userdata/")
        sys.exit(1)

    print(f"Loaded {len(spins)} spins from userdata/")
    run_backtest(spins, warmup=50)
