#!/usr/bin/env python3
"""
Comprehensive Model Weight Backtester — Long-Running Deep Analysis
===================================================================
Runs for hours testing every possible combination of model weights,
parameters, and strategies against real spin data.

This is a STANDALONE script — does NOT modify any app code.

Tests:
  1. All model weight combinations (fine-grained 5% steps)
  2. Individual model performance per spin
  3. Frequency analyzer: different flat/recent blend ratios
  4. Markov chain: different 1st/2nd order blend ratios
  5. LSTM/GRU included (trains from scratch during backtest)
  6. Confidence threshold sweep
  7. Rolling window analysis (which combos work best at different data sizes)
  8. Monte Carlo simulation (shuffle data, test statistical significance)
  9. Anchor selection: probability-first vs top-N naive
  10. Time-of-day / position-in-data analysis

Results written to: backtest_results.txt (continuously updated)

Usage:
    source venv/bin/activate && python backtest_comprehensive.py
"""

import sys
import os
import time
import numpy as np
from collections import defaultdict, Counter
from itertools import product
import random
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TOTAL_NUMBERS, WHEEL_ORDER, NUMBER_TO_POSITION,
    TOP_PREDICTIONS_COUNT, USERDATA_DIR,
    MARKOV_ORDER_1_WEIGHT, MARKOV_ORDER_2_WEIGHT,
    FREQUENCY_FLAT_WEIGHT, FREQUENCY_RECENT_WEIGHT,
    FREQUENCY_DECAY_FACTOR,
)
from app.ml.frequency_analyzer import FrequencyAnalyzer
from app.ml.markov_chain import MarkovChain
from app.ml.pattern_detector import PatternDetector

# Try importing LSTM (may fail if torch not available)
try:
    from app.ml.lstm_predictor import LSTMPredictor
    HAS_LSTM = True
except Exception:
    HAS_LSTM = False

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results.txt')
TOP_N = TOP_PREDICTIONS_COUNT  # 12

# ─── Data Loading ────────────────────────────────────────────────────

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


def get_top_n(probs, n=TOP_N):
    """Get top N numbers by probability."""
    return set(np.argsort(probs)[-n:])


def log(msg, also_print=True):
    """Write to results file and optionally print."""
    with open(RESULTS_FILE, 'a') as f:
        f.write(msg + '\n')
    if also_print:
        print(msg)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Fine-Grained Weight Grid Search (3-model)
# ═══════════════════════════════════════════════════════════════════════

def test_weight_grid(spins, warmup=50, step=5):
    """Test all 3-model weight combinations at given step size (%)."""
    log(f"\n{'='*80}")
    log(f"  TEST 1: WEIGHT GRID SEARCH (step={step}%, 3 models: Freq/Markov/Pattern)")
    log(f"  Spins: {len(spins)} | Warmup: {warmup} | Evaluated: {len(spins)-warmup}")
    log(f"  Random baseline: {TOP_N}/{TOTAL_NUMBERS} = {TOP_N/TOTAL_NUMBERS*100:.1f}%")
    log(f"{'='*80}")

    # Generate all weight combos where freq + markov + pattern = 100
    combos = []
    for fw in range(0, 101, step):
        for mw in range(0, 101 - fw, step):
            pw = 100 - fw - mw
            combos.append((fw/100, mw/100, pw/100))

    log(f"  Testing {len(combos)} weight combinations...")

    # Pre-compute model predictions for each spin (avoid redundant computation)
    freq = FrequencyAnalyzer()
    markov = MarkovChain()
    pattern = PatternDetector()

    # Store predictions: list of (freq_probs, markov_probs, pattern_probs, actual)
    predictions = []

    for i, actual in enumerate(spins):
        if i >= warmup:
            fp = freq.get_number_probabilities()
            mp = markov.get_probabilities()
            pp = pattern.get_number_probabilities()
            predictions.append((fp, mp, pp, actual))

        freq.update(actual)
        markov.update(actual)
        pattern.update(actual)

    # Now test each combo
    results = []
    for fw, mw, pw in combos:
        hits = 0
        for fp, mp, pp, actual in predictions:
            ensemble = fp * fw + mp * mw + pp * pw
            if ensemble.sum() > 0:
                ensemble /= ensemble.sum()
            top = get_top_n(ensemble)
            if actual in top:
                hits += 1
        rate = hits / len(predictions) * 100
        results.append((fw, mw, pw, hits, rate))

    # Sort by hit rate
    results.sort(key=lambda x: x[4], reverse=True)

    # Print top 30
    log(f"\n  {'Rank':<6} {'Freq%':<8} {'Markov%':<9} {'Pattern%':<10} {'Hits':<8} {'Rate':<8} {'vs Random':<10}")
    log(f"  {'-'*60}")

    baseline = TOP_N / TOTAL_NUMBERS * 100
    for i, (fw, mw, pw, hits, rate) in enumerate(results[:30]):
        marker = ""
        if i == 0:
            marker = " 🏆"
        elif abs(fw - 0.35) < 0.01 and abs(mw - 0.40) < 0.01:
            marker = " ← current"
        log(f"  {i+1:<6} {fw*100:<8.0f} {mw*100:<9.0f} {pw*100:<10.0f} {hits:<8} {rate:<7.1f}% {rate-baseline:>+8.1f}%{marker}")

    # Also print worst 5 for contrast
    log(f"\n  --- Worst 5 ---")
    for i, (fw, mw, pw, hits, rate) in enumerate(results[-5:]):
        log(f"  {len(results)-4+i:<6} {fw*100:<8.0f} {mw*100:<9.0f} {pw*100:<10.0f} {hits:<8} {rate:<7.1f}% {rate-baseline:>+8.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Individual Model Performance Per Spin (Detailed)
# ═══════════════════════════════════════════════════════════════════════

def test_individual_models(spins, warmup=50):
    """Detailed per-model analysis: top-1, top-5, top-12 hit rates."""
    log(f"\n{'='*80}")
    log(f"  TEST 2: INDIVIDUAL MODEL DETAILED ANALYSIS")
    log(f"{'='*80}")

    freq = FrequencyAnalyzer()
    markov = MarkovChain()
    pattern = PatternDetector()

    model_stats = {
        'frequency': {'top1': 0, 'top5': 0, 'top12': 0, 'avg_rank': [], 'prob_assigned': []},
        'markov': {'top1': 0, 'top5': 0, 'top12': 0, 'avg_rank': [], 'prob_assigned': []},
        'pattern': {'top1': 0, 'top5': 0, 'top12': 0, 'avg_rank': [], 'prob_assigned': []},
    }
    total_eval = 0

    for i, actual in enumerate(spins):
        if i >= warmup:
            total_eval += 1
            for name, model_fn in [('frequency', freq.get_number_probabilities),
                                    ('markov', markov.get_probabilities),
                                    ('pattern', pattern.get_number_probabilities)]:
                probs = model_fn()
                sorted_nums = np.argsort(probs)[::-1]
                rank = int(np.where(sorted_nums == actual)[0][0]) + 1

                model_stats[name]['avg_rank'].append(rank)
                model_stats[name]['prob_assigned'].append(float(probs[actual]))

                if rank <= 1:
                    model_stats[name]['top1'] += 1
                if rank <= 5:
                    model_stats[name]['top5'] += 1
                if rank <= 12:
                    model_stats[name]['top12'] += 1

        freq.update(actual)
        markov.update(actual)
        pattern.update(actual)

    log(f"\n  {'Model':<15} {'Top-1':<10} {'Top-5':<10} {'Top-12':<10} {'Avg Rank':<12} {'Avg Prob':<12}")
    log(f"  {'-'*70}")

    random_top1 = 1/37*100
    random_top5 = 5/37*100
    random_top12 = 12/37*100

    for name in ['frequency', 'markov', 'pattern']:
        s = model_stats[name]
        t1 = s['top1'] / total_eval * 100
        t5 = s['top5'] / total_eval * 100
        t12 = s['top12'] / total_eval * 100
        avg_r = np.mean(s['avg_rank'])
        avg_p = np.mean(s['prob_assigned'])
        log(f"  {name:<15} {t1:<9.1f}% {t5:<9.1f}% {t12:<9.1f}% {avg_r:<12.1f} {avg_p:<12.6f}")

    log(f"  {'RANDOM':<15} {random_top1:<9.1f}% {random_top5:<9.1f}% {random_top12:<9.1f}% {19.0:<12.1f} {1/37:<12.6f}")

    return model_stats


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Frequency Analyzer — Flat vs Recent Blend Sweep
# ═══════════════════════════════════════════════════════════════════════

def test_frequency_blend(spins, warmup=50):
    """Test different flat/recent blend ratios for frequency analyzer."""
    log(f"\n{'='*80}")
    log(f"  TEST 3: FREQUENCY ANALYZER — FLAT vs RECENT BLEND SWEEP")
    log(f"{'='*80}")

    blend_ratios = [(i/100, 1-i/100) for i in range(0, 101, 5)]
    results = []

    for flat_w, recent_w in blend_ratios:
        freq = FrequencyAnalyzer()
        hits = 0
        total_eval = 0

        for i, actual in enumerate(spins):
            if i >= warmup:
                total_eval += 1
                # Manually compute blended probs with this ratio
                if len(freq.spin_history) >= 5:
                    total_spins = len(freq.spin_history)
                    flat_probs = np.array([
                        (freq.frequency_counts.get(j, 0) + 1) / (total_spins + TOTAL_NUMBERS)
                        for j in range(TOTAL_NUMBERS)
                    ])
                    flat_probs /= flat_probs.sum()
                    recent_probs = freq.get_recent_probabilities()
                    blended = flat_w * flat_probs + recent_w * recent_probs
                    blended /= blended.sum()
                else:
                    blended = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

                top = get_top_n(blended)
                if actual in top:
                    hits += 1

            freq.update(actual)

        rate = hits / total_eval * 100
        results.append((flat_w, recent_w, hits, rate))

    results.sort(key=lambda x: x[3], reverse=True)

    log(f"\n  {'Rank':<6} {'Flat%':<8} {'Recent%':<9} {'Hits':<8} {'Rate':<8}")
    log(f"  {'-'*40}")

    for i, (fw, rw, hits, rate) in enumerate(results[:20]):
        marker = ""
        if i == 0:
            marker = " 🏆"
        elif abs(fw - 0.35) < 0.02:
            marker = " ← current"
        log(f"  {i+1:<6} {fw*100:<8.0f} {rw*100:<9.0f} {hits:<8} {rate:<7.1f}%{marker}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Markov Chain — 1st vs 2nd Order Blend Sweep
# ═══════════════════════════════════════════════════════════════════════

def test_markov_blend(spins, warmup=50):
    """Test different 1st/2nd order blend ratios for Markov chain."""
    log(f"\n{'='*80}")
    log(f"  TEST 4: MARKOV CHAIN — 1st vs 2nd ORDER BLEND SWEEP")
    log(f"{'='*80}")

    blend_ratios = [(i/100, 1-i/100) for i in range(0, 101, 5)]
    results = []

    # Build a Markov chain with full history first
    base_markov = MarkovChain()
    for s in spins[:warmup]:
        base_markov.update(s)

    # Now test incrementally
    for o1_w, o2_w in blend_ratios:
        markov = MarkovChain()
        hits = 0
        total_eval = 0

        for i, actual in enumerate(spins):
            if i >= warmup:
                total_eval += 1
                p1 = markov.predict_first_order()
                p2 = markov.predict_second_order()
                combined = o1_w * p1 + o2_w * p2
                total = combined.sum()
                if total > 0:
                    combined /= total
                top = get_top_n(combined)
                if actual in top:
                    hits += 1

            markov.update(actual)

        rate = hits / total_eval * 100
        results.append((o1_w, o2_w, hits, rate))

    results.sort(key=lambda x: x[3], reverse=True)

    log(f"\n  {'Rank':<6} {'1st Order%':<12} {'2nd Order%':<12} {'Hits':<8} {'Rate':<8}")
    log(f"  {'-'*46}")

    for i, (o1, o2, hits, rate) in enumerate(results[:20]):
        marker = ""
        if i == 0:
            marker = " 🏆"
        elif abs(o1 - 0.60) < 0.02:
            marker = " ← current"
        log(f"  {i+1:<6} {o1*100:<12.0f} {o2*100:<12.0f} {hits:<8} {rate:<7.1f}%{marker}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Rolling Window Analysis — How Do Models Perform Over Time?
# ═══════════════════════════════════════════════════════════════════════

def test_rolling_performance(spins, warmup=50, window=100):
    """Track hit rate over time with rolling window for each model and combos."""
    log(f"\n{'='*80}")
    log(f"  TEST 5: ROLLING PERFORMANCE (window={window} spins)")
    log(f"  Shows how model performance changes as more data accumulates")
    log(f"{'='*80}")

    freq = FrequencyAnalyzer()
    markov = MarkovChain()
    pattern = PatternDetector()

    rolling = {
        'frequency': [],
        'markov': [],
        'pattern': [],
        'best_static_combo': [],  # Will use best combo from Test 1
        'F50/M50': [],
        'current': [],
    }
    hits_buffer = {k: [] for k in rolling}

    # Some interesting checkpoints
    checkpoints = [100, 200, 300, 500, 750, 1000, 1250, 1477]

    for i, actual in enumerate(spins):
        if i >= warmup:
            fp = freq.get_number_probabilities()
            mp = markov.get_probabilities()
            pp = pattern.get_number_probabilities()

            # Individual models
            for name, probs in [('frequency', fp), ('markov', mp), ('pattern', pp)]:
                hit = 1 if actual in get_top_n(probs) else 0
                hits_buffer[name].append(hit)
                if len(hits_buffer[name]) > window:
                    hits_buffer[name].pop(0)
                rolling[name].append(sum(hits_buffer[name]) / len(hits_buffer[name]) * 100)

            # F50/M50
            ens = 0.5 * fp + 0.5 * mp
            ens /= ens.sum()
            hit = 1 if actual in get_top_n(ens) else 0
            hits_buffer['F50/M50'].append(hit)
            if len(hits_buffer['F50/M50']) > window:
                hits_buffer['F50/M50'].pop(0)
            rolling['F50/M50'].append(sum(hits_buffer['F50/M50']) / len(hits_buffer['F50/M50']) * 100)

            # Current config
            ens = 0.35 * fp + 0.40 * mp + 0.05 * pp
            ens /= ens.sum()
            hit = 1 if actual in get_top_n(ens) else 0
            hits_buffer['current'].append(hit)
            if len(hits_buffer['current']) > window:
                hits_buffer['current'].pop(0)
            rolling['current'].append(sum(hits_buffer['current']) / len(hits_buffer['current']) * 100)

        freq.update(actual)
        markov.update(actual)
        pattern.update(actual)

    # Print at checkpoints
    log(f"\n  {'Spin':<8}", also_print=True)
    header = f"  {'Spin':<8}"
    for name in ['frequency', 'markov', 'pattern', 'F50/M50', 'current']:
        header += f" {name:<12}"
    log(header)
    log(f"  {'-'*70}")

    for cp in checkpoints:
        idx = cp - warmup - 1
        if 0 <= idx < len(rolling['frequency']):
            line = f"  {cp:<8}"
            for name in ['frequency', 'markov', 'pattern', 'F50/M50', 'current']:
                val = rolling[name][idx]
                line += f" {val:<11.1f}%"
            log(line)

    return rolling


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Monte Carlo — Is Performance Real or Just Luck?
# ═══════════════════════════════════════════════════════════════════════

def test_monte_carlo(spins, warmup=50, n_shuffles=200):
    """Shuffle data and compare — if real data beats shuffled, signal is real."""
    log(f"\n{'='*80}")
    log(f"  TEST 6: MONTE CARLO SIGNIFICANCE TEST ({n_shuffles} shuffles)")
    log(f"  If real data beats shuffled data, the signal is real (not luck)")
    log(f"{'='*80}")

    baseline = TOP_N / TOTAL_NUMBERS * 100

    # First get real performance
    combos_to_test = [
        ("FREQUENCY", 1.0, 0.0, 0.0),
        ("MARKOV", 0.0, 1.0, 0.0),
        ("PATTERN", 0.0, 0.0, 1.0),
        ("CURRENT", 0.35, 0.40, 0.05),
        ("F50/M50", 0.5, 0.5, 0.0),
        ("MARKOV_HEAVY", 0.2, 0.8, 0.0),
    ]

    def run_one(data, combos, warmup_n):
        freq = FrequencyAnalyzer()
        markov = MarkovChain()
        pattern = PatternDetector()
        combo_hits = {name: 0 for name, _, _, _ in combos}
        total_eval = 0

        for i, actual in enumerate(data):
            if i >= warmup_n:
                total_eval += 1
                fp = freq.get_number_probabilities()
                mp = markov.get_probabilities()
                pp = pattern.get_number_probabilities()

                for name, fw, mw, pw in combos:
                    ens = fp * fw + mp * mw + pp * pw
                    s = ens.sum()
                    if s > 0:
                        ens /= s
                    if actual in get_top_n(ens):
                        combo_hits[name] += 1

            freq.update(actual)
            markov.update(actual)
            pattern.update(actual)

        rates = {name: combo_hits[name] / max(1, total_eval) * 100 for name in combo_hits}
        return rates

    # Real data performance
    real_rates = run_one(spins, combos_to_test, warmup)

    # Shuffled performance
    log(f"\n  Running {n_shuffles} shuffled trials...")
    shuffle_rates = {name: [] for name, _, _, _ in combos_to_test}

    for trial in range(n_shuffles):
        shuffled = list(spins)
        random.shuffle(shuffled)
        rates = run_one(shuffled, combos_to_test, warmup)
        for name in rates:
            shuffle_rates[name].append(rates[name])

        if (trial + 1) % 50 == 0:
            log(f"    ... {trial+1}/{n_shuffles} shuffles done")

    # Report
    log(f"\n  {'Combo':<18} {'Real%':<8} {'Shuffle Mean':<14} {'Shuffle Std':<12} {'P-value':<10} {'Signal?':<8}")
    log(f"  {'-'*72}")

    for name, _, _, _ in combos_to_test:
        real = real_rates[name]
        shuf_mean = np.mean(shuffle_rates[name])
        shuf_std = np.std(shuffle_rates[name])
        # How many shuffled trials beat real?
        beats = sum(1 for s in shuffle_rates[name] if s >= real)
        p_value = beats / n_shuffles
        signal = "YES ✓" if p_value < 0.05 else "NO"
        log(f"  {name:<18} {real:<7.1f}% {shuf_mean:<13.1f}% {shuf_std:<11.2f}% {p_value:<9.3f} {signal}")

    return real_rates, shuffle_rates


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: LSTM/GRU Backtest (if available)
# ═══════════════════════════════════════════════════════════════════════

def test_lstm_model(spins, warmup=100):
    """Train LSTM from scratch and test its predictions."""
    if not HAS_LSTM:
        log(f"\n{'='*80}")
        log(f"  TEST 7: LSTM/GRU — SKIPPED (PyTorch not available)")
        log(f"{'='*80}")
        return None

    log(f"\n{'='*80}")
    log(f"  TEST 7: LSTM/GRU MODEL BACKTEST")
    log(f"  Training from scratch with incremental data")
    log(f"{'='*80}")

    lstm = LSTMPredictor()
    hits_untrained = 0
    hits_trained = 0
    total_untrained = 0
    total_trained = 0

    train_points = [200, 500, 1000]
    trained_at = []

    for i, actual in enumerate(spins):
        if i >= warmup:
            probs = lstm.predict()
            top = get_top_n(probs)
            hit = actual in top

            if lstm.is_trained:
                total_trained += 1
                if hit:
                    hits_trained += 1
            else:
                total_untrained += 1
                if hit:
                    hits_untrained += 1

        lstm.update(actual)

        # Train at specific points
        if i + 1 in train_points and not lstm.is_trained:
            log(f"  Training LSTM at spin {i+1}...")
            try:
                result = lstm.train()
                if result:
                    log(f"    → Trained successfully!")
                    trained_at.append(i + 1)
            except Exception as e:
                log(f"    → Training failed: {e}")

    log(f"\n  LSTM Results:")
    if total_untrained > 0:
        log(f"    Before training: {hits_untrained}/{total_untrained} = {hits_untrained/total_untrained*100:.1f}%")
    if total_trained > 0:
        log(f"    After training:  {hits_trained}/{total_trained} = {hits_trained/total_trained*100:.1f}%")
    log(f"    Random baseline: {TOP_N/TOTAL_NUMBERS*100:.1f}%")
    log(f"    Trained at spins: {trained_at}")

    return {'untrained': hits_untrained, 'trained': hits_trained,
            'total_untrained': total_untrained, 'total_trained': total_trained}


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: 4-Model Weight Grid (with LSTM)
# ═══════════════════════════════════════════════════════════════════════

def test_4model_grid(spins, warmup=50, step=10):
    """Test 4-model combinations including LSTM (coarser grid due to computation)."""
    if not HAS_LSTM:
        log(f"\n{'='*80}")
        log(f"  TEST 8: 4-MODEL GRID — SKIPPED (no LSTM)")
        log(f"{'='*80}")
        return None

    log(f"\n{'='*80}")
    log(f"  TEST 8: 4-MODEL WEIGHT GRID (Freq/Markov/Pattern/LSTM, step={step}%)")
    log(f"{'='*80}")

    # Pre-train LSTM with first portion of data
    lstm = LSTMPredictor()
    for s in spins[:warmup]:
        lstm.update(s)

    log(f"  Pre-training LSTM on first {warmup} spins...")
    try:
        lstm.train()
        log(f"  LSTM trained: {lstm.is_trained}")
    except:
        log(f"  LSTM training failed, using untrained predictions")

    # Pre-compute all model predictions
    freq = FrequencyAnalyzer()
    markov = MarkovChain()
    pattern = PatternDetector()

    predictions = []
    for i, actual in enumerate(spins):
        if i >= warmup:
            fp = freq.get_number_probabilities()
            mp = markov.get_probabilities()
            pp = pattern.get_number_probabilities()
            lp = lstm.predict()
            predictions.append((fp, mp, pp, lp, actual))

        freq.update(actual)
        markov.update(actual)
        pattern.update(actual)
        lstm.update(actual)

    # Generate 4-model combos
    combos = []
    for fw in range(0, 101, step):
        for mw in range(0, 101 - fw, step):
            for pw in range(0, 101 - fw - mw, step):
                lw = 100 - fw - mw - pw
                combos.append((fw/100, mw/100, pw/100, lw/100))

    log(f"  Testing {len(combos)} 4-model combinations...")

    results = []
    for fw, mw, pw, lw in combos:
        hits = 0
        for fp, mp, pp, lp, actual in predictions:
            ensemble = fp * fw + mp * mw + pp * pw + lp * lw
            s = ensemble.sum()
            if s > 0:
                ensemble /= s
            if actual in get_top_n(ensemble):
                hits += 1
        rate = hits / len(predictions) * 100
        results.append((fw, mw, pw, lw, hits, rate))

    results.sort(key=lambda x: x[5], reverse=True)

    log(f"\n  {'Rank':<6} {'Freq%':<8} {'Mark%':<8} {'Patt%':<8} {'LSTM%':<8} {'Hits':<8} {'Rate':<8}")
    log(f"  {'-'*56}")

    for i, (fw, mw, pw, lw, hits, rate) in enumerate(results[:30]):
        marker = ""
        if i == 0:
            marker = " 🏆"
        log(f"  {i+1:<6} {fw*100:<8.0f} {mw*100:<8.0f} {pw*100:<8.0f} {lw*100:<8.0f} {hits:<8} {rate:<7.1f}%{marker}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# TEST 9: Anchor Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════

def test_anchor_strategies(spins, warmup=50):
    """Compare different selection strategies for choosing top-12 numbers."""
    log(f"\n{'='*80}")
    log(f"  TEST 9: TOP-12 SELECTION STRATEGIES")
    log(f"  How should we pick the final 12 numbers from probability distribution?")
    log(f"{'='*80}")

    freq = FrequencyAnalyzer()
    markov = MarkovChain()
    pattern = PatternDetector()

    strategies = {
        'naive_top12': 0,       # Just pick top 12 by probability
        'top12_with_neighbours': 0,  # Top 6 + their wheel neighbours
        'cluster_based': 0,     # Group by wheel position, pick densest clusters
        'diversified': 0,       # Top 8 by prob + 4 from cold sectors
    }
    total_eval = 0

    for i, actual in enumerate(spins):
        if i >= warmup:
            total_eval += 1
            fp = freq.get_number_probabilities()
            mp = markov.get_probabilities()
            # Using current weights
            ens = 0.35 * fp + 0.40 * mp + 0.05 * pattern.get_number_probabilities()
            ens /= ens.sum()

            sorted_idx = np.argsort(ens)[::-1]

            # Strategy 1: Naive top 12
            top12 = set(sorted_idx[:12])
            if actual in top12:
                strategies['naive_top12'] += 1

            # Strategy 2: Top 6 + wheel neighbours
            top6 = list(sorted_idx[:6])
            selection = set(top6)
            for num in top6:
                pos = NUMBER_TO_POSITION.get(int(num), 0)
                for offset in [-1, 1]:
                    nb_pos = (pos + offset) % len(WHEEL_ORDER)
                    selection.add(WHEEL_ORDER[nb_pos])
                if len(selection) >= 12:
                    break
            # Fill to 12 with next highest prob
            for num in sorted_idx:
                if len(selection) >= 12:
                    break
                selection.add(int(num))
            if actual in selection:
                strategies['top12_with_neighbours'] += 1

            # Strategy 3: Cluster-based (group top-18 by wheel position, pick densest)
            top18 = [int(x) for x in sorted_idx[:18]]
            positions = [(n, NUMBER_TO_POSITION.get(n, 0)) for n in top18]
            positions.sort(key=lambda x: x[1])
            clusters = []
            current = [positions[0]]
            for j in range(1, len(positions)):
                if positions[j][1] - current[-1][1] <= 2:
                    current.append(positions[j])
                else:
                    clusters.append(current)
                    current = [positions[j]]
            clusters.append(current)
            # Sort by cluster density (prob per number)
            clusters.sort(key=lambda c: sum(ens[n] for n, _ in c) / len(c), reverse=True)
            selection = set()
            for cluster in clusters:
                for n, _ in cluster:
                    if len(selection) < 12:
                        selection.add(n)
            if actual in selection:
                strategies['cluster_based'] += 1

            # Strategy 4: Diversified (top 8 + 4 from unexplored sectors)
            top8 = set(int(x) for x in sorted_idx[:8])
            top8_sectors = set(NUMBER_TO_POSITION.get(n, 0) // 5 for n in top8)
            other_nums = []
            for num in sorted_idx[8:]:
                sector = NUMBER_TO_POSITION.get(int(num), 0) // 5
                if sector not in top8_sectors:
                    other_nums.append(int(num))
                    top8_sectors.add(sector)
                if len(other_nums) >= 4:
                    break
            selection = top8 | set(other_nums[:4])
            # Fill remaining
            for num in sorted_idx:
                if len(selection) >= 12:
                    break
                selection.add(int(num))
            if actual in selection:
                strategies['diversified'] += 1

        freq.update(actual)
        markov.update(actual)
        pattern.update(actual)

    log(f"\n  {'Strategy':<25} {'Hits':<8} {'Rate':<8} {'vs Random':<10}")
    log(f"  {'-'*52}")

    baseline = TOP_N / TOTAL_NUMBERS * 100
    for name, hits in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
        rate = hits / total_eval * 100
        marker = " 🏆" if rate == max(v / total_eval * 100 for v in strategies.values()) else ""
        log(f"  {name:<25} {hits:<8} {rate:<7.1f}% {rate-baseline:>+8.1f}%{marker}")

    return strategies


# ═══════════════════════════════════════════════════════════════════════
# TEST 10: Data Quality Analysis
# ═══════════════════════════════════════════════════════════════════════

def test_data_quality(spins):
    """Analyze the spin data itself for patterns, distribution, randomness."""
    log(f"\n{'='*80}")
    log(f"  TEST 10: DATA QUALITY ANALYSIS")
    log(f"{'='*80}")

    counts = Counter(spins)
    total = len(spins)
    expected = total / TOTAL_NUMBERS

    log(f"\n  Total spins: {total}")
    log(f"  Expected per number: {expected:.1f}")
    log(f"  Unique numbers seen: {len(counts)}")

    # Distribution stats
    count_values = [counts.get(i, 0) for i in range(TOTAL_NUMBERS)]
    log(f"\n  Distribution:")
    log(f"    Min count:  {min(count_values)} (number {count_values.index(min(count_values))})")
    log(f"    Max count:  {max(count_values)} (number {count_values.index(max(count_values))})")
    log(f"    Std dev:    {np.std(count_values):.1f}")
    log(f"    Expected std (Poisson): {np.sqrt(expected):.1f}")

    # Chi-square test
    from scipy import stats
    observed = np.array(count_values)
    expected_arr = np.full(TOTAL_NUMBERS, expected)
    chi2, p_value = stats.chisquare(observed, expected_arr)
    log(f"\n  Chi-square test:")
    log(f"    Statistic:  {chi2:.2f}")
    log(f"    P-value:    {p_value:.6f}")
    log(f"    Significant: {'YES - data has bias!' if p_value < 0.05 else 'NO - data looks random'}")

    # Runs test (consecutive same number)
    repeats = sum(1 for i in range(1, len(spins)) if spins[i] == spins[i-1])
    expected_repeats = total / TOTAL_NUMBERS
    log(f"\n  Repeats (back-to-back same number):")
    log(f"    Actual:   {repeats}")
    log(f"    Expected: {expected_repeats:.1f}")

    # Hot and cold numbers
    hot = [(i, c) for i, c in enumerate(count_values) if c > expected * 1.3]
    cold = [(i, c) for i, c in enumerate(count_values) if c < expected * 0.7]
    hot.sort(key=lambda x: x[1], reverse=True)
    cold.sort(key=lambda x: x[1])

    log(f"\n  Hot numbers (>130% expected):")
    for num, cnt in hot[:10]:
        log(f"    #{num}: {cnt} times ({cnt/expected*100:.0f}% of expected)")

    log(f"\n  Cold numbers (<70% expected):")
    for num, cnt in cold[:10]:
        log(f"    #{num}: {cnt} times ({cnt/expected*100:.0f}% of expected)")


# ═══════════════════════════════════════════════════════════════════════
# TEST 11: Confidence Threshold Analysis
# ═══════════════════════════════════════════════════════════════════════

def test_confidence_thresholds(spins, warmup=50):
    """Test different BET thresholds — when should the system say BET vs WAIT?"""
    log(f"\n{'='*80}")
    log(f"  TEST 11: CONFIDENCE THRESHOLD ANALYSIS")
    log(f"  Finding the optimal BET threshold")
    log(f"{'='*80}")

    from app.ml.confidence import ConfidenceEngine

    freq = FrequencyAnalyzer()
    markov = MarkovChain()
    pattern = PatternDetector()
    conf_engine = ConfidenceEngine()

    # Collect (confidence, hit) pairs
    conf_hit_pairs = []

    for i, actual in enumerate(spins):
        if i >= warmup:
            fp = freq.get_number_probabilities()
            mp = markov.get_probabilities()
            pp = pattern.get_number_probabilities()

            ens = 0.35 * fp + 0.40 * mp + 0.05 * pp
            ens /= ens.sum()

            top12 = set(np.argsort(ens)[-12:])
            hit = actual in top12

            # Get confidence
            pattern_strength = pattern.get_pattern_strength()
            all_dists = [fp, mp, pp, np.full(TOTAL_NUMBERS, 1.0/TOTAL_NUMBERS)]
            breakdown = conf_engine.get_breakdown(
                all_dists, len(freq.spin_history), pattern_strength, freq.spin_history
            )
            confidence = breakdown['overall']

            conf_hit_pairs.append((confidence, hit))

            # Record for confidence engine learning
            pred_record = {
                'numbers': list(top12)[:12],
                'color': None, 'dozen': None,
                'high_low': None, 'odd_even': None,
                'actual_color': None, 'actual_dozen': None,
            }
            conf_engine.record_prediction(pred_record, actual)

        freq.update(actual)
        markov.update(actual)
        pattern.update(actual)

    # Test different thresholds
    thresholds = list(range(10, 91, 5))
    log(f"\n  {'Threshold':<12} {'BET spins':<12} {'BET hits':<10} {'BET rate':<10} {'WAIT spins':<12} {'WAIT hits':<11} {'WAIT rate':<10} {'Better?'}")
    log(f"  {'-'*88}")

    for threshold in thresholds:
        bet_hits = sum(1 for c, h in conf_hit_pairs if c >= threshold and h)
        bet_total = sum(1 for c, h in conf_hit_pairs if c >= threshold)
        wait_hits = sum(1 for c, h in conf_hit_pairs if c < threshold and h)
        wait_total = sum(1 for c, h in conf_hit_pairs if c < threshold)

        bet_rate = bet_hits / max(1, bet_total) * 100
        wait_rate = wait_hits / max(1, wait_total) * 100
        better = "✓ BET > WAIT" if bet_rate > wait_rate else "✗ WAIT ≥ BET"

        marker = ""
        if abs(threshold - 35) < 1:
            marker = " ← current"

        log(f"  {threshold:<12} {bet_total:<12} {bet_hits:<10} {bet_rate:<9.1f}% {wait_total:<12} {wait_hits:<11} {wait_rate:<9.1f}% {better}{marker}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN — Run Everything
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    start_time = time.time()

    # Clear results file
    with open(RESULTS_FILE, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"  COMPREHENSIVE MODEL BACKTEST — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")

    spins = load_userdata()
    if not spins:
        print("ERROR: No spin data found in userdata/")
        sys.exit(1)

    log(f"Loaded {len(spins)} spins from userdata/")
    log(f"Started at {time.strftime('%H:%M:%S')}")

    # ─── Run all tests ───────────────────────────────────────────
    t0 = time.time()

    # Test 1: Fine-grained weight grid (5% steps)
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 1...")
    grid_results = test_weight_grid(spins, warmup=50, step=5)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 2: Individual model analysis
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 2...")
    model_stats = test_individual_models(spins, warmup=50)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 3: Frequency blend sweep
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 3...")
    freq_results = test_frequency_blend(spins, warmup=50)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 4: Markov blend sweep
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 4...")
    markov_results = test_markov_blend(spins, warmup=50)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 5: Rolling performance
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 5...")
    rolling = test_rolling_performance(spins, warmup=50)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 6: Monte Carlo (this is the slow one)
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 6 (Monte Carlo — will take a while)...")
    mc_real, mc_shuffle = test_monte_carlo(spins, warmup=50, n_shuffles=500)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 7: LSTM backtest
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 7 (LSTM)...")
    lstm_results = test_lstm_model(spins, warmup=100)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 8: 4-model grid (coarser)
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 8 (4-model grid)...")
    grid4_results = test_4model_grid(spins, warmup=50, step=10)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 9: Anchor strategies
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 9...")
    anchor_results = test_anchor_strategies(spins, warmup=50)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 10: Data quality
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 10...")
    test_data_quality(spins)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # Test 11: Confidence thresholds
    t0 = time.time()
    log(f"\n[{time.strftime('%H:%M:%S')}] Starting Test 11...")
    test_confidence_thresholds(spins, warmup=50)
    log(f"  Completed in {time.time()-t0:.0f}s")

    # ─── Final Summary ───────────────────────────────────────────
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    log(f"\n{'='*80}")
    log(f"  ALL TESTS COMPLETE — Total time: {hours}h {minutes}m {seconds}s")
    log(f"  Results saved to: {RESULTS_FILE}")
    log(f"{'='*80}")

    # Extract top recommendations
    if grid_results:
        best = grid_results[0]
        log(f"\n  🏆 BEST 3-MODEL COMBO: F{best[0]*100:.0f}/M{best[1]*100:.0f}/P{best[2]*100:.0f} = {best[4]:.1f}%")

    if grid4_results:
        best4 = grid4_results[0]
        log(f"  🏆 BEST 4-MODEL COMBO: F{best4[0]*100:.0f}/M{best4[1]*100:.0f}/P{best4[2]*100:.0f}/L{best4[3]*100:.0f} = {best4[5]:.1f}%")

    log(f"  📊 Random baseline: {TOP_N/TOTAL_NUMBERS*100:.1f}%")
    log(f"\n  Full results at: {RESULTS_FILE}")
