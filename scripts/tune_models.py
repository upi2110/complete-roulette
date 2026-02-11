#!/usr/bin/env python3
"""
Model Tuning Script — Walk-forward validation on real userdata only.

For each model, tests different parameter combinations by:
  1. Loading all real userdata (data1-data5.txt)
  2. Walking through the data chronologically
  3. At each step: predict top-N, then reveal actual → measure hit rate
  4. Reports best params per model

NO test data is used. Only userdata/*.txt files.
"""

import sys
import os
import time
import json
import numpy as np
from collections import Counter, defaultdict
from itertools import product as iprod

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    TOTAL_NUMBERS, WHEEL_ORDER, NUMBER_TO_POSITION,
    RED_NUMBERS, BLACK_NUMBERS, WHEEL_SECTORS,
    HOT_NUMBER_THRESHOLD, COLD_NUMBER_THRESHOLD,
    USERDATA_DIR,
)


# ─── Data Loading ──────────────────────────────────────────────────────
def load_all_userdata():
    """Load all spin numbers from userdata/*.txt files."""
    all_spins = []
    data_dir = USERDATA_DIR
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.txt'):
            fpath = os.path.join(data_dir, fname)
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        n = int(line)
                        if 0 <= n <= 36:
                            all_spins.append(n)
    return all_spins


# ─── Walk-Forward Evaluation ───────────────────────────────────────────
def evaluate_model(model_fn, data, warmup=100, top_n=12):
    """Walk-forward: train on data[:i], predict i, check hit.

    model_fn(history) -> 37-element probability array

    Returns dict with hit rates for different top-N sizes.
    """
    hits = {k: 0 for k in [3, 5, 8, 10, 12]}
    total = 0

    for i in range(warmup, len(data)):
        history = data[:i]
        actual = data[i]

        probs = model_fn(history)
        ranked = np.argsort(probs)[::-1]

        total += 1
        for k in hits:
            if actual in ranked[:k]:
                hits[k] += 1

    rates = {k: round(hits[k] / total * 100, 2) if total > 0 else 0
             for k in hits}
    rates['total_predictions'] = total
    # Baselines: random top-K out of 37
    rates['baselines'] = {k: round(k / 37 * 100, 2) for k in hits}
    return rates


# ═══════════════════════════════════════════════════════════════════════
# MODEL 1: FREQUENCY ANALYZER TUNING
# ═══════════════════════════════════════════════════════════════════════
def tune_frequency(data):
    print("\n" + "=" * 70)
    print("TUNING: Frequency Analyzer")
    print("=" * 70)

    # Parameters to test
    decay_factors = [0.990, 0.995, 0.998, 0.999, 1.0]
    flat_weights = [0.2, 0.4, 0.6, 0.8, 1.0]

    best_rate = 0
    best_params = {}
    results = []

    for decay, flat_w in iprod(decay_factors, flat_weights):
        recent_w = 1.0 - flat_w

        def model_fn(history, _decay=decay, _flat_w=flat_w, _recent_w=recent_w):
            counts = Counter(history)
            total = len(history)

            # Flat Laplace-smoothed
            flat_probs = np.array([
                (counts.get(i, 0) + 1) / (total + TOTAL_NUMBERS)
                for i in range(TOTAL_NUMBERS)
            ])
            flat_probs /= flat_probs.sum()

            # Time-weighted
            n = len(history)
            weighted_counts = np.ones(TOTAL_NUMBERS, dtype=np.float64)
            for idx, num in enumerate(history):
                weight = _decay ** (n - 1 - idx)
                weighted_counts[num] += weight
            recent_probs = weighted_counts / weighted_counts.sum()

            blended = _flat_w * flat_probs + _recent_w * recent_probs
            blended /= blended.sum()
            return blended

        rates = evaluate_model(model_fn, data, warmup=100)
        result = {
            'decay': decay,
            'flat_weight': flat_w,
            'recent_weight': recent_w,
            **rates
        }
        results.append(result)

        if rates[12] > best_rate:
            best_rate = rates[12]
            best_params = result

        print(f"  decay={decay:.3f} flat={flat_w:.1f} → "
              f"top3={rates[3]:.1f}% top5={rates[5]:.1f}% "
              f"top8={rates[8]:.1f}% top12={rates[12]:.1f}%")

    print(f"\n  BEST: decay={best_params['decay']} flat={best_params['flat_weight']} "
          f"→ top12={best_rate:.2f}% (baseline={best_params['baselines'][12]:.1f}%)")
    return best_params, results


# ═══════════════════════════════════════════════════════════════════════
# MODEL 2: MARKOV CHAIN TUNING
# ═══════════════════════════════════════════════════════════════════════
def tune_markov(data):
    print("\n" + "=" * 70)
    print("TUNING: Markov Chain")
    print("=" * 70)

    # Parameters to test
    order1_weights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    smoothing_vals = [0.1, 0.3, 0.5, 1.0, 2.0]

    best_rate = 0
    best_params = {}
    results = []

    for o1w, smooth in iprod(order1_weights, smoothing_vals):
        o2w = 1.0 - o1w

        def model_fn(history, _o1w=o1w, _o2w=o2w, _smooth=smooth):
            if len(history) < 3:
                return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

            # Build transition matrices
            counts_1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
            counts_2 = defaultdict(lambda: np.zeros(TOTAL_NUMBERS))

            for j in range(1, len(history)):
                counts_1[history[j - 1]][history[j]] += 1
                if j >= 2:
                    key = (history[j - 2], history[j - 1])
                    counts_2[key][history[j]] += 1

            current = history[-1]
            smoothed_1 = counts_1[current] + _smooth
            p1 = smoothed_1 / smoothed_1.sum()

            key = (history[-2], history[-1])
            if key in counts_2:
                smoothed_2 = counts_2[key] + _smooth
                p2 = smoothed_2 / smoothed_2.sum()
            else:
                p2 = p1

            combined = _o1w * p1 + _o2w * p2
            combined /= combined.sum()
            return combined

        rates = evaluate_model(model_fn, data, warmup=100)
        result = {
            'order1_weight': o1w,
            'order2_weight': o2w,
            'smoothing': smooth,
            **rates
        }
        results.append(result)

        if rates[12] > best_rate:
            best_rate = rates[12]
            best_params = result

        print(f"  o1={o1w:.1f} o2={o2w:.1f} smooth={smooth:.1f} → "
              f"top3={rates[3]:.1f}% top5={rates[5]:.1f}% "
              f"top8={rates[8]:.1f}% top12={rates[12]:.1f}%")

    print(f"\n  BEST: o1={best_params['order1_weight']} o2={best_params['order2_weight']} "
          f"smooth={best_params['smoothing']} → top12={best_rate:.2f}% "
          f"(baseline={best_params['baselines'][12]:.1f}%)")
    return best_params, results


# ═══════════════════════════════════════════════════════════════════════
# MODEL 3: PATTERN DETECTOR TUNING
# ═══════════════════════════════════════════════════════════════════════
def tune_patterns(data):
    print("\n" + "=" * 70)
    print("TUNING: Pattern Detector")
    print("=" * 70)

    # Parameters to test
    sector_thresholds = [1.2, 1.3, 1.4, 1.5]
    sector_boosts = [1.1, 1.2, 1.3, 1.5]
    repeater_boosts = [0.5, 1.0, 1.5, 2.0]
    lookbacks = [10, 20, 30, 50]

    best_rate = 0
    best_params = {}
    results = []

    for thresh, sboost, rboost, lookback in iprod(
            sector_thresholds, sector_boosts, repeater_boosts, lookbacks):

        def model_fn(history, _thresh=thresh, _sboost=sboost,
                     _rboost=rboost, _lookback=lookback):
            probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
            if len(history) < 20:
                return probs

            recent = history[-_lookback:] if len(history) >= _lookback else history

            # Sector bias boost
            total_recent = len(recent)
            for sector_name, sector_nums in WHEEL_SECTORS.items():
                sector_count = sum(1 for n in recent if n in sector_nums)
                expected = total_recent * len(sector_nums) / TOTAL_NUMBERS
                if expected > 0:
                    ratio = sector_count / expected
                    if ratio >= _thresh:
                        for n in sector_nums:
                            probs[n] *= _sboost

            # Repeater boost
            recent_counts = Counter(recent)
            for num, count in recent_counts.items():
                if count >= 2:
                    repeat_rate = count / total_recent
                    probs[num] *= (1 + repeat_rate * _rboost)

            probs /= probs.sum()
            return probs

        rates = evaluate_model(model_fn, data, warmup=100)
        result = {
            'sector_threshold': thresh,
            'sector_boost': sboost,
            'repeater_boost': rboost,
            'lookback': lookback,
            **rates
        }
        results.append(result)

        if rates[12] > best_rate:
            best_rate = rates[12]
            best_params = result

    print(f"  Tested {len(results)} combinations")
    print(f"  BEST: thresh={best_params['sector_threshold']} "
          f"sboost={best_params['sector_boost']} "
          f"rboost={best_params['repeater_boost']} "
          f"lookback={best_params['lookback']} "
          f"→ top12={best_rate:.2f}% (baseline={best_params['baselines'][12]:.1f}%)")

    # Show top 5
    results.sort(key=lambda x: x[12], reverse=True)
    print("\n  Top 5 combinations:")
    for r in results[:5]:
        print(f"    thresh={r['sector_threshold']} sboost={r['sector_boost']} "
              f"rboost={r['repeater_boost']} lb={r['lookback']} → top12={r[12]:.1f}%")

    return best_params, results


# ═══════════════════════════════════════════════════════════════════════
# MODEL 4: ENSEMBLE WEIGHT TUNING
# ═══════════════════════════════════════════════════════════════════════
def tune_ensemble(data, best_freq_params, best_markov_params, best_pattern_params):
    print("\n" + "=" * 70)
    print("TUNING: Ensemble Weights")
    print("=" * 70)

    # Use best params from individual models
    freq_decay = best_freq_params.get('decay', 0.998)
    freq_flat = best_freq_params.get('flat_weight', 0.4)
    markov_o1 = best_markov_params.get('order1_weight', 0.6)
    markov_smooth = best_markov_params.get('smoothing', 0.5)
    pat_thresh = best_pattern_params.get('sector_threshold', 1.4)
    pat_sboost = best_pattern_params.get('sector_boost', 1.3)
    pat_rboost = best_pattern_params.get('repeater_boost', 1.0)
    pat_lookback = best_pattern_params.get('lookback', 20)

    # Ensemble weight combinations (must sum to ~1.0)
    # freq, markov, pattern weights (LSTM tested separately)
    weight_combos = [
        (0.30, 0.50, 0.20),  # Markov heavy
        (0.30, 0.40, 0.30),  # Balanced freq/pattern
        (0.40, 0.40, 0.20),  # Freq+Markov
        (0.50, 0.30, 0.20),  # Frequency heavy
        (0.20, 0.60, 0.20),  # Markov dominant
        (0.35, 0.35, 0.30),  # Balanced
        (0.25, 0.50, 0.25),  # Markov focused
        (0.40, 0.30, 0.30),  # Freq+Pattern
        (0.20, 0.40, 0.40),  # Pattern heavy
        (0.33, 0.34, 0.33),  # Equal
        (0.45, 0.45, 0.10),  # Freq+Markov, low pattern
        (0.50, 0.40, 0.10),  # Current-like (freq+markov heavy)
        (0.30, 0.60, 0.10),  # Markov dominant, low pattern
        (0.60, 0.30, 0.10),  # Frequency dominant
        (0.20, 0.70, 0.10),  # Strong Markov
    ]

    best_rate = 0
    best_params = {}
    results = []

    for fw, mw, pw in weight_combos:

        def model_fn(history, _fw=fw, _mw=mw, _pw=pw):
            if len(history) < 20:
                return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

            # Frequency model
            counts = Counter(history)
            total = len(history)
            flat_probs = np.array([
                (counts.get(i, 0) + 1) / (total + TOTAL_NUMBERS)
                for i in range(TOTAL_NUMBERS)
            ])
            flat_probs /= flat_probs.sum()
            n = len(history)
            weighted_counts = np.ones(TOTAL_NUMBERS, dtype=np.float64)
            for idx, num in enumerate(history):
                weight = freq_decay ** (n - 1 - idx)
                weighted_counts[num] += weight
            recent_probs = weighted_counts / weighted_counts.sum()
            freq_probs = freq_flat * flat_probs + (1.0 - freq_flat) * recent_probs
            freq_probs /= freq_probs.sum()

            # Markov model
            if len(history) >= 3:
                counts_1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
                counts_2 = defaultdict(lambda: np.zeros(TOTAL_NUMBERS))
                for j in range(1, len(history)):
                    counts_1[history[j - 1]][history[j]] += 1
                    if j >= 2:
                        key = (history[j - 2], history[j - 1])
                        counts_2[key][history[j]] += 1
                current = history[-1]
                smoothed_1 = counts_1[current] + markov_smooth
                p1 = smoothed_1 / smoothed_1.sum()
                key = (history[-2], history[-1])
                if key in counts_2:
                    smoothed_2 = counts_2[key] + markov_smooth
                    p2 = smoothed_2 / smoothed_2.sum()
                else:
                    p2 = p1
                markov_probs = markov_o1 * p1 + (1.0 - markov_o1) * p2
                markov_probs /= markov_probs.sum()
            else:
                markov_probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

            # Pattern model
            pat_probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
            recent = history[-pat_lookback:] if len(history) >= pat_lookback else history
            total_recent = len(recent)
            for sector_name, sector_nums in WHEEL_SECTORS.items():
                sector_count = sum(1 for nn in recent if nn in sector_nums)
                expected = total_recent * len(sector_nums) / TOTAL_NUMBERS
                if expected > 0 and sector_count / expected >= pat_thresh:
                    for nn in sector_nums:
                        pat_probs[nn] *= pat_sboost
            recent_counts = Counter(recent)
            for num, count in recent_counts.items():
                if count >= 2:
                    pat_probs[num] *= (1 + (count / total_recent) * pat_rboost)
            pat_probs /= pat_probs.sum()

            # Ensemble blend
            ensemble = _fw * freq_probs + _mw * markov_probs + _pw * pat_probs
            ensemble /= ensemble.sum()
            return ensemble

        rates = evaluate_model(model_fn, data, warmup=100)
        result = {
            'freq_weight': fw,
            'markov_weight': mw,
            'pattern_weight': pw,
            **rates
        }
        results.append(result)

        if rates[12] > best_rate:
            best_rate = rates[12]
            best_params = result

        print(f"  F={fw:.2f} M={mw:.2f} P={pw:.2f} → "
              f"top3={rates[3]:.1f}% top5={rates[5]:.1f}% "
              f"top8={rates[8]:.1f}% top12={rates[12]:.1f}%")

    print(f"\n  BEST: F={best_params['freq_weight']} M={best_params['markov_weight']} "
          f"P={best_params['pattern_weight']} → top12={best_rate:.2f}%")
    return best_params, results


# ═══════════════════════════════════════════════════════════════════════
# MODEL 5: PREDICTION CONFIDENCE FACTOR TUNING
# ═══════════════════════════════════════════════════════════════════════
def tune_prediction_factor(data, best_ensemble_params, best_freq, best_markov, best_pattern):
    print("\n" + "=" * 70)
    print("TUNING: Prediction Confidence Factor (number selection threshold)")
    print("=" * 70)

    factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]

    fw = best_ensemble_params.get('freq_weight', 0.30)
    mw = best_ensemble_params.get('markov_weight', 0.40)
    pw = best_ensemble_params.get('pattern_weight', 0.30)

    freq_decay = best_freq.get('decay', 0.998)
    freq_flat = best_freq.get('flat_weight', 0.4)
    markov_o1 = best_markov.get('order1_weight', 0.6)
    markov_smooth = best_markov.get('smoothing', 0.5)
    pat_thresh = best_pattern.get('sector_threshold', 1.4)
    pat_sboost = best_pattern.get('sector_boost', 1.3)
    pat_rboost = best_pattern.get('repeater_boost', 1.0)
    pat_lookback = best_pattern.get('lookback', 20)

    warmup = 100
    results = []

    for factor in factors:
        hits = 0
        total = 0
        avg_numbers = []

        for i in range(warmup, len(data)):
            history = data[:i]
            actual = data[i]

            if len(history) < 20:
                continue

            # Build ensemble (same as tune_ensemble best)
            counts = Counter(history)
            tot = len(history)
            flat_p = np.array([(counts.get(j, 0) + 1) / (tot + TOTAL_NUMBERS)
                               for j in range(TOTAL_NUMBERS)])
            flat_p /= flat_p.sum()
            n = len(history)
            wc = np.ones(TOTAL_NUMBERS, dtype=np.float64)
            for idx, num in enumerate(history):
                wc[num] += freq_decay ** (n - 1 - idx)
            rp = wc / wc.sum()
            freq_probs = freq_flat * flat_p + (1 - freq_flat) * rp
            freq_probs /= freq_probs.sum()

            if len(history) >= 3:
                c1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
                c2 = defaultdict(lambda: np.zeros(TOTAL_NUMBERS))
                for j in range(1, len(history)):
                    c1[history[j - 1]][history[j]] += 1
                    if j >= 2:
                        c2[(history[j - 2], history[j - 1])][history[j]] += 1
                cur = history[-1]
                s1 = c1[cur] + markov_smooth
                p1 = s1 / s1.sum()
                k = (history[-2], history[-1])
                if k in c2:
                    s2 = c2[k] + markov_smooth
                    p2 = s2 / s2.sum()
                else:
                    p2 = p1
                markov_probs = markov_o1 * p1 + (1 - markov_o1) * p2
                markov_probs /= markov_probs.sum()
            else:
                markov_probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

            pat_probs = np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)
            recent = history[-pat_lookback:]
            tr = len(recent)
            for sname, snums in WHEEL_SECTORS.items():
                sc = sum(1 for nn in recent if nn in snums)
                exp = tr * len(snums) / TOTAL_NUMBERS
                if exp > 0 and sc / exp >= pat_thresh:
                    for nn in snums:
                        pat_probs[nn] *= pat_sboost
            rc = Counter(recent)
            for num, cnt in rc.items():
                if cnt >= 2:
                    pat_probs[num] *= (1 + (cnt / tr) * pat_rboost)
            pat_probs /= pat_probs.sum()

            ensemble = fw * freq_probs + mw * markov_probs + pw * pat_probs
            ensemble /= ensemble.sum()

            # Apply confidence factor threshold
            uniform = 1.0 / TOTAL_NUMBERS
            threshold = uniform * factor
            selected = [j for j in range(TOTAL_NUMBERS) if ensemble[j] >= threshold]

            # At least 3 numbers
            if len(selected) < 3:
                ranked = np.argsort(ensemble)[::-1]
                selected = [int(ranked[j]) for j in range(3)]

            avg_numbers.append(len(selected))
            total += 1
            if actual in selected:
                hits += 1

        hit_rate = round(hits / total * 100, 2) if total > 0 else 0
        avg_n = round(np.mean(avg_numbers), 1) if avg_numbers else 0
        baseline = round(avg_n / 37 * 100, 1) if avg_n > 0 else 0
        edge = round(hit_rate - baseline, 2)

        results.append({
            'factor': factor,
            'hit_rate': hit_rate,
            'avg_numbers': avg_n,
            'baseline': baseline,
            'edge': edge,
            'total': total,
        })

        print(f"  factor={factor:.1f} → hit={hit_rate:.1f}% avg_nums={avg_n:.0f} "
              f"baseline={baseline:.1f}% edge={edge:+.1f}%")

    # Best by edge over baseline
    best = max(results, key=lambda x: x['edge'])
    print(f"\n  BEST by edge: factor={best['factor']} → "
          f"hit={best['hit_rate']:.1f}% nums={best['avg_numbers']:.0f} "
          f"edge={best['edge']:+.1f}%")
    return best, results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("AI ROULETTE MODEL TUNING — Walk-Forward Validation")
    print("Using ONLY real userdata (no test data)")
    print("=" * 70)

    data = load_all_userdata()
    print(f"\nLoaded {len(data)} spins from userdata/")
    print(f"Unique numbers: {len(set(data))}/37")
    print(f"Random baseline (top-12): {12/37*100:.1f}%")

    start = time.time()

    # Tune individual models
    best_freq, freq_results = tune_frequency(data)
    best_markov, markov_results = tune_markov(data)
    best_pattern, pattern_results = tune_patterns(data)

    # Tune ensemble with best individual params
    best_ensemble, ens_results = tune_ensemble(
        data, best_freq, best_markov, best_pattern)

    # Tune prediction threshold
    best_factor, factor_results = tune_prediction_factor(
        data, best_ensemble, best_freq, best_markov, best_pattern)

    elapsed = time.time() - start

    # ─── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Time: {elapsed:.0f}s")
    print(f"Data: {len(data)} spins")
    print(f"Random baseline (top-12): {12/37*100:.1f}%")

    print(f"\n--- Frequency Analyzer ---")
    print(f"  CURRENT: decay=0.998 flat=0.80 recent=0.20")
    print(f"  BEST:    decay={best_freq['decay']} flat={best_freq['flat_weight']} "
          f"recent={best_freq['recent_weight']}")
    print(f"  top12: {best_freq[12]}%")

    print(f"\n--- Markov Chain ---")
    print(f"  CURRENT: order1=0.6 order2=0.4 smoothing=0.5")
    print(f"  BEST:    order1={best_markov['order1_weight']} "
          f"order2={best_markov['order2_weight']} "
          f"smoothing={best_markov['smoothing']}")
    print(f"  top12: {best_markov[12]}%")

    print(f"\n--- Pattern Detector ---")
    print(f"  CURRENT: sector_thresh=1.4 sector_boost=1.3 repeater_boost=1.0 lookback=20")
    print(f"  BEST:    thresh={best_pattern['sector_threshold']} "
          f"boost={best_pattern['sector_boost']} "
          f"rboost={best_pattern['repeater_boost']} "
          f"lookback={best_pattern['lookback']}")
    print(f"  top12: {best_pattern[12]}%")

    print(f"\n--- Ensemble Weights ---")
    print(f"  CURRENT: freq=0.30 markov=0.40 pattern=0.05 (lstm=0.25 excluded)")
    print(f"  BEST:    freq={best_ensemble['freq_weight']} "
          f"markov={best_ensemble['markov_weight']} "
          f"pattern={best_ensemble['pattern_weight']}")
    print(f"  top12: {best_ensemble[12]}%")

    print(f"\n--- Prediction Confidence Factor ---")
    print(f"  CURRENT: 1.5")
    print(f"  BEST:    {best_factor['factor']} "
          f"(avg {best_factor['avg_numbers']:.0f} numbers, "
          f"edge={best_factor['edge']:+.1f}%)")

    # Save results
    output = {
        'data_size': len(data),
        'best_frequency': best_freq,
        'best_markov': best_markov,
        'best_pattern': best_pattern,
        'best_ensemble': best_ensemble,
        'best_factor': best_factor,
        'elapsed_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(BASE_DIR, 'scripts', 'tuning_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
