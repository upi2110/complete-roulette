#!/usr/bin/env python3
"""
Backtest: Compare anchor+neighbours vs pure top-12 individual number selection.

Method 1: Current anchor+neighbours (±2) - uses _smart_anchor_selection
Method 2: Pure top-12 by probability - just take highest 12 numbers

Tests on all real spin data from userdata/ folder.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso')

import config
from app.ml.ensemble import EnsemblePredictor
import numpy as np

# Configuration
BET_PER_NUM = 2.0
NUM_PICKS = 12
WARMUP = 30
PROGRESS_INTERVAL = 500

def load_userdata():
    """Load all spin data from userdata/ folder."""
    userdata_dir = os.path.join(config.BASE_DIR, 'userdata')
    datasets = {}

    if not os.path.exists(userdata_dir):
        print(f"ERROR: userdata directory not found at {userdata_dir}")
        return datasets

    for filename in sorted(os.listdir(userdata_dir)):
        if filename.endswith('.txt') or filename.endswith('.csv'):
            filepath = os.path.join(userdata_dir, filename)
            numbers = []

            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and line.isdigit():
                            num = int(line)
                            if 0 <= num <= 36:
                                numbers.append(num)
            except Exception as e:
                print(f"ERROR reading {filename}: {e}")
                continue

            if numbers:
                datasets[filename] = numbers
                print(f"Loaded {filename}: {len(numbers)} spins")

    return datasets


def test_anchor_method(spins, dataset_name=""):
    """Method 1: Current anchor+neighbours system using _smart_anchor_selection."""
    predictor = EnsemblePredictor()
    hits = 0
    total = 0
    profit = 0.0
    prediction_sizes = []

    for i in range(len(spins) - 1):
        predictor.update(spins[i])

        if i < WARMUP:
            continue

        # Use the full predict() method which includes _smart_anchor_selection
        prediction = predictor.predict()
        predicted = set(int(n) for n in prediction['top_numbers'][:NUM_PICKS])
        prediction_sizes.append(len(predicted))
        actual = spins[i + 1]

        total += 1
        total_bet = BET_PER_NUM * len(predicted)

        if actual in predicted:
            hits += 1
            # Win on 1 number: get 35:1 payout + original bet back, minus all bets
            profit += (BET_PER_NUM * 35 + BET_PER_NUM) - total_bet
        else:
            profit -= total_bet

        # Progress indicator
        if total > 0 and total % PROGRESS_INTERVAL == 0:
            current_rate = (hits / total) * 100
            print(f"  [{dataset_name}] Anchor method: {total} predictions, {hits} hits ({current_rate:.1f}%)")

    avg_size = sum(prediction_sizes) / len(prediction_sizes) if prediction_sizes else 0

    return {
        'hits': hits,
        'total': total,
        'profit': profit,
        'avg_prediction_size': avg_size
    }


def test_individual_method(spins, dataset_name=""):
    """Method 2: Pure top-12 by individual probability (no anchors, no wheel grouping)."""
    predictor = EnsemblePredictor()
    hits = 0
    total = 0
    profit = 0.0
    prediction_sizes = []

    for i in range(len(spins) - 1):
        predictor.update(spins[i])

        if i < WARMUP:
            continue

        # Get raw ensemble probabilities
        ensemble_probs, individual_distributions = predictor.get_ensemble_probabilities()

        # Pick top 12 numbers by raw probability (no wheel logic)
        top_indices = np.argsort(ensemble_probs)[::-1][:NUM_PICKS]
        predicted = set(int(idx) for idx in top_indices)
        prediction_sizes.append(len(predicted))
        actual = spins[i + 1]

        total += 1
        total_bet = BET_PER_NUM * len(predicted)

        if actual in predicted:
            hits += 1
            profit += (BET_PER_NUM * 35 + BET_PER_NUM) - total_bet
        else:
            profit -= total_bet

        # Progress indicator
        if total > 0 and total % PROGRESS_INTERVAL == 0:
            current_rate = (hits / total) * 100
            print(f"  [{dataset_name}] Individual method: {total} predictions, {hits} hits ({current_rate:.1f}%)")

    avg_size = sum(prediction_sizes) / len(prediction_sizes) if prediction_sizes else 0

    return {
        'hits': hits,
        'total': total,
        'profit': profit,
        'avg_prediction_size': avg_size
    }


def print_results(method_name, results, random_baseline=32.43):
    """Print formatted results for a method."""
    hits = results['hits']
    total = results['total']
    profit = results['profit']
    avg_size = results['avg_prediction_size']

    if total == 0:
        print(f"\n{method_name}: No predictions made")
        return

    hit_rate = (hits / total) * 100
    edge = hit_rate - random_baseline

    print(f"\n{method_name}:")
    print(f"  Total Predictions: {total}")
    print(f"  Hits: {hits}")
    print(f"  Hit Rate: {hit_rate:.2f}%")
    print(f"  Edge vs Random: {edge:+.2f}%")
    print(f"  Profit/Loss: ${profit:.2f}")
    print(f"  Avg Numbers Predicted: {avg_size:.1f}")


def main():
    print("=" * 70)
    print("BACKTEST: Anchor+Neighbours vs Pure Top-12 Individual")
    print("=" * 70)

    # Load all datasets
    datasets = load_userdata()

    if not datasets:
        print("No data found in userdata/ folder")
        return

    # Combine all spins
    all_spins = []
    for filename in sorted(datasets.keys()):
        all_spins.extend(datasets[filename])

    print(f"\nTotal spins across {len(datasets)} files: {len(all_spins)}")
    print(f"Warmup period: {WARMUP} spins")
    print(f"Bet per number: ${BET_PER_NUM}")
    print(f"Numbers picked: {NUM_PICKS}")
    print(f"Random baseline (12/37): 32.43%")
    print("\n" + "=" * 70)

    # Test both methods on combined data
    print("\n### RUNNING OVERALL BACKTEST (all data combined) ###\n")

    print("Testing Method 1: Anchor+Neighbours (current system)...")
    anchor_results = test_anchor_method(all_spins, "Overall")

    print("\nTesting Method 2: Pure Top-12 Individual...")
    individual_results = test_individual_method(all_spins, "Overall")

    # Print overall results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print_results("Method 1: Anchor+Neighbours", anchor_results)
    print_results("Method 2: Pure Top-12 Individual", individual_results)

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    if anchor_results['total'] > 0 and individual_results['total'] > 0:
        anchor_rate = (anchor_results['hits'] / anchor_results['total']) * 100
        individual_rate = (individual_results['hits'] / individual_results['total']) * 100

        better = "Anchor+Neighbours" if anchor_rate > individual_rate else "Pure Individual"
        diff = abs(anchor_rate - individual_rate)

        print(f"\nWinner: {better}")
        print(f"Difference: {diff:.2f}%")
        print(f"\nProfit comparison:")
        print(f"  Anchor+Neighbours: ${anchor_results['profit']:.2f}")
        print(f"  Pure Individual: ${individual_results['profit']:.2f}")
        profit_diff = individual_results['profit'] - anchor_results['profit']
        print(f"  Individual advantage: ${profit_diff:+.2f}")

    # Per-dataset analysis
    print("\n" + "=" * 70)
    print("PER-DATASET ANALYSIS")
    print("=" * 70)

    for filename in sorted(datasets.keys()):
        spins = datasets[filename]
        print(f"\n### Dataset: {filename} ({len(spins)} spins) ###")

        anchor_ds = test_anchor_method(spins, filename)
        individual_ds = test_individual_method(spins, filename)

        print(f"\n{filename}:")

        if anchor_ds['total'] > 0:
            anchor_rate = (anchor_ds['hits'] / anchor_ds['total']) * 100
            print(f"  Anchor: {anchor_ds['hits']}/{anchor_ds['total']} = {anchor_rate:.1f}%, ${anchor_ds['profit']:.2f}")
        else:
            print(f"  Anchor: No predictions")

        if individual_ds['total'] > 0:
            ind_rate = (individual_ds['hits'] / individual_ds['total']) * 100
            print(f"  Individual: {individual_ds['hits']}/{individual_ds['total']} = {ind_rate:.1f}%, ${individual_ds['profit']:.2f}")
        else:
            print(f"  Individual: No predictions")


if __name__ == '__main__':
    main()
