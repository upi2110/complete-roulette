"""
Comprehensive Model Diagnosis Script
=====================================
Analyzes why roulette prediction models are not achieving 50% hit rate.

Tests all 7 parts:
1. Data Analysis (chi-square, biases)
2. Walk-Forward Prediction Diagnosis
3. Model-by-Model Breakdown
4. Probability Calibration Check
5. Near-Miss Analysis
6. Autocorrelation / Pattern Check
7. Oracle Upper Bound
"""

import sys
import os
import numpy as np
from collections import Counter
from scipy import stats

# Add project root to path
sys.path.insert(0, '/Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso')

import config
from app.ml.ensemble import EnsemblePredictor

MODEL_NAMES = ['frequency', 'markov', 'patterns', 'lstm', 'wheel_strategy', 'hot_number']


def load_userdata_files():
    """Load all files from userdata/ directory."""
    userdata_dir = config.USERDATA_DIR
    datasets = {}

    for filename in sorted(os.listdir(userdata_dir)):
        if filename.endswith('.txt') and not filename.startswith('.'):
            filepath = os.path.join(userdata_dir, filename)
            with open(filepath, 'r') as f:
                numbers = [int(line.strip()) for line in f if line.strip().isdigit()]
            datasets[filename] = numbers

    return datasets


def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def chi_square_uniformity(data):
    """Test if data is uniformly distributed across 0-36."""
    counts = Counter(data)
    observed = [counts.get(i, 0) for i in range(37)]
    expected = [len(data) / 37] * 37

    chi2, p_value = stats.chisquare(observed, expected)
    return chi2, p_value, observed, expected


def part1_data_analysis(datasets):
    """Part 1: Analyze all userdata files for biases."""
    print_section("PART 1: DATA ANALYSIS")

    all_spins = []

    for filename, data in datasets.items():
        print(f"\n{filename}:")
        print(f"  Total spins: {len(data)}")

        # Chi-square test
        chi2, p_value, observed, expected = chi_square_uniformity(data)
        print(f"  Chi-square: {chi2:.2f}, p-value: {p_value:.4f}")

        if p_value < 0.05:
            print(f"  ⚠️  SIGNIFICANT DEVIATION from uniform (p < 0.05)")
        else:
            print(f"  ✓ Consistent with uniform distribution")

        # Most/least common numbers
        counts = Counter(data)
        most_common = counts.most_common(5)
        least_common = sorted(counts.items(), key=lambda x: x[1])[:5]

        print(f"  Most common: {[(num, cnt) for num, cnt in most_common]}")
        print(f"  Least common: {[(num, cnt) for num, cnt in least_common]}")

        # Expected count per number
        expected_count = len(data) / 37
        print(f"  Expected per number: {expected_count:.2f}")

        # Check for strong biases (>50% deviation from expected)
        strong_bias = []
        for num in range(37):
            actual = counts.get(num, 0)
            deviation = abs(actual - expected_count) / expected_count
            if deviation > 0.5:
                strong_bias.append((num, actual, deviation))

        if strong_bias:
            print(f"  ⚠️  Strong biases (>50% deviation):")
            for num, cnt, dev in strong_bias:
                print(f"     Number {num}: {cnt} times ({dev*100:.1f}% deviation)")

        all_spins.extend(data)

    # Combined analysis
    print(f"\n\nCOMBINED ANALYSIS (all {len(all_spins)} spins):")
    chi2, p_value, observed, expected = chi_square_uniformity(all_spins)
    print(f"  Chi-square: {chi2:.2f}, p-value: {p_value:.4f}")

    counts = Counter(all_spins)
    most_common = counts.most_common(5)
    least_common = sorted(counts.items(), key=lambda x: x[1])[:5]
    print(f"  Most common: {[(num, cnt) for num, cnt in most_common]}")
    print(f"  Least common: {[(num, cnt) for num, cnt in least_common]}")

    return all_spins


def part2_walkforward_diagnosis(data):
    """Part 2: Walk-forward prediction diagnosis."""
    print_section("PART 2: WALK-FORWARD PREDICTION DIAGNOSIS")

    print(f"\nUsing {len(data)} spins for walk-forward test")
    print("Starting predictions after 30 spins...\n")

    predictor = EnsemblePredictor()

    results = {
        'hits': 0,
        'misses': 0,
        'ranks': [],  # Rank of actual number in probability list
        'actual_probs': [],  # Probability assigned to actual number
        'threshold_probs': [],  # Probability of 12th-ranked number
    }

    # Feed spins one by one
    for i in range(len(data)):
        actual_number = data[i]

        if i >= 30:  # Start predicting after 30 spins
            # Get prediction
            prediction = predictor.predict()
            top_12 = prediction['top_numbers']

            # Get ensemble probabilities
            ensemble_probs, individual_dists = predictor.get_ensemble_probabilities()

            # Check if hit
            hit = actual_number in top_12
            if hit:
                results['hits'] += 1
            else:
                results['misses'] += 1

            # Find rank of actual number
            sorted_indices = np.argsort(ensemble_probs)[::-1]  # Descending
            rank = np.where(sorted_indices == actual_number)[0][0] + 1  # 1-indexed
            results['ranks'].append(rank)

            # Probability assigned to actual number
            actual_prob = ensemble_probs[actual_number]
            results['actual_probs'].append(actual_prob)

            # Probability of 12th-ranked number (threshold)
            threshold_prob = ensemble_probs[sorted_indices[11]]
            results['threshold_probs'].append(threshold_prob)

        # Update models
        predictor.update(actual_number)

    # Analysis
    total_predictions = results['hits'] + results['misses']
    hit_rate = results['hits'] / total_predictions * 100 if total_predictions > 0 else 0

    print(f"Total predictions: {total_predictions}")
    print(f"Hits: {results['hits']}")
    print(f"Misses: {results['misses']}")
    print(f"Hit Rate: {hit_rate:.2f}%")
    print(f"Expected (12/37): {12/37*100:.2f}%")
    print(f"Difference: {hit_rate - 12/37*100:+.2f}%")

    print(f"\nRank Statistics:")
    print(f"  Mean rank: {np.mean(results['ranks']):.1f}")
    print(f"  Median rank: {np.median(results['ranks']):.1f}")
    print(f"  Min rank: {min(results['ranks'])}")
    print(f"  Max rank: {max(results['ranks'])}")

    print(f"\nProbability Statistics:")
    print(f"  Mean prob (actual): {np.mean(results['actual_probs'])*100:.2f}%")
    print(f"  Mean prob (12th): {np.mean(results['threshold_probs'])*100:.2f}%")
    print(f"  Uniform: {1/37*100:.2f}%")

    return results, predictor


def part3_model_breakdown(data, results_from_part2):
    """Part 3: Model-by-model breakdown."""
    print_section("PART 3: MODEL-BY-MODEL BREAKDOWN")

    print(f"\nAnalyzing individual model predictions...")

    predictor = EnsemblePredictor()

    model_stats = {name: {'ranks': []} for name in MODEL_NAMES}

    # Feed spins one by one
    for i in range(len(data)):
        actual_number = data[i]

        if i >= 30:
            # Get individual model distributions
            ensemble_probs, individual_dists = predictor.get_ensemble_probabilities()

            # individual_dists = [freq, markov, patterns, lstm, wheel, hot]
            for idx, name in enumerate(MODEL_NAMES):
                model_probs = individual_dists[idx]

                # Find rank of actual number in this model's predictions
                sorted_indices = np.argsort(model_probs)[::-1]
                rank = np.where(sorted_indices == actual_number)[0][0] + 1
                model_stats[name]['ranks'].append(rank)

        predictor.update(actual_number)

    # Analysis
    print(f"\n{'Model':<20} {'Mean Rank':<12} {'Median Rank':<12} {'Top-12 Hit%':<12}")
    print("-" * 60)

    for name in MODEL_NAMES:
        ranks = model_stats[name]['ranks']
        if ranks:
            mean_rank = np.mean(ranks)
            median_rank = np.median(ranks)
            top_12_hits = sum(1 for r in ranks if r <= 12)
            hit_rate = top_12_hits / len(ranks) * 100

            print(f"{name:<20} {mean_rank:<12.1f} {median_rank:<12.0f} {hit_rate:<12.1f}")

    return model_stats


def part4_calibration_check(results):
    """Part 4: Probability calibration check."""
    print_section("PART 4: PROBABILITY CALIBRATION CHECK")

    print("\nGrouping predictions by probability assigned to actual number...")

    # Define bins
    bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0]
    bin_labels = ['0-1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-100%']

    bin_counts = {label: {'total': 0, 'hits': 0} for label in bin_labels}

    # We need to re-run to get hit information per prediction
    # For now, approximate from ranks
    for prob, rank in zip(results['actual_probs'], results['ranks']):
        hit = rank <= 12

        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            if low <= prob < high:
                label = bin_labels[i]
                bin_counts[label]['total'] += 1
                if hit:
                    bin_counts[label]['hits'] += 1
                break

    print(f"\n{'Prob Range':<12} {'Count':<10} {'Hit Rate':<12} {'Expected':<12}")
    print("-" * 50)

    for label in bin_labels:
        count = bin_counts[label]['total']
        hits = bin_counts[label]['hits']
        if count > 0:
            hit_rate = hits / count * 100
            expected = 32.4  # 12/37
            print(f"{label:<12} {count:<10} {hit_rate:<12.1f} {expected:<12.1f}")
        else:
            print(f"{label:<12} {count:<10} {'N/A':<12} {'N/A':<12}")


def part5_near_miss_analysis(results):
    """Part 5: Near-miss analysis."""
    print_section("PART 5: NEAR-MISS ANALYSIS")

    print("\nDistribution of actual number ranks:")

    ranks = results['ranks']

    # Define bins
    bins = [
        (1, 12, "Top 12 (HITS)"),
        (13, 15, "Just missed (13-15)"),
        (16, 20, "Close miss (16-20)"),
        (21, 37, "No clue (21+)")
    ]

    print(f"\n{'Range':<25} {'Count':<10} {'Percentage':<12}")
    print("-" * 50)

    for low, high, label in bins:
        count = sum(1 for r in ranks if low <= r <= high)
        percentage = count / len(ranks) * 100 if ranks else 0
        print(f"{label:<25} {count:<10} {percentage:<12.1f}")

    # Histogram
    print("\nRank histogram (1-37):")
    rank_counts = Counter(ranks)
    for r in range(1, 38):
        count = rank_counts.get(r, 0)
        bar = '█' * (count // 5)
        print(f"  Rank {r:2d}: {count:4d} {bar}")


def part6_autocorrelation_check(data):
    """Part 6: Autocorrelation / pattern check."""
    print_section("PART 6: AUTOCORRELATION / PATTERN CHECK")

    print("\nChecking for autocorrelation in the data...")

    # Simple lag-1 autocorrelation: does a number predict the next?
    # For each number, track what follows it
    transition_counts = {i: Counter() for i in range(37)}

    for i in range(len(data) - 1):
        current = data[i]
        next_num = data[i + 1]
        transition_counts[current][next_num] += 1

    # Check if any number has a strong preference for what follows
    strong_transitions = []
    for num in range(37):
        total = sum(transition_counts[num].values())
        if total >= 5:  # At least 5 occurrences
            most_common = transition_counts[num].most_common(1)
            if most_common:
                next_num, count = most_common[0]
                expected = total / 37
                if count > expected * 2:  # 2x expected
                    strong_transitions.append((num, next_num, count, total))

    if strong_transitions:
        print("\n⚠️  Strong transition patterns found:")
        for num, next_num, count, total in strong_transitions:
            prob = count / total * 100
            expected = 100 / 37
            print(f"  {num} → {next_num}: {count}/{total} ({prob:.1f}%, expected {expected:.1f}%)")
    else:
        print("\n✓ No strong transition patterns detected (consistent with random)")

    # Check for number repeats (same number twice in a row)
    repeats = sum(1 for i in range(len(data) - 1) if data[i] == data[i + 1])
    expected_repeats = len(data) / 37
    print(f"\nRepeats (same number twice in a row):")
    print(f"  Observed: {repeats}")
    print(f"  Expected: {expected_repeats:.1f}")
    print(f"  Ratio: {repeats / expected_repeats:.2f}x")

    # Test frequency model alone
    print("\n\nComparing FREQUENCY MODEL ALONE vs ENSEMBLE:")

    predictor = EnsemblePredictor()
    freq_hits = 0
    ensemble_hits = 0
    total_preds = 0

    for i in range(len(data)):
        actual_number = data[i]

        if i >= 30:
            # Frequency alone
            freq_probs = predictor.frequency.get_number_probabilities()
            freq_top_12 = np.argsort(freq_probs)[::-1][:12]
            if actual_number in freq_top_12:
                freq_hits += 1

            # Ensemble
            prediction = predictor.predict()
            if actual_number in prediction['top_numbers']:
                ensemble_hits += 1

            total_preds += 1

        predictor.update(actual_number)

    print(f"  Frequency alone: {freq_hits}/{total_preds} ({freq_hits/total_preds*100:.2f}%)")
    print(f"  Ensemble:        {ensemble_hits}/{total_preds} ({ensemble_hits/total_preds*100:.2f}%)")
    print(f"  Difference:      {(ensemble_hits - freq_hits)/total_preds*100:+.2f}%")


def part7_oracle_upper_bound(data):
    """Part 7: Oracle upper bound."""
    print_section("PART 7: ORACLE UPPER BOUND")

    print("\nWhat's the theoretical BEST hit rate achievable?")

    # If data is truly uniform, max is 12/37 = 32.4%
    uniform_max = 12 / 37 * 100
    print(f"\nIf data is uniform: {uniform_max:.2f}%")

    # If there are biases, we can do better by always picking top-12 most frequent
    counts = Counter(data)
    top_12_most_frequent = [num for num, _ in counts.most_common(12)]

    # Test oracle: always bet on these 12 numbers
    oracle_hits = sum(1 for num in data if num in top_12_most_frequent)
    oracle_rate = oracle_hits / len(data) * 100

    print(f"\nOracle (always bet top-12 most frequent): {oracle_rate:.2f}%")
    print(f"Top-12 numbers: {sorted(top_12_most_frequent)}")

    # Compare to our model
    print(f"\nOur model needs to approach {oracle_rate:.2f}% to be optimal")
    print(f"Gap between uniform and oracle: {oracle_rate - uniform_max:.2f}%")

    # Check entropy
    probs = np.array([counts.get(i, 0) / len(data) for i in range(37)])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(37)

    print(f"\nData entropy: {entropy:.3f} (max: {max_entropy:.3f})")
    print(f"Entropy ratio: {entropy/max_entropy*100:.1f}% (100% = perfectly random)")


def main():
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  COMPREHENSIVE ROULETTE MODEL DIAGNOSIS".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")

    # Load data
    datasets = load_userdata_files()
    print(f"\nLoaded {len(datasets)} files from userdata/")

    # Part 1: Data Analysis
    all_spins = part1_data_analysis(datasets)

    # Find largest dataset
    largest_file = max(datasets.items(), key=lambda x: len(x[1]))
    largest_data = largest_file[1]
    print(f"\n\nUsing LARGEST dataset ({largest_file[0]}) for detailed analysis:")
    print(f"  {len(largest_data)} spins")

    # Part 2: Walk-Forward Diagnosis
    results, predictor = part2_walkforward_diagnosis(largest_data)

    # Part 3: Model-by-Model Breakdown
    model_stats = part3_model_breakdown(largest_data, results)

    # Part 4: Calibration Check
    part4_calibration_check(results)

    # Part 5: Near-Miss Analysis
    part5_near_miss_analysis(results)

    # Part 6: Autocorrelation Check
    part6_autocorrelation_check(largest_data)

    # Part 7: Oracle Upper Bound
    part7_oracle_upper_bound(largest_data)

    print("\n" + "="*80)
    print("  DIAGNOSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
