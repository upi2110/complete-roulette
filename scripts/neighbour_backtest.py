#!/usr/bin/env python3
"""
Backtest comparing neighbour spread ±1, ±2, and ±3 on real spin data.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import config
from app.ml.ensemble import EnsemblePredictor
from collections import defaultdict


def load_spin_files(userdata_dir):
    """Load all spin data files from userdata/ directory."""
    userdata_path = Path(userdata_dir)
    if not userdata_path.exists():
        print(f"Warning: {userdata_dir} does not exist")
        return {}

    datasets = {}
    for file_path in userdata_path.glob("*.txt"):
        spins = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    num = int(line)
                    if 0 <= num <= 36:
                        spins.append(num)
        if spins:
            datasets[file_path.name] = spins
            print(f"Loaded {len(spins)} spins from {file_path.name}")

    return datasets


def run_backtest_for_neighbour_setting(datasets, neighbours_per_anchor, min_spins=30):
    """
    Run walk-forward backtest for a specific neighbour setting.

    Returns dict with results per dataset and aggregated.
    """
    print(f"\n{'='*80}")
    print(f"Testing NEIGHBOURS_PER_ANCHOR = ±{neighbours_per_anchor}")
    print(f"{'='*80}")

    # Override config
    config.NEIGHBOURS_PER_ANCHOR = neighbours_per_anchor
    config.TOP_PREDICTIONS_MAX = 12
    config.TOP_PREDICTIONS_COUNT = 12

    # Force reload of ensemble module to pick up new config values
    import importlib
    from app.ml import ensemble as ensemble_module
    importlib.reload(ensemble_module)
    from app.ml.ensemble import EnsemblePredictor

    results_per_dataset = {}
    total_predictions = 0
    total_hits = 0
    total_profit_loss = 0.0
    total_numbers_predicted = 0

    for dataset_name, spins in datasets.items():
        print(f"\nProcessing {dataset_name}...")

        # Create fresh predictor for this dataset
        predictor = EnsemblePredictor()

        hits = 0
        misses = 0
        profit_loss = 0.0
        numbers_predicted_sum = 0
        predictions_made = 0

        # Walk forward through spins
        for i in range(len(spins)):
            current_spin = spins[i]
            predictor.update(current_spin)

            # After min_spins, start making predictions
            if i >= min_spins and i < len(spins) - 1:
                # Get prediction
                prediction = predictor.predict()
                top_numbers = prediction.get('top_numbers', [])

                if len(top_numbers) > 0:
                    predictions_made += 1
                    numbers_predicted_sum += len(top_numbers)

                    # Next actual spin
                    next_spin = spins[i + 1]

                    # Check if next spin is in predicted set
                    if next_spin in top_numbers:
                        hits += 1
                        # Win: bet $2 per number, win on one number = $2*35 + $2 back - total bet
                        total_bet = 2 * len(top_numbers)
                        win_amount = 2 * 35 + 2  # $70 + $2 = $72
                        profit_loss += (win_amount - total_bet)
                    else:
                        misses += 1
                        # Loss: lose all bets
                        total_bet = 2 * len(top_numbers)
                        profit_loss -= total_bet

        # Store results for this dataset
        hit_rate = (hits / predictions_made * 100) if predictions_made > 0 else 0
        avg_numbers = (numbers_predicted_sum / predictions_made) if predictions_made > 0 else 0

        results_per_dataset[dataset_name] = {
            'predictions': predictions_made,
            'hits': hits,
            'misses': misses,
            'hit_rate': hit_rate,
            'profit_loss': profit_loss,
            'avg_numbers': avg_numbers
        }

        total_predictions += predictions_made
        total_hits += hits
        total_profit_loss += profit_loss
        total_numbers_predicted += numbers_predicted_sum

        print(f"  Predictions: {predictions_made}, Hits: {hits}, Hit Rate: {hit_rate:.2f}%, P/L: ${profit_loss:.2f}")

    # Calculate aggregated metrics
    overall_hit_rate = (total_hits / total_predictions * 100) if total_predictions > 0 else 0
    avg_numbers_overall = (total_numbers_predicted / total_predictions) if total_predictions > 0 else 0
    random_hit_rate = 32.43  # 12/37 * 100
    edge_vs_random = overall_hit_rate - random_hit_rate

    return {
        'neighbours': neighbours_per_anchor,
        'total_predictions': total_predictions,
        'total_hits': total_hits,
        'overall_hit_rate': overall_hit_rate,
        'random_hit_rate': random_hit_rate,
        'edge_vs_random': edge_vs_random,
        'total_profit_loss': total_profit_loss,
        'avg_numbers': avg_numbers_overall,
        'per_dataset': results_per_dataset
    }


def print_results_table(all_results):
    """Print formatted results table."""
    print("\n" + "="*100)
    print("BACKTEST RESULTS SUMMARY")
    print("="*100)
    print(f"{'Neighbour':<12} {'Predictions':<12} {'Hits':<8} {'Hit Rate':<12} {'Random':<10} {'Edge':<10} {'P/L':<12} {'Avg #s':<8}")
    print(f"{'Spread':<12} {'Made':<12} {'':<8} {'%':<12} {'%':<10} {'%':<10} {'($)':<12} {'':<8}")
    print("-"*100)

    for result in all_results:
        print(f"±{result['neighbours']:<11} "
              f"{result['total_predictions']:<12} "
              f"{result['total_hits']:<8} "
              f"{result['overall_hit_rate']:<12.2f} "
              f"{result['random_hit_rate']:<10.2f} "
              f"{result['edge_vs_random']:<+10.2f} "
              f"${result['total_profit_loss']:<11.2f} "
              f"{result['avg_numbers']:<8.2f}")

    print("="*100)

    # Print per-dataset breakdown for best performer
    if all_results:
        best = max(all_results, key=lambda x: x['overall_hit_rate'])
        print(f"\nPer-Dataset Breakdown for Best Performer (±{best['neighbours']}):")
        print("-"*100)
        print(f"{'Dataset':<30} {'Predictions':<12} {'Hits':<8} {'Hit Rate %':<12} {'P/L ($)':<12}")
        print("-"*100)
        for dataset_name, data in best['per_dataset'].items():
            print(f"{dataset_name:<30} "
                  f"{data['predictions']:<12} "
                  f"{data['hits']:<8} "
                  f"{data['hit_rate']:<12.2f} "
                  f"${data['profit_loss']:<11.2f}")
        print("-"*100)


def main():
    print("Neighbour Spread Backtest")
    print("="*100)

    userdata_dir = project_root / "userdata"

    # Load all datasets
    print("\nLoading spin data...")
    datasets = load_spin_files(userdata_dir)

    if not datasets:
        print("No data files found in userdata/")
        return

    total_spins = sum(len(spins) for spins in datasets.values())
    print(f"\nTotal datasets: {len(datasets)}")
    print(f"Total spins: {total_spins}")

    # Test each neighbour setting
    neighbour_settings = [1, 2, 3]
    all_results = []

    for neighbours in neighbour_settings:
        result = run_backtest_for_neighbour_setting(datasets, neighbours)
        all_results.append(result)

    # Print summary table
    print_results_table(all_results)

    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
