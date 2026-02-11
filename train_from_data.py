#!/usr/bin/env python3
"""
Training Script - Load historical spin data from file and train all AI models.

Usage:
    python train_from_data.py <data_file>
    python train_from_data.py data/my_spins.txt
    python train_from_data.py data/my_spins.csv

Supported formats:
    - .txt or .csv with one number (0-36) per line
    - Lines with non-numeric content are skipped (headers, comments)
    - Most recent number should be at the bottom of the file
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SESSIONS_DIR, MODELS_DIR, TOTAL_NUMBERS,
    LSTM_EPOCHS, RETRAIN_INTERVAL, get_number_color
)
from app.ml.frequency_analyzer import FrequencyAnalyzer
from app.ml.markov_chain import MarkovChain
from app.ml.pattern_detector import PatternDetector
from app.ml.lstm_predictor import LSTMPredictor
from app.ml.confidence import ConfidenceEngine
from app.ml.ensemble import EnsemblePredictor


def load_data(filepath):
    """Load spin numbers from a text/csv file (one number per line)."""
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        sys.exit(1)

    numbers = []
    skipped = 0
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Handle CSV with multiple columns - take first column
            parts = line.split(',')
            value = parts[0].strip()

            try:
                num = int(value)
                if 0 <= num <= 36:
                    numbers.append(num)
                else:
                    skipped += 1
                    print(f"  WARNING: Line {line_num}: '{num}' out of range (0-36), skipped")
            except ValueError:
                skipped += 1
                if line_num <= 3:
                    # Likely a header row
                    pass
                else:
                    print(f"  WARNING: Line {line_num}: '{line}' is not a number, skipped")

    return numbers, skipped


def validate_data(numbers):
    """Basic validation of the dataset."""
    from collections import Counter
    counts = Counter(numbers)

    print(f"\n  Dataset Validation:")
    print(f"  {'─' * 40}")
    print(f"  Total spins:        {len(numbers)}")
    print(f"  Unique numbers:     {len(counts)}/37")
    print(f"  Min number:         {min(numbers)}")
    print(f"  Max number:         {max(numbers)}")

    # Check for any dominant biases
    most_common = counts.most_common(5)
    least_common = counts.most_common()[-5:]
    expected = len(numbers) / TOTAL_NUMBERS

    print(f"  Expected frequency: {expected:.1f} per number")
    print(f"\n  Top 5 most frequent:")
    for num, count in most_common:
        ratio = count / expected
        color = get_number_color(num)
        bar = "█" * int(ratio * 10)
        print(f"    #{num:2d} ({color:5s}): {count:3d} hits ({ratio:.2f}x expected) {bar}")

    print(f"\n  Top 5 least frequent:")
    for num, count in least_common:
        ratio = count / expected
        color = get_number_color(num)
        print(f"    #{num:2d} ({color:5s}): {count:3d} hits ({ratio:.2f}x expected)")

    # Color distribution
    reds = sum(1 for n in numbers if get_number_color(n) == 'red')
    blacks = sum(1 for n in numbers if get_number_color(n) == 'black')
    greens = sum(1 for n in numbers if get_number_color(n) == 'green')
    total = len(numbers)

    print(f"\n  Color Distribution:")
    print(f"    Red:   {reds:3d} ({reds/total*100:.1f}%) [expected 48.6%]")
    print(f"    Black: {blacks:3d} ({blacks/total*100:.1f}%) [expected 48.6%]")
    print(f"    Green: {greens:3d} ({greens/total*100:.1f}%) [expected 2.7%]")

    return True


def save_as_session(numbers, source_file):
    """Save the imported data as a training session for future use."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    session_id = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = {
        'id': session_id,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'status': 'imported',
        'source_file': os.path.basename(source_file),
        'spins': [
            {
                'spin_number': i + 1,
                'actual_number': num,
                'timestamp': datetime.now().isoformat()
            }
            for i, num in enumerate(numbers)
        ],
        'predictions': [],
        'bets': [],
        'bankroll_history': [],
        'stats': {
            'total_spins': len(numbers),
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'skips': 0,
            'starting_bankroll': 4000,
            'ending_bankroll': 4000,
            'peak_bankroll': 4000,
            'lowest_bankroll': 4000,
            'max_consecutive_losses': 0,
            'best_prediction_streak': 0,
            'lstm_trained': False
        }
    }

    filepath = os.path.join(SESSIONS_DIR, f'session_{session_id}.json')
    with open(filepath, 'w') as f:
        json.dump(session, f, indent=2)

    print(f"\n  Session saved: {filepath}")
    return session_id


def train_all_models(numbers):
    """Train all ML models on the dataset."""

    print(f"\n{'═' * 60}")
    print(f"  TRAINING ALL MODELS")
    print(f"{'═' * 60}")

    # ─── 1. Frequency Analyzer ──────────────────────────────
    print(f"\n  [1/5] Frequency Analyzer...")
    t0 = time.time()
    freq = FrequencyAnalyzer()
    freq.load_history(numbers)
    chi = freq.get_chi_square_result()
    bias = freq.get_bias_score()
    hot = freq.get_hot_numbers(5)
    cold = freq.get_cold_numbers(5)
    t1 = time.time()

    print(f"        Chi-square statistic: {chi['statistic']:.2f}")
    print(f"        P-value:              {chi['p_value']:.4f}")
    print(f"        Significant bias:     {'YES ⚠️' if chi['significant'] else 'No'}")
    print(f"        Bias score:           {bias}/100")
    if hot:
        hot_str = ', '.join(f"#{h['number']}({h['ratio']}x)" for h in hot)
        print(f"        Hot numbers:          {hot_str}")
    if cold:
        cold_str = ', '.join(f"#{c['number']}({c['ratio']}x)" for c in cold)
        print(f"        Cold numbers:         {cold_str}")
    print(f"        Time: {t1-t0:.2f}s ✓")

    # ─── 2. Markov Chain ────────────────────────────────────
    print(f"\n  [2/5] Markov Chain...")
    t0 = time.time()
    markov = MarkovChain()
    markov.load_history(numbers)
    top_markov = markov.get_top_predictions(5)
    strength = markov.get_transition_strength()
    t1 = time.time()

    print(f"        Transition strength:  {strength}/100")
    markov_str = ', '.join(f"#{p['number']}({p['probability']:.3f})" for p in top_markov)
    print(f"        Top predictions:      {markov_str}")
    print(f"        Time: {t1-t0:.2f}s ✓")

    # ─── 3. Pattern Detector ────────────────────────────────
    print(f"\n  [3/5] Pattern Detector...")
    t0 = time.time()
    patterns = PatternDetector()
    patterns.load_history(numbers)
    pat_strength = patterns.get_pattern_strength()
    color_pat = patterns.analyze_color_patterns()
    dozen_pat = patterns.analyze_dozen_patterns()
    suggestions = patterns.get_bet_suggestions()
    t1 = time.time()

    print(f"        Pattern strength:     {pat_strength}/100")
    print(f"        Color streak:         {color_pat['streak']} {color_pat['streak_color']}")
    print(f"        Dominant dozen:       {dozen_pat.get('dominant', 'none')}")
    if suggestions:
        for s in suggestions[:3]:
            print(f"        Suggestion:           {s['type']} → {s['value']} (conf: {s['confidence']}%)")
    print(f"        Time: {t1-t0:.2f}s ✓")

    # ─── 4. GRU Neural Network (PyTorch) ────────────────────
    print(f"\n  [4/5] GRU Neural Network (PyTorch)...")
    t0 = time.time()
    lstm = LSTMPredictor()
    lstm.load_history(numbers)

    if lstm.can_train():
        print(f"        Device:               {lstm.device}")
        print(f"        Dataset size:         {len(numbers)} spins")
        print(f"        Training sequences:   {len(numbers) - 20}")
        print(f"        Training epochs:      {LSTM_EPOCHS}")
        print(f"        Training...")

        result = lstm.train(epochs=LSTM_EPOCHS)

        if result['status'] == 'trained':
            print(f"        Final loss:           {result['final_loss']:.4f}")
            print(f"        Dataset used:         {result['dataset_size']} sequences")

            # Show predictions
            top_lstm = lstm.get_top_predictions(5)
            confidence = lstm.get_confidence_score()
            print(f"        GRU confidence:       {confidence}%")
            lstm_str = ', '.join(f"#{p['number']}({p['probability']:.3f})" for p in top_lstm)
            print(f"        Top predictions:      {lstm_str}")
        else:
            print(f"        Training result:      {result['status']}")
    else:
        print(f"        Not enough data to train (need {40} spins, have {len(numbers)})")

    t1 = time.time()
    print(f"        Time: {t1-t0:.2f}s ✓")

    # ─── 5. Ensemble Predictor ──────────────────────────────
    print(f"\n  [5/5] Ensemble Predictor...")
    t0 = time.time()
    ensemble = EnsemblePredictor()
    ensemble.load_history(numbers)

    # Reload the trained LSTM
    ensemble.lstm = lstm

    prediction = ensemble.predict()
    t1 = time.time()

    print(f"        Overall confidence:   {prediction['confidence']}%")
    print(f"        Mode:                 {prediction['mode']}")
    print(f"        Top numbers:          {prediction['top_numbers']}")

    colors = [get_number_color(n) for n in prediction['top_numbers']]
    print(f"        Top colors:           {colors}")

    gp = prediction['group_probabilities']
    print(f"        Red vs Black:         {gp['red']*100:.1f}% vs {gp['black']*100:.1f}%")
    print(f"        Dozens:               1st={gp['dozen_1st']*100:.1f}% 2nd={gp['dozen_2nd']*100:.1f}% 3rd={gp['dozen_3rd']*100:.1f}%")
    print(f"        High vs Low:          {gp['high']*100:.1f}% vs {gp['low']*100:.1f}%")
    print(f"        Odd vs Even:          {gp['odd']*100:.1f}% vs {gp['even']*100:.1f}%")

    if prediction['bets']:
        print(f"\n        Bet Suggestions:")
        for bet in prediction['bets'][:5]:
            print(f"          → {bet['type']:12s} {bet['value']:20s} (prob: {bet['probability']*100:.1f}%, payout: {bet['payout']})")

    print(f"        Time: {t1-t0:.2f}s ✓")

    return prediction


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python train_from_data.py <data_file>")
        print("\nExample:")
        print("  python train_from_data.py data/my_spins.txt")
        print("  python train_from_data.py data/roulette_history.csv")
        print("\nFile format: one number (0-36) per line, most recent at bottom")
        sys.exit(1)

    filepath = sys.argv[1]

    print(f"\n{'═' * 60}")
    print(f"  AI ROULETTE PREDICTOR - MODEL TRAINING")
    print(f"{'═' * 60}")
    print(f"\n  Loading data from: {filepath}")

    # Load data
    numbers, skipped = load_data(filepath)

    if not numbers:
        print("  ERROR: No valid numbers found in file!")
        sys.exit(1)

    print(f"  Loaded:  {len(numbers)} spin numbers")
    if skipped > 0:
        print(f"  Skipped: {skipped} invalid lines")

    # Validate
    validate_data(numbers)

    # Save as session for future training
    session_id = save_as_session(numbers, filepath)

    # Train all models
    prediction = train_all_models(numbers)

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Session ID:       {session_id}")
    print(f"  Spins trained on: {len(numbers)}")
    print(f"  Models saved to:  {MODELS_DIR}")
    print(f"  Session saved to: {SESSIONS_DIR}")
    print(f"\n  Next step: Run 'python run.py' and open http://localhost:5050")
    print(f"  The trained models will automatically load when you start a session!")
    print(f"{'═' * 60}\n")


if __name__ == '__main__':
    main()
