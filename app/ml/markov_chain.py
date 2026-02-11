"""
Markov Chain Model - First and second order transition probability matrices.
Predicts next number based on transition patterns from previous spins.
"""

import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, '.')
from config import TOTAL_NUMBERS, MARKOV_ORDER_1_WEIGHT, MARKOV_ORDER_2_WEIGHT

# Laplace smoothing: add this many pseudo-counts to every transition
# Prevents 100%/0% spiky predictions with sparse data
MARKOV_SMOOTHING = 0.5


class MarkovChain:
    def __init__(self):
        self.spin_history = []
        # First order: P(next | current)
        self.transition_1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
        self.counts_1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
        # Second order: P(next | prev, current) stored as dict
        self.counts_2 = defaultdict(lambda: np.zeros(TOTAL_NUMBERS))
        self.transition_2 = {}

    def update(self, number):
        self.spin_history.append(number)
        n = len(self.spin_history)

        # Update first order (with Laplace smoothing)
        if n >= 2:
            prev = self.spin_history[-2]
            self.counts_1[prev][number] += 1
            smoothed = self.counts_1[prev] + MARKOV_SMOOTHING
            self.transition_1[prev] = smoothed / smoothed.sum()

        # Update second order (with Laplace smoothing)
        if n >= 3:
            prev2 = self.spin_history[-3]
            prev1 = self.spin_history[-2]
            key = (prev2, prev1)
            self.counts_2[key][number] += 1
            smoothed = self.counts_2[key] + MARKOV_SMOOTHING
            self.transition_2[key] = smoothed / smoothed.sum()

    def load_history(self, history):
        self.spin_history = []
        self.transition_1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
        self.counts_1 = np.zeros((TOTAL_NUMBERS, TOTAL_NUMBERS))
        self.counts_2 = defaultdict(lambda: np.zeros(TOTAL_NUMBERS))
        self.transition_2 = {}
        for num in history:
            self.update(num)

    def predict_first_order(self):
        """Predict next number using first-order transitions with Laplace smoothing."""
        if len(self.spin_history) < 2:
            return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        current = self.spin_history[-1]
        # Always apply smoothing so we never get 100%/0%
        smoothed = self.counts_1[current] + MARKOV_SMOOTHING
        probs = smoothed / smoothed.sum()
        return probs

    def predict_second_order(self):
        """Predict next number using second-order transitions with Laplace smoothing."""
        if len(self.spin_history) < 3:
            return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        key = (self.spin_history[-2], self.spin_history[-1])
        if key in self.counts_2:
            smoothed = self.counts_2[key] + MARKOV_SMOOTHING
            probs = smoothed / smoothed.sum()
            return probs

        return self.predict_first_order()

    def get_probabilities(self):
        """Combined first and second order prediction."""
        p1 = self.predict_first_order()
        p2 = self.predict_second_order()

        combined = MARKOV_ORDER_1_WEIGHT * p1 + MARKOV_ORDER_2_WEIGHT * p2

        total = combined.sum()
        if total > 0:
            combined /= total

        return combined

    def get_top_predictions(self, top_n=5):
        """Get top predicted numbers with probabilities."""
        probs = self.get_probabilities()
        top_indices = np.argsort(probs)[-top_n:][::-1]

        return [
            {'number': int(idx), 'probability': round(float(probs[idx]), 4)}
            for idx in top_indices
        ]

    def get_transition_strength(self):
        """How strong are the transition patterns (0-100)."""
        if len(self.spin_history) < 10:
            return 0.0

        probs = self.get_probabilities()
        uniform = 1.0 / TOTAL_NUMBERS

        # KL divergence from uniform
        kl_div = 0
        for p in probs:
            if p > 0:
                kl_div += p * np.log(p / uniform)

        # Normalize to 0-100
        strength = min(100, kl_div * 100)
        return round(float(strength), 1)

    def get_repeater_probability(self):
        """Probability that the last number will repeat."""
        if not self.spin_history:
            return 1.0 / TOTAL_NUMBERS
        last = self.spin_history[-1]
        probs = self.get_probabilities()
        return float(probs[last])

    def get_summary(self):
        return {
            'total_spins': len(self.spin_history),
            'top_predictions': self.get_top_predictions(),
            'transition_strength': self.get_transition_strength(),
            'repeater_prob': round(self.get_repeater_probability(), 4)
        }
