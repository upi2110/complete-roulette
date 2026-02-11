"""
Transformer-GRU Hybrid Predictor — Industrial-grade neural network for
roulette sequence prediction.

Architecture:
  Input (85-dim rich features per timestep, sequence of 30)
    → LayerNorm
    → 2-layer GRU (hidden=192, dropout=0.3)
    → Multi-head Self-Attention (4 heads)
    → Residual connection + LayerNorm
    → FC(192 → 96) + GELU + Dropout(0.2)
    → FC(96 → 37) → Softmax

Training improvements over the old simple GRU:
  - Rich 85-dim feature vectors instead of 37-dim one-hot
  - Validation split with early stopping
  - Learning rate scheduling (ReduceLROnPlateau)
  - Label smoothing (0.1) to prevent overconfidence
  - Multi-head self-attention for long-range correlations
  - Residual connections to prevent gradient vanishing
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, '.')
from config import (
    TOTAL_NUMBERS, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_SEQUENCE_LENGTH, LSTM_LEARNING_RATE, LSTM_EPOCHS,
    LSTM_BATCH_SIZE, LSTM_MIN_TRAINING_SPINS, LSTM_MODEL_PATH, MODELS_DIR,
    ATTENTION_HEADS, LSTM_LABEL_SMOOTHING,
    EARLY_STOPPING_PATIENCE, LR_SCHEDULE_PATIENCE, LR_SCHEDULE_FACTOR,
    VALIDATION_SPLIT, FEATURE_DIM,
)
from app.ml.feature_engine import FeatureEngine


# ─── Dataset ─────────────────────────────────────────────────────────

class RouletteDataset(Dataset):
    """Sliding window dataset using rich feature vectors."""

    def __init__(self, spin_history, seq_length=LSTM_SEQUENCE_LENGTH,
                 feature_engine=None):
        self.seq_length = seq_length
        self.data = []
        self.targets = []

        if feature_engine is None:
            feature_engine = FeatureEngine()

        n = len(spin_history)
        if n > seq_length:
            for i in range(n - seq_length):
                # Extract rich feature matrix for this window
                features = feature_engine.extract_sequence(
                    spin_history, seq_start=i, seq_length=seq_length
                )
                target = spin_history[i + seq_length]
                self.data.append(features)
                self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.data[idx]),
                torch.LongTensor([self.targets[idx]])[0])


# ─── Neural Network Architecture ─────────────────────────────────────

class RouletteGRU(nn.Module):
    """Transformer-GRU Hybrid for roulette number prediction.

    GRU captures sequential patterns (what came after what).
    Self-attention finds long-range correlations (number X tends to
    appear N spins after number Y).
    Residual connection + LayerNorm stabilise training.
    GELU activation is smoother than ReLU for probability estimation.
    """

    def __init__(self, input_size=FEATURE_DIM, hidden_size=LSTM_HIDDEN_SIZE,
                 num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT,
                 num_heads=ATTENTION_HEADS):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input normalisation
        self.input_norm = nn.LayerNorm(input_size)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Multi-head self-attention over GRU outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, TOTAL_NUMBERS),
        )

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)

        # 1. Input LayerNorm
        x = self.input_norm(x)

        # 2. GRU
        gru_out, hidden = self.gru(x, hidden)
        # gru_out: (batch, seq_len, hidden_size)

        # 3. Multi-head self-attention + residual
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        # Residual connection + LayerNorm
        combined = self.attn_norm(gru_out + attn_out)

        # 4. Use last timestep output
        last = combined[:, -1, :]

        # 5. Classification
        logits = self.fc(last)
        return logits, hidden


# ─── Predictor Wrapper ────────────────────────────────────────────────

class LSTMPredictor:
    """High-level wrapper: manages training, prediction, persistence."""

    def __init__(self):
        self.device = self._get_device()
        self.feature_engine = FeatureEngine()
        self.model = RouletteGRU().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LSTM_LEARNING_RATE
        )
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=LSTM_LABEL_SMOOTHING
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=LR_SCHEDULE_FACTOR,
            patience=LR_SCHEDULE_PATIENCE,
        )
        self.is_trained = False
        self.training_loss_history = []
        self.validation_loss_history = []
        self.spin_history = []
        self.spins_since_last_train = 0
        # NOTE: No auto-load here. LSTM checkpoint is loaded explicitly
        # via load_checkpoint() when Train AI is clicked or state is restored.

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def load_checkpoint(self):
        """Load LSTM weights from disk. Called explicitly, not on __init__."""
        if os.path.exists(LSTM_MODEL_PATH):
            try:
                checkpoint = torch.load(
                    LSTM_MODEL_PATH, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint['model_state'])
                self.is_trained = checkpoint.get('is_trained', True)
                self.training_loss_history = checkpoint.get('loss_history', [])
                self.validation_loss_history = checkpoint.get('val_loss_history', [])
                print(f"[LSTM] Model loaded from {LSTM_MODEL_PATH}")
            except Exception as e:
                print(f"[LSTM] Could not load model (architecture may have changed): {e}")
                print("[LSTM] Starting with fresh weights — will retrain on next Train AI")

    def save_model(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        checkpoint = {
            'model_state': self.model.state_dict(),
            'is_trained': self.is_trained,
            'loss_history': self.training_loss_history[-100:],
            'val_loss_history': self.validation_loss_history[-100:],
        }
        torch.save(checkpoint, LSTM_MODEL_PATH)

    def update(self, number):
        self.spin_history.append(number)
        self.spins_since_last_train += 1

    def load_history(self, history):
        self.spin_history = list(history)

    def can_train(self):
        return len(self.spin_history) >= LSTM_MIN_TRAINING_SPINS

    def needs_retrain(self, interval):
        return self.spins_since_last_train >= interval and self.can_train()

    def train(self, epochs=LSTM_EPOCHS):
        """Train with validation split, early stopping, and LR scheduling."""
        if not self.can_train():
            return {'status': 'insufficient_data', 'spins': len(self.spin_history)}

        # Build dataset with rich features
        full_dataset = RouletteDataset(
            self.spin_history,
            feature_engine=self.feature_engine,
        )
        if len(full_dataset) == 0:
            return {'status': 'no_sequences', 'spins': len(self.spin_history)}

        # Train / validation split (temporal: last 15% for validation)
        total = len(full_dataset)
        val_size = max(1, int(total * VALIDATION_SPLIT))
        train_size = total - val_size

        # Temporal split — don't shuffle the split boundary
        train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total))

        train_loader = DataLoader(
            train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False
        )

        # Try to yield to eventlet between epochs
        try:
            import eventlet
            _sleep = eventlet.sleep
        except ImportError:
            _sleep = lambda x: None

        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        actual_epochs = 0

        for epoch in range(epochs):
            actual_epochs = epoch + 1

            # ── Training pass ──
            self.model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                output, _ = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            self.training_loss_history.append(avg_train_loss)

            # ── Validation pass ──
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    output, _ = self.model(batch_x)
                    loss = self.criterion(output, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            self.validation_loss_history.append(avg_val_loss)

            # LR scheduling
            self.scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"[LSTM] Early stopping at epoch {actual_epochs} "
                          f"(val_loss={avg_val_loss:.4f}, best={best_val_loss:.4f})")
                    break

            # Yield to event loop every 5 epochs
            if epoch % 5 == 0:
                _sleep(0)

        # Restore best model weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_trained = True
        self.spins_since_last_train = 0
        self.save_model()

        return {
            'status': 'trained',
            'epochs': actual_epochs,
            'final_loss': round(best_val_loss, 4),
            'train_loss': round(self.training_loss_history[-1], 4),
            'dataset_size': total,
            'train_size': train_size,
            'val_size': val_size,
            'early_stopped': patience_counter >= EARLY_STOPPING_PATIENCE,
        }

    def predict(self):
        """Predict probability distribution for next spin using rich features."""
        if not self.is_trained or len(self.spin_history) < LSTM_SEQUENCE_LENGTH:
            return np.full(TOTAL_NUMBERS, 1.0 / TOTAL_NUMBERS)

        self.model.eval()
        with torch.no_grad():
            # Build feature sequence for last LSTM_SEQUENCE_LENGTH spins
            seq_start = len(self.spin_history) - LSTM_SEQUENCE_LENGTH
            features = self.feature_engine.extract_sequence(
                self.spin_history, seq_start, LSTM_SEQUENCE_LENGTH
            )
            # Add batch dimension: (1, seq_len, feature_dim)
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            output, _ = self.model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        return probs

    def get_top_predictions(self, top_n=5):
        """Get top N predicted numbers."""
        probs = self.predict()
        top_indices = np.argsort(probs)[-top_n:][::-1]
        return [
            {'number': int(idx), 'probability': round(float(probs[idx]), 4)}
            for idx in top_indices
        ]

    def get_prediction_entropy(self):
        """Measure uncertainty of prediction (lower = more confident)."""
        probs = self.predict()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(TOTAL_NUMBERS)
        return float(entropy / max_entropy) if max_entropy > 0 else 1.0

    def get_confidence_score(self):
        """Convert entropy to confidence score 0-100."""
        if not self.is_trained:
            return 0.0
        entropy_ratio = self.get_prediction_entropy()
        confidence = (1 - entropy_ratio) * 100
        return round(max(0, min(100, confidence)), 1)

    def get_summary(self):
        return {
            'is_trained': self.is_trained,
            'device': str(self.device),
            'total_spins': len(self.spin_history),
            'can_train': self.can_train(),
            'top_predictions': self.get_top_predictions() if self.is_trained else [],
            'confidence': self.get_confidence_score(),
            'last_loss': self.training_loss_history[-1] if self.training_loss_history else None,
            'last_val_loss': self.validation_loss_history[-1] if self.validation_loss_history else None,
        }
