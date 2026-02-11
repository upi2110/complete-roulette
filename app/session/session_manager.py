"""
Session Manager - Persistent session storage, training data aggregation,
cross-session model retraining, and performance analytics.
"""

import os
import json
import time
from datetime import datetime

import sys
sys.path.insert(0, '.')
from config import SESSIONS_DIR, DATA_DIR


class SessionManager:
    def __init__(self):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.current_session = None
        self.session_id = None

    def create_session(self):
        """Start a new session."""
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_session = {
            'id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'status': 'active',
            'spins': [],
            'predictions': [],
            'bets': [],
            'bankroll_history': [],
            'stats': {
                'total_spins': 0,
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
        self._save()
        return self.session_id

    def record_spin(self, number, prediction=None, bet_info=None, bankroll_status=None):
        """Record a single spin with its prediction and bet result."""
        if not self.current_session:
            self.create_session()

        spin_record = {
            'spin_number': self.current_session['stats']['total_spins'] + 1,
            'timestamp': datetime.now().isoformat(),
            'actual_number': number,
        }

        if prediction:
            spin_record['prediction'] = {
                'top_numbers': prediction.get('top_numbers', []),
                'confidence': prediction.get('confidence', 0),
                'mode': prediction.get('mode', 'WAIT'),
                'suggested_color': prediction.get('suggested_color'),
                'suggested_dozen': prediction.get('suggested_dozen'),
            }
            self.current_session['predictions'].append({
                'spin': spin_record['spin_number'],
                'confidence': prediction.get('confidence', 0),
                'correct_number': number in prediction.get('top_numbers', []),
                'mode': prediction.get('mode', 'WAIT')
            })

        if bet_info:
            spin_record['bet'] = bet_info
            self.current_session['bets'].append(bet_info)

        if bankroll_status:
            spin_record['bankroll'] = bankroll_status.get('bankroll', 0)
            self.current_session['bankroll_history'].append(bankroll_status.get('bankroll', 0))

            # Update stats
            stats = self.current_session['stats']
            stats['ending_bankroll'] = bankroll_status.get('bankroll', 0)
            stats['peak_bankroll'] = max(stats['peak_bankroll'], bankroll_status.get('bankroll', 0))
            stats['lowest_bankroll'] = min(stats['lowest_bankroll'], bankroll_status.get('bankroll', 0))
            stats['max_consecutive_losses'] = max(
                stats['max_consecutive_losses'],
                bankroll_status.get('max_consecutive_losses', 0)
            )

        self.current_session['spins'].append(spin_record)
        self.current_session['stats']['total_spins'] += 1
        self.current_session['updated_at'] = datetime.now().isoformat()

        # Auto-save every 5 spins
        if self.current_session['stats']['total_spins'] % 5 == 0:
            self._save()

    def update_bet_stats(self, won):
        """Update win/loss stats."""
        if not self.current_session:
            return

        stats = self.current_session['stats']
        stats['total_bets'] += 1
        if won:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

    def record_skip(self):
        """Record when AI suggests WAIT."""
        if self.current_session:
            self.current_session['stats']['skips'] += 1

    def end_session(self):
        """End current session and save final state."""
        if self.current_session:
            self.current_session['status'] = 'completed'
            self.current_session['ended_at'] = datetime.now().isoformat()
            self._save()

            session_id = self.session_id
            self.current_session = None
            self.session_id = None
            return session_id
        return None

    def _save(self):
        """Save current session to JSON file."""
        if not self.current_session:
            return

        filepath = os.path.join(SESSIONS_DIR, f'session_{self.session_id}.json')
        with open(filepath, 'w') as f:
            json.dump(self.current_session, f, indent=2, default=str)

    def load_session(self, session_id):
        """Load a specific session."""
        filepath = os.path.join(SESSIONS_DIR, f'session_{session_id}.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.current_session = json.load(f)
                self.session_id = session_id
                return self.current_session
        return None

    def get_all_sessions(self):
        """List all saved sessions."""
        sessions = []
        if not os.path.exists(SESSIONS_DIR):
            return sessions

        for filename in sorted(os.listdir(SESSIONS_DIR), reverse=True):
            if filename.startswith('session_') and filename.endswith('.json'):
                filepath = os.path.join(SESSIONS_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        session = json.load(f)
                        sessions.append({
                            'id': session.get('id'),
                            'created_at': session.get('created_at'),
                            'status': session.get('status'),
                            'total_spins': session['stats']['total_spins'],
                            'total_bets': session['stats']['total_bets'],
                            'wins': session['stats']['wins'],
                            'losses': session['stats']['losses'],
                            'starting_bankroll': session['stats']['starting_bankroll'],
                            'ending_bankroll': session['stats']['ending_bankroll'],
                            'pnl': round(session['stats']['ending_bankroll'] -
                                        session['stats']['starting_bankroll'], 2)
                        })
                except (json.JSONDecodeError, KeyError):
                    continue

        return sessions

    def clear_all_sessions(self):
        """Delete all saved session JSON files."""
        if not os.path.exists(SESSIONS_DIR):
            return 0
        count = 0
        for filename in os.listdir(SESSIONS_DIR):
            if filename.startswith('session_') and filename.endswith('.json'):
                os.remove(os.path.join(SESSIONS_DIR, filename))
                count += 1
        self.current_session = None
        self.session_id = None
        print(f"[RESET] Cleared {count} session files")
        return count

    def get_all_training_data(self):
        """Aggregate all spin numbers from all sessions for model training."""
        all_spins = []

        if not os.path.exists(SESSIONS_DIR):
            return all_spins

        for filename in sorted(os.listdir(SESSIONS_DIR)):
            if filename.startswith('session_') and filename.endswith('.json'):
                filepath = os.path.join(SESSIONS_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        session = json.load(f)
                        for spin in session.get('spins', []):
                            all_spins.append(spin['actual_number'])
                except (json.JSONDecodeError, KeyError):
                    continue

        return all_spins

    def get_all_training_clusters(self):
        """Get training data as separate clusters (each session/import = independent cluster).
        Returns list of lists - each inner list is an independent data cluster."""
        clusters = []

        if not os.path.exists(SESSIONS_DIR):
            return clusters

        for filename in sorted(os.listdir(SESSIONS_DIR)):
            if filename.startswith('session_') and filename.endswith('.json'):
                filepath = os.path.join(SESSIONS_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        session = json.load(f)
                        spins = [s['actual_number'] for s in session.get('spins', [])]
                        if spins:
                            clusters.append({
                                'id': session.get('id', ''),
                                'status': session.get('status', ''),
                                'spins': spins,
                                'size': len(spins)
                            })
                except (json.JSONDecodeError, KeyError):
                    continue

        return clusters

    def undo_last_spin(self):
        """Remove the last recorded spin from current session."""
        if not self.current_session:
            return None

        spins = self.current_session.get('spins', [])
        if not spins:
            return None

        removed = spins.pop()
        self.current_session['stats']['total_spins'] = len(spins)
        self.current_session['updated_at'] = datetime.now().isoformat()
        self._save()

        return removed.get('actual_number')

    def get_performance_analytics(self):
        """Cross-session performance analytics."""
        sessions = self.get_all_sessions()
        if not sessions:
            return {
                'total_sessions': 0,
                'total_spins': 0,
                'total_bets': 0,
                'overall_win_rate': 0,
                'best_session_pnl': 0,
                'worst_session_pnl': 0,
                'avg_session_pnl': 0,
                'total_pnl': 0,
                'profitable_sessions': 0
            }

        total_spins = sum(s['total_spins'] for s in sessions)
        total_bets = sum(s['total_bets'] for s in sessions)
        total_wins = sum(s['wins'] for s in sessions)
        pnls = [s['pnl'] for s in sessions]

        return {
            'total_sessions': len(sessions),
            'total_spins': total_spins,
            'total_bets': total_bets,
            'overall_win_rate': round(total_wins / total_bets * 100, 1) if total_bets > 0 else 0,
            'best_session_pnl': max(pnls) if pnls else 0,
            'worst_session_pnl': min(pnls) if pnls else 0,
            'avg_session_pnl': round(sum(pnls) / len(pnls), 2) if pnls else 0,
            'total_pnl': round(sum(pnls), 2),
            'profitable_sessions': sum(1 for p in pnls if p > 0),
            'sessions': sessions
        }

    def get_current_session_spins(self):
        """Get spin numbers from current session."""
        if not self.current_session:
            return []
        return [s['actual_number'] for s in self.current_session.get('spins', [])]
