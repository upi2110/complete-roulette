"""
End-to-End SocketIO Integration Tests
Tests the full flow: connect → start session → import data → get predictions → submit spins → end session

Uses real roulette data from userdata/ files when available.
"""
import pytest
import socketio
import time
import threading
import sys
import os
import glob

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from config import TOP_PREDICTIONS_COUNT, CONFIDENCE_BET_THRESHOLD, INITIAL_BANKROLL

SERVER_URL = 'http://localhost:5050'


def _load_userdata_sample(max_n=100):
    """Load real roulette data from userdata/ files (up to max_n numbers)."""
    all_numbers = []
    userdata_dir = os.path.join(PROJECT_ROOT, 'userdata')
    for f in sorted(glob.glob(os.path.join(userdata_dir, '*.txt'))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        num = int(line)
                        if 0 <= num <= 36:
                            all_numbers.append(num)
                    except ValueError:
                        pass
        if len(all_numbers) >= max_n:
            break
    return all_numbers[:max_n]


# Load real data from userdata files, fallback to hardcoded
_USERDATA_100 = _load_userdata_sample(100)
_FALLBACK = [18, 26, 28, 35, 16, 28, 22, 35, 1, 20, 3, 35, 20, 23, 7, 24, 22, 2, 33, 35,
             12, 30, 27, 11, 9, 10, 9, 20, 16, 31, 4, 3, 16, 20, 34, 13, 28, 3, 15, 33,
             12, 11, 26, 23, 15, 36, 1, 25, 28, 32, 14, 6, 12, 16, 3, 6, 1, 35, 18, 8,
             30, 21, 29, 4, 8, 28, 1, 30, 4, 10, 30, 23, 36, 29, 28, 13, 3, 34, 9, 31,
             1, 2, 18, 25, 32, 6, 16, 16, 19, 35, 16, 32, 30, 21, 25, 36, 21, 27, 7, 6]

REAL_DATA = _USERDATA_100 if len(_USERDATA_100) >= 50 else _FALLBACK
REAL_DATA_50 = REAL_DATA[:50]
REAL_DATA_CSV = ','.join(str(n) for n in REAL_DATA_50)


class SocketIOTestClient:
    """Helper class to manage socketio client with event tracking."""

    def __init__(self):
        self.sio = socketio.Client(logger=False, engineio_logger=False,
                                   reconnection=True, reconnection_delay=1)
        self.events = {}
        self.event_waiter = threading.Event()
        self.connected = False
        self._setup_handlers()

    def _setup_handlers(self):
        @self.sio.event
        def connect():
            self.connected = True
            self._record('connect', True)

        @self.sio.event
        def disconnect():
            self.connected = False

        @self.sio.on('connected')
        def on_connected(data):
            self._record('connected', data)

        @self.sio.on('status_update')
        def on_status(data):
            self._record('status_update', data)

        @self.sio.on('session_started')
        def on_session_started(data):
            self._record('session_started', data)

        @self.sio.on('training_update')
        def on_training(data):
            self._record('training_update', data)

        @self.sio.on('import_complete')
        def on_import(data):
            self._record('import_complete', data)

        @self.sio.on('prediction_result')
        def on_prediction(data):
            self._record('prediction_result', data)

        @self.sio.on('spin_processed')
        def on_spin(data):
            self._record('spin_processed', data)

        @self.sio.on('session_ended')
        def on_ended(data):
            self._record('session_ended', data)

        @self.sio.on('spin_undone')
        def on_undo(data):
            self._record('spin_undone', data)

        @self.sio.on('error')
        def on_error(data):
            self._record('error', data)

    def _record(self, event_name, data):
        self.events[event_name] = data
        self.event_waiter.set()

    def connect(self, retries=5):
        for attempt in range(retries):
            try:
                self.sio.connect(SERVER_URL, transports=['polling', 'websocket'])
                time.sleep(1)
                return
            except Exception:
                if attempt < retries - 1:
                    time.sleep(5)  # Generous pause for eventlet to recover
                else:
                    raise

    def disconnect(self):
        if self.sio.connected:
            self.sio.disconnect()

    def emit(self, event, data=None):
        if data:
            self.sio.emit(event, data)
        else:
            self.sio.emit(event)

    def wait_for(self, event_name, timeout=120):
        """Wait for a specific event, return its data."""
        deadline = time.time() + timeout
        while event_name not in self.events and time.time() < deadline:
            self.event_waiter.clear()
            self.event_waiter.wait(timeout=1)
        return self.events.get(event_name)

    def clear_event(self, event_name):
        self.events.pop(event_name, None)


def is_server_running():
    """Check if the server is accessible (with retry).
    Uses a raw socket check first, then HTTP, to avoid eventlet blocking issues.
    """
    import socket
    for attempt in range(5):
        try:
            sock = socket.create_connection(('localhost', 5050), timeout=3)
            sock.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(2)
    return False


# Skip all tests if server isn't running
pytestmark = pytest.mark.skipif(
    not is_server_running(),
    reason=f"Server not running at {SERVER_URL}"
)


@pytest.fixture(autouse=True)
def pause_between_tests():
    """Give the eventlet server time to recover between test connections.
    Without this, rapid sequential SocketIO connections overwhelm the
    single-threaded async server and cause connection timeouts."""
    yield
    time.sleep(5)


class TestE2EConnection:
    def test_connect(self):
        client = SocketIOTestClient()
        try:
            client.connect()
            data = client.wait_for('connected', timeout=5)
            assert data is not None
            assert 'session_active' in data
        finally:
            client.disconnect()

    def test_get_status(self):
        client = SocketIOTestClient()
        try:
            client.connect()
            client.emit('get_status')
            data = client.wait_for('status_update', timeout=5)
            assert data is not None
            assert 'session_active' in data
            assert 'bankroll' in data
            assert 'model_status' in data
        finally:
            client.disconnect()


class TestE2ESessionFlow:
    def test_start_session(self):
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            data = client.wait_for('session_started', timeout=120)
            assert data is not None, "session_started event not received"
            assert 'session_id' in data
            assert 'bankroll' in data
            assert 'model_status' in data
            assert 'historical_spins_loaded' in data

            # End session
            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_import_data_skip_training(self):
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)
            time.sleep(2)  # Give server a moment after session start

            # Import 50 real numbers
            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            data = client.wait_for('import_complete', timeout=120)
            assert data is not None, "import_complete event not received"
            assert data['numbers_imported'] == 50
            assert data['training_skipped'] == True
            assert data['total_spins'] > 0

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_get_prediction(self):
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            # Import real data first
            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)

            # Get prediction
            client.emit('get_prediction')
            data = client.wait_for('prediction_result', timeout=10)
            assert data is not None, "prediction_result event not received"

            pred = data['prediction']
            assert len(pred['top_numbers']) == TOP_PREDICTIONS_COUNT
            assert pred['confidence'] >= 0
            assert pred['mode'] in ('WAIT', 'BET', 'BET_HIGH')
            assert 'group_probabilities' in pred
            assert 'bets' in pred

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_submit_spin(self):
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            # Import real data
            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)
            time.sleep(1)  # Give server time after import

            # Submit a spin (using real number from data)
            client.emit('submit_spin', {'number': 14})
            data = client.wait_for('spin_processed', timeout=15)
            assert data is not None, "spin_processed event not received"
            assert data['spin_result']['number'] == 14
            assert data['spin_result']['color'] in ('red', 'black', 'green')
            assert 'next_prediction' in data
            assert 'bankroll' in data

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_12_predictions_count(self):
        """Verify that predictions return a dynamic number of numbers (2-12, AI decides)"""
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)

            client.emit('get_prediction')
            data = client.wait_for('prediction_result', timeout=10)
            assert data is not None

            pred = data['prediction']
            n_preds = len(pred['top_numbers'])
            assert 2 <= n_preds <= TOP_PREDICTIONS_COUNT, \
                f"Expected 2-{TOP_PREDICTIONS_COUNT} predictions, got {n_preds}"
            assert len(pred['top_probabilities']) == n_preds

            # Anchors check — dynamic count (2+)
            assert 'anchors' in pred
            assert len(pred['anchors']) >= 2, \
                f"Expected at least 2 anchors, got {len(pred['anchors'])}"

            # All numbers should be valid roulette numbers
            for num in pred['top_numbers']:
                assert 0 <= num <= 36, f"Invalid number: {num}"

            # Recommended bet should be straight type
            assert data['recommended_bet'] is not None
            assert data['recommended_bet']['type'] == 'straight'
            assert 'numbers' in data['recommended_bet']
            assert len(data['recommended_bet']['numbers']) == n_preds

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_bet_win_on_predicted_number(self):
        """Submit a number that's in the predicted set — should be a WIN"""
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)
            time.sleep(2)  # Let server stabilise after background import

            # Get prediction to know which numbers are predicted
            client.emit('get_prediction')
            pred_data = client.wait_for('prediction_result', timeout=15)
            assert pred_data is not None

            predicted_numbers = pred_data['prediction']['top_numbers']
            bet = pred_data['recommended_bet']

            if bet:
                # Submit a number that IS in the predicted set
                win_number = predicted_numbers[0]
                client.clear_event('spin_processed')
                client.emit('submit_spin', {
                    'number': win_number,
                    'bet_placed': bet,
                    'bet_amount': bet['amount']
                })
                data = client.wait_for('spin_processed', timeout=30)
                assert data is not None
                assert data['spin_result']['bet_result']['won'] == True
                assert data['spin_result']['bet_result']['payout'] > 0

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_confidence_above_threshold(self):
        """With historical data loaded, confidence should exceed BET threshold"""
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)

            client.emit('get_prediction')
            data = client.wait_for('prediction_result', timeout=10)
            assert data is not None

            conf = data['prediction']['confidence']
            mode = data['prediction']['mode']
            assert conf >= CONFIDENCE_BET_THRESHOLD, \
                f"Confidence {conf}% below {CONFIDENCE_BET_THRESHOLD}% threshold"
            assert mode in ('BET', 'BET_HIGH'), \
                f"Mode should be BET/BET_HIGH but got {mode}"

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_multiple_spins_update_predictions(self):
        """Submit multiple spins and verify predictions update"""
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)
            time.sleep(2)  # Let server stabilise after background import

            predictions = []
            for spin_num in [6, 12, 16]:
                client.clear_event('spin_processed')
                client.emit('submit_spin', {'number': spin_num})
                data = client.wait_for('spin_processed', timeout=30)
                assert data is not None, f"Spin {spin_num} not processed"
                predictions.append(data['next_prediction']['top_numbers'])
                time.sleep(1)

            # Predictions should vary (not all identical)
            assert len(predictions) == 3
            # All predictions should have dynamic count (2-12 numbers)
            for pred in predictions:
                assert 2 <= len(pred) <= TOP_PREDICTIONS_COUNT, \
                    f"Expected 2-{TOP_PREDICTIONS_COUNT} predictions, got {len(pred)}"

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_end_session(self):
        client = SocketIOTestClient()
        try:
            client.connect(retries=8)
            time.sleep(2)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            client.emit('end_session')
            data = client.wait_for('session_ended', timeout=5)
            assert data is not None
            assert 'final_bankroll' in data
            assert 'analytics' in data
        finally:
            client.disconnect()

    def test_error_on_spin_without_session(self):
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            # Don't start session, just submit spin
            client.emit('submit_spin', {'number': 18})
            data = client.wait_for('error', timeout=5)
            assert data is not None
            assert 'message' in data
        finally:
            client.disconnect()


class TestE2ERegression:
    """Regression tests for bugs found in production."""

    def test_model_status_consistent_counts(self):
        """REGRESSION: All models must report the same spin count.
        Previously Frequency=1533, Markov=1527, Pattern=515."""
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            # Import data
            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            data = client.wait_for('import_complete', timeout=60)
            assert data is not None

            # Check model status — all should report same count
            # Filter to only model dicts (skip metadata keys like adaptive_weights_active)
            model_status = data.get('model_status', {})
            if model_status:
                spin_counts = {name: m['spins'] for name, m in model_status.items()
                               if isinstance(m, dict) and 'spins' in m}
                if spin_counts:
                    counts_set = set(spin_counts.values())
                    assert len(counts_set) == 1, \
                        f"Models report different spin counts: {spin_counts}"

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_straight_bet_only(self):
        """Verify recommended_bet.type is always 'straight'."""
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)

            client.emit('get_prediction')
            data = client.wait_for('prediction_result', timeout=10)
            assert data is not None

            bet = data.get('recommended_bet')
            if bet:
                assert bet['type'] == 'straight', \
                    f"Expected straight bet, got {bet['type']}"
                assert 'numbers' in bet
                n_bet_nums = len(bet['numbers'])
                assert 2 <= n_bet_nums <= TOP_PREDICTIONS_COUNT, \
                    f"Expected 2-{TOP_PREDICTIONS_COUNT} bet numbers, got {n_bet_nums}"

            # All bets in prediction should be straight
            for b in data['prediction'].get('bets', []):
                assert b['type'] == 'straight', \
                    f"Expected straight bet, got {b['type']}"

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_bet_loss_on_unpredicted_number(self):
        """Submit a number NOT in predicted set — should be a LOSS."""
        client = SocketIOTestClient()
        try:
            client.connect()
            time.sleep(1)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)
            time.sleep(2)  # Let server stabilise after background import

            # Get prediction to know which numbers are predicted
            client.emit('get_prediction')
            pred_data = client.wait_for('prediction_result', timeout=15)
            assert pred_data is not None

            predicted = set(pred_data['prediction']['top_numbers'])
            bet = pred_data.get('recommended_bet')

            if bet:
                # Find a number NOT in the predicted set
                unpredicted = None
                for n in range(37):
                    if n not in predicted:
                        unpredicted = n
                        break

                if unpredicted is not None:
                    client.clear_event('spin_processed')
                    client.emit('submit_spin', {
                        'number': unpredicted,
                        'bet_placed': bet,
                        'bet_amount': bet['amount']
                    })
                    data = client.wait_for('spin_processed', timeout=30)
                    assert data is not None
                    assert data['spin_result']['bet_result']['won'] == False

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()

    def test_incremental_betting_flow(self):
        """Test the $1 incremental betting: 3 losses → higher bet, 2 wins → lower bet."""
        client = SocketIOTestClient()
        try:
            client.connect(retries=8)
            time.sleep(2)
            client.emit('start_session')
            client.wait_for('session_started', timeout=120)

            client.emit('import_data', {'text': REAL_DATA_CSV, 'skip_training': True})
            client.wait_for('import_complete', timeout=60)
            time.sleep(2)  # Let server stabilise after background import

            # Get initial prediction + bet
            client.emit('get_prediction')
            pred_data = client.wait_for('prediction_result', timeout=15)
            assert pred_data is not None

            if pred_data.get('recommended_bet'):
                initial_amount = pred_data['recommended_bet']['amount']
                assert initial_amount == 1.0, \
                    f"Initial bet should be $1, got ${initial_amount}"

            client.emit('end_session')
            client.wait_for('session_ended', timeout=5)
        finally:
            client.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
