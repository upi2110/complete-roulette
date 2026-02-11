"""
SocketIO Event Handlers - Real-time WebSocket events for the dashboard.
Handles spin input, predictions, session management, and model training.
"""

from flask_socketio import emit
from flask import request
from app import socketio
import eventlet
import time

import sys
sys.path.insert(0, '.')
from config import get_number_color, PAYOUTS, RED_NUMBERS, BLACK_NUMBERS, \
    FIRST_DOZEN, SECOND_DOZEN, THIRD_DOZEN, LOW_NUMBERS, HIGH_NUMBERS, \
    ODD_NUMBERS, EVEN_NUMBERS, TOP_PREDICTIONS_COUNT, RETRAIN_INTERVAL, \
    USERDATA_DIR

from app.ml.ensemble import EnsemblePredictor
from app.money.bankroll_manager import BankrollManager
from app.session.session_manager import SessionManager

# Global instances
predictor = EnsemblePredictor()
bankroll = BankrollManager()
session_mgr = SessionManager()

# ─── No auto-load on startup ──────────────────────────────────────────
# Models start idle. User must click "Train AI" to load data.
# This prevents stale data from showing "Trained" before the user acts.
print("[Startup] Models start idle — click 'Train AI' to load data")

# Track current state
current_state = {
    'session_active': False,
    'last_prediction': None,
    'last_bet_placed': None
}


def _check_bet_result(actual_number, bet):
    """Check if a bet won based on the actual result."""
    bet_type = bet['type']
    bet_value = bet['value']

    if bet_type == 'red_black':
        if bet_value == 'Red':
            return actual_number in RED_NUMBERS
        else:
            return actual_number in BLACK_NUMBERS

    elif bet_type == 'high_low':
        if 'High' in bet_value:
            return actual_number in HIGH_NUMBERS
        else:
            return actual_number in LOW_NUMBERS

    elif bet_type == 'odd_even':
        if bet_value == 'Odd':
            return actual_number in ODD_NUMBERS
        else:
            return actual_number in EVEN_NUMBERS

    elif bet_type == 'dozen':
        if '1st' in bet_value:
            return actual_number in FIRST_DOZEN
        elif '2nd' in bet_value:
            return actual_number in SECOND_DOZEN
        else:
            return actual_number in THIRD_DOZEN

    elif bet_type == 'column':
        # Simplified column check
        if actual_number == 0:
            return False
        col = ((actual_number - 1) % 3) + 1
        if '1st' in bet_value:
            return col == 1
        elif '2nd' in bet_value:
            return col == 2
        else:
            return col == 3

    elif bet_type == 'straight':
        return actual_number == int(bet_value)

    elif bet_type == 'sector':
        return actual_number in bet.get('numbers', [])

    return False


@socketio.on('connect')
def handle_connect():
    emit('connected', {
        'message': 'Connected to AI Roulette Predictor',
        'session_active': current_state['session_active'],
        'model_status': predictor.get_model_status(),
        'total_spins': len(predictor.spin_history),
        'session_start_time': current_state.get('session_start_time'),
    })


@socketio.on('start_session')
def handle_start_session():
    global current_state

    # Check cache freshness BEFORE create_session writes a new JSON file
    cache_is_fresh = predictor.is_state_fresh()

    session_id = session_mgr.create_session()
    current_state['session_active'] = True
    current_state['session_start_time'] = time.time()
    current_state['last_prediction'] = None
    current_state['last_bet_placed'] = None

    # Capture the client's session ID so we can emit to them from background
    sid = request.sid

    def _do_start():
        """Load model state: use what's in memory, cache, or rebuild from sessions."""
        total_spins = 0
        clusters_loaded = 0
        load_mode = 'none'

        # If predictor already has data in memory (from startup auto-load
        # or a previous Train AI), skip redundant disk loading
        if len(predictor.spin_history) > 0:
            total_spins = len(predictor.spin_history)
            load_mode = 'memory'
            # Re-save so cache stays newer than the empty session file
            predictor.save_state()
            print(f"[Session] Using existing model state: {total_spins} spins")

        elif cache_is_fresh:
            # FAST PATH: load from cached pickle (~instant)
            if predictor.load_state():
                total_spins = len(predictor.spin_history)
                load_mode = 'cache'
                predictor.save_state()
                print(f"[Session] Fast start from cache: {total_spins} spins")

        if load_mode == 'none':
            # SLOW PATH: rebuild from session JSON files
            clusters = session_mgr.get_all_training_clusters()
            if clusters:
                cluster_result = predictor.load_clusters(clusters)
                total_spins = cluster_result['total_spins']
                clusters_loaded = cluster_result['clusters_loaded']
                eventlet.sleep(0)
            # Save state for fast next start
            predictor.save_state()
            load_mode = 'rebuild'

            # Train LSTM in background AFTER session_started is emitted
            # so the UI responds instantly
            if predictor.lstm.can_train():
                def _bg_train():
                    result = predictor.train_lstm()
                    predictor.save_state()
                    socketio.emit('training_update', result, to=sid)
                socketio.start_background_task(_bg_train)

        bankroll.full_reset()

        # Send last 10 numbers so UI can display them immediately
        last_10 = list(predictor.spin_history[-10:]) if predictor.spin_history else []

        socketio.emit('session_started', {
            'session_id': session_id,
            'bankroll': bankroll.get_status(),
            'model_status': predictor.get_model_status(),
            'historical_spins_loaded': total_spins,
            'clusters_loaded': clusters_loaded,
            'load_mode': load_mode,
            'last_numbers': last_10,
            'session_start_time': current_state.get('session_start_time'),
        }, to=sid)

    socketio.start_background_task(_do_start)


@socketio.on('end_session')
def handle_end_session():
    global current_state

    session_id = session_mgr.end_session()
    current_state['session_active'] = False

    # Emit immediately so the UI responds instantly
    sid = request.sid
    emit('session_ended', {
        'session_id': session_id,
        'final_bankroll': bankroll.get_status(),
        'analytics': session_mgr.get_performance_analytics()
    })

    # Train LSTM + save state in background (user doesn't wait)
    # Capture spin count to detect if a reset happened while we were training
    spin_count_before = len(predictor.spin_history)

    def _bg_train_and_save():
        # Guard: if reset_all cleared everything, don't re-save stale state
        if len(predictor.spin_history) == 0 and spin_count_before > 0:
            print("[EndSession] Skipping background save — reset detected")
            return
        if predictor.lstm.can_train():
            result = predictor.train_lstm()
            socketio.emit('training_update', result, to=sid)
        predictor.save_state()

    socketio.start_background_task(_bg_train_and_save)


@socketio.on('reset_all')
def handle_reset_all():
    """Completely reset all training data, models, and bankroll to fresh state."""
    global current_state

    # End session if active
    if current_state['session_active']:
        session_mgr.end_session()
        current_state['session_active'] = False

    # Reset all ML models (clears history, weights, saved files)
    predictor.full_reset()

    # Reset bankroll
    bankroll.full_reset()

    # Clear session history files
    session_mgr.clear_all_sessions()

    current_state['last_prediction'] = None
    current_state['last_bet_placed'] = None

    emit('reset_complete', {
        'message': 'All training data cleared. Fresh start.',
        'bankroll': bankroll.get_status(),
        'model_status': predictor.get_model_status()
    })


@socketio.on('get_prediction')
def handle_get_prediction():
    """Get next prediction without submitting a spin result."""
    if not current_state['session_active']:
        emit('error', {'message': 'No active session. Start a session first.'})
        return

    prediction = predictor.predict()

    # Check if we should bet
    should_bet, reason = bankroll.should_bet(prediction['confidence'], prediction['mode'])

    # Calculate bet amount — straight bets on all predicted numbers (dynamic count)
    n_preds = len(prediction['top_numbers'])
    bet_per_number = bankroll.calculate_bet_amount(prediction['confidence'], 'straight', num_predictions=n_preds)
    total_bet = round(bet_per_number * n_preds, 2)

    recommended_bet = None
    if should_bet and prediction['top_numbers']:
        recommended_bet = {
            'type': 'straight',
            'value': str(prediction['anchors'][0]) if prediction.get('anchors') else str(prediction['top_numbers'][0]),
            'numbers': prediction['top_numbers'],
            'amount': bet_per_number,
            'total_bet': total_bet,
            'payout': '35:1',
            'risk': 'high',
            'probability': round(len(prediction['top_numbers']) / 37, 4)
        }

    # Get money management advice
    money_advice = bankroll.get_advice(
        ai_confidence=prediction['confidence'],
        ai_mode=prediction['mode'],
        prediction=prediction
    )

    response = {
        'prediction': prediction,
        'should_bet': should_bet,
        'bet_reason': reason,
        'recommended_bet': recommended_bet,
        'bet_amount': bet_per_number,
        'total_bet': total_bet,
        'bankroll': bankroll.get_status(),
        'money_advice': money_advice
    }

    current_state['last_prediction'] = prediction
    emit('prediction_result', response)


@socketio.on('submit_spin')
def handle_submit_spin(data):
    """Submit the actual spin result."""
    if not current_state['session_active']:
        emit('error', {'message': 'No active session. Start a session first.'})
        return

    try:
        number = int(data.get('number', -1))
    except (ValueError, TypeError):
        emit('error', {'message': 'Invalid number. Enter 0-36.'})
        return

    if number < 0 or number > 36:
        emit('error', {'message': 'Invalid number. Enter 0-36.'})
        return

    # Get the bet that was placed (if any)
    bet_placed = data.get('bet_placed')
    bet_per_number = float(data.get('bet_amount', 0))

    # Process bet result — straight bets on predicted numbers
    bet_info = None
    won = False
    payout = 0

    if bet_placed and bet_per_number > 0:
        # For straight bets: check if actual number is in the predicted set
        predicted_numbers = bet_placed.get('numbers', [])
        num_bets = len(predicted_numbers) if predicted_numbers else 1
        total_bet = round(bet_per_number * num_bets, 2)

        if number in predicted_numbers:
            won = True
            payout = bet_per_number * 35  # 35:1 payout on the winning number
            # Net profit = winnings + stake back - total cost
            # e.g. $35 + $1 - $12 = $24 when betting $1 on 12 numbers
            net_profit = round(payout + bet_per_number - total_bet, 2)
        else:
            won = False
            payout = 0
            net_profit = round(-total_bet, 2)

        bankroll.process_result(total_bet, won, payout)
        session_mgr.update_bet_stats(won)

        bet_info = {
            'type': 'straight',
            'value': f'{num_bets} numbers',
            'amount': total_bet,
            'per_number': bet_per_number,
            'numbers': predicted_numbers,
            'won': won,
            'payout': round(payout + bet_per_number, 2) if won else 0,  # Total return (winnings + stake)
            'net': net_profit
        }
    else:
        # Tick wait counter even if not betting
        bankroll.tick_wait()
        bankroll.total_skips += 1
        session_mgr.record_skip()

    # Update all models with the new number
    predictor.update(number)

    # Auto-retrain LSTM in background if enough new spins have accumulated
    if predictor.lstm.needs_retrain(RETRAIN_INTERVAL):
        _sid = request.sid
        def _bg_retrain():
            result = predictor.train_lstm()
            predictor.save_state()
            socketio.emit('training_update', result, to=_sid)
        socketio.start_background_task(_bg_retrain)

    # Record in session
    prediction_for_record = current_state.get('last_prediction')
    session_mgr.record_spin(number, prediction_for_record, bet_info, bankroll.get_status())

    # Generate next prediction
    next_prediction = predictor.predict()
    should_bet, reason = bankroll.should_bet(next_prediction['confidence'], next_prediction['mode'])

    next_n_preds = len(next_prediction['top_numbers'])
    next_bet_per_number = bankroll.calculate_bet_amount(next_prediction['confidence'], 'straight', num_predictions=next_n_preds)
    next_total_bet = round(next_bet_per_number * next_n_preds, 2)

    next_recommended_bet = None
    if should_bet and next_prediction['top_numbers']:
        next_recommended_bet = {
            'type': 'straight',
            'value': str(next_prediction['anchors'][0]) if next_prediction.get('anchors') else str(next_prediction['top_numbers'][0]),
            'numbers': next_prediction['top_numbers'],
            'amount': next_bet_per_number,
            'total_bet': next_total_bet,
            'payout': '35:1',
            'risk': 'high',
            'probability': round(len(next_prediction['top_numbers']) / 37, 4)
        }

    current_state['last_prediction'] = next_prediction

    # Get money management advice for next spin
    money_advice = bankroll.get_advice(
        ai_confidence=next_prediction['confidence'],
        ai_mode=next_prediction['mode'],
        prediction=next_prediction
    )

    response = {
        'spin_result': {
            'number': number,
            'color': get_number_color(number),
            'bet_result': bet_info
        },
        'next_prediction': next_prediction,
        'should_bet': should_bet,
        'bet_reason': reason,
        'recommended_bet': next_recommended_bet,
        'bet_amount': next_bet_per_number,
        'total_bet': next_total_bet,
        'bankroll': bankroll.get_status(),
        'model_status': predictor.get_model_status(),
        'money_advice': money_advice
    }

    emit('spin_processed', response)


@socketio.on('train_model')
def handle_train_model():
    """Train AI from all files in the userdata/ folder.
    Scans userdata/ for .txt and .csv files, parses numbers from each,
    feeds them into models as clusters, trains LSTM, and returns a
    detailed summary of what was processed.
    """
    import os
    import time
    from collections import Counter
    from datetime import datetime

    sid = request.sid

    def _do_train():
        train_start = time.time()

        # Step 1: Scan userdata/ folder for .txt and .csv files
        if not os.path.exists(USERDATA_DIR):
            os.makedirs(USERDATA_DIR, exist_ok=True)
            socketio.emit('training_complete', {
                'status': 'no_data',
                'message': f'No files found in userdata/ folder. Place .txt or .csv files there with one number (0-36) per line.',
                'model_status': predictor.get_model_status()
            }, to=sid)
            return

        files_found = []
        for fname in sorted(os.listdir(USERDATA_DIR)):
            if fname.lower().endswith(('.txt', '.csv')) and not fname.startswith('.'):
                files_found.append(fname)

        if not files_found:
            socketio.emit('training_complete', {
                'status': 'no_data',
                'message': f'No .txt or .csv files found in userdata/ folder.',
                'model_status': predictor.get_model_status()
            }, to=sid)
            return

        # Step 2: Parse numbers from each file
        file_details = []
        all_numbers = []
        clusters = []

        for fname in files_found:
            filepath = os.path.join(USERDATA_DIR, fname)
            file_numbers = []
            skipped = 0
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Handle CSV: take first column
                        parts = line.replace(',', '\n').split('\n')
                        for part in parts:
                            part = part.strip()
                            if not part:
                                continue
                            try:
                                num = int(part)
                                if 0 <= num <= 36:
                                    file_numbers.append(num)
                                else:
                                    skipped += 1
                            except ValueError:
                                skipped += 1
            except Exception as e:
                file_details.append({
                    'filename': fname,
                    'numbers': 0,
                    'skipped': 0,
                    'error': str(e)
                })
                continue

            file_details.append({
                'filename': fname,
                'numbers': len(file_numbers),
                'skipped': skipped,
                'error': None
            })

            if file_numbers:
                all_numbers.extend(file_numbers)
                clusters.append({
                    'spins': file_numbers,
                    'size': len(file_numbers),
                    'source': fname
                })

            eventlet.sleep(0)  # Keep heartbeats alive

        if not all_numbers:
            socketio.emit('training_complete', {
                'status': 'no_valid_data',
                'message': 'Files found but no valid numbers (0-36) in them.',
                'files': file_details,
                'model_status': predictor.get_model_status()
            }, to=sid)
            return

        # Step 3: APPEND data into models (don't reset existing knowledge)
        # If models are empty (first train), use load_clusters for proper
        # cluster-aware loading. Otherwise use update_incremental to append.
        existing_spins = len(predictor.spin_history)

        if existing_spins == 0:
            # Models are empty — do a full cluster load (no data to lose)
            cluster_result = predictor.load_clusters(clusters)
            load_mode = 'initial_load'
            new_spins_added = cluster_result['total_spins']
        else:
            # Models already have data — append incrementally
            # Feed all userdata numbers on top of existing model knowledge
            update_result = predictor.update_incremental(all_numbers)
            load_mode = 'incremental_append'
            new_spins_added = update_result['numbers_added']

        eventlet.sleep(0)

        # Step 4: Train LSTM
        train_result = {'status': 'insufficient_data'}
        if predictor.lstm.can_train():
            train_result = predictor.train_lstm()
            eventlet.sleep(0)

        # Step 5: Save state for fast next session start
        predictor.save_state()

        train_time = round(time.time() - train_start, 2)

        # Step 6: Compute summary statistics
        freq_counts = Counter(all_numbers)
        total = len(all_numbers)
        expected = total / 37

        # Hot/cold numbers
        hot_numbers = []
        cold_numbers = []
        for num in range(37):
            count = freq_counts.get(num, 0)
            ratio = round(count / expected, 2) if expected > 0 else 0
            if ratio >= 1.5:
                hot_numbers.append({'number': num, 'count': count, 'ratio': ratio})
            elif ratio <= 0.5:
                cold_numbers.append({'number': num, 'count': count, 'ratio': ratio})

        hot_numbers.sort(key=lambda x: -x['ratio'])
        cold_numbers.sort(key=lambda x: x['ratio'])

        # Color distribution
        reds = sum(1 for n in all_numbers if get_number_color(n) == 'red')
        blacks = sum(1 for n in all_numbers if get_number_color(n) == 'black')
        greens = sum(1 for n in all_numbers if get_number_color(n) == 'green')

        # Last 10 for UI (from full model history, not just new files)
        last_10 = list(predictor.spin_history[-10:])

        total_model_spins = len(predictor.spin_history)
        mode_label = 'loaded' if load_mode == 'initial_load' else 'appended'
        msg = f'Successfully {mode_label} {new_spins_added} spins from {len(files_found)} files in {train_time}s.'
        if existing_spins > 0 and load_mode == 'incremental_append':
            msg += f' Total model knowledge: {total_model_spins} spins ({existing_spins} existing + {new_spins_added} new).'

        socketio.emit('training_complete', {
            'status': 'trained',
            'message': msg,
            'files': file_details,
            'summary': {
                'total_files': len(files_found),
                'total_spins': total_model_spins,
                'unique_numbers': len(freq_counts),
                'training_time': train_time,
                'lstm_result': train_result,
                'hot_numbers': hot_numbers[:5],
                'cold_numbers': cold_numbers[:5],
                'color_distribution': {
                    'red': reds,
                    'red_pct': round(reds / total * 100, 1) if total > 0 else 0,
                    'black': blacks,
                    'black_pct': round(blacks / total * 100, 1) if total > 0 else 0,
                    'green': greens,
                    'green_pct': round(greens / total * 100, 1) if total > 0 else 0,
                }
            },
            'last_numbers': last_10,
            'model_status': predictor.get_model_status()
        }, to=sid)

    socketio.start_background_task(_do_train)


@socketio.on('get_sessions')
def handle_get_sessions():
    """Get all historical sessions."""
    analytics = session_mgr.get_performance_analytics()
    emit('sessions_list', analytics)


def _parse_numbers(raw_text):
    """Parse roulette numbers from text (newline or comma separated)."""
    numbers = []
    lines = raw_text.replace(',', '\n').split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            num = int(line)
            if 0 <= num <= 36:
                numbers.append(num)
        except ValueError:
            continue
    return numbers


@socketio.on('import_data')
def handle_import_data(data):
    """Import historical spin data INCREMENTALLY into current trained models.
    New data is applied on top of existing model knowledge - models are NOT reset.
    Data is also saved as a cluster for future session bootstrapping."""
    import traceback

    print(f"[import_data] Received request, skip_training={data.get('skip_training', False)}")
    raw_text = data.get('text', '')
    if not raw_text.strip():
        emit('error', {'message': 'No data provided.'})
        return

    numbers = _parse_numbers(raw_text)
    print(f"[import_data] Parsed {len(numbers)} numbers")

    if not numbers:
        emit('error', {'message': 'No valid numbers (0-36) found in data.'})
        return

    # Capture client session ID for background emit
    sid = request.sid
    skip_training = data.get('skip_training', False)

    def _do_import():
        try:
            # Save as independent imported cluster/session (for persistence)
            from app.session.session_manager import SessionManager
            import_mgr = SessionManager()
            import_mgr.create_session()
            import_mgr.current_session['status'] = 'imported'
            import_mgr.current_session['stats']['total_spins'] = len(numbers)
            import_mgr.current_session['spins'] = [
                {'spin_number': i + 1, 'actual_number': num}
                for i, num in enumerate(numbers)
            ]
            import_mgr.end_session()
            print("[import_data] Saved cluster")

            # INCREMENTAL: Apply new data on top of existing trained models
            # Models are NOT reset - new numbers are fed one by one via update()
            update_result = predictor.update_incremental(numbers)
            print(f"[import_data] Incremental update done: {update_result}")

            # Yield to eventlet to keep heartbeats alive
            eventlet.sleep(0)

            # Save state so next session start uses cache
            predictor.save_state()

            train_result = {'status': 'skipped' if skip_training else 'pending_background'}

            total_clusters = len(session_mgr.get_all_training_clusters())

            # Send last 10 numbers so UI can display them immediately
            last_10 = list(predictor.spin_history[-10:])

            socketio.emit('import_complete', {
                'numbers_imported': len(numbers),
                'clusters_total': total_clusters,
                'total_spins': update_result['total_spins'],
                'train_result': train_result,
                'training_skipped': skip_training,
                'model_status': predictor.get_model_status(),
                'update_mode': 'incremental',
                'last_numbers': last_10,
                'message': f'Imported {len(numbers)} spins incrementally into current models. Total: {update_result["total_spins"]} spins.{"(Neural Net training skipped)" if skip_training else ""}'
            }, to=sid)
            print("[import_data] Emitted import_complete successfully")

            # Train LSTM in background AFTER emitting import_complete
            if not skip_training and predictor.lstm.can_train():
                result = predictor.train_lstm()
                predictor.save_state()
                socketio.emit('training_update', result, to=sid)
                print(f"[import_data] Background training done: {result}")
        except Exception as e:
            print(f"[import_data] ERROR: {e}")
            traceback.print_exc()
            socketio.emit('error', {'message': f'Import failed: {str(e)}'}, to=sid)

    socketio.start_background_task(_do_import)


@socketio.on('undo_spin')
def handle_undo_spin():
    """Undo/revert the last entered spin number."""
    if not current_state['session_active']:
        emit('error', {'message': 'No active session.'})
        return

    # Remove from session storage
    removed = session_mgr.undo_last_spin()
    if removed is None:
        emit('error', {'message': 'No spins to undo.'})
        return

    # Remove from predictor models
    predictor.undo_last()

    # Reverse bankroll state if a bet was placed on the undone spin
    bankroll.undo_last_bet()

    # Persist updated state
    predictor.save_state()

    # Get fresh prediction after undo
    prediction = predictor.predict() if len(predictor.spin_history) >= 3 else None
    should_bet = False
    reason = 'WAIT'
    recommended_bet = None

    if prediction:
        should_bet, reason = bankroll.should_bet(prediction['confidence'], prediction['mode'])
        if should_bet and prediction['bets']:
            primary_bet = prediction['bets'][0]
            bet_amount = bankroll.calculate_bet_amount(prediction['confidence'], primary_bet['type'])
            recommended_bet = {**primary_bet, 'amount': bet_amount}

    current_state['last_prediction'] = prediction

    # Get money management advice
    money_advice = bankroll.get_advice(
        ai_confidence=prediction['confidence'] if prediction else 0,
        ai_mode=prediction['mode'] if prediction else 'WAIT',
        prediction=prediction
    )

    emit('spin_undone', {
        'removed_number': removed,
        'removed_color': get_number_color(removed),
        'prediction': prediction,
        'should_bet': should_bet,
        'bet_reason': reason,
        'recommended_bet': recommended_bet,
        'bankroll': bankroll.get_status(),
        'model_status': predictor.get_model_status(),
        'remaining_spins': len(predictor.spin_history),
        'money_advice': money_advice
    })


@socketio.on('run_test')
def handle_run_test(data):
    """Test mode: feed a dataset and get accuracy report showing how well AI predicts."""
    raw_text = data.get('text', '')
    if not raw_text.strip():
        emit('error', {'message': 'No test data provided.'})
        return

    numbers = _parse_numbers(raw_text)

    if len(numbers) < 5:
        emit('error', {'message': f'Need at least 5 numbers for test. Got {len(numbers)}.'})
        return

    # Run test
    report = predictor.run_test(numbers)

    emit('test_complete', report)


@socketio.on('get_status')
def handle_get_status():
    """Get current system status."""
    emit('status_update', {
        'session_active': current_state['session_active'],
        'bankroll': bankroll.get_status(),
        'model_status': predictor.get_model_status()
    })
