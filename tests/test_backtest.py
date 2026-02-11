"""
Comprehensive Backtest Module for AI Roulette Predictor.

Tests the full prediction + money management system against real historical data.

Rules:
- Session 1: Train on numbers[0:50], bet from numbers[50:]
- Session 2: Train on numbers[0:51], bet from numbers[51:]
- Session N: Train on numbers[0:49+N], bet from numbers[49+N:]
- Each session starts with $4,000, target +$100 profit
- Session ends when: target reached, stop-loss hit, or data runs out
- Data1 and Data2 tested independently
"""

import sys
import os
import json
import csv
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Remove LSTM model file BEFORE importing anything that triggers loading
_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'models')
_lstm_path = os.path.join(_models_dir, 'lstm_model.pth')
if os.path.exists(_lstm_path):
    os.remove(_lstm_path)
    print(f"[BACKTEST] Removed {_lstm_path} for fair backtest")

from app.ml.ensemble import EnsemblePredictor
from app.money.bankroll_manager import BankrollManager
from config import (
    INITIAL_BANKROLL, SESSION_TARGET, STOP_LOSS_THRESHOLD,
    TOP_PREDICTIONS_COUNT, CONFIDENCE_BET_THRESHOLD
)


def load_data(filepath):
    """Load numbers from a text file (one number per line)."""
    numbers = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    num = int(line)
                    if 0 <= num <= 36:
                        numbers.append(num)
                except ValueError:
                    continue
    return numbers


def run_session(all_numbers, train_count, session_num):
    """Run a single betting session.

    Args:
        all_numbers: Full list of spin numbers
        train_count: How many numbers to use for training (from start)
        session_num: Session number for labelling

    Returns:
        dict with session results and spin-by-spin history
    """
    if train_count >= len(all_numbers):
        return None  # No betting data left

    train_data = all_numbers[:train_count]
    bet_data = all_numbers[train_count:]

    if not bet_data:
        return None

    # Fresh predictor and bankroll for each session
    predictor = EnsemblePredictor()
    bankroll = BankrollManager()

    # Reset LSTM/GRU to truly untrained state for fair backtest
    # Reinitialize all model weights to random (undo any loaded checkpoint)
    for module in predictor.lstm.model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    predictor.lstm.is_trained = False
    predictor.lstm.training_loss_history = []
    predictor.lstm.spin_history = []
    # Disable auto-retrain: monkey-patch needs_retrain to prevent
    # automatic LSTM training during backtest (user said "skip training")
    predictor.lstm.needs_retrain = lambda interval: False

    # Feed training data (no LSTM training — matches "Import Without Training")
    for num in train_data:
        predictor.update(num)

    # Betting phase
    spin_history = []
    spins_played = 0

    for i, actual_number in enumerate(bet_data):
        # Get prediction BEFORE revealing the number
        prediction = predictor.predict()
        confidence = prediction['confidence']
        mode = prediction['mode']
        predicted_numbers = prediction.get('top_numbers', [])
        anchors = prediction.get('anchors', [])

        # Money management decision
        should_bet, reason = bankroll.should_bet(confidence, mode)
        bet_per_number = bankroll.calculate_bet_amount(confidence, 'straight')

        # Process the spin
        won = False
        payout = 0
        net_profit = 0
        total_bet = 0
        bet_placed = False

        if should_bet and predicted_numbers:
            bet_placed = True
            num_bets = len(predicted_numbers)
            total_bet = round(bet_per_number * num_bets, 2)

            if actual_number in predicted_numbers:
                won = True
                payout = bet_per_number * 35
                net_profit = round(payout + bet_per_number - total_bet, 2)
            else:
                won = False
                net_profit = round(-total_bet, 2)

            bankroll.process_result(total_bet, won, payout)
        else:
            # Not betting — tick wait counter
            bankroll.tick_wait()
            bankroll.total_skips += 1

        # Record spin
        spin_record = {
            'spin_num': i + 1,
            'data_index': train_count + i + 1,  # 1-based index in original data
            'actual': actual_number,
            'predicted': predicted_numbers[:4] if predicted_numbers else [],  # Show anchors
            'all_predicted': predicted_numbers,
            'confidence': confidence,
            'mode': mode,
            'bet_placed': bet_placed,
            'total_bet': total_bet,
            'per_number': bet_per_number if bet_placed else 0,
            'won': won if bet_placed else None,
            'net': net_profit,
            'bankroll': round(bankroll.bankroll, 2),
            'reason': reason if not should_bet else 'BET',
            'hit': actual_number in predicted_numbers if predicted_numbers else False
        }
        spin_history.append(spin_record)
        spins_played += 1

        # Update predictor with actual result
        predictor.update(actual_number)

        # Check session end conditions
        if bankroll.profit_loss >= SESSION_TARGET:
            break
        if bankroll.stop_loss_hit:
            break

    # Build session summary
    bets_placed = [s for s in spin_history if s['bet_placed']]
    wins = [s for s in bets_placed if s['won']]
    losses = [s for s in bets_placed if not s['won']]
    skips = [s for s in spin_history if not s['bet_placed']]
    hits = [s for s in spin_history if s['hit']]

    result = {
        'session_num': session_num,
        'train_count': train_count,
        'bet_start_index': train_count + 1,
        'total_spins': spins_played,
        'total_bets': len(bets_placed),
        'total_skips': len(skips),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / len(bets_placed) * 100, 1) if bets_placed else 0,
        'hit_rate': round(len(hits) / spins_played * 100, 1) if spins_played else 0,
        'final_bankroll': round(bankroll.bankroll, 2),
        'profit_loss': round(bankroll.profit_loss, 2),
        'target_reached': bankroll.profit_loss >= SESSION_TARGET,
        'stop_loss_hit': bankroll.stop_loss_hit,
        'data_exhausted': (not bankroll.profit_loss >= SESSION_TARGET) and (not bankroll.stop_loss_hit),
        'peak_bankroll': round(bankroll.peak_bankroll, 2),
        'max_drawdown': round(bankroll.peak_bankroll - bankroll.trough_bankroll, 2),
        'max_consecutive_losses': bankroll.max_consecutive_losses,
        'total_wagered': round(sum(s['total_bet'] for s in bets_placed), 2),
        'total_won': round(sum(s['net'] for s in wins), 2),
        'total_lost': round(sum(abs(s['net']) for s in losses), 2),
        'spin_history': spin_history
    }

    return result


def run_backtest(data_filepath, data_label):
    """Run full backtest on a dataset with multiple sessions.

    Each session trains on an increasing number of numbers,
    then bets on the rest until target is reached or data runs out.
    """
    all_numbers = load_data(data_filepath)
    total_numbers = len(all_numbers)

    print(f"\n{'='*80}")
    print(f"  BACKTEST: {data_label}")
    print(f"  File: {data_filepath}")
    print(f"  Total numbers: {total_numbers}")
    print(f"  Initial bankroll: ${INITIAL_BANKROLL:.2f}")
    print(f"  Session target: +${SESSION_TARGET:.2f}")
    print(f"  Stop loss: ${STOP_LOSS_THRESHOLD:.2f}")
    print(f"  Bet: ${1:.2f}/number × {TOP_PREDICTIONS_COUNT} numbers = ${TOP_PREDICTIONS_COUNT:.2f}/spin")
    print(f"{'='*80}")

    sessions = []
    session_num = 0
    train_count = 50  # Start with 50 training numbers

    while train_count < total_numbers:
        session_num += 1
        print(f"\n--- Session {session_num}: Train on [1..{train_count}], Bet from [{train_count+1}..] ---")

        result = run_session(all_numbers, train_count, session_num)
        if result is None:
            print(f"  No betting data left. Stopping.")
            break

        sessions.append(result)

        # Print session summary
        status = "TARGET REACHED" if result['target_reached'] else \
                 "STOP-LOSS HIT" if result['stop_loss_hit'] else \
                 "DATA EXHAUSTED"

        print(f"  Status: {status}")
        print(f"  Spins: {result['total_spins']} | Bets: {result['total_bets']} | Skips: {result['total_skips']}")
        print(f"  Wins: {result['wins']} | Losses: {result['losses']} | Win Rate: {result['win_rate']}%")
        print(f"  Hit Rate: {result['hit_rate']}% (number in predicted set)")
        print(f"  P&L: ${result['profit_loss']:+.2f} | Final: ${result['final_bankroll']:.2f}")
        print(f"  Peak: ${result['peak_bankroll']:.2f} | Max Drawdown: ${result['max_drawdown']:.2f}")

        # Next session: train on one more number
        train_count += 1

        # If target was not reached and data exhausted, stop
        if result['data_exhausted']:
            print(f"  Data exhausted at session {session_num}. No more sessions possible.")
            break

    return sessions


def print_detailed_report(sessions, data_label):
    """Print a comprehensive report with spin-by-spin history for each session."""
    print(f"\n\n{'#'*80}")
    print(f"  DETAILED REPORT: {data_label}")
    print(f"{'#'*80}")

    # Overall summary
    total_sessions = len(sessions)
    targets_reached = sum(1 for s in sessions if s['target_reached'])
    stop_losses = sum(1 for s in sessions if s['stop_loss_hit'])
    data_exhausted = sum(1 for s in sessions if s['data_exhausted'])
    all_bets = sum(s['total_bets'] for s in sessions)
    all_wins = sum(s['wins'] for s in sessions)
    all_losses_count = sum(s['losses'] for s in sessions)
    all_spins = sum(s['total_spins'] for s in sessions)
    total_profit = sum(s['profit_loss'] for s in sessions)
    total_wagered = sum(s['total_wagered'] for s in sessions)

    print(f"\n  OVERALL SUMMARY")
    print(f"  {'─'*60}")
    print(f"  Total Sessions:          {total_sessions}")
    print(f"  Targets Reached ($100):  {targets_reached} ({round(targets_reached/total_sessions*100,1) if total_sessions else 0}%)")
    print(f"  Stop-Loss Hit:           {stop_losses}")
    print(f"  Data Exhausted:          {data_exhausted}")
    print(f"  Total Spins Played:      {all_spins}")
    print(f"  Total Bets Placed:       {all_bets}")
    print(f"  Total Wins:              {all_wins}")
    print(f"  Total Losses:            {all_losses_count}")
    print(f"  Overall Win Rate:        {round(all_wins/all_bets*100,1) if all_bets else 0}%")
    print(f"  Total Wagered:           ${total_wagered:.2f}")
    print(f"  Combined P&L:            ${total_profit:+.2f}")
    print(f"  Avg P&L per Session:     ${total_profit/total_sessions:+.2f}" if total_sessions else "")
    print(f"  Avg Spins to Target:     {round(sum(s['total_spins'] for s in sessions if s['target_reached'])/targets_reached, 1) if targets_reached else 'N/A'}")

    # Per-session detailed history
    for session in sessions:
        sn = session['session_num']
        status = "TARGET REACHED" if session['target_reached'] else \
                 "STOP-LOSS HIT" if session['stop_loss_hit'] else \
                 "DATA EXHAUSTED"

        print(f"\n\n  {'='*76}")
        print(f"  SESSION {sn} — {status}")
        print(f"  Training: numbers [1..{session['train_count']}] | Betting from: number {session['bet_start_index']}")
        print(f"  {'='*76}")
        print(f"  Spins: {session['total_spins']} | Bets: {session['total_bets']} | Wins: {session['wins']} | Losses: {session['losses']}")
        print(f"  Win Rate: {session['win_rate']}% | Hit Rate: {session['hit_rate']}%")
        print(f"  Final Bankroll: ${session['final_bankroll']:.2f} | P&L: ${session['profit_loss']:+.2f}")
        print(f"  Peak: ${session['peak_bankroll']:.2f} | Max Drawdown: ${session['max_drawdown']:.2f}")
        print(f"  Total Wagered: ${session['total_wagered']:.2f} | Won: ${session['total_won']:.2f} | Lost: ${session['total_lost']:.2f}")
        print(f"  Max Consecutive Losses: {session['max_consecutive_losses']}")

        # Spin-by-spin table
        print(f"\n  {'─'*76}")
        print(f"  {'#':>4} {'Data#':>5} {'Actual':>6} {'Predicted':>20} {'Conf':>5} {'Action':>10} {'Bet':>7} {'W/L':>5} {'Net':>8} {'Bankroll':>10}")
        print(f"  {'─'*76}")

        for spin in session['spin_history']:
            pred_str = ','.join(str(n) for n in spin['predicted']) if spin['predicted'] else '-'
            if len(pred_str) > 18:
                pred_str = pred_str[:17] + '…'

            if spin['bet_placed']:
                action = 'BET'
                wl = 'WIN' if spin['won'] else 'LOSS'
                bet_str = f"${spin['total_bet']:.2f}"
                net_str = f"${spin['net']:+.2f}"
            else:
                action = spin['reason'][:10] if spin['reason'] else 'SKIP'
                wl = '-'
                bet_str = '-'
                net_str = '-'

            hit_marker = '*' if spin['hit'] else ' '

            print(f"  {spin['spin_num']:>4} {spin['data_index']:>5} {hit_marker}{spin['actual']:>5} {pred_str:>20} {spin['confidence']:>4.0f}% {action:>10} {bet_str:>7} {wl:>5} {net_str:>8} ${spin['bankroll']:>9.2f}")

        print(f"  {'─'*76}")


def save_report_to_file(sessions, data_label, filepath):
    """Save the detailed report as JSON for further analysis."""
    report = {
        'data_label': data_label,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'initial_bankroll': INITIAL_BANKROLL,
            'session_target': SESSION_TARGET,
            'stop_loss': STOP_LOSS_THRESHOLD,
            'predictions_per_spin': TOP_PREDICTIONS_COUNT,
            'confidence_threshold': CONFIDENCE_BET_THRESHOLD
        },
        'sessions': []
    }

    for s in sessions:
        session_data = {k: v for k, v in s.items() if k != 'spin_history'}
        session_data['spins'] = []
        for spin in s['spin_history']:
            session_data['spins'].append({
                'spin_num': spin['spin_num'],
                'data_index': spin['data_index'],
                'actual': spin['actual'],
                'predicted_anchors': spin['predicted'],
                'confidence': spin['confidence'],
                'bet_placed': spin['bet_placed'],
                'total_bet': spin['total_bet'],
                'won': spin['won'],
                'net': spin['net'],
                'bankroll': spin['bankroll'],
                'hit': spin['hit']
            })
        report['sessions'].append(session_data)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  JSON report saved to: {filepath}")


def save_csv_reports(sessions, data_label, report_dir):
    """Save two CSV files: session summary + spin-by-spin detail."""

    # ── 1) Session Summary CSV ──────────────────────────────────────────
    summary_path = os.path.join(report_dir, f'backtest_{data_label.lower()}_sessions.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Session', 'Train_Count', 'Bet_Start', 'Status',
            'Total_Spins', 'Bets_Placed', 'Skips', 'Wins', 'Losses',
            'Win_Rate_%', 'Hit_Rate_%', 'Final_Bankroll', 'Profit_Loss',
            'Peak_Bankroll', 'Max_Drawdown', 'Max_Consec_Losses',
            'Total_Wagered', 'Total_Won', 'Total_Lost'
        ])
        for s in sessions:
            status = 'TARGET_REACHED' if s['target_reached'] else \
                     'STOP_LOSS' if s['stop_loss_hit'] else 'DATA_EXHAUSTED'
            writer.writerow([
                s['session_num'], s['train_count'], s['bet_start_index'], status,
                s['total_spins'], s['total_bets'], s['total_skips'],
                s['wins'], s['losses'],
                s['win_rate'], s['hit_rate'],
                s['final_bankroll'], s['profit_loss'],
                s['peak_bankroll'], s['max_drawdown'],
                s['max_consecutive_losses'],
                s['total_wagered'], s['total_won'], s['total_lost']
            ])
    print(f"  CSV session summary saved to: {summary_path}")

    # ── 2) Spin-by-Spin Detail CSV ──────────────────────────────────────
    detail_path = os.path.join(report_dir, f'backtest_{data_label.lower()}_spins.csv')
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Session', 'Spin_Num', 'Data_Index', 'Actual_Number',
            'Predicted_Anchors', 'All_Predicted', 'Confidence_%',
            'Action', 'Bet_Per_Number', 'Total_Bet',
            'Result', 'Net_Profit', 'Bankroll', 'Hit'
        ])
        for s in sessions:
            for spin in s['spin_history']:
                anchors = '|'.join(str(n) for n in spin['predicted']) if spin['predicted'] else ''
                all_pred = '|'.join(str(n) for n in spin['all_predicted']) if spin['all_predicted'] else ''

                if spin['bet_placed']:
                    action = 'BET'
                    result = 'WIN' if spin['won'] else 'LOSS'
                else:
                    action = spin.get('reason', 'SKIP')
                    result = ''

                writer.writerow([
                    s['session_num'], spin['spin_num'], spin['data_index'],
                    spin['actual'],
                    anchors, all_pred, spin['confidence'],
                    action, spin['per_number'], spin['total_bet'],
                    result, spin['net'], spin['bankroll'],
                    'YES' if spin['hit'] else 'NO'
                ])
    print(f"  CSV spin detail saved to: {detail_path}")


def main():
    """Run backtests on both data files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'userdata')
    report_dir = os.path.join(base_dir, 'data', 'reports')
    os.makedirs(report_dir, exist_ok=True)

    data_files = [
        (os.path.join(data_dir, 'data1.txt'), 'DATA1'),
        (os.path.join(data_dir, 'data2.txt'), 'DATA2'),
    ]

    all_results = {}

    for filepath, label in data_files:
        if not os.path.exists(filepath):
            print(f"\nWARNING: {filepath} not found, skipping.")
            continue

        # Run backtest
        sessions = run_backtest(filepath, label)

        # Print detailed report
        print_detailed_report(sessions, label)

        # Save JSON
        json_path = os.path.join(report_dir, f'backtest_{label.lower()}.json')
        save_report_to_file(sessions, label, json_path)

        # Save CSVs
        save_csv_reports(sessions, label, report_dir)

        all_results[label] = sessions

    # Final comparison
    if len(all_results) == 2:
        print(f"\n\n{'='*80}")
        print(f"  COMPARISON: DATA1 vs DATA2")
        print(f"{'='*80}")
        for label, sessions in all_results.items():
            targets = sum(1 for s in sessions if s['target_reached'])
            total = len(sessions)
            total_pnl = sum(s['profit_loss'] for s in sessions)
            total_bets = sum(s['total_bets'] for s in sessions)
            total_wins = sum(s['wins'] for s in sessions)
            avg_spins = round(sum(s['total_spins'] for s in sessions if s['target_reached'])/targets, 1) if targets else 'N/A'

            print(f"\n  {label}:")
            print(f"    Sessions: {total} | Targets Hit: {targets}/{total} ({round(targets/total*100,1) if total else 0}%)")
            print(f"    Combined P&L: ${total_pnl:+.2f}")
            print(f"    Win Rate: {round(total_wins/total_bets*100,1) if total_bets else 0}%")
            print(f"    Avg Spins to Target: {avg_spins}")

    print(f"\n{'='*80}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
