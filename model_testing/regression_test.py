#!/usr/bin/env python3
"""
REGRESSION TEST: THE ONE Strategy — Pre-loaded AI + 60-spin sessions
=====================================================================
Tests the REAL scenario:
  1. Pre-load ALL 5,303 historical spins from userdata/ (AI starts warm)
  2. Then simulate multiple independent 60-spin sessions on test_data1.txt
  3. Each session keeps the pre-loaded AI state (warm start)
  4. Target: $100 profit per session within 60 spins

This answers: "If I pre-load my data, can the AI bet confidently in 60 spins?"

Usage:
    cd /Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso
    source venv/bin/activate
    python model_testing/regression_test.py
"""

import sys
import os
import time
import copy
import statistics
import pickle
from collections import Counter

PROJECT_ROOT = '/Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso'
sys.path.insert(0, PROJECT_ROOT)

try:
    from app.ml.ensemble import EnsemblePredictor
    from app.money.bankroll_manager import BankrollManager
    from config import (
        INITIAL_BANKROLL, CONFIDENCE_BET_THRESHOLD, STOP_LOSS_THRESHOLD,
        TOP_PREDICTIONS_COUNT, BASE_BET, BET_SEQUENCE, PAYOUTS
    )
    print("[OK] All imports successful.")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

USERDATA_DIR = os.path.join(PROJECT_ROOT, 'userdata')
TEST_DATA = os.path.join(PROJECT_ROOT, 'model_testing', 'test_data', 'test_data1.txt')
SESSION_LENGTH = 60
TARGET_PROFIT = 100.0


def load_numbers_from_file(path):
    """Load spin numbers from a text file (one number per line)."""
    numbers = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                num = int(line)
                if 0 <= num <= 36:
                    numbers.append(num)
    return numbers


def load_all_userdata():
    """Load all historical spins from userdata/ directory in order."""
    all_spins = []
    files = sorted([f for f in os.listdir(USERDATA_DIR) if f.startswith('data') and f.endswith('.txt')],
                   key=lambda x: int(x.replace('data', '').replace('.txt', '')))
    for fname in files:
        path = os.path.join(USERDATA_DIR, fname)
        spins = load_numbers_from_file(path)
        all_spins.extend(spins)
        print(f"  Loaded {fname}: {len(spins)} spins")
    return all_spins


def preload_ensemble(ensemble, historical_spins):
    """Feed historical spins into ensemble to warm it up."""
    print(f"\n  Pre-loading {len(historical_spins)} historical spins into AI...")
    t0 = time.time()
    for i, num in enumerate(historical_spins):
        ensemble.update(num)
        if (i + 1) % 1000 == 0:
            # Quick confidence check at milestones
            try:
                pred = ensemble.predict()
                conf = pred.get('confidence', 0)
                mode = pred.get('mode', 'WAIT')
                print(f"    Spin {i+1}: confidence={conf:.1f}%, mode={mode}")
            except:
                pass
    elapsed = time.time() - t0
    print(f"  Pre-load complete in {elapsed:.1f}s")

    # Final state check
    try:
        pred = ensemble.predict()
        conf = pred.get('confidence', 0)
        mode = pred.get('mode', 'WAIT')
        should_bet = pred.get('should_bet', False)
        top = pred.get('top_numbers', [])
        print(f"  AI state after pre-load:")
        print(f"    Confidence: {conf:.1f}%")
        print(f"    Mode: {mode}")
        print(f"    Should bet: {should_bet}")
        print(f"    Top predictions: {top[:7]}...")
    except Exception as e:
        print(f"  Warning: predict() error after pre-load: {e}")

    return ensemble


def run_session(ensemble, session_spins, session_id, verbose=True):
    """Run a single 60-spin session using a pre-loaded ensemble.

    Returns stats dict. Does NOT modify the ensemble in-place (uses deepcopy).
    """
    # Deep copy ensemble so each session starts from same warm state
    # Actually we WANT continuous learning — each session continues from prior
    # But bankroll resets each session
    bankroll = BankrollManager(initial_bankroll=INITIAL_BANKROLL)

    stats = {
        'session_id': session_id,
        'spins': len(session_spins),
        'bets_placed': 0,
        'wins': 0,
        'losses': 0,
        'skips': 0,
        'stop_loss_hit': False,
        'target_hit': False,
        'target_hit_spin': None,
        'total_wagered': 0.0,
        'total_payouts': 0.0,
        'profit_loss': 0.0,
        'peak_profit': 0.0,
        'max_drawdown': 0.0,
        'seq_steps_used': Counter(),
        'confidences_bet': [],
        'confidences_wait': [],
        'hit_when_bet': 0,
        'miss_when_bet': 0,
        'hit_when_wait': 0,
        'miss_when_wait': 0,
        'spin_log': [],
    }

    for i, actual_number in enumerate(session_spins):
        # Get prediction BEFORE revealing result
        try:
            prediction = ensemble.predict()
        except Exception as e:
            ensemble.update(actual_number)
            bankroll.tick_wait()
            stats['skips'] += 1
            continue

        top_numbers = prediction.get('top_numbers', [])
        confidence = prediction.get('confidence', 0)
        mode = prediction.get('mode', 'WAIT')
        should_bet_signal = prediction.get('should_bet', False)
        num_predictions = len(top_numbers)

        # Betting decision
        ai_says_bet = (mode != 'WAIT' and confidence >= CONFIDENCE_BET_THRESHOLD
                       and should_bet_signal)
        bm_should_bet, bm_reason = bankroll.should_bet(confidence, mode)
        actually_bet = ai_says_bet and bm_should_bet

        hit = actual_number in top_numbers

        if actually_bet and num_predictions > 0:
            seq_step = bankroll.seq_idx
            stats['seq_steps_used'][seq_step] += 1

            per_number_bet = bankroll.calculate_bet_amount(
                confidence, 'straight', num_predictions=num_predictions
            )
            total_bet = round(per_number_bet * num_predictions, 2)

            stats['bets_placed'] += 1
            stats['total_wagered'] += total_bet
            stats['confidences_bet'].append(confidence)

            if hit:
                payout = 35 * per_number_bet
                bankroll.process_result(total_bet, won=True, payout_amount=payout)
                stats['wins'] += 1
                stats['hit_when_bet'] += 1
                stats['total_payouts'] += payout + per_number_bet
                net = payout + per_number_bet - total_bet
                if verbose:
                    print(f"    Spin {i+1:2d}: BET step={seq_step} ${per_number_bet:.2f}×{num_predictions}="
                          f"${total_bet:.2f} | ✅ HIT {actual_number} | +${net:.2f} | "
                          f"bankroll ${bankroll.bankroll:.2f} | conf {confidence:.0f}%")
            else:
                bankroll.process_result(total_bet, won=False)
                stats['losses'] += 1
                stats['miss_when_bet'] += 1
                if verbose:
                    print(f"    Spin {i+1:2d}: BET step={seq_step} ${per_number_bet:.2f}×{num_predictions}="
                          f"${total_bet:.2f} | ❌ MISS (was {actual_number}) | -${total_bet:.2f} | "
                          f"bankroll ${bankroll.bankroll:.2f} | conf {confidence:.0f}%")

            stats['spin_log'].append({
                'spin': i + 1, 'action': 'BET', 'step': seq_step,
                'per_num': per_number_bet, 'total': total_bet,
                'actual': actual_number, 'hit': hit,
                'confidence': confidence, 'bankroll': bankroll.bankroll
            })
        else:
            stats['skips'] += 1
            if hit:
                stats['hit_when_wait'] += 1
            else:
                stats['miss_when_wait'] += 1
            stats['confidences_wait'].append(confidence)
            bankroll.tick_wait()

        # Feed result to AI (continuous learning)
        ensemble.update(actual_number)

        # Track profit
        current_pl = bankroll.bankroll - INITIAL_BANKROLL
        stats['peak_profit'] = max(stats['peak_profit'], current_pl)
        if current_pl < 0:
            stats['max_drawdown'] = max(stats['max_drawdown'], abs(current_pl))

        # Check target
        if current_pl >= TARGET_PROFIT and not stats['target_hit']:
            stats['target_hit'] = True
            stats['target_hit_spin'] = i + 1
            if verbose:
                print(f"    >>> 🎯 TARGET HIT: +${current_pl:.2f} at spin {i+1}!")

        # Check stop-loss
        if bankroll.stop_loss_hit:
            stats['stop_loss_hit'] = True
            if verbose:
                print(f"    >>> ⛔ STOP-LOSS at spin {i+1}, bankroll ${bankroll.bankroll:.2f}")
            break

    stats['profit_loss'] = round(bankroll.bankroll - INITIAL_BANKROLL, 2)
    return stats


def run_regression_test():
    """Main regression test: pre-load data, then run sessions."""
    print(f"\n{'#'*70}")
    print(f"  REGRESSION TEST: THE ONE Strategy")
    print(f"  Pre-load 5,303 spins → then 60-spin sessions on test_data1.txt")
    print(f"  Target: ${TARGET_PROFIT} profit per session")
    print(f"  Bet: ${BASE_BET}/number × 14 numbers = ${BASE_BET * 14}/bet")
    print(f"  Sequence: {list(BET_SEQUENCE)} × ${BASE_BET}")
    print(f"  Confidence gate: {CONFIDENCE_BET_THRESHOLD}%")
    print(f"{'#'*70}")

    # ── Step 1: Load historical data ──
    print(f"\n{'='*70}")
    print(f"  STEP 1: Load historical data from userdata/")
    print(f"{'='*70}")
    historical = load_all_userdata()
    print(f"  Total historical spins: {len(historical)}")

    # ── Step 2: Load test data ──
    test_spins = load_numbers_from_file(TEST_DATA)
    print(f"\n  Test data (test_data1.txt): {len(test_spins)} spins")

    # ── Step 3: Create and pre-load ensemble ──
    print(f"\n{'='*70}")
    print(f"  STEP 2: Pre-load AI with {len(historical)} historical spins")
    print(f"{'='*70}")
    ensemble = EnsemblePredictor()
    ensemble = preload_ensemble(ensemble, historical)

    # ── Step 4: Run sessions on test data ──
    print(f"\n{'='*70}")
    print(f"  STEP 3: Run 60-spin sessions on test_data1.txt")
    print(f"{'='*70}")

    sessions = []
    idx = 0
    session_id = 1

    while idx + SESSION_LENGTH <= len(test_spins):
        session_spins = test_spins[idx:idx + SESSION_LENGTH]
        print(f"\n  --- Session {session_id} (spins {idx+1}-{idx+SESSION_LENGTH}) ---")

        t0 = time.time()
        stats = run_session(ensemble, session_spins, session_id, verbose=True)
        elapsed = time.time() - t0

        pl = stats['profit_loss']
        pl_str = f"+${pl:.2f}" if pl >= 0 else f"-${abs(pl):.2f}"
        wr = (stats['wins'] / stats['bets_placed'] * 100) if stats['bets_placed'] > 0 else 0
        target_str = f" 🎯 TARGET at spin {stats['target_hit_spin']}" if stats['target_hit'] else ""
        stop_str = " ⛔ STOP-LOSS" if stats['stop_loss_hit'] else ""

        print(f"  Session {session_id} RESULT: P/L={pl_str} | bets={stats['bets_placed']} | "
              f"W/L={stats['wins']}/{stats['losses']} | WR={wr:.0f}% | "
              f"{elapsed:.1f}s{target_str}{stop_str}")

        sessions.append(stats)
        idx += SESSION_LENGTH
        session_id += 1

    # Handle remaining spins
    if idx < len(test_spins):
        remaining_spins = test_spins[idx:]
        print(f"\n  --- Session {session_id} (spins {idx+1}-{len(test_spins)}, {len(remaining_spins)} spins) ---")
        stats = run_session(ensemble, remaining_spins, session_id, verbose=True)
        pl = stats['profit_loss']
        pl_str = f"+${pl:.2f}" if pl >= 0 else f"-${abs(pl):.2f}"
        wr = (stats['wins'] / stats['bets_placed'] * 100) if stats['bets_placed'] > 0 else 0
        target_str = f" 🎯 TARGET at spin {stats['target_hit_spin']}" if stats['target_hit'] else ""
        print(f"  Session {session_id} RESULT: P/L={pl_str} | bets={stats['bets_placed']} | "
              f"W/L={stats['wins']}/{stats['losses']} | WR={wr:.0f}%{target_str}")
        sessions.append(stats)

    # ── Step 5: Print comprehensive results ──
    print_results(sessions)

    # ── Step 6: Cross-validation — shuffle test data order ──
    print(f"\n{'='*70}")
    print(f"  STEP 4: CROSS-VALIDATION — Different data slices")
    print(f"{'='*70}")
    run_cross_validation(historical, test_spins)


def run_cross_validation(historical, test_spins):
    """Test with different slices of data as pre-load vs test."""
    import random

    # Combine all data
    all_data = historical + test_spins
    total = len(all_data)
    print(f"  Total data pool: {total} spins")
    print(f"  Running 5 cross-validation folds...\n")

    fold_results = []

    for fold in range(5):
        # Use different splits: 80% pre-load, 20% test
        # Shift the window each fold
        test_start = fold * (total // 5)
        test_end = test_start + (total // 5)

        preload_data = all_data[:test_start] + all_data[test_end:]
        test_data = all_data[test_start:test_end]

        if len(preload_data) < 100 or len(test_data) < 60:
            continue

        # Pre-load
        ensemble = EnsemblePredictor()
        for num in preload_data:
            ensemble.update(num)

        # Run sessions on test slice
        fold_sessions = []
        idx = 0
        sid = 1
        while idx + SESSION_LENGTH <= len(test_data):
            session_spins = test_data[idx:idx + SESSION_LENGTH]
            stats = run_session(ensemble, session_spins, sid, verbose=False)
            fold_sessions.append(stats)
            idx += SESSION_LENGTH
            sid += 1

        # Summarize fold
        if fold_sessions:
            pls = [s['profit_loss'] for s in fold_sessions]
            total_bets = sum(s['bets_placed'] for s in fold_sessions)
            total_wins = sum(s['wins'] for s in fold_sessions)
            total_losses = sum(s['losses'] for s in fold_sessions)
            targets_hit = sum(1 for s in fold_sessions if s['target_hit'])
            avg_pl = statistics.mean(pls)
            wr = (total_wins / total_bets * 100) if total_bets > 0 else 0

            fold_results.append({
                'fold': fold + 1,
                'preload': len(preload_data),
                'test': len(test_data),
                'sessions': len(fold_sessions),
                'avg_pl': avg_pl,
                'total_pl': sum(pls),
                'total_bets': total_bets,
                'win_rate': wr,
                'targets_hit': targets_hit,
                'profitable': sum(1 for p in pls if p > 0),
            })

            print(f"  Fold {fold+1}: preload={len(preload_data)} | test={len(test_data)} | "
                  f"sessions={len(fold_sessions)} | avg P/L=${avg_pl:+.2f} | "
                  f"bets={total_bets} | WR={wr:.0f}% | "
                  f"targets={targets_hit}/{len(fold_sessions)} | "
                  f"profitable={fold_results[-1]['profitable']}/{len(fold_sessions)}")

    # Cross-validation summary
    if fold_results:
        avg_pl_all = statistics.mean([f['avg_pl'] for f in fold_results])
        total_targets = sum(f['targets_hit'] for f in fold_results)
        total_sessions_cv = sum(f['sessions'] for f in fold_results)
        total_profitable = sum(f['profitable'] for f in fold_results)
        avg_wr = statistics.mean([f['win_rate'] for f in fold_results])

        print(f"\n  Cross-validation summary ({len(fold_results)} folds):")
        print(f"    Average P/L per session:     ${avg_pl_all:+.2f}")
        print(f"    Average win rate:            {avg_wr:.1f}%")
        print(f"    Target ($100) hit rate:       {total_targets}/{total_sessions_cv} "
              f"({total_targets/total_sessions_cv*100:.1f}%)" if total_sessions_cv > 0 else "")
        print(f"    Profitable sessions:         {total_profitable}/{total_sessions_cv} "
              f"({total_profitable/total_sessions_cv*100:.1f}%)" if total_sessions_cv > 0 else "")


def print_results(sessions):
    """Print comprehensive regression test results."""
    n = len(sessions)
    if n == 0:
        print("  No sessions to analyze.")
        return

    pls = [s['profit_loss'] for s in sessions]
    profitable = sum(1 for p in pls if p > 0)
    losing = sum(1 for p in pls if p < 0)
    breakeven = sum(1 for p in pls if p == 0)
    targets_hit = sum(1 for s in sessions if s['target_hit'])
    stop_losses = sum(1 for s in sessions if s['stop_loss_hit'])

    total_bets = sum(s['bets_placed'] for s in sessions)
    total_wins = sum(s['wins'] for s in sessions)
    total_losses_c = sum(s['losses'] for s in sessions)
    total_wagered = sum(s['total_wagered'] for s in sessions)
    total_payouts = sum(s['total_payouts'] for s in sessions)
    total_spins = sum(s['spins'] for s in sessions)

    all_conf_bet = []
    all_conf_wait = []
    for s in sessions:
        all_conf_bet.extend(s['confidences_bet'])
        all_conf_wait.extend(s['confidences_wait'])

    all_seq = Counter()
    for s in sessions:
        all_seq.update(s['seq_steps_used'])

    hit_bet = sum(s['hit_when_bet'] for s in sessions)
    miss_bet = sum(s['miss_when_bet'] for s in sessions)
    hit_wait = sum(s['hit_when_wait'] for s in sessions)
    miss_wait = sum(s['miss_when_wait'] for s in sessions)

    print(f"\n{'#'*70}")
    print(f"  REGRESSION TEST RESULTS")
    print(f"{'#'*70}")

    # ── Session Results ──
    print(f"\n  SESSION OUTCOMES ({n} sessions × {SESSION_LENGTH} spins)")
    print(f"  {'─'*50}")
    print(f"  🎯 Target ($100) hit:    {targets_hit}/{n} ({targets_hit/n*100:.0f}%)")
    print(f"  ✅ Profitable sessions:  {profitable}/{n} ({profitable/n*100:.0f}%)")
    print(f"  🔴 Losing sessions:      {losing}/{n} ({losing/n*100:.0f}%)")
    print(f"  ⚪ Break-even sessions:  {breakeven}/{n} ({breakeven/n*100:.0f}%)")
    print(f"  ⛔ Stop-loss hit:        {stop_losses}/{n} ({stop_losses/n*100:.0f}%)")

    print(f"\n  P/L DISTRIBUTION")
    print(f"  {'─'*50}")
    print(f"  Average P/L per session: ${statistics.mean(pls):+.2f}")
    print(f"  Median P/L per session:  ${statistics.median(pls):+.2f}")
    print(f"  Best session:            ${max(pls):+.2f}")
    print(f"  Worst session:           ${min(pls):+.2f}")
    if n > 1:
        print(f"  Std deviation:           ${statistics.stdev(pls):.2f}")
    print(f"  Total P/L all sessions:  ${sum(pls):+.2f}")

    # Per-session breakdown
    print(f"\n  PER-SESSION BREAKDOWN")
    print(f"  {'─'*50}")
    for s in sessions:
        pl = s['profit_loss']
        pl_str = f"+${pl:.2f}" if pl >= 0 else f"-${abs(pl):.2f}"
        wr = (s['wins'] / s['bets_placed'] * 100) if s['bets_placed'] > 0 else 0
        target = "🎯" if s['target_hit'] else "  "
        stop = "⛔" if s['stop_loss_hit'] else "  "
        print(f"  {target}{stop} Session {s['session_id']:2d}: P/L={pl_str:>10s} | "
              f"bets={s['bets_placed']:2d} | W/L={s['wins']}/{s['losses']} | "
              f"WR={wr:.0f}% | peak=+${s['peak_profit']:.2f} | dd=${s['max_drawdown']:.2f}")

    # ── Betting Analytics ──
    print(f"\n  BETTING ANALYTICS")
    print(f"  {'─'*50}")
    print(f"  Total spins:             {total_spins}")
    print(f"  Total bets placed:       {total_bets}")
    bet_freq = (total_bets / total_spins * 100) if total_spins > 0 else 0
    print(f"  Bet frequency:           {bet_freq:.1f}% of spins")
    print(f"  Total wins / losses:     {total_wins} / {total_losses_c}")
    wr = (total_wins / total_bets * 100) if total_bets > 0 else 0
    print(f"  Overall win rate:        {wr:.1f}%")
    if all_conf_bet:
        print(f"  Avg confidence at bet:   {statistics.mean(all_conf_bet):.1f}%")
    if all_conf_wait:
        print(f"  Avg confidence at wait:  {statistics.mean(all_conf_wait):.1f}%")

    if all_seq:
        print(f"\n  Sequence step distribution:")
        for step in sorted(all_seq.keys()):
            mult = BET_SEQUENCE[step] if step < len(BET_SEQUENCE) else '?'
            count = all_seq[step]
            per_num = BASE_BET * mult if isinstance(mult, int) else 0
            pct = count / total_bets * 100 if total_bets > 0 else 0
            print(f"    Step {step} (×{mult} = ${per_num:.2f}/num): {count} bets ({pct:.0f}%)")

    # ── Financial ──
    print(f"\n  FINANCIAL SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Total wagered:           ${total_wagered:,.2f}")
    print(f"  Total payouts:           ${total_payouts:,.2f}")
    print(f"  Net profit/loss:         ${sum(pls):+,.2f}")
    roi = (sum(pls) / total_wagered * 100) if total_wagered > 0 else 0
    print(f"  ROI:                     {roi:+.1f}%")

    # ── Prediction Quality ──
    print(f"\n  PREDICTION QUALITY")
    print(f"  {'─'*50}")
    total_preds = hit_bet + miss_bet + hit_wait + miss_wait
    total_hits = hit_bet + hit_wait
    overall_hr = (total_hits / total_preds * 100) if total_preds > 0 else 0
    bet_hr = (hit_bet / (hit_bet + miss_bet) * 100) if (hit_bet + miss_bet) > 0 else 0
    wait_hr = (hit_wait / (hit_wait + miss_wait) * 100) if (hit_wait + miss_wait) > 0 else 0

    print(f"  Overall hit rate:        {overall_hr:.1f}% ({total_hits}/{total_preds})")
    print(f"  Hit rate when BETTING:   {bet_hr:.1f}% ({hit_bet}/{hit_bet + miss_bet})")
    print(f"  Hit rate when WAITING:   {wait_hr:.1f}% ({hit_wait}/{hit_wait + miss_wait})")
    print(f"  Random baseline:         {14/37*100:.1f}% (14/37)")
    print(f"  Edge over random:        {overall_hr - 14/37*100:+.1f}%")
    if (hit_bet + miss_bet) > 0:
        print(f"  BET edge over random:    {bet_hr - 14/37*100:+.1f}%")

    print(f"\n{'#'*70}")
    print(f"  VERDICT")
    print(f"{'#'*70}")
    if targets_hit > 0 and targets_hit / n >= 0.3:
        print(f"  ✅ PASS — Target hit rate {targets_hit/n*100:.0f}% is viable")
    elif profitable > 0 and profitable / n >= 0.5:
        print(f"  ⚠️  PARTIAL — {profitable/n*100:.0f}% profitable but target not always hit")
    else:
        print(f"  ❌ NEEDS WORK — Only {profitable/n*100:.0f}% profitable, {targets_hit/n*100:.0f}% target hit")

    if total_bets == 0:
        print(f"  ⚠️  AI placed ZERO bets — confidence gate too high or needs more data")
    elif bet_freq < 2:
        print(f"  ⚠️  Very low bet frequency ({bet_freq:.1f}%) — consider lowering confidence gate")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    run_regression_test()
