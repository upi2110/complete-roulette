#!/usr/bin/env python3
"""
Backtest: THE ONE Strategy on test_data1.txt (509 spins)
=========================================================
Simulates 60-spin sessions and a continuous rolling run using the actual
app.ml.ensemble.EnsemblePredictor and app.money.bankroll_manager.BankrollManager.

Usage:
    cd /Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso
    source venv/bin/activate
    python model_testing/backtest_the_one.py
"""

import sys
import os
import time
import statistics
from collections import Counter, defaultdict

# ── Project imports ──────────────────────────────────────────────────
PROJECT_ROOT = '/Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso'
sys.path.insert(0, PROJECT_ROOT)

try:
    from app.ml.ensemble import EnsemblePredictor
    from app.money.bankroll_manager import BankrollManager
    from config import (
        INITIAL_BANKROLL, CONFIDENCE_BET_THRESHOLD, STOP_LOSS_THRESHOLD,
        TOP_PREDICTIONS_COUNT, BASE_BET, BET_SEQUENCE, PAYOUTS,
        CONDITIONAL_BET_THRESHOLD
    )
    print("[OK] All project modules imported successfully.")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Make sure you run from the project root with the venv activated.")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────
DATA_FILE = os.path.join(PROJECT_ROOT, 'model_testing', 'test_data', 'test_data1.txt')
SESSION_LENGTH = 60
WARMUP_SPINS = 10   # Feed data without betting for first N spins per session


def load_test_data(path):
    """Load spin numbers from text file (one number per line)."""
    numbers = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                num = int(line)
                if 0 <= num <= 36:
                    numbers.append(num)
                else:
                    print(f"  [WARN] Skipping out-of-range number: {num}")
    return numbers


def run_session(spins, session_id, verbose=False):
    """Run a single 60-spin session. Returns a stats dict."""
    ensemble = EnsemblePredictor()
    bankroll = BankrollManager(initial_bankroll=INITIAL_BANKROLL)

    stats = {
        'session_id': session_id,
        'spins_processed': 0,
        'bets_placed': 0,
        'wins': 0,
        'losses': 0,
        'skips': 0,
        'stop_loss_hit': False,
        'total_wagered': 0.0,
        'total_payouts': 0.0,
        'start_bankroll': INITIAL_BANKROLL,
        'end_bankroll': 0.0,
        'profit_loss': 0.0,
        'max_drawdown_pct': 0.0,
        'max_drawdown_dollar': 0.0,
        'peak_bankroll': INITIAL_BANKROLL,
        'seq_steps_used': Counter(),
        'confidences_bet': [],
        'confidences_wait': [],
        'hit_when_bet': 0,
        'miss_when_bet': 0,
        'hit_when_wait': 0,
        'miss_when_wait': 0,
        'wait_reasons': Counter(),
    }

    peak = INITIAL_BANKROLL

    for i, actual_number in enumerate(spins):
        stats['spins_processed'] += 1

        # ── Warmup: feed data, no betting ──
        if i < WARMUP_SPINS:
            ensemble.update(actual_number)
            bankroll.tick_wait()
            continue

        # ── Get prediction ──
        try:
            prediction = ensemble.predict()
        except Exception as e:
            if verbose:
                print(f"    Spin {i}: predict() error: {e}")
            ensemble.update(actual_number)
            bankroll.tick_wait()
            stats['skips'] += 1
            continue

        top_numbers = prediction.get('top_numbers', [])
        confidence = prediction.get('confidence', 0)
        mode = prediction.get('mode', 'WAIT')
        should_bet_signal = prediction.get('should_bet', False)
        num_predictions = len(top_numbers)

        # ── Determine if we should bet ──
        # Confidence gate from ensemble
        ai_says_bet = (mode != 'WAIT' and confidence >= CONFIDENCE_BET_THRESHOLD
                       and should_bet_signal)

        # Bankroll manager gate
        bm_should_bet, bm_reason = bankroll.should_bet(confidence, mode)

        actually_bet = ai_says_bet and bm_should_bet

        # Check if actual number is in predictions (for quality tracking)
        hit = actual_number in top_numbers

        if actually_bet and num_predictions > 0:
            # ── Place bet ──
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
                # WIN: payout is 35 × per_number_bet (the winning number's stake)
                payout = 35 * per_number_bet
                bankroll.process_result(total_bet, won=True, payout_amount=payout)
                stats['wins'] += 1
                stats['hit_when_bet'] += 1
                stats['total_payouts'] += payout + per_number_bet  # 35:1 + stake back
                if verbose:
                    print(f"    Spin {i}: BET step={seq_step} ${total_bet:.2f} on "
                          f"{num_predictions} nums | HIT {actual_number} | "
                          f"payout ${payout:.2f} | bankroll ${bankroll.bankroll:.2f}")
            else:
                # LOSS
                bankroll.process_result(total_bet, won=False)
                stats['losses'] += 1
                stats['miss_when_bet'] += 1
                if verbose:
                    print(f"    Spin {i}: BET step={seq_step} ${total_bet:.2f} on "
                          f"{num_predictions} nums | MISS (actual={actual_number}) | "
                          f"bankroll ${bankroll.bankroll:.2f}")
        else:
            # ── Skip / Wait ──
            stats['skips'] += 1
            if not ai_says_bet:
                reason = f"AI:{mode}/conf={confidence:.0f}"
            else:
                reason = bm_reason
            stats['wait_reasons'][reason] += 1
            stats['confidences_wait'].append(confidence)
            if hit:
                stats['hit_when_wait'] += 1
            else:
                stats['miss_when_wait'] += 1

        # Tick bankroll cooldowns
        if not actually_bet:
            bankroll.tick_wait()

        # ── Feed actual result to ensemble ──
        ensemble.update(actual_number)

        # Track peak and drawdown
        if bankroll.bankroll > peak:
            peak = bankroll.bankroll
        dd_dollar = peak - bankroll.bankroll
        dd_pct = (dd_dollar / peak * 100) if peak > 0 else 0
        if dd_dollar > stats['max_drawdown_dollar']:
            stats['max_drawdown_dollar'] = dd_dollar
            stats['max_drawdown_pct'] = dd_pct

        # ── Stop-loss check ──
        if bankroll.stop_loss_hit:
            stats['stop_loss_hit'] = True
            if verbose:
                print(f"    *** STOP-LOSS HIT at spin {i} | bankroll ${bankroll.bankroll:.2f}")
            break

    stats['end_bankroll'] = round(bankroll.bankroll, 2)
    stats['profit_loss'] = round(bankroll.bankroll - INITIAL_BANKROLL, 2)
    stats['peak_bankroll'] = round(peak, 2)

    return stats


def run_all_sessions(all_spins):
    """Run multiple independent 60-spin sessions across the data."""
    sessions = []
    total_spins = len(all_spins)
    idx = 0
    session_id = 1

    print(f"\n{'='*70}")
    print(f"  INDEPENDENT SESSION MODE  ({SESSION_LENGTH}-spin sessions)")
    print(f"{'='*70}")
    print(f"  Total spins available: {total_spins}")
    print(f"  Sessions possible: {total_spins // SESSION_LENGTH}")
    print(f"  Warmup spins per session: {WARMUP_SPINS}")
    print(f"  Initial bankroll: ${INITIAL_BANKROLL:,.2f}")
    print(f"  Stop-loss: ${STOP_LOSS_THRESHOLD:,.2f}")
    print(f"  Confidence gate: {CONFIDENCE_BET_THRESHOLD}%")
    print(f"  Bet sequence: {list(BET_SEQUENCE)} x ${BASE_BET}")
    print(f"{'='*70}\n")

    while idx + SESSION_LENGTH <= total_spins:
        session_spins = all_spins[idx:idx + SESSION_LENGTH]
        t0 = time.time()
        stats = run_session(session_spins, session_id, verbose=False)
        elapsed = time.time() - t0

        pl_str = f"+${stats['profit_loss']:.2f}" if stats['profit_loss'] >= 0 else f"-${abs(stats['profit_loss']):.2f}"
        hit_rate = (stats['wins'] / stats['bets_placed'] * 100) if stats['bets_placed'] > 0 else 0
        stop_str = " [STOP-LOSS]" if stats['stop_loss_hit'] else ""

        print(f"  Session {session_id:2d} | spins {idx+1:3d}-{idx+SESSION_LENGTH:3d} | "
              f"P/L: {pl_str:>10s} | bets: {stats['bets_placed']:2d} | "
              f"W/L: {stats['wins']}/{stats['losses']} | "
              f"hit: {hit_rate:5.1f}% | "
              f"dd: ${stats['max_drawdown_dollar']:.0f} | "
              f"{elapsed:.1f}s{stop_str}")

        sessions.append(stats)
        idx += SESSION_LENGTH
        session_id += 1

    # Handle remaining spins (partial session)
    if idx < total_spins:
        remaining = total_spins - idx
        session_spins = all_spins[idx:]
        t0 = time.time()
        stats = run_session(session_spins, session_id, verbose=False)
        elapsed = time.time() - t0

        pl_str = f"+${stats['profit_loss']:.2f}" if stats['profit_loss'] >= 0 else f"-${abs(stats['profit_loss']):.2f}"
        hit_rate = (stats['wins'] / stats['bets_placed'] * 100) if stats['bets_placed'] > 0 else 0
        stop_str = " [STOP-LOSS]" if stats['stop_loss_hit'] else ""

        print(f"  Session {session_id:2d} | spins {idx+1:3d}-{total_spins:3d} ({remaining} spins) | "
              f"P/L: {pl_str:>10s} | bets: {stats['bets_placed']:2d} | "
              f"W/L: {stats['wins']}/{stats['losses']} | "
              f"hit: {hit_rate:5.1f}% | "
              f"dd: ${stats['max_drawdown_dollar']:.0f} | "
              f"{elapsed:.1f}s{stop_str}")
        sessions.append(stats)

    return sessions


def run_rolling(all_spins):
    """Run a single continuous session across ALL spins, reporting every 60."""
    print(f"\n{'='*70}")
    print(f"  ROLLING (CONTINUOUS) MODE  — single ensemble across all {len(all_spins)} spins")
    print(f"{'='*70}\n")

    ensemble = EnsemblePredictor()
    bankroll = BankrollManager(initial_bankroll=INITIAL_BANKROLL)

    # Aggregate stats
    total_bets = 0
    total_wins = 0
    total_losses = 0
    total_skips = 0
    total_wagered = 0.0
    total_payouts = 0.0
    confidences_bet = []
    confidences_wait = []
    hit_when_bet = 0
    miss_when_bet = 0
    hit_when_wait = 0
    miss_when_wait = 0
    seq_steps_used = Counter()
    peak = INITIAL_BANKROLL
    max_dd_dollar = 0
    max_dd_pct = 0
    segment_start_bankroll = INITIAL_BANKROLL
    stop_loss_hit = False

    # Segment reporting
    segment_results = []

    t0 = time.time()
    for i, actual_number in enumerate(all_spins):
        # Warmup for the first N spins
        if i < WARMUP_SPINS:
            ensemble.update(actual_number)
            bankroll.tick_wait()
            continue

        # Predict
        try:
            prediction = ensemble.predict()
        except Exception as e:
            ensemble.update(actual_number)
            bankroll.tick_wait()
            total_skips += 1
            continue

        top_numbers = prediction.get('top_numbers', [])
        confidence = prediction.get('confidence', 0)
        mode = prediction.get('mode', 'WAIT')
        should_bet_signal = prediction.get('should_bet', False)
        num_predictions = len(top_numbers)

        ai_says_bet = (mode != 'WAIT' and confidence >= CONFIDENCE_BET_THRESHOLD
                       and should_bet_signal)
        bm_should_bet, bm_reason = bankroll.should_bet(confidence, mode)
        actually_bet = ai_says_bet and bm_should_bet

        hit = actual_number in top_numbers

        if actually_bet and num_predictions > 0:
            seq_step = bankroll.seq_idx
            seq_steps_used[seq_step] += 1

            per_number_bet = bankroll.calculate_bet_amount(
                confidence, 'straight', num_predictions=num_predictions
            )
            total_bet_amt = round(per_number_bet * num_predictions, 2)

            total_bets += 1
            total_wagered += total_bet_amt
            confidences_bet.append(confidence)

            if hit:
                payout = 35 * per_number_bet
                bankroll.process_result(total_bet_amt, won=True, payout_amount=payout)
                total_wins += 1
                hit_when_bet += 1
                total_payouts += payout + per_number_bet
            else:
                bankroll.process_result(total_bet_amt, won=False)
                total_losses += 1
                miss_when_bet += 1
        else:
            total_skips += 1
            confidences_wait.append(confidence)
            if hit:
                hit_when_wait += 1
            else:
                miss_when_wait += 1
            bankroll.tick_wait()

        ensemble.update(actual_number)

        # Track peak/drawdown
        if bankroll.bankroll > peak:
            peak = bankroll.bankroll
        dd_d = peak - bankroll.bankroll
        dd_p = (dd_d / peak * 100) if peak > 0 else 0
        if dd_d > max_dd_dollar:
            max_dd_dollar = dd_d
            max_dd_pct = dd_p

        # Segment report every SESSION_LENGTH spins
        if (i + 1) % SESSION_LENGTH == 0 or i == len(all_spins) - 1:
            seg_pl = bankroll.bankroll - segment_start_bankroll
            seg_num = (i + 1) // SESSION_LENGTH
            pl_str = f"+${seg_pl:.2f}" if seg_pl >= 0 else f"-${abs(seg_pl):.2f}"
            cumul = bankroll.bankroll - INITIAL_BANKROLL
            cumul_str = f"+${cumul:.2f}" if cumul >= 0 else f"-${abs(cumul):.2f}"
            print(f"  Spins {max(1,i+2-SESSION_LENGTH):3d}-{i+1:3d} | "
                  f"segment P/L: {pl_str:>10s} | "
                  f"cumulative: {cumul_str:>10s} | "
                  f"bankroll: ${bankroll.bankroll:,.2f}")
            segment_results.append({
                'segment': seg_num,
                'segment_pl': seg_pl,
                'cumulative_pl': cumul,
                'bankroll': bankroll.bankroll,
            })
            segment_start_bankroll = bankroll.bankroll

        # Stop-loss
        if bankroll.stop_loss_hit:
            stop_loss_hit = True
            print(f"\n  *** STOP-LOSS HIT at spin {i+1} | bankroll ${bankroll.bankroll:.2f}")
            break

    elapsed = time.time() - t0

    rolling_stats = {
        'total_spins': len(all_spins),
        'spins_processed': min(i + 1, len(all_spins)),
        'total_bets': total_bets,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'total_skips': total_skips,
        'total_wagered': total_wagered,
        'total_payouts': total_payouts,
        'start_bankroll': INITIAL_BANKROLL,
        'end_bankroll': bankroll.bankroll,
        'profit_loss': bankroll.bankroll - INITIAL_BANKROLL,
        'peak_bankroll': peak,
        'max_drawdown_dollar': max_dd_dollar,
        'max_drawdown_pct': max_dd_pct,
        'stop_loss_hit': stop_loss_hit,
        'confidences_bet': confidences_bet,
        'confidences_wait': confidences_wait,
        'hit_when_bet': hit_when_bet,
        'miss_when_bet': miss_when_bet,
        'hit_when_wait': hit_when_wait,
        'miss_when_wait': miss_when_wait,
        'seq_steps_used': seq_steps_used,
        'elapsed': elapsed,
        'segments': segment_results,
    }
    return rolling_stats


def print_session_analytics(sessions):
    """Print comprehensive session-based analytics."""
    n = len(sessions)
    if n == 0:
        print("  No sessions to analyze.")
        return

    pls = [s['profit_loss'] for s in sessions]
    profitable = [p for p in pls if p > 0]
    losing = [p for p in pls if p < 0]
    breakeven = [p for p in pls if p == 0]
    bets_per_session = [s['bets_placed'] for s in sessions]
    hit_rates = [(s['wins'] / s['bets_placed'] * 100) if s['bets_placed'] > 0 else 0
                 for s in sessions]
    stop_losses = sum(1 for s in sessions if s['stop_loss_hit'])

    # Aggregate sequence steps
    all_seq_steps = Counter()
    for s in sessions:
        all_seq_steps.update(s['seq_steps_used'])

    # Aggregate confidences
    all_conf_bet = []
    all_conf_wait = []
    for s in sessions:
        all_conf_bet.extend(s['confidences_bet'])
        all_conf_wait.extend(s['confidences_wait'])

    # Financial aggregates
    total_wagered = sum(s['total_wagered'] for s in sessions)
    total_payouts = sum(s['total_payouts'] for s in sessions)
    total_bets_count = sum(s['bets_placed'] for s in sessions)
    total_wins = sum(s['wins'] for s in sessions)
    total_losses_count = sum(s['losses'] for s in sessions)
    total_skips = sum(s['skips'] for s in sessions)
    total_spins = sum(s['spins_processed'] for s in sessions)
    total_pl = sum(pls)

    # Prediction quality aggregates
    hit_bet_all = sum(s['hit_when_bet'] for s in sessions)
    miss_bet_all = sum(s['miss_when_bet'] for s in sessions)
    hit_wait_all = sum(s['hit_when_wait'] for s in sessions)
    miss_wait_all = sum(s['miss_when_wait'] for s in sessions)

    # ── SESSION ANALYTICS ──
    print(f"\n{'='*70}")
    print(f"  SESSION ANALYTICS")
    print(f"{'='*70}")
    print(f"  Total sessions played:       {n}")
    print(f"  Profitable sessions:         {len(profitable)} ({len(profitable)/n*100:.1f}%)")
    print(f"  Losing sessions:             {len(losing)} ({len(losing)/n*100:.1f}%)")
    print(f"  Break-even sessions:         {len(breakeven)} ({len(breakeven)/n*100:.1f}%)")
    print(f"  Average P/L per session:     ${statistics.mean(pls):+.2f}")
    print(f"  Median session result:       ${statistics.median(pls):+.2f}")
    print(f"  Best session profit:         ${max(pls):+.2f}")
    print(f"  Worst session loss:          ${min(pls):+.2f}")
    print(f"  Std dev of session P/L:      ${statistics.stdev(pls):.2f}" if n > 1 else "")
    print(f"  Total P/L across all:        ${total_pl:+.2f}")
    print(f"  Avg bets placed / session:   {statistics.mean(bets_per_session):.1f}")
    print(f"  Avg hit rate / session:      {statistics.mean(hit_rates):.1f}%")
    print(f"  Stop-loss hit count:         {stop_losses} ({stop_losses/n*100:.1f}%)")

    # ── BETTING ANALYTICS ──
    print(f"\n{'='*70}")
    print(f"  BETTING ANALYTICS")
    print(f"{'='*70}")
    print(f"  Total spins processed:       {total_spins}")
    print(f"  Total bets placed:           {total_bets_count}")
    print(f"  Total skips (WAIT):          {total_skips}")
    bet_freq = (total_bets_count / total_spins * 100) if total_spins > 0 else 0
    print(f"  Bet frequency:               {bet_freq:.1f}% of spins")
    print(f"  Total wins:                  {total_wins}")
    print(f"  Total losses:                {total_losses_count}")
    wr = (total_wins / total_bets_count * 100) if total_bets_count > 0 else 0
    print(f"  Overall win rate:            {wr:.1f}%")
    avg_conf_bet = statistics.mean(all_conf_bet) if all_conf_bet else 0
    avg_conf_wait = statistics.mean(all_conf_wait) if all_conf_wait else 0
    print(f"  Avg confidence when betting: {avg_conf_bet:.1f}%")
    print(f"  Avg confidence when waiting: {avg_conf_wait:.1f}%")

    print(f"\n  Sequence step distribution:")
    for step in sorted(all_seq_steps.keys()):
        mult = BET_SEQUENCE[step] if step < len(BET_SEQUENCE) else '?'
        count = all_seq_steps[step]
        pct = count / total_bets_count * 100 if total_bets_count > 0 else 0
        print(f"    Step {step} (x{mult}): {count:3d} bets ({pct:.1f}%)")

    # ── FINANCIAL ANALYTICS ──
    print(f"\n{'='*70}")
    print(f"  FINANCIAL ANALYTICS")
    print(f"{'='*70}")
    print(f"  Total money wagered:         ${total_wagered:,.2f}")
    print(f"  Total payouts received:      ${total_payouts:,.2f}")
    print(f"  Net profit/loss:             ${total_pl:+,.2f}")
    roi = (total_pl / total_wagered * 100) if total_wagered > 0 else 0
    print(f"  ROI (net / wagered):         {roi:+.2f}%")
    avg_bet = total_wagered / total_bets_count if total_bets_count > 0 else 0
    print(f"  Average total bet size:      ${avg_bet:.2f} (across {TOP_PREDICTIONS_COUNT} numbers)")
    avg_per_num = avg_bet / TOP_PREDICTIONS_COUNT if TOP_PREDICTIONS_COUNT > 0 else 0
    print(f"  Average per-number bet:      ${avg_per_num:.2f}")

    # Max drawdown across sessions
    max_dd = max(s['max_drawdown_dollar'] for s in sessions)
    max_dd_pct = max(s['max_drawdown_pct'] for s in sessions)
    print(f"  Worst session drawdown:      ${max_dd:,.2f} ({max_dd_pct:.1f}%)")

    # ── PREDICTION QUALITY ──
    print(f"\n{'='*70}")
    print(f"  PREDICTION QUALITY")
    print(f"{'='*70}")
    total_predictions = hit_bet_all + miss_bet_all + hit_wait_all + miss_wait_all
    total_hits = hit_bet_all + hit_wait_all
    overall_hit = (total_hits / total_predictions * 100) if total_predictions > 0 else 0
    print(f"  Total predictions evaluated: {total_predictions}")
    print(f"  Overall hit rate:            {overall_hit:.1f}% ({total_hits}/{total_predictions})")

    bet_preds = hit_bet_all + miss_bet_all
    hit_rate_bet = (hit_bet_all / bet_preds * 100) if bet_preds > 0 else 0
    print(f"  Hit rate when BETTING:       {hit_rate_bet:.1f}% ({hit_bet_all}/{bet_preds})")

    wait_preds = hit_wait_all + miss_wait_all
    hit_rate_wait = (hit_wait_all / wait_preds * 100) if wait_preds > 0 else 0
    print(f"  Hit rate when WAITING:       {hit_rate_wait:.1f}% ({hit_wait_all}/{wait_preds})")

    # Theoretical random hit rate for reference
    theoretical = TOP_PREDICTIONS_COUNT / 37 * 100
    print(f"  Theoretical random hit rate: {theoretical:.1f}% ({TOP_PREDICTIONS_COUNT}/37)")
    edge = overall_hit - theoretical
    print(f"  Edge over random:            {edge:+.1f}%")

    # Confidence calibration
    print(f"\n  Confidence calibration:")
    if all_conf_bet:
        # Bin by confidence ranges
        bins = [(65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 100)]
        # Collect per-session data into per-bin
        all_bet_conf_hit = list(zip(all_conf_bet,
                                     [True]*hit_bet_all + [False]*miss_bet_all))
        # Actually we need matched pairs. Let me collect them properly from sessions.
        # Since we don't have matched pairs in aggregate, report average confidence vs actual hit rate
        print(f"    Avg reported confidence:   {avg_conf_bet:.1f}%")
        print(f"    Actual hit rate at bet:    {hit_rate_bet:.1f}%")
        if avg_conf_bet > 0:
            calibration = hit_rate_bet / avg_conf_bet * 100
            print(f"    Calibration ratio:         {calibration:.1f}% (100% = perfectly calibrated)")


def print_rolling_analytics(stats):
    """Print rolling/continuous session analytics."""
    print(f"\n{'='*70}")
    print(f"  ROLLING SESSION ANALYTICS")
    print(f"{'='*70}")
    print(f"  Total spins:                 {stats['spins_processed']}")
    print(f"  Total bets placed:           {stats['total_bets']}")
    print(f"  Total skips:                 {stats['total_skips']}")
    bet_freq = (stats['total_bets'] / stats['spins_processed'] * 100) if stats['spins_processed'] > 0 else 0
    print(f"  Bet frequency:               {bet_freq:.1f}%")
    print(f"  Wins / Losses:               {stats['total_wins']} / {stats['total_losses']}")
    wr = (stats['total_wins'] / stats['total_bets'] * 100) if stats['total_bets'] > 0 else 0
    print(f"  Win rate:                    {wr:.1f}%")
    print(f"  ")
    print(f"  Start bankroll:              ${stats['start_bankroll']:,.2f}")
    print(f"  End bankroll:                ${stats['end_bankroll']:,.2f}")
    pl = stats['profit_loss']
    print(f"  Net profit/loss:             ${pl:+,.2f}")
    print(f"  Peak bankroll:               ${stats['peak_bankroll']:,.2f}")
    print(f"  Max drawdown:                ${stats['max_drawdown_dollar']:,.2f} ({stats['max_drawdown_pct']:.1f}%)")
    print(f"  Stop-loss hit:               {'YES' if stats['stop_loss_hit'] else 'No'}")
    print(f"  Total wagered:               ${stats['total_wagered']:,.2f}")
    print(f"  Total payouts:               ${stats['total_payouts']:,.2f}")
    roi = (pl / stats['total_wagered'] * 100) if stats['total_wagered'] > 0 else 0
    print(f"  ROI:                         {roi:+.2f}%")
    print(f"  Elapsed time:                {stats['elapsed']:.1f}s")

    # Prediction quality
    total_preds = (stats['hit_when_bet'] + stats['miss_when_bet'] +
                   stats['hit_when_wait'] + stats['miss_when_wait'])
    total_hits = stats['hit_when_bet'] + stats['hit_when_wait']
    overall_hit = (total_hits / total_preds * 100) if total_preds > 0 else 0
    bet_preds = stats['hit_when_bet'] + stats['miss_when_bet']
    hit_rate_bet = (stats['hit_when_bet'] / bet_preds * 100) if bet_preds > 0 else 0
    wait_preds = stats['hit_when_wait'] + stats['miss_when_wait']
    hit_rate_wait = (stats['hit_when_wait'] / wait_preds * 100) if wait_preds > 0 else 0

    print(f"\n  Prediction quality:")
    print(f"    Overall hit rate:          {overall_hit:.1f}% ({total_hits}/{total_preds})")
    print(f"    Hit rate when BETTING:     {hit_rate_bet:.1f}% ({stats['hit_when_bet']}/{bet_preds})")
    print(f"    Hit rate when WAITING:     {hit_rate_wait:.1f}% ({stats['hit_when_wait']}/{wait_preds})")

    if stats['confidences_bet']:
        avg_cb = statistics.mean(stats['confidences_bet'])
        print(f"    Avg confidence at bet:     {avg_cb:.1f}%")
    if stats['confidences_wait']:
        avg_cw = statistics.mean(stats['confidences_wait'])
        print(f"    Avg confidence at wait:    {avg_cw:.1f}%")

    # Sequence steps
    if stats['seq_steps_used']:
        print(f"\n  Sequence step distribution:")
        for step in sorted(stats['seq_steps_used'].keys()):
            mult = BET_SEQUENCE[step] if step < len(BET_SEQUENCE) else '?'
            count = stats['seq_steps_used'][step]
            pct = count / stats['total_bets'] * 100 if stats['total_bets'] > 0 else 0
            print(f"    Step {step} (x{mult}): {count:3d} bets ({pct:.1f}%)")


# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    print(f"\n{'#'*70}")
    print(f"  BACKTEST: THE ONE STRATEGY")
    print(f"  Data: {DATA_FILE}")
    print(f"{'#'*70}")

    # Load data
    all_spins = load_test_data(DATA_FILE)
    print(f"\n  Loaded {len(all_spins)} spin numbers.")
    print(f"  Number distribution: min={min(all_spins)}, max={max(all_spins)}, "
          f"unique={len(set(all_spins))}")

    # Quick distribution check
    freq = Counter(all_spins)
    most_common = freq.most_common(5)
    least_common = freq.most_common()[-5:]
    print(f"  Most frequent:  {[(n, c) for n, c in most_common]}")
    print(f"  Least frequent: {[(n, c) for n, c in least_common]}")

    # ── Part 1: Independent Sessions ──
    t_start = time.time()
    sessions = run_all_sessions(all_spins)
    t_sessions = time.time() - t_start

    print_session_analytics(sessions)
    print(f"\n  [Sessions completed in {t_sessions:.1f}s]")

    # ── Part 2: Rolling (Continuous) ──
    t_start = time.time()
    rolling = run_rolling(all_spins)
    t_rolling = time.time() - t_start

    print_rolling_analytics(rolling)
    print(f"\n  [Rolling completed in {t_rolling:.1f}s]")

    # ── Final Summary ──
    print(f"\n{'#'*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'#'*70}")
    session_pl = sum(s['profit_loss'] for s in sessions)
    rolling_pl = rolling['profit_loss']
    print(f"  Independent sessions total P/L:  ${session_pl:+,.2f}")
    print(f"  Rolling continuous P/L:          ${rolling_pl:+,.2f}")

    session_bets = sum(s['bets_placed'] for s in sessions)
    session_wins = sum(s['wins'] for s in sessions)
    swr = (session_wins / session_bets * 100) if session_bets > 0 else 0
    rwr = (rolling['total_wins'] / rolling['total_bets'] * 100) if rolling['total_bets'] > 0 else 0
    print(f"  Session win rate:                {swr:.1f}% ({session_wins}/{session_bets})")
    print(f"  Rolling win rate:                {rwr:.1f}% ({rolling['total_wins']}/{rolling['total_bets']})")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
