/**
 * Main Application - SocketIO client, UI state management, event handlers.
 */

const RED_NUMBERS = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);

function getNumberColor(n) {
    if (n === 0) return 'green';
    return RED_NUMBERS.has(n) ? 'red' : 'black';
}

class App {
    constructor() {
        this.socket = io({ transports: ['polling', 'websocket'], upgrade: true, reconnectionDelay: 1000, reconnectionDelayMax: 5000 });
        this.wheel = new RouletteWheel('wheelCanvas');
        this.charts = new ChartsManager();

        this.sessionActive = false;
        this.lastPrediction = null;
        this.currentBet = null;
        this.spinHistory = [];
        this.spinCount = 0;
        this._pendingInitialData = null;
        this._skipTraining = false;
        this.sessionStartTime = null;
        this.timerInterval = null;

        this.bindSocket();
        this.bindUI();
        this.buildNumberGrid();
        this.loadTheme();
    }

    // ─── Socket Events ─────────────────────────────────────
    bindSocket() {
        this.socket.on('connect', () => {
            this.setConnectionStatus('connected', 'Connected');
            this.socket.emit('get_status');

            // Retry pending initial data import if socket reconnected mid-flow
            if (this._pendingImportRetry) {
                const { text, skipTraining } = this._pendingImportRetry;
                console.log('[Reconnect] Retrying pending import_data...');
                this.socket.emit('import_data', { text, skip_training: skipTraining });
            }
        });

        this.socket.on('disconnect', () => {
            this.setConnectionStatus('disconnected', 'Disconnected');
        });

        this.socket.on('connected', (data) => {
            this.sessionActive = data.session_active;
            this.updateSessionButtons();
            // Show model status immediately on page load (from startup auto-load)
            if (data.model_status) {
                this.updateModelStatus(data.model_status);
            }
            if (data.total_spins > 0) {
                document.getElementById('spinCount').textContent = `${data.total_spins} spins (saved)`;
            }
            // Resume session timer on reconnect
            if (data.session_active && data.session_start_time) {
                this.sessionStartTime = data.session_start_time * 1000;
                this.startTimer();
            }
        });

        this.socket.on('session_started', (data) => {
            this.sessionActive = true;
            this.updateSessionButtons();
            this.updateBankroll(data.bankroll);
            this.updateModelStatus(data.model_status);
            this.charts.resetFrequency();
            this.spinHistory = [];
            this.spinCount = 0;
            this.clearHistory();

            // Start session timer
            this.sessionStartTime = data.session_start_time ? data.session_start_time * 1000 : Date.now();
            this.startTimer();

            // Show last 10 numbers from loaded historical data (if any)
            if (data.last_numbers && data.last_numbers.length > 0) {
                this.spinHistory = data.last_numbers.slice();
                this.spinCount = data.historical_spins_loaded || data.last_numbers.length;
                document.getElementById('spinCount').textContent = `${this.spinCount} spins`;
            }
            this.updateLastNumbers(this.spinHistory.slice(-10));

            // Check if we have pending initial data to import
            if (this._pendingInitialData) {
                const text = this._pendingInitialData;
                const skipTraining = this._skipTraining;
                this._pendingInitialData = null;
                this._skipTraining = false;
                // Store retry data in case socket reconnects before import_complete
                this._pendingImportRetry = { text, skipTraining };
                // Import the initial data into the session
                this.socket.emit('import_data', { text: text, skip_training: skipTraining });
                // The import_complete handler will close the modal and show toast
                return;
            }

            if (data.historical_spins_loaded > 0) {
                const clusterMsg = data.clusters_loaded ? ` (${data.clusters_loaded} clusters)` : '';
                this.toast(`Session started. Loaded ${data.historical_spins_loaded} historical spins${clusterMsg}.`, 'success');
            } else {
                this.toast('New session started. Begin entering spin results.', 'success');
            }

            this.setMode('wait', 'WAIT', 'Enter first spin result to get predictions');
            document.getElementById('submitSpin').disabled = false;
            document.getElementById('undoSpin').disabled = false;


            // Get initial prediction
            this.socket.emit('get_prediction');
        });

        this.socket.on('session_ended', (data) => {
            this.sessionActive = false;
            this.stopTimer();
            this.updateSessionButtons();
            document.getElementById('submitSpin').disabled = true;

            this.setMode('', 'SESSION ENDED', `Final P&L: $${data.final_bankroll.profit_loss}`);
            this.toast('Session ended and saved.', 'info');
        });

        this.socket.on('reset_complete', (data) => {
            this.sessionActive = false;
            this.stopTimer();
            this.spinCount = 0;
            this.lastPrediction = null;
            this.currentBet = null;
            this.updateSessionButtons();
            document.getElementById('submitSpin').disabled = true;

            // Clear spin history table
            const tbody = document.querySelector('#spinHistory tbody');
            if (tbody) tbody.innerHTML = '';
            // Clear Last 10
            const last10 = document.getElementById('last10Numbers');
            if (last10) last10.innerHTML = '';
            // Clear prediction display
            const predNums = document.getElementById('predictionNumbers');
            if (predNums) predNums.innerHTML = '<div class="placeholder-text">Start session to see predictions</div>';
            // Clear wheel
            if (this.wheel) this.wheel.reset();
            // Update bankroll
            if (data.bankroll) this.updateBankroll(data.bankroll);
            // Reset mode
            this.setMode('', 'RESET', 'All training data cleared. Start a fresh session.');
            this.toast('✓ All data cleared. AI is completely fresh.', 'success');
        });

        this.socket.on('prediction_result', (data) => {
            this.lastPrediction = data.prediction;
            this.currentBet = data.recommended_bet;
            this.updatePrediction(data);
            if (data.money_advice) this.updateMoneyAdvice(data.money_advice);
        });

        this.socket.on('spin_processed', (data) => {
            const result = data.spin_result;
            this.spinCount++;

            // Update wheel
            this.wheel.highlightNumber(result.number);
            document.getElementById('wheelResult').textContent = result.number;
            document.getElementById('wheelResult').className = `wheel-result ${result.color}`;

            // Update charts
            this.charts.updateFrequency(result.number);
            this.charts.updateBankrollChart(data.bankroll.bankroll_history);

            // Update spin history display
            this.spinHistory.push(result.number);
            this.updateLastNumbers(this.spinHistory.slice(-10));

            // Update bankroll
            this.updateBankroll(data.bankroll);

            // Update model status
            this.updateModelStatus(data.model_status);

            // Add to history table
            this.addHistoryRow(result, data);

            // Update prediction for next spin
            this.lastPrediction = data.next_prediction;
            this.currentBet = data.recommended_bet;
            this.updatePrediction({
                prediction: data.next_prediction,
                should_bet: data.should_bet,
                bet_reason: data.bet_reason,
                recommended_bet: data.recommended_bet,
                bet_amount: data.bet_amount,
                bankroll: data.bankroll
            });

            // Update money management advice
            if (data.money_advice) this.updateMoneyAdvice(data.money_advice);

            // Toast for bet result
            if (result.bet_result) {
                if (result.bet_result.won) {
                    this.toast(`WIN! Net +$${result.bet_result.net.toFixed(2)} (payout $${result.bet_result.payout.toFixed(2)})`, 'success');
                } else {
                    this.toast(`Loss: -$${result.bet_result.amount.toFixed(2)}`, 'error');
                }
            }

            document.getElementById('spinCount').textContent = `${this.spinCount} spins`;

            // Re-enable submit button after processing
            document.getElementById('submitSpin').disabled = false;
            document.getElementById('spinInput').focus();
        });

        this.socket.on('training_update', (data) => {
            if (data.status === 'trained') {
                this.toast(`Neural Net trained! Loss: ${data.final_loss}`, 'success');
            } else {
                this.toast(`Training: ${data.status}`, 'warning');
            }
        });

        this.socket.on('training_complete', (data) => {
            document.getElementById('trainModel').disabled = false;
            this.updateModelStatus(data.model_status);

            // Update last 10 numbers if provided
            if (data.last_numbers && data.last_numbers.length > 0) {
                this.spinHistory = data.last_numbers.slice();
                this.updateLastNumbers(this.spinHistory.slice(-10));
                if (data.summary) {
                    this.spinCount = data.summary.total_spins;
                    document.getElementById('spinCount').textContent = `${this.spinCount} spins`;
                }
            }

            // Build results HTML for the modal
            this.showTrainingResults(data);

            if (data.status === 'trained') {
                this.toast(`AI trained on ${data.summary.total_spins} spins from ${data.summary.total_files} files!`, 'success');
            } else {
                this.toast(data.message || 'Training completed with issues.', 'warning');
            }
        });

        this.socket.on('sessions_list', (data) => {
            this.showSessionsModal(data);
        });

        this.socket.on('status_update', (data) => {
            this.sessionActive = data.session_active;
            this.updateSessionButtons();
            this.updateBankroll(data.bankroll);
            this.updateModelStatus(data.model_status);
        });

        this.socket.on('import_complete', (data) => {
            // Clear retry flag — import succeeded
            this._pendingImportRetry = null;

            // Close import modal
            document.getElementById('importProgress').classList.add('hidden');
            document.getElementById('importSubmit').disabled = false;
            document.getElementById('importWithoutTraining').disabled = false;
            document.getElementById('importModal').classList.add('hidden');

            // Close initial data modal (if it was used for session startup)
            document.getElementById('initialDataModal').classList.add('hidden');
            document.getElementById('initialDataProgress').classList.add('hidden');
            document.getElementById('startWithData').disabled = false;
            document.getElementById('startWithoutTraining').disabled = false;
            const skipBtn = document.getElementById('skipInitialData');
            if (skipBtn) skipBtn.disabled = false;

            this.updateModelStatus(data.model_status);

            // Update last 10 numbers and spin count from imported data
            if (data.last_numbers && data.last_numbers.length > 0) {
                this.spinHistory = data.last_numbers.slice();
                this.spinCount = data.total_spins || data.numbers_imported;
                this.updateLastNumbers(this.spinHistory.slice(-10));
                document.getElementById('spinCount').textContent = `${this.spinCount} spins`;
            }

            const trainMsg = data.training_skipped
                ? `Loaded ${data.numbers_imported} spins (no Neural Net training). Ready!`
                : `Loaded ${data.numbers_imported} spins. Neural Net training in background...`;
            this.toast(trainMsg, 'success');

            // Enable buttons

            document.getElementById('submitSpin').disabled = false;
            document.getElementById('undoSpin').disabled = false;

            const modeMsg = data.training_skipped
                ? `Loaded ${data.total_spins} spins (Neural Net not trained). Enter next spin result.`
                : `AI trained on ${data.total_spins} total spins. Enter next spin result.`;
            this.setMode('wait', 'READY', modeMsg);

            // Get initial prediction
            this.socket.emit('get_prediction');
        });

        this.socket.on('spin_undone', (data) => {
            this.toast(`Undone: removed #${data.removed_number} (${data.removed_color})`, 'warning');

            // Remove last from local history
            this.spinHistory.pop();
            this.spinCount = Math.max(0, this.spinCount - 1);
            this.updateLastNumbers(this.spinHistory.slice(-10));
            document.getElementById('spinCount').textContent = `${this.spinCount} spins`;

            // Remove last row from history table
            const tbody = document.getElementById('historyBody');
            if (tbody.firstChild) tbody.removeChild(tbody.firstChild);

            // Update bankroll and models
            this.updateBankroll(data.bankroll);
            this.updateModelStatus(data.model_status);

            // Update prediction
            if (data.prediction) {
                this.lastPrediction = data.prediction;
                this.updatePrediction({
                    prediction: data.prediction,
                    should_bet: data.should_bet,
                    bet_reason: data.bet_reason,
                    recommended_bet: data.recommended_bet,
                    bankroll: data.bankroll
                });
            }

            // Update wheel
            if (this.spinHistory.length > 0) {
                const last = this.spinHistory[this.spinHistory.length - 1];
                this.wheel.highlightNumber(last);
                document.getElementById('wheelResult').textContent = last;
            } else {
                document.getElementById('wheelResult').textContent = '-';
            }

            // Update money management advice
            if (data.money_advice) this.updateMoneyAdvice(data.money_advice);
        });

        this.socket.on('test_complete', (data) => {
            document.getElementById('testProgress').classList.add('hidden');
            document.getElementById('testSubmit').disabled = false;
            this.showTestResults(data);
        });

        this.socket.on('error', (data) => {
            this.toast(data.message, 'error');
            // Re-enable buttons on error
            document.getElementById('importSubmit').disabled = false;
            document.getElementById('importProgress').classList.add('hidden');
            document.getElementById('testSubmit').disabled = false;
            document.getElementById('testProgress').classList.add('hidden');
            // Re-enable spin submit in case it was disabled during submit
            if (this.sessionActive) {
                document.getElementById('submitSpin').disabled = false;
            }
        });
    }

    // ─── UI Binding ─────────────────────────────────────────
    bindUI() {
        // Session controls — show initial data popup instead of starting immediately
        document.getElementById('startSession').addEventListener('click', () => {
            document.getElementById('initialDataModal').classList.remove('hidden');
            document.getElementById('initialDataTextarea').value = '';
            document.getElementById('startWithData').disabled = true;
            this.updateInitialDataCount();
            document.getElementById('initialDataTextarea').focus();
        });

        document.getElementById('endSession').addEventListener('click', () => {
            this.socket.emit('end_session');
        });

        // Initial Data modal handlers
        document.getElementById('closeInitialDataModal').addEventListener('click', () => {
            document.getElementById('initialDataModal').classList.add('hidden');
        });

        document.getElementById('initialDataModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('initialDataModal')) {
                document.getElementById('initialDataModal').classList.add('hidden');
            }
        });

        document.getElementById('initialDataTextarea').addEventListener('input', () => {
            this.updateInitialDataCount();
        });

        // Start session WITH initial data AND train Neural Net
        document.getElementById('startWithData').addEventListener('click', () => {
            const text = document.getElementById('initialDataTextarea').value.trim();
            const numbers = this._parseNumbers(text);
            if (numbers.length < 5) {
                this.toast('Enter at least 5 numbers to start.', 'warning');
                return;
            }
            document.getElementById('startWithData').disabled = true;
            document.getElementById('startWithoutTraining').disabled = true;
            document.getElementById('skipInitialData').disabled = true;
            document.getElementById('initialDataProgress').classList.remove('hidden');
            document.getElementById('initialDataProgressText').textContent =
                `Training AI on ${numbers.length} spins... This may take a moment.`;

            // Start session first, then import data with training
            this._skipTraining = false;
            this.socket.emit('start_session');
            this._pendingInitialData = text;
        });

        // Start session WITH data but WITHOUT training Neural Net (for fake/test data)
        document.getElementById('startWithoutTraining').addEventListener('click', () => {
            const text = document.getElementById('initialDataTextarea').value.trim();
            const numbers = this._parseNumbers(text);
            if (numbers.length < 5) {
                this.toast('Enter at least 5 numbers to start.', 'warning');
                return;
            }
            document.getElementById('startWithData').disabled = true;
            document.getElementById('startWithoutTraining').disabled = true;
            document.getElementById('skipInitialData').disabled = true;
            document.getElementById('initialDataProgress').classList.remove('hidden');
            document.getElementById('initialDataProgressText').textContent =
                `Loading ${numbers.length} spins (no Neural Net training)...`;

            // Start session, then import WITHOUT training
            this._skipTraining = true;
            this.socket.emit('start_session');
            this._pendingInitialData = text;
        });

        // Skip — start session without initial data
        document.getElementById('skipInitialData').addEventListener('click', () => {
            document.getElementById('initialDataModal').classList.add('hidden');
            this.socket.emit('start_session');
        });

        document.getElementById('trainModel').addEventListener('click', () => {
            // Show training modal with spinner
            document.getElementById('trainingModal').classList.remove('hidden');
            document.getElementById('trainingModalBody').innerHTML = `
                <div style="display:flex;align-items:center;gap:8px;">
                    <div class="spinner"></div>
                    <span style="font-size:0.85rem;">Scanning userdata/ folder and training models...</span>
                </div>`;

            this.toast('Training AI from userdata/ files...', 'info');
            this.socket.emit('train_model');
        });

        // Close training modal
        document.getElementById('closeTrainingModal').addEventListener('click', () => {
            document.getElementById('trainingModal').classList.add('hidden');
        });
        document.getElementById('trainingModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('trainingModal')) {
                document.getElementById('trainingModal').classList.add('hidden');
            }
        });

        // Spin input
        document.getElementById('submitSpin').addEventListener('click', () => this.submitSpin());

        document.getElementById('spinInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.submitSpin();
        });

        // Undo last spin
        document.getElementById('undoSpin').addEventListener('click', () => {
            if (confirm('Undo the last entered number?')) {
                this.socket.emit('undo_spin');
            }
        });

        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => this.toggleTheme());

        // Sessions modal
        document.getElementById('sessionsBtn').addEventListener('click', () => {
            this.socket.emit('get_sessions');
            document.getElementById('sessionsModal').classList.remove('hidden');
        });

        document.getElementById('closeModal').addEventListener('click', () => {
            document.getElementById('sessionsModal').classList.add('hidden');
        });

        document.getElementById('sessionsModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('sessionsModal')) {
                document.getElementById('sessionsModal').classList.add('hidden');
            }
        });

        // Import Data modal
        document.getElementById('importDataBtn').addEventListener('click', () => {
            document.getElementById('importModal').classList.remove('hidden');
        });

        document.getElementById('closeImportModal').addEventListener('click', () => {
            document.getElementById('importModal').classList.add('hidden');
        });

        document.getElementById('importModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('importModal')) {
                document.getElementById('importModal').classList.add('hidden');
            }
        });

        // File upload handling
        document.getElementById('importFile').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            document.getElementById('importFileName').textContent = file.name;
            const reader = new FileReader();
            reader.onload = (ev) => {
                document.getElementById('importTextarea').value = ev.target.result;
                this.updateImportCount();
            };
            reader.readAsText(file);
        });

        // Live count of numbers in textarea
        document.getElementById('importTextarea').addEventListener('input', () => {
            this.updateImportCount();
        });

        // Submit import
        document.getElementById('importSubmit').addEventListener('click', () => {
            const text = document.getElementById('importTextarea').value.trim();
            if (!text) {
                this.toast('Please paste or upload your data first.', 'warning');
                return;
            }
            document.getElementById('importSubmit').disabled = true;
            document.getElementById('importWithoutTraining').disabled = true;
            document.getElementById('importProgress').classList.remove('hidden');
            this.socket.emit('import_data', { text: text, skip_training: false });
        });

        // Import data WITHOUT training Neural Net
        document.getElementById('importWithoutTraining').addEventListener('click', () => {
            const text = document.getElementById('importTextarea').value.trim();
            if (!text) {
                this.toast('Please paste or upload your data first.', 'warning');
                return;
            }
            document.getElementById('importSubmit').disabled = true;
            document.getElementById('importWithoutTraining').disabled = true;
            document.getElementById('importProgress').classList.remove('hidden');
            this.socket.emit('import_data', { text: text, skip_training: true });
        });

        // ─── Test Mode ─────────────────────────────────────────
        document.getElementById('resetAllBtn').addEventListener('click', () => {
            const first = confirm('WARNING: This will DELETE all training data, saved models, and session history.\n\nThe AI will start completely fresh with zero knowledge.\n\nAre you sure?');
            if (!first) return;
            const second = confirm('FINAL CONFIRMATION: All data will be permanently deleted.\n\nClick OK to confirm reset.');
            if (!second) return;
            this.socket.emit('reset_all');
            this.toast('Resetting all data...', 'info');
        });

        document.getElementById('testModeBtn').addEventListener('click', () => {
            document.getElementById('testModal').classList.remove('hidden');
        });

        document.getElementById('closeTestModal').addEventListener('click', () => {
            document.getElementById('testModal').classList.add('hidden');
        });

        document.getElementById('testModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('testModal')) {
                document.getElementById('testModal').classList.add('hidden');
            }
        });

        document.getElementById('testFile').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            document.getElementById('testFileName').textContent = file.name;
            const reader = new FileReader();
            reader.onload = (ev) => {
                document.getElementById('testTextarea').value = ev.target.result;
                this.updateTestCount();
            };
            reader.readAsText(file);
        });

        document.getElementById('testTextarea').addEventListener('input', () => {
            this.updateTestCount();
        });

        document.getElementById('testSubmit').addEventListener('click', () => {
            const text = document.getElementById('testTextarea').value.trim();
            if (!text) {
                this.toast('Please paste or upload test data first.', 'warning');
                return;
            }
            document.getElementById('testSubmit').disabled = true;
            document.getElementById('testProgress').classList.remove('hidden');
            document.getElementById('testResults').classList.add('hidden');
            this.socket.emit('run_test', { text: text });
        });
    }

    updateImportCount() {
        const text = document.getElementById('importTextarea').value;
        const numbers = this._parseNumbers(text);
        document.getElementById('importCount').textContent = `${numbers.length} numbers detected`;
    }

    updateTestCount() {
        const text = document.getElementById('testTextarea').value;
        const numbers = this._parseNumbers(text);
        document.getElementById('testCount').textContent = `${numbers.length} numbers detected`;
    }

    updateInitialDataCount() {
        const text = document.getElementById('initialDataTextarea').value;
        const numbers = this._parseNumbers(text);
        const count = numbers.length;
        const target = 50;

        // Update counter number
        document.getElementById('counterNumber').textContent = count;
        document.getElementById('initialDataCount').textContent = `${count} / ${target} numbers`;

        // Update ring progress (circumference = 2 * PI * 35 ≈ 220)
        const circumference = 220;
        const progress = Math.min(count / target, 1);
        const offset = circumference - (progress * circumference);
        const ring = document.getElementById('counterRingFill');
        ring.style.strokeDashoffset = offset;

        // Color coding based on progress
        ring.classList.remove('warning', 'ready');
        const statusEl = document.getElementById('counterStatus');
        statusEl.classList.remove('needs-more', 'almost', 'ready', 'excellent');

        if (count === 0) {
            statusEl.textContent = 'Enter numbers to begin...';
            statusEl.classList.add('needs-more');
        } else if (count < 20) {
            statusEl.textContent = `Need ${target - count} more numbers`;
            statusEl.classList.add('needs-more');
        } else if (count < target) {
            ring.classList.add('warning');
            statusEl.textContent = `${target - count} more for best results`;
            statusEl.classList.add('almost');
        } else if (count < 100) {
            ring.classList.add('ready');
            statusEl.textContent = 'Ready to start!';
            statusEl.classList.add('ready');
        } else {
            ring.classList.add('ready');
            statusEl.textContent = `Excellent! ${count} spins loaded`;
            statusEl.classList.add('excellent');
        }

        // Enable/disable start buttons (minimum 5 numbers to start)
        document.getElementById('startWithData').disabled = count < 5;
        document.getElementById('startWithoutTraining').disabled = count < 5;
    }

    _parseNumbers(text) {
        return text.replace(/,/g, '\n').split('\n')
            .map(s => s.trim())
            .filter(s => s !== '' && !isNaN(parseInt(s)))
            .filter(s => { const n = parseInt(s); return n >= 0 && n <= 36; });
    }

    showTestResults(report) {
        const div = document.getElementById('testResults');
        div.classList.remove('hidden');

        if (report.error) {
            div.innerHTML = `<p style="color:var(--danger);">${report.error}</p>`;
            return;
        }

        const acc = report.accuracy;
        const exp = report.expected_random;

        const indicator = (actual, expected) => {
            if (actual > expected * 1.1) return '<span style="color:var(--success);">&#9650;</span>';
            if (actual < expected * 0.9) return '<span style="color:var(--danger);">&#9660;</span>';
            return '<span style="color:var(--text-muted);">&#9644;</span>';
        };

        div.innerHTML = `
            <div style="padding:16px;background:var(--bg-tertiary);border-radius:8px;">
                <h3 style="margin-bottom:12px;font-size:1rem;">Test Results: ${report.total_spins} Spins, ${report.total_predictions} Predictions</h3>
                <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
                    <thead>
                        <tr style="border-bottom:2px solid var(--border);">
                            <th style="text-align:left;padding:6px;">Category</th>
                            <th style="text-align:right;padding:6px;">AI Accuracy</th>
                            <th style="text-align:right;padding:6px;">Random Expected</th>
                            <th style="text-align:center;padding:6px;">vs Random</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid var(--border);">
                            <td style="padding:6px;font-weight:600;">Number in Top 5</td>
                            <td style="text-align:right;padding:6px;font-weight:700;">${acc.top5_number}%</td>
                            <td style="text-align:right;padding:6px;color:var(--text-muted);">${exp.top5_number}%</td>
                            <td style="text-align:center;padding:6px;">${indicator(acc.top5_number, exp.top5_number)}</td>
                        </tr>
                        <tr style="border-bottom:1px solid var(--border);">
                            <td style="padding:6px;font-weight:600;">Color (Red/Black)</td>
                            <td style="text-align:right;padding:6px;font-weight:700;">${acc.color}%</td>
                            <td style="text-align:right;padding:6px;color:var(--text-muted);">${exp.color}%</td>
                            <td style="text-align:center;padding:6px;">${indicator(acc.color, exp.color)}</td>
                        </tr>
                        <tr style="border-bottom:1px solid var(--border);">
                            <td style="padding:6px;font-weight:600;">Dozen (1st/2nd/3rd)</td>
                            <td style="text-align:right;padding:6px;font-weight:700;">${acc.dozen}%</td>
                            <td style="text-align:right;padding:6px;color:var(--text-muted);">${exp.dozen}%</td>
                            <td style="text-align:center;padding:6px;">${indicator(acc.dozen, exp.dozen)}</td>
                        </tr>
                        <tr style="border-bottom:1px solid var(--border);">
                            <td style="padding:6px;font-weight:600;">High/Low</td>
                            <td style="text-align:right;padding:6px;font-weight:700;">${acc.high_low}%</td>
                            <td style="text-align:right;padding:6px;color:var(--text-muted);">${exp.high_low}%</td>
                            <td style="text-align:center;padding:6px;">${indicator(acc.high_low, exp.high_low)}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px;font-weight:600;">Odd/Even</td>
                            <td style="text-align:right;padding:6px;font-weight:700;">${acc.odd_even}%</td>
                            <td style="text-align:right;padding:6px;color:var(--text-muted);">${exp.odd_even}%</td>
                            <td style="text-align:center;padding:6px;">${indicator(acc.odd_even, exp.odd_even)}</td>
                        </tr>
                    </tbody>
                </table>
                <p style="font-size:0.72rem;color:var(--text-muted);margin-top:8px;">
                    &#9650; = Better than random &nbsp; &#9660; = Worse than random &nbsp; &#9644; = Similar to random
                </p>
            </div>
            ${report.details && report.details.length > 0 ? `
            <div style="margin-top:12px;max-height:250px;overflow-y:auto;">
                <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
                    <thead>
                        <tr style="border-bottom:2px solid var(--border);position:sticky;top:0;background:var(--bg-secondary);">
                            <th style="padding:4px 6px;text-align:left;">#</th>
                            <th style="padding:4px 6px;text-align:left;">Actual</th>
                            <th style="padding:4px 6px;text-align:left;">Predicted Top 5</th>
                            <th style="padding:4px 6px;text-align:center;">Hit?</th>
                            <th style="padding:4px 6px;text-align:center;">Color</th>
                            <th style="padding:4px 6px;text-align:right;">Conf</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${report.details.map(d => `
                        <tr style="border-bottom:1px solid var(--border);">
                            <td style="padding:4px 6px;">${d.spin}</td>
                            <td style="padding:4px 6px;"><span class="td-number ${d.actual_color}" style="width:24px;height:24px;font-size:0.7rem;">${d.actual}</span></td>
                            <td style="padding:4px 6px;font-family:var(--font-mono);font-size:0.72rem;">${d.predicted_top5.join(', ')}</td>
                            <td style="padding:4px 6px;text-align:center;">${d.hit_top5 ? '<span style="color:var(--success);">&#10003;</span>' : '<span style="color:var(--text-muted);">&#10007;</span>'}</td>
                            <td style="padding:4px 6px;text-align:center;">${d.hit_color ? '<span style="color:var(--success);">&#10003;</span>' : '<span style="color:var(--text-muted);">&#10007;</span>'}</td>
                            <td style="padding:4px 6px;text-align:right;">${d.confidence}%</td>
                        </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            ` : ''}
        `;
    }

    showTrainingResults(data) {
        const body = document.getElementById('trainingModalBody');

        // ─── "Add More Data" section (shown in all cases) ─────────────
        const addMoreHtml = `
            <div style="margin-top:20px;padding-top:16px;border-top:2px solid var(--border);">
                <h3 style="font-size:0.9rem;font-weight:700;margin-bottom:8px;">➕ Add More Data</h3>
                <p style="font-size:0.8rem;color:var(--text-muted);margin-bottom:8px;">
                    Paste numbers below or upload a .txt/.csv file. New data will be added on top of existing training.
                </p>
                <div style="margin-bottom:8px;">
                    <label class="btn btn-secondary btn-sm" style="cursor:pointer;">
                        Choose File
                        <input type="file" id="trainAddFile" accept=".txt,.csv" style="display:none;">
                    </label>
                    <span id="trainAddFileName" style="font-size:0.8rem;color:var(--text-muted);margin-left:8px;">No file selected</span>
                </div>
                <textarea id="trainAddTextarea" rows="6" placeholder="Paste numbers here (one per line or comma-separated, 0-36)..." style="width:100%;padding:10px;border:1px solid var(--border);border-radius:8px;background:var(--bg-tertiary);color:var(--text-primary);font-family:var(--font-mono);font-size:0.85rem;resize:vertical;box-sizing:border-box;"></textarea>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
                    <span id="trainAddCount" style="font-size:0.8rem;color:var(--text-muted);">0 numbers detected</span>
                    <button class="btn btn-primary btn-sm" id="trainAddSubmit">Add & Retrain</button>
                </div>
                <div class="hidden" id="trainAddProgress" style="margin-top:8px;padding:10px;background:var(--bg-tertiary);border-radius:8px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div class="spinner"></div>
                        <span style="font-size:0.85rem;">Adding data and retraining...</span>
                    </div>
                </div>
            </div>`;

        if (data.status === 'no_data' || data.status === 'no_valid_data') {
            body.innerHTML = `
                <div style="text-align:center;padding:24px;">
                    <div style="font-size:2rem;margin-bottom:12px;">📂</div>
                    <p style="font-size:1rem;font-weight:600;margin-bottom:8px;">${data.message}</p>
                    <p style="font-size:0.85rem;color:var(--text-muted);">
                        Place your .txt or .csv files in the <code style="background:var(--bg-tertiary);padding:2px 6px;border-radius:4px;">userdata/</code> folder.<br>
                        Each file should contain one roulette number (0-36) per line.
                    </p>
                </div>
                ${addMoreHtml}`;
            this._bindTrainAddHandlers();
            return;
        }

        const s = data.summary || {};
        const files = data.files || [];

        // Files table
        let filesHtml = files.map(f => {
            const statusIcon = f.error ? '<span style="color:var(--danger);">&#10007;</span>' : '<span style="color:var(--success);">&#10003;</span>';
            const detail = f.error ? `<span style="color:var(--danger);font-size:0.75rem;">${f.error}</span>` : `${f.numbers} numbers${f.skipped > 0 ? `, ${f.skipped} skipped` : ''}`;
            return `<tr style="border-bottom:1px solid var(--border);">
                <td style="padding:6px 8px;">${statusIcon}</td>
                <td style="padding:6px 8px;font-family:var(--font-mono);font-size:0.82rem;">${f.filename}</td>
                <td style="padding:6px 8px;text-align:right;">${detail}</td>
            </tr>`;
        }).join('');

        // LSTM result
        const lstmStatus = s.lstm_result || {};
        let lstmHtml = '';
        if (lstmStatus.status === 'trained') {
            lstmHtml = `<span style="color:var(--success);font-weight:600;">Trained</span> (loss: ${lstmStatus.final_loss}, ${lstmStatus.epochs} epochs, ${lstmStatus.dataset_size} sequences)`;
        } else if (lstmStatus.status === 'insufficient_data') {
            lstmHtml = `<span style="color:var(--warning);">Not enough data</span> (need 40+ spins)`;
        } else {
            lstmHtml = `<span style="color:var(--text-muted);">${lstmStatus.status || 'N/A'}</span>`;
        }

        // Hot numbers
        const hotHtml = (s.hot_numbers || []).map(h =>
            `<span class="number-pill ${getNumberColor(h.number)}" style="font-size:0.75rem;">${h.number}</span><small style="color:var(--text-muted);margin-right:6px;">${h.ratio}x</small>`
        ).join('') || '<span style="color:var(--text-muted);">None detected</span>';

        // Cold numbers
        const coldHtml = (s.cold_numbers || []).map(c =>
            `<span class="number-pill ${getNumberColor(c.number)}" style="font-size:0.75rem;opacity:0.6;">${c.number}</span><small style="color:var(--text-muted);margin-right:6px;">${c.ratio}x</small>`
        ).join('') || '<span style="color:var(--text-muted);">None detected</span>';

        // Color dist
        const cd = s.color_distribution || {};

        body.innerHTML = `
            <div style="margin-bottom:16px;padding:16px;background:var(--bg-tertiary);border-radius:8px;text-align:center;">
                <div style="font-size:1.5rem;font-weight:800;color:var(--success);margin-bottom:4px;">Training Complete</div>
                <div style="font-size:0.9rem;color:var(--text-secondary);">${s.total_spins || 0} spins from ${s.total_files || 0} files in ${s.training_time || 0}s</div>
            </div>

            <!-- Files Used -->
            <div style="margin-bottom:16px;">
                <h3 style="font-size:0.9rem;font-weight:700;margin-bottom:8px;">📁 Files Processed</h3>
                <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
                    <thead>
                        <tr style="border-bottom:2px solid var(--border);">
                            <th style="padding:4px 8px;text-align:left;width:30px;"></th>
                            <th style="padding:4px 8px;text-align:left;">Filename</th>
                            <th style="padding:4px 8px;text-align:right;">Details</th>
                        </tr>
                    </thead>
                    <tbody>${filesHtml}</tbody>
                </table>
            </div>

            <!-- Model Training -->
            <div style="margin-bottom:16px;">
                <h3 style="font-size:0.9rem;font-weight:700;margin-bottom:8px;">🧠 Neural Network (GRU)</h3>
                <div style="padding:10px;background:var(--bg-tertiary);border-radius:6px;font-size:0.85rem;">
                    ${lstmHtml}
                </div>
            </div>

            <!-- Statistics -->
            <div style="margin-bottom:16px;">
                <h3 style="font-size:0.9rem;font-weight:700;margin-bottom:8px;">📊 Data Summary</h3>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
                    <div style="padding:10px;background:var(--bg-tertiary);border-radius:6px;text-align:center;">
                        <div style="font-size:0.7rem;color:var(--text-muted);">TOTAL SPINS</div>
                        <div style="font-size:1.3rem;font-weight:800;">${s.total_spins || 0}</div>
                    </div>
                    <div style="padding:10px;background:var(--bg-tertiary);border-radius:6px;text-align:center;">
                        <div style="font-size:0.7rem;color:var(--text-muted);">UNIQUE NUMBERS</div>
                        <div style="font-size:1.3rem;font-weight:800;">${s.unique_numbers || 0}/37</div>
                    </div>
                    <div style="padding:10px;background:var(--bg-tertiary);border-radius:6px;text-align:center;">
                        <div style="font-size:0.7rem;color:var(--text-muted);">TRAINING TIME</div>
                        <div style="font-size:1.3rem;font-weight:800;">${s.training_time || 0}s</div>
                    </div>
                </div>

                <!-- Color Distribution -->
                <div style="display:flex;gap:16px;margin-bottom:12px;font-size:0.82rem;">
                    <span><span style="color:#e74c3c;font-weight:700;">Red:</span> ${cd.red || 0} (${cd.red_pct || 0}%)</span>
                    <span><span style="color:#2c3e50;font-weight:700;">Black:</span> ${cd.black || 0} (${cd.black_pct || 0}%)</span>
                    <span><span style="color:#27ae60;font-weight:700;">Green:</span> ${cd.green || 0} (${cd.green_pct || 0}%)</span>
                </div>

                <!-- Hot Numbers -->
                <div style="margin-bottom:8px;">
                    <span style="font-size:0.82rem;font-weight:600;">🔥 Hot Numbers: </span>${hotHtml}
                </div>

                <!-- Cold Numbers -->
                <div>
                    <span style="font-size:0.82rem;font-weight:600;">🧊 Cold Numbers: </span>${coldHtml}
                </div>
            </div>

            <div style="padding:10px;background:var(--accent-light);border-radius:6px;font-size:0.82rem;color:var(--accent);">
                All 4 models (Frequency, Markov, Pattern, Neural Net) have been trained. Start a session to use predictions!
            </div>

            ${addMoreHtml}
        `;

        this._bindTrainAddHandlers();
    }

    _bindTrainAddHandlers() {
        const textarea = document.getElementById('trainAddTextarea');
        const countEl = document.getElementById('trainAddCount');
        const fileInput = document.getElementById('trainAddFile');
        const fileNameEl = document.getElementById('trainAddFileName');
        const submitBtn = document.getElementById('trainAddSubmit');
        const progressEl = document.getElementById('trainAddProgress');

        if (!textarea || !submitBtn) return;

        // Live counter as user types/pastes
        textarea.addEventListener('input', () => {
            const nums = this._parseNumbers(textarea.value);
            countEl.textContent = `${nums.length} numbers detected`;
        });

        // File upload fills the textarea
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            fileNameEl.textContent = file.name;
            const reader = new FileReader();
            reader.onload = (ev) => {
                textarea.value = ev.target.result;
                textarea.dispatchEvent(new Event('input'));
            };
            reader.readAsText(file);
        });

        // Add & Retrain button
        submitBtn.addEventListener('click', () => {
            const text = textarea.value.trim();
            const numbers = this._parseNumbers(text);
            if (numbers.length < 1) {
                this.toast('Enter at least 1 number to add.', 'warning');
                return;
            }

            // Show progress, disable button
            submitBtn.disabled = true;
            progressEl.classList.remove('hidden');

            // Send as import_data (incremental, adds on top of existing)
            this.socket.emit('import_data', { text: text, skip_training: false });

            // When import completes, update the modal
            const onImportDone = (importData) => {
                this.socket.off('import_complete', onImportDone);
                progressEl.classList.add('hidden');
                submitBtn.disabled = false;
                textarea.value = '';
                countEl.textContent = '0 numbers detected';
                fileNameEl.textContent = 'No file selected';
                fileInput.value = '';

                this.toast(`Added ${importData.numbers_imported} spins. Total: ${importData.total_spins} spins.`, 'success');

                // Update model status in the main UI
                this.updateModelStatus(importData.model_status);
                if (importData.last_numbers && importData.last_numbers.length > 0) {
                    this.spinHistory = importData.last_numbers.slice();
                    this.spinCount = importData.total_spins || importData.numbers_imported;
                    this.updateLastNumbers(this.spinHistory.slice(-10));
                    document.getElementById('spinCount').textContent = `${this.spinCount} spins`;
                }
            };
            this.socket.on('import_complete', onImportDone);
        });
    }

    buildNumberGrid() {
        const grid = document.getElementById('numberGrid');
        // Numbers in display order
        const numbers = [
            0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34,
            35, 36
        ];

        numbers.forEach(num => {
            const btn = document.createElement('button');
            btn.className = `num-btn ${getNumberColor(num)}`;
            btn.textContent = num;
            btn.addEventListener('click', () => {
                document.getElementById('spinInput').value = num;
            });
            grid.appendChild(btn);
        });
    }

    // ─── Submit Spin ────────────────────────────────────────
    submitSpin() {
        const input = document.getElementById('spinInput');
        const submitBtn = document.getElementById('submitSpin');
        const num = parseInt(input.value);

        if (isNaN(num) || num < 0 || num > 36) {
            this.toast('Enter a number between 0 and 36', 'warning');
            return;
        }

        // Debounce: disable button immediately to prevent duplicate submissions
        submitBtn.disabled = true;

        // Determine if we're placing a bet based on current recommendation
        const betData = {
            number: num,
            bet_placed: this.currentBet || null,
            bet_amount: this.currentBet ? this.currentBet.amount : 0
        };

        this.socket.emit('submit_spin', betData);
        input.value = '';
    }

    // ─── Update Functions ───────────────────────────────────
    updatePrediction(data) {
        const pred = data.prediction;
        const confidence = pred.confidence;

        // Update confidence badge
        const badge = document.getElementById('confidenceBadge');
        badge.textContent = `${confidence}%`;
        badge.className = 'confidence-badge';
        if (confidence >= 80) badge.classList.add('very-high');
        else if (confidence >= 65) badge.classList.add('high');
        else if (confidence >= 40) badge.classList.add('medium');
        else badge.classList.add('low');

        // Update confidence bar
        document.getElementById('confidenceFill').style.width = `${confidence}%`;
        document.getElementById('confidenceValue').textContent = `${confidence}%`;

        // Update BET/WAIT decision inline
        const decisionDiv = document.getElementById('predictionDecision');
        const decisionLabel = document.getElementById('decisionLabel');
        const decisionDetail = document.getElementById('decisionDetail');
        const betInfoDiv = document.getElementById('predictionBetInfo');

        if (data.should_bet) {
            const modeClass = pred.mode === 'BET_HIGH' ? 'bet-high' : 'bet';
            decisionDiv.className = `prediction-decision ${modeClass}`;
            decisionLabel.textContent = pred.mode === 'BET_HIGH' ? 'BET (HIGH)' : 'BET';
            const betAmt = data.recommended_bet ? data.recommended_bet.amount : 0;
            const totalBet = data.recommended_bet ? data.recommended_bet.total_bet : 0;
            decisionDetail.textContent = `$${betAmt.toFixed(2)}/number × ${pred.top_numbers.length} = $${totalBet.toFixed(2)} total`;
            betInfoDiv.classList.remove('hidden');
            document.getElementById('betPerNumber').textContent = `$${betAmt.toFixed(2)}`;
            document.getElementById('betTotal').textContent = `$${totalBet.toFixed(2)}`;
            document.getElementById('betCoverage').textContent = `${pred.top_numbers.length}/37 (${(pred.top_numbers.length/37*100).toFixed(1)}%)`;
            document.getElementById('betWinPayout').textContent = '35:1';
        } else {
            decisionDiv.className = 'prediction-decision wait';
            decisionLabel.textContent = 'WAIT';
            decisionDetail.textContent = data.bet_reason || `Confidence ${confidence}% below threshold`;
            betInfoDiv.classList.add('hidden');
        }

        // Top predicted numbers grouped by anchor with spread info
        const anchorDetails = pred.anchor_details || [];
        const anchorsSet = new Set((pred.anchors || []).map(n => parseInt(n)));

        // Build a lookup: number → anchor detail for spread label
        const numToAnchor = {};
        for (const ad of anchorDetails) {
            for (const n of (ad.numbers || [])) {
                numToAnchor[parseInt(n)] = ad;
            }
        }

        const topDiv = document.getElementById('topNumbers');
        if (pred.top_numbers && pred.top_numbers.length > 0) {
            // Group numbers by anchor for display
            let html = '';
            const rendered = new Set();

            // Render anchor groups first
            for (const ad of anchorDetails) {
                const anchorNum = parseInt(ad.number);
                const spread = ad.spread || 1;
                const groupNums = (ad.numbers || []).map(n => parseInt(n));

                html += `<div class="anchor-group">`;
                html += `<div class="anchor-group-label">${anchorNum} <span class="spread-badge">\u00b1${spread}</span></div>`;
                html += `<div class="anchor-group-numbers">`;
                for (const num of groupNums) {
                    const isAnchor = num === anchorNum;
                    const roleClass = isAnchor ? 'anchor' : 'neighbour';
                    const idx = pred.top_numbers.indexOf(num);
                    const prob = idx >= 0 ? (pred.top_probabilities[idx] * 100).toFixed(1) : '?';
                    html += `
                        <div class="pred-number">
                            <div class="pred-num-circle ${getNumberColor(num)} ${roleClass}">${num}</div>
                            <span class="pred-prob">${prob}%</span>
                        </div>`;
                    rendered.add(num);
                }
                html += `</div></div>`;
            }

            // Render any remaining numbers not in anchor groups
            const remaining = pred.top_numbers.filter(n => !rendered.has(parseInt(n)));
            if (remaining.length > 0) {
                html += `<div class="anchor-group">`;
                html += `<div class="anchor-group-label">Extra</div>`;
                html += `<div class="anchor-group-numbers">`;
                for (const num of remaining) {
                    const idx = pred.top_numbers.indexOf(num);
                    const prob = idx >= 0 ? (pred.top_probabilities[idx] * 100).toFixed(1) : '?';
                    html += `
                        <div class="pred-number">
                            <div class="pred-num-circle ${getNumberColor(num)} neighbour">${num}</div>
                            <span class="pred-prob">${prob}%</span>
                        </div>`;
                }
                html += `</div></div>`;
            }

            topDiv.innerHTML = html;

            // Highlight predicted numbers + anchors on the wheel
            this.wheel.setPredictedNumbers(
                pred.top_numbers.map(n => parseInt(n)),
                (pred.anchors || []).map(n => parseInt(n))
            );
        }

        // Confidence breakdown (6 factors)
        if (pred.confidence_breakdown && pred.confidence_breakdown.factors) {
            const bd = pred.confidence_breakdown.factors;
            let breakdownHtml = `
                <div class="cb-item"><span class="cb-label">Model Agreement</span><span class="cb-score">${bd.model_agreement.score.toFixed(0)}</span></div>
                <div class="cb-item"><span class="cb-label">Hist. Accuracy</span><span class="cb-score">${bd.historical_accuracy.score.toFixed(0)}</span></div>
                <div class="cb-item"><span class="cb-label">Pattern Strength</span><span class="cb-score">${bd.pattern_strength.score.toFixed(0)}</span></div>
                <div class="cb-item"><span class="cb-label">Sample Size</span><span class="cb-score">${bd.sample_size.score.toFixed(0)}</span></div>
                <div class="cb-item"><span class="cb-label">Momentum</span><span class="cb-score">${bd.streak_momentum.score.toFixed(0)}</span></div>
            `;
            if (bd.recent_hit_rate) {
                breakdownHtml += `<div class="cb-item"><span class="cb-label">Hit Rate</span><span class="cb-score">${bd.recent_hit_rate.score.toFixed(0)}</span></div>`;
            }
            document.getElementById('confidenceBreakdown').innerHTML = breakdownHtml;
        }

        // Mode banner — show bet amount when betting
        // Exploration mode indicator
        let explorationSuffix = '';
        if (pred.exploration_active) {
            explorationSuffix = ` | 🔄 Exploring (${pred.consecutive_misses || '?'} misses)`;
        }

        const betSizeEl = document.getElementById('spinBetSize');
        if (data.should_bet) {
            const modeClass = pred.mode === 'BET_HIGH' ? 'bet-high' : 'bet';
            const modeLabel = pred.mode === 'BET_HIGH' ? 'BET (HIGH CONFIDENCE)' : 'BET';
            const totalBet = data.recommended_bet ? data.recommended_bet.total_bet : 0;
            this.setMode(modeClass, modeLabel, `Confidence: ${confidence}%${explorationSuffix}`, totalBet);
            // Show bet size in sticky input bar
            if (betSizeEl) {
                betSizeEl.textContent = `Bet: $${data.bet_amount}/num × ${pred.top_numbers.length} = $${totalBet}`;
                betSizeEl.style.color = 'var(--success)';
            }
        } else {
            if (data.bet_reason && data.bet_reason.includes('TARGET')) {
                this.setMode('danger', 'TARGET REACHED', data.bet_reason);
            } else if (data.bet_reason && data.bet_reason.includes('STOP_LOSS')) {
                this.setMode('danger', 'STOP LOSS', data.bet_reason);
            } else if (data.bet_reason && data.bet_reason.includes('LOSS_STREAK')) {
                this.setMode('wait', `CAUTION`, data.bet_reason + explorationSuffix);
            } else {
                this.setMode('wait', 'WAIT', (data.bet_reason || `Confidence ${confidence}% below threshold`) + explorationSuffix);
            }
            // Show WAIT in sticky input bar
            if (betSizeEl) {
                betSizeEl.textContent = 'WAIT';
                betSizeEl.style.color = 'var(--warning)';
            }
        }

        // Group probabilities
        if (pred.group_probabilities) {
            const gp = pred.group_probabilities;
            this.updateProbBar('probRed', 'probRedVal', gp.red);
            this.updateProbBar('probBlack', 'probBlackVal', gp.black);
            this.updateProbBar('probD1', 'probD1Val', gp.dozen_1st);
            this.updateProbBar('probD2', 'probD2Val', gp.dozen_2nd);
            this.updateProbBar('probD3', 'probD3Val', gp.dozen_3rd);
            this.updateProbBar('probHigh', 'probHighVal', gp.high);
            this.updateProbBar('probLow', 'probLowVal', gp.low);
            this.updateProbBar('probOdd', 'probOddVal', gp.odd);
            this.updateProbBar('probEven', 'probEvenVal', gp.even);
        }
    }

    updateProbBar(fillId, valId, prob) {
        const pct = (prob * 100).toFixed(1);
        document.getElementById(fillId).style.width = `${pct}%`;
        document.getElementById(valId).textContent = `${pct}%`;
    }

    // ─── Money Management Advisor ────────────────────────────
    updateMoneyAdvice(advice) {
        if (!advice) return;

        // Update strategy badge
        const badge = document.getElementById('strategyBadge');
        badge.textContent = advice.strategy;
        badge.className = 'strategy-badge';
        const stratLower = (advice.strategy || '').toLowerCase();
        if (stratLower === 'aggressive') badge.classList.add('aggressive');
        else if (stratLower === 'standard') badge.classList.add('standard');
        else if (stratLower === 'recovery' || stratLower === 'loss recovery') badge.classList.add('recovery');
        else if (stratLower === 'defensive' || stratLower === 'capital preservation') badge.classList.add('defensive');
        else badge.classList.add('selective');

        // Update action banner
        const actionDiv = document.getElementById('advisorAction');
        const actionIcon = document.getElementById('advisorActionIcon');
        const actionLabel = document.getElementById('advisorActionLabel');
        const actionReason = document.getElementById('advisorActionReason');

        actionLabel.textContent = advice.action_label;
        actionReason.textContent = advice.reason;

        // Reset classes
        actionDiv.className = 'advisor-action';
        if (advice.action === 'BET') {
            if (advice.action_label.includes('HIGH')) {
                actionDiv.classList.add('bet-high');
                actionIcon.innerHTML = '&#9654;&#9654;';
            } else {
                actionDiv.classList.add('bet');
                actionIcon.innerHTML = '&#9654;';
            }
        } else if (advice.action === 'STOP') {
            actionDiv.classList.add('stop');
            actionIcon.innerHTML = '&#9888;';
        } else {
            actionDiv.classList.add('wait');
            actionIcon.innerHTML = '&#9208;';
        }

        // Update bet sizing grid — straight bets only (dynamic prediction count)
        const sizingDiv = document.getElementById('advisorSizing');
        const sizingGrid = document.getElementById('sizingGrid');
        if (advice.action === 'BET' && advice.bet_sizing && advice.bet_sizing.straight) {
            sizingDiv.classList.remove('hidden');
            const perNum = advice.bet_sizing.straight;
            // Use actual prediction count from last prediction (dynamic)
            const numPreds = (this.lastPrediction && this.lastPrediction.top_numbers) ? this.lastPrediction.top_numbers.length : 12;
            const totalBet = perNum * numPreds;
            sizingGrid.innerHTML = `
                <div class="sizing-item recommended">
                    <span class="sizing-type">Per Number</span>
                    <span class="sizing-amount">$${perNum.toFixed(2)}</span>
                    <span class="sizing-payout">35:1</span>
                </div>
                <div class="sizing-item">
                    <span class="sizing-type">Total (${numPreds} nums)</span>
                    <span class="sizing-amount">$${totalBet.toFixed(2)}</span>
                    <span class="sizing-payout">Straight</span>
                </div>`;
        } else {
            sizingDiv.classList.add('hidden');
        }

        // Update risk dashboard
        const riskLevel = document.getElementById('riskLevel');
        riskLevel.textContent = (advice.risk_level || '--').toUpperCase();
        riskLevel.className = 'risk-value';
        if (advice.risk_level === 'low') riskLevel.classList.add('risk-low');
        else if (advice.risk_level === 'medium') riskLevel.classList.add('risk-medium');
        else if (advice.risk_level === 'high') riskLevel.classList.add('risk-high');
        else if (advice.risk_level === 'critical') riskLevel.classList.add('risk-critical');

        document.getElementById('riskDrawdown').textContent = `${advice.drawdown || 0}%`;
        const drawdownEl = document.getElementById('riskDrawdown');
        drawdownEl.className = 'risk-value';
        if ((advice.drawdown || 0) > 15) drawdownEl.classList.add('risk-high');
        else if ((advice.drawdown || 0) > 5) drawdownEl.classList.add('risk-medium');
        else drawdownEl.classList.add('risk-low');

        const momentum = advice.momentum || 0;
        const momEl = document.getElementById('riskMomentum');
        momEl.textContent = momentum > 0 ? `+${momentum}` : momentum.toString();
        momEl.className = 'risk-value';
        if (momentum > 30) momEl.classList.add('risk-low');
        else if (momentum < -30) momEl.classList.add('risk-high');
        else momEl.classList.add('risk-medium');

        document.getElementById('riskToTarget').textContent = `$${(advice.distance_to_target || 0).toFixed(0)}`;
        document.getElementById('riskToStop').textContent = `$${(advice.distance_to_stop_loss || 0).toFixed(0)}`;
        document.getElementById('riskStrategy').textContent = advice.strategy || '--';

        // Update warnings
        const warningsDiv = document.getElementById('advisorWarnings');
        const warningsList = document.getElementById('warningsList');
        if (advice.warnings && advice.warnings.length > 0) {
            warningsDiv.classList.remove('hidden');
            warningsList.innerHTML = advice.warnings.map(w =>
                `<div class="warning-item">${w}</div>`
            ).join('');
        } else {
            warningsDiv.classList.add('hidden');
        }

        // Update tips
        const tipsDiv = document.getElementById('advisorTips');
        const tipsList = document.getElementById('tipsList');
        if (advice.tips && advice.tips.length > 0) {
            tipsDiv.classList.remove('hidden');
            tipsList.innerHTML = advice.tips.map(t =>
                `<div class="tip-item">${t}</div>`
            ).join('');
        } else {
            tipsDiv.classList.add('hidden');
        }
    }

    updateBankroll(status) {
        if (!status) return;

        document.getElementById('bankrollAmount').textContent = `$${status.bankroll.toLocaleString('en-US', {minimumFractionDigits: 2})}`;

        const pnlEl = document.getElementById('bankrollPnl');
        const pnl = status.profit_loss;
        pnlEl.textContent = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`;
        pnlEl.className = `bankroll-pnl ${pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral'}`;

        document.getElementById('targetValue').textContent = `$${status.session_target}`;
        document.getElementById('targetProgress').textContent = `${status.target_progress}%`;
        document.getElementById('winRate').textContent = `${status.win_rate}%`;
        document.getElementById('totalBets').textContent = status.total_bets;
        document.getElementById('winLoss').textContent = `${status.total_wins} / ${status.total_losses}`;
        document.getElementById('consecLosses').textContent = status.consecutive_losses;

        // Color-code consecutive losses
        const clEl = document.getElementById('consecLosses');
        if (status.consecutive_losses >= 2) clEl.style.color = 'var(--danger)';
        else if (status.consecutive_losses >= 1) clEl.style.color = 'var(--warning)';
        else clEl.style.color = '';

        // Target progress bar
        document.getElementById('targetFill').style.width = `${Math.max(0, status.target_progress)}%`;

        // Bankroll chart
        if (status.bankroll_history) {
            this.charts.updateBankrollChart(status.bankroll_history);
        }
    }

    updateModelStatus(models) {
        if (!models) return;

        const setStatus = (id, status, text) => {
            const el = document.getElementById(id);
            el.textContent = text;
            el.className = `model-status ${status}`;
        };

        if (models.frequency.status === 'idle') {
            setStatus('modelFreq', 'idle', 'Idle — No data');
        } else {
            setStatus('modelFreq', 'active', `Active (Bias: ${models.frequency.bias_score}) — ${models.frequency.spins} spins`);
        }

        if (models.markov.status === 'idle') {
            setStatus('modelMarkov', 'idle', 'Idle — No data');
        } else {
            setStatus('modelMarkov', 'active', `Active (Str: ${models.markov.strength}) — ${models.markov.spins} spins`);
        }

        if (models.patterns.status === 'idle') {
            setStatus('modelPattern', 'idle', 'Idle — No data');
        } else {
            setStatus('modelPattern', 'active', `Active (Str: ${models.patterns.strength}) — ${models.patterns.spins} spins`);
        }

        if (models.lstm.trained) {
            setStatus('modelLstm', 'active', `Trained (${models.lstm.device}) — ${models.lstm.spins} spins`);
        } else if (models.lstm.spins > 0) {
            setStatus('modelLstm', 'collecting', `Collecting (${models.lstm.spins}/40)`);
        } else {
            setStatus('modelLstm', 'idle', 'Idle — No data');
        }
    }

    updateLastNumbers(numbers) {
        const pills = document.getElementById('numberPills');
        // Reverse so most recent appears first (leftmost)
        const reversed = numbers.slice().reverse();
        pills.innerHTML = reversed.map(n => `
            <span class="number-pill ${getNumberColor(n)}">${n}</span>
        `).join('');
    }

    setMode(modeClass, label, reason, totalBet) {
        const banner = document.getElementById('modeBanner');
        banner.className = `mode-banner ${modeClass}`;
        document.getElementById('modeLabel').textContent = label;
        document.getElementById('modeReason').textContent = reason;

        const icon = document.getElementById('modeIcon');
        const betAmountEl = document.getElementById('modeBetAmount');
        if (modeClass === 'bet' || modeClass === 'bet-high') {
            icon.innerHTML = '&#9654;'; // Play triangle
            // Show bet amount in banner
            if (totalBet && totalBet > 0) {
                betAmountEl.innerHTML = `<span class="mode-bet-label">Total Bet</span>$${totalBet.toFixed(2)}`;
                betAmountEl.classList.remove('hidden');
            } else {
                betAmountEl.classList.add('hidden');
            }
        } else if (modeClass === 'danger') {
            icon.innerHTML = '&#9888;'; // Warning
            betAmountEl.classList.add('hidden');
        } else {
            icon.innerHTML = '&#9208;'; // Pause
            betAmountEl.classList.add('hidden');
        }
    }

    updateSessionButtons() {
        const startBtn = document.getElementById('startSession');
        const endBtn = document.getElementById('endSession');

        if (this.sessionActive) {
            startBtn.classList.add('hidden');
            endBtn.classList.remove('hidden');
        } else {
            startBtn.classList.remove('hidden');
            endBtn.classList.add('hidden');
        }
    }

    addHistoryRow(result, data) {
        const tbody = document.getElementById('historyBody');
        const row = document.createElement('tr');

        const betResult = result.bet_result;
        const bankroll = data.bankroll;

        let betCell = '<span class="td-skip">SKIP</span>';
        let resultCell = '<span class="td-skip">-</span>';
        let pnlCell = '<span class="td-skip">$0</span>';

        if (betResult) {
            betCell = `$${betResult.per_number ? betResult.per_number.toFixed(2) : betResult.amount.toFixed(2)} × ${betResult.numbers ? betResult.numbers.length : '-'} nums`;
            resultCell = betResult.won
                ? `<span class="td-win">WIN +$${betResult.net.toFixed(2)}</span>`
                : `<span class="td-loss">LOSS</span>`;
            pnlCell = betResult.net >= 0
                ? `<span class="td-win">+$${betResult.net.toFixed(2)}</span>`
                : `<span class="td-loss">-$${Math.abs(betResult.net).toFixed(2)}</span>`;
        }

        const prevPred = this.lastPrediction;
        // Show anchors with spread in prediction column (e.g., "21±2, 5±1, 33±1")
        let predNums = '-';
        if (prevPred && prevPred.anchor_details && prevPred.anchor_details.length > 0) {
            predNums = prevPred.anchor_details.map(ad =>
                `${ad.number}\u00b1${ad.spread}`
            ).join(', ');
        } else if (prevPred && prevPred.anchors) {
            predNums = prevPred.anchors.join(', ');
        } else if (prevPred && prevPred.top_numbers) {
            predNums = prevPred.top_numbers.slice(0, 4).join(', ');
        }
        const predConf = prevPred ? `${prevPred.confidence}%` : '-';
        const predMode = prevPred ? prevPred.mode : '-';

        row.innerHTML = `
            <td>${this.spinCount}</td>
            <td><span class="td-number ${result.color}">${result.number}</span></td>
            <td>${result.color}</td>
            <td>${predNums}</td>
            <td>${predConf}</td>
            <td>${predMode}</td>
            <td>${betCell}</td>
            <td>${resultCell}</td>
            <td>${pnlCell}</td>
            <td>$${bankroll.bankroll.toFixed(2)}</td>
        `;

        // Insert at top (most recent first)
        tbody.insertBefore(row, tbody.firstChild);
    }

    clearHistory() {
        document.getElementById('historyBody').innerHTML = '';
        document.getElementById('spinCount').textContent = '0 spins';
    }

    // ─── Connection Status ──────────────────────────────────
    setConnectionStatus(status, text) {
        const dot = document.querySelector('.status-dot');
        const textEl = document.querySelector('.status-text');
        dot.className = `status-dot ${status}`;
        textEl.textContent = text;
    }

    // ─── Sessions Modal ─────────────────────────────────────
    showSessionsModal(data) {
        const body = document.getElementById('sessionsBody');

        if (!data.sessions || data.sessions.length === 0) {
            body.innerHTML = '<p style="text-align:center;color:var(--text-muted)">No sessions yet. Start your first session!</p>';
            return;
        }

        let html = `
            <div style="margin-bottom:16px;padding:12px;background:var(--bg-tertiary);border-radius:8px;">
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;text-align:center;">
                    <div><div style="font-size:0.7rem;color:var(--text-muted)">TOTAL SESSIONS</div><div style="font-size:1.2rem;font-weight:800">${data.total_sessions}</div></div>
                    <div><div style="font-size:0.7rem;color:var(--text-muted)">TOTAL P&L</div><div style="font-size:1.2rem;font-weight:800;color:${data.total_pnl >= 0 ? 'var(--success)' : 'var(--danger)'}">$${data.total_pnl.toFixed(2)}</div></div>
                    <div><div style="font-size:0.7rem;color:var(--text-muted)">WIN RATE</div><div style="font-size:1.2rem;font-weight:800">${data.overall_win_rate}%</div></div>
                </div>
            </div>
        `;

        html += data.sessions.map(s => `
            <div class="session-row">
                <div>
                    <div class="session-id">${s.id}</div>
                    <div style="font-size:0.72rem;color:var(--text-muted)">${s.total_spins} spins, ${s.total_bets} bets</div>
                </div>
                <div>
                    <span class="session-pnl ${s.pnl >= 0 ? 'positive' : 'negative'}">
                        ${s.pnl >= 0 ? '+' : ''}$${s.pnl.toFixed(2)}
                    </span>
                </div>
            </div>
        `).join('');

        body.innerHTML = html;
    }

    // ─── Theme ──────────────────────────────────────────────
    loadTheme() {
        const saved = localStorage.getItem('roulette-theme') || 'light';
        document.documentElement.setAttribute('data-theme', saved);
    }

    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('roulette-theme', next);
        this.wheel.draw();
        this.charts.updateTheme();
    }

    // ─── Session Timer ─────────────────────────────────────
    startTimer() {
        this.stopTimer();  // Clear any existing interval
        const timerEl = document.getElementById('sessionTimer');
        if (!timerEl) return;

        const updateDisplay = () => {
            if (!this.sessionStartTime) return;
            const elapsed = Date.now() - this.sessionStartTime;
            const totalSecs = Math.floor(elapsed / 1000);
            const mins = Math.floor(totalSecs / 60);
            const secs = totalSecs % 60;
            const display = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            timerEl.textContent = display;

            // Yellow warning after 20 minutes
            if (mins >= 20) {
                timerEl.style.color = 'var(--warning)';
                timerEl.style.fontWeight = '700';
            } else {
                timerEl.style.color = 'var(--text-muted)';
                timerEl.style.fontWeight = '';
            }
        };

        updateDisplay();  // Immediate first update
        this.timerInterval = setInterval(updateDisplay, 1000);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        this.sessionStartTime = null;
        const timerEl = document.getElementById('sessionTimer');
        if (timerEl) {
            timerEl.textContent = '--:--';
            timerEl.style.color = 'var(--text-muted)';
            timerEl.style.fontWeight = '';
        }
    }

    // ─── Toast ──────────────────────────────────────────────
    toast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
