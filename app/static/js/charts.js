/**
 * Charts Manager - Chart.js visualizations for bankroll and frequency.
 */

class ChartsManager {
    constructor() {
        this.bankrollChart = null;
        this.frequencyChart = null;
        this.frequencyData = new Array(37).fill(0);
        this.initCharts();
    }

    getThemeColors() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        return {
            text: isDark ? '#9299ae' : '#5f6577',
            grid: isDark ? '#2d3244' : '#e2e5ed',
            accent: isDark ? '#6b8aff' : '#4f6ef7',
            success: isDark ? '#34d369' : '#22c55e',
            danger: isDark ? '#f87171' : '#ef4444',
            bg: isDark ? '#1a1d28' : '#ffffff'
        };
    }

    initCharts() {
        const colors = this.getThemeColors();

        // Bankroll Chart
        const bankrollCtx = document.getElementById('bankrollChart');
        if (bankrollCtx) {
            this.bankrollChart = new Chart(bankrollCtx, {
                type: 'line',
                data: {
                    labels: ['Start'],
                    datasets: [{
                        label: 'Bankroll',
                        data: [4000],
                        borderColor: colors.accent,
                        backgroundColor: colors.accent + '20',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHitRadius: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            grid: { color: colors.grid },
                            ticks: { color: colors.text, font: { size: 10 } }
                        }
                    },
                    interaction: { intersect: false, mode: 'index' }
                }
            });
        }

        // Frequency Chart
        const freqCtx = document.getElementById('frequencyChart');
        if (freqCtx) {
            const redNumbers = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);

            const barColors = [];
            for (let i = 0; i < 37; i++) {
                if (i === 0) barColors.push('#16a34a');
                else if (redNumbers.has(i)) barColors.push('#dc2626');
                else barColors.push('#374151');
            }

            this.frequencyChart = new Chart(freqCtx, {
                type: 'bar',
                data: {
                    labels: Array.from({length: 37}, (_, i) => i.toString()),
                    datasets: [{
                        label: 'Frequency',
                        data: new Array(37).fill(0),
                        backgroundColor: barColors,
                        borderRadius: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: {
                                color: colors.text,
                                font: { size: 8 },
                                maxRotation: 0
                            }
                        },
                        y: {
                            grid: { color: colors.grid },
                            ticks: { color: colors.text, font: { size: 10 }, stepSize: 1 },
                            beginAtZero: true
                        }
                    }
                }
            });
        }
    }

    updateBankrollChart(history) {
        if (!this.bankrollChart) return;

        const labels = history.map((_, i) => i === 0 ? 'Start' : `Spin ${i}`);
        this.bankrollChart.data.labels = labels;
        this.bankrollChart.data.datasets[0].data = history;

        const colors = this.getThemeColors();
        const lastValue = history[history.length - 1];
        const firstValue = history[0];

        this.bankrollChart.data.datasets[0].borderColor =
            lastValue >= firstValue ? colors.success : colors.danger;
        this.bankrollChart.data.datasets[0].backgroundColor =
            (lastValue >= firstValue ? colors.success : colors.danger) + '20';

        this.bankrollChart.update('none');
    }

    updateFrequency(number) {
        if (!this.frequencyChart) return;

        this.frequencyData[number]++;
        this.frequencyChart.data.datasets[0].data = [...this.frequencyData];
        this.frequencyChart.update('none');
    }

    resetFrequency() {
        this.frequencyData = new Array(37).fill(0);
        if (this.frequencyChart) {
            this.frequencyChart.data.datasets[0].data = [...this.frequencyData];
            this.frequencyChart.update('none');
        }
    }

    updateTheme() {
        const colors = this.getThemeColors();

        if (this.bankrollChart) {
            this.bankrollChart.options.scales.y.grid.color = colors.grid;
            this.bankrollChart.options.scales.y.ticks.color = colors.text;
            this.bankrollChart.update('none');
        }

        if (this.frequencyChart) {
            this.frequencyChart.options.scales.x.ticks.color = colors.text;
            this.frequencyChart.options.scales.y.grid.color = colors.grid;
            this.frequencyChart.options.scales.y.ticks.color = colors.text;
            this.frequencyChart.update('none');
        }
    }
}
