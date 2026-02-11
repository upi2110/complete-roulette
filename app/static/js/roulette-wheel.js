/**
 * Roulette Wheel - Static flat European roulette wheel
 * matching reference image: light blue inner area with blue radial lines,
 * outer ring with number boxes (red/black/green).
 * Prediction circles shown as a ring outside the wheel for clear visibility.
 * No spin animation.
 */

class RouletteWheel {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.centerX = this.canvas.width / 2;
        this.centerY = this.canvas.height / 2;
        // Shrink main wheel to leave room for prediction ring outside
        this.radius = Math.min(this.centerX, this.centerY) - 30;

        // European wheel order
        this.numbers = [
            0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36,
            11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9,
            22, 18, 29, 7, 28, 12, 35, 3, 26
        ];

        this.redNumbers = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
        this.sliceAngle = (2 * Math.PI) / this.numbers.length;

        this.highlightedNumber = null;      // Last spin result
        this.predictedNumbers = [];          // All predicted numbers (anchors + neighbours)
        this.anchorNumbers = [];             // Anchor numbers (subset of predicted)

        this.draw();
    }

    getColor(num) {
        if (num === 0) return '#16a34a';
        return this.redNumbers.has(num) ? '#c0392b' : '#1a1a2e';
    }

    draw() {
        const ctx = this.ctx;
        const cx = this.centerX;
        const cy = this.centerY;
        const r = this.radius;
        const predictedSet = new Set(this.predictedNumbers);
        const anchorSet = new Set(this.anchorNumbers);

        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // === Dimensions ===
        const outerR = r;                 // Outer edge of number boxes
        const innerR = r * 0.76;          // Inner edge of number boxes
        const boxDepth = outerR - innerR;
        const centerDotR = 3;

        // === Prediction ring radius (outside wheel) ===
        const predRingR = outerR + 18;    // Center of prediction circles
        const predCircleR = 10;           // Radius of each prediction circle

        // === 1. Draw prediction ring (circles outside the wheel) ===
        if (this.predictedNumbers.length > 0) {
            for (let i = 0; i < this.numbers.length; i++) {
                const num = this.numbers[i];
                const isAnchor = anchorSet.has(num);
                const isPredicted = predictedSet.has(num);
                const isResult = this.highlightedNumber === num;

                if (!isPredicted && !isAnchor) continue;

                const midAngle = i * this.sliceAngle - Math.PI / 2 + this.sliceAngle / 2;
                const px = cx + predRingR * Math.cos(midAngle);
                const py = cy + predRingR * Math.sin(midAngle);

                // Circle background
                ctx.beginPath();
                ctx.arc(px, py, predCircleR, 0, 2 * Math.PI);

                if (isResult) {
                    // Last spin result that was also predicted — bright green
                    ctx.fillStyle = '#2ecc71';
                    ctx.strokeStyle = '#27ae60';
                    ctx.lineWidth = 2.5;
                } else if (isAnchor) {
                    ctx.fillStyle = '#f39c12';
                    ctx.strokeStyle = '#d35400';
                    ctx.lineWidth = 2;
                } else {
                    ctx.fillStyle = '#f5d76e';
                    ctx.strokeStyle = '#e67e22';
                    ctx.lineWidth = 1.5;
                }
                ctx.fill();
                ctx.stroke();

                // Number text inside circle
                ctx.save();
                ctx.fillStyle = '#ffffff';
                ctx.font = `bold 10px -apple-system, "Segoe UI", sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.shadowColor = 'rgba(0,0,0,0.3)';
                ctx.shadowBlur = 1;
                ctx.fillText(num.toString(), px, py);
                ctx.restore();
            }
        }

        // === 2. Draw light inner circle (light blue-white) ===
        ctx.beginPath();
        ctx.arc(cx, cy, innerR, 0, 2 * Math.PI);
        ctx.fillStyle = '#e8f0fe';
        ctx.fill();

        // === 3. Draw blue radial lines from center to inner ring edge ===
        ctx.strokeStyle = '#7bafd4';
        ctx.lineWidth = 0.8;
        for (let i = 0; i < this.numbers.length; i++) {
            const angle = i * this.sliceAngle - Math.PI / 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + innerR * Math.cos(angle), cy + innerR * Math.sin(angle));
            ctx.stroke();
        }

        // === 4. Center dot ===
        ctx.beginPath();
        ctx.arc(cx, cy, centerDotR, 0, 2 * Math.PI);
        ctx.fillStyle = '#7bafd4';
        ctx.fill();

        // === 5. Inner ring border ===
        ctx.beginPath();
        ctx.arc(cx, cy, innerR, 0, 2 * Math.PI);
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2;
        ctx.stroke();

        // === 6. Draw number boxes on the outer ring ===
        for (let i = 0; i < this.numbers.length; i++) {
            const startAngle = i * this.sliceAngle - Math.PI / 2;
            const endAngle = startAngle + this.sliceAngle;
            const num = this.numbers[i];

            const isResult = this.highlightedNumber === num;
            const isAnchor = anchorSet.has(num);
            const isPredicted = predictedSet.has(num);

            // Draw the box (arc segment)
            ctx.beginPath();
            ctx.arc(cx, cy, outerR, startAngle, endAngle);
            ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
            ctx.closePath();

            // Box fill color — keep standard roulette colors on the wheel itself
            if (isResult) {
                ctx.fillStyle = '#2ecc71'; // Bright green for last result
            } else if (num === 0) {
                ctx.fillStyle = '#16a34a';
            } else if (this.redNumbers.has(num)) {
                ctx.fillStyle = '#c0392b';
            } else {
                ctx.fillStyle = '#1a1a2e';
            }
            ctx.fill();

            // Box border — highlight predicted ones with a glow border
            if (isResult) {
                ctx.strokeStyle = '#27ae60';
                ctx.lineWidth = 2.5;
            } else if (isAnchor) {
                ctx.strokeStyle = '#f39c12';
                ctx.lineWidth = 2.5;
            } else if (isPredicted) {
                ctx.strokeStyle = '#f5d76e';
                ctx.lineWidth = 2;
            } else {
                ctx.strokeStyle = '#334155';
                ctx.lineWidth = 1;
            }
            ctx.stroke();

            // === 7. Draw number text ===
            const midAngle = startAngle + this.sliceAngle / 2;
            const textR = innerR + boxDepth / 2;
            const tx = cx + textR * Math.cos(midAngle);
            const ty = cy + textR * Math.sin(midAngle);

            ctx.save();
            ctx.translate(tx, ty);
            ctx.rotate(midAngle + Math.PI / 2);

            const fontSize = r < 110 ? 9 : 12;
            ctx.fillStyle = '#ffffff';
            ctx.font = `bold ${fontSize}px -apple-system, "Segoe UI", sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            ctx.fillText(num.toString(), 0, 0);
            ctx.restore();
        }

        // === 8. Outer border ring ===
        ctx.beginPath();
        ctx.arc(cx, cy, outerR + 1, 0, 2 * Math.PI);
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // === 9. Show info in center ===
        if (this.highlightedNumber !== null) {
            const resultColor = this.getColor(this.highlightedNumber);
            ctx.fillStyle = resultColor;
            ctx.font = `bold ${r < 110 ? 18 : 24}px -apple-system, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(this.highlightedNumber.toString(), cx, cy - 6);

            ctx.fillStyle = '#64748b';
            ctx.font = `600 ${r < 110 ? 7 : 9}px -apple-system, sans-serif`;
            ctx.fillText('LAST SPIN', cx, cy + 14);
        } else if (this.predictedNumbers.length > 0) {
            ctx.fillStyle = '#e67e22';
            ctx.font = `bold ${r < 110 ? 14 : 18}px -apple-system, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${this.predictedNumbers.length}`, cx, cy - 6);

            ctx.fillStyle = '#64748b';
            ctx.font = `600 ${r < 110 ? 7 : 9}px -apple-system, sans-serif`;
            ctx.fillText('PREDICTED', cx, cy + 14);
        }
    }

    /**
     * Set predicted numbers and anchors to highlight on the wheel.
     * @param {number[]} numbers - All predicted numbers (anchors + neighbours)
     * @param {number[]} anchors - Anchor numbers (subset)
     */
    setPredictedNumbers(numbers, anchors) {
        this.predictedNumbers = numbers || [];
        this.anchorNumbers = anchors || [];
        this.draw();
    }

    /**
     * Highlight the last spin result number (no animation, just visual update).
     */
    highlightNumber(num) {
        this.highlightedNumber = num;
        this.draw();
    }

    reset() {
        this.highlightedNumber = null;
        this.predictedNumbers = [];
        this.anchorNumbers = [];
        this.draw();
    }
}
