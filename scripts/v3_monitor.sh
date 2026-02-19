#!/bin/bash
# Auto-monitor marathon V3 every 30 minutes
LOG="/Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso/scripts/v3_updates.txt"
OUTPUT="/Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso/scripts/marathon_v3_output.txt"

while true; do
    echo "=== UPDATE $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"

    # Check if running
    PID=$(pgrep -f "marathon_v3_24h" | head -1)
    if [ -z "$PID" ]; then
        echo "  MARATHON V3 STOPPED" >> "$LOG"
        echo "  Final output lines: $(wc -l < "$OUTPUT")" >> "$LOG"
        echo "  Last 10 lines:" >> "$LOG"
        tail -10 "$OUTPUT" >> "$LOG"
        break
    fi

    CPU=$(ps -p "$PID" -o %cpu= 2>/dev/null)
    echo "  PID: $PID | CPU: ${CPU}%" >> "$LOG"
    echo "  Output lines: $(wc -l < "$OUTPUT")" >> "$LOG"

    # Get current phase
    PHASE=$(grep "^PHASE:" "$OUTPUT" | tail -1)
    echo "  $PHASE" >> "$LOG"

    # Get tested count
    TESTED=$(grep "Tested:" "$OUTPUT" | tail -1)
    echo "  $TESTED" >> "$LOG"

    # Get latest leaderboard top 5
    echo "  --- Latest Top 5 ---" >> "$LOG"
    grep -A 6 "AFTER PHASE" "$OUTPUT" | tail -20 >> "$LOG"

    # Get any errors
    ERRORS=$(grep -i "error\|traceback\|exception" "$OUTPUT" | wc -l)
    echo "  Errors: $ERRORS" >> "$LOG"

    echo "---" >> "$LOG"

    sleep 1800  # 30 minutes
done
