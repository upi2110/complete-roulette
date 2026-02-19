#!/bin/bash
# Hourly monitoring for marathon_v2
OUTPUT="/Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso/scripts/marathon_v2_output.txt"
LOG="/Users/ubusan-nb-ecr/Documents/UpenderImp/CompleteCaso/scripts/hourly_updates.txt"

echo "========================================" >> "$LOG"
echo "HOURLY MONITORING STARTED: $(date)" >> "$LOG"
echo "========================================" >> "$LOG"

for hour in $(seq 1 12); do
    sleep 3600
    echo "" >> "$LOG"
    echo "════════════════════════════════════════════════════════════════" >> "$LOG"
    echo "HOUR $hour UPDATE — $(date)" >> "$LOG"
    echo "════════════════════════════════════════════════════════════════" >> "$LOG"
    
    # Check if process is alive
    if ps aux | grep "marathon_v2.py" | grep -v grep > /dev/null 2>&1; then
        CPU=$(ps aux | grep "marathon_v2.py" | grep -v grep | awk '{print $3}')
        echo "  Process: RUNNING at ${CPU}% CPU" >> "$LOG"
    else
        echo "  Process: STOPPED" >> "$LOG"
    fi
    
    LINES=$(wc -l < "$OUTPUT")
    echo "  Output lines: $LINES" >> "$LOG"
    
    # Show phases reached
    echo "  Phases reached:" >> "$LOG"
    grep "^PHASE" "$OUTPUT" | while read line; do
        echo "    $line" >> "$LOG"
    done
    
    # Check for errors
    ERRORS=$(grep -c "Traceback\|Error" "$OUTPUT" 2>/dev/null || echo 0)
    echo "  Errors: $ERRORS" >> "$LOG"
    
    # Show current leaderboard top 5 (last one printed)
    echo "  --- Latest Leaderboard Top 5 ---" >> "$LOG"
    # Get the last leaderboard block
    grep -A 8 "AFTER PHASE" "$OUTPUT" | tail -15 >> "$LOG"
    
    echo "  --- Last 5 lines of output ---" >> "$LOG"
    tail -5 "$OUTPUT" >> "$LOG"
    
    # If process died, note it
    if ! ps aux | grep "marathon_v2.py" | grep -v grep > /dev/null 2>&1; then
        echo "" >> "$LOG"
        echo "  *** MARATHON FINISHED OR CRASHED ***" >> "$LOG"
        echo "  Check scripts/marathon_v2_output.txt for full results" >> "$LOG"
        break
    fi
done

echo "" >> "$LOG"
echo "════════════════════════════════════════════════════════════════" >> "$LOG"
echo "MONITORING COMPLETE — $(date)" >> "$LOG"
echo "════════════════════════════════════════════════════════════════" >> "$LOG"
