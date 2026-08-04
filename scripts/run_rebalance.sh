#!/bin/bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_FILE="$PROJECT_DIR/logs/launchd.log"

mkdir -p "$(dirname "$LOG_FILE")"
cd "$PROJECT_DIR"

echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') : run starting ===" >> "$LOG_FILE"

OUTPUT="$("$PYTHON" -m scripts.rebalance --execute 2>&1)"
EXIT_CODE=$?

echo "$OUTPUT" >> "$LOG_FILE"
echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') : run finished, exit code $EXIT_CODE ===" >> "$LOG_FILE"

# --- Alerting ---
# Keyed off rebalance.py's own documented exit codes (0/1/2/3), not a
# blanket "non-zero = crash" assumption, since 1 is its normal halt code.
WEEKDAY="$(date '+%u')"   # 1 (Mon) through 7 (Sun)

notify() {
  say -v Samantha "$1"
}

if echo "$OUTPUT" | grep -q "Traceback (most recent call last)"; then
  notify "T212 rebalance crashed with an unhandled error. Check the log."
elif [ "$EXIT_CODE" -eq 2 ]; then
  notify "T212 rebalance produced no target. Check the log for a warmup or data problem."
elif [ "$EXIT_CODE" -eq 3 ]; then
  notify "T212 rebalance placed some orders but not all. This needs manual review."
elif [ "$EXIT_CODE" -eq 1 ]; then
  if echo "$OUTPUT" | grep -q "HALTED" && [ "$WEEKDAY" -le 5 ]; then
    notify "T212 rebalance halted on a trading day. Check the log for the reason."
  fi
elif [ "$EXIT_CODE" -ne 0 ]; then
  notify "T212 rebalance exited with unexpected code $EXIT_CODE. Check the log."
fi
