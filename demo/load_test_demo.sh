#!/usr/bin/env bash
# load_test_demo.sh – 60-second live load test for the class demo.
#
# Shows Cloud Run auto-scaling in real time.
# Watch the GCP Cloud Run console while this runs.
#
# Usage:
#   HOST=https://your-cloud-run-url PRECISION=fp32 ./demo/load_test_demo.sh

set -euo pipefail

HOST="${HOST:-http://localhost:8080}"
PRECISION="${PRECISION:-fp32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USERS="${USERS:-50}"
SPAWN_RATE="${SPAWN_RATE:-5}"
RUN_TIME="${RUN_TIME:-60s}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results/demo_run"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " Live Load Test"
echo " Host:       $HOST"
echo " Precision:  $PRECISION"
echo " Users:      $USERS (ramping at $SPAWN_RATE/s)"
echo " Duration:   $RUN_TIME"
echo " Watch Cloud Run scaling in GCP console ↗"
echo "============================================"

PRECISION=$PRECISION BATCH_SIZE=$BATCH_SIZE locust \
  --headless \
  --locustfile "$SCRIPT_DIR/../benchmarks/locustfile.py" \
  --host "$HOST" \
  --users "$USERS" \
  --spawn-rate "$SPAWN_RATE" \
  --run-time "$RUN_TIME" \
  --csv "$RESULTS_DIR/demo" \
  --html "$RESULTS_DIR/demo_report.html" \
  --only-summary

echo ""
echo "Results saved to $RESULTS_DIR/"
echo "Open $RESULTS_DIR/demo_report.html for the full report."
