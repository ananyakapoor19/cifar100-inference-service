#!/usr/bin/env bash
# run_experiments.sh – Execute the full experiment matrix and collect CSV results.
#
# Experiment matrix:
#   precision  ∈ {fp32, int8}
#   batch_size ∈ {1, 8}
#   concurrency ∈ {10, 50, 200}
#
# Usage:
#   HOST=http://localhost:8080 ./benchmarks/run_experiments.sh
#
# Results are written to ./results/<label>/  as Locust CSV files.
# After all runs, calls analysis/plot_results.py to generate charts.

set -euo pipefail

HOST="${HOST:-http://localhost:8080}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
RUN_TIME="${RUN_TIME:-60s}"          # Duration of each Locust run
SPAWN_RATE="${SPAWN_RATE:-10}"       # Users spawned per second

PRECISIONS=("fp32" "int8")
BATCH_SIZES=("1" "8")
CONCURRENCIES=("10" "50" "200")

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo " CIFAR-100 Inference Benchmark Suite"
echo " Host:        $HOST"
echo " Run time:    $RUN_TIME"
echo " Matrix size: $(( ${#PRECISIONS[@]} * ${#BATCH_SIZES[@]} * ${#CONCURRENCIES[@]} )) cells"
echo "============================================================"

cell=0
total=$(( ${#PRECISIONS[@]} * ${#BATCH_SIZES[@]} * ${#CONCURRENCIES[@]} ))

for precision in "${PRECISIONS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for concurrency in "${CONCURRENCIES[@]}"; do
      cell=$(( cell + 1 ))
      label="${precision}_b${batch_size}_c${concurrency}"
      out_dir="${RESULTS_DIR}/${label}"
      mkdir -p "$out_dir"

      echo ""
      echo "─────────────────────────────────────────────────────────"
      echo " Cell $cell/$total: precision=$precision  batch=$batch_size  users=$concurrency"
      echo " Output: $out_dir"
      echo "─────────────────────────────────────────────────────────"

      PRECISION="$precision" \
      BATCH_SIZE="$batch_size" \
      python3 -m locust \
        -f benchmarks/locustfile.py \
        --headless \
        --users "$concurrency" \
        --spawn-rate "$SPAWN_RATE" \
        --run-time "$RUN_TIME" \
        --host "$HOST" \
        --csv "${out_dir}/stats" \
        --html "${out_dir}/report.html" \
        2>&1 | tee "${out_dir}/locust.log" || true  # non-zero on failures is expected under load

      echo " ✓ Cell $cell done."
    done
  done
done

echo ""
echo "============================================================"
echo " All $total cells complete. Generating analysis charts…"
echo "============================================================"

python3 benchmarks/analysis/plot_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir  "$RESULTS_DIR/charts"

echo "Charts saved to $RESULTS_DIR/charts"
echo "Done."
