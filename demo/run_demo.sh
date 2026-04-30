#!/usr/bin/env bash
# run_demo.sh – Start the Streamlit demo app.
#
# Usage:
#   ./demo/run_demo.sh                                        # uses localhost:8080
#   CIFAR100_API_URL=https://your-cloud-run-url ./demo/run_demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python -c "import streamlit" 2>/dev/null; then
  echo "Installing demo dependencies..."
  pip install -q -r "$SCRIPT_DIR/requirements.txt"
fi

export CIFAR100_API_URL="${CIFAR100_API_URL:-http://localhost:8080}"

echo "============================================"
echo " CIFAR-100 Inference Demo"
echo " API: $CIFAR100_API_URL"
echo " Opening http://localhost:8501"
echo "============================================"

streamlit run "$SCRIPT_DIR/app.py" \
  --server.port 8501 \
  --server.headless false \
  --browser.gatherUsageStats false
