#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  CascadeWatch — install dependencies & launch the dashboard
#  Usage:  bash run_dashboard.sh
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

echo "==> Installing / verifying Dash & Plotly..."
"$PYTHON" -m pip install --quiet --upgrade dash plotly

echo "==> Starting CascadeWatch dashboard..."
echo "    Open  http://127.0.0.1:8050  in your browser."
echo "    Press Ctrl+C to stop."
echo ""
"$PYTHON" "$SCRIPT_DIR/dashboard.py"
