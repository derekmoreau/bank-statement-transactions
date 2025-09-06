#!/usr/bin/env bash
set -euo pipefail

# Absolute project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda base path and environment name
CONDA_BIN="/opt/anaconda3/bin/conda"
ACTIVATE="/opt/anaconda3/bin/activate"
ENV_NAME="bankparser"

echo "[run.sh] Ensuring conda env '$ENV_NAME' exists..."
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[run.sh] Creating env '$ENV_NAME' (python=3.13)..."
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.13
fi

echo "[run.sh] Activating env '$ENV_NAME'..."
source "$ACTIVATE" "$ENV_NAME"

echo "[run.sh] Installing/upgrading dependencies..."
pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"

cd "$PROJECT_DIR"
echo "[run.sh] Launching Streamlit..."
python -m streamlit run enhanced_parser.py


