#!/bin/bash
set -euo pipefail
MODE="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
# Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv_install.sh
    sh /tmp/uv_install.sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if [ "$MODE" == "--train" ]; then
    uv sync --extra train --extra dev
else
    uv sync --extra dev
fi
echo "Bootstrap complete."
uv run python scripts/doctor.py
