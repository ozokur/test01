#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN=""
if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if [ -z "${PYTHON_BIN}" ]; then
  if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display alert "Python 3 not found" message "Install Python 3.9 or newer, then re-run the launcher."'
  else
    echo "Python 3 not found. Install Python 3.9 or newer, then re-run the launcher." >&2
  fi
  exit 1
fi
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" yolo_gui.py "$@"
