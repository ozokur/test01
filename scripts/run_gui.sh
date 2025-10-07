#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
  echo "python command not found. Please install Python 3.9+ first." >&2
  exit 1
fi

cd "${REPO_ROOT}"

python yolo_gui.py "$@"
