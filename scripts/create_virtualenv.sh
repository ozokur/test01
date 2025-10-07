#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
VENV_DIR=${1:-.venv}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} command not found. Please install Python 3.9+ first." >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [ -d "${VENV_DIR}" ]; then
  echo "Virtual environment directory '${VENV_DIR}' already exists." >&2
  echo "Activate it with: source ${VENV_DIR}/bin/activate" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

cat <<INFO
Created virtual environment at '${VENV_DIR}'.
Activate it with:

  source ${VENV_DIR}/bin/activate

When you're done, deactivate it with:

  deactivate
INFO
