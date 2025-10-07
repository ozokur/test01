#!/bin/bash
set -euo pipefail

show_alert() {
  local title=${1:-"YOLO GUI"}
  local message=${2:-""}

  if command -v osascript >/dev/null 2>&1; then
    osascript <<OSA
display alert "$title" message "$message"
OSA
  else
    >&2 echo "$title: $message"
  fi
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements.txt"
BOOTSTRAP_MARKER="${VENV_DIR}/.bootstrap-complete"

ensure_python() {
  local python_bin=""

  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  elif command -v python >/dev/null 2>&1; then
    python_bin="python"
  fi

  if [ -z "${python_bin}" ]; then
    show_alert "Python 3 not found" "Install Python 3.9 or newer, then re-run the launcher."
    exit 1
  fi

  echo "${python_bin}"
}

bootstrap_virtualenv() {
  local base_python
  base_python=$(ensure_python)

  if [ ! -x "${VENV_PYTHON}" ]; then
    "${base_python}" -m venv "${VENV_DIR}"
  fi

  if [ ! -x "${VENV_PYTHON}" ]; then
    show_alert "Virtualenv error" "Failed to create a virtual environment in ${VENV_DIR}."
    exit 1
  fi

  if [ ! -f "${BOOTSTRAP_MARKER}" ]; then
    "${VENV_PYTHON}" -m pip install --upgrade pip

    if [ -f "${REQUIREMENTS_FILE}" ]; then
      "${VENV_PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"
    fi

    touch "${BOOTSTRAP_MARKER}"
  fi
}

bootstrap_virtualenv

cd "${REPO_ROOT}"
exec "${VENV_PYTHON}" yolo_gui.py "$@"
