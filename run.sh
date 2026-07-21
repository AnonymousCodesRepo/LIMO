#!/bin/bash
# Convenience wrapper: run a repository script with the repo root on PYTHONPATH.
# Repo root is derived from the script location, so it works on any clone.
# Override the Python interpreter with MOP_PYTHON if needed.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${MOP_PYTHON:-python}"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"
exec "$PYTHON" "$@"
