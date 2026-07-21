#!/bin/bash
# Run the main cost_at_accuracy_target experiment (method = Ours / LIMO) for one
# dataset at its winning operating point.
#
#   scripts/run_main.sh opp115     # Policy
#   scripts/run_main.sh hoc        # Cancer
#   scripts/run_main.sh cuad       # CUAD
#
# Requires the small + large + embedding endpoints to be up (see README):
# pretrain.mode:fit calls the live models for query synthesis + active labeling;
# the eval phase replays the large model from examples/predictions/.
set -euo pipefail

DS="${1:-}"
case "$DS" in
  opp115|hoc|cuad) ;;
  *) echo "usage: scripts/run_main.sh <opp115|hoc|cuad>" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/run.sh" -m eval --config "$ROOT/configs/main/${DS}.yaml"
