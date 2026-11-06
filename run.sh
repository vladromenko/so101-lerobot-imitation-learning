#!/usr/bin/env bash
# Client only: the persistent 8031 service is the only Wristlab executor.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
exec /home/jetsonnx/Projects/so101_autonomous_pick_place/.venv/bin/python -m wristlab.cli "$@"
