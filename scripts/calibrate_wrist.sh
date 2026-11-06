#!/usr/bin/env bash
# Collect wrist-camera calibration views of a stationary cube (the arm moves!) and fit them.
#   ./scripts/calibrate_wrist.sh            -> plan and simulate only
#   ./scripts/calibrate_wrist.sh --execute  -> move the arm (operator at STOP), then fit
# Views adapt to where the cube is and keep away from the box; the box opening comes from this program.
set -euo pipefail
cd "$(dirname "$0")/.."
P="$HOME/Projects/so101_autonomous_pick_place"
PY="$P/.venv/bin/python"
SEED=$(ls -t "$P"/logs/wrist_extraction_preparation/*/qualified-candidate.json | head -1)
EXEC=""
[[ "${1:-}" == "--execute" ]] && EXEC="--execute"
OUT=calibration/run-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
"$PY" scripts/run_project_calibrator.py "$P/scripts/calibrate_wrist_camera.py" --markerless $EXEC --silhouette-geometry-seed "$SEED" | tee "$OUT/acquisition.out"
[[ -z "$EXEC" ]] && exit 0
DIR=$(sed -n 's/^ARTIFACTS //p' "$OUT/acquisition.out" | tail -1)
"$PY" scripts/fit_wrist_top_face.py "$DIR/observations.json" --out "$OUT/candidate.json" | tee "$OUT/fit.out"
