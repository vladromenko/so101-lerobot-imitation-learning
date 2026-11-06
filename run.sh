#!/usr/bin/env bash
# Usage: ./run.sh                      -> measure + plan + simulate (no motor commands)
#        ./run.sh --execute --confirm-attended [--max-cubes N]
# The extra --stop-compat argument lets the existing UI STOP button recognise this process.
cd "$(dirname "$0")"
exec "$HOME/Projects/so101_autonomous_pick_place/.venv/bin/python" scripts/extract_from_box.py --stop-compat single_cube_pick_place.py "$@"
