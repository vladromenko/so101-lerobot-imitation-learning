#!/usr/bin/env bash
# Web interface: http://homebrain.local:8030 (from the Mac). Stays up until stopped with ./ui.sh stop.
cd "$(dirname "$0")"
PY="$HOME/Projects/so101_autonomous_pick_place/.venv/bin/python"
case "${1:-start}" in
  stop)  pkill -f "box_extract/ui/server.py" && echo stopped ;;
  fg)    exec "$PY" box_extract/ui/server.py ;;
  *)     pkill -f "box_extract/ui/server.py" 2>/dev/null; mkdir -p logs/ui
         nohup "$PY" box_extract/ui/server.py > logs/ui/server.log 2>&1 < /dev/null & disown
         sleep 2; echo "started: http://$(hostname).local:8030" ;;
esac
