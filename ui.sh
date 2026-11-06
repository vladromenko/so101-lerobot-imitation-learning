#!/usr/bin/env bash
# Only this copy's named service; never pkill a server or touch port 8030.
set -euo pipefail
case "${1:-start}" in
  start) systemctl --user start so101-wristlab-web.service ;;
  restart) systemctl --user restart so101-wristlab-web.service ;;
  stop) systemctl --user stop so101-wristlab-web.service ;;
  status) systemctl --user status so101-wristlab-web.service --no-pager ;;
  *) echo 'Usage: ./ui.sh [start|restart|stop|status]' >&2; exit 2 ;;
esac
