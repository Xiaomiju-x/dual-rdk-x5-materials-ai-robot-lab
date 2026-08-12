#!/usr/bin/env bash
# Restore the service owner that runtime evidence preparation temporarily stopped.

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREP_REPORT="${EMBODIED_V3_RUNTIME_PREP:-$HOME/embodied_v3_runtime/latest_prepare.json}"

[ -f "$TOOLS_DIR/start_embodied_v3_stack.sh" ] || {
  echo "ERR missing $TOOLS_DIR/start_embodied_v3_stack.sh" >&2
  exit 2
}

# The acceptance helper owns these temporary processes. Stop only that known stack.
bash "$TOOLS_DIR/start_embodied_v3_stack.sh" stop

scope="none"
if [ -s "$PREP_REPORT" ]; then
  scope="$(python3 - "$PREP_REPORT" <<'PY'
import json
import sys
from pathlib import Path

try:
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(str(report.get("managed_service_stopped") or "none"))
except Exception:
    print("none")
PY
)"
fi

case ",$scope," in
  *,system,*)
    sudo -n systemctl start embodied_brain.service
    ;;
esac
case ",$scope," in
  *,user,*)
    systemctl --user start embodied_brain.service
    ;;
esac

echo "EMBODIED_V3_RUNTIME_RESTORED"
echo "managed_service_scope: $scope"
if [[ ",$scope," == *,system,* ]]; then
  echo "system_service: $(systemctl is-active embodied_brain.service 2>/dev/null || true)"
fi
if [[ ",$scope," == *,user,* ]]; then
  echo "user_service: $(systemctl --user is-active embodied_brain.service 2>/dev/null || true)"
fi
echo "safety: this script never clears F407 estop"
