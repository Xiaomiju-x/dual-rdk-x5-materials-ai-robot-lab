#!/usr/bin/env bash
# Deprecated compatibility entry. Direct in-place deployment is intentionally disabled.
set -euo pipefail

cat >&2 <<'EOF'
deploy.sh is disabled because it bypasses the immutable manifest, candidate
gate, browser/origin evidence, staged health checks and verified rollback.

Build an immutable candidate tree, then run its tools/deploy_staged.sh entry.
EOF
exit 64
