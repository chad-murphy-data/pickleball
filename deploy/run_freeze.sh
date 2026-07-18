#!/usr/bin/env bash
# Run the lineup-freeze watcher for today's event window, then commit and
# push any captured freezes. Started daily by pickleball-freeze.timer;
# exits within minutes on non-event days (nothing scheduled → idles out).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export TZ="${POLLER_TZ:-America/Los_Angeles}"

PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

"$PY" scraper/lineup_freeze.py --interval "${FREEZE_INTERVAL:-90}"

if [[ -n "$(git status --porcelain -- live/)" ]]; then
    git add live/
    git commit -m "live: $(date +%F) lineup freezes"
fi
if [[ -n "$(git log --oneline '@{u}..HEAD' 2>/dev/null)" ]]; then
    git pull --rebase --autostash
    git push origin HEAD
fi
