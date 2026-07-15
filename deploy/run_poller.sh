#!/usr/bin/env bash
# Run the Tier-1 live poller for today's event window, then commit and push
# any captured JSONL. Started daily by pickleball-live.timer; exits within
# seconds on non-event days (the poller discovers nothing and quits).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# The poller keys "today" off the machine-local date. Pin it to the tour's
# home timezone so a UTC/European box never rolls the date over while late
# Pacific-evening matches are still in play.
export TZ="${POLLER_TZ:-America/Los_Angeles}"

PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

"$PY" scraper/live_poller.py --interval "${POLLER_INTERVAL:-25}"

# Preserve the day's events. A failed push leaves the commit behind; the
# next successful push (systemd retry or tomorrow's run) carries it along.
if [[ -n "$(git status --porcelain -- live/)" ]]; then
    git add live/
    git commit -m "live: $(date +%F) events"
fi
if [[ -n "$(git log --oneline '@{u}..HEAD' 2>/dev/null)" ]]; then
    git pull --rebase --autostash
    git push origin HEAD
fi
