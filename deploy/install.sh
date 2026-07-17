#!/usr/bin/env bash
# One-shot setup on a fresh VPS. Run as your login user (not root):
#   creates .venv, installs httpx, installs + enables a user-level systemd
#   timer that starts the poller every morning at 09:15 Pacific.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
command -v systemctl >/dev/null || { echo "systemd is required"; exit 1; }

python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install --quiet --upgrade pip httpx

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
for unit in pickleball-live pickleball-logs; do
    sed "s|@REPO@|$REPO|g" "$REPO/deploy/$unit.service" \
        > "$UNIT_DIR/$unit.service"
    cp "$REPO/deploy/$unit.timer" "$UNIT_DIR/$unit.timer"
done

systemctl --user daemon-reload
systemctl --user enable --now pickleball-live.timer
# The nightly log-harvest timer is OPT-IN: the GitHub Action
# (.github/workflows/rally-logs.yml) is the default collector — never run
# both, they'd double-hit the API for the same files.
if [[ "${WITH_LOGS_TIMER:-0}" == "1" ]]; then
    systemctl --user enable --now pickleball-logs.timer
fi
# Keep user services alive after you log out of the box.
loginctl enable-linger "$(whoami)"

echo
echo "Installed. Verify with:"
echo "  systemctl --user list-timers pickleball-live.timer"
echo "  $REPO/.venv/bin/python $REPO/scraper/live_poller.py --once"
echo
echo "Full dress rehearsal of the daily unit (quick exit on a non-event day):"
echo "  systemctl --user start pickleball-live.service"
echo "  journalctl --user -u pickleball-live -e"
