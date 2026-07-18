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
for unit in pickleball-live pickleball-logs pickleball-freeze; do
    sed "s|@REPO@|$REPO|g" "$REPO/deploy/$unit.service" \
        > "$UNIT_DIR/$unit.service"
    cp "$REPO/deploy/$unit.timer" "$UNIT_DIR/$unit.timer"
done

systemctl --user daemon-reload
systemctl --user enable --now pickleball-live.timer
systemctl --user enable --now pickleball-freeze.timer
# Nightly log harvest runs here by default. The GitHub Action variant
# (.github/workflows/rally-logs.yml) has its schedule commented out —
# never run both collectors, they'd double-hit the API for the same
# files. Skip with WITH_LOGS_TIMER=0.
if [[ "${WITH_LOGS_TIMER:-1}" == "1" ]]; then
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
