"""One command for the whole POC. Run this on YOUR machine, not a server.

    python vision/run_poc.py --url "<youtube url>" --pick midseason-womens
    python vision/run_poc.py --url "<youtube url>" --match <uuid>
    python vision/run_poc.py --check "Florida Smash v Bay Area Breakers" --date 2026-07-08

Does, in order: rally timeline -> download audio -> detect contacts ->
sync + report.  Skips any step whose output already exists, so re-running
after a crash is cheap.

WHY LOCAL AND NOT THE DROPLET: YouTube bot-checks datacenter IPs.  The
agent sandbox this was written in gets "Sign in to confirm you're not a
bot", and a DigitalOcean box will get the same.  A laptop on residential
internet does not.

Dependencies: pip install yt-dlp numpy imageio-ffmpeg
(imageio-ffmpeg ships a static ffmpeg, so no brew/apt needed.)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "vision"


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        sys.exit(f"failed: {cmd[0]}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="YouTube URL of the matchup VOD")
    ap.add_argument("--pick", help="curated match name (see rally_timeline.py)")
    ap.add_argument("--match", help="match uuid")
    ap.add_argument("--check", help='matchup as "Team A v Team B" — just report '
                                    "whether its games have usable rally windows")
    ap.add_argument("--date", help="YYYY-MM-DD, used with --check")
    ap.add_argument("--audio", default=str(OUT / "match_audio.m4a"))
    ap.add_argument("--k", type=float, default=4.0)
    args = ap.parse_args()
    py = sys.executable

    if args.check:
        run([py, str(ROOT / "vision" / "rally_timeline.py"),
             "--teams", args.check] + (["--date", args.date] if args.date else []))
        return

    if not args.url or not (args.pick or args.match):
        ap.error("need --url plus one of --pick/--match (or use --check)")

    # 1. rally timeline (network: the open BFF, works anywhere)
    sel = ["--pick", args.pick] if args.pick else ["--match", args.match]
    run([py, str(ROOT / "vision" / "rally_timeline.py")] + sel)
    tls = sorted(OUT.glob("rally_timeline_*.csv"),
                 key=lambda p: p.stat().st_mtime)
    timeline = tls[-1]

    # 2. audio
    audio = Path(args.audio)
    audio.parent.mkdir(parents=True, exist_ok=True)
    if audio.exists():
        print(f"\n[skip] audio already at {audio}")
    else:
        run(["yt-dlp", "-f", "bestaudio", "-o", str(audio), args.url])

    # 3. contacts
    contacts = OUT / "contacts.csv"
    if contacts.exists():
        print(f"\n[skip] contacts already at {contacts}")
    else:
        run([py, str(ROOT / "vision" / "audio_contacts.py"),
             "--audio", str(audio), "--out", str(contacts), "--k", str(args.k)])

    # 4. report
    run([py, str(ROOT / "vision" / "poc_report.py"),
         "--timeline", str(timeline), "--contacts", str(contacts)])

    print("\n" + "=" * 68)
    print("A matchup VOD holds ALL FOUR games, so the same audio can be scored\n"
          "against each game's timeline — four independent checks from one\n"
          "download. Re-run poc_report.py with a different --timeline to do it.")


if __name__ == "__main__":
    main()
