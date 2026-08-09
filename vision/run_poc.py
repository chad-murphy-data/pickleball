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

# Known full-matchup VODs. MLP only publishes whole matchups, which is an
# UPGRADE, not a limitation: all four games share one absolute clock, so the
# offset can be fitted per game and the four estimates must agree — a
# cross-validation of the sync a single game cannot give you.
VODS = {
    "chicago-0725": dict(
        url="https://www.youtube.com/watch?v=QOhu67FAeY4",
        matchup="Chicago Slice v Utah Black Diamonds", date="2026-07-25",
        note="all four games start-marked (100/100/100/98%), 193 rallies, "
             "no DreamBreaker — the whole VOD is logged"),
}


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        sys.exit(f"failed: {cmd[0]}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vod", choices=sorted(VODS),
                    help="a known matchup VOD — supplies url, teams and date")
    ap.add_argument("--url", help="YouTube URL of the matchup VOD")
    ap.add_argument("--matchup", help='"Team A v Team B" (with --date)')
    ap.add_argument("--date", help="YYYY-MM-DD")
    ap.add_argument("--pick", help="curated single game (see rally_timeline.py)")
    ap.add_argument("--match", help="match uuid")
    ap.add_argument("--check", help='matchup as "Team A v Team B" — just report '
                                    "whether its games have usable rally windows")
    ap.add_argument("--audio", default=str(OUT / "match_audio.m4a"))
    ap.add_argument("--k", type=float, default=4.0)
    args = ap.parse_args()
    py = sys.executable

    if args.check:
        run([py, str(ROOT / "vision" / "rally_timeline.py"),
             "--teams", args.check] + (["--date", args.date] if args.date else []))
        return

    url, matchup, date = args.url, args.matchup, args.date
    if args.vod:
        v = VODS[args.vod]
        url = url or v["url"]
        matchup, date = matchup or v["matchup"], date or v["date"]
        print(f"VOD {args.vod}: {v['matchup']} {v['date']}\n  {v['note']}")
    if not url or not (matchup or args.pick or args.match):
        ap.error("need a URL plus one of --vod/--matchup/--pick/--match")

    # 1. rally timeline (network: the open BFF, works anywhere)
    if matchup:
        sel = ["--matchup", matchup] + (["--date", date] if date else [])
    elif args.pick:
        sel = ["--pick", args.pick]
    else:
        sel = ["--match", args.match]
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
