"""Step 0 of the vision POC — the sync spine.

    python vision/rally_timeline.py --match <uuid>
    python vision/rally_timeline.py --pick columbus-womens     # named POC match

Pulls the referee log for one match off the open BFF and emits a per-rally
table: index, start/end wall-clock, duration, score string, server,
receiver and outcome.  Nothing here touches video — this is the TARGET the
video gets aligned to, and it is the reason the POC needs almost no hand
labelling: every downstream detector can be scored against it.

What the log actually gives (verified 2026-08-09 on MLP Orlando):
  * log_type 12  = RALLY START
  * log_type 14/16/23 = RALLY END (point / side-out / second server)
  and they interleave perfectly, so every rally has a [start, end] window.

CAUTION 1, and it shapes everything downstream: these timestamps are when a
REFEREE PRESSED A BUTTON.  1-second resolution, lagging the real event by
human reaction time, so they are good to about +/-1-2 s.  That is plenty to
WINDOW the video (rallies run ~14-20 s with a few seconds between) and
useless for labelling individual contacts.  Window with the log; find
contacts inside the window with audio/vision.

CAUTION 2 — REFEREE LOGGING STYLE IS BIMODAL (measured 2026-08-09 over nine
MLP 2026 matches, one per event).  Type 12's timing is a workflow artifact,
not a guaranteed rally start:

    informative (96-100% of rallies, median lead 16-20 s)
        Austin, Chicago, Mid-Season, New York, San Diego
    batch-entered (0-5%, lead 0 s — start and outcome logged in the same
    second, several rallies at a time)
        Columbus, St. Petersburg, Dallas, St. Louis

So the script measures the style per match and picks its windowing:
  * informative  -> window = [rally start, rally end].  Leaves real DEAD
    TIME between rallies, which is what lets the audio check validate
    itself with no hand labels.
  * batch        -> window = [previous rally end, this rally end].  Tiles
    the match with no dead time, so the density contrast is unavailable and
    validation has to lean on the negative controls instead.
PREFER AN INFORMATIVE MATCH FOR THE POC. The pick below does.

Output: data/vision/rally_timeline_<match8>.csv  (+ _meta.json)
Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "vision"

BFF = "https://pickleball.com/api/v1/results/getListLogs?id={}"
RALLY_START = 12
RALLY_END = {14: "point", 16: "sideout", 23: "second"}
TIMEOUT_LOG = 18          # useful negative control: no contacts during these
GAME_OVER, MATCH_OVER = 4, 6

# Curated POC matches. Rationale in vision/README.md.
PICKS = {
    # THE POC PICK. Women's (most dinking -> the interval histogram's best
    # chance to show two modes), 74 rallies (near the densest in MLP 2026),
    # referee style INFORMATIVE at 96% with a 20 s median lead (so real rally
    # windows and real dead time), and a marquee event so it is among the
    # likeliest to have a clean featured-court upload.
    "midseason-womens": dict(date="2026-07-08",
                             event="Edward Jones Mid-Season",
                             context="womens"),
    # Backup, same properties, men's: 81 rallies, 96% informative, 16 s lead.
    "austin-mens": dict(date="2026-06-13", event="MLP Austin", context="mens"),
    # The coverage-dial / freeze-out target for LATER, once the pipeline
    # works: widest within-gender gap in MLP 2026 women's doubles (0.90).
    # NB check its referee style before relying on rally windows.
    "newyork-gap": dict(date="2026-06-28", event="MLP New York",
                        context="womens"),
}


def resolve_pick(name):
    """Find the match_id for a curated pick, choosing the longest game."""
    want = PICKS[name]
    rally = {r["match_id"]: int(r["n_rallies"] or 0)
             for r in csv.DictReader((DATA / "match_rally_summary.csv").open())}
    best = None
    for g in csv.DictReader((DATA / "games.csv").open()):
        if (g["date"] != want["date"] or g["context"] != want["context"]
                or want["event"] not in g["event_name"]):
            continue
        n = rally.get(g["match_id"], 0)
        if best is None or n > best[0]:
            best = (n, g)
    if best is None:
        sys.exit(f"pick {name!r} matched no game in games.csv")
    return best[1]["match_id"], best[1]


def fetch(match_id):
    req = urllib.request.Request(
        BFF.format(match_id),
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)["data"]


def parse_ts(row):
    return dt.datetime.fromisoformat(row["date_created"].replace("Z", "+00:00"))


def build(log):
    """Pair each RALLY_END with the RALLY_START that precedes it, then decide
    whether the start timestamps are informative and set the window mode.

    Returns (rows, mode, informative_fraction, median_lead_s).
    """
    rows = sorted(log, key=lambda r: r["log_index"])
    out, pending, game = [], None, 1
    for r in rows:
        t = r["log_type"]
        if t == RALLY_START:
            pending = r
        elif t in RALLY_END:
            src = pending if pending is not None else r
            out.append({
                "rally": len(out) + 1,
                "game": game,
                "t_logged_start": parse_ts(src).isoformat(),
                "t_end": parse_ts(r).isoformat(),
                "lead_s": (parse_ts(r) - parse_ts(src)).total_seconds(),
                "start_score": src.get("start_score_current_game_string"),
                "end_score": r.get("end_score_current_game_string"),
                "outcome": RALLY_END[t],
                "server_uuid": (src.get("server_uuid") or "").lower(),
                "receiver_uuid": (src.get("receiver_uuid") or "").lower(),
            })
            pending = None
        elif t == GAME_OVER:
            game += 1
    if not out:
        return out, "none", 0.0, 0.0

    leads = sorted(r["lead_s"] for r in out)
    frac = sum(1 for r in out if r["lead_s"] > 2) / len(out)
    med = leads[len(leads) // 2]
    mode = "start-marked" if frac >= 0.5 else "end-to-end"

    for i, r in enumerate(out):
        if mode == "start-marked":
            r["t_start"] = r["t_logged_start"]
        else:
            # no usable start marker: the rally happened between the previous
            # outcome and this one.
            r["t_start"] = out[i - 1]["t_end"] if i else r["t_logged_start"]
        r["duration_s"] = (dt.datetime.fromisoformat(r["t_end"])
                           - dt.datetime.fromisoformat(r["t_start"])).total_seconds()
    for i in range(len(out) - 1):
        a = dt.datetime.fromisoformat(out[i]["t_end"])
        b = dt.datetime.fromisoformat(out[i + 1]["t_start"])
        out[i]["gap_after_s"] = (b - a).total_seconds()
    out[-1]["gap_after_s"] = ""
    return out, mode, frac, med


def timeouts(log):
    """Timeout windows — a free NEGATIVE control: contact detectors should
    find essentially nothing inside them."""
    return [parse_ts(r).isoformat() for r in log if r["log_type"] == TIMEOUT_LOG]


def survey_matchup(teams, date=None):
    """Vet a candidate VOD before downloading it.

    Referee style turns out to vary BY MATCH, not by event — two courts at
    the same tournament can log completely differently — so a matchup found
    on YouTube has to be checked before you trust its rally windows.
    """
    import time
    want = {t.strip().lower()
            for t in teams.replace(" vs ", " v ").split(" v ")}
    rows = [r for r in csv.DictReader((DATA / "mlp_matchups_2026.csv").open())
            if {r["team_one"].lower(), r["team_two"].lower()} == want
            and (date is None or r["date"] == date)]
    if not rows:
        sys.exit(f"no 2026 matchup found for {teams!r}"
                 + (f" on {date}" if date else ""))
    games = {g["match_id"]: g for g in csv.DictReader((DATA / "games.csv").open())}
    rally = {r["match_id"]: int(r["n_rallies"] or 0)
             for r in csv.DictReader((DATA / "match_rally_summary.csv").open())}
    seen = set()
    for r in sorted(rows, key=lambda r: (r["date"], r["game_slot"])):
        if r["matchup_id"] in seen and len(rows) > 8:
            continue
        g = games.get(r["match_id"])
        if not g:
            continue
        try:
            _, mode, frac, med = build(fetch(r["match_id"]))
        except Exception as exc:
            print(f"  {r['date']} slot {r['game_slot']}: fetch failed ({exc})")
            continue
        flag = "USABLE" if mode == "start-marked" else "degenerate windows"
        print(f"  {r['date']} slot {r['game_slot']} {g['context']:7s} "
              f"{rally.get(r['match_id'], 0):3d} rallies  {mode:12s} "
              f"({frac:.0%}, lead {med:.0f}s)  {flag}")
        print(f"      match_id {r['match_id']}")
        time.sleep(1.1)
    print("\n  Only 'start-marked' games leave real dead time between rallies,\n"
          "  which is what lets the contact check validate itself with no labels.")


def build_matchup(teams, date=None):
    """One timeline spanning every game of a matchup — the right target for a
    full-matchup VOD, which is all MLP publishes.

    This is an UPGRADE over a single game, not a compromise. All games share
    one absolute (UTC) clock, so a continuous VOD needs exactly ONE offset to
    map video time onto all four. Fitting the offset per game then gives four
    estimates that must agree — which is a free cross-validation of the sync
    method that a single game cannot provide. If they disagree, the
    disagreements are the broadcast's edit points, which is also worth
    knowing.

    Bonus: the changeovers between games are long stretches of genuinely
    dead time, much better negative controls than the 1-4 s gaps between
    rallies inside a game.
    """
    import time
    want = {t.strip().lower()
            for t in teams.replace(" vs ", " v ").split(" v ")}
    rows = [r for r in csv.DictReader((DATA / "mlp_matchups_2026.csv").open())
            if {r["team_one"].lower(), r["team_two"].lower()} == want
            and (date is None or r["date"] == date)]
    if not rows:
        sys.exit(f"no 2026 matchup found for {teams!r}")
    dates = sorted({r["date"] for r in rows})
    if len(dates) > 1:
        sys.exit(f"{teams} played on {dates} — pass --date to choose")
    games = {g["match_id"]: g for g in csv.DictReader((DATA / "games.csv").open())}
    names = {r["player_id"].lower(): r["full_name"]
             for r in csv.DictReader((DATA / "v2_players.csv").open())}

    allr, meta_games, modes = [], [], []
    for r in sorted(rows, key=lambda r: int(r["game_slot"])):
        g = games.get(r["match_id"])
        if not g:
            continue
        rl, mode, frac, med = build(fetch(r["match_id"]))
        if not rl:
            print(f"  slot {r['game_slot']}: no rally rows, skipped")
            continue
        for x in rl:
            x["slot"] = r["game_slot"]
            x["match_id"] = r["match_id"]
        allr.extend(rl)
        modes.append(mode)
        meta_games.append({
            "slot": r["game_slot"], "match_id": r["match_id"],
            "context": g["context"], "score": f"{g['t1_score']}-{g['t2_score']}",
            "n_rallies": len(rl), "window_mode": mode,
            "informative_fraction": frac, "median_lead_s": med,
            "t_start": rl[0]["t_start"], "t_end": rl[-1]["t_end"],
            "players": [names.get((g[k] or "").lower(), "?")
                        for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]})
        time.sleep(1.1)
    if not allr:
        sys.exit("no rallies found for this matchup")

    allr.sort(key=lambda x: x["t_start"])
    for i, x in enumerate(allr):
        x["rally"] = i + 1

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"matchup_{dates[0].replace('-', '')}_{rows[0]['matchup_id'][:8]}"
    cols = ["rally", "slot", "match_id", "game", "t_start", "t_end",
            "duration_s", "gap_after_s", "lead_s", "start_score", "end_score",
            "outcome", "server_uuid", "receiver_uuid"]
    with (OUT / f"rally_timeline_{stem}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for x in allr:
            w.writerow({c: x.get(c, "") for c in cols})
    meta = {"matchup": teams, "date": dates[0], "games": meta_games,
            "n_rallies": len(allr),
            "all_start_marked": all(m == "start-marked" for m in modes)}
    (OUT / f"rally_timeline_{stem}_meta.json").write_text(json.dumps(meta, indent=2))

    t0 = dt.datetime.fromisoformat(allr[0]["t_start"])
    print(f"matchup {teams}  {dates[0]}")
    for m in meta_games:
        rel = (dt.datetime.fromisoformat(m["t_start"]) - t0).total_seconds()
        print(f"  slot {m['slot']} {m['context']:7s} {m['n_rallies']:3d} rallies "
              f"{m['score']:>6s}  {m['window_mode']:12s}  starts +{rel/60:5.1f} min")
        print(f"          {' / '.join(p.split()[-1] for p in m['players'])}")
    span = (dt.datetime.fromisoformat(allr[-1]["t_end"]) - t0).total_seconds()
    live = sum(x["duration_s"] for x in allr)
    print(f"  TOTAL {len(allr)} rallies over {span/60:.1f} min, "
          f"{live/span:.0%} live")
    if not meta["all_start_marked"]:
        print("  WARNING: not every game is start-marked — the density check "
              "will be weaker for those")
    print(f"  wrote {(OUT / f'rally_timeline_{stem}.csv').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", help="match uuid")
    ap.add_argument("--pick", choices=sorted(PICKS), help="curated POC match")
    ap.add_argument("--teams", help='"Team A v Team B" — report the referee '
                                    "style of every game in that matchup, so a "
                                    "candidate VOD can be vetted before you "
                                    "download it")
    ap.add_argument("--matchup", help='"Team A v Team B" — build ONE combined '
                                      "timeline covering every game in the "
                                      "matchup. This is what to use with a "
                                      "full-matchup VOD.")
    ap.add_argument("--date", help="YYYY-MM-DD, narrows --teams/--matchup")
    args = ap.parse_args()
    if args.teams:
        survey_matchup(args.teams, args.date)
        return
    if args.matchup:
        build_matchup(args.matchup, args.date)
        return
    if not args.match and not args.pick:
        ap.error("give --match, --pick, --matchup or --teams")

    meta = {}
    if args.pick:
        match_id, g = resolve_pick(args.pick)
        meta = {k: g[k] for k in ("date", "event_name", "context", "stage",
                                  "t1_score", "t2_score", "scoring_format")}
        meta["players"] = [g[k] for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
    else:
        match_id = args.match

    log = fetch(match_id)
    rallies, mode, frac_inf, med_lead = build(log)
    if not rallies:
        sys.exit("no rally rows found — event may predate digital refereeing")

    names = {}
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        names[r["player_id"].lower()] = r["full_name"]
    meta.update(match_id=match_id, n_log_rows=len(log), n_rallies=len(rallies),
                log_types=dict(Counter(r["log_type"] for r in log)),
                timeouts=timeouts(log), window_mode=mode,
                informative_start_fraction=frac_inf, median_lead_s=med_lead,
                t0=rallies[0]["t_start"], t_last=rallies[-1]["t_end"],
                player_names=[names.get((p or "").lower(), "?")
                              for p in meta.get("players", [])])

    OUT.mkdir(parents=True, exist_ok=True)
    stem = match_id[:8]
    cols = ["rally", "game", "t_start", "t_end", "duration_s", "gap_after_s",
            "lead_s", "start_score", "end_score", "outcome",
            "server_uuid", "receiver_uuid"]
    with (OUT / f"rally_timeline_{stem}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rallies:
            w.writerow({c: r.get(c, "") for c in cols})
    (OUT / f"rally_timeline_{stem}_meta.json").write_text(json.dumps(meta, indent=2))

    durs = sorted(r["duration_s"] for r in rallies)
    gaps = sorted(r["gap_after_s"] for r in rallies if r["gap_after_s"] != "")
    span = (dt.datetime.fromisoformat(rallies[-1]["t_end"])
            - dt.datetime.fromisoformat(rallies[0]["t_start"])).total_seconds()
    print(f"match {match_id}")
    if meta.get("player_names"):
        print("  " + " / ".join(meta["player_names"]) + f"   {meta.get('date','')}"
              f"  {meta.get('event_name','')}")
    print(f"  {len(rallies)} rallies over {span/60:.1f} min "
          f"({len(log)} log rows)")
    print(f"  referee style   {mode}  ({frac_inf:.0%} of rallies carry a "
          f"start marker >2s before the outcome, median lead {med_lead:.0f}s)")
    print(f"  rally duration  median {durs[len(durs)//2]:.0f}s  "
          f"p10 {durs[len(durs)//10]:.0f}s  p90 {durs[9*len(durs)//10]:.0f}s")
    if gaps:
        print(f"  dead time       median {gaps[len(gaps)//2]:.0f}s  "
              f"p90 {gaps[9*len(gaps)//10]:.0f}s")
    live = sum(r["duration_s"] for r in rallies)
    print(f"  LIVE fraction   {live/span:.0%} of the match window is inside a "
          f"rally")
    if mode == "start-marked":
        print("                  ^ the audio check can use density inside vs "
              "outside these windows")
    else:
        print("                  ^ windows TILE the match (no start markers), so "
              "there is no dead-time\n                    contrast here — "
              "validation must lean on the timeout negative control")
    print(f"  wrote {(OUT / f'rally_timeline_{stem}.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
