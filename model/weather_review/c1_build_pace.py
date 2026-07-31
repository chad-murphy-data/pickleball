"""Build the per-game PACE table (duration, points, game-hour weather).

    python model/weather_review/c1_build_pace.py

Writes /tmp/.../scratchpad/c1_pace.csv and prints a defect audit.
Nothing is written into the repo's data/ directory.

Timestamp semantics (verified here, not assumed):
  * `g1_end_utc..g5_end_utc` in data/match_times.csv are TRUE UTC.
  * `start_local` / `completed_local` are venue-LOCAL wall clock carrying a
    spurious trailing 'Z'. The feed's own `tz_abbrev` is WRONG for a
    handful of events (every Phoenix event is stamped MDT although
    Arizona does not observe DST; the Jan-2024 Palm Springs event is
    stamped PDT). So instead of trusting tz_abbrev we CALIBRATE the
    offset per event from median(completed_local - last game end),
    rounded to the hour. Games 2+ never need this: their duration is a
    difference of two true-UTC stamps.

Duration definitions:
  game n>=2 : end(n) - end(n-1)   [includes the ~2 min between-game break]
  game 1    : end(1) - match start (calibrated-offset local -> UTC)
A `first_game` flag lets the regressions absorb the different overhead.
"""
from __future__ import annotations

import csv
import datetime as dt
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_lib import (ROOT, event_tz, game_eta, get_tz, label_arms,  # noqa: E402
                    load_games, load_hourly, load_v2, local_hour_key,
                    parse_utc, read_csv)

OUT = Path("/tmp/claude-0/-home-user-pickleball/"
           "a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad/c1_pace.csv")

END_COLS = ["g1_end_utc", "g2_end_utc", "g3_end_utc", "g4_end_utc", "g5_end_utc"]


def naive_local(s):
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        return None


def calibrate_offsets(mt_rows, tzs):
    """event_id -> timedelta correcting the feed's local stamps.

    The feed's local wall clock should equal UTC converted with the venue
    tz. Where it does not (Phoenix, a couple of stale DST abbrevs) we
    recover an integer-hour correction from completed_local vs the last
    true-UTC game end.
    """
    resid = defaultdict(list)
    for r in mt_rows:
        tz = get_tz(tzs.get(r["event_id"], ""))
        if tz is None:
            continue
        ends = [parse_utc(r[c]) for c in END_COLS]
        ends = [e for e in ends if e]
        comp = naive_local(r["completed_local"])
        if not ends or comp is None:
            continue
        # what the local clock SHOULD read at the last game end
        want = ends[-1].astimezone(tz).replace(tzinfo=None)
        resid[r["event_id"]].append((comp - want).total_seconds())
    off = {}
    for ev, vs in resid.items():
        if len(vs) < 5:
            continue
        m = statistics.median(vs)
        off[ev] = dt.timedelta(hours=round(m / 3600.0))
    return off


def main():
    mt_rows = read_csv(ROOT / "data/match_times.csv")
    mt = {r["match_id"]: r for r in mt_rows}
    tzs = event_tz()
    offsets = calibrate_offsets(mt_rows, tzs)
    nz = {e: o for e, o in offsets.items() if o.total_seconds()}
    print(f"calibrated local-clock offsets: {len(offsets)} events, "
          f"{len(nz)} needed a non-zero correction "
          f"({sorted({int(o.total_seconds()//3600) for o in nz.values()})} h)")

    hourly = load_hourly()
    arms = label_arms()
    v2 = load_v2()
    by_match = load_games()

    defect = Counter()
    rows = []
    for mid, gs in by_match.items():
        t = mt.get(mid)
        if not t:
            defect["no match_times row"] += 1
            continue
        ev = gs[0]["event_id"]
        tzname = tzs.get(ev)
        tz = get_tz(tzname or "")
        if tz is None:
            defect["no usable timezone"] += 1
            continue
        ends = {}
        for i, c in enumerate(END_COLS, 1):
            ts = parse_utc(t[c])
            if ts:
                ends[i] = ts
        if not ends:
            defect["no game-end stamps"] += 1
            continue
        start = None
        sl = naive_local(t["start_local"])
        if sl is not None and ev in offsets:
            start = (sl - offsets[ev]).replace(tzinfo=tz).astimezone(dt.timezone.utc)
        for g in gs:
            gn = int(g["game_number"])
            if gn not in ends:
                defect["game end missing"] += 1
                continue
            if gn - 1 in ends:
                dur = (ends[gn] - ends[gn - 1]).total_seconds()
                first = 0
            elif gn == 1 and start is not None:
                dur = (ends[gn] - start).total_seconds()
                first = 1
            else:
                defect["no usable start for game 1"] += 1
                continue
            pts = int(g["t1_score"]) + int(g["t2_score"])
            if pts <= 0:
                defect["zero points"] += 1
                continue
            hk = local_hour_key(ends[gn], tzname)
            wx = hourly.get((ev, hk))
            if wx is None or wx["wind"] is None:
                defect["no hourly weather"] += 1
                continue
            eta = game_eta(g, v2)
            rows.append({
                "game_id": g["game_id"], "match_id": mid, "event_id": ev,
                "date": g["date"], "tour": g["tour"],
                "fmt": g["scoring_format"], "game_number": gn,
                "best_of": g["best_of"], "first_game": first,
                "dur": round(dur, 1), "points": pts,
                "margin": abs(int(g["margin"])),
                "wind": wx["wind"], "gust": wx["gust"], "temp": wx["temp"],
                "precip": wx["precip"] or 0.0,
                "eta": "" if eta is None else round(eta, 6),
                "published": arms["published"].get(ev) or "",
                "corrected_all": arms["corrected_all"].get(ev) or "",
                "corrected_hi": arms["corrected_hi"].get(ev) or "",
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} game-duration rows -> {OUT}")

    print("\nDROPS")
    for k, v in defect.most_common():
        print(f"  {k:28s} {v}")

    def qs(vals, label, fmt="%.1f"):
        vals = sorted(vals)
        n = len(vals) - 1
        pts = [vals[int(p * n)] for p in (0, .01, .05, .5, .95, .99, 1)]
        print(label + "  " + "  ".join(fmt % v for v in pts))

    print("\n%-28s %s" % ("", "min      p1      p5     p50     p95     p99     max"))
    for f in (0, 1):
        sub = [r for r in rows if r["first_game"] == f]
        tag = "game 1" if f else "games 2+"
        qs([r["dur"] for r in sub], "%-22s n=%5d" % ("DURATION s " + tag, len(sub)),
           "%7.0f")
        qs([r["dur"] / r["points"] for r in sub],
           "%-22s n=%5d" % ("SEC/POINT " + tag, len(sub)), "%7.1f")

    print("\nlabel arm corrected_all:",
          dict(Counter(r["corrected_all"] or "dropped" for r in rows)))
    print("tour:", dict(Counter(r["tour"] for r in rows)))
    print("format:", dict(Counter(r["fmt"] for r in rows)))
    print("year:", dict(Counter(r["date"][:4] for r in rows)))
    print("events:", len({r["event_id"] for r in rows}))
    o = [r for r in rows if r["corrected_all"] == "outdoor"]
    ws = sorted(r["wind"] for r in o)
    print("outdoor(corrected) n=%d  wind p50 %.1f p90 %.1f p99 %.1f max %.1f;"
          " >=14mph %d  >=20mph %d"
          % (len(ws), ws[len(ws) // 2], ws[int(.9 * len(ws))],
             ws[int(.99 * len(ws))], ws[-1],
             sum(1 for w in ws if w >= 14), sum(1 for w in ws if w >= 20)))


if __name__ == "__main__":
    main()
