"""Backfill historical referee logs (getListLogs) for every archived match.

The open BFF serves the complete referee log — every rally's server,
receiver, outcome and timestamp — for completed matches (discovered
2026-07-16; schemas + log_type enum in recon.md "getListLogs"). This
harvester walks every unique match in data/games.csv (+ singles) at a
polite pace and caches each log to raw/match_logs/ forever (completed
matches are immutable). ~30.5k matches ≈ one overnight run; fully
resumable — re-running skips everything already cached.

Usage:
    python scraper/harvest_logs.py                   # harvest everything missing
    python scraper/harvest_logs.py --dry-run         # count what would be fetched
    python scraper/harvest_logs.py --limit 100       # bounded slice
    python scraper/harvest_logs.py --since 2026-01-01 --doubles-only
    python scraper/harvest_logs.py --summarize       # offline: raw/ → data/*.csv

Harvest order is newest-first so partial runs are maximally useful.
Matches inside pb_api's volatile window (last 3 days) are skipped —
their logs may still be growing; the next run picks them up.

--summarize (no network) tallies every cached log into:
    data/match_rally_summary.csv   one row per match: rally/point counts,
                                   serve-rally win rate, score validation
    data/player_serve_rallies.csv  (player, discipline, tour, year):
                                   serve rallies + wins AND return rallies +
                                   wins → full serve/return splits per player
                                   (return is team-attributed in doubles; see
                                   summarize() for the double-count caveat)
It also prints the aggregate serve-rally win rate k by tour — the
empirical replacement for the 0.35–0.45 the win-prob DP assumes.

Attribution rule (verified against a hand-decoded game, recon.md): a
rally is opened by a type-12 row carrying that rally's server/receiver;
the next 14/16/23 row is its outcome (point / side-out / second server).
Type 16/23 rows themselves name the POST-transition server — never
attribute by them. All archived games are side-out (games.csv contains
sideout_11/sideout_15 only), so every point belongs to the rally server.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_api import PBClient, RAW, VOLATILE_DAYS

log = logging.getLogger("logs")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RALLY, POINT, SIDEOUT, SECOND = 12, 14, 16, 23


def load_matches(doubles=True, singles=True):
    """Unique non-forfeit matches with metadata, newest first."""
    seen = {}
    if doubles:
        for r in csv.DictReader(open(DATA / "games.csv")):
            if r["is_forfeit"] == "True":
                continue
            m = r["match_id"].lower()
            if m not in seen:
                seen[m] = {"match_id": m, "discipline": "doubles",
                           "tour": r["tour"], "date": r["date"],
                           "sides": ({r["t1_p1"].lower(), r["t1_p2"].lower()},
                                     {r["t2_p1"].lower(), r["t2_p2"].lower()}),
                           "games": {}}
            seen[m]["games"][int(r["game_number"])] = (int(r["t1_score"]),
                                                       int(r["t2_score"]))
    if singles:
        for r in csv.DictReader(open(DATA / "singles_games.csv")):
            if r["is_forfeit"] == "True":
                continue
            m = r["match_id"].lower()
            if m not in seen:
                seen[m] = {"match_id": m, "discipline": "singles",
                           "tour": "PPA", "date": r["date"],
                           "sides": ({r["p1"].lower()}, {r["p2"].lower()}),
                           "games": {}}
            seen[m]["games"][int(r["game_number"])] = (int(r["s1"]), int(r["s2"]))
    return sorted(seen.values(), key=lambda x: x["date"], reverse=True)


# ---------- harvest ----------

def harvest(matches, interval, limit, dry_run, max_hours=None):
    cutoff = (dt.date.today() - dt.timedelta(days=VOLATILE_DAYS)).isoformat()
    todo = []
    for m in matches:
        if m["date"] >= cutoff:
            continue
        p = RAW / "match_logs" / m["match_id"][:2] / f"{m['match_id']}.json"
        if not p.exists():
            todo.append(m)
    if limit:
        todo = todo[:limit]
    eta_h = len(todo) * interval / 3600
    log.info("cached: %d | to fetch: %d (~%.1f h at %.1fs)",
             len(matches) - len(todo), len(todo), eta_h, interval)
    if dry_run or not todo:
        return

    stop = {"flag": False}
    def _sig(signum, frame):
        stop["flag"] = True
        log.info("signal received — finishing current match then exiting")
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    c = PBClient()
    t0, done, empty, errs = time.monotonic(), 0, 0, 0
    deadline = t0 + max_hours * 3600 if max_hours else None
    for m in todo:
        if stop["flag"]:
            break
        if deadline and time.monotonic() > deadline:
            log.info("runtime cap (%.1f h) reached — exiting cleanly; "
                     "re-run resumes where this left off", max_hours)
            break
        tick = time.monotonic()
        try:
            data = c.match_logs(m["match_id"])
            done += 1
            if not data:
                empty += 1
        except Exception as e:
            errs += 1
            log.warning("error on %s (%s %s): %s — will retry next run",
                        m["match_id"], m["tour"], m["date"], e)
            if errs >= 20 and errs > done:
                log.error("error rate too high — aborting; investigate first")
                break
        if done and done % 100 == 0:
            rate = done / (time.monotonic() - t0)
            log.info("%d/%d fetched (%d empty/404, %d errors) — ETA %.1f h",
                     done, len(todo), empty, errs, (len(todo) - done) / rate / 3600)
        time.sleep(max(0.0, interval - (time.monotonic() - tick)))
    log.info("harvest done: %d fetched, %d empty/404, %d errors, %d remaining",
             done, empty, errs, len(todo) - done)


# ---------- summarize (offline) ----------

def _point_delta(r):
    """Score change of a type-14 row: (delta, team_uuid).

    Neither signal is reliable alone (both observed in the wild):
    - rewind corrections garble the score STRINGS (mixed perspectives,
      e.g. 2-11 → 10-11) while the payload is clean (11→10);
    - phantom double-entries repeat the payload (+1, start None) while
      the string correctly does not move;
    - explicit undo rows have payload end_score None and only the string
      shows the -1.
    When both parse and disagree, the smaller-magnitude delta is the
    truthful one in every observed case, so that's the rule.
    """
    team = ((r.get("point_log") or {}).get("team_uuid") or "").lower()
    pd = None
    pl = r.get("point_log") or {}
    if pl.get("end_score") is not None:
        start = pl.get("start_score")
        pd = pl["end_score"] - (start if start is not None else pl["end_score"] - 1)
    sd = None
    try:
        a = r["start_score_current_game_string"].split("-")
        b = r["end_score_current_game_string"].split("-")
        sd = int(b[0]) - int(a[0])
    except (KeyError, ValueError, IndexError):
        pass
    if pd is None and sd is None:
        return 0, team
    if pd is None:
        return sd, team
    if sd is None or sd == pd:
        return pd, team
    return (sd if abs(sd) <= abs(pd) else pd), team


def side_of(uuid, sides):
    for i, s in enumerate(sides):
        if uuid in s:
            return i
    return None


def tally(rows, sides):
    """One log → (per-player serve tallies, per-game points-by-side, counts).

    sides = ({side-0 player uuids}, {side-1 player uuids}) from the archive.
    Points are attributed via the payload team_uuid; the team→side mapping
    is learned from the log's own normal points (in side-out scoring the
    scoring team IS the serving team, and rally rows name the serving
    player). Corrections retract the win from that team's last credited
    server.
    """
    rows = sorted(rows, key=lambda x: x.get("log_index", 0))

    def confirmed_by_next(i):
        """Disambiguate 'payload +1, own string unmoved': a REAL point with
        a stale string is confirmed by the next row carrying the score
        forward incremented; a phantom double-entry is not. Side-outs flip
        the string perspective (server score becomes component 2)."""
        try:
            cur0 = int(rows[i]["start_score_current_game_string"].split("-")[0])
        except (KeyError, ValueError, IndexError):
            return False
        for nxt in rows[i + 1:]:
            try:
                parts = [int(x) for x in
                         nxt["start_score_current_game_string"].split("-")]
            except (KeyError, ValueError, IndexError):
                continue
            comp = 1 if nxt.get("log_type") == SIDEOUT else 0
            return parts[comp] == cur0 + 1
        return False

    team_side = {}
    for r in rows:                       # pass 1: learn team → side
        if r.get("log_type") == POINT:
            delta, team = _point_delta(r)
            side = side_of((r.get("server_uuid") or "").lower(), sides)
            if delta > 0 and team and side is not None:
                team_side.setdefault(team, side)

    serves = defaultdict(lambda: [0, 0])   # server_uuid -> [rallies, wins]
    pts = defaultdict(lambda: [0, 0])      # game_number -> [side0, side1]
    counts = defaultdict(int)
    win_stack = defaultdict(list)          # side -> servers credited, in order
    current = None
    for i, r in enumerate(rows):           # pass 2: tally
        t = r.get("log_type")
        counts[t] += 1
        if t == RALLY:
            current = r
        elif t == POINT:
            delta, team = _point_delta(r)
            if delta == 0 and (r.get("point_log") or {}).get("end_score") \
                    is not None and confirmed_by_next(i):
                delta = 1              # stale string, real point
            server = ((current or r).get("server_uuid") or "").lower()
            side = team_side.get(team)
            if side is None:
                side = side_of(server, sides)
            if side is not None:
                pts[r.get("game_number")][side] += delta
                # scoring opened late: the string starts below the payload
                # (e.g. string 0-0 but payload 2→3) — the gap is real score
                # with no logged rallies; credit it to the side, not to k
                pl = r.get("point_log") or {}
                if delta > 0 and pl.get("start_score") is not None:
                    try:
                        s0 = int(r["start_score_current_game_string"].split("-")[0])
                        gap = pl["start_score"] - s0
                        if 0 < gap < 15:
                            pts[r.get("game_number")][side] += gap
                    except (KeyError, ValueError, IndexError):
                        pass
            if delta > 0 and current is not None and server:
                serves[server][0] += 1
                serves[server][1] += 1
                if side is not None:
                    win_stack[side].append(server)
            elif delta < 0 and side is not None:
                for _ in range(-delta):    # retract win(s), not the rally
                    who = win_stack[side].pop() if win_stack[side] else server
                    if who:
                        serves[who][1] -= 1
            current = None
        elif t in (SIDEOUT, SECOND):
            if current is not None and current.get("server_uuid"):
                serves[current["server_uuid"].lower()][0] += 1
            current = None
    return serves, pts, counts


def check_scores(m, pts):
    """Compare log-derived points per side with the archived game scores."""
    if not m["games"]:
        return "no-games"
    for game, expected in m["games"].items():
        if tuple(pts.get(game, [0, 0])) != expected:
            return "mismatch"
    return "ok"


def summarize(matches, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_match, per_player = [], defaultdict(lambda: [0, 0, 0, 0])
    k_num, k_den = defaultdict(int), defaultdict(int)
    missing = 0
    for m in matches:
        p = RAW / "match_logs" / m["match_id"][:2] / f"{m['match_id']}.json"
        if not p.exists():
            missing += 1
            continue
        body = json.loads(p.read_text())
        rows = body.get("data") if isinstance(body, dict) else body
        if not rows:
            per_match.append({**{k: m[k] for k in ("match_id", "discipline", "tour", "date")},
                              "n_logs": 0, "n_rallies": 0, "n_points": 0,
                              "n_sideouts": 0, "n_second": 0, "k_match": "",
                              "score_check": "empty" if rows == [] else "404"})
            continue
        serves, pts, counts = tally(rows, m["sides"])
        rallies = sum(v[0] for v in serves.values())
        wins = sum(v[1] for v in serves.values())
        key = (m["discipline"], m["tour"])
        k_num[key] += wins
        k_den[key] += rallies
        year = m["date"][:4]
        for s, (n, w) in serves.items():
            agg = per_player[(s, m["discipline"], m["tour"], year)]
            agg[0] += n
            agg[1] += w
        # Return rallies, reconstructed without a receiver field: a side's
        # return rallies ARE the opposing side's serve rallies, and the side
        # WINS the ones the opponent lost (every side-out is a return-team
        # win). Singles sides are one player, so this is an exact individual
        # split; doubles credits BOTH partners with the full team total — the
        # right denominator for a per-player return rate, but it double-counts
        # if you sum return_rallies across a doubles team, so never do that.
        side_sr = [0, 0]
        side_sw = [0, 0]
        for i, s in enumerate(m["sides"]):
            for u in s:
                sn, sw = serves.get(u, (0, 0))
                side_sr[i] += sn
                side_sw[i] += sw
        for i, s in enumerate(m["sides"]):
            opp = 1 - i
            rr, rw = side_sr[opp], side_sr[opp] - side_sw[opp]
            for u in s:
                agg = per_player[(u, m["discipline"], m["tour"], year)]
                agg[2] += rr
                agg[3] += rw
        per_match.append({**{k: m[k] for k in ("match_id", "discipline", "tour", "date")},
                          "n_logs": len(rows), "n_rallies": counts[RALLY],
                          "n_points": counts[POINT], "n_sideouts": counts[SIDEOUT],
                          "n_second": counts[SECOND],
                          "k_match": f"{wins / rallies:.3f}" if rallies else "",
                          "score_check": check_scores(m, pts)})
    mp = out_dir / "match_rally_summary.csv"
    with mp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_match[0].keys()))
        w.writeheader()
        w.writerows(per_match)
    pp = out_dir / "player_serve_rallies.csv"
    with pp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_uuid", "discipline", "tour", "year",
                    "serve_rallies", "serve_wins",
                    "return_rallies", "return_wins"])
        for (s, disc, tour, year), (sr, sw, rr, rw) in sorted(per_player.items()):
            w.writerow([s, disc, tour, year, sr, sw, rr, rw])
    ok = sum(1 for r in per_match if r["score_check"] == "ok")
    bad = sum(1 for r in per_match if r["score_check"] == "mismatch")
    emp = sum(1 for r in per_match if r["score_check"] in ("empty", "404"))
    log.info("summarized %d matches (%d not yet harvested): "
             "%d score-validated ok, %d mismatch, %d empty/404 → %s, %s",
             len(per_match), missing, ok, bad, emp, mp, pp)
    for key in sorted(k_den):
        if k_den[key]:
            log.info("k[%s %s] = %d/%d = %.3f",
                     key[0], key[1], k_num[key], k_den[key],
                     k_num[key] / k_den[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.1,
                    help="seconds per request (>=1.0 enforced)")
    ap.add_argument("--limit", type=int, help="fetch at most N matches")
    ap.add_argument("--since", help="only matches on/after YYYY-MM-DD")
    ap.add_argument("--until", help="only matches on/before YYYY-MM-DD")
    ap.add_argument("--doubles-only", action="store_true")
    ap.add_argument("--singles-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-hours", type=float,
                    help="stop cleanly after this many hours (resumable)")
    ap.add_argument("--summarize", action="store_true",
                    help="no network: tally raw/match_logs → data/*.csv")
    ap.add_argument("--out", default=str(DATA),
                    help="summary output dir (default data/)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.interval < 1.0:
        raise SystemExit("interval below 1s is impolite; refusing")
    matches = load_matches(doubles=not args.singles_only,
                           singles=not args.doubles_only)
    if args.since:
        matches = [m for m in matches if m["date"] >= args.since]
    if args.until:
        matches = [m for m in matches if m["date"] <= args.until]
    if args.summarize:
        summarize(matches, Path(args.out))
    else:
        harvest(matches, args.interval, args.limit, args.dry_run,
                max_hours=args.max_hours)


if __name__ == "__main__":
    main()
