"""Replay a completed MLP matchup's referee logs into live win-prob charts.

The display prototype for ROADMAP Pillar 5: per-rally win probability from
the serve-aware DP (sitelib/winprob.py), anchored to the same calibrated
pre-match probabilities the receipts ledger grades, rendered as one
self-contained HTML file (inline SVG, no dependencies).

Usage:
    python web/replay_winprob.py --matchup <uuid> [--out charts.html]
                                 [--k 0.43] [--p-db 0.5]

Reads rally logs via the open BFF (getListLogs; cached in raw/match_logs),
lineups + results from getResultsMatchupData, values from
data/v2_players.csv. Works for any completed matchup whose courts were
digitally refereed. The matchup track combines the live current-game
probability with pre-match probabilities for games not yet played
(2-2 goes to the DreamBreaker at --p-db for team one).

This is a REPLAY tool — the real-time version is the same engine fed by
the Tier-1 poller / Tier-2 SSE feed on the droplet.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "scraper"))

from sitelib.race import (calibrate, race_dist, set_calibration, sigmoid,
                          team_eta)                          # noqa: E402
from sitelib.winprob import (A1, A2, B1, B2, K_DOUBLES, ServeDP,
                             display_floor, eta_anchor,
                             rally_race_p)                   # noqa: E402
from pb_api import PBClient                                  # noqa: E402

RALLY, POINT, SIDEOUT, SECOND = 12, 14, 16, 23
MARKS = {18: "T", 35: "T+", 2: "C", 37: "C", 45: "C"}   # timeout / challenge

T1_COLOR, T2_COLOR = "#d1495b", "#1d6fa5"


def load_values():
    vals = {}
    with open(ROOT / "data" / "v2_players.csv") as fh:
        for r in csv.DictReader(fh):
            vals[r["player_id"].lower()] = (r["full_name"],
                                            float(r["value_now_mean"]))
    return vals


def load_cal():
    try:
        c = json.loads((ROOT / "web" / "calibration.json").read_text())
        set_calibration(c["a"], c["b"], c["eps"])
    except FileNotFoundError:
        pass


def match_series(logs, side1, dp):
    """[(rally_i, p_raw, ts)] at each rally start + annotation marks."""
    pts, marks = [], []
    n = 0
    for r in sorted(logs, key=lambda x: x.get("log_index", 0)):
        t = r.get("log_type")
        if t in MARKS:
            team = None
            for key in ("timeout_log", "additional_timeout_log",
                        "challenge_log", "video_challenge_log",
                        "line_review_log"):
                if isinstance(r.get(key), dict):
                    team = (r[key].get("team_uuid") or "").lower()
            if pts:                      # skip pre-game admin rows
                marks.append((n, MARKS[t], team))
            continue
        if t != RALLY:
            continue
        try:
            s_srv, s_rcv, num = (int(x) for x in
                                 r["start_score_current_game_string"].split("-"))
        except (KeyError, ValueError):
            continue
        srv = (r.get("server_uuid") or "").lower()
        if srv in side1:
            a, b = s_srv, s_rcv
            state = A1 if num == 1 else A2
        else:
            a, b = s_rcv, s_srv
            state = B1 if num == 1 else B2
        n += 1
        pts.append((n, dp.p(a, b, state), r.get("date_created")))
    return pts, marks


def db_series(logs, side1, p_rally):
    pts, n = [], 0
    for r in sorted(logs, key=lambda x: x.get("log_index", 0)):
        if r.get("log_type") != RALLY:
            continue
        try:
            parts = [int(x) for x in
                     r["start_score_current_game_string"].split("-")]
            s_srv, s_rcv = parts[0], parts[1]   # DB strings have no server #
        except (KeyError, ValueError, IndexError):
            continue
        srv = (r.get("server_uuid") or "").lower()
        a, b = (s_srv, s_rcv) if srv in side1 else (s_rcv, s_srv)
        n += 1
        pts.append((n, rally_race_p(a, b, p_rally), r.get("date_created")))
    return pts, []


def matchup_prob(w1, w2, future_ps, p_db):
    if w1 >= 3:
        return 1.0
    if w2 >= 3:
        return 0.0
    if not future_ps:
        return p_db                      # 2-2 → DreamBreaker
    p = future_ps[0]
    return (p * matchup_prob(w1 + 1, w2, future_ps[1:], p_db)
            + (1 - p) * matchup_prob(w1, w2 + 1, future_ps[1:], p_db))


def svg_step(pts, marks, width=760, hgt=150, color=T1_COLOR, floor=True):
    if not pts:
        return "<p class='nolog'>no rally log for this court</p>"
    n = max(p[0] for p in pts)
    def X(i):
        return 45 + (width - 60) * i / max(n, 1)
    def Y(p):
        return 12 + (hgt - 30) * (1 - p)
    disp = [(i, display_floor(p) if floor else p) for i, p, _ in pts]
    d = f"M {X(disp[0][0]):.1f} {Y(disp[0][1]):.1f} " + " ".join(
        f"H {X(i):.1f} V {Y(p):.1f}" for i, p in disp[1:])
    ticks = "".join(
        f"<line x1='{X(i):.1f}' y1='{hgt-18}' x2='{X(i):.1f}' y2='{hgt-8}' "
        f"stroke='#888' stroke-width='1.5'><title>{lab}</title></line>"
        f"<text x='{X(i):.1f}' y='{hgt-2}' font-size='8' fill='#888' "
        f"text-anchor='middle'>{lab}</text>"
        for i, lab, _ in marks)
    grid = "".join(
        f"<line x1='45' y1='{Y(v)}' x2='{width-15}' y2='{Y(v)}' "
        f"stroke='#ddd' stroke-width='{1.4 if v == .5 else .6}'/>"
        f"<text x='40' y='{Y(v)+3}' font-size='9' fill='#999' "
        f"text-anchor='end'>{int(v*100)}%</text>"
        for v in (0.1, 0.5, 0.9))
    return (f"<svg viewBox='0 0 {width} {hgt}' width='100%'>{grid}"
            f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>"
            f"{ticks}</svg>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=float, default=K_DOUBLES)
    ap.add_argument("--p-db", type=float, default=0.5,
                    help="team-one DreamBreaker win prob (pre-match)")
    args = ap.parse_args()

    load_cal()
    vals = load_values()
    c = PBClient()
    md = c._get_json("/api/v2/results/getResultsMatchupData"
                     f"?matchupId={args.matchup}").get("data") or {}
    t1, t2 = md.get("teamOneTitle") or "Team 1", md.get("teamTwoTitle") or "Team 2"
    title = f"{md.get('matchupGroupTitle') or 'MLP'} — {t1} vs {t2}"

    games, db_game = [], None
    for m in md.get("matches") or []:
        pl1 = [(m.get("teamOnePlayerOneUuid") or "").lower(),
               (m.get("teamOnePlayerTwoUuid") or "").lower()]
        pl2 = [(m.get("teamTwoPlayerOneUuid") or "").lower(),
               (m.get("teamTwoPlayerTwoUuid") or "").lower()]
        g = {"uuid": (m.get("matchUuid") or "").lower(),
             "abbrev": m.get("matchAbbreviation"),
             "side1": {u for u in pl1 if u}, "side2": {u for u in pl2 if u},
             "score": f"{m.get('teamOneGameOneScore')}-{m.get('teamTwoGameOneScore')}",
             "winner": m.get("winner"), "status": m.get("matchStatus")}
        if m.get("isTieBreaker"):
            if g["status"] == 4 and g["winner"] in (1, 2):
                db_game = g
            continue
        if g["status"] != 4 or g["winner"] not in (1, 2):
            continue
        names, missing, vsum = [], [], {}
        for side, key in ((pl1, "v1"), (pl2, "v2")):
            vv = []
            for u in side:
                if u in vals:
                    vv.append(vals[u][1])
                    names.append(vals[u][0])
                else:
                    vv.append(0.0)
                    names.append("?")
                    missing.append(u)
            vsum[key] = vv
        eta = team_eta(vsum["v1"][0], vsum["v1"][1], vsum["v2"][0], vsum["v2"][1])
        p0 = calibrate(race_dist(round(sigmoid(eta), 4), 11)["p_win"])
        g.update(eta=eta, p0=p0, names=names, missing=missing)
        games.append(g)

    # per-game series
    for g in games:
        dp = ServeDP(eta_anchor(g["p0"], args.k), args.k)
        logs = c.match_logs(g["uuid"]) or []
        g["pts"], g["marks"] = match_series(logs, g["side1"], dp)
        g["pts"].append((len(g["pts"]) + 1, 1.0 if g["winner"] == 1 else 0.0, None))

    if db_game:
        lo, hi = 1e-4, 1 - 1e-4
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if rally_race_p(0, 0, mid) < args.p_db else (lo, mid)
        p_rally = 0.5 * (lo + hi)
        logs = c.match_logs(db_game["uuid"]) or []
        db_game["pts"], db_game["marks"] = db_series(logs, db_game["side1"], p_rally)
        db_game["pts"].append((len(db_game["pts"]) + 1,
                               1.0 if db_game["winner"] == 1 else 0.0, None))
        db_game["p0"] = args.p_db

    # matchup track: live game prob × pre-match probs of unplayed games
    track, w1, w2, x = [], 0, 0, 0
    seq = games + ([db_game] if db_game else [])
    for gi, g in enumerate(seq):
        future = [gg["p0"] for gg in games[gi + 1:]]
        for i, p, _ in g["pts"][:-1]:
            live = (p * matchup_prob(w1 + 1, w2, future, args.p_db)
                    + (1 - p) * matchup_prob(w1, w2 + 1, future, args.p_db)
                    ) if g is not db_game else p
            track.append((x + i, live, None))
        x += len(g["pts"])
        if g["winner"] == 1:
            w1 += 1
        else:
            w2 += 1
    track.append((x, 1.0 if (w1 > w2 or (db_game and db_game["winner"] == 1)) else 0.0, None))
    pre = matchup_prob(0, 0, [g["p0"] for g in games], args.p_db)

    panels = "".join(
        f"<div class='card'><h3>{g['abbrev']}: "
        f"{html.escape(' / '.join(g['names'][:2]))} vs "
        f"{html.escape(' / '.join(g['names'][2:]))}"
        f" — <b>{g['score']}</b> <span class='pre'>(pre-match "
        f"{display_floor(g['p0'])*100:.0f}%{', unrated: ' + str(len(g['missing'])) if g['missing'] else ''}"
        f")</span></h3>{svg_step(g['pts'], g['marks'])}</div>"
        for g in games) + (
        f"<div class='card'><h3>DreamBreaker: {db_game['score']} "
        f"<span class='pre'>(pre-match {display_floor(args.p_db)*100:.0f}%)</span></h3>"
        f"{svg_step(db_game['pts'], db_game['marks'], color='#8a5fbf')}</div>"
        if db_game else "")

    out = args.out or str(ROOT / f"winprob_{args.matchup[:8]}.html")
    Path(out).write_text(f"""<meta charset='utf-8'>
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:820px;margin:2em auto;
     padding:0 1em;color:#222;background:#fafafa}}
.card{{background:#fff;border:1px solid #e2e2e2;border-radius:8px;
      padding:.8em 1em;margin:.8em 0}}
h1{{font-size:1.25em}} h3{{font-size:.95em;margin:.2em 0 .4em;font-weight:500}}
.pre{{color:#777;font-size:.85em}} .foot{{color:#888;font-size:.78em;line-height:1.5}}
.t1{{color:{T1_COLOR};font-weight:600}} .t2{{color:{T2_COLOR};font-weight:600}}
.nolog{{color:#999;font-style:italic}}
</style>
<h1>{html.escape(title)}</h1>
<p>Rally-by-rally win probability for <span class='t1'>{html.escape(t1)}</span>
 vs <span class='t2'>{html.escape(t2)}</span> — final {md.get('teamOneScore')}–{md.get('teamTwoScore')}.
 Curves show P({html.escape(t1)} wins), replayed from the referee log.</p>
<div class='card'><h3>Matchup win probability
 <span class='pre'>(pre-match {display_floor(pre)*100:.0f}%)</span></h3>
{svg_step(track, [], hgt=190, color='#333')}</div>
{panels}
<p class='foot'>Engine: serve-aware side-out DP (4 serve states, exact
cell algebra), league serve-rally win rate k={args.k} measured from
referee logs; each game anchored to the calibrated pre-match probability
so the receipts ledger and these curves agree at rally zero. Ticks:
T=timeout, C=challenge/review. Probabilities are floored — nothing is
ever 0% or 100%. In-game calibration is pending the full rally-log
backfill; treat mid-game numbers as provisional.</p>
""")
    print(f"wrote {out}")
    for g in seq:
        print(f"  {g['abbrev']:4s} pre {g['p0']:.3f} → {g['score']} "
              f"({len(g['pts'])-1} rallies)")


if __name__ == "__main__":
    main()
