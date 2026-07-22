"""Doubles->singles correction for non-ranked-singles players, from
DreamBreaker rally logs (v2 parser, 2026-07-22).

Question: players with no meaningful PICKLE singles record get a singles
value imputed from doubles (make_forecast.SINGLES_IMPUTE, fitted regression
singles ~= 0.28 + 1.14*doubles). db_model.md flagged the selection bias:
they don't play singles because they're worse at it. DreamBreakers ARE
singles points, so measure the bias directly: pool every non-ranked
player's DB rallies and ask how they perform vs what the imputation
predicts, controlling for opposition strength and their own doubles rating
(both enter through the value gap in the rally logistic below).

v2 parser (replaces the score-string inference of v1, which only
score-validated 37/94 matches): DreamBreaker referee logs contain
  * type 12  rally rows (server_uuid reliable; receiver NOT),
  * type 14  POINT rows -- point_log.team_uuid is the scoring FRANCHISE
    uuid, start/end_score give the correction-aware delta
    (harvest_logs._point_delta: rewinds, phantom double-entries, stale
    strings all observed in the wild and handled),
  * type 32  substitute_players_log rows -- every 4-point rotation is
    announced explicitly: {player_in_uuid, player_out_uuid, team_uuid}.
So the on-court singles matchup for every rally is EXPLICIT (track the
current player per franchise from the sub rows; the initial player is the
first sub row's player_out), the winner of every rally is EXPLICIT (POINT
team_uuid), and each team's ANNOUNCED ROTATION ORDER falls out for free.

Validation: every match's reconstructed per-franchise totals must equal
the official final (dreambreakers.csv t1/t2, oriented via the matchup
record's teamOneUuid/teamTwoUuid). Non-validating matches are EXCLUDED
from the fit and reported.

Outputs:
  data/db_rallies.csv  one row per reconstructed rally (players, winner)
  data/db_orders.csv   announced rotation order per team per DreamBreaker
  fit report on stdout (logistic + cluster bootstrap + sensitivity)

Run: python model/db_impute.py   (offline after first run; logs cached)
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scraper"))
from pb_api import PBClient                                   # noqa: E402
from harvest_logs import _point_delta                          # noqa: E402

RALLY, POINT, SUB, START = 12, 14, 32, 22
K_TEAM_LEVEL = 0.42          # db_model.md team-level fit, for reference
RANKED_MIN_GAMES = 10        # a "real" PICKLE singles score (make_forecast)
IMPUTE_A, IMPUTE_B = 0.28, 1.14   # UN-shrunk imputation (what we test)
BOOT = 2000
SEED = 20260722


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_players():
    sv, sg, gen, dbl, names = {}, {}, {}, {}, {}
    with open(ROOT / "data" / "singles_players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            sv[u] = float(r["singles_value"]); sg[u] = int(r["singles_games"])
            gen[u] = r["gender"]; names[u] = r["full_name"]
    with open(ROOT / "data" / "v2_players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            dbl[u] = float(r["value_now_mean"])
            gen.setdefault(u, r["gender"]); names.setdefault(u, r["full_name"])
    return sv, sg, gen, dbl, names


def load_dreambreakers():
    out = []
    with open(ROOT / "data" / "dreambreakers.csv") as fh:
        for r in csv.DictReader(fh):
            try:
                t1, t2 = int(r["t1_score"]), int(r["t2_score"])
            except ValueError:
                continue          # a few rows have shifted columns; skip
            if len(r["match_id"]) == 36:
                out.append((r["match_id"].lower(), r["matchup_id"].lower(),
                            t1, t2, r["date"]))
    return out


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def confirmed_by_next(rows, i):
    """Port of harvest_logs' stale-string disambiguation (component 0 only:
    DB strings carry no server number and no side-out perspective flips)."""
    try:
        cur0 = int(rows[i]["start_score_current_game_string"].split("-")[0])
    except (KeyError, ValueError, IndexError, AttributeError):
        return False
    for nxt in rows[i + 1:]:
        try:
            parts = [int(x) for x in
                     nxt["start_score_current_game_string"].split("-")]
        except (KeyError, ValueError, IndexError, AttributeError):
            continue
        return parts[0] == cur0 + 1 or parts[1] == cur0 + 1
    return False


def parse_db(rows, team1, team2):
    """Reconstruct one DreamBreaker.

    Returns dict:
      rallies  [(p1, p2, winner_team)]  p1/p2 = on-court player of team1/2
      totals   {team_uuid: points}
      orders   {team_uuid: [players in announced rotation order]}
      flags    list of anomaly strings
    """
    rows = sorted(rows, key=lambda r: r.get("log_index", 0))
    flags = []
    cur = {}                       # team_uuid -> on-court player
    orders = defaultdict(list)     # team_uuid -> rotation sequence
    first_out_seen = set()

    # initial on-court player per team = first sub row's player_out
    for r in rows:
        if r.get("log_type") == SUB:
            sl = r.get("substitute_players_log") or {}
            t = (sl.get("team_uuid") or "").lower()
            if t and t not in first_out_seen:
                first_out_seen.add(t)
                po = (sl.get("player_out_uuid") or "").lower()
                cur[t] = po
                orders[t].append(po)
    if set(cur) != {team1, team2}:
        flags.append(f"teams-from-subs {sorted(cur)} != matchup teams")
        return {"rallies": [], "totals": {}, "orders": dict(orders),
                "flags": flags}

    totals = {team1: 0, team2: 0}
    rallies = []                   # (p1, p2, winner_team)
    # The 4th point of a segment is logged AFTER the two substitution rows
    # (observed ordering: rally, SUB, SUB, POINT), so on-court players must
    # be snapshotted at the RALLY row, not read at POINT time.
    pending = None                 # (p1, p2, server) at the rally row
    for i, r in enumerate(rows):
        t = r.get("log_type")
        if t == SUB:
            sl = r.get("substitute_players_log") or {}
            tm = (sl.get("team_uuid") or "").lower()
            pin = (sl.get("player_in_uuid") or "").lower()
            pout = (sl.get("player_out_uuid") or "").lower()
            if tm in cur:
                if cur[tm] != pout:
                    flags.append(f"sub out-mismatch row {r.get('log_index')}")
                cur[tm] = pin
                if pin not in orders[tm]:
                    orders[tm].append(pin)
        elif t == RALLY:
            pending = (cur.get(team1), cur.get(team2),
                       (r.get("server_uuid") or "").lower())
        elif t == POINT:
            delta, team = _point_delta(r)
            team = team.lower()
            if team not in totals:
                if delta:
                    flags.append(f"unknown scoring team {team[:8]}")
                continue
            # Cross-check against the payload's per-team cumulative score
            # (clean on everything except rewind rows, which _point_delta
            # already resolves to a negative via the smaller-|delta| rule):
            #   end == total+1  -> normal point (also rescues real points
            #                      whose string was stale or pre-advanced)
            #   end <= total    -> duplicate/phantom entry: drop
            #   end >  total+1  -> unlogged score gap: credit the gap points
            #                      to the team WITHOUT rally attribution
            end = (r.get("point_log") or {}).get("end_score")
            if delta >= 0 and end is not None:
                if end == totals[team] + 1:
                    delta = 1
                elif end <= totals[team]:
                    if delta:
                        flags.append(f"dup-entry dropped row {r.get('log_index')}")
                    delta = 0
                else:
                    gap = end - totals[team] - 1
                    flags.append(f"score gap +{gap} row {r.get('log_index')}")
                    totals[team] += gap        # real score, no logged rallies
                    delta = 1
            if delta > 0:
                if delta > 1:
                    flags.append(f"multi-point delta {delta}")
                if pending is None:
                    # POINT with no preceding rally row: on-court players
                    # cannot be pinned (cur may already reflect a
                    # substitution) -- credit the score, skip attribution
                    flags.append(f"unattributed point row {r.get('log_index')}")
                    totals[team] += delta
                    continue
                p1, p2, srv = pending
                if srv and srv not in (p1, p2):
                    flags.append(f"server not on court row {r.get('log_index')}")
                for _ in range(delta):
                    rallies.append((p1, p2, team))
                    totals[team] += 1
                pending = None
            elif delta < 0:
                for _ in range(-delta):
                    # retract the scoring team's most recent credited rally
                    for j in range(len(rallies) - 1, -1, -1):
                        if rallies[j][2] == team:
                            rallies.pop(j)
                            totals[team] -= 1
                            break
                    else:
                        flags.append("retraction with no prior point")
    return {"rallies": rallies, "totals": totals,
            "orders": dict(orders), "flags": flags}


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def logistic_fit(data):
    """data rows: (gap, impdiff, wins_i, n).  Fit
    P(i wins) = sigmoid(b0 + b1*gap + b2*impdiff); return coefs + SEs."""
    b = [0.0, 0.3, 0.0]
    H = None
    for _ in range(200):
        g = [0.0] * 3
        H = [[0.0] * 3 for _ in range(3)]
        for gap, imd, wi, n in data:
            eta = b[0] + b[1] * gap + b[2] * imd
            p = 1 / (1 + math.exp(-eta))
            x = (1.0, gap, float(imd))
            r_ = wi - n * p
            w = n * p * (1 - p)
            for u in range(3):
                g[u] += r_ * x[u]
                for v in range(3):
                    H[u][v] += w * x[u] * x[v]
        M = [row[:] + [g[k]] for k, row in enumerate(H)]
        for c in range(3):
            pv = M[c][c]
            if abs(pv) < 1e-12:
                break
            for cc in range(c, 4):
                M[c][cc] /= pv
            for rr in range(3):
                if rr != c:
                    f = M[rr][c]
                    for cc in range(c, 4):
                        M[rr][cc] -= f * M[c][cc]
        step = [M[0][3], M[1][3], M[2][3]]
        b = [bi + s for bi, s in zip(b, step)]
        if sum(abs(s) for s in step) < 1e-10:
            break
    # invert H for SEs
    n_ = 3
    A = [row[:] for row in H]
    I = [[1.0 if i == j else 0.0 for j in range(n_)] for i in range(n_)]
    for c in range(n_):
        pv = A[c][c]
        for j in range(n_):
            A[c][j] /= pv; I[c][j] /= pv
        for rr in range(n_):
            if rr != c:
                f = A[rr][c]
                for j in range(n_):
                    A[rr][j] -= f * A[c][j]; I[rr][j] -= f * I[c][j]
    se = [I[i][i] ** 0.5 for i in range(n_)]
    return b, se


def run_fit(rally_rows, sv, sg, gen, dbl, ranked_min, label):
    """rally_rows: (match_id, u1, u2, y1). Build per-(match,pair) binomials,
    same-gender only; fit; cluster bootstrap by match."""
    def value(u):
        if u in sv and sg.get(u, 0) >= ranked_min:
            return sv[u], False
        if u in dbl:
            return IMPUTE_A + IMPUTE_B * dbl[u], True
        if u in sv:
            return sv[u], True     # tiny singles sample: treat as non-ranked
        return None, None

    per_match = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    skipped_gender, skipped_value = 0, 0
    for mid, u1, u2, y1 in rally_rows:
        if gen.get(u1) != gen.get(u2) or gen.get(u1) not in ("M", "F"):
            skipped_gender += 1
            continue
        v1, imp1 = value(u1)
        v2, imp2 = value(u2)
        if v1 is None or v2 is None:
            skipped_value += 1
            continue
        key = (u1, u2, v1 - v2, (1 if imp1 else 0) - (1 if imp2 else 0))
        rec = per_match[mid][key]
        rec[0] += y1
        rec[1] += 1

    def dataset(matches):
        return [(gap, imd, w, n)
                for m in matches
                for (u1, u2, gap, imd), (w, n) in per_match[m].items()]

    matches = sorted(per_match)
    data = dataset(matches)
    n_rallies = sum(n for *_, n in data)
    n_imp = sum(n for gap, imd, w, n in data if imd != 0)
    b, se = logistic_fit(data)
    delta = -b[2] / b[1] if abs(b[1]) > 1e-9 else float("nan")

    rng = random.Random(SEED)
    boots = []
    for _ in range(BOOT):
        sample = [matches[rng.randrange(len(matches))]
                  for _ in range(len(matches))]
        bb, _ = logistic_fit(dataset(sample))
        if abs(bb[1]) > 1e-9:
            boots.append(-bb[2] / bb[1])
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    neg_share = sum(1 for x in boots if x <= 0) / len(boots)

    print(f"\n--- fit [{label}] (ranked = >= {ranked_min} singles games) ---")
    print(f"  same-gender rallies {n_rallies} "
          f"(imputed-vs-other: {n_imp}); "
          f"skipped cross-gender {skipped_gender}, unratable {skipped_value}")
    print(f"  P(win) = sigmoid({b[0]:+.3f} + {b[1]:.3f}*gap "
          f"+ {b[2]:+.3f}*impdiff)")
    print(f"  empirical rally k = {b[1]:.3f} +/- {se[1]:.3f} "
          f"  (team-level fit {K_TEAM_LEVEL})")
    print(f"  imputed penalty   = {b[2]:+.3f} +/- {se[2]:.3f} "
          f"(z = {b[2]/se[2]:.2f})")
    print(f"  correction (value-scale shrink) = {delta:+.3f}")
    print(f"  cluster bootstrap ({BOOT}x by match): "
          f"95% CI [{lo:+.3f}, {hi:+.3f}], P(shrink<=0) = {neg_share:.3f}")
    return delta, (lo, hi)


# ---------------------------------------------------------------------------
def main():
    sv, sg, gen, dbl, names = load_players()
    dbs = load_dreambreakers()
    c = PBClient()
    print(f"{len(dbs)} DreamBreakers on record; parsing referee logs...")

    ok, no_logs, bad = [], [], []
    all_rallies = []           # (match_id, p_team1, p_team2, y_team1)
    order_rows = []
    for mid, muid, t1s, t2s, date in dbs:
        rows = c.match_logs(mid) or []
        if not rows:
            no_logs.append(mid)
            continue
        md = c.matchup_data(muid, volatile=False)
        team1 = (md.get("teamOneUuid") or "").lower()
        team2 = (md.get("teamTwoUuid") or "").lower()
        res = parse_db(rows, team1, team2)
        v = res["totals"]
        exact = (v.get(team1) == t1s and v.get(team2) == t2s)
        # benign flags = the parser handling known log quirks correctly;
        # risky flags = possible on-court attribution error -> exclude
        BENIGN = ("dup-entry dropped", "score gap", "unattributed point")
        risky = [f for f in res["flags"] if not f.startswith(BENIGN)]
        if exact and not risky:
            ok.append(mid)
            for p1, p2, w in res["rallies"]:
                all_rallies.append((mid, p1, p2, 1 if w == team1 else 0))
            for tm, order in res["orders"].items():
                order_rows.append({
                    "match_id": mid, "date": date, "team_uuid": tm,
                    "team_title": (md.get("teamOneTitle") if tm == team1
                                   else md.get("teamTwoTitle")),
                    "order": "|".join(order),
                    "order_names": "|".join(names.get(u, "?") for u in order),
                    "order_genders": "".join(gen.get(u, "?") for u in order),
                })
        else:
            bad.append((mid, v.get(team1), v.get(team2), t1s, t2s,
                        risky[:2] or res["flags"][:1]))
    print(f"  logs found: {len(ok) + len(bad)}/{len(dbs)}  "
          f"(no digital log: {len(no_logs)})")
    print(f"  EXACT final-score validation: {len(ok)}/{len(ok) + len(bad)}")
    if bad:
        print("  excluded (mismatch/flags):")
        for mid, a, b_, t1, t2, fl in bad[:10]:
            print(f"    {mid[:8]} recon {a}-{b_} vs official {t1}-{t2} {fl}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")

    # persist rally + order datasets
    with open(ROOT / "data" / "db_rallies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "player_team1", "player_team2", "team1_won"])
        w.writerows(all_rallies)
    with open(ROOT / "data" / "db_orders.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(order_rows[0].keys()))
        w.writeheader()
        w.writerows(order_rows)
    print(f"  wrote data/db_rallies.csv ({len(all_rallies)} rallies), "
          f"data/db_orders.csv ({len(order_rows)} team-orders)")

    # empirical league ordering behaviour (Anna's premise)
    m_first = sum(1 for r in order_rows if r["order_genders"][:1] == "M")
    m_top2 = sum(r["order_genders"][:2].count("M") for r in order_rows)
    n_ord = len(order_rows)
    print(f"\nEmpirical announced orders ({n_ord} team-orders):")
    print(f"  slot 1 is a man: {m_first}/{n_ord} ({m_first/n_ord*100:.0f}%)")
    print(f"  men share of slots 1-2: {m_top2}/{2*n_ord} "
          f"({m_top2/(2*n_ord)*100:.0f}%)")

    # main fit + sensitivity
    run_fit(all_rallies, sv, sg, gen, dbl, RANKED_MIN_GAMES, "primary")
    run_fit(all_rallies, sv, sg, gen, dbl, 1, "sensitivity: ranked >= 1 game")
    run_fit(all_rallies, sv, sg, gen, dbl, 30, "sensitivity: ranked >= 30")


if __name__ == "__main__":
    main()
