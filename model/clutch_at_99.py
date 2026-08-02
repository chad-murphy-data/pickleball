"""Does clutch predict who WINS a game that reached 9-9?

The sharpest surviving version of the predictive question.  clutch.md killed
clutch as a game predictor six ways, but every one of those tests was priced
at the START of a game (projected toss-ups, final margin <= 2, raw / reliable
/ top / bottom / residual / closeness).  None of them conditioned on the game
actually ARRIVING at the highest-leverage score in the sport.

That matters mechanically.  Clutch is mean-zero within a game by
construction, so over a whole game it must wash out.  But 9-9 is a
CONDITIONAL slice: from here on, EVERY remaining rally is a high-leverage
rally, so the within-game averaging-out no longer applies.  If a clutch
player really redistributes point-wins toward big points, 9-9 is where the
redistribution is all upside and none of the offsetting downside.  This is
the one slice where a mean-zero trait could still move a result.

Design
------
* Population: every doubles game to 11 in the referee-log archive that
  reached 9-9 (both sides on 9).  Source pb_rally (Supabase).
* Baseline: the exact serve-aware DP (web/sitelib/winprob.py) evaluated AT
  the 9-9 state with the actual serve state (which side, first or second
  server), with eta from month-of-game v2 values through the weakest-link
  team_eta and anchored the same way the live engine anchors.  So the
  baseline already knows skill AND the serve situation.
* Test: does the team clutch differential explain the residual
  (won - predicted) from 9-9?
* Leakage control: clutch (data/clutch_players.csv) was measured on
  Jan-May 2026 rallies, so games in that window are contaminated -- the
  9-9 rallies themselves fed the clutch estimate.  Every headline number is
  reported on the CLEAN subset (all games outside Jan-May 2026) as well as
  the full sample.

Run: python model/clutch_at_99.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from sitelib import race, winprob  # noqa: E402

CACHE = ROOT / "model" / "_clutch99_cache.json"
SB_URL = "https://nwgxyytowbluuykbdcfc.supabase.co"
SB_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
CONTAM = ("2026-01-01", "2026-06-01")   # clutch measurement window


# ---------------------------------------------------------------- fetching
def _page(table, params, chunk=1000):
    """PostgREST caps a page at db max-rows (1000 here), so a bigger `limit`
    silently returns 1000 — page in 1000s and always send a stable order, or
    offset paging repeats/drops rows."""
    import httpx
    out, off = [], 0
    hdr = {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"}
    while True:
        p = dict(params)
        p["limit"] = chunk
        p["offset"] = off
        r = httpx.get(f"{SB_URL}/rest/v1/{table}", params=p, headers=hdr, timeout=180)
        r.raise_for_status()
        rows = r.json()
        out.extend(rows)
        if len(rows) < chunk:
            return out
        off += chunk


def fetch():
    """Rallies at or past 9-9, plus the per-match rosters."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    # month-sliced: an unordered full scan trips the statement timeout, and
    # ordering only stays cheap inside a narrow date window
    rallies = []
    for y in (2024, 2025, 2026):
        for m in range(1, 13):
            lo = f"{y}-{m:02d}-01"
            hi = f"{y + 1}-01-01" if m == 12 else f"{y}-{m + 1:02d}-01"
            rallies += _page("pb_rally", {
                "select": ("match_id,game_number,rally_number,server_side,server_number,"
                           "server_score,receiver_score,won,outcome,match_date,tour"),
                "discipline": "eq.doubles",
                "server_score": "gte.9",
                "receiver_score": "gte.9",
                "match_date": f"gte.{lo}",
                "and": f"(match_date.lt.{hi})",
                "order": "match_id.asc,game_number.asc,rally_number.asc",
            })
    mids = sorted({r["match_id"] for r in rallies})
    roster = []
    for i in range(0, len(mids), 200):
        batch = ",".join(mids[i:i + 200])
        roster += _page("pb_side2", {
            "select": "match_id,side,p1,p2",
            "match_id": f"in.({batch})",
            "order": "match_id.asc,side.asc",
        })
    blob = {"rallies": rallies, "roster": roster}
    CACHE.write_text(json.dumps(blob))
    return blob


# ------------------------------------------------------------------ inputs
def load_values():
    """uuid -> (name, gender, current value, {month: value})."""
    cur, traj = {}, defaultdict(dict)
    for r in csv.DictReader((ROOT / "data" / "v2_players.csv").open()):
        cur[r["player_id"].lower()] = {
            "name": r["full_name"], "gender": r["gender"],
            "v": float(r["value_now_mean"]), "sd": float(r["value_now_sd"]),
            "games": int(r["games"])}
    for r in csv.DictReader((ROOT / "data" / "v2_trajectories.csv").open()):
        traj[r["player_id"].lower()][r["month"]] = float(r["value_mean"])
    return cur, traj


def load_clutch(cur):
    """clutch_players.csv is keyed by NAME; resolve to uuid via the v2 value
    it carries (the convention used by clutch_closeness_test.py)."""
    by_name = defaultdict(list)
    for u, d in cur.items():
        by_name[d["name"]].append((d["v"], u))
    clutch, z = {}, {}
    miss = []
    for r in csv.DictReader((ROOT / "data" / "clutch_players.csv").open()):
        cand = by_name.get(r["name"], [])
        if not cand:
            miss.append(r["name"])
            continue
        u = min(cand, key=lambda x: abs(x[0] - float(r["value"])))[1]
        clutch[u] = float(r["clutch"])
        z[u] = float(r["z"])
    if miss:
        print(f"  (unmatched clutch names: {miss})")
    return clutch, z


# ------------------------------------------------------------------- games
def build_games(blob, cur, traj):
    by_game = defaultdict(list)
    for r in blob["rallies"]:
        by_game[(r["match_id"], r["game_number"])].append(r)
    roster = defaultdict(dict)          # match -> side(0/1) -> [p1, p2]
    for r in blob["roster"]:
        if r["p1"] and r["p2"]:
            roster[r["match_id"]][r["side"]] = [r["p1"].lower(), r["p2"].lower()]

    games, skipped = [], defaultdict(int)
    for (mid, gn), rs in by_game.items():
        rs.sort(key=lambda r: r["rally_number"])
        # must actually pass through 9-9 (the gte.9 filter also catches 10-9 etc.)
        if not any(r["server_score"] == 9 and r["receiver_score"] == 9 for r in rs):
            skipped["never at 9-9"] += 1
            continue
        # format: games to 15 also pass 9-9; keep only to-11
        last = rs[-1]
        hi = max(last["server_score"] + (1 if last["won"] else 0),
                 last["receiver_score"] + (0 if last["won"] else 1))
        if hi > 13:
            skipped["not to-11"] += 1
            continue
        # winner of the game: whoever took the final rally (sides are 0/1)
        win_side = last["server_side"] if last["won"] else (1 - last["server_side"])
        # first rally at exactly 9-9 fixes the serve state we price from
        e = next(r for r in rs if r["server_score"] == 9 and r["receiver_score"] == 9)
        s1, s2 = roster[mid].get(0, []), roster[mid].get(1, [])
        if len(s1) != 2 or len(s2) != 2:
            skipped["roster != 2v2"] += 1
            continue
        us = s1 + s2
        if not all(u in cur for u in us):
            skipped["player not in v2"] += 1
            continue
        month = e["match_date"][:7]
        vals = [traj[u].get(month, cur[u]["v"]) for u in us]
        games.append({
            "mid": mid, "gn": gn, "date": e["match_date"], "tour": e["tour"],
            "us": us, "vals": vals,
            "serve_side": e["server_side"], "server_number": e["server_number"],
            "won1": 1 if win_side == 0 else 0,   # "side 1" = pb side 0
        })
    return games, skipped


def price(g):
    """P(side 1 wins | 9-9, actual serve state), exact serve-aware DP.

    eta is anchored the same way the live engine anchors: the DP's
    start-of-game probability is forced to equal the calibrated race-model
    probability, so mid-game numbers stay consistent with graded receipts.
    """
    v = g["vals"]
    eta = race.team_eta(v[0], v[1], v[2], v[3])
    p0 = race.calibrate(race.game_win_prob_uncertain(eta, race.SD_MATCH, 11))
    eta_a = winprob.eta_anchor(p0)
    dp = winprob.ServeDP(eta_a, winprob.K_DOUBLES, 11)
    # DP labels the serving side A; map to side 1
    n = g["server_number"] or 2
    if g["serve_side"] == 0:
        st = winprob.A1 if n == 1 else winprob.A2
        return dp.p(9, 9, st)
    st = winprob.B1 if n == 1 else winprob.B2
    return dp.p(9, 9, st)


# -------------------------------------------------------------------- test
def boot(x, y, fn, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    vals = np.array([fn(x[i], y[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    return np.percentile(vals, [2.5, 97.5])


def logistic(X, y, ridge=1e-4, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * b
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def cluster_boot(games, vals, resid, n=4000, seed=3):
    """Bootstrap over MATCHES, not games — a PPA best-of-3 contributes up to
    three 9-9 games with the same four players, so game-level resampling
    understates the standard error."""
    rng = np.random.default_rng(seed)
    idx_by_match = defaultdict(list)
    for i, g in enumerate(games):
        idx_by_match[g["mid"]].append(i)
    keys = list(idx_by_match)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_by_match[keys[k]] for k in pick])
        v, r = vals[idx], resid[idx]
        if v.std() < 1e-12:
            continue
        out.append(np.corrcoef(v, r)[0, 1])
    return np.percentile(out, [2.5, 97.5])


def placebo(games, clutch, zmap, label):
    """The control that decides whether the 9-9 signal is CLUTCH or just a
    mispriced baseline.

    Clutch correlates r ~ 0.6-0.7 with skill, so any residual skill the 9-9
    baseline fails to absorb will show up as a fake clutch effect (exactly the
    artifact clutch_residual_test.py caught).  Three arms:
      SKILL PLACEBO  -- swap clutch for the skill differential, zero clutch
                        content.  If it scores the same, it's the baseline.
      ORTHOGONAL     -- clutch differential with the skill differential
                        projected out.  This is clutch's OWN contribution.
      SHUFFLE        -- clutch values permuted across players, 200 draws.
    """
    resid = np.array([g["won1"] - g["p99"] for g in games])
    skill = np.array([g["skill_diff"] for g in games])
    cl = np.array([sum(clutch.get(u, 0.0) for u in g["us"][:2])
                   - sum(clutch.get(u, 0.0) for u in g["us"][2:]) for g in games])

    def rc(x):
        return float(np.corrcoef(x, resid)[0, 1])

    orth = cl - skill * (np.dot(cl, skill) / np.dot(skill, skill))
    lo_s, hi_s = cluster_boot(games, skill, resid)
    lo_o, hi_o = cluster_boot(games, orth, resid)
    lo_c, hi_c = cluster_boot(games, cl, resid)

    rng = np.random.default_rng(11)
    us_all = sorted({u for g in games for u in g["us"]})
    cvals = [clutch.get(u, 0.0) for u in us_all]
    sh = []
    for _ in range(200):
        perm = dict(zip(us_all, rng.permutation(cvals)))
        x = np.array([sum(perm.get(u, 0.0) for u in g["us"][:2])
                      - sum(perm.get(u, 0.0) for u in g["us"][2:]) for g in games])
        sh.append(rc(x))
    slo, shi = np.percentile(sh, [2.5, 97.5])

    print(f"{label} FALSIFICATION  n={len(games)}")
    print(f"    clutch differential      corr {rc(cl):+.3f}  cluster-CI[{lo_c:+.3f},{hi_c:+.3f}]")
    print(f"    SKILL PLACEBO (no clutch) corr {rc(skill):+.3f}  cluster-CI[{lo_s:+.3f},{hi_s:+.3f}]")
    print(f"    clutch ORTHOGONAL to skill corr {rc(orth):+.3f}  cluster-CI[{lo_o:+.3f},{hi_o:+.3f}]")
    print(f"    shuffled-clutch null      95% [{slo:+.3f},{shi:+.3f}]")


def report(games, clutch, zmap, label, hi_only=False):
    if len(games) < 30:
        print(f"{label}: n={len(games)} — too few, skipped")
        return
    tag = "reliable |z|>=1.5 only" if hi_only else "all players"
    src = ({u: (c if abs(zmap[u]) >= 1.5 else 0.0) for u, c in clutch.items()}
           if hi_only else clutch)
    cl, pred, won = [], [], []
    for g in games:
        c = [src.get(u, 0.0) for u in g["us"]]
        cl.append((c[0] + c[1]) - (c[2] + c[3]))
        pred.append(g["p99"])
        won.append(g["won1"])
    cl, pred, won = np.array(cl), np.array(pred), np.array(won)
    resid = won - pred
    r = float(np.corrcoef(cl, resid)[0, 1])
    lo, hi = boot(cl, resid, lambda a, b: np.corrcoef(a, b)[0, 1])

    lg = np.log(np.clip(pred, 1e-6, 1 - 1e-6) / np.clip(1 - pred, 1e-6, 1))
    X = np.column_stack([np.ones(len(cl)), lg, cl])
    b = logistic(X, won.astype(float))
    bs = []
    rng = np.random.default_rng(7)
    for idx in rng.integers(0, len(cl), (1500, len(cl))):
        try:
            bs.append(logistic(X[idx], won[idx].astype(float))[2])
        except np.linalg.LinAlgError:
            pass
    blo, bhi = np.percentile(bs, [2.5, 97.5])

    # effect size at a realistic gap: sd of the clutch differential
    sd = float(np.std(cl))
    pp = 100 * (1 / (1 + math.exp(-(b[2] * sd))) - 0.5) * 2 * 0.25 * 4  # ~slope in pp
    pp = 100 * (1 / (1 + np.exp(-(0.0 + b[2] * sd))) - 0.5)
    print(f"{label} [{tag}]  n={len(games)}")
    print(f"    base rate side1 wins from 9-9: {won.mean():.3f}   model says {pred.mean():.3f}")
    print(f"    corr(clutch diff, resid) = {r:+.3f}  CI[{lo:+.3f},{hi:+.3f}]")
    print(f"    logit weight on clutch    = {b[2]:+.2f}  CI[{blo:+.2f},{bhi:+.2f}]"
          f"   (+1 sd of clutch gap = {pp:+.1f} pp)")


def main():
    print("Fetching rally states at 9-9 ...")
    blob = fetch()
    cur, traj = load_values()
    clutch, zmap = load_clutch(cur)
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])

    games, skipped = build_games(blob, cur, traj)
    print(f"  {len(games)} doubles games to 11 reached 9-9   (dropped: {dict(skipped)})")
    for g in games:
        g["p99"] = price(g)
        v = g["vals"]
        g["skill_diff"] = race.team_eta(v[0], v[1], v[2], v[3])

    covered = [g for g in games if any(u in clutch for u in g["us"])]
    clean = [g for g in covered if not (CONTAM[0] <= g["date"] < CONTAM[1])]
    print(f"  {len(covered)} with >=1 rated player;  {len(clean)} outside the "
          f"clutch measurement window (clean)\n")

    print("=" * 74)
    print("Q: does the clutch differential explain who wins FROM 9-9?")
    print("=" * 74)
    for gs, lab in ((covered, "FULL   "), (clean, "CLEAN  ")):
        for hi in (False, True):
            report(gs, clutch, zmap, lab, hi_only=hi)
        print()
        placebo(gs, clutch, zmap, lab)
        print()

    # ---- sanity: is the 9-9 baseline itself calibrated?
    p = np.array([g["p99"] for g in covered])
    w = np.array([g["won1"] for g in covered])
    print(f"9-9 baseline calibration: mean pred {p.mean():.3f} vs obs {w.mean():.3f}, "
          f"Brier {np.mean((p - w) ** 2):.4f} (coin flip 0.2500)")

    # ---- how much room is there at all?  spread of the clutch differential
    cl = np.array([sum(clutch.get(u, 0.0) for u in g["us"][:2])
                   - sum(clutch.get(u, 0.0) for u in g["us"][2:]) for g in covered])
    print(f"clutch differential: sd {cl.std():.4f}, range [{cl.min():+.4f},{cl.max():+.4f}]")


if __name__ == "__main__":
    main()
