"""Clutch as an SRM-style latent-variable fit over ALL FOUR players in a rally.

Why this replaces the marginal average in big_points.py
-------------------------------------------------------
The frozen index is  clutch_u = mean(levz * residual)  over u's OWN SERVING
rallies.  Two problems, both structural:

1. **It sees ~a quarter of a player's rallies.**  Return rallies are dropped
   entirely, and in side-out scoring you cannot score on return -- every
   comeback is BUILT on return rallies.  The metric is blind to the half of
   the game where deficits get erased.  (The stated reason, "you can't pin a
   return rally on one of two receivers", does not hold: pb_rally carries
   receiver_uuid on 100% of doubles rallies.  And the argument is
   self-inconsistent -- the server's partner plays the rally too, so serve
   attribution is exactly as team-contaminated as return attribution.)

2. **It never adjusts for who else was on court.**  Play your big points
   next to Ben Johns and his clutch leaks into your average.  It is raw
   plus-minus, not a rating.

This module does for clutch what the SRM does for skill: every rally is one
observation constraining all four players at once,

    logit P(serving side wins rally) = logit(k_side)          <- skill, FIXED
                                     + levz * (a_srv + a_srvp
                                               - d_rcv - d_rcvp)

where levz is the within-game standardized leverage (exact serve-aware DP
swing, same construction as big_points), and each player carries TWO
parameters:

    a_u = ATTACK clutch  -- applies on every rally where u's side serves
    d_u = DEFEND clutch  -- applies on every rally where u's side receives

Both are identified because every player appears on both sides of the serve.
The skill term enters as a fixed OFFSET (from v2 through the serve-aware
DP), so clutch is identified purely as the INTERACTION with leverage and
cannot absorb skill -- the confound that inflated the 9-9 result by a third.

A Gaussian prior on (a, d) gives partial pooling, so the noisy middle shrinks
toward zero on its own rather than being cut off post hoc at |z| > 1.5.

Run: python model/clutch_srm.py          # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from sitelib import race, winprob  # noqa: E402
from sitelib.winprob import _table, serve_probs  # noqa: E402

CACHE = ROOT / "model" / "_clutch_srm_cache.json"
SB_URL = "https://nwgxyytowbluuykbdcfc.supabase.co"
SB_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
K_LEAGUE = 0.443            # same league serve rate big_points measured
MIN_RALLIES = 300           # match the frozen index's inclusion bar
ETA_ROUND = 0.05            # DP table cache granularity


# ---------------------------------------------------------------- fetching
def _get(table, params):
    import httpx
    hdr = {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"}
    for attempt in range(4):
        try:
            r = httpx.get(f"{SB_URL}/rest/v1/{table}", params=params,
                          headers=hdr, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 3:
                raise
    return []


def _match_ids():
    """Distinct doubles match ids, via the one-row-per-player-per-match table
    (68k rows) rather than a distinct over 1M rallies."""
    out, off = [], 0
    while True:
        rows = _get("pb_match_player_serve", {
            "select": "match_id", "discipline": "eq.doubles",
            "order": "match_id.asc,player_uuid.asc", "limit": 1000, "offset": off})
        out.extend(rows)
        if len(rows) < 1000:
            return out
        off += 1000


def fetch():
    """All doubles rallies + per-match pairs.  PostgREST caps a page at 1000
    rows regardless of `limit`, and offset paging without a stable `order`
    repeats and drops rows -- both silent, so page in 1000s with an order."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    sel = ("match_id,game_number,rally_number,server_side,server_number,"
           "server_uuid,receiver_uuid,server_score,receiver_score,won,"
           "match_date")
    # Deep OFFSET into a date slice re-scans and trips the statement timeout.
    # Batch on match_id instead: the filtered set stays tiny (PK-indexed), so
    # offsets never get deep enough to matter.
    mids = sorted({r["match_id"] for r in _match_ids()})
    print(f"  {len(mids)} doubles matches")
    batches = [mids[i:i + 30] for i in range(0, len(mids), 30)]

    def one(batch):
        out, off = [], 0
        while True:
            rows = _get("pb_rally", {
                "select": sel, "match_id": f"in.({','.join(batch)})",
                "order": "match_id.asc,game_number.asc,rally_number.asc",
                "limit": 1000, "offset": off})
            out.extend(rows)
            if len(rows) < 1000:
                return out
            off += 1000

    with ThreadPoolExecutor(max_workers=12) as ex:
        rallies = [r for part in ex.map(one, batches) for r in part]
    print(f"  fetched {len(rallies)} doubles rallies")

    def pairs(batch):
        return _get("pb_side2", {"select": "match_id,side,p1,p2",
                                 "match_id": f"in.({','.join(batch)})",
                                 "order": "match_id.asc,side.asc",
                                 "limit": 1000})

    chunks = [mids[i:i + 200] for i in range(0, len(mids), 200)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        roster = [r for part in ex.map(pairs, chunks) for r in part]
    blob = {"rallies": rallies, "roster": roster}
    CACHE.write_text(json.dumps(blob))
    return blob


# ------------------------------------------------------------------ inputs
def load_values():
    cur, traj = {}, defaultdict(dict)
    for r in csv.DictReader((ROOT / "data" / "v2_players.csv").open()):
        cur[r["player_id"].lower()] = {"name": r["full_name"],
                                       "gender": r["gender"],
                                       "v": float(r["value_now_mean"])}
    for r in csv.DictReader((ROOT / "data" / "v2_trajectories.csv").open()):
        traj[r["player_id"].lower()][r["month"]] = float(r["value_mean"])
    return cur, traj


def leverage_of(V, T, a, b, state, side_A_serving):
    """|win-prob swing| on this rally — verbatim from big_points.py so the
    leverage scale stays identical to the frozen index."""
    def val(aa, bb, ss):
        if aa >= T and aa - bb >= 2:
            return 1.0
        if bb >= T and bb - aa >= 2:
            return 0.0
        return V.get((aa, bb, ss), 0.5)
    if side_A_serving:
        w = val(a + 1, b, state)
        l = val(a, b, 1) if state == 0 else val(a, b, 2)
    else:
        w = val(a, b + 1, state)
        l = val(a, b, 3) if state == 2 else val(a, b, 0)
    return abs(w - l)


# ------------------------------------------------------------- design build
def build(blob, cur, traj):
    by_game = defaultdict(list)
    for r in blob["rallies"]:
        by_game[(r["match_id"], r["game_number"])].append(r)
    roster = defaultdict(dict)
    for r in blob["roster"]:
        if r["p1"] and r["p2"]:
            roster[r["match_id"]][r["side"]] = [r["p1"].lower(), r["p2"].lower()]

    rows, skipped = [], defaultdict(int)
    for (mid, gn), rs in by_game.items():
        s0, s1 = roster[mid].get(0, []), roster[mid].get(1, [])
        if len(s0) != 2 or len(s1) != 2:
            skipped["no 2v2 roster"] += 1
            continue
        us = s0 + s1
        if not all(u in cur for u in us):
            skipped["unrated player"] += 1
            continue
        rs.sort(key=lambda r: r["rally_number"])
        if any(r["server_score"] is None or r["receiver_score"] is None
               or r["server_side"] is None for r in rs):
            skipped["null score/side"] += 1
            continue
        last = rs[-1]
        top = max(last["server_score"] + (1 if last["won"] else 0),
                  last["receiver_score"] + (0 if last["won"] else 1))
        # target: to-11 (incl. deuce runs) vs to-15.  Deuce games can finish
        # ABOVE 11, so a flat cutoff would throw out exactly the wildest
        # endgames -- split on which target the run is consistent with.
        T = 11 if top <= 13 else 15
        if top > 17:
            skipped["odd format"] += 1
            continue
        month = rs[0]["match_date"][:7]
        v = [traj[u].get(month, cur[u]["v"]) for u in us]
        eta = race.team_eta(v[0], v[1], v[2], v[3])
        kA, kB = serve_probs(round(eta / ETA_ROUND) * ETA_ROUND, K_LEAGUE)
        V = _table(round(kA, 6), round(kB, 6), T, T + 40)

        levs, recs = [], []
        bad = False
        for r in rs:
            side = r["server_side"]
            n = r["server_number"] or 2
            if (r["server_score"] is None or r["receiver_score"] is None
                    or side is None):
                bad = True
                break
            # DP state: 0/1 = side0 first/second server, 2/3 = side1
            state = (0 if n == 1 else 1) if side == 0 else (2 if n == 1 else 3)
            a = r["server_score"] if side == 0 else r["receiver_score"]
            b = r["receiver_score"] if side == 0 else r["server_score"]
            levs.append(leverage_of(V, T, a, b, state, side == 0))
            srv = (r["server_uuid"] or "").lower()
            rcv = (r["receiver_uuid"] or "").lower()
            ss = s0 if side == 0 else s1
            rr = s1 if side == 0 else s0
            if srv not in ss or rcv not in rr:
                recs.append(None)
                continue
            srvp = ss[0] if ss[1] == srv else ss[1]
            rcvp = rr[0] if rr[1] == rcv else rr[1]
            recs.append((srv, srvp, rcv, rcvp, kA if side == 0 else kB,
                         r["won"]))
        if bad:
            skipped["null score/side"] += 1
            continue
        levs = np.array(levs)
        if levs.std() < 1e-9:
            skipped["flat leverage"] += 1
            continue
        levz = (levs - levs.mean()) / levs.std()
        for rec, lz in zip(recs, levz):
            if rec is None:
                continue
            srv, srvp, rcv, rcvp, k, won = rec
            rows.append((srv, srvp, rcv, rcvp, float(lz),
                         math.log(k / (1 - k)), int(won), mid, gn))
    return rows, skipped


# -------------------------------------------------------------------- fit
def fit(rows, index, lam, iters=400, lam_main=0.25):
    """MAP logistic fit.  Params: [m | n | a | d], P each.

        logit P(serve side wins) = offset
                                 + (m_s1 + m_s2) - (n_r1 + n_r2)      <- LEVEL
                                 + levz * ((a_s1 + a_s2)
                                           - (d_r1 + d_r2))           <- CLUTCH

    The main effects m (serving) and n (receiving) are NOT decoration -- they
    are what makes the clutch terms mean anything.  levz is mean-zero within a
    GAME, but not within the subset of rallies where one particular side
    serves, so without a level term any miscalibration in the skill offset
    flows straight into a and d.  A first cut without m/n produced z-scores
    near +30 in exact skill order: it was re-estimating skill, not clutch.

    With the level terms in, a and d are identified only off the leverage
    GRADIENT -- does this player do better than their own baseline
    specifically when the point is big -- which is the definition.
    """
    P = len(index)
    n = len(rows)
    srv = np.array([[index[r[0]], index[r[1]]] for r in rows])
    rcv = np.array([[index[r[2]], index[r[3]]] for r in rows])
    lz = np.array([r[4] for r in rows])
    off = np.array([r[5] for r in rows])
    y = np.array([r[6] for r in rows], dtype=float)
    pri = np.concatenate([np.full(2 * P, lam_main), np.full(2 * P, lam)])

    def eta_of(beta):
        m, nn = beta[:P], beta[P:2 * P]
        a, d = beta[2 * P:3 * P], beta[3 * P:]
        return (off
                + m[srv[:, 0]] + m[srv[:, 1]] - nn[rcv[:, 0]] - nn[rcv[:, 1]]
                + lz * (a[srv[:, 0]] + a[srv[:, 1]]
                        - d[rcv[:, 0]] - d[rcv[:, 1]]))

    def negll(beta):
        """Penalised negative log-likelihood and its exact gradient.

        A diagonal-Newton step diverges here: teammates' parameters are
        strongly correlated (they co-occur in every rally of a match) and the
        diagonal preconditioner ignores exactly that structure, so full steps
        overshoot.  L-BFGS builds the curvature it needs from the gradient
        history instead.
        """
        e = np.clip(eta_of(beta), -30, 30)
        p = 1 / (1 + np.exp(-e))
        f = -np.sum(y * np.log(np.clip(p, 1e-12, None))
                    + (1 - y) * np.log(np.clip(1 - p, 1e-12, None)))
        f += 0.5 * np.sum((beta / pri) ** 2)
        r = p - y
        gl = r * lz
        g = np.zeros(4 * P)
        for c in (0, 1):
            np.add.at(g, srv[:, c], r)                 # m
            np.add.at(g, P + rcv[:, c], -r)            # n
            np.add.at(g, 2 * P + srv[:, c], gl)        # a
            np.add.at(g, 3 * P + rcv[:, c], -gl)       # d
        g += beta / pri ** 2
        return f, g

    from scipy.optimize import minimize
    res = minimize(negll, np.zeros(4 * P), jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "maxfun": iters * 2})
    beta = res.x
    e = eta_of(beta)
    p = np.clip(1 / (1 + np.exp(-e)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # se from the diagonal of the observed information
    w = p * (1 - p)
    h = np.zeros(4 * P)
    hl = w * lz * lz
    for c in (0, 1):
        np.add.at(h, srv[:, c], w)
        np.add.at(h, P + rcv[:, c], w)
        np.add.at(h, 2 * P + srv[:, c], hl)
        np.add.at(h, 3 * P + rcv[:, c], hl)
    h += 1 / pri ** 2
    return beta, 1 / np.sqrt(h), ll, n


def holdout_ll(rows, index, lam, seed=0):
    """Held-out log-loss, split by MATCH so a match never spans the split."""
    rng = np.random.default_rng(seed)
    mids = sorted({r[7] for r in rows})
    test = set(np.array(mids)[rng.random(len(mids)) < 0.25])
    tr = [r for r in rows if r[7] not in test]
    te = [r for r in rows if r[7] in test]
    beta, _, _, _ = fit(tr, index, lam)
    P = len(index)
    m, nn = beta[:P], beta[P:2 * P]
    a, d = beta[2 * P:3 * P], beta[3 * P:]
    ll = 0.0
    for r in te:
        i1, i2 = index[r[0]], index[r[1]]
        j1, j2 = index[r[2]], index[r[3]]
        e = (r[5] + m[i1] + m[i2] - nn[j1] - nn[j2]
             + r[4] * (a[i1] + a[i2] - d[j1] - d[j2]))
        p = min(max(1 / (1 + math.exp(-max(min(e, 30), -30))), 1e-12), 1 - 1e-12)
        ll += math.log(p) if r[6] else math.log(1 - p)
    return ll / len(te), len(te)


def main():
    print("Fetching rallies ...")
    blob = fetch()
    cur, traj = load_values()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])

    rows, skipped = build(blob, cur, traj)
    print(f"  {len(rows)} rallies with all four players identified "
          f"(dropped: {dict(skipped)})")

    count = defaultdict(int)
    for r in rows:
        for u in r[:4]:
            count[u] += 1
    keep = {u for u, c in count.items() if c >= MIN_RALLIES}
    rows = [r for r in rows if all(u in keep for u in r[:4])]
    index = {u: i for i, u in enumerate(sorted(keep))}
    print(f"  {len(index)} players with >= {MIN_RALLIES} rallies; "
          f"{len(rows)} rallies with all four kept\n")

    print("Choosing the prior sd by held-out log-loss (split by match):")
    best, best_lam = None, None
    for lam in (0.02, 0.05, 0.10, 0.20, 0.40):
        ll, nte = holdout_ll(rows, index, lam)
        flag = ""
        if best is None or ll > best:
            best, best_lam, flag = ll, lam, "  <-- best"
        print(f"    prior sd {lam:.2f}:  held-out ll/rally {ll:+.6f}{flag}")
    print(f"  using prior sd = {best_lam}\n")

    beta, se, ll, n = fit(rows, index, best_lam)
    P = len(index)
    inv = {i: u for u, i in index.items()}
    out = []
    for u, i in index.items():
        out.append({"uuid": u, "name": cur[u]["name"], "gender": cur[u]["gender"],
                    "n": count[u],
                    "attack": float(beta[2 * P + i]),
                    "attack_se": float(se[2 * P + i]),
                    "defend": float(beta[3 * P + i]),
                    "defend_se": float(se[3 * P + i]),
                    "level_serve": float(beta[i]),
                    "level_return": float(beta[P + i]),
                    "total": float(beta[2 * P + i] + beta[3 * P + i])})
    out.sort(key=lambda d: -d["total"])

    print("=" * 78)
    print("TOP 15 by TOTAL clutch (attack + defend), teammate-adjusted")
    print("=" * 78)
    print(f"{'player':24}{'g':>3}{'rallies':>9}{'attack':>10}{'defend':>10}{'total':>9}")
    for d in out[:15]:
        print(f"{d['name']:24}{d['gender']:>3}{d['n']:>9}"
              f"{d['attack']:+10.4f}{d['defend']:+10.4f}{d['total']:+9.4f}")

    print(f"\n{'-' * 78}\nTOP 10 DEFENDERS (clutch on the return side — invisible "
          f"to the frozen index)\n{'-' * 78}")
    for d in sorted(out, key=lambda d: -d["defend"])[:10]:
        print(f"{d['name']:24}{d['gender']:>3}{d['n']:>9}"
              f"{d['defend']:+10.4f}  (se {d['defend_se']:.4f}, "
              f"z {d['defend'] / d['defend_se']:+.1f})")

    print(f"\n{'-' * 78}\nTOP 10 ATTACKERS (clutch on serve — comparable to the "
          f"frozen index)\n{'-' * 78}")
    for d in sorted(out, key=lambda d: -d["attack"])[:10]:
        print(f"{d['name']:24}{d['gender']:>3}{d['n']:>9}"
              f"{d['attack']:+10.4f}  (se {d['attack_se']:.4f}, "
              f"z {d['attack'] / d['attack_se']:+.1f})")

    # --- are attack and defend the same trait?
    A = np.array([d["attack"] for d in out])
    D = np.array([d["defend"] for d in out])
    print(f"\ncorr(attack, defend) = {np.corrcoef(A, D)[0, 1]:+.3f}   "
          f"sd(attack) {A.std():.4f}  sd(defend) {D.std():.4f}")

    # --- how does this compare to the frozen server-only index?
    frozen = {}
    for r in csv.DictReader((ROOT / "data" / "clutch_players.csv").open()):
        frozen[r["name"]] = float(r["clutch"])
    pair = [(frozen[d["name"]], d["attack"], d["total"])
            for d in out if d["name"] in frozen]
    if len(pair) > 10:
        f = np.array([p[0] for p in pair])
        print(f"vs frozen index (n={len(pair)}): corr with attack "
              f"{np.corrcoef(f, [p[1] for p in pair])[0, 1]:+.3f}, "
              f"with total {np.corrcoef(f, [p[2] for p in pair])[0, 1]:+.3f}")

    vv = np.array([cur[d["uuid"]]["v"] for d in out])
    T = np.array([d["total"] for d in out])
    A = np.array([d["attack"] for d in out])
    # a strong player has BOTH a high serve level and a high return level
    # (return enters negated), so the skill-facing combination is the SUM.
    L = np.array([d["level_serve"] + d["level_return"] for d in out])
    print(f"\nSKILL CHECK  corr(v2, clutch total) = {np.corrcoef(vv, T)[0, 1]:+.3f}"
          f"   corr(v2, LEVEL total) = {np.corrcoef(vv, L)[0, 1]:+.3f}")
    print(f"  attack: mean {A.mean():+.4f} sd {A.std():.4f}  "
          f"pct [{np.percentile(A, 5):+.3f} {np.percentile(A, 50):+.3f} "
          f"{np.percentile(A, 95):+.3f}]")

    # ---- PERMUTATION NULL: shuffle levz WITHIN each game and refit.
    # This is the decisive check.  Everything except the leverage ordering is
    # preserved -- same players, same rallies, same outcomes, same offsets --
    # so any spread that survives is NOT a leverage effect.
    print("\nPERMUTATION NULL (levz shuffled within game, full refit):")
    rng = np.random.default_rng(17)
    bygame = defaultdict(list)
    for i, r in enumerate(rows):
        bygame[(r[7], r[8])].append(i)
    perm = list(rows)
    for idxs in bygame.values():
        vals = [rows[i][4] for i in idxs]
        rng.shuffle(vals)
        for i, v in zip(idxs, vals):
            perm[i] = perm[i][:4] + (v,) + perm[i][5:]
    nb, nse, _, _ = fit(perm, index, best_lam)
    nA = nb[2 * P:3 * P]
    nD = nb[3 * P:]
    print(f"    real  sd(attack) {A.std():.4f}   max |z| {np.max(np.abs(A / np.array([d['attack_se'] for d in out]))):.1f}")
    print(f"    null  sd(attack) {nA.std():.4f}   max |z| {np.max(np.abs(nA / nse[2 * P:3 * P])):.1f}")
    print(f"    ratio real/null  {A.std() / max(nA.std(), 1e-9):.2f}x"
          "   (1.0 = the whole spread is an artifact, not clutch)")

    dest = ROOT / "data" / "clutch_srm.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {dest}  ({len(out)} players)")


if __name__ == "__main__":
    main()
