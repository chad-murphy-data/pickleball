"""Big points and big games — the out-of-the-box corner of the shootout.

Two player skills folklore swears by, measured properly for the first
time in pro pickleball:

  CLUTCH ("wins big points").  Every logged rally gets an EXACT leverage
  number: reconstruct the full state (score AND serve state — A#1 / A#2 /
  B#1 / B#2) from the referee log, then ask the serve-aware DP
  (web/sitelib/winprob.py) how much the game's win probability swings on
  that rally.  A rally at 9-9 on second serve is objectively bigger than
  one at 3-1; the DP quantifies by how much, using the sport's own
  side-out structure.  Clutch for a server = covariance between their
  rally outcomes (vs matchup expectation) and within-game leverage
  z-scores.  Tested for existence (permutation null: shuffle leverage
  within game), persistence (odd/even match split), and predictive value.

  DURABILITY ("more durable against stronger opposition").  Per player:
  the slope of their team's per-game residual (observed minus expected
  point share, v2 expectation) on opponent pairing strength.  Positive
  slope = raises their game against the strong ("big-game player");
  negative = flat-track bully.  Same trio of tests.

Run:  python model/big_points.py            # durability (games only)
      python model/big_points.py --clutch   # + clutch (needs raw/match_logs)

Writes model/big_points_summary.json; prose lives in
model/spec_shootout.md.  Reuses spec_shootout's loaders/frames so the
predictive rows land on the identical 926-game holdout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))

import spec_shootout as sx  # noqa: E402

RALLY, POINT, SIDEOUT, SECOND = 12, 14, 16, 23
MIN_GAMES_DUR = 60
MIN_RALLIES_CL = 300
N_PERM = 200
SEED = 20260718


# ------------------------------------------------------------ durability --


def durability(games, players, chem, racer):
    train = [g for g in games if g["date"] < sx.SPLIT]
    rows = defaultdict(list)      # player -> [(opp_strength, signed resid, date)]
    strengths = []

    def team_val(u1, u2):
        v1, v2 = players[u1]["v"], players[u2]["v"]
        return v1 + v2 + sx.GAMMA_V2 * abs(v1 - v2)

    for g in train:
        if any(u not in players for u in g["us"]):
            continue
        t1 = team_val(g["us"][0], g["us"][1])
        t2 = team_val(g["us"][2], g["us"][3])
        c1 = chem.get(frozenset((players[g["us"][0]]["name"],
                                 players[g["us"][1]]["name"])), 0.0)
        c2 = chem.get(frozenset((players[g["us"][2]]["name"],
                                 players[g["us"][3]]["name"])), 0.0)
        eta = t1 - t2 + c1 - c2
        share = g["s1"] / (g["s1"] + g["s2"])
        resid = share - 1.0 / (1.0 + math.exp(-eta))
        strengths.extend([t1, t2])
        for j, u in enumerate(g["us"]):
            opp = t2 if j < 2 else t1
            rows[u].append((opp, resid if j < 2 else -resid, g["date"]))

    mu, sd = float(np.mean(strengths)), float(np.std(strengths))
    rng = np.random.default_rng(SEED)
    per_player = {}
    for u, obs in rows.items():
        if len(obs) < MIN_GAMES_DUR:
            continue
        x = (np.array([o[0] for o in obs]) - mu) / sd
        y = np.array([o[1] for o in obs])
        if x.std() < 1e-6:
            continue

        def ols_slope(xv, yv):
            xc = xv - xv.mean()
            den = (xc ** 2).sum()
            return float((xc * yv).sum() / den) if den > 1e-9 else np.nan

        slope = ols_slope(x, y)
        # permutation null: break the (x, y) pairing within the player
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            null[i] = ols_slope(x, rng.permutation(y))
        # split-half by date for reliability
        order = np.argsort([o[2] for o in obs])
        h1, h2 = order[: len(order) // 2], order[len(order) // 2:]
        s1, s2 = ols_slope(x[h1], y[h1]), ols_slope(x[h2], y[h2])
        per_player[u] = dict(
            name=players[u]["name"], n=len(obs), slope=slope,
            null_sd=float(null.std()), z=slope / float(null.std()),
            half1=s1, half2=s2)

    zs = np.array([p["z"] for p in per_player.values()])
    h1 = np.array([p["half1"] for p in per_player.values()])
    h2 = np.array([p["half2"] for p in per_player.values()])
    ok = np.isfinite(h1) & np.isfinite(h2)
    rel = float(np.corrcoef(h1[ok], h2[ok])[0, 1]) if ok.sum() > 10 else None
    # confound check: if the model is globally mis-curved, better players
    # get positive slopes MECHANICALLY (more residual headroom vs strong
    # opposition).  Regress slope on own value; the residual dispersion is
    # the idiosyncratic "trait" part.
    vv = np.array([players[u]["v"] for u in per_player])
    sl = np.array([p["slope"] for p in per_player.values()])
    ns = np.array([p["null_sd"] for p in per_player.values()])
    b, a = np.polyfit(vv, sl, 1)
    z_resid = (sl - (a + b * vv)) / ns
    hr1 = (h1 - (a + b * vv)) if rel is not None else h1
    hr2 = (h2 - (a + b * vv)) if rel is not None else h2
    okr = np.isfinite(hr1) & np.isfinite(hr2)
    rel_resid = float(np.corrcoef(hr1[okr], hr2[okr])[0, 1]) if okr.sum() > 10 else None
    # excess dispersion: Var(z) > 1 means real heterogeneity beyond noise
    summary = dict(
        n_players=len(per_player),
        z_var=float(zs.var()), z_mean=float(zs.mean()),
        split_half_r=rel,
        slope_vs_value=dict(corr=float(np.corrcoef(vv, sl)[0, 1]),
                            trend_per_logit=float(b)),
        z_var_residual=float(z_resid.var()),
        split_half_r_residual=rel_resid,
        top_biggame=sorted(
            ({"name": p["name"], "slope": round(p["slope"], 4),
              "z": round(p["z"], 2), "n": p["n"]}
             for p in per_player.values()), key=lambda d: -d["z"])[:8],
        top_flattrack=sorted(
            ({"name": p["name"], "slope": round(p["slope"], 4),
              "z": round(p["z"], 2), "n": p["n"]}
             for p in per_player.values()), key=lambda d: d["z"])[:8])
    return per_player, summary, (mu, sd)


def durability_strategy(games, players, chem, racer, per_player, norm):
    """Predictive row: weak_gamma + b * durability adjustment."""
    mu, sd = norm
    slope = {u: p["slope"] for u, p in per_player.items()}

    def team_val(u1, u2):
        v1, v2 = players[u1]["v"], players[u2]["v"]
        return v1 + v2 + sx.GAMMA_V2 * abs(v1 - v2)

    def feature(gs):
        out = np.zeros(len(gs))
        for i, g in enumerate(gs):
            t1 = team_val(g["us"][0], g["us"][1])
            t2 = team_val(g["us"][2], g["us"][3])
            s1 = np.mean([slope.get(u, 0.0) for u in g["us"][:2]])
            s2 = np.mean([slope.get(u, 0.0) for u in g["us"][2:]])
            out[i] = s1 * (t2 - mu) / sd - s2 * (t1 - mu) / sd
        return out

    train = [g for g in games if sx.RECENT <= g["date"] < sx.SPLIT]
    hold = [g for g in games if g["date"] >= sx.SPLIT
            and all(u in players and players[u]["games"] >= sx.MIN_TRAIN_GAMES
                    for u in g["us"])]
    Ftr, Fho = sx.Frame(train, players, chem), sx.Frame(hold, players, chem)
    ftr, fho = feature(train), feature(hold)

    from scipy.optimize import minimize

    def eta(frame, f, x):
        return x[0] * (frame.d_sum() + x[1] * frame.d_gap()) + x[2] * f

    def nll(x):
        pw = racer.win(eta(Ftr, ftr, x), Ftr.T)
        pc = np.clip(pw, 1e-9, 1 - 1e-9)
        return -np.mean(np.where(Ftr.won, np.log(pc), np.log(1 - pc)))

    x = minimize(nll, [1.0, sx.GAMMA_V2, 0.0], method="Nelder-Mead",
                 options=dict(xatol=1e-4)).x
    pw = racer.win(eta(Fho, fho, x), Fho.T)
    ref = racer.win(Fho.eta_v2(), Fho.T)
    m = sx.metrics(pw, Fho.won)
    m["params"] = [round(float(t), 4) for t in x]
    m["vs_ref"] = sx.paired_boot(pw, ref, Fho.won)
    return m


# --------------------------------------------------------------- clutch --


def reconstruct_states(rows, sides, games_by_no):
    """One match's log -> per rally: (game_no, a, b, state, server_uuid,
    server_side, won).  Score is tracked by the state machine itself
    (side-out scoring: only the serving side scores); games whose final
    reconstructed score disagrees with the archive are discarded upstream.
    States follow winprob.py: 0=A#1 1=A#2 2=B#1 3=B#2; A = archive team 1;
    games open on the starting side's #2 server."""
    out = []
    by_game = defaultdict(list)
    for r in sorted(rows, key=lambda x: x.get("log_index", 0)):
        if r.get("log_type") in (RALLY, POINT, SIDEOUT, SECOND):
            by_game[r.get("game_number")].append(r)
    for gno, ev in by_game.items():
        if gno not in games_by_no:
            continue
        a = b = 0
        state = None
        current = None
        seq = []
        bad = False
        for r in ev:
            t = r.get("log_type")
            if t == RALLY:
                current = r
                continue
            if current is None:
                continue
            su = (current.get("server_uuid") or "").lower()
            side = 0 if su in sides[0] else 1 if su in sides[1] else None
            if side is None:
                current = None
                continue
            if state is None:                 # opening: #2 of starting side
                state = 1 if side == 0 else 3
            exp_side = 0 if state in (0, 1) else 1
            if side != exp_side:
                # resync on the log's own attribution (rare referee fixups;
                # the end-of-game score check discards anything garbled)
                state = 1 if side == 0 else 3
            won = t == POINT
            seq.append((gno, a, b, state, su, side, won))
            if t == POINT:
                if side == 0:
                    a += 1
                else:
                    b += 1
            elif t == SECOND:
                if state in (0, 2):
                    state += 1
                else:
                    bad = True
                    break
            elif t == SIDEOUT:
                state = 2 if state in (0, 1) else 0
            current = None
        exp = games_by_no[gno]
        if not bad and (a, b) == exp:
            out.extend(seq)
    return out


def leverage_of(V, T, a, b, state, side_A_serving):
    """|win prob swing| of this rally, team-A perspective, exact."""
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


def clutch(games, players, chem):
    from sitelib.winprob import serve_probs, _table
    raw = ROOT / "raw" / "match_logs"
    train = [g for g in games if sx.RECENT <= g["date"] < sx.SPLIT]
    by_match = defaultdict(list)
    for g in train:
        by_match[g["match"]].append(g)

    def eta_of(g):
        if any(u not in players for u in g["us"]):
            return None
        v = [players[u]["v"] for u in g["us"]]
        c1 = chem.get(frozenset((players[g["us"][0]]["name"],
                                 players[g["us"][1]]["name"])), 0.0)
        c2 = chem.get(frozenset((players[g["us"][2]]["name"],
                                 players[g["us"][3]]["name"])), 0.0)
        return (v[0] + v[1] + sx.GAMMA_V2 * abs(v[0] - v[1])
                - v[2] - v[3] - sx.GAMMA_V2 * abs(v[2] - v[3]) + c1 - c2)

    k_league = 0.443          # measured on this harvest (printed by --rally)
    recs = []                 # (server, match, levz-slot, resid) built per game
    n_ok = n_bad = 0
    for m, gs in by_match.items():
        p = raw / m[:2] / f"{m}.json"
        if not p.exists():
            continue
        body = json.loads(p.read_text())
        rows = body.get("data") if isinstance(body, dict) else body
        if not rows:
            continue
        sides = (frozenset(u.lower() for u in gs[0]["us"][:2]),
                 frozenset(u.lower() for u in gs[0]["us"][2:]))
        games_by_no = {g["gn"]: (g["s1"], g["s2"]) for g in gs}
        eta_by_no = {g["gn"]: eta_of(g) for g in gs}
        T_by_no = {g["gn"]: g["T"] for g in gs}
        seq = reconstruct_states(rows, sides, games_by_no)
        if not seq:
            n_bad += 1
            continue
        n_ok += 1
        by_game = defaultdict(list)
        for s in seq:
            by_game[s[0]].append(s)
        for gno, rallies in by_game.items():
            e = eta_by_no.get(gno)
            if e is None:
                continue
            kA, kB = serve_probs(e, k_league)
            T = T_by_no[gno]
            V = _table(round(kA, 6), round(kB, 6), T, T + 40)
            levs, srv_expect = [], []
            for (_, a, b, state, su, side, won) in rallies:
                levs.append(leverage_of(V, T, a, b, state, side == 0))
                srv_expect.append(kA if side == 0 else kB)
            levs = np.array(levs)
            if levs.std() < 1e-9:
                continue
            levz = (levs - levs.mean()) / levs.std()
            for (rly, lz, ke) in zip(rallies, levz, srv_expect):
                _, a, b, state, su, side, won = rly
                recs.append((su, m, float(lz), float(won) - ke))
    print(f"clutch: reconstructed {n_ok} matches ok, {n_bad} discarded "
          f"(score mismatch/empty), {len(recs)} server-rallies")

    by_p = defaultdict(list)
    for su, m, lz, res in recs:
        by_p[su].append((m, lz, res))
    rng = np.random.default_rng(SEED)
    per_player = {}
    for u, obs in by_p.items():
        if len(obs) < MIN_RALLIES_CL or u not in players:
            continue
        lz = np.array([o[1] for o in obs])
        res = np.array([o[2] for o in obs])
        cl = float(np.mean(lz * res))
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            null[i] = np.mean(rng.permutation(lz) * res)
        matches = sorted({o[0] for o in obs})
        pick = {m: i % 2 for i, m in enumerate(matches)}
        e1 = [lz[i] * res[i] for i, o in enumerate(obs) if pick[o[0]] == 0]
        e2 = [lz[i] * res[i] for i, o in enumerate(obs) if pick[o[0]] == 1]
        per_player[u] = dict(
            name=players[u]["name"], n=len(obs), clutch=cl,
            null_sd=float(null.std()),
            z=cl / float(null.std()) if null.std() > 0 else 0.0,
            half1=float(np.mean(e1)) if e1 else np.nan,
            half2=float(np.mean(e2)) if e2 else np.nan)

    zs = np.array([p["z"] for p in per_player.values()])
    h1 = np.array([p["half1"] for p in per_player.values()])
    h2 = np.array([p["half2"] for p in per_player.values()])
    ok = np.isfinite(h1) & np.isfinite(h2)
    rel = float(np.corrcoef(h1[ok], h2[ok])[0, 1]) if ok.sum() > 10 else None
    # is "clutch" just "good"?  close-game winners won the big rallies by
    # construction, so skill leaks into the metric; regress it out.
    corr_v = z_var_resid = None
    if len(per_player) > 10:
        vv = np.array([players[u]["v"] for u in per_player])
        cl = np.array([p["clutch"] for p in per_player.values()])
        ns = np.array([p["null_sd"] for p in per_player.values()])
        b, a = np.polyfit(vv, cl, 1)
        corr_v = float(np.corrcoef(vv, cl)[0, 1])
        z_var_resid = float(((cl - (a + b * vv)) / ns).var())
    summary = dict(
        clutch_vs_value_corr=corr_v, z_var_residual=z_var_resid,
        n_matches_ok=n_ok, n_discarded=n_bad, n_server_rallies=len(recs),
        n_players=len(per_player), z_var=float(zs.var()) if len(zs) else None,
        z_mean=float(zs.mean()) if len(zs) else None, split_half_r=rel,
        top_clutch=sorted(
            ({"name": p["name"], "clutch": round(p["clutch"], 4),
              "z": round(p["z"], 2), "n": p["n"]}
             for p in per_player.values()), key=lambda d: -d["z"])[:8],
        top_anticlutch=sorted(
            ({"name": p["name"], "clutch": round(p["clutch"], 4),
              "z": round(p["z"], 2), "n": p["n"]}
             for p in per_player.values()), key=lambda d: d["z"])[:8])
    return per_player, summary


def biggest_points(k=0.443, T=11):
    """The DP's own answer to 'which point is biggest?' — top leverage
    states between equal teams (flavor for the writeup)."""
    from sitelib.winprob import _table
    V = _table(k, k, T, T + 40)
    out = []
    names = {0: "A#1", 1: "A#2", 2: "B#1", 3: "B#2"}
    for (a, b, s), _ in V.items():
        if a <= T and b <= T:
            lev = leverage_of(V, T, a, b, s, s in (0, 1))
            out.append((round(lev, 3), f"{a}-{b} {names[s]}"))
    out.sort(reverse=True)
    dedup = []
    for lev, tag in out:
        if len(dedup) < 6:
            dedup.append({"state": tag, "swing": lev})
    return dedup


# ----------------------------------------------------------------- main --


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clutch", action="store_true",
                    help="include the rally-level clutch analysis "
                         "(needs raw/match_logs)")
    args = ap.parse_args()

    games = sx.load_games()
    players, chem = sx.load_v2()
    racer = sx.Racer()

    out = {"generated": "2026-07-18"}

    per, summ, norm = durability(games, players, chem, racer)
    print(f"durability: {summ['n_players']} players, z-var={summ['z_var']:.2f} "
          f"(1.0 = pure noise), split-half r={summ['split_half_r']}")
    row = durability_strategy(games, players, chem, racer, per, norm)
    print(f"A_durability: acc={row['accuracy']:.4f} brier={row['brier']:.4f} "
          f"b={row['params'][2]}")
    out["durability"] = dict(summary=summ, strategy_row=row)

    if args.clutch:
        perc, summc = clutch(games, players, chem)
        print(f"clutch: {summc['n_players']} servers, z-var={summc['z_var']}, "
              f"split-half r={summc['split_half_r']}")
        out["clutch"] = dict(summary=summc)
        out["biggest_points"] = biggest_points()

    (ROOT / "model" / "big_points_summary.json").write_text(
        json.dumps(out, indent=1))
    print("wrote model/big_points_summary.json")


if __name__ == "__main__":
    main()
