"""Is "singles surplus" a real second dimension of skill?

    python model/singles_dimension.py            # both gates
    python model/singles_dimension.py --perms 400

THE IDEA.  v2 gives each player ONE number, and that scalar is a sufficient
statistic for doubles outcomes by construction.  Any decomposition of it
into physical / selection / strategy is likelihood-flat — the same class of
non-identifiability as the cross-gender offset.  Escaping it needs an
auxiliary channel that loads on the components DIFFERENTLY.

Singles is the best such channel in this archive: 26k games, and
singles~doubles r = 0.74, so ~45% of singles variance is orthogonal to
doubles.  Candidate dimension = SINGLES SURPLUS, a player's singles value
minus what their doubles value predicts.  Two gates decide whether it is a
dimension or just a rotation of the same number:

  GATE 1 — RELIABILITY.  Split-half.  The trap: the two halves must not
  share rating error.  If you residualise both halves against ONE doubles
  value, the shared error in that value lands in both residuals and fakes
  the correlation.  So BOTH disciplines are refit independently on each
  half, with the same estimator.  Read the answer against the reliability
  of the singles and doubles ratings themselves, not against zero — those
  set the ceiling.
  Second trap: SHRINKAGE.  A player with few singles games is pulled to the
  prior, which manufactures a surplus that correlates with game count — and
  game count is stable across halves, so it would fake reliability.  The
  residualisation therefore carries log-count covariates, and the run
  reports the number both with and without them.

  GATE 2 — INCREMENTAL VALIDITY, INSIDE DOUBLES.  Note what does NOT count:
  finding 6 (singles value predicts DreamBreakers better than the doubles
  proxy) is nearly tautological, because DreamBreakers ARE singles.  The
  real question is whether the surplus says anything about DOUBLES.  And
  the main effect cannot — the doubles rating already absorbs it, by
  construction.  So the test has to be an INTERACTION: does the surplus
  predict doubles performance in physically harder conditions relative to
  the same players in easier ones?  Two arms, both physical-flavoured and
  both with data in hand: game 3 of a match (fatigue) and match-hour heat.

GENDER.  Men and women never play each other in singles, so the two
singles scales are connected only through the prior and their relative
level is arbitrary.  Everything here is residualised WITHIN GENDER.

ORIENTATION.  games.csv is not side-neutral (t1 wins 67.8% — finding 11),
so the gate-2 panel randomises side orientation before fitting.

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import GAMMA, sigmoid  # noqa: E402

DATA = ROOT / "data"
SD_DOUBLES = 0.40        # per-point logit prior (v2 spread ~0.38)
SD_SINGLES = 0.60        # matches fit_singles.py; singles spreads wider
MIN_SINGLES = 50         # per player, TOTAL across both halves
MIN_DOUBLES = 100
ITERS = 250
OUT_MD = ROOT / "model" / "singles_dimension.md"
OUT_JSON = ROOT / "model" / "singles_dimension_summary.json"


# ------------------------------------------------------------------ data ---

def load_doubles():
    """(players4, s1, s2, meta) per graded doubles game."""
    out = []
    for g in csv.DictReader((DATA / "games.csv").open()):
        if g["is_dreambreaker"] in ("True", "1") or g["is_forfeit"] in ("True", "1"):
            continue
        try:
            s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        except ValueError:
            continue
        if s1 + s2 < 5:
            continue
        ps = [(g[k] or "").lower()
              for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(ps):
            continue
        out.append((ps, s1, s2, g))
    return out


def load_singles():
    out = []
    for r in csv.DictReader((DATA / "singles_games.csv").open()):
        if r["is_forfeit"] != "False":
            continue
        try:
            s1, s2 = int(r["s1"]), int(r["s2"])
        except ValueError:
            continue
        if s1 + s2 < 5:
            continue
        p1, p2 = r["p1"].lower(), r["p2"].lower()
        if not p1 or not p2:
            continue
        out.append((p1, p2, s1, s2, r["context"], r["date"]))
    return out


# --------------------------------------------------------------- fitters ---

def fit_singles(games, sd=SD_SINGLES, iters=ITERS):
    """Per-point Binomial MAP, v ~ N(0, sd). Same shape as fit_singles.py but
    with no recency weighting: the two halves must be exchangeable, and the
    object here is a career-average value, not current form."""
    idx = {}
    obs = []
    for p1, p2, s1, s2, _ctx, _dt in games:
        for p in (p1, p2):
            idx.setdefault(p, len(idx))
        obs.append((idx[p1], idx[p2], s1, s2))
    v = [0.0] * len(idx)
    pts = [0.0] * len(idx)
    for i, j, s1, s2 in obs:
        pts[i] += s1 + s2
        pts[j] += s1 + s2
    pre = [0.25 * t + 1.0 / sd ** 2 for t in pts]
    for _ in range(iters):
        grad = [-x / sd ** 2 for x in v]
        for i, j, s1, s2 in obs:
            p = sigmoid(v[i] - v[j])
            g = s1 - (s1 + s2) * p
            grad[i] += g
            grad[j] -= g
        for k in range(len(v)):
            v[k] += grad[k] / pre[k]
    return {p: v[i] for p, i in idx.items()}


def fit_doubles(games, sd=SD_DOUBLES, iters=ITERS, gamma=GAMMA):
    """Same likelihood with v2's team structure: team = sum + gamma*|gap|.
    gamma is held at v2's fitted value, not refit — this is a rating
    estimator for the split-half design, not a gamma study."""
    idx = {}
    obs = []
    for ps, s1, s2, _g in games:
        for p in ps:
            idx.setdefault(p, len(idx))
        obs.append((tuple(idx[p] for p in ps), s1, s2))
    v = [0.0] * len(idx)
    pts = [0.0] * len(idx)
    for ii, s1, s2 in obs:
        for k in ii:
            pts[k] += s1 + s2
    pre = [0.25 * t + 1.0 / sd ** 2 for t in pts]
    for _ in range(iters):
        grad = [-x / sd ** 2 for x in v]
        for (a, b, c, d), s1, s2 in obs:
            va, vb, vc, vd = v[a], v[b], v[c], v[d]
            # d(team)/d(v_stronger) = 1 + gamma ; d/d(v_weaker) = 1 - gamma
            ga_a = 1.0 + (gamma if va >= vb else -gamma)
            ga_b = 1.0 + (gamma if vb > va else -gamma)
            gc_c = 1.0 + (gamma if vc >= vd else -gamma)
            gc_d = 1.0 + (gamma if vd > vc else -gamma)
            eta = ((va + vb + gamma * abs(va - vb))
                   - (vc + vd + gamma * abs(vc - vd)))
            g = s1 - (s1 + s2) * sigmoid(eta)
            grad[a] += g * ga_a
            grad[b] += g * ga_b
            grad[c] -= g * gc_c
            grad[d] -= g * gc_d
        for k in range(len(v)):
            v[k] += grad[k] / pre[k]
    return {p: v[i] for p, i in idx.items()}


# --------------------------------------------------------------- helpers ---

def ols(rows, ykey, xkeys):
    p = len(xkeys) + 1
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for r in rows:
        x = [1.0] + [r[k] for k in xkeys]
        for i in range(p):
            xty[i] += x[i] * r[ykey]
            for j in range(p):
                xtx[i][j] += x[i] * x[j]
    m = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r_: abs(m[r_][c]))
        m[c], m[piv] = m[piv], m[c]
        if abs(m[c][c]) < 1e-12:
            return None
        for r_ in range(p):
            if r_ != c:
                f = m[r_][c] / m[c][c]
                for cc in range(c, p + 1):
                    m[r_][cc] -= f * m[c][cc]
    return [m[i][p] / m[i][i] for i in range(p)]


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def surplus(sing, doub, counts_s, counts_d, gender, players, covars=True):
    """Residualise singles on doubles WITHIN GENDER, optionally also on the
    log game counts that drive shrinkage. Returns {player: surplus}."""
    out = {}
    for gsex in ("M", "F"):
        rows = []
        for p in players:
            if gender.get(p) != gsex or p not in sing or p not in doub:
                continue
            rows.append({"p": p, "s": sing[p], "d": doub[p],
                         "ls": math.log(counts_s[p]), "ld": math.log(counts_d[p])})
        if len(rows) < 12:
            continue
        xk = ["d"] + (["ls", "ld"] if covars else [])
        c = ols(rows, "s", xk)
        if c is None:
            continue
        for r in rows:
            pred = c[0] + sum(c[i + 1] * r[k] for i, k in enumerate(xk))
            out[r["p"]] = r["s"] - pred
    return out


# ----------------------------------------------------------------- gates ---

def gate1(dbl, sng, rng, perms, split="random"):
    """Split-half reliability of the singles surplus.

    split='random'  — games dealt at random; measures pure estimation noise.
    split='era'     — 2024-25 vs 2026. Strictly harder: it also has to
                      survive ageing, form and a changing field, and the two
                      halves share no event context at all.
    """
    ha_d, hb_d, ha_s, hb_s = [], [], [], []
    if split == "era":
        for g in dbl:
            (hb_d if g[3]["date"][:4] == "2026" else ha_d).append(g)
        for g in sng:
            (hb_s if g[5][:4] == "2026" else ha_s).append(g)
    else:
        for g in dbl:
            (ha_d if rng.random() < 0.5 else hb_d).append(g)
        for g in sng:
            (ha_s if rng.random() < 0.5 else hb_s).append(g)

    gender, cs, cd = {}, defaultdict(int), defaultdict(int)
    for p1, p2, _s1, _s2, ctx, _dt in sng:
        gx = "F" if ctx == "womens_singles" else "M"
        gender[p1] = gx
        gender[p2] = gx
        cs[p1] += 1
        cs[p2] += 1
    for ps, _s1, _s2, _g in dbl:
        for p in ps:
            cd[p] += 1

    print(f"  fitting halves: doubles {len(ha_d)}/{len(hb_d)} games, "
          f"singles {len(ha_s)}/{len(hb_s)} games ...")
    dA, dB = fit_doubles(ha_d), fit_doubles(hb_d)
    sA, sB = fit_singles(ha_s), fit_singles(hb_s)

    elig = [p for p in cs
            if cs[p] >= MIN_SINGLES and cd[p] >= MIN_DOUBLES
            and p in dA and p in dB and p in sA and p in sB]
    csA, csB, cdA, cdB = (defaultdict(int) for _ in range(4))
    for h, c in ((ha_s, csA), (hb_s, csB)):
        for p1, p2, *_ in h:
            c[p1] += 1
            c[p2] += 1
    for h, c in ((ha_d, cdA), (hb_d, cdB)):
        for ps, *_ in h:
            for p in ps:
                c[p] += 1
    elig = [p for p in elig if min(csA[p], csB[p]) >= 10 and min(cdA[p], cdB[p]) >= 20]
    print(f"  eligible players: {len(elig)}")

    res = {}
    for covars in (True, False):
        rA = surplus(sA, dA, csA, cdA, gender, elig, covars)
        rB = surplus(sB, dB, csB, cdB, gender, elig, covars)
        common = [p for p in elig if p in rA and p in rB]
        xs = [rA[p] for p in common]
        ys = [rB[p] for p in common]
        r = corr(xs, ys)
        null = []
        for _ in range(perms):
            sh = ys[:]
            rng.shuffle(sh)
            null.append(corr(xs, sh))
        null.sort()
        lo, hi = null[int(0.025 * len(null))], null[int(0.975 * len(null))]
        res["with_covariates" if covars else "no_covariates"] = {
            "n": len(common), "r": r, "null95": [lo, hi],
            "p": (sum(1 for x in null if x >= r) + 1) / (len(null) + 1)}

    # ceilings: how reliable are the two ratings themselves?
    for lbl, A, B in (("singles_rating", sA, sB), ("doubles_rating", dA, dB)):
        xs = [A[p] for p in elig]
        ys = [B[p] for p in elig]
        res[lbl + "_reliability"] = corr(xs, ys)

    # full-data surplus for gate 2
    dF, sF = fit_doubles(dbl), fit_singles(sng)
    full = surplus(sF, dF, cs, cd, gender,
                   [p for p in cs if cs[p] >= MIN_SINGLES and cd[p] >= MIN_DOUBLES
                    and p in sF and p in dF], covars=True)
    return res, full, gender


def gate2(dbl, surp, rng, perms):
    """Does the surplus predict doubles performance CONDITIONALLY?

    Panel: doubles games where all four players have a surplus. x = (team A
    summed surplus - team B summed surplus). y = observed point share -
    v2-expected share. The MAIN effect must be ~0 (the doubles rating is
    sufficient); the test is whether the slope differs between hard and easy
    conditions. Side orientation randomised (finding 11)."""
    v2 = {}
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        v2[r["player_id"].lower()] = float(r["value_now_mean"])
    # match-hour temperature, if the weather join is present
    temp, start_hour = {}, {}
    hp = DATA / "event_weather_hourly.csv"
    if hp.exists():
        for r in csv.DictReader(hp.open()):
            try:                                  # same join as favorites_wind
                temp[(r["event_id"], r["local_time"][:13])] = \
                    float(r["temperature_2m"])
            except (TypeError, ValueError):
                pass
    mp = DATA / "match_times.csv"
    if mp.exists():
        for r in csv.DictReader(mp.open()):
            ts = r["start_local"] or r["planned_start_local"]
            if ts:
                start_hour[r["match_id"]] = ts[:13]

    bym = defaultdict(list)
    for ps, s1, s2, g in dbl:
        bym[g["match_id"]].append((ps, s1, s2, g))

    # SAME-DAY MATCH LOAD: pro players run several disciplines a day, so
    # "how many matches has this player already finished today" is a much
    # better-powered fatigue probe than deciders alone (it applies to every
    # game, not just game 3). Needs match start times.
    played_before = {}
    order = []
    for mid, gs in bym.items():
        g0 = gs[0][3]
        order.append((g0["date"], start_hour.get(mid, ""), mid, gs[0][0]))
    order.sort(key=lambda t: (t[0], t[1]))
    seen = defaultdict(int)
    for dt, _h, mid, ps in order:
        played_before[mid] = {p: seen[(dt, p)] for p in ps}
        for p in ps:
            seen[(dt, p)] += 1

    rows = []
    n_full = 0
    for mid, gs in bym.items():
        gs = sorted(gs, key=lambda t: int(t[3]["game_number"] or 1))
        for k, (ps, s1, s2, g) in enumerate(gs):
            if not all(p in v2 for p in ps):
                continue
            # A player with no singles record has no MEASURED surplus; 0 is
            # "average surplus", which is the honest imputation (it adds
            # noise, never a systematic tilt). Require at least half the
            # court measured so x is not mostly imputation.
            meas = [p for p in ps if p in surp]
            if len(meas) < 2:
                continue
            if len(meas) == 4:
                n_full += 1
            if rng.random() < 0.5:                      # neutralise t1 bias
                ps = ps[2:] + ps[:2]
                s1, s2 = s2, s1
            va, vb, vc, vd = (v2[p] for p in ps)
            eta = ((va + vb + GAMMA * abs(va - vb))
                   - (vc + vd + GAMMA * abs(vc - vd)))
            sp = [surp.get(p, 0.0) for p in ps]
            x = (sp[0] + sp[1]) - (sp[2] + sp[3])
            t = temp.get((g["event_id"], start_hour.get(mid, "")))
            pb = played_before.get(mid, {})
            load = max((pb.get(p, 0) for p in ps), default=0)
            rows.append({"y": s1 / (s1 + s2) - sigmoid(eta), "x": x,
                         "decider": 1.0 if (len(gs) >= 3 and k == 2) else 0.0,
                         "temp": t, "load": load, "ev": g["event_id"]})
    sdx = math.sqrt(sum(r["x"] ** 2 for r in rows) / len(rows)
                    - (sum(r["x"] for r in rows) / len(rows)) ** 2)
    print(f"  gate-2 panel: {len(rows)} games ({n_full} with all four "
          f"players measured), sd(x) = {sdx:.3f}")
    print(f"    {sum(1 for r in rows if r['decider']):.0f} deciders, "
          f"{sum(1 for r in rows if r['temp'] is not None)} with match-hour "
          f"temp, {sum(1 for r in rows if r['load'] >= 2)} at same-day load >= 2")

    def slope_diff(rs, key, hot):
        """Slope of y on x among 'hot' rows minus among the rest."""
        a = [r for r in rs if hot(r)]
        b = [r for r in rs if not hot(r)]
        if len(a) < 200 or len(b) < 200:
            return None
        ca, cb = ols(a, "y", ["x"]), ols(b, "y", ["x"])
        if ca is None or cb is None:
            return None
        return {"slope_hard": ca[1], "slope_easy": cb[1],
                "diff": ca[1] - cb[1], "n_hard": len(a), "n_easy": len(b)}

    def boot(rs, hot, n=400):
        clus = defaultdict(list)
        for r in rs:
            clus[r["ev"]].append(r)
        keys = list(clus)
        out = []
        for _ in range(n):
            s = []
            for _ in keys:
                s.extend(clus[rng.choice(keys)])
            d = slope_diff(s, None, hot)
            if d:
                out.append(d["diff"])
        out.sort()
        return (out[int(0.025 * len(out))], out[int(0.975 * len(out))]) if out else None

    res = {"sd_x": sdx, "n_games": len(rows)}
    main = ols(rows, "y", ["x"])
    res["main_effect_slope"] = main[1] if main else None

    # --- WITHIN-MATCH decider arm (the one to trust) ----------------------
    # The naive between-games decider contrast returns a marginal negative
    # slope, and it is an ARTIFACT: reaching 1-1 selects on the match-level
    # shock (clutch_decider.py), which is common to all three games. An eta
    # control does NOT remove it (the artifact is the match shock, not the
    # skill gap). Differencing INSIDE the match cancels it exactly.
    wm = []
    for mid, gs in bym.items():
        gs = sorted(gs, key=lambda t: int(t[3]["game_number"] or 1))
        if len(gs) < 3:
            continue
        ps0 = gs[0][0]
        if not all(p in v2 for p in ps0) or len([p for p in ps0 if p in surp]) < 2:
            continue
        flip = rng.random() < 0.5
        rr, ok = [], True
        for ps, s1, s2, g in gs[:3]:
            if set(ps) != set(ps0):
                ok = False
                break
            pp = list(ps)
            a, b = s1, s2
            if flip:
                pp, a, b = pp[2:] + pp[:2], b, a
            va, vb, vc, vd = (v2[p] for p in pp)
            eta = ((va + vb + GAMMA * abs(va - vb))
                   - (vc + vd + GAMMA * abs(vc - vd)))
            rr.append(a / (a + b) - sigmoid(eta))
        if not ok:
            continue
        pp = list(ps0)
        if flip:
            pp = pp[2:] + pp[:2]
        sp = [surp.get(p, 0.0) for p in pp]
        wm.append({"y": rr[2] - (rr[0] + rr[1]) / 2.0,
                   "x": (sp[0] + sp[1]) - (sp[2] + sp[3]),
                   "ev": gs[0][3]["event_id"]})
    if wm:
        c = ols(wm, "y", ["x"])
        clus = defaultdict(list)
        for r in wm:
            clus[r["ev"]].append(r)
        keys = list(clus)
        draws = []
        for _ in range(400):
            s = []
            for _ in keys:
                s.extend(clus[rng.choice(keys)])
            cc = ols(s, "y", ["x"])
            if cc:
                draws.append(cc[1])
        draws.sort()
        res["decider_within_match"] = {
            "slope": c[1], "n": len(wm),
            "ci95": [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]}
        ci = res["decider_within_match"]["ci95"]
        print(f"  {'decider (within-match)':22s} slope {c[1]:+.4f} "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  (n={len(wm)} three-game matches)"
              "   <- the trustworthy version")

    arms = [("decider (naive)", lambda r: r["decider"] > 0.5, rows),
            ("same-day load>=2", lambda r: r["load"] >= 2, rows)]
    hot_rows = [r for r in rows if r["temp"] is not None]
    if hot_rows:
        ts = sorted(r["temp"] for r in hot_rows)
        cut = ts[int(0.75 * len(ts))]
        arms.append((f"heat>{cut:.0f}F", lambda r: r["temp"] >= cut, hot_rows))
    for name, hot, rs in arms:
        d = slope_diff(rs, None, hot)
        if d is None:
            continue
        d["ci95"] = boot(rs, hot)
        res[name] = d
        ci = d["ci95"]
        print(f"  {name:14s} slope hard {d['slope_hard']:+.4f} (n={d['n_hard']}) "
              f"vs easy {d['slope_easy']:+.4f} (n={d['n_easy']})  "
              f"diff {d['diff']:+.4f}" +
              (f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    dbl, sng = load_doubles(), load_singles()
    print(f"doubles games {len(dbl)}, singles games {len(sng)}")

    print("\nGATE 1 — split-half reliability of the singles surplus")
    g1, surp, gender = gate1(dbl, sng, rng, args.perms)
    for k in ("singles_rating_reliability", "doubles_rating_reliability"):
        print(f"  {k:32s} r = {g1[k]:+.3f}")
    for k in ("with_covariates", "no_covariates"):
        d = g1[k]
        print(f"  surplus reliability ({k:15s}) r = {d['r']:+.3f}  "
              f"null95 [{d['null95'][0]:+.3f}, {d['null95'][1]:+.3f}]  "
              f"p={d['p']:.3f}  (n={d['n']})")

    # Harder version of the same gate: the two halves are ERAS, so the
    # surplus must also survive ageing, form and a changing field, and the
    # halves share no event context at all.
    print("\nGATE 1b — ERA split (2024-25 vs 2026), the harder test")
    g1b, _s, _g = gate1(dbl, sng, rng, args.perms, split="era")
    for k in ("singles_rating_reliability", "doubles_rating_reliability"):
        print(f"  {k:32s} r = {g1b[k]:+.3f}")
    for k in ("with_covariates", "no_covariates"):
        d = g1b[k]
        print(f"  surplus reliability ({k:15s}) r = {d['r']:+.3f}  "
              f"null95 [{d['null95'][0]:+.3f}, {d['null95'][1]:+.3f}]  "
              f"p={d['p']:.3f}  (n={d['n']})")

    print("\nGATE 2 — incremental validity inside doubles")
    g2 = gate2(dbl, surp, rng, args.perms)
    print(f"  main effect (must be ~0, rating is sufficient): "
          f"{g2['main_effect_slope']:+.5f}")

    OUT_JSON.write_text(json.dumps(
        {"gate1": g1, "gate1_era": g1b, "gate2": g2,
         "n_surplus_players": len(surp),
         "min_singles": MIN_SINGLES, "min_doubles": MIN_DOUBLES}, indent=2))
    with (DATA / "singles_surplus.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "gender", "singles_surplus"])
        for p, s in sorted(surp.items(), key=lambda kv: -kv[1]):
            w.writerow([p, gender.get(p, ""), f"{s:.4f}"])
    print(f"\nwrote {OUT_JSON.name}, data/singles_surplus.csv")


if __name__ == "__main__":
    main()
