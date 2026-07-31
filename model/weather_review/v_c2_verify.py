"""ADVERSARIAL VERIFICATION of task C2 (wind as an out-of-sample predictor).

Independent re-derivation: own race DP (recursive, cross-checked against a
Monte-Carlo), own weather join, own optimizer (finite-difference damped
Newton, not Nelder-Mead), own bootstrap seeds and fold seeds.

Nothing is imported from windlib except the path constants.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
SPLIT = "2026-06-01"
RECENT = "2026-01-01"
MIN_TRAIN_GAMES = 10
GAMMA = -0.16596178710460663
VSEED = 987654321          # deliberately different from the tester's 20260731


# ------------------------------------------------------------- race DP --
# Independent implementation: forward recursion over states with an explicit
# deuce absorbing state, written from the rules rather than copied.

def race_prob(p, T):
    """P(A reaches T with a 2-point margin), A wins each point w.p. p."""
    q = 1.0 - p
    # states (a,b) with a,b <= T-1 reachable before someone wins
    prob = {(0, 0): 1.0}
    win = 0.0
    deuce = 0.0
    for tot in range(0, 2 * T):
        cur = [(a, b) for (a, b) in prob if a + b == tot]
        for (a, b) in cur:
            m = prob.pop((a, b))
            if m == 0.0:
                continue
            # A scores
            if a + 1 >= T and (a + 1) - b >= 2:
                win += m * p
            elif a + 1 == T - 1 and b == T - 1:
                deuce += m * p
            else:
                prob[(a + 1, b)] = prob.get((a + 1, b), 0.0) + m * p
            # B scores
            if b + 1 >= T and (b + 1) - a >= 2:
                pass                       # A loses
            elif a == T - 1 and b + 1 == T - 1:
                deuce += m * q
            else:
                prob[(a, b + 1)] = prob.get((a, b + 1), 0.0) + m * q
    # from deuce (T-1,T-1): A must win by 2 -> p^2/(p^2+q^2)
    return win + deuce * (p * p / (p * p + q * q)) if (p * p + q * q) > 0 else win


def mc_race(p, T, n=400000, seed=1):
    rng = np.random.default_rng(seed)
    a = b = 0
    wins = 0
    for _ in range(n):
        a = b = 0
        while True:
            if rng.random() < p:
                a += 1
            else:
                b += 1
            if a >= T and a - b >= 2:
                wins += 1
                break
            if b >= T and b - a >= 2:
                break
    return wins / n


class Race:
    def __init__(self, m=1401):
        self.grid = np.linspace(0.005, 0.995, m)
        self.tab = {T: np.array([race_prob(float(p), T) for p in self.grid])
                    for T in (11, 15)}

    def win(self, eta, T):
        p = 1.0 / (1.0 + np.exp(-np.asarray(eta, float)))
        out = np.empty_like(p)
        for t in (11, 15):
            m = (T == t)
            if m.any():
                out[m] = np.interp(p[m], self.grid, self.tab[t])
        return out


# ---------------------------------------------------------------- data --

def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_all(label="override"):
    pl = {r["player_id"]: (float(r["value_now_mean"]), int(r["games"]),
                           r["full_name"]) for r in rd(DATA / "v2_players_train.csv")}
    chem = {frozenset((r["p1_name"], r["p2_name"])): float(r["chem_logit_mean"])
            for r in rd(DATA / "v2_dyads_train.csv")}
    traj = {}
    for r in rd(DATA / "v2_trajectories_train.csv"):
        traj.setdefault(r["player_id"], {})[r["month"]] = float(r["value_mean"])

    geo = {r["event_id"].lower(): r.get("setting", "unknown")
           for r in rd(DATA / "event_geo.csv")}
    ovr = {r["event_id"].lower(): r["setting"] for r in rd(DATA / "venue_overrides.csv")}
    daily = {(r["event_id"].lower(), r["date"]): r for r in rd(DATA / "event_weather.csv")}
    hourly = {(r["event_id"].lower(), r["local_time"][:13]): r
              for r in rd(DATA / "event_weather_hourly.csv")}
    times = {}
    for r in rd(DATA / "match_times.csv"):
        times[r["match_id"].lower()] = (r["start_local"] or r["planned_start_local"],
                                        bool(r["start_local"]))

    games = []
    for g in rd(DATA / "games.csv"):
        if g["is_forfeit"] != "False":
            continue
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        us = (g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"])
        if not all(u in pl and pl[u][1] >= MIN_TRAIN_GAMES for u in us):
            continue
        ev = g["event_id"].lower()
        mt = g["match_id"].lower()
        hs = times.get(mt)
        wind = gust = temp = None
        src = "none"
        if hs and hs[0]:
            row = hourly.get((ev, hs[0][:13]))
            if row is not None:
                wind, gust, temp = (fnum(row["windspeed_10m"]),
                                    fnum(row["windgusts_10m"]),
                                    fnum(row["temperature_2m"]))
                src = "hour_actual" if hs[1] else "hour_planned"
        if wind is None:
            row = daily.get((ev, g["date"]))
            if row is not None:
                wind, gust, temp = (fnum(row["windspeed_10m_max"]),
                                    fnum(row["windgusts_10m_max"]),
                                    fnum(row["temperature_2m_max"]))
                src = "day"
        if wind is None:
            continue
        setting = (ovr.get(ev) or geo.get(ev, "unknown")) if label == "override" \
            else geo.get(ev, "unknown")
        games.append(dict(date=g["date"], event=ev, match=mt, us=us,
                          T=11 if g["scoring_format"] == "sideout_11" else 15,
                          won=int(g["margin"]) > 0, wind=wind, gust=gust,
                          temp=temp if temp is not None else 75.0,
                          src=src, setting=setting))
    games.sort(key=lambda r: (r["date"], r["match"]))
    return games, pl, chem, traj


def eta_of(games, pl, chem, traj=None):
    out = np.empty(len(games))
    for i, g in enumerate(games):
        us = g["us"]
        if traj is None:
            v = [pl[u][0] for u in us]
        else:
            mo = g["date"][:7]
            v = []
            for u in us:
                t = traj.get(u)
                if not t:
                    v.append(pl[u][0])
                    continue
                if mo in t:
                    v.append(t[mo])
                else:
                    ks = sorted(t)
                    v.append(t[ks[0]] if mo < ks[0] else t[ks[-1]])
        c1 = chem.get(frozenset((pl[us[0]][2], pl[us[1]][2])), 0.0)
        c2 = chem.get(frozenset((pl[us[2]][2], pl[us[3]][2])), 0.0)
        out[i] = (v[0] + v[1] + GAMMA * abs(v[0] - v[1])
                  - v[2] - v[3] - GAMMA * abs(v[2] - v[3]) + c1 - c2)
    return out


def feats(games):
    w = np.array([g["wind"] for g in games], float)
    gu = np.array([g["gust"] for g in games], float)
    tp = np.array([g["temp"] for g in games], float)
    o = np.array([g["setting"] == "outdoor" for g in games], float)
    return dict(W=o * (w - 8.0) / 10.0, G=o * (gu - 14.0) / 10.0,
                TH=o * (w >= 14.0), TP=o * (tp - 75.0) / 10.0,
                wind=w, out=o)


# --------------------------------------------------------------- models --

def pred(name, x, eta, F, T, race):
    if name == "base":
        return race.win(eta, T)
    if name == "scale":
        return race.win(x[0] * eta, T)
    if name == "a":
        return race.win(x[0] * eta * (1.0 + x[1] * F["W"]), T)
    if name == "c":
        return race.win(x[0] * eta * (1.0 + x[1] * F["G"]), T)
    if name == "d":
        return race.win(x[0] * eta * (1.0 + x[1] * F["TH"]), T)
    if name == "e":
        p = np.clip(race.win(x[0] * eta, T), 1e-9, 1 - 1e-9)
        z = np.log(p / (1 - p)) * (1.0 + x[1] * F["W"])
        return 1.0 / (1.0 + np.exp(-z))
    if name == "f":
        return race.win(x[0] * eta * (1.0 + x[1] * F["TP"]), T)
    raise KeyError(name)


def newton_fit(name, eta, F, T, won, race, x0):
    """Damped finite-difference Newton (independent of the tester's NM)."""
    x = np.array(x0, float)
    n = len(x)

    def nll(v):
        p = np.clip(pred(name, v, eta, F, T, race), 1e-9, 1 - 1e-9)
        return -float(np.mean(np.where(won, np.log(p), np.log(1 - p))))

    f0 = nll(x)
    h = 1e-4
    for _ in range(60):
        g = np.zeros(n)
        H = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n); e[i] = h
            fp, fm = nll(x + e), nll(x - e)
            g[i] = (fp - fm) / (2 * h)
            H[i, i] = (fp - 2 * f0 + fm) / h ** 2
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros(n); ei[i] = h
                ej = np.zeros(n); ej[j] = h
                H[i, j] = H[j, i] = (nll(x + ei + ej) - nll(x + ei - ej)
                                     - nll(x - ei + ej) + nll(x - ei - ej)) / (4 * h ** 2)
        # regularise to positive definite
        lam = 1e-8
        for _ in range(50):
            try:
                step = np.linalg.solve(H + lam * np.eye(n), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            if np.dot(step, g) < 0:
                break
            lam *= 10
        else:
            break
        t = 1.0
        for _ in range(30):
            xn = x + t * step
            fn = nll(xn)
            if fn < f0 - 1e-14:
                break
            t *= 0.5
        else:
            break
        if abs(f0 - fn) < 1e-12 and np.max(np.abs(t * step)) < 1e-7:
            x, f0 = xn, fn
            break
        x, f0 = xn, fn
    return x


def brier(p, won):
    return float(np.mean((np.asarray(p) - np.asarray(won, float)) ** 2))


def acc(p, won):
    return float(np.mean((np.asarray(p) > 0.5) == np.asarray(won, bool)))


def ll(p, won):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(-np.where(won, np.log(pc), np.log(1 - pc))))


def cluster_boot(p, ref, won, ev, seed=VSEED, n=4000):
    won = np.asarray(won, bool)
    d = (np.asarray(p) - won) ** 2 - (np.asarray(ref) - won) ** 2
    uniq, inv = np.unique(np.asarray(ev), return_inverse=True)
    K = len(uniq)
    s = np.bincount(inv, weights=d, minlength=K)
    c = np.bincount(inv, minlength=K).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, K, size=(n, K))
    mb = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    return dict(d=float(d.mean()),
                ci=[float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))],
                p_better=float(np.mean(mb < 0)))


# ----------------------------------------------------------------- main --

def main():
    out = {}
    race = Race()

    # 0. DP sanity
    chk = []
    for p, T in ((0.5, 11), (0.55, 11), (0.60, 15), (0.45, 15)):
        chk.append(dict(p=p, T=T, dp=race_prob(p, T)))
    out["dp_check"] = chk
    out["dp_mc_0.55_11"] = mc_race(0.55, 11, 200000, seed=7)
    print("DP check:", [(c["p"], c["T"], round(c["dp"], 5)) for c in chk],
          "MC(0.55,11)=", round(out["dp_mc_0.55_11"], 4))

    games, pl, chem, traj = load_all("override")
    print(f"games with weather + ratings: {len(games)}  events={len(set(g['event'] for g in games))}")
    F = feats(games)
    T = np.array([g["T"] for g in games])
    won = np.array([g["won"] for g in games])
    ev = np.array([g["event"] for g in games])
    dates = np.array([g["date"] for g in games])
    out["n_all"] = len(games)
    out["n_events"] = int(len(set(ev)))
    out["n_outdoor"] = int(F["out"].sum())
    out["wx_src"] = {k: int(sum(1 for g in games if g["src"] == k))
                     for k in ("hour_actual", "hour_planned", "day")}
    out["mean_outdoor_wind"] = float(np.mean([g["wind"] for g in games
                                              if g["setting"] == "outdoor"]))

    # ---- ARM 1 -------------------------------------------------------
    eta_now = eta_of(games, pl, chem, None)
    fit_i = np.flatnonzero((dates >= RECENT) & (dates < SPLIT))
    hold_i = np.flatnonzero(dates >= SPLIT)
    print(f"ARM1 fit={len(fit_i)} hold={len(hold_i)} hold_events={len(set(ev[hold_i]))}")

    hset = [games[i] for i in hold_i]
    comp = {}
    for g in hset:
        comp[g["setting"]] = comp.get(g["setting"], 0) + 1
    ow = [g["wind"] for g in hset if g["setting"] == "outdoor"]
    out["arm1"] = dict(n_hold=len(hold_i), n_fit=len(fit_i),
                       hold_events=int(len(set(ev[hold_i]))),
                       hold_setting=comp,
                       hold_outdoor_wind=dict(
                           n=len(ow), median=float(np.median(ow)),
                           p95=float(np.percentile(ow, 95)), max=float(max(ow)),
                           n_ge_14=int(sum(w >= 14 for w in ow))))
    print("ARM1 holdout settings:", comp,
          "outdoor wind max=", round(max(ow), 1), "n>=14:", sum(w >= 14 for w in ow))

    def sl(idx):
        return {k: v[idx] for k, v in F.items()}

    Ff, Fh = sl(fit_i), sl(hold_i)
    ef, eh, Tf, Th, wf, wh = (eta_now[fit_i], eta_now[hold_i], T[fit_i],
                              T[hold_i], won[fit_i], won[hold_i])
    evh = ev[hold_i]

    p_base = pred("base", None, eh, Fh, Th, race)
    x_s = newton_fit("scale", ef, Ff, Tf, wf, race, [1.0])
    p_scale = pred("scale", x_s, eh, Fh, Th, race)
    arm1_models = dict(
        base=dict(n=len(hold_i), acc=acc(p_base, wh), brier=brier(p_base, wh),
                  ll=ll(p_base, wh)),
        scale=dict(params=[float(x_s[0])], acc=acc(p_scale, wh),
                   brier=brier(p_scale, wh), ll=ll(p_scale, wh),
                   vs_base=cluster_boot(p_scale, p_base, wh, evh)))
    print(f"  base  brier={arm1_models['base']['brier']:.5f} "
          f"acc={arm1_models['base']['acc']:.4f} ll={arm1_models['base']['ll']:.5f}")
    print(f"  scale s={x_s[0]:.4f} brier={arm1_models['scale']['brier']:.5f} "
          f"vs_base d={arm1_models['scale']['vs_base']['d']:+.6f} "
          f"P={arm1_models['scale']['vs_base']['p_better']:.3f}")
    for nm in ("a", "c", "d", "e", "f"):
        x = newton_fit(nm, ef, Ff, Tf, wf, race, [1.0, 0.0])
        p = pred(nm, x, eh, Fh, Th, race)
        b = cluster_boot(p, p_scale, wh, evh)
        arm1_models[nm] = dict(params=[float(v) for v in x], acc=acc(p, wh),
                               brier=brier(p, wh), ll=ll(p, wh), vs_scale=b)
        print(f"  {nm}: par=({x[0]:.4f},{x[1]:+.4f}) brier={brier(p, wh):.5f} "
              f"dB={b['d']:+.6f} CI[{b['ci'][0]:+.6f},{b['ci'][1]:+.6f}] "
              f"P(better)={b['p_better']:.3f}")
    out["arm1"]["models"] = arm1_models

    # ---- ARM 2: event cross-fit, several fold seeds / K ---------------
    eta_traj = eta_of(games, pl, chem, traj)
    uniq = np.array(sorted(set(ev)))
    arm2 = {}
    for (K, seed) in ((10, VSEED), (10, 42), (5, VSEED)):
        rng = np.random.default_rng(seed)
        fold_of = {e: int(i) for i, e in zip(rng.permutation(len(uniq)) % K, uniq)}
        folds = np.array([fold_of[e] for e in ev])
        preds = {}
        for nm, x0 in (("base", None), ("scale", [1.0]), ("a", [1.0, 0.0]),
                       ("d", [1.0, 0.0]), ("e", [1.0, 0.0]), ("f", [1.0, 0.0])):
            p = np.empty(len(games))
            for k in range(K):
                te = folds == k
                tr = ~te
                if x0 is None:
                    p[te] = pred("base", None, eta_traj[te], sl(te), T[te], race)
                    continue
                x = newton_fit(nm, eta_traj[tr], sl(tr), T[tr], won[tr], race, x0)
                p[te] = pred(nm, x, eta_traj[te], sl(te), T[te], race)
            preds[nm] = p
        row = dict(K=K, seed=seed,
                   base=dict(brier=brier(preds["base"], won), acc=acc(preds["base"], won),
                             ll=ll(preds["base"], won)),
                   scale=dict(brier=brier(preds["scale"], won),
                              acc=acc(preds["scale"], won), ll=ll(preds["scale"], won),
                              vs_base=cluster_boot(preds["scale"], preds["base"], won, ev)))
        for nm in ("a", "d", "e", "f"):
            row[nm] = dict(brier=brier(preds[nm], won), acc=acc(preds[nm], won),
                           vs_scale=cluster_boot(preds[nm], preds["scale"], won, ev))
        arm2[f"K{K}_s{seed}"] = row
        print(f"ARM2 K={K} seed={seed}: base={row['base']['brier']:.5f} "
              f"acc={row['base']['acc']:.4f} scale={row['scale']['brier']:.5f}")
        for nm in ("a", "d", "e", "f"):
            v = row[nm]["vs_scale"]
            print(f"   {nm}: brier={row[nm]['brier']:.5f} acc={row[nm]['acc']:.4f} "
                  f"dB={v['d']:+.7f} CI[{v['ci'][0]:+.7f},{v['ci'][1]:+.7f}] "
                  f"P={v['p_better']:.3f}")
    out["arm2_crossfit"] = arm2

    json.dump(out, open(Path(__file__).parent / "v_c2_verify.json", "w"), indent=1)
    print("wrote v_c2_verify.json")


if __name__ == "__main__":
    main()
