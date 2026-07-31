"""C2 verification, part 2: the load-bearing BOUND and the value-of-knowing.

  (A) full-archive wind coefficients, cold-start optimiser, 1000-resample
      event-cluster bootstrap (the tester used 400 warm-started NM fits)
  (B) a confound check the tester did not run: free OUTDOOR scale alongside
      the wind slope, so b cannot absorb an indoor/outdoor calibration gap
  (C) subgroup scoring by wind bin
  (D) ORACLE value-of-wind: the maximum Brier a *true* effect of size d
      could ever buy on this archive (analytic, no simulation noise)
  (E) analytic MDE from the coefficient standard error
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_c2_verify import (Race, VSEED, RECENT, SPLIT, cluster_boot, eta_of,  # noqa
                         feats, load_all, newton_fit, pred, brier, acc, ll)

OUT = Path(__file__).resolve().parent


def pred2(name, x, eta, F, T, race):
    """Extra specs with a free outdoor scale."""
    o = F["out"]
    if name == "a2":     # s*(1 + c*outdoor + b*W) * eta
        return race.win(x[0] * eta * (1.0 + x[1] * o + x[2] * F["W"]), T)
    if name == "d2":     # s*(1 + c*outdoor + b*1[w>=14]) * eta
        return race.win(x[0] * eta * (1.0 + x[1] * o + x[2] * F["TH"]), T)
    raise KeyError(name)


def newton2(name, eta, F, T, won, race, x0):
    x = np.array(x0, float)
    n = len(x)

    def nll(v):
        p = np.clip(pred2(name, v, eta, F, T, race), 1e-9, 1 - 1e-9)
        return -float(np.mean(np.where(won, np.log(p), np.log(1 - p))))

    f0 = nll(x); h = 1e-4
    for _ in range(60):
        g = np.zeros(n); H = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n); e[i] = h
            fp, fm = nll(x + e), nll(x - e)
            g[i] = (fp - fm) / (2 * h); H[i, i] = (fp - 2 * f0 + fm) / h ** 2
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros(n); ei[i] = h
                ej = np.zeros(n); ej[j] = h
                H[i, j] = H[j, i] = (nll(x + ei + ej) - nll(x + ei - ej)
                                     - nll(x - ei + ej) + nll(x - ei - ej)) / (4 * h ** 2)
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
            xn = x + t * step; fn = nll(xn)
            if fn < f0 - 1e-14:
                break
            t *= 0.5
        else:
            break
        conv = abs(f0 - fn) < 1e-12
        x, f0 = xn, fn
        if conv:
            break
    return x


def main():
    race = Race()
    games, pl, chem, traj = load_all("override")
    F = feats(games)
    T = np.array([g["T"] for g in games])
    won = np.array([g["won"] for g in games])
    ev = np.array([g["event"] for g in games])
    eta = eta_of(games, pl, chem, traj)
    uniq = np.array(sorted(set(ev)))
    ev_idx = {e: np.flatnonzero(ev == e) for e in uniq}
    out = {}

    ow = np.array([g["wind"] for g in games if g["setting"] == "outdoor"])
    out["outdoor_wind"] = dict(n=len(ow), mean=float(ow.mean()),
                               median=float(np.median(ow)),
                               p90=float(np.percentile(ow, 90)),
                               max=float(ow.max()),
                               n_ge14=int((ow >= 14).sum()),
                               n_ge20=int((ow >= 20).sum()))
    print("outdoor wind:", out["outdoor_wind"])

    def sl(idx):
        return {k: v[idx] for k, v in F.items()}

    # ---- (A) coefficients + 1000-resample cold-start cluster bootstrap ----
    coefs = {}
    NB = 1000
    for nm, x0 in (("a", [1.0, 0.0]), ("d", [1.0, 0.0]),
                   ("c", [1.0, 0.0]), ("f", [1.0, 0.0])):
        xf = newton_fit(nm, eta, F, T, won, race, x0)
        rng = np.random.default_rng(VSEED + hash(nm) % 1000)
        bs = []
        for _ in range(NB):
            pick = uniq[rng.integers(0, len(uniq), len(uniq))]
            idx = np.concatenate([ev_idx[e] for e in pick])
            xb = newton_fit(nm, eta[idx], sl(idx), T[idx], won[idx], race, x0)
            bs.append(float(xb[1]))
        bs = np.array(bs)
        coefs[nm] = dict(b=float(xf[1]), s=float(xf[0]),
                         ci=[float(np.percentile(bs, 2.5)),
                             float(np.percentile(bs, 97.5))],
                         se=float(bs.std()), n_boot=NB,
                         p_neg=float((bs < 0).mean()))
        print(f"  coef {nm}: b={xf[1]:+.4f} s={xf[0]:.4f} "
              f"CI[{coefs[nm]['ci'][0]:+.4f},{coefs[nm]['ci'][1]:+.4f}] "
              f"se={coefs[nm]['se']:.4f} P(b<0)={coefs[nm]['p_neg']:.3f}")
    out["coefs"] = coefs

    # ---- (B) confound check: free outdoor scale alongside the wind slope --
    conf = {}
    for nm in ("a2", "d2"):
        xf = newton2(nm, eta, F, T, won, race, [1.0, 0.0, 0.0])
        rng = np.random.default_rng(VSEED + 7)
        bs = []
        for _ in range(400):
            pick = uniq[rng.integers(0, len(uniq), len(uniq))]
            idx = np.concatenate([ev_idx[e] for e in pick])
            xb = newton2(nm, eta[idx], sl(idx), T[idx], won[idx], race,
                         [1.0, 0.0, 0.0])
            bs.append(float(xb[2]))
        bs = np.array(bs)
        conf[nm] = dict(s=float(xf[0]), outdoor_scale=float(xf[1]),
                        b=float(xf[2]),
                        ci=[float(np.percentile(bs, 2.5)),
                            float(np.percentile(bs, 97.5))], n_boot=400)
        print(f"  {nm}: s={xf[0]:.4f} outdoor_extra={xf[1]:+.4f} "
              f"b={xf[2]:+.4f} CI[{conf[nm]['ci'][0]:+.4f},{conf[nm]['ci'][1]:+.4f}]")
    out["outdoor_scale_control"] = conf

    # ---- (C) subgroups on my own 10-fold cross-fit ------------------------
    K = 10
    rng = np.random.default_rng(VSEED)
    fold_of = {e: int(i) for i, e in zip(rng.permutation(len(uniq)) % K, uniq)}
    folds = np.array([fold_of[e] for e in ev])
    P = {}
    for nm, x0 in (("scale", [1.0]), ("a", [1.0, 0.0]), ("d", [1.0, 0.0])):
        p = np.empty(len(games))
        for k in range(K):
            te = folds == k; tr = ~te
            x = newton_fit(nm, eta[tr], sl(tr), T[tr], won[tr], race, x0)
            p[te] = pred(nm, x, eta[te], sl(te), T[te], race)
        P[nm] = p
    wind = F["wind"]; o = F["out"] == 1
    subs = {}
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 99)):
        m = o & (wind >= lo) & (wind < hi)
        if m.sum() < 30:
            continue
        row = dict(n=int(m.sum()), n_events=int(len(set(ev[m]))))
        for nm in ("a", "d"):
            row[nm] = cluster_boot(P[nm][m], P["scale"][m], won[m], ev[m])
        subs[f"{lo}-{hi}"] = row
        print(f"  wind {lo}-{hi}: n={row['n']} ev={row['n_events']} "
              f"a dB={row['a']['d']:+.6f} CI[{row['a']['ci'][0]:+.6f},"
              f"{row['a']['ci'][1]:+.6f}] P={row['a']['p_better']:.3f} | "
              f"d dB={row['d']['d']:+.6f} P={row['d']['p_better']:.3f}")
    out["subgroups"] = subs

    # ---- (D) ORACLE: max Brier a TRUE effect of size d could buy ----------
    s0 = float(newton_fit("scale", eta, F, T, won, race, [1.0])[0])
    p_ctrl = race.win(s0 * eta, T)
    oracle = []
    for d in (-0.05, -0.072, -0.117, -0.15, -0.21, -0.40):
        p_true = race.win(s0 * eta * (1.0 + d * F["W"]), T)
        # expected Brier under truth p_true
        def eb(q):
            return float(np.mean(p_true * (1 - q) ** 2 + (1 - p_true) * q ** 2))
        gain_all = eb(p_true) - eb(p_ctrl)
        mo = o
        def ebm(q, m):
            return float(np.mean(p_true[m] * (1 - q[m]) ** 2
                                 + (1 - p_true[m]) * q[m] ** 2))
        gain_out = ebm(p_true, mo) - ebm(p_ctrl, mo)
        mw = o & (wind >= 14)
        gain_windy = ebm(p_true, mw) - ebm(p_ctrl, mw)
        oracle.append(dict(d=d, gain_brier_all=gain_all,
                           gain_brier_outdoor=gain_out,
                           gain_brier_windy14=gain_windy))
        print(f"  oracle d={d:+.3f}: max Brier gain all={gain_all:+.7f} "
              f"outdoor={gain_out:+.7f} windy14+={gain_windy:+.7f}")
    out["oracle"] = oracle
    out["scale_fullsample"] = s0

    # ---- (E) pp translation of the threshold bound ------------------------
    mw = o & (wind >= 14) & (wind < 20)
    bt, lo_t, hi_t = coefs["d"]["b"], coefs["d"]["ci"][0], coefs["d"]["ci"][1]
    e_abs = np.abs(eta[mw]) * s0
    Tm = T[mw]
    base_fav = race.win(e_abs, Tm)
    def shift(b):
        return float(np.mean(race.win(e_abs * (1.0 + b), Tm) - base_fav) * 100)
    out["pp_threshold"] = dict(n=int(mw.sum()), mean_fav_pred=float(base_fav.mean()),
                               point=shift(bt), lo=shift(lo_t), hi=shift(hi_t))
    print(f"  pp shift 14-20mph favourite: {shift(bt):+.2f} "
          f"[{shift(lo_t):+.2f},{shift(hi_t):+.2f}] pp "
          f"(n={int(mw.sum())}, mean fav pred {base_fav.mean()*100:.1f}%)")

    # analytic MDE from the coefficient se (2.8*se at 80% power, one-sided .05)
    out["mde_analytic_coef"] = {k: -2.486 * v["se"] for k, v in coefs.items()}
    print("  analytic 80%-power MDE on b (coef test):",
          {k: round(v, 3) for k, v in out["mde_analytic_coef"].items()})

    json.dump(out, open(OUT / "v_c2_verify2.json", "w"), indent=1)
    print("wrote v_c2_verify2.json")


if __name__ == "__main__":
    main()
