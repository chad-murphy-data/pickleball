"""C3(B) — wind skill as a VARIANCE COMPONENT, plus a pooled style x wind test.

    python model/weather_review/c3b_wind_skill.py [--sims 12]

model/wind_skill.py tested the "F1 rain driver" hypothesis by split-half
correlation of 552 per-player OLS wind slopes (r = +0.06 against a
permutation band).  Correlating noisy slopes is a weak instrument and it
cannot produce an upper bound.  This script estimates the dispersion of
per-player wind slopes DIRECTLY as a random-effects variance component:

    share_g - 1/2 = a + b*skill_g + c*w_g + d*skill_g*w_g
                    + sum_{i in T1}(u0_i + u1_i*wc_g)
                    - sum_{j in T2}(u0_j + u1_j*wc_g) + eps_g

    u0_i ~ N(0, tau0^2)      player's own over/under-performance vs v2
    u1_i ~ N(0, tau1^2)      PLAYER WIND SLOPE  <- the object of interest
    eps_g ~ N(0, phi_bin(g) * v_g)

wc = (match-hour wind - mean)/10 mph, so u1 is in units of point share
per 10 mph.  v_g is the EXACT race-to-T point-share variance implied by
the v2 per-point probability, so blowout/close-game heteroskedasticity is
not mistaken for player structure, and phi is free per wind bin so that
wind-driven noise inflation is not mistaken for wind SKILL: tau1 is then
identified only by the fact that the same player recurs across games with
different wind.

Estimation: ML by Woodbury (the random-effect design is sparse; every
likelihood evaluation is one Cholesky of a 2p x 2p matrix).  Interval:
profile likelihood, checked against a parametric bootstrap that also
reports the minimum detectable tau1.

Part 2 (style x wind): a hypothesis-driven axis has far more power than
552 individual slopes.  Player style indices are built from the committed
rally aggregates (serve-rally win rate, return-rally win rate, their
difference = serve-leaning vs return-leaning, and match pace k) and the
team style GAP is interacted with wind.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # the box is shared;
os.environ.setdefault("OMP_NUM_THREADS", "1")        # BLAS threads thrash

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b6_lib import ShareMoments  # noqa: E402
from c3_lib import cluster_se, load_frame, read_csv  # noqa: E402
from windlib import nelder_mead  # noqa: E402

DATA = HERE.parent.parent / "data"
WIND_BINS = [0.0, 8.0, 14.0, 1e9]          # mph, for the free residual scale


# ------------------------------------------------------------ aggregation --

def aggregate_matches(rows):
    """Collapse games to MATCHES.

    Every regressor (skill, match-hour wind) is constant within a match and
    so is any player random effect, so the match total carries all the
    information about tau1 while the within-match game-to-game replication
    -- which a game-level fit would mistake for player structure -- is
    dropped.  y_m = team-1 share of all points in the match; v_m is the
    exact iid race variance of that aggregate.
    """
    sm = ShareMoments()
    by = defaultdict(list)
    for r in rows:
        by[r["match"]].append(r)
    out = []
    for mid, gs in by.items():
        n1 = sum(g["s1"] for g in gs)
        n2 = sum(g["s2"] for g in gs)
        tot = n1 + n2
        num = 0.0
        for g in gs:
            p = 1.0 / (1.0 + math.exp(-g["eta"]))
            sd = sm.moments(p, g["T"])[1]
            num += (g["s1"] + g["s2"]) ** 2 * sd * sd
        r0 = gs[0]
        out.append(dict(match=mid, event=r0["event"], date=r0["date"],
                        tour=r0["tour"], us=r0["us"], eta=r0["eta"],
                        skill=r0["skill"], wind=r0["wind"], w=r0["w"],
                        actual=r0["actual"], games=len(gs),
                        y=n1 / tot - 0.5, v=num / tot ** 2))
    out.sort(key=lambda r: (r["date"], r["match"]))
    return out


# ------------------------------------------------------------ model core --

class MixedWind:
    """ML for the two-component random-effects model described above."""

    def __init__(self, rows, min_games=20, wind_mode="global", verbose=True):
        rows = aggregate_matches(rows)
        self.rows = rows
        n = len(rows)
        counts = defaultdict(int)
        for r in rows:
            for u in r["us"]:
                counts[u] += 1
        self.players = sorted(u for u, c in counts.items() if c >= min_games)
        pid = {u: i for i, u in enumerate(self.players)}
        self.p = len(self.players)
        P2 = 2 * self.p
        self.P2 = P2

        wind = np.array([r["wind"] for r in rows])
        self.wbar = float(wind.mean())
        if wind_mode == "within_event":
            # slope covariate = deviation from the EVENT's own mean wind, so a
            # player's event-to-event FORM cannot masquerade as a wind slope
            ev = np.array([r["event"] for r in rows])
            wc = np.empty(n)
            for e in set(ev.tolist()):
                m = ev == e
                wc[m] = (wind[m] - wind[m].mean()) / 10.0
        else:
            wc = (wind - self.wbar) / 10.0
        self.wind_mode = wind_mode
        skill = np.array([r["skill"] for r in rows])
        w = wind / 10.0
        self.y = np.array([r["y"] for r in rows])
        self.X = np.column_stack([np.ones(n), skill, w, skill * w])
        self.wc = wc
        self.wind = wind

        self.v = np.array([r["v"] for r in rows])
        self.bin = np.digitize(wind, WIND_BINS[1:-1])
        self.nbin = len(WIND_BINS) - 1

        # sparse random-effect design: 8 slots per game (4 players x 2 REs)
        idx = np.zeros((n, 8), np.int64)
        val = np.zeros((n, 8))
        for g, r in enumerate(rows):
            for j, u in enumerate(r["us"]):
                s = 1.0 if j < 2 else -1.0
                i = pid.get(u)
                if i is None:
                    continue
                idx[g, j] = i
                val[g, j] = s
                idx[g, 4 + j] = self.p + i
                val[g, 4 + j] = s * wc[g]
        self.idx, self.val = idx, val

        # per-bin sufficient statistics with weight 1/v (phi factored out)
        self.M, self.Zy, self.ZX = [], [], []
        self.Syy, self.SXy, self.SXX, self.nb, self.logv = [], [], [], [], []
        for b in range(self.nbin):
            m = self.bin == b
            dinv = 1.0 / self.v[m]
            I, V = idx[m], val[m]
            Mb = np.zeros(P2 * P2)
            for a1 in range(8):
                for a2 in range(8):
                    flat = I[:, a1] * P2 + I[:, a2]
                    Mb += np.bincount(flat, weights=dinv * V[:, a1] * V[:, a2],
                                      minlength=P2 * P2)
            self.M.append(Mb.reshape(P2, P2))
            zy = np.zeros(P2)
            zx = np.zeros((P2, self.X.shape[1]))
            for a1 in range(8):
                zy += np.bincount(I[:, a1], weights=dinv * V[:, a1] * self.y[m],
                                  minlength=P2)
                for k in range(self.X.shape[1]):
                    zx[:, k] += np.bincount(I[:, a1],
                                            weights=dinv * V[:, a1] * self.X[m, k],
                                            minlength=P2)
            self.Zy.append(zy)
            self.ZX.append(zx)
            self.Syy.append(float((self.y[m] ** 2 * dinv).sum()))
            self.SXy.append(self.X[m].T @ (self.y[m] * dinv))
            self.SXX.append(self.X[m].T @ (self.X[m] * dinv[:, None]))
            self.nb.append(int(m.sum()))
            self.logv.append(float(np.log(self.v[m]).sum()))
        if verbose:
            print(f"  mixed model: n={n} MATCHES, p={self.p} players with RE "
                  f"(>= {min_games} matches), wind_mode={wind_mode}, "
                  f"bins n={self.nb}")

    # ---- likelihood -----------------------------------------------------
    def neg2ll(self, tau0, tau1, phi, y=None):
        P2, p = self.P2, self.p
        A = np.zeros((P2, P2))
        zy = np.zeros(P2)
        zx = np.zeros((P2, self.X.shape[1]))
        Syy = 0.0
        SXy = np.zeros(self.X.shape[1])
        SXX = np.zeros((self.X.shape[1],) * 2)
        logdetD = 0.0
        yv = self.y if y is None else y
        for b in range(self.nbin):
            f = 1.0 / phi[b]
            A += f * self.M[b]
            if y is None:
                zy += f * self.Zy[b]
                Syy += f * self.Syy[b]
                SXy += f * self.SXy[b]
            zx += f * self.ZX[b]
            SXX += f * self.SXX[b]
            logdetD += self.nb[b] * math.log(phi[b]) + self.logv[b]
        if y is not None:      # recompute the y-dependent parts (simulation)
            for b in range(self.nbin):
                m = self.bin == b
                dinv = 1.0 / (self.v[m] * phi[b])
                I, V = self.idx[m], self.val[m]
                for a1 in range(8):
                    zy += np.bincount(I[:, a1], weights=dinv * V[:, a1] * yv[m],
                                      minlength=self.P2)
                Syy += float((yv[m] ** 2 * dinv).sum())
                SXy += self.X[m].T @ (yv[m] * dinv)
        t = np.concatenate([np.full(p, tau0 ** 2), np.full(p, tau1 ** 2)])
        t = np.maximum(t, 1e-14)
        C = A + np.diag(1.0 / t)
        L = np.linalg.cholesky(C)
        logdetC = 2.0 * np.log(np.diag(L)).sum()
        logdetT = float(np.log(t).sum())

        sol = np.linalg.solve(C, np.column_stack([zy, zx]))
        Ciy, CiX = sol[:, 0], sol[:, 1:]
        XVX = SXX - zx.T @ CiX
        XVy = SXy - zx.T @ Ciy
        yVy = Syy - zy @ Ciy
        beta = np.linalg.solve(XVX, XVy)
        quad = yVy - 2 * beta @ XVy + beta @ XVX @ beta
        logdetV = logdetD + logdetT + logdetC
        return logdetV + quad, beta

    def fit(self, tau1_fixed=None, start=None, y=None, fix_phi=None):
        """ML over (tau0, tau1, phi_b); tau1 can be pinned for profiling."""
        pinned = tau1_fixed is not None
        phi_fixed = None if fix_phi is None else np.asarray(fix_phi, float)

        def unpack(z):
            tau0 = math.exp(z[0])
            if pinned:
                phi = phi_fixed if phi_fixed is not None else np.exp(z[1:])
                return tau0, tau1_fixed, phi
            phi = phi_fixed if phi_fixed is not None else np.exp(z[2:])
            return tau0, math.exp(z[1]), phi

        def obj(z):
            try:
                val, _ = self.neg2ll(*unpack(z), y=y)
            except np.linalg.LinAlgError:
                return 1e12
            return val if np.isfinite(val) else 1e12

        nphi = 0 if phi_fixed is not None else self.nbin
        want = (1 if pinned else 2) + nphi
        z0 = np.array([math.log(0.02), math.log(0.01)] + [0.0] * self.nbin)
        if start is not None:
            z0 = np.asarray(start, float).copy()
        head = z0[:1] if pinned else np.array([z0[0], z0[1] if len(z0) > 1
                                               else math.log(0.01)])
        tail = z0[-self.nbin:] if len(z0) >= self.nbin else np.zeros(self.nbin)
        z0 = head if nphi == 0 else np.concatenate([head, tail])
        assert len(z0) == want
        z = nelder_mead(obj, z0, step=0.35, xtol=1e-6, ftol=1e-8, maxiter=4000)
        tau0, tau1, phi = unpack(z)
        val, beta = self.neg2ll(tau0, tau1, phi, y=y)
        return dict(tau0=tau0, tau1=tau1, phi=phi.tolist(), neg2ll=val,
                    beta=beta.tolist(), z=z.tolist(), pinned=pinned)

    def simulate(self, tau0, tau1, phi, beta, rng):
        n = len(self.y)
        u0 = rng.normal(0, tau0, self.p)
        u1 = rng.normal(0, tau1, self.p)
        u = np.concatenate([u0, u1])
        eff = np.zeros(n)
        for a1 in range(8):
            eff += self.val[:, a1] * u[self.idx[:, a1]]
        sd = np.sqrt(self.v * np.array(phi)[self.bin])
        return self.X @ np.asarray(beta) + eff + rng.normal(0, sd)


# ------------------------------------------------------------ style axis --

def player_styles(rows):
    """Crude banger-vs-grinder proxies from the committed rally aggregates."""
    sv, svn, rt, rtn = defaultdict(float), defaultdict(float), \
        defaultdict(float), defaultdict(float)
    for r in read_csv(DATA / "player_serve_rallies.csv"):
        if r["discipline"] != "doubles":
            continue
        u = r["player_uuid"]
        sv[u] += float(r["serve_wins"])
        svn[u] += float(r["serve_rallies"])
        rt[u] += float(r["return_wins"])
        rtn[u] += float(r["return_rallies"])
    kmatch = {r["match_id"]: float(r["k_match"])
              for r in read_csv(DATA / "match_rally_summary.csv")
              if r["discipline"] == "doubles" and r["k_match"]}
    pace, pacen = defaultdict(float), defaultdict(int)
    for r in rows:
        k = kmatch.get(r["match"])
        if k is None:
            continue
        for u in r["us"]:
            pace[u] += k
            pacen[u] += 1
    out = {}
    for u in set(svn) | set(pacen):
        d = {}
        if svn[u] >= 200 and rtn[u] >= 200:
            d["serve"] = sv[u] / svn[u]
            d["ret"] = rt[u] / rtn[u]
            d["serve_minus_ret"] = d["serve"] - d["ret"]
        if pacen[u] >= 10:
            d["pace"] = pace[u] / pacen[u]
        if d:
            out[u] = d
    return out


def style_reliability(rows, key):
    """Split-half (odd vs even year) reliability of the serve/return index."""
    agg = defaultdict(lambda: defaultdict(float))
    for r in read_csv(DATA / "player_serve_rallies.csv"):
        if r["discipline"] != "doubles":
            continue
        h = int(r["year"]) % 2
        a = agg[r["player_uuid"]]
        for c in ("serve_wins", "serve_rallies", "return_wins", "return_rallies"):
            a[f"{c}{h}"] += float(r[c])
    xs, ys = [], []
    for u, a in agg.items():
        if min(a["serve_rallies0"], a["serve_rallies1"],
               a["return_rallies0"], a["return_rallies1"]) < 150:
            continue
        def val(h):
            s = a[f"serve_wins{h}"] / a[f"serve_rallies{h}"]
            r_ = a[f"return_wins{h}"] / a[f"return_rallies{h}"]
            return s - r_ if key == "serve_minus_ret" else s
        xs.append(val(0))
        ys.append(val(1))
    if len(xs) < 20:
        return None, len(xs)
    return float(np.corrcoef(xs, ys)[0, 1]), len(xs)


def style_test(rows, styles, key, boot=1000, seed=7, player_fe=False):
    """y = a + b*skill + c*w + d*skill*w + e*dstyle + f*dstyle*w."""
    import c3a_fixed_effects as fe_mod
    use = []
    vals = []
    for r in rows:
        s = [styles.get(u, {}).get(key) for u in r["us"]]
        if any(v is None for v in s):
            continue
        use.append(r)
        vals.append(0.5 * (s[0] + s[1]) - 0.5 * (s[2] + s[3]))
    if len(use) < 500:
        return None
    ds = np.array(vals)
    sd = ds.std()
    ds = ds / sd
    D = fe_mod.pack(use)
    n = len(use)
    cols = [D["skill"], D["w"], D["skill"] * D["w"], ds, ds * D["w"]]
    M = np.column_stack([D["y"]] + cols)
    if player_fe:
        femat = fe_mod.make_fe("L3", D, np.zeros(n, np.int64))
        from c3_lib import absorb
        M, _ = absorb(femat, M)
    X = np.column_stack([np.ones(n)] + [M[:, i] for i in range(1, M.shape[1])])
    beta = np.linalg.solve(X.T @ X, X.T @ M[:, 0])
    cov, G = cluster_se(X, M[:, 0], beta, D["ev"])
    se = float(np.sqrt(cov[5, 5]))
    # event cluster bootstrap
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, e in enumerate(D["ev"]):
        by[int(e)].append(i)
    keys = list(by)
    idxs = {k: np.array(v) for k, v in by.items()}
    draws = []
    for _ in range(boot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idxs[keys[i]] for i in pick])
        Xb, yb = X[sel], M[sel, 0]
        try:
            draws.append(np.linalg.solve(Xb.T @ Xb, Xb.T @ yb)[5])
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(key=key, n=n, events=G, style_sd=float(sd),
                f=float(beta[5]), lo=float(lo), hi=float(hi), cr_se=se,
                e=float(beta[4]), d=float(beta[3]), player_fe=player_fe)


# ------------------------------------------------------------------ main --

def profile_ci(prof, base_n2ll, crit=3.84):
    """Profile-likelihood interval from a (tau1, -2ll) grid (None = open)."""
    xs = [t for t, _ in prof]
    ds = [v - base_n2ll for _, v in prof]
    kmin = min(range(len(ds)), key=lambda k: ds[k])

    def cross(k0, k1):
        a, b = ds[k0], ds[k1]
        if (a - crit) * (b - crit) > 0 or a == b:
            return None
        return xs[k0] + (crit - a) / (b - a) * (xs[k1] - xs[k0])

    lo = hi = None
    for k in range(kmin, 0, -1):
        c = cross(k, k - 1)
        if c is not None:
            lo = c
            break
    for k in range(kmin, len(xs) - 1):
        c = cross(k, k + 1)
        if c is not None:
            hi = c
            break
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=6)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-games", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    rows = load_frame()
    pools = {"outdoor": [r for r in rows if r["setting_c"] == "outdoor"],
             "indoor": [r for r in rows if r["setting_c"] == "indoor"]}
    out = {"pools": {}, "style": [], "reliability": {}}
    grid = [1e-6, 0.003, 0.006, 0.009, 0.012, 0.016, 0.020, 0.026, 0.034, 0.045]

    for name, rr in pools.items():
        for wm in ("global", "within_event"):
            tag = f"{name}/{wm}"
            print(f"\n== {tag}: {len(rr)} games")
            mm = MixedWind(rr, min_games=a.min_games, wind_mode=wm)
            base = mm.fit()
            print(f"  ML tau0={base['tau0']:.4f} tau1={base['tau1']:.4f} "
                  f"phi={[round(x,3) for x in base['phi']]} "
                  f"-2ll={base['neg2ll']:.2f} ({time.time()-t0:.0f}s)")
            prof = []
            for t1 in grid:
                f = mm.fit(tau1_fixed=t1, start=np.array(base["z"]))
                prof.append((t1, f["neg2ll"]))
                print(f"    tau1={t1:.4f}  d(-2ll)={f['neg2ll']-base['neg2ll']:+.3f}")
            lo, hi = profile_ci(prof, base["neg2ll"])
            lr0 = prof[0][1] - base["neg2ll"]
            print(f"  profile 95% CI on tau1: [{lo}, {hi}];  LR vs 0 = {lr0:.2f}")
            rec = dict(base=base, profile=prof, n_matches=len(mm.y), p=mm.p,
                       ci_lo=lo, ci_hi=hi, lr_vs_zero=lr0,
                       wind_sd=float(np.std(mm.wind)),
                       slope_sd=float(np.std(mm.wc * 10)))
            if name == "outdoor" and a.sims:
                rng = np.random.default_rng(99)
                sims = {}
                for kind, t1 in (("null", 1e-6), ("alt", 0.02)):
                    vals = []
                    for k in range(a.sims):
                        ysim = mm.simulate(base["tau0"], t1, base["phi"],
                                           base["beta"], rng)
                        f = mm.fit(start=np.array(base["z"]), y=ysim,
                                   fix_phi=base["phi"])
                        f0 = mm.fit(tau1_fixed=1e-6, start=np.array(base["z"]),
                                    y=ysim, fix_phi=base["phi"])
                        vals.append((f["tau1"], f0["neg2ll"] - f["neg2ll"]))
                        print(f"    sim {kind} {k}: tau1_hat={f['tau1']:.4f} "
                              f"LR={vals[-1][1]:.2f}")
                    sims[kind] = vals
                rec["sims"] = sims
            out["pools"][tag] = rec

    styles = player_styles(rows)
    print(f"\nstyle indices for {len(styles)} players")
    for key in ("serve_minus_ret", "serve", "pace"):
        r, nrel = style_reliability(rows, key)
        out["reliability"][key] = dict(r=r, n=nrel)
        print(f"  reliability({key}) = {r} on n={nrel}")
        for name, rr in pools.items():
            for pfe in (False, True):
                res = style_test(rr, styles, key, boot=a.boot, player_fe=pfe)
                if res:
                    res["pool"] = name
                    out["style"].append(res)
                    print(f"  {key:16s} {name:8s} playerFE={pfe} "
                          f"f={res['f']:+.4f} [{res['lo']:+.4f},{res['hi']:+.4f}] "
                          f"CRse={res['cr_se']:.4f} n={res['n']}")
    (HERE / "c3b_wind_skill.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", HERE / "c3b_wind_skill.json", f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
