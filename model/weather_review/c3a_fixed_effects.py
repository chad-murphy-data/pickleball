"""C3(A) — the favourites x wind interaction under WITHIN-UNIT identification.

    python model/weather_review/c3a_fixed_effects.py [--boot 1000]

Published spec (model/favorites_wind.py, regression 1), game level:

    share - 1/2 = a + b*skill + c*w + d*(skill*w)

    skill = sigmoid(v2 eta) - 1/2   (team-1 orientation)
    w     = match-hour wind / 10 mph
    d < 0 outdoors  ==  wind compresses the favourite's edge.

Phase-2 test B2a showed d is not stable across EVENT SETS (verified-outdoor
events +0.114, unaudited events -0.111): the pooled estimator is comparing
different games, played by different people, in different places.  This
script re-identifies d from within-unit variation only, adding fixed
effects that are ANTISYMMETRIC where the outcome is (player and pair
dummies enter +1 for the two team-1 slots and -1 for the team-2 slots,
which is the correct paired-comparison encoding) and event-level slope
dummies where the confound is a per-event skill slope:

  L0  pooled (replicates the published spec)
  L1  event intercepts
  L2  event intercepts + event x skill      <- kills per-event calibration
  L3  player fixed effects
  L4  pair (dyad) fixed effects
  L5  pair x event fixed effects            <- same pair, same event,
                                               calm vs windy hours
  L6  player FE + event x skill
  L7  pair x event FE + event x skill       <- strongest

Fixed effects are absorbed by Jacobi-preconditioned CG (Frisch-Waugh:
y and the three regressors are residualised on the FE span, then a
4-column OLS recovers b, c, d).  Inference = cluster bootstrap over
EVENTS; fixed effects that nest inside an event are re-keyed per drawn
copy so a duplicated event does not share its FE across copies.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c3_lib import FEBlocks, absorb, cluster_se, load_frame  # noqa: E402

HERE = Path(__file__).resolve().parent
SPECS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
LABEL = {
    "L0": "pooled (published spec)",
    "L1": "event FE",
    "L2": "event FE + event x skill",
    "L3": "player FE",
    "L4": "pair FE",
    "L5": "pair x event FE",
    "L6": "player FE + event x skill",
    "L7": "pair x event FE + event x skill",
}


# --------------------------------------------------------------- design --

def pack(rows):
    """Row dicts -> integer-coded arrays."""
    def code(keys):
        u = {}
        out = np.empty(len(keys), np.int64)
        for i, k in enumerate(keys):
            out[i] = u.setdefault(k, len(u))
        return out, len(u)

    ev, nev = code([r["event"] for r in rows])
    players = sorted({u for r in rows for u in r["us"]})
    pidx = {u: i for i, u in enumerate(players)}
    P = np.array([[pidx[u] for u in r["us"]] for r in rows], np.int64)
    pairs = sorted({r["pair1"] for r in rows} | {r["pair2"] for r in rows})
    qidx = {u: i for i, u in enumerate(pairs)}
    Q = np.array([[qidx[r["pair1"]], qidx[r["pair2"]]] for r in rows], np.int64)
    return dict(
        y=np.array([r["y"] for r in rows]),
        skill=np.array([r["skill"] for r in rows]),
        w=np.array([r["w"] for r in rows]),
        wind=np.array([r["wind"] for r in rows]),
        ev=ev, nev=nev, P=P, nplayer=len(players), Q=Q, npair=len(pairs),
        actual=np.array([r["actual"] for r in rows]),
    )


def _recode(*cols):
    """Combine integer columns into a dense 0..K-1 code."""
    keys = cols[0].astype(np.int64)
    for c in cols[1:]:
        keys = keys * (int(c.max()) + 1) + c.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    return inv.astype(np.int64), len(uniq)


def make_fe(spec, D, copy):
    """Build the FE design for a (possibly bootstrapped) sample.

    `copy` labels which bootstrap copy of an event each row came from;
    event-nested effects are re-keyed by it.
    """
    n = len(D["y"])
    blocks, p = [], 0
    ev_c, nev_c = _recode(D["ev"], copy)
    one = np.ones(n)
    if spec in ("L1", "L2", "L6", "L7"):
        blocks.append((ev_c + p, one))
        p += nev_c
    if spec in ("L2", "L6", "L7"):
        blocks.append((ev_c + p, D["skill"]))
        p += nev_c
    if spec in ("L3", "L6"):
        for j in range(4):
            blocks.append((D["P"][:, j] + p, one * (1.0 if j < 2 else -1.0)))
        p += D["nplayer"]
    if spec == "L4":
        blocks.append((D["Q"][:, 0] + p, one))
        blocks.append((D["Q"][:, 1] + p, -one))
        p += D["npair"]
    if spec in ("L5", "L7"):
        # one shared parameter space for both sides: pair x event x copy
        keys = np.concatenate([D["Q"][:, 0], D["Q"][:, 1]])
        evs = np.concatenate([D["ev"], D["ev"]])
        cc = np.concatenate([copy, copy])
        pe, k = _recode(keys, evs, cc)
        blocks.append((pe[:n] + p, one))
        blocks.append((pe[n:] + p, -one))
        p += k
    if not blocks:
        return None
    return FEBlocks(n, blocks, p)


def fit(spec, D, copy=None, fast=False):
    """Returns (beta, surviving-sd fraction of skill*w, residuals, X)."""
    n = len(D["y"])
    if copy is None:
        copy = np.zeros(n, np.int64)
    M = np.column_stack([D["y"], D["skill"], D["w"], D["skill"] * D["w"]])
    fe = make_fe(spec, D, copy)
    if fe is None:
        R = M - M.mean(axis=0)
        frac = 1.0
    else:
        R, _ = absorb(fe, M, tol=(1e-7 if fast else 1e-9),
                      maxit=(250 if fast else 400))
        sw0 = M[:, 3] - M[:, 3].mean()
        frac = float(np.std(R[:, 3]) / max(np.std(sw0), 1e-12))
    X = np.column_stack([np.ones(n), R[:, 1], R[:, 2], R[:, 3]])
    beta = np.linalg.solve(X.T @ X, X.T @ R[:, 0])
    return beta, frac, R, X


def boot(spec, D, events, B, seed=1234):
    rng = np.random.default_rng(seed)
    idx_by_ev = defaultdict(list)
    for i, e in enumerate(D["ev"]):
        idx_by_ev[int(e)].append(i)
    idx_by_ev = {k: np.array(v) for k, v in idx_by_ev.items()}
    keys = list(idx_by_ev)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        parts, copies = [], []
        for c, i in enumerate(pick):
            rr = idx_by_ev[keys[i]]
            parts.append(rr)
            copies.append(np.full(len(rr), c, np.int64))
        sel = np.concatenate(parts)
        copy = np.concatenate(copies)
        Db = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[0] == len(D["y"])
                  else v) for k, v in D.items()}
        try:
            beta, _, _, _ = fit(spec, Db, copy, fast=True)
        except np.linalg.LinAlgError:
            continue
        out.append(beta)
    return np.array(out)


def variation_report(spec, D):
    """How much identifying variation survives the restriction."""
    n = len(D["y"])
    wind = D["wind"]
    if spec in ("L1", "L2"):
        unit, wind = D["ev"], D["wind"]
    elif spec in ("L3", "L6"):
        unit = np.concatenate([D["P"][:, j] for j in range(4)])
        wind = np.tile(D["wind"], 4)
    elif spec == "L4":
        unit = np.concatenate([D["Q"][:, 0], D["Q"][:, 1]])
        wind = np.tile(D["wind"], 2)
    elif spec in ("L5", "L7"):
        unit, _ = _recode(np.concatenate([D["Q"][:, 0], D["Q"][:, 1]]),
                          np.tile(D["ev"], 2))
        wind = np.tile(D["wind"], 2)
    else:
        return dict(units=0, units_var=0, games_in_var=n)
    by = defaultdict(list)
    for u, wv in zip(unit, wind):
        by[int(u)].append(wv)
    units = len(by)
    uvar = sum(1 for v in by.values() if len(v) > 1 and np.std(v) >= 2.0)
    games = sum(len(v) for v in by.values() if len(v) > 1 and np.std(v) >= 2.0)
    return dict(units=units, units_var=uvar, games_in_var=games)


# ------------------------------------------------------------------ main --

def run_pool(name, rows, specs, B, seed):
    D = pack(rows)
    events = sorted(set(D["ev"].tolist()))
    res = []
    for spec in specs:
        t0 = time.time()
        beta, frac, R, X = fit(spec, D)
        cov, G = cluster_se(X, R[:, 0], beta, D["ev"])
        se = float(np.sqrt(cov[3, 3]))
        draws = boot(spec, D, events, B, seed) if B else np.zeros((0, 4))
        if len(draws):
            lo, hi = np.percentile(draws[:, 3], [2.5, 97.5])
        else:
            lo, hi = beta[3] - 1.96 * se, beta[3] + 1.96 * se
        v = variation_report(spec, D)
        res.append(dict(pool=name, spec=spec, label=LABEL[spec],
                        n=len(rows), events=G, b=float(beta[1]),
                        c=float(beta[2]), d=float(beta[3]),
                        d_lo=float(lo), d_hi=float(hi), d_cr_se=se,
                        frac_sw=frac, secs=round(time.time() - t0, 1), **v))
        print(f"  {name:22s} {spec} d={beta[3]:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"CRse={se:.4f} surviving sd(sw)={frac:.3f} "
              f"({v['units']} units, {v['units_var']} w/ wind spread) "
              f"{res[-1]['secs']}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot-heavy", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rows = load_frame()
    print(f"frame: {len(rows)} games with a match-hour wind join "
          f"({100*np.mean([r['actual'] for r in rows]):.0f}% actual start times)")

    pools = {
        "outdoor (arm c)": [r for r in rows if r["setting_c"] == "outdoor"],
        "indoor (arm c)": [r for r in rows if r["setting_c"] == "indoor"],
        "outdoor (arm a=published)": [r for r in rows if r["setting_a"] == "outdoor"],
        "outdoor c: audited": [r for r in rows
                               if r["setting_c"] == "outdoor" and r["audited"]],
        "outdoor c: unaudited": [r for r in rows
                                 if r["setting_c"] == "outdoor" and not r["audited"]],
        "outdoor c: actual times": [r for r in rows if r["setting_c"] == "outdoor"
                                    and r["actual"]],
    }
    reps = {"L0": a.boot, "L1": a.boot, "L2": a.boot, "L4": a.boot_heavy,
            "L3": a.boot_heavy, "L5": a.boot_heavy,
            "L6": max(a.boot_heavy // 2, 100), "L7": max(a.boot_heavy // 2, 100)}
    out = []
    for name, rr in pools.items():
        if len(rr) < 500:
            continue
        specs = SPECS if name.startswith(("outdoor (arm c)", "indoor (arm c)")) \
            else ["L0", "L2", "L5"]
        print(f"\n== {name}: {len(rr)} games")
        for spec in specs:
            B = reps[spec] if name.startswith(("outdoor (arm c)", "indoor (arm c)")) \
                else min(reps[spec], a.boot_heavy)
            out += run_pool(name, rr, [spec], B, a.seed)
    (HERE / "c3a_fixed_effects.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", HERE / "c3a_fixed_effects.json")


if __name__ == "__main__":
    main()
