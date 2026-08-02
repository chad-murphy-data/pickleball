"""Sweep entry states and player filters — with a search-corrected null.

The 9-9 test came back null. Natural follow-ups: enter earlier (7-7, 8-8),
enter on "anyone reaches 9" rather than a tie, and restrict to players whose
clutch is actually distinguishable from zero. Any of those could plausibly
be where a real effect lives.

But this is now a SEARCH, and the best cell of a grid always looks better
than it is. So the sweep is paired with a shuffled-clutch null that runs the
IDENTICAL grid: permute the lift values across players and re-sweep, many
times, recording the best cell each time. If the real grid's best cell sits
inside the null's distribution of best cells, we found nothing — no matter
how good that one cell's own CI looks.

Every prediction remains 2-fold cross-fitted BY MATCH, so no model ever sees
the match it predicts.

Run: python model/clutch_endgame_sweep.py       # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "model"))

from sitelib import race                                      # noqa: E402
from sitelib.winprob import (A1, A2, B1, B2, eta_anchor,      # noqa: E402
                             serve_probs, _table)
import clutch_srm as cs                                        # noqa: E402
import clutch_regimes as cr                                    # noqa: E402
from clutch_at_99_regimes import game_target                   # noqa: E402

K_LEAGUE = 0.443
N_SHUFFLE = 40

ENTRIES = [
    ("7-7  tie",        lambda a, b: a == 7 and b == 7),
    ("8-8  tie",        lambda a, b: a == 8 and b == 8),
    ("9-9  tie",        lambda a, b: a == 9 and b == 9),
    ("10-10 tie",       lambda a, b: a == 10 and b == 10),
    ("either >= 7",     lambda a, b: max(a, b) >= 7),
    ("either >= 9",     lambda a, b: max(a, b) >= 9),
    ("either >= 10",    lambda a, b: max(a, b) >= 10),
    ("both >= 7",       lambda a, b: min(a, b) >= 7),
    ("both >= 9",       lambda a, b: min(a, b) >= 9),
    ("within 1, >= 8",  lambda a, b: abs(a - b) <= 1 and max(a, b) >= 8),
]
FILTERS = [("all players", 0.0), ("all four |z|>=1", 1.0),
           ("all four |z|>=2", 2.0)]

_TCACHE = {}


def dp_at(kA, kB, a, b, side, n):
    key = (round(kA, 5), round(kB, 5))
    V = _TCACHE.get(key)
    if V is None:
        V = _table(round(kA, 6), round(kB, 6), 11, 51)
        _TCACHE[key] = V
    st = ((A1 if n == 1 else A2) if side == 0 else (B1 if n == 1 else B2))
    if a >= 11 and a - b >= 2:
        return 1.0
    if b >= 11 and b - a >= 2:
        return 0.0
    return V.get((a, b, st), 0.5)


def collect(blob, cur, traj):
    """One record per game, carrying the FULL rally sequence so any entry
    condition can be applied without re-walking the archive."""
    by_game = defaultdict(list)
    for r in blob["rallies"]:
        by_game[(r["match_id"], r["game_number"])].append(r)
    roster = defaultdict(dict)
    for r in blob["roster"]:
        if r["p1"] and r["p2"]:
            roster[r["match_id"]][r["side"]] = [r["p1"].lower(), r["p2"].lower()]
    out = []
    for (mid, gn), rs in by_game.items():
        s0, s1 = roster[mid].get(0, []), roster[mid].get(1, [])
        if len(s0) != 2 or len(s1) != 2:
            continue
        us = s0 + s1
        if not all(u in cur for u in us):
            continue
        rs.sort(key=lambda r: r["rally_number"])
        if any(r["server_score"] is None or r["receiver_score"] is None
               or r["server_side"] is None for r in rs):
            continue
        last = rs[-1]
        a = last["server_score"] + (1 if last["won"] else 0)
        b = last["receiver_score"] + (0 if last["won"] else 1)
        if game_target(max(a, b), min(a, b)) != 11:
            continue
        seq = []
        for r in rs:
            sa = r["server_score"] if r["server_side"] == 0 else r["receiver_score"]
            sb = r["receiver_score"] if r["server_side"] == 0 else r["server_score"]
            seq.append((sa, sb, r["server_side"], r["server_number"] or 2))
        month = rs[0]["match_date"][:7]
        out.append({
            "mid": mid, "us": us,
            "vals": [traj[u].get(month, cur[u]["v"]) for u in us],
            "seq": seq,
            "won1": 1 if (last["server_side"] if last["won"]
                          else 1 - last["server_side"]) == 0 else 0})
    return out


def main():
    print("Loading ...")
    blob = cs.fetch()
    cur, traj = cs.load_values()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])
    games = collect(blob, cur, traj)
    print(f"  {len(games)} to-11 doubles games\n")

    rows, _ = cs.build(blob, cur, traj)
    lz = np.array([r[4] for r in rows])
    hi_cut, lo_cut = np.quantile(lz, cr.HI_Q), np.quantile(lz, cr.LO_Q)
    keep = ([r + (True,) for r in rows if r[4] > hi_cut]
            + [r + (False,) for r in rows if r[4] < lo_cut])
    nhi, nlo = defaultdict(int), defaultdict(int)
    for r in keep:
        for u in r[:4]:
            (nhi if r[9] else nlo)[u] += 1
    ok = {u for u in nhi if nhi[u] >= cr.MIN_EACH and nlo[u] >= cr.MIN_EACH}
    keep = [r for r in keep if all(u in ok for u in r[:4])]
    index = {u: i for i, u in enumerate(sorted(ok))}
    P = len(index)
    mids = sorted({r[7] for r in keep})
    fold = {m: i % 2 for i, m in enumerate(mids)}

    def fit_fold(f):
        sub = [r for r in keep if fold[r[7]] == f]
        hi = np.array([r[9] for r in sub])
        y = np.array([r[6] for r in sub], dtype=float)
        rH, rL = y[hi].mean(), y[~hi].mean()
        off = np.where(hi, math.log(rH / (1 - rH)), math.log(rL / (1 - rL)))
        sub2 = [r[:5] + (float(off[i]),) + r[6:] for i, r in enumerate(sub)]
        b, se = cr.fit_regimes(sub2, index, anchor=True)
        lift = (b[2 * P:3 * P] + b[3 * P:]) - (b[:P] + b[P:2 * P])
        se_l = np.sqrt(se[:P] ** 2 + se[P:2 * P] ** 2
                       + se[2 * P:3 * P] ** 2 + se[3 * P:] ** 2)
        return lift, np.abs(lift) / se_l

    print("  fitting fold 0 ..."); L0, Z0 = fit_fold(0)
    print("  fitting fold 1 ..."); L1, Z1 = fit_fold(1)
    print(f"  {P} players; |z|>=1: {int((Z0 >= 1).sum())}, "
          f"|z|>=2: {int((Z0 >= 2).sum())}\n")

    # pre-compute per game: skill entry state + prob, and the player indices
    prepared = []
    for g in games:
        if g["mid"] not in fold or not all(u in index for u in g["us"]):
            continue
        v = g["vals"]
        eta = race.team_eta(v[0], v[1], v[2], v[3])
        p0 = race.calibrate(race.game_win_prob_uncertain(eta, race.SD_MATCH, 11))
        prepared.append({"mid": g["mid"], "seq": g["seq"], "won1": g["won1"],
                         "eta": eta_anchor(p0),
                         "i": [index[u] for u in g["us"]],
                         "f": fold[g["mid"]]})

    def run(lifts, zs, ent_fn, zmin, w):
        """(y, p_skill, p_lift, match) for every game meeting the condition."""
        out = []
        for g in prepared:
            L, Z = (lifts[1], zs[1]) if g["f"] == 0 else (lifts[0], zs[0])
            i = g["i"]
            if zmin > 0 and not all(Z[k] >= zmin for k in i):
                continue
            hit = next((s for s in g["seq"] if ent_fn(s[0], s[1])), None)
            if hit is None:
                continue
            a, b, side, n = hit
            kA, kB = serve_probs(g["eta"], K_LEAGUE)
            ps = dp_at(kA, kB, a, b, side, n)
            d = float(L[i[0]] + L[i[1]] - L[i[2]] - L[i[3]])
            kA2, kB2 = serve_probs(g["eta"] + w * d, K_LEAGUE)
            pl = dp_at(kA2, kB2, a, b, side, n)
            out.append((g["won1"], ps, pl, g["mid"]))
        return out

    WEIGHTS = (0.1, 0.2, 0.4)

    def best_delta(lifts, zs, verbose=False):
        best = 0.0
        table = []
        for lab, fn in ENTRIES:
            for flab, zmin in FILTERS:
                cells = []
                for w in WEIGHTS:
                    r = run(lifts, zs, fn, zmin, w)
                    if len(r) < 100:
                        continue
                    y = np.array([x[0] for x in r], dtype=float)
                    a = np.array([x[1] for x in r])
                    b = np.array([x[2] for x in r])
                    cells.append((np.mean((b - y) ** 2) - np.mean((a - y) ** 2),
                                  len(r), w, r))
                if not cells:
                    continue
                d, nn, w, r = min(cells, key=lambda c: c[0])
                best = min(best, d)
                table.append((lab, flab, nn, w, d, r))
        return best, table

    obs_best, table = best_delta((L0, L1), (Z0, Z1))
    print("=" * 84)
    print("SWEEP — Brier delta vs skill-only (negative = clutch helps), best "
          "weight per cell")
    print("=" * 84)
    print(f"  {'entry':<16}{'filter':<18}{'n':>6}{'w':>5}{'dBrier':>10}")
    for lab, flab, nn, w, d, _ in table:
        star = "  <<<" if d == obs_best else ""
        print(f"  {lab:<16}{flab:<18}{nn:>6}{w:>5.1f}{d:>+10.5f}{star}")

    # cluster CI for the single best cell (the number one would be tempted
    # to quote — shown so it can be compared against the null below)
    lab, flab, nn, w, d, r = min(table, key=lambda t: t[4])
    y = np.array([x[0] for x in r], dtype=float)
    a = np.array([x[1] for x in r])
    b = np.array([x[2] for x in r])
    bym = defaultdict(list)
    for k, x in enumerate(r):
        bym[x[3]].append(k)
    keys = list(bym)
    rng = np.random.default_rng(9)
    ds = []
    for _ in range(3000):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([bym[keys[k]] for k in pick])
        yy = y[idx]
        ds.append(np.mean((b[idx] - yy) ** 2) - np.mean((a[idx] - yy) ** 2))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"\n  BEST CELL: {lab} / {flab}  n={nn}  w={w}  "
          f"dBrier {d:+.5f}  CI[{lo:+.5f},{hi:+.5f}]")
    print(f"  Taken alone that CI would read as {'a real effect' if hi < 0 else 'null'}."
          f"  Now the search correction:")

    # ---- the search-corrected null
    print(f"\n{'-' * 84}\nSHUFFLED-CLUTCH NULL — same grid, lift values permuted "
          f"across players\n{'-' * 84}")
    nrng = np.random.default_rng(21)
    nulls = []
    for s in range(N_SHUFFLE):
        pa, pb = nrng.permutation(P), nrng.permutation(P)
        nb, _ = best_delta((L0[pa], L1[pb]), (Z0[pa], Z1[pb]))
        nulls.append(nb)
        if (s + 1) % 10 == 0:
            print(f"    {s + 1}/{N_SHUFFLE} shuffles ...")
    nulls = np.array(nulls)
    p = float(np.mean(nulls <= obs_best))
    print(f"\n  observed best cell   {obs_best:+.5f}")
    print(f"  null best cells      median {np.median(nulls):+.5f}  "
          f"5th pct {np.percentile(nulls, 5):+.5f}  min {nulls.min():+.5f}")
    print(f"  P(null best <= observed best) = {p:.3f}")
    print(f"\n  => {'SURVIVES the search correction' if p < 0.05 else 'DOES NOT survive — a grid this size produces a cell this good by chance'}")


if __name__ == "__main__":
    main()
