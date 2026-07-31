"""ADVERSARIAL VERIFICATION of B2a, part 3.

Is the verified/unaudited split of the corrected-outdoor pool SPECIAL, or is
d simply unstable along ANY event-level cut?  Plus: confound audit of the
split (tour, era, format, wind range), a stage-robust within-event estimator,
and an independent spot-check of H1.

    python model/weather_review/v_b2a_attack2.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_b2a_verify import build_games, labels, rd, fnum  # noqa: E402
from v_b2a_attack import Suff, boot_ci  # noqa: E402


def cell_within_d(rows, keyfn, min_games=40):
    """Absorb a cell-specific intercept AND skill slope, then estimate the
    common (c, d).  keyfn defines the cell (event, or event x date)."""
    by = defaultdict(list)
    for r in rows:
        by[keyfn(r)].append(r)
    Y, W, SW, CL = [], [], [], []
    for k, rs in by.items():
        if len(rs) < min_games or np.std([r["wind"] for r in rs]) < 1e-9:
            continue
        A = np.array([[1.0, r["skill"]] for r in rs])
        M = np.eye(len(rs)) - A @ np.linalg.pinv(A)
        Y.append(M @ np.array([r["y"] for r in rs]))
        W.append(M @ np.array([r["w"] for r in rs]))
        SW.append(M @ np.array([r["sw"] for r in rs]))
        CL += [r["ev"] for r in rs]
    if not Y:
        return None
    y = np.concatenate(Y); X = np.column_stack([np.concatenate(W), np.concatenate(SW)])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    res = y - X @ b
    cl = np.array(CL)
    meat = np.zeros((2, 2))
    for e in np.unique(cl):
        m = cl == e
        s = X[m].T @ res[m]
        meat += np.outer(s, s)
    G = len(np.unique(cl))
    V = XtXi @ meat @ XtXi * (G / (G - 1.0))
    return float(b[1]), float(math.sqrt(V[1, 1])), G, len(y), len(by)


def main():
    games = build_games()
    geo, ov, arms = labels()
    P = print
    by_ev = defaultdict(list)
    for r in games:
        by_ev[r["ev"]].append(r)
    suff = Suff(games)
    ev_all = list(suff.S)
    c_out = [e for e in ev_all if arms["c"].get(e) == "outdoor"]

    def cls(e):
        o = ov.get(e)
        return f"{geo[e]}->{o['setting'] if o else '(unaudited)'}"

    # ---- how unstable is d along ARBITRARY event-level cuts? ---------------
    P("=" * 74)
    P("d ACROSS EVENT-LEVEL CUTS OF THE CORRECTED-OUTDOOR POOL (arm c)")
    P("=" * 74)
    P(f"  whole pool: d = {suff.d(c_out):+.4f}  ({len(c_out)} events, {suff.N(c_out)} games)")

    def show(label, groups):
        parts = []
        for g, evs in groups.items():
            if not evs:
                continue
            d = suff.d(evs)
            if d is None:
                continue
            parts.append(f"{g}: {d:+.4f} (ev={len(evs)},n={suff.N(evs)})")
        P(f"  {label:22s} " + " | ".join(parts))

    # tour of the event (majority)
    tour_of = {e: max({"MLP": sum(1 for r in by_ev[e] if r["tour"] == "MLP"),
                       "PPA": sum(1 for r in by_ev[e] if r["tour"] == "PPA")}.items(),
                      key=lambda kv: kv[1])[0] for e in c_out}
    show("by tour", {t: [e for e in c_out if tour_of[e] == t] for t in ("PPA", "MLP")})
    yr_of = {e: by_ev[e][0]["date"][:4] for e in c_out}
    show("by year (first date)", {y: [e for e in c_out if yr_of[e] == y]
                                  for y in ("2024", "2025", "2026")})
    med = np.median([suff.n[e] for e in c_out])
    show("by event size", {"small": [e for e in c_out if suff.n[e] <= med],
                           "large": [e for e in c_out if suff.n[e] > med]})
    mw = {e: np.mean([r["wind"] for r in by_ev[e]]) for e in c_out}
    mmw = np.median(list(mw.values()))
    show("by event mean wind", {"calmer": [e for e in c_out if mw[e] <= mmw],
                                "windier": [e for e in c_out if mw[e] > mmw]})
    lat = {r["event_id"]: fnum(r["lat"]) for r in rd("data/event_geo.csv")}
    mlat = np.median([lat[e] for e in c_out if lat.get(e) is not None])
    show("by latitude", {"south": [e for e in c_out if (lat.get(e) or 0) <= mlat],
                         "north": [e for e in c_out if (lat.get(e) or 0) > mlat]})
    show("by audit status", {"verified": [e for e in c_out if cls(e) == "outdoor->outdoor"],
                             "unaudited": [e for e in c_out if cls(e) == "outdoor->(unaudited)"],
                             "newly": [e for e in c_out if cls(e) == "indoor->outdoor"]})
    # arbitrary placebo: hash of the event uuid
    show("placebo (uuid hash)", {"h0": [e for e in c_out if int(e[:8], 16) % 2 == 0],
                                 "h1": [e for e in c_out if int(e[:8], 16) % 2 == 1]})

    # ---- does the audit gap survive tour / era controls? -------------------
    P("\n" + "=" * 74)
    P("CONFOUND AUDIT OF THE VERIFIED-vs-UNAUDITED GAP")
    P("=" * 74)
    ver = [e for e in c_out if cls(e) == "outdoor->outdoor"]
    una = [e for e in c_out if cls(e) == "outdoor->(unaudited)"]
    gap = suff.d(ver) - suff.d(una)
    P(f"  full gap: {gap:+.4f}")

    def subgap(label, filt):
        rows = [r for r in games if filt(r)]
        s = Suff(rows)
        v = [e for e in ver if e in s.S]
        u = [e for e in una if e in s.S]
        dv, du = s.d(v), s.d(u)
        if dv is None or du is None:
            P(f"  {label:26s} n/a")
            return
        P(f"  {label:26s} verified {dv:+.4f} (ev={len(v)},n={s.N(v)})  "
          f"unaudited {du:+.4f} (ev={len(u)},n={s.N(u)})  gap {dv-du:+.4f}")

    subgap("PPA games only", lambda r: r["tour"] == "PPA")
    for y in ("2024", "2025", "2026"):
        subgap(f"{y} games only", lambda r, y=y: r["date"][:4] == y)
    subgap("sideout_11 only", lambda r: r["fmt"] == "sideout_11")
    subgap("wind < 12 mph only", lambda r: r["wind"] < 12)
    subgap("actual (not planned) times", lambda r: not r["planned"])

    # ---- stage-robust within-event estimator -------------------------------
    P("\n" + "=" * 74)
    P("COMPOSITION-FREE ESTIMATORS (absorb cell intercept + cell skill slope)")
    P("=" * 74)
    for lab, arm in (("arm a", "a"), ("arm c", "c")):
        for s in ("outdoor", "indoor"):
            rows = [r for r in games if arms[arm].get(r["ev"]) == s]
            for cellname, keyfn, mg in (("event", lambda r: r["ev"], 40),
                                        ("event x day", lambda r: (r["ev"], r["date"]), 40)):
                out = cell_within_d(rows, keyfn, mg)
                if out is None:
                    continue
                d, se, G, n, ncell = out
                P(f"  {lab} {s:7s} within {cellname:11s}: d = {d:+.4f} +/- {se:.4f} "
                  f"[{d-1.96*se:+.4f},{d+1.96*se:+.4f}]  ({ncell} cells, {n} games)")

    # ---- independent H1 spot check -----------------------------------------
    P("\n" + "=" * 74)
    P("H1 SPOT CHECK (independent build): serve-point rate vs match-hour wind")
    P("=" * 74)
    hourly = {}
    for r in rd("data/event_weather_hourly.csv"):
        w = fnum(r["windspeed_10m"])
        if w is not None:
            hourly[(r["event_id"], r["local_time"][:13])] = w
    hour = {}
    for r in rd("data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            hour[r["match_id"]] = ts[:13]
    ev_of = {r["match_id"]: r["event_id"] for r in rd("data/match_times.csv")}
    rows = []
    for r in rd("data/match_rally_summary.csv"):
        if r["discipline"] != "doubles" or int(r["n_rallies"]) < 20:
            continue
        e = ev_of.get(r["match_id"])
        w = hourly.get((e, hour.get(r["match_id"], "")))
        if w is None:
            continue
        rows.append((e, w, int(r["n_points"]) / int(r["n_rallies"]), int(r["n_rallies"])))
    for arm in ("a", "c"):
        for s in ("outdoor", "indoor"):
            sub = [t for t in rows if arms[arm].get(t[0]) == s]
            w = np.array([t[1] for t in sub]); y = np.array([t[2] for t in sub])
            wt = np.array([t[3] for t in sub], float)
            X = np.column_stack([np.ones(len(w)), w])
            beta = np.linalg.solve(X.T @ (X * wt[:, None]), X.T @ (y * wt))
            # event-clustered SE
            res = y - X @ beta
            XtXi = np.linalg.inv(X.T @ (X * wt[:, None]))
            meat = np.zeros((2, 2))
            evs = np.array([t[0] for t in sub])
            for e in np.unique(evs):
                m = evs == e
                sc = (X[m] * wt[m, None]).T @ res[m]
                meat += np.outer(sc, sc)
            V = XtXi @ meat @ XtXi
            se = math.sqrt(V[1, 1]) * 10
            P(f"  arm {arm} {s:7s}: slope/+10mph = {beta[1]*10:+.4f} +/- {se:.4f} "
              f"[{beta[1]*10-1.96*se:+.4f},{beta[1]*10+1.96*se:+.4f}]  n={len(sub)}")


if __name__ == "__main__":
    main()
