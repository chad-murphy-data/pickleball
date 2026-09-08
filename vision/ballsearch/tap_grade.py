"""Tap-keyed grader — the bounce pipeline against the owner's bounce TAPS
(2026-09-08).

Until now every bounce number in this directory was keyed on the
fitter's own bounce calls on the owner's click path ("human bounces",
`bound_oracle.human_bounces`).  The bounce coder gave the first direct
key: `data/vision/bounce_labels_chicago0725.csv`, one call per flight
(bounce with a landing spot / volley / no bounce / unsure).  This module
grades any bounds variant x demotion x typing rule against those taps:

  matched   taps within 0.30 s of a called bounce (greedy nearest time)
  false     called bounces with no tap within 0.30 s on a flight the
            owner called volley / no-bounce (unsure and uncoded flights
            are not counted either way)
  land ft   landing error for matched taps that carry a spot (floor
            homography of the tap pixel vs the fitter's bounce_xy)
  contacts  contact taps matched within 0.25 s / junk bounds
  intact    tap-holding flights whose two contacts are both bound and
            nothing is bound in between (bound_oracle.intact_flights,
            keyed on taps)

Bounds variants: `shipped` (the production claim + shipped crossing
demotion, = bounce_autopsy.tracked), `claimer` (claimer_bounds_r{N}.json,
the 09-05 model with the window-end leak), `claimer_ne`
(claimer_bounds_ne_r{N}.json, `claimer.py --no-end-feats --key taps`).
Demotion: none / shipped.  Typing: off, or the AT-FEET rule
(`feet_type`): an ok ARC flight whose fitted arc reaches z <= Z_FEET ft
inside the last W_FEET s before the next contact is a bounce at the
receiver's feet — the fitter's blind spot the taps exposed (13% of all
bounces sit within 0.10 s of the next contact and were all typed arc).

Fits are cached per (rally, bounds, demotion); the typing rule is
applied on the cached segments so its grid costs nothing.

Seal discipline: r20 / r21 refuse to run unless `--seal` is given AND
`bounce_gate.md` carries a live verdict line; r22+ never load.

    python3 tap_grade.py fit 7 --bounds claimer_ne --demotion none
    python3 tap_grade.py grade 2 3 4 5 6 7 17 --bounds shipped
    python3 tap_grade.py tune-feet            # grid on train, frozen rule
    python3 tap_grade.py grade 20 21 --bounds claimer_ne --feet --seal
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import ball_replicate as br                       # noqa: E402
import court3d as c3                              # noqa: E402
import bound_oracle as bo                         # noqa: E402
from bounce_autopsy import tracked as autopsy_tracked   # noqa: E402
from claim_lab import load as c3load              # noqa: E402

DATA = HERE.parent.parent / "data" / "vision"
LABELS = DATA / "bounce_labels_chicago0725.csv"
GATE = HERE / "bounce_gate.md"
TRAIN = [2, 3, 4, 5, 6, 7, 17]
SEAL = [20, 21]
BMATCH = 0.30            # tap <-> call time match (= br.BOUNCE_MATCH_S)
FEET_GRID = dict(W=[0.15, 0.20, 0.25, 0.30], Z=[0.0, 0.25, 0.5, 1.0])
TUNE_JSON = HERE / "tap_feet_tune.json"


# ------------------------------------------------------------- truth

def truth(rally):
    """Owner's bounce taps + per-flight calls for one rally.
    flights: [(t_from, t_to, term, call, taps[(t,x,y)])] in order."""
    rows = defaultdict(list)
    for r in csv.DictReader(open(LABELS)):
        if int(r["rally_cum"]) == rally:
            rows[int(r["flight"])].append(r)
    if not rows:
        raise SystemExit(f"r{rally}: no bounce labels")
    # loader cleanup lives in make_bounce_audit.read_labels (dups, out-of-window)
    sys.path.insert(0, str(HERE.parent))
    from make_bounce_audit import read_labels
    clean = read_labels(str(LABELS), quiet=True)[rally]
    fl = []
    for fi in sorted(rows):
        r0 = rows[fi][0]
        e = clean.get(fi - 1, {"call": r0["call"], "b": []})
        fl.append(dict(t0=float(r0["t_from_s"]), t1=float(r0["t_to_s"]),
                       term=(r0["hitter_to"] == ""), call=e["call"],
                       taps=list(e["b"])))
    return fl


def flight_of(fl, t):
    for f in fl:
        if f["t0"] - 0.15 <= t <= f["t1"] + 0.15:
            return f
    return None


# -------------------------------------------------------------- cells

def check_seal(rally, seal):
    if rally >= 22:
        raise SystemExit(f"r{rally} is temporal-gate HOLDOUT — never loaded")
    if rally in SEAL:
        live = GATE.exists() and "VERDICT: LIVE" in GATE.read_text()
        if not (seal and live):
            raise SystemExit(f"r{rally} is a SEAL: needs --seal and a live "
                             f"verdict in {GATE.name}")


def bounds_for(c, kind):
    r = c["rally"]
    obs, sh_bounds, sh_evs, anchors = bo.predem(c, "raw")
    if kind == "tracked":
        return obs, list(sh_bounds), list(sh_evs), anchors
    p = HERE / (f"claimer_bounds_r{r}.json" if kind == "claimer"
                else f"claimer_bounds_ne_r{r}.json")
    if not p.exists():
        raise SystemExit(f"{p.name} missing")
    with open(p) as f:
        d = json.load(f)
    return obs, list(d["bounds"]), list(d["evs"]), anchors


def fit_cell(rally, kind, demotion, seal=False):
    """(segs, bounds) for a bounds variant; cached."""
    check_seal(rally, seal)
    cp = HERE / f"tapgrade_{kind}_{demotion}_r{rally}.pkl"
    if cp.exists():
        with open(cp, "rb") as f:
            return pickle.load(f)
    c = c3load(rally)
    if kind == "shipped":
        _, segs, bounds, _ = autopsy_tracked(c)
    else:
        obs, bounds, evs, anchors = bounds_for(c, kind)
        P, floors = c["P"], c["floors"]
        if demotion == "none":
            pa = br.bound_anchor_positions(bounds, anchors, floors)
            segs, _ = br.reconstruct(P, obs, list(bounds), list(evs), pa,
                                     corridor=True)
        else:
            segs, _, bounds, _ = br.crossing_demotion(
                P, obs, list(bounds), list(evs), floors, anchors)
    out = (segs, list(bounds))
    with open(cp, "wb") as f:
        pickle.dump(out, f)
    return out


# ------------------------------------------------------------- typing

def feet_type(segs, bounds, W, Z, dead):
    """AT-FEET rule.  For each ok ARC flight that ends at a real bound
    (not the rally end): sample the fitted arc over the last W s before
    the bound; if its lowest point is <= Z ft it is a bounce at the
    receiver's feet, timed at the first floor crossing (or the minimum
    when the arc never reaches 0).  Returns [(ts, xy, k)]."""
    out = []
    for k, s in enumerate(segs):
        if not s or not s.get("ok") or s["kind"] != "arc":
            continue
        t1 = bounds[k + 1]
        if t1 >= dead - 1e-6:
            continue
        a0, a1, th = s["arcs"][-1]
        tt = np.arange(max(a0, t1 - W), t1 + 1e-9, 1 / 120)
        if len(tt) < 2:
            continue
        X = c3.arc_pos(th, tt - a0)
        z = X[:, 2]
        if z.min() > Z:
            continue
        below = np.where(z <= max(Z, 0.0))[0]
        i = int(below[0]) if len(below) else int(np.argmin(z))
        xy = X[i, :2]
        if not (-2 <= xy[0] <= 22 and -1 <= xy[1] <= 45):
            continue
        out.append((float(tt[i]), np.array(xy), k))
    return out


def calls_for(segs, bounds, feet, dead):
    calls = [(float(s["ts"]), np.asarray(s["bounce_xy"]), k)
             for k, s in enumerate(segs)
             if s and s.get("ok") and s["kind"] == "bounce"]
    if feet:
        W, Z = feet
        calls += feet_type(segs, bounds, W, Z, dead)
    return sorted(calls, key=lambda x: x[0])


# -------------------------------------------------------------- grade

def _match(taps, calls, tol=BMATCH):
    pairs = sorted((abs(t[0] - c[0]), i, j) for i, t in enumerate(taps)
                   for j, c in enumerate(calls) if abs(t[0] - c[0]) <= tol)
    ui, uj, out = set(), set(), []
    for _, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); out.append((i, j))
    return out


SHIFT = 0.0          # --shift S: slide every machine bounce call by +S s (the null)


def grade(rally, kind, demotion, feet=None, seal=False):
    fl = truth(rally)
    c = c3load(rally)
    segs, bounds = fit_cell(rally, kind, demotion, seal)
    dead = float(c["dead"])
    calls = calls_for(segs, bounds, feet, dead)
    if SHIFT:                       # null: every machine call slid in time
        calls = [(ts + SHIFT, xy, k) for (ts, xy, k) in calls]
    taps = [(t, x, y) for f in fl for (t, x, y) in f["taps"]]
    pairs = _match(taps, calls)
    H = br.floor_homography(c["P"])
    land = []
    for i, j in pairs:
        t, x, y = taps[i]
        if x is None:
            continue
        v = H @ np.array([x, y, 1.0])
        land.append(math.hypot(v[0] / v[2] - calls[j][1][0],
                               v[1] / v[2] - calls[j][1][1]))
    matched_j = {j for _, j in pairs}
    false = 0
    for j, (ts, xy, k) in enumerate(calls):
        if j in matched_j:
            continue
        f = flight_of(fl, ts)
        if f and f["call"] in ("volley", "nobounce"):
            false += 1
        elif f and f["call"] == "bounce" and not any(
                abs(t - ts) <= BMATCH for t, _, _ in f["taps"]):
            false += 1        # bounce flight, but not where the owner put it
    imps = list(c["imps"])
    cm = sum(1 for hc in imps if bo.is_real(hc, bounds))
    junk = sum(1 for b in bounds[:-1] if not bo.is_real(b, imps))
    tap_ts = [t for f in fl if not f["term"] for (t, _, _) in f["taps"]]
    intact = bo.intact_flights(bounds, imps, tap_ts)
    feet_taps = [t for f in fl if not f["term"] for (t, _, _) in f["taps"]
                 if f["t1"] - t <= 0.10]
    feet_hit = sum(1 for i, j in pairs if taps[i][0] in feet_taps)
    return dict(rally=rally, taps=len(taps), calls=len(calls),
                matched=len(pairs), false=false, land=land,
                contacts=[cm, len(imps)], junk=junk,
                intact=[intact, len(tap_ts)],
                feet=[feet_hit, len(feet_taps)])


def fmt(g):
    l = (f"{np.median(g['land']):.1f}" if g["land"] else "  - ")
    return (f"{g['matched']:>3}/{g['taps']:<3} false {g['false']:>2} "
            f"land {l:>4} | feet {g['feet'][0]}/{g['feet'][1]} | "
            f"cont {g['contacts'][0]:>2}/{g['contacts'][1]:<2} junk {g['junk']:>2} "
            f"intact {g['intact'][0]:>2}/{g['intact'][1]:<2}")


def pooled(gs):
    p = dict(rally="all", taps=sum(g["taps"] for g in gs),
             calls=sum(g["calls"] for g in gs),
             matched=sum(g["matched"] for g in gs),
             false=sum(g["false"] for g in gs),
             land=[x for g in gs for x in g["land"]],
             contacts=[sum(g["contacts"][0] for g in gs), sum(g["contacts"][1] for g in gs)],
             junk=sum(g["junk"] for g in gs),
             intact=[sum(g["intact"][0] for g in gs), sum(g["intact"][1] for g in gs)],
             feet=[sum(g["feet"][0] for g in gs), sum(g["feet"][1] for g in gs)])
    return p


def run_grade(rallies, kind, demotion, feet, seal, quiet=False):
    gs = []
    for r in rallies:
        g = grade(r, kind, demotion, feet, seal)
        gs.append(g)
        if not quiet:
            print(f"r{r:<3} {fmt(g)}")
    p = pooled(gs)
    if not quiet:
        print(f"all  {fmt(p)}")
    return gs, p


# ---------------------------------------------------------------- tune

def tune_feet(kind, demotion):
    """FROZEN RULE (written before any number): over the train rallies,
    the AT-FEET cell maximises pooled (matched - false) vs the same
    bounds without typing; ties -> smaller W, then smaller Z (the more
    conservative cell).  Must beat no-typing by >= +2 net or the rule
    is DEAD.  Written to tap_feet_tune.json."""
    _, base = run_grade(TRAIN, kind, demotion, None, False, quiet=True)
    net0 = base["matched"] - base["false"]
    print(f"train, {kind}/{demotion}, no typing: {fmt(base)}  net {net0}")
    rows = []
    for W in FEET_GRID["W"]:
        for Z in FEET_GRID["Z"]:
            _, p = run_grade(TRAIN, kind, demotion, (W, Z), False, quiet=True)
            net = p["matched"] - p["false"]
            rows.append((net, -W, -Z, W, Z, p))
            print(f"  W {W:.2f} Z {Z:.2f}: {fmt(p)}  net {net:+d} vs {net0:+d}")
    rows.sort(reverse=True)
    net, _, _, W, Z, p = rows[0]
    verdict = "LIVE" if net >= net0 + 2 else "DEAD"
    out = dict(kind=kind, demotion=demotion, W=W, Z=Z, net=net, net0=net0,
               verdict=verdict, base=fmt(base), best=fmt(p))
    with open(TUNE_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nVERDICT feet rule: {verdict} — W {W} Z {Z}, net {net:+d} vs {net0:+d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "grade", "tune-feet"])
    ap.add_argument("rallies", nargs="*", type=int)
    ap.add_argument("--bounds", default="shipped",
                    choices=["shipped", "tracked", "claimer", "claimer_ne"])
    ap.add_argument("--demotion", default="shipped", choices=["none", "shipped"])
    ap.add_argument("--feet", nargs="*", type=float,
                    help="apply the at-feet rule: no args = the frozen "
                         "tune (tap_feet_tune.json), or W Z")
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="null: slide every machine call by +S s before matching")
    a = ap.parse_args()
    global SHIFT
    SHIFT = a.shift
    feet = None
    if a.feet is not None:
        if len(a.feet) == 2:
            feet = (a.feet[0], a.feet[1])
        else:
            d = json.load(open(TUNE_JSON))
            feet = (d["W"], d["Z"])
    if a.cmd == "fit":
        for r in a.rallies:
            segs, bounds = fit_cell(r, a.bounds, a.demotion, a.seal)
            print(f"r{r} {a.bounds}/{a.demotion}: {len(bounds)} bounds, "
                  f"{sum(1 for s in segs if s and s.get('ok'))} ok segs")
    elif a.cmd == "grade":
        run_grade(a.rallies or TRAIN, a.bounds, a.demotion, feet, a.seal)
    else:
        tune_feet(a.bounds, a.demotion)


if __name__ == "__main__":
    main()
