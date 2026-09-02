"""train_read — the first read of the incumbent stack on a NEW TRAIN rally
(2026-09-02, written for r17, the first click-package delivery).

Train rallies only: this is a diagnostic, not a grade. It refuses the
evaluation rallies (r9 / r10) and the seals (r20 / r21) outright —
those go through the registered one-shots, never this script.

What it prints, for `python3 train_read.py <rally>`:
  1. the incumbent path-first track (adopted tune cell) scored against
     the owner's V/S clicks the usual way (cdp.score: at-click, r@8/12/20,
     prec@12), split by click kind (V clean / S streak), plus the two
     nulls (displaced, time-shift);
  2. gap fill v2 (adopted product): tracked half bit-identical check,
     inferred stratum r@12 / prec@12 on its own;
  3. WHERE the misses are, by the owner's contact labels: each V/S click
     is assigned to the nearest manual contact within +-0.30 s and the
     hit rate is reported per shot type (serve / return / slow / fast /
     lunge) and for clicks that sit between contacts (flight middles);
  4. a per-flight table (span, n, rms, launch) as pathfirst prints.

Nothing is tuned here; no knob is read from anything but the committed
tune records. Output is meant to be tee'd to train_read_r{N}.txt.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, "/home/user/pickleball/vision")

import pathfirst as pf          # noqa: E402
import gapfill as gf            # noqa: E402
import geom_fix                 # noqa: E402
import corridor_dp as cdp       # noqa: E402
import court3d as c3            # noqa: E402

CONTACTS = Path("/home/user/pickleball/data/vision/contact_labels_chicago0725.csv")
SPLIT = Path("/home/user/pickleball/data/vision/label_split.csv")
FORBIDDEN = {9, 10, 20, 21}     # evaluation + seals: registered one-shots only
T_ASSIGN = 0.30                 # s: a click belongs to a contact within this
FPS = 60


def contacts_for(rally):
    out = []
    for r in csv.DictReader(open(CONTACTS)):
        if int(r["rally_cum"]) != rally or r["source"] not in ("manual", "divergent"):
            continue
        if r.get("contact", "1") == "0":
            continue
        out.append((float(r["t_refined_s"] or r["t_tap_s"]), r["shot_type"]))
    return sorted(out)


def assert_train(rally):
    if rally in FORBIDDEN:
        raise SystemExit(f"rally {rally} is evaluation/seal — use the registered grade, not train_read")
    for r in csv.DictReader(open(SPLIT)):
        if int(r["rally_cum"]) == rally:
            if r["split"] != "train":
                raise SystemExit(f"rally {rally} is {r['split']} — untouchable")
            return
    raise SystemExit(f"rally {rally} not in label_split.csv")


def hit(track, t, tx, ty, t0):
    f = int(round((t - t0) * FPS))
    p = track.get(f) or track.get(f - 1) or track.get(f + 1)
    if p is None:
        return None
    return float(np.hypot(p[0] - tx, p[1] - ty)) <= cdp.R_MAIN


def main():
    rally = int(sys.argv[1])
    assert_train(rally)
    pc = json.loads(pf.TUNE_JSON.read_text())
    assert not pc.get("dead")
    cell = dict(p_seed=pc["p_seed"], s_min=pc["s_min"], gap=int(pc["gap"]))
    g2 = json.loads((SP / "gapfill_tune2.json").read_text())
    g2 = dict(gap_max=float(g2["gap_max"]), d_meet=float(g2["d_meet"]))

    ctx = pf.context(rally)
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    print(f"rally {rally} TRAIN READ: {len(truth)} V/S clicks, decode@12 "
          f"{sum(dec)}/{len(dec)}; path-first cell {cell} (p-cache '{ctx['pxs']}'), "
          f"gap fill v2 {g2}")

    # 1. incumbent path-first track
    cdp.W_P_SOFT = 25.0
    res = pf.run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
    tr = res["track"]
    print(f"  path-first: hyp {res['n_hyp']} kept {res['n_kept']} flights "
          f"{res['n_fl']} selected {len(res['chosen'])}")
    cdp.score(tr, truth, t0, dec, "path-first")
    for vis in ("V", "S"):
        tt = [x for x in truth if x[3] == vis]
        dd = [d for x, d in zip(truth, dec) if x[3] == vis]
        cdp.score(tr, tt, t0, dd, f"  pf[{vis}]")
    rng = np.random.default_rng(pf.NULL_SEED)
    cdp.score(pf.displaced(tr, rng), truth, t0, dec, "null-disp")
    cdp.score(pf.timeshift(tr, ctx, rng), truth, t0, dec, "null-tshift")

    # 2. gap fill v2 product
    r2 = gf.run(ctx, g2)
    tracked = {f: xy for f, xy in r2["track"].items() if f not in r2["inferred"]}
    print(f"  gap fill v2: tracked half bit-identical to path-first: {tracked == tr}; "
          f"inferred frames {len(r2['inferred'])}")
    cdp.score(r2["track"], truth, t0, dec, "tracked+inf")
    inf_only = {f: xy for f, xy in r2["track"].items() if f in r2["inferred"]}
    cdp.score(inf_only, truth, t0, dec, "inferred")
    h_d, h_t = gf.nulls(ctx, inf_only, rng)
    print(f"  inferred nulls: displaced {h_d}  time-shift {h_t}")
    full = r2["track"]

    # 3. where the misses are, by the owner's contact labels
    cons = contacts_for(rally)
    print(f"-- misses by contact type ({len(cons)} manual contacts, clicks assigned within "
          f"+-{T_ASSIGN:.2f} s; 'middle' = no contact that close)")
    print("    bucket      n   pf-hit  pf-miss  no-pt | +fill-hit")
    rows = {}
    for (t, tx, ty, vis) in truth:
        near = min(cons, key=lambda c: abs(c[0] - t)) if cons else None
        key = near[1] if near and abs(near[0] - t) <= T_ASSIGN else "middle"
        rows.setdefault(key, []).append((t, tx, ty, vis))
    order = ["serve", "return", "slow", "fast", "lunge", "middle"]
    for key in order + [k for k in rows if k not in order]:
        if key not in rows:
            continue
        pts = rows[key]
        h = m = n = hf = 0
        for (t, tx, ty, vis) in pts:
            r = hit(tr, t, tx, ty, t0)
            if r is None:
                n += 1
            elif r:
                h += 1
            else:
                m += 1
            hf += bool(hit(full, t, tx, ty, t0))
        print(f"    {key:10s} {len(pts):3d}   {h:4d}    {m:4d}    {n:4d} | {hf:4d}")
    # same split, streak clicks only
    print("    (S clicks only)")
    for key in order:
        pts = [p for p in rows.get(key, []) if p[3] == "S"]
        if not pts:
            continue
        h = sum(1 for (t, tx, ty, vis) in pts if hit(tr, t, tx, ty, t0))
        n = sum(1 for (t, tx, ty, vis) in pts if hit(tr, t, tx, ty, t0) is None)
        print(f"    {key:10s} {len(pts):3d}   {h:4d}    {len(pts) - h - n:4d}    {n:4d}")

    # 3b. per contact: the clicks in its +-T_ASSIGN window and how many the
    #     path-first track / the filled track cover (coverage per shot)
    print(f"-- per contact (clicks within +-{T_ASSIGN:.2f} s): t type  n  pf  +fill  S-clicks")
    for (tc, typ) in cons:
        pts = [p for p in truth if abs(p[0] - tc) <= T_ASSIGN]
        h = sum(1 for (t, tx, ty, vis) in pts if hit(tr, t, tx, ty, t0))
        hf = sum(1 for (t, tx, ty, vis) in pts if hit(full, t, tx, ty, t0))
        ns = sum(1 for p in pts if p[3] == "S")
        print(f"   {tc:7.2f} {typ:6s} {len(pts):3d} {h:3d} {hf:5d}  {ns:3d}")

    # 4. flights
    print("-- flights: span | n w rms | density | launch")
    for fl in res["chosen"]:
        v0 = c3.arc_vel(fl["theta"], 0.0)
        sp = np.linalg.norm(v0)
        loft = np.degrees(np.arctan2(v0[2], np.hypot(v0[0], v0[1])))
        print(f"   {t0 + fl['fa'] / FPS:7.2f}-{t0 + fl['fb'] / FPS:7.2f} "
              f"{(fl['fb'] - fl['fa'] + 1) / FPS:4.2f}s | {fl['n']:3d} "
              f"{fl['w']:5.1f} {fl['rms']:4.1f} | {fl['density']:.2f} | "
              f"{sp:5.1f} ft/s loft {loft:5.1f} k {fl['theta'][6]:.2f}")


if __name__ == "__main__":
    main()
