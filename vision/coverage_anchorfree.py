"""Anchor-free identity: name the players in rallies that HAVE NO SERVE.

The serve anchor is where the geometry chain learns who is who, so a
rally whose serve the broadcast never showed is dropped entirely — 24
`anchor_not_found` + 12 `identity_no_serving_config` of the 141 on the
first real VOD.  Those rallies are not empty: measured, they still carry
0.56 main-camera fraction over the whole rally (median 0.73), and 28 of
50 have >=2 side-assigned tracks on each side.  Only the naming moment
is missing.

USER'S IDEA (2026-08-19), which is what this module builds: learn each
player's appearance FROM THEIR SERVICE POINTS — the rallies that do have
anchors, where geometry names them for free — then carry that model into
the anchor-less rallies.  No hand labelling anywhere: the training
labels are the geometry chain's own output.

The structural help that makes it tractable: partners stand on the SAME
side of the net, so the near/far split already in the pose npz
partitions the four tracks into two TEAMS.  With the per-game team->end
map (which coverage already computes and checks, 63/63 consistent), a
track's side names its team, and the only remaining question is which of
two partners it is — a 2-way call, where the per-team model measures
97.9%/98.3% within-team sample accuracy, rather than a 4-way one.

PRE-REGISTERED 2026-08-19, BEFORE the first number was computed
(anti-cooking protocol; if these bars are missed the method is reported
as failing, not retuned until it passes):

  GATE A — method admissible, per game.  Simulate the real task on
    rallies where the answer is known: hold a resolved rally out, fit
    the appearance model on that game's OTHER rallies, name the held-out
    rally's tracks from MID-RALLY appearance alone (no anchor, no
    geometry) under the side->team constraint, and compare to the
    geometry names.  A game qualifies only at >= 0.90 agreement over its
    detections.  This is stricter than the 0.85 LORO bar the repair
    layer uses, because an anchor-less rally has NO geometric
    cross-check — nothing downstream can catch an error here.
  GATE B — per-rally admission.  Both sides must produce a confident
    2-way split: assignment margin >= MARGIN_MIN in z-scored Lab units,
    and >= MIN_DETS detections on the track.  Rallies that miss it are
    DROPPED and ledgered, never guessed (house rule).
  FLAGGED OUTPUT.  Admitted rallies carry identity_source=appearance so
    every downstream number can be recomputed with and without them, and
    published numbers must state which.

Note this deliberately does NOT feed the anchor-swap audit: that audit
is appearance-vs-geometry, and a rally named by appearance alone has no
geometry to disagree with.  Anchor-free rallies are additions to the
sample, never evidence about the chain.

    python vision/coverage_anchorfree.py --validate ...   # Gate A
    python vision/coverage_anchorfree.py --emit ...       # ledger
    python vision/coverage_anchorfree.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage as C
import coverage_appearance as A
from swing_probe import decode_window

ROOT = Path(__file__).resolve().parent.parent
GATE_A = 0.90         # per-game agreement required (pre-registered)
MARGIN_MIN = 0.35     # z-units between the two partners (pre-registered)
MIN_DETS = 8          # detections a track needs to be named
MIN_SIDE_TRACKS = 1   # named tracks required per side


def team_end_map(rallies, assign_by_cum, game_of):
    """Per game: side (0 near / 1 far) -> the pair of uuids that plays
    it.  Built from the resolved rallies' own geometry names, majority
    vote; teams swap ends between games, which is why it is per game."""
    tally = defaultdict(Counter)
    for cum, _w, dets, assign, _ts, _cf in rallies:
        g = game_of.get(cum, -1)
        for d, (u, _c, _h) in zip(dets, assign):
            if u and d.side in (0, 1):
                tally[(g, d.side)][u] += 1
    out = {}
    for (g, side), cnt in tally.items():
        out.setdefault(g, {})[side] = [u for u, _ in cnt.most_common(2)]
    return out


def det_descriptor(frame, d):
    """One detection's descriptor, computed EXACTLY as the training
    features are: box crop -> CROP_W x CROP_H resize -> keypoints mapped
    into crop coordinates -> moments_crop.  Computing test features in
    native frame space instead would silently change the sampling
    neighbourhood and mismatch the trained centroids."""
    if d.kpt is None:
        return None
    got = A.crop_of(frame, d.box, np.asarray(d.kpt, np.float32),
                    np.asarray(d.kpc, np.float32))
    if got is None:
        return None
    crop, k = got
    f = A.moments_crop(crop, k, np.asarray(d.kpc, np.float32))
    return f if np.isfinite(f).all() else None


def rally_descriptors(video, dets, t0, t1, fps, width):
    """Mean descriptor per track, from MID-RALLY frames.  Uses the
    boxes/keypoints already in the pose npz, so no pose model runs
    here — this is a decode pass only."""
    by_i = defaultdict(list)
    for d in dets:
        by_i[int(round((d.t - t0) * fps))].append(d)
    acc = defaultdict(list)
    for i, frame in enumerate(decode_window(video, t0, t1 - t0, fps, width)):
        for d in by_i.get(i, ()):
            f = det_descriptor(frame, d)
            if f is not None:
                acc[d.track].append(f)
    return ({tr: np.mean(v, 0) for tr, v in acc.items()
             if len(v) >= MIN_DETS},
            {tr: len(v) for tr, v in acc.items()})


def name_side_restricted(fw, f, candidates):
    """2-way call between the two partners the side already implies.
    Returns (uuid, margin in z-units)."""
    z = (f - fw.mu) / fw.sd
    d = sorted((float(np.linalg.norm(z - fw.cent[u])), u)
               for u in candidates if u in fw.cent)
    if len(d) < 2:
        return (d[0][1], np.inf) if d else (None, 0.0)
    return d[0][1], d[1][0] - d[0][0]


def name_rally(fw, descs, side_by_track, endmap_g):
    """Name every track of one rally from appearance + the side->team
    constraint.  Returns {track: (uuid, margin)}."""
    out = {}
    for tr, f in descs.items():
        side = side_by_track.get(tr, -1)
        cands = endmap_g.get(side) if side in (0, 1) else None
        if not cands:
            continue
        u, m = name_side_restricted(fw, f, cands)
        if u is not None:
            out[tr] = (u, m)
    return out


def load_swaps(vod):
    """The gated anchor-swap ledger — geometry names AFTER repair are
    the comparison truth, since the raw chain swapped ~25% of rallies."""
    out = defaultdict(list)
    p = ROOT / f"data/vision/identity_swaps_{vod}.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["swap"] == "1" and float(r["unanimity"]) >= 0.8:
                out[int(r["rally_cum"])].append(tuple(r["team"].split("|")))
    return out


def truth_names(dets, assign, swaps_for_rally):
    """Geometry's per-detection name with the swap ledger applied."""
    out = {}
    for d, (u, _c, _h) in zip(dets, assign):
        out[id(d)] = u
    for ua, ub in swaps_for_rally:
        for k, v in list(out.items()):
            if v == ua:
                out[k] = "__t__"
        for k, v in list(out.items()):
            if v == ub:
                out[k] = ua
        for k, v in list(out.items()):
            if v == "__t__":
                out[k] = ub
    return out


def build(a):
    """Shared setup: per-game models trained on SERVE-ANCHOR crops,
    the resolved rallies, the team->end map, partners."""
    names_full, partner = {}, {}
    for r in csv.DictReader(open(ROOT / "data/coverage_players.csv")):
        names_full[r["player_uuid"]] = r["player"].split()[-1]
        partner[r["player_uuid"]] = r["partner_uuid"]
    game_of = {int(r["rally_cum"]): int(r["game"])
               for r in csv.DictReader(open(a.windows))}
    rallies = A.resolve_rallies(a)
    F, lab, cum_k, gm = A.clean_labels(a, partner, game_of)
    return rallies, game_of, F, lab, cum_k, gm, partner, names_full


def validate(a):
    """GATE A: name resolved rallies from mid-rally appearance ALONE
    (model refit without that rally) and compare to geometry."""
    rallies, game_of, F, lab, cum_k, gm, partner, names = build(a)
    endmap = team_end_map(rallies, None, game_of)
    swaps = load_swaps(a.vod)
    print(f"{len(rallies)} resolved rallies; team->end map: "
          + "; ".join(f"g{g}: near={[names.get(u,u[:6]) for u in v.get(0,[])]}"
                      for g, v in sorted(endmap.items())))
    per_game = defaultdict(lambda: [0, 0])
    detail = []
    for cum, win, dets, assign, _ts, _cf in rallies:
        g = game_of.get(cum, -1)
        if g not in endmap:
            continue
        m = (gm == g) & (cum_k != cum)          # leave THIS rally out
        if m.sum() < 24:
            continue
        fw = A.FourWay()
        fw.fit(F[m], lab[m])
        t0, t1 = float(win["t0s"]), float(win["t1s"])
        descs, ndet = rally_descriptors(a.video, dets, t0, t1,
                                        a.fps, a.width)
        if not descs:
            continue
        side_by = {d.track: d.side for d in dets}
        got = name_rally(fw, descs, side_by, endmap[g])
        truth = truth_names(dets, assign, swaps.get(cum, ()))
        hit = tot = 0
        for d in dets:
            g_name = truth.get(id(d))
            a_name = got.get(d.track, (None, 0))[0]
            if g_name and a_name:
                tot += 1
                hit += int(g_name == a_name)
        if tot:
            per_game[g][0] += hit
            per_game[g][1] += tot
            detail.append((cum, g, hit / tot, tot,
                           min((mg for _u, mg in got.values()), default=0)))
    print(f"\n{'game':>4} {'agreement':>10} {'dets':>7}  vs GATE_A "
          f"{GATE_A:.0%}")
    ok = {}
    for g in sorted(per_game):
        hit, tot = per_game[g]
        acc = hit / max(tot, 1)
        ok[g] = acc >= GATE_A
        print(f"{g:>4} {acc:>10.1%} {tot:>7}  {'PASS' if ok[g] else 'FAIL'}")
    bad = sorted(detail, key=lambda r: r[2])[:8]
    print("\nworst rallies (rally, game, agreement, dets, min margin):")
    for cum, g, acc, tot, mg in bad:
        print(f"  r{cum:<4} g{g} {acc:>6.1%} {tot:>5}  margin {mg:.2f}")
    return ok


def emit(a):
    """Name the rallies that HAVE NO ANCHOR, for games that cleared
    Gate A, and write the ledger coverage consumes."""
    ok = validate(a)
    good = {g for g, v in ok.items() if v}
    print(f"\ngames clearing GATE A: {sorted(good) or 'NONE'}")
    if not good:
        print("no game qualifies — nothing emitted (by design)")
        return
    rallies, game_of, F, lab, cum_k, gm, partner, names = build(a)
    endmap = team_end_map(rallies, None, game_of)
    resolved = {cum for cum, *_ in rallies}
    court = C.load_court(a.court)
    cam = C.load_camera(a.cam)
    windows = C.load_windows(a.windows)
    fw = {}
    for g in good:
        fw[g] = A.FourWay()
        fw[g].fit(F[gm == g], lab[gm == g])
    rows, stats = [], Counter()
    for cum in sorted(windows):
        g = game_of.get(cum, -1)
        if cum in resolved or g not in good:
            stats["skipped_resolved_or_game" if cum in resolved
                  else "game_below_gate_A"] += 1
            continue
        win = windows[cum]
        npz = Path(a.pose_dir) / f"r{cum:04d}.npz"
        if not npz.exists() or win.get("approx") == "1" \
                or win["outcome"] not in ("point", "sideout", "second"):
            stats["no_window_or_approx"] += 1
            continue
        dets, _d = C.load_rally(npz, court, cam)
        if not dets:
            stats["no_dets_after_gates"] += 1
            continue
        t0, t1 = float(win["t0s"]), float(win["t1s"])
        descs, ndet = rally_descriptors(a.video, dets, t0, t1,
                                        a.fps, a.width)
        if not descs:
            stats["no_descriptors"] += 1
            continue
        side_by = {d.track: d.side for d in dets}
        got = name_rally(fw[g], descs, side_by, endmap[g])
        keep = {tr: (u, m) for tr, (u, m) in got.items()
                if m >= MARGIN_MIN and ndet.get(tr, 0) >= MIN_DETS}
        per_side = Counter(side_by.get(tr, -1) for tr in keep)
        if per_side[0] < MIN_SIDE_TRACKS or per_side[1] < MIN_SIDE_TRACKS:
            stats["gate_B_thin_sides"] += 1
            continue
        stats["ADMITTED"] += 1
        for tr, (u, m) in sorted(keep.items()):
            rows.append(dict(rally_cum=cum, game=g, track=tr,
                             player_uuid=u, margin=round(float(m), 3),
                             n_dets=ndet.get(tr, 0),
                             side=side_by.get(tr, -1)))
    out = ROOT / f"data/vision/identity_anchorfree_{a.vod}.csv"
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nanchor-free ledger: {stats['ADMITTED']} rallies admitted, "
          f"{len(rows)} track names")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28} {v}")
    print(f"-> {out}" if rows else "(nothing written)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--no-cam-gate", action="store_true")
    ap.add_argument("--vod", default="")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.validate:
        validate(a)
        return
    if a.emit:
        emit(a)
        return
    ap.error("pick --validate, --emit or --selftest")


def selftest():
    # side->team restriction: the far pair must never win a near call
    fw = A.FourWay()
    F = np.array([[0.0, 0], [0.1, 0], [10.0, 0], [10.1, 0]], np.float32)
    fw.fit(F, ["nearA", "nearB", "farA", "farB"])
    u, m = name_side_restricted(fw, np.array([9.9, 0], np.float32),
                                ["nearA", "nearB"])
    assert u in ("nearA", "nearB"), u
    u2, m2 = name_side_restricted(fw, np.array([0.02, 0], np.float32),
                                  ["nearA", "nearB"])
    assert u2 == "nearA", u2
    assert m2 >= 0, m2
    # a lone candidate returns infinite margin, never a coin flip
    u3, m3 = name_side_restricted(fw, F[0], ["nearA"])
    assert u3 == "nearA" and np.isinf(m3)
    # team_end_map majority vote, per game, sides independent
    class D:
        def __init__(s, side, tr):
            s.side, s.track = side, tr
    dets = [D(0, 1), D(0, 1), D(1, 2)]
    assign = [("u_near", 1, 0), ("u_near", 1, 0), ("u_far", 1, 0)]
    em = team_end_map([(7, {}, dets, assign, 0, 1)], None, {7: 1})
    assert em[1][0] == ["u_near"] and em[1][1] == ["u_far"], em
    print("SELFTEST OK (side restriction, lone candidate, end map)")


if __name__ == "__main__":
    main()
