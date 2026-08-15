"""Steps 1 & 3 of the vision POC — sync the audio to the referee log, then
report the four numbers that decide whether Tier 2 is worth building.

    python vision/poc_report.py --timeline data/vision/rally_timeline_809fe252.csv \
                                --contacts data/vision/contacts.csv

SYNC WITHOUT OCR.  The referee log is a square wave: ~20 s inside a rally,
a few seconds out, 73 times, with irregular durations.  That pattern is
distinctive enough to align against on its own — slide the contact list
against the rally windows and take the offset that puts the most contacts
inside them.  No scorebug reading, one free parameter.  (Scorebug OCR is
the more robust upgrade and unlocks clip-linking for the whole archive, but
it is not needed to answer the POC's question.)

THE FOUR NUMBERS.
  1. sync quality     — how sharply the alignment peaks. A flat curve means
                        the contacts are not tracking the rally structure at
                        all, which is itself the answer.
  2. density contrast — contacts per second inside rally windows vs outside.
                        Free validation: no hand labels, straight from the log.
  3. contacts/rally   — should be >=1 with a sane spread (pro rallies run
                        ~4-14 shots). Zeros and 40s are both detector failure.
  4. interval modes   — the actual hypothesis: does the inter-contact
                        interval distribution separate a slow dink mode
                        (~0.5 s) from a fast speed-up mode (~0.2 s)?

Two more checks that cost nothing because the log provides them:
  * TIMEOUT CONTROL — contact density inside logged timeouts should collapse.
  * LATE CLUSTERING — the rally window starts when the referee marks it,
    which is before the serve, so contacts should sit in the LATTER part of
    each window rather than uniformly. Uniform means we are detecting noise.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_timeline(path):
    rows = []
    for r in csv.DictReader(Path(path).open()):
        rows.append({
            "rally": int(r["rally"]),
            "slot": r.get("slot") or "1",
            "start": dt.datetime.fromisoformat(r["t_start"]),
            "end": dt.datetime.fromisoformat(r["t_end"]),
            "outcome": r["outcome"],
        })
    t0 = rows[0]["start"]
    for r in rows:
        r["a"] = (r["start"] - t0).total_seconds()
        r["b"] = (r["end"] - t0).total_seconds()
    return rows, t0


def inside_count(windows, times, offset):
    """How many contact times land inside a rally window at this offset.

    Half-open [a, b) — consecutive rally windows can share a boundary when
    the referee marks the next start in the same second the last one ended,
    and closed intervals double-count a contact sitting exactly there.
    """
    t = np.sort(times - offset)
    tot = 0
    for a, b in windows:
        tot += np.searchsorted(t, b, "left") - np.searchsorted(t, a, "left")
    return int(tot)


def _end_lag_spread(windows, times, offset):
    """Spread of (window end - last contact in that window).

    The count objective is FLAT: contacts sit in the middle of a ~20 s
    window, so shifting the offset by seconds moves nobody across an edge
    and the count does not change. That plateau made a continuous VOD look
    edited in testing.

    This is the sharp anchor. The referee presses at the moment the rally
    ENDS, with a roughly constant reaction time, so the gap between the
    final contact and the logged end should be near-constant across
    rallies. Minimising its spread pins the offset far tighter than
    counting can, and it uses the one timestamp refereeing actually nails.
    """
    t = np.sort(times - offset)
    lags, n_hit = [], 0
    for a, b in windows:
        i = np.searchsorted(t, a, "left")
        j = np.searchsorted(t, b, "left")
        if j > i:
            lags.append(b - t[j - 1])
            n_hit += 1
    if n_hit < max(5, 0.6 * len(windows)):
        return np.inf, 0
    lags = np.array(lags)
    q1, q3 = np.percentile(lags, [25, 75])
    return float(q3 - q1), n_hit


def rally_end_candidates(times, min_gap=2.5):
    """Contacts followed by a long silence — i.e. rally-ending shots.

    Rallies are separated by seconds of dead time and shots inside one are
    under a second apart, so the gap structure alone marks the ends. These
    are SHARP events, unlike window membership, which is why the offset is
    refined against them.
    """
    t = np.sort(times)
    if len(t) < 2:
        return t
    gaps = np.diff(t)
    idx = np.flatnonzero(gaps >= min_gap)
    return np.concatenate([t[idx], t[-1:]])


def refine_offset(windows, times, off0, max_shift=12.0, min_gap=2.5):
    """Match detected rally-ends to logged rally-ends; take the median shift.

    Window-membership plateaus because contacts sit in the middle of ~20 s
    windows, so seconds of shift move nobody across an edge — in testing
    that plateau alone left one game 5 s out and made a continuous VOD read
    as edited. Matching END EVENTS instead uses the one timestamp
    refereeing nails, and the median over ~200 rallies absorbs the
    detector's false positives and misses.
    """
    ends = rally_end_candidates(times, min_gap) - off0
    if len(ends) < 5:
        return off0, np.nan
    ends = np.sort(ends)
    resid = []
    for _a, b in windows:
        i = np.searchsorted(ends, b)
        for j in (i - 1, i):
            if 0 <= j < len(ends) and abs(ends[j] - b) <= max_shift:
                resid.append(ends[j] - b)
    if len(resid) < max(5, 0.4 * len(windows)):
        return off0, np.nan
    resid = np.array(resid)
    # residuals cluster at the true shift plus the referee's reaction lag
    med = float(np.median(resid))
    keep = resid[np.abs(resid - med) <= 3.0]
    return off0 + float(np.median(keep)), float(np.percentile(keep, 75)
                                                - np.percentile(keep, 25))


def slot_candidates(windows, times, lo, hi, coarse=1.0, topk=12):
    """Top local maxima of the inside-count curve for one game.

    A single game's rally pattern is not unique enough to identify itself
    across a whole broadcast — 34 windows of ~20 s will align plausibly
    against a DIFFERENT game's contacts. In testing, game 1 locked onto
    game 2 and scored HIGHER than at its true offset. So each game offers
    candidates and the ordering constraint picks between them.
    """
    grid = np.arange(lo, hi, coarse)
    counts = np.array([inside_count(windows, times, o) for o in grid])
    order = np.argsort(counts)[::-1]
    picked = []
    for i in order:
        if len(picked) >= topk:
            break
        if all(abs(grid[i] - grid[j]) > 30 for j in picked):
            picked.append(i)
    return [(float(grid[i]), int(counts[i])) for i in picked]


def order_constrained_offsets(rallies, slots, times, lo, hi, coarse=1.0):
    """Choose one offset per game, jointly, so the games appear in the video
    in the same ORDER they were played and do not overlap.

    Games are broadcast in sequence, so game k cannot start before game k-1
    has finished. That constraint is what disambiguates the near-miss
    alignments above; greedy per-game fitting cannot use it, and a greedy
    pass locks in game 1's mistake. Dynamic programming over each game's
    candidate offsets maximises total contacts placed subject to the order.
    """
    spans, cands = [], []
    for s in slots:
        w = [(r["a"], r["b"]) for r in rallies if r["slot"] == s]
        spans.append((min(a for a, _ in w), max(b for _, b in w), w))
        cands.append(slot_candidates(w, times, lo, hi, coarse))
    # Constraint: games START in broadcast order. NOT "game k begins after
    # game k-1's full span" — this VOD is CONDENSED (80 min of video over
    # 107 min of play, and 97 min of that is in-game, so dead time is cut
    # from inside games as well as between them). A game's footage is
    # therefore SHORTER than its wall-clock span, and requiring non-overlap
    # of wall-clock spans rejects the true solution.
    n = len(slots)
    best = [[(-1, None)] * len(cands[i]) for i in range(n)]
    for j, (o, c) in enumerate(cands[0]):
        best[0][j] = (c, None)
    for i in range(1, n):
        a_i, a_prev = spans[i][0], spans[i - 1][0]
        for j, (o, c) in enumerate(cands[i]):
            start_audio = a_i + o
            bestprev, arg = -1, None
            for k, (po, _) in enumerate(cands[i - 1]):
                if best[i - 1][k][0] < 0:
                    continue
                if start_audio >= a_prev + po - 5.0:
                    if best[i - 1][k][0] > bestprev:
                        bestprev, arg = best[i - 1][k][0], k
            if arg is not None:
                best[i][j] = (bestprev + c, arg)
    endj = max(range(len(cands[n - 1])), key=lambda j: best[n - 1][j][0])
    if best[n - 1][endj][0] < 0:                     # no ordered path: give up
        return {s: slot_candidates(
            [(r["a"], r["b"]) for r in rallies if r["slot"] == s],
            times, lo, hi, coarse, topk=1)[0][0] for s in slots}, False
    chain, j = [None] * n, endj
    for i in range(n - 1, -1, -1):
        chain[i] = cands[i][j][0]
        j = best[i][j][1]
    out = {}
    for i, s in enumerate(slots):
        o, _ = refine_offset(spans[i][2], times, chain[i])
        o, _ = refine_offset(spans[i][2], times, o, max_shift=5.0)
        out[s] = float(o)
    return out, True


def find_offset(windows, times, lo, hi, coarse=1.0, fine=0.05, refine=True):
    """Coarse count-based search, then refine against rally-end events."""
    grid = np.arange(lo, hi, coarse)
    counts = np.array([inside_count(windows, times, o) for o in grid])
    best = float(grid[int(np.argmax(counts))])
    if not refine:
        fgrid = np.arange(best - coarse, best + coarse, fine)
        fc = np.array([inside_count(windows, times, o) for o in fgrid])
        return float(fgrid[int(np.argmax(fc))]), grid, counts
    o1, _ = refine_offset(windows, times, best)
    o2, _ = refine_offset(windows, times, o1, max_shift=5.0)
    return float(o2), grid, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--search-lo", type=float, default=-600.0,
                    help="seconds; video may start well before the first rally")
    ap.add_argument("--search-hi", type=float, default=3600.0)
    ap.add_argument("--fast-cut", type=float, default=0.35,
                    help="interval below this counts as a fast/speed-up shot")
    # absolute so the script behaves the same from any working directory
    ap.add_argument("--out", default=str(ROOT / "data" / "vision" / "poc_report.json"))
    args = ap.parse_args()

    rallies, t0 = load_timeline(args.timeline)
    windows = [(r["a"], r["b"]) for r in rallies]
    times = np.array([float(r["t_audio_s"])
                      for r in csv.DictReader(Path(args.contacts).open())])
    span = windows[-1][1] - windows[0][0]
    live = sum(b - a for a, b in windows)
    print(f"timeline: {len(rallies)} rallies, {span/60:.1f} min, "
          f"{live/span:.0%} live")
    print(f"contacts: {len(times)} onsets over "
          f"{(times.max()-times.min())/60:.1f} min of audio\n")

    # ---- 1. sync -------------------------------------------------------
    off, grid, counts = find_offset(windows, times, args.search_lo, args.search_hi)
    inside = inside_count(windows, times, off)
    base = np.median(counts)
    peak_ratio = inside / base if base > 0 else float("inf")
    print(f"1. SYNC            offset {off:+.2f}s")
    print(f"                   peak/median of the alignment curve = "
          f"{peak_ratio:.2f}x  (flat ~1.0 means no real alignment)")
    print("                   NB the offset absorbs the referee's reaction lag "
          "as a constant\n                   (~1s): it is refined by matching "
          "rally-END events, and the ref\n                   presses just after "
          "the point. Fine for windowing and for the\n                   "
          "cross-game agreement below; not a broadcast-delay measurement.")

    # ---- 1b. per-game offsets: the free cross-validation ---------------
    # All games of a matchup share one absolute clock, so a CONTINUOUS VOD
    # needs one offset for all of them. Fitting each game separately gives
    # independent estimates that must agree — and if they don't, the
    # disagreements locate the broadcast's edits. Either outcome is useful,
    # which is why a full-matchup VOD beats a single game.
    slots = sorted({r["slot"] for r in rallies}, key=lambda s: (len(s), s))
    per_slot = {}
    if len(slots) > 1:
        print("\n1b. PER-GAME SYNC  (all games share one clock, so these should agree)")
        # Each game searches the FULL range (a broadcast that cuts the breaks
        # runs shorter than the wall clock it covers, so later games can sit
        # many minutes off a global fit), but the choice is made JOINTLY under
        # the constraint that games appear in broadcast order.
        per_slot, ordered = order_constrained_offsets(
            rallies, slots, times, args.search_lo, args.search_hi)
        for s in slots:
            w = [(r["a"], r["b"]) for r in rallies if r["slot"] == s]
            print(f"    slot {s}: offset {per_slot[s]:+9.2f}s   "
                  f"{inside_count(w, times, per_slot[s])} contacts in "
                  f"{len(w)} rally windows")
        if not ordered:
            print("    WARNING: no ordering-consistent solution — the games do "
                  "not appear in\n             broadcast order, so at least one "
                  "offset is unreliable")
        drift = [per_slot[s] - per_slot[slots[0]] for s in slots]
        if max(abs(d) for d in drift) > 30:
            print("    cumulative drift vs game 1: "
                  + "  ".join(f"{d/60:+.1f} min" for d in drift))
            print("    -> the broadcast removes the breaks between games; each "
                  "game needs its\n       own offset, which is what the metrics "
                  "below use.")
        spread = max(per_slot.values()) - min(per_slot.values())
        # Judge the spread against the ESTIMATOR'S OWN RESOLUTION, not an
        # arbitrary threshold. Contacts sit in the middle of ~20 s windows,
        # so a range of offsets puts exactly the same contacts inside — the
        # objective has a plateau, and offsets cannot be resolved finer than
        # that. Calling a VOD "edited" on a spread smaller than the plateau
        # would be reading noise.
        print(f"    plateau half-widths: ", end="")
        halves = []
        for s in slots:
            w = [(r["a"], r["b"]) for r in rallies if r["slot"] == s]
            o = per_slot[s]
            g = np.arange(o - 20, o + 20, 0.25)
            c = np.array([inside_count(w, times, x) for x in g])
            ok = g[c >= c.max() - max(1, 0.002 * c.max())]
            halves.append((ok.max() - ok.min()) / 2.0)
            print(f"{halves[-1]:.1f}s ", end="")
        res = float(np.median(halves))
        print(f" (median {res:.1f}s)")
        print(f"    spread {spread:.2f}s vs resolution ~{res:.1f}s  ->  ", end="")
        if spread <= 2 * res:
            print("CONSISTENT: within what this estimator can resolve, so the\n"
                  "                    VOD reads as continuous and the sync "
                  "checks out four ways")
        else:
            print("DIVERGENT: larger than the estimator's resolution, so the\n"
                  "                    broadcast is edited between games — each "
                  "game still works\n                    on its own offset, and "
                  "the jumps locate the cuts")
    # Downstream, work in AUDIO time: shift each game's windows by its own
    # offset rather than shifting the contacts. Strictly more accurate when
    # the broadcast is edited, identical when it isn't.
    # The refinement anchors each rally's LAST contact on the logged end, so
    # that contact sits exactly on the boundary and the half-open rule drops
    # it — one lost contact per rally, ~10% of the total. Pad the end by a
    # hair, clipped at the next window's start so nothing double-counts.
    t = times
    aw = []
    for i, r in enumerate(rallies):
        o = per_slot.get(r["slot"], off)
        a, b = r["a"] + o, r["b"] + o
        if i + 1 < len(rallies):
            nxt = rallies[i + 1]["a"] + per_slot.get(rallies[i + 1]["slot"], off)
            b = min(b + 0.5, max(b, nxt - 1e-6))
        else:
            b += 0.5
        aw.append((a, b))
    # Count each contact AT MOST ONCE. With per-game offsets the audio-time
    # windows of different games can overlap, and summing per-window counts
    # then reports more contacts inside than exist (104% in testing).
    ts = np.sort(t)
    claimed = np.zeros(len(ts), dtype=bool)
    for a, b in aw:
        claimed[np.searchsorted(ts, a, "left"):np.searchsorted(ts, b, "left")] = True
    inside = int(claimed.sum())
    live = sum(b - a for a, b in aw)
    span_audio = max(b for _, b in aw) - min(a for a, _ in aw)
    dead = max(1e-9, span_audio - live)

    # ---- 2. density contrast ------------------------------------------
    dens_in = inside / live
    dens_out = max(0, len(t) - inside) / dead
    print(f"\n2. DENSITY         {inside}/{len(times)} contacts inside rally "
          f"windows ({inside/len(times):.0%})")
    print(f"                   inside {dens_in:.3f}/s   outside {dens_out:.3f}/s"
          f"   ratio {dens_in/max(dens_out,1e-9):.1f}x")

    # ---- 3. contacts per rally ----------------------------------------
    per = np.array([int(np.searchsorted(ts, b, "left")
                        - np.searchsorted(ts, a, "left")) for a, b in aw])
    print(f"3. PER RALLY       median {np.median(per):.0f}   "
          f"p10 {np.percentile(per,10):.0f}   p90 {np.percentile(per,90):.0f}   "
          f"zeros {int(np.sum(per==0))}/{len(per)}")

    # ---- late clustering (free check) ---------------------------------
    rel = []
    for a, b in aw:
        sel = t[(t >= a) & (t <= b)]
        if len(sel) and b > a:
            rel.extend((sel - a) / (b - a))
    rel = np.array(rel)
    if len(rel):
        print(f"   late-clustering  mean position in window "
              f"{rel.mean():.2f} (0.5 = uniform; >0.5 expected, the referee "
              f"marks the start before the serve)")

    # ---- 4. interval modes --------------------------------------------
    iv = []
    for a, b in aw:
        sel = np.sort(t[(t >= a) & (t <= b)])
        if len(sel) > 1:
            iv.extend(np.diff(sel))
    iv = np.array([x for x in iv if x < 2.0])
    print(f"4. INTERVALS       n={len(iv)}")
    if len(iv) > 30:
        fast, slow = iv[iv < args.fast_cut], iv[iv >= args.fast_cut]
        print(f"   fast (<{args.fast_cut:.2f}s)  n={len(fast):5d}  "
              f"median {np.median(fast)*1000:.0f} ms")
        print(f"   slow (>={args.fast_cut:.2f}s) n={len(slow):5d}  "
              f"median {np.median(slow)*1000:.0f} ms")
        edges = np.arange(0, 1.25, 0.05)
        h, _ = np.histogram(iv, bins=edges)
        top = h.max()
        print("\n   inter-contact interval histogram")
        for i, c in enumerate(h):
            if edges[i] > 1.2:
                break
            bar = "#" * int(40 * c / max(top, 1))
            print(f"   {edges[i]:4.2f}-{edges[i+1]:4.2f}s |{bar} {c}")
        # crude bimodality: is there a dip between two peaks?
        sm = np.convolve(h, np.ones(3) / 3, mode="same")
        pk = [i for i in range(1, len(sm) - 1) if sm[i] > sm[i-1] and sm[i] >= sm[i+1]]
        print(f"\n   smoothed peaks at: "
              f"{[f'{edges[i]:.2f}s' for i in pk] or 'none — unimodal'}")

    rep = {"offset_s": off, "per_game_offsets": per_slot,
           "per_game_spread_s": (max(per_slot.values()) - min(per_slot.values()))
           if len(per_slot) > 1 else 0.0,
           "n_contacts": int(len(times)),
           "inside_fraction": float(inside / len(times)),
           "peak_ratio": float(peak_ratio),
           "density_in": float(dens_in), "density_out": float(dens_out),
           "per_rally_median": float(np.median(per)),
           "per_rally_zeros": int(np.sum(per == 0)),
           "mean_window_position": float(rel.mean()) if len(rel) else None,
           "n_intervals": int(len(iv)),
           "fast_median_ms": float(np.median(iv[iv < args.fast_cut]) * 1000)
           if len(iv) and np.any(iv < args.fast_cut) else None,
           "slow_median_ms": float(np.median(iv[iv >= args.fast_cut]) * 1000)
           if len(iv) and np.any(iv >= args.fast_cut) else None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
