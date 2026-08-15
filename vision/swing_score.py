"""Gate B scorer: turn probe output + hand labels into the gate verdict.

Two stages, kept strictly apart so the hand labels are touched ONCE:

STAGE 1 — LABEL-FREE operating point.
    The probe emitted every wrist-speed peak and every audio onset above
    low floors.  A contact = swing peak (v >= theta_v) with a coincident
    pop (z >= theta_z, |dt| <= coinc).  The operating point is chosen on
    label-free criteria only: maximize side-ALTERNATION subject to a
    plausible contact rate (0.30-0.85 contacts/s; real play is ~0.58).
    Side alternation is physics: every legal shot crosses the net, so
    consecutive contacts alternate near/far.  Misses and false positives
    both break it, which makes it a cleanliness score no labels can leak
    into.

STAGE 2 — score the frozen point against the labels.
    Detected contact sequences are aligned to the labeled shot sequences
    per rally (order-preserving Needleman-Wunsch on the TEAM sequence —
    the labels carry who hit; the detector carries near/far mapped to
    teams by the serve anchor).  Out come recall per shot type, overall
    precision, and the pre-registered verdict.

ALTERNATION vs RECALL — the consistency curve (pre-registered 2026-08-12,
BEFORE any probe data existed; this corrects the original G1 bar, which
was arithmetically inconsistent with the G2 bar):
    truth alternates sides strictly, so a detected consecutive pair is
    same-side iff an ODD number of shots was missed between them.  With
    iid recall r the gap is geometric and P(odd gap) = 1/(2-r); falses
    (precision q) land on a ~random side, and with sparse falses a pair
    involving one alternates at ~1/2, so
        alt(r, q) ~ q^2 / (2 - r) + (1 - q^2) / 2
    (verified by simulation in --selftest: 0.741 observed vs 0.743
    predicted at r=0.75, q=0.90).  A detector at exactly the G2 bars
    shows ~74% alternation, so the original 85% bar could never be
    passed by a detector that passes G2.  G1 therefore scores
    CONSISTENCY: observed alternation within +-0.08 of alt(r_hat, q_hat),
    with an absolute junk-kill at < 0.45 — structured junk repeats on one
    side and lands BELOW the 0.5 random floor (the old ball streams sat
    at 9-37%).  The decisive detection gate is G2.

    python vision/swing_score.py                      # default file paths
    python vision/swing_score.py --selftest           # no files needed
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/vision"

# The audit tool PREFILLS shots 1-2 (serve, return) for any rally the
# labeler merely opens - both are log-known, so they arrive with a real
# uuid and a real type and look fully coded. Rallies 17/18/43/49 carry
# exactly those two rows from stray clicks and were never watched; scored
# as genuine 2-shot rallies they would charge the detector a false
# positive for every real shot it correctly found. A length rule cannot
# separate them: rallies 13 and 15 really are two shots (return went
# out / missed return). So the coded set is explicit - game 1, rallies
# 1-16, the frozen 203-shot ground truth.
CODED_RALLIES = frozenset(range(1, 17))

FAST_TYPES = {"speed-up", "counter", "smash"}
# counter vs. smash is the labeler's least confident split (both are fast,
# hard-contact shots to the eye); dink/fast/lob is a firm split. Keeping
# both in FAST_TYPES makes that ambiguity harmless to the gate - a shot
# mislabeled counter<->smash still lands in the same stratum either way.
BARS = {"recall_overall": 0.75, "recall_fast": 0.60, "precision": 0.90,
        "alt_tolerance": 0.08, "alt_junk_kill": 0.45,
        "kill_recall_overall": 0.60}
GRID_V = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28]
GRID_Z = [2.0, 3.0, 4.0, 6.0, 8.0]
RATE_BAND = (0.30, 0.85)          # plausible contacts per second of rally


def alt_expected(r, q):
    return q * q / (2 - min(r, 1.0)) + (1 - q * q) / 2


# ---------------------------------------------------------------- loads


def load_windows(path):
    out = {}
    for r in csv.DictReader(open(path)):
        a, b = r["start_score"].split("-")[:2]
        out[int(r["rally_cum"])] = {
            "cum": int(r["rally_cum"]), "game": int(r["game"]),
            "t0": float(r["t0s"]), "t1": float(r["t1s"]),
            "dur": float(r["dur_s"]),
            "conf": r.get("approx", "0") == "0",
            "half": 1 if max(int(a), int(b)) >= 6 else 0,
            "server": r["server_uuid"],
            "teamA": set(r["teamA_uuids"].split("|")),
            "teamB": set(r["teamB_uuids"].split("|")),
        }
    return out


def load_swings(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(int(r["rally_cum"]), []).append(
            (float(r["t_video"]), r["slot"][:4].replace("L", "").replace("R", ""),
             float(r["v_boxh_per_frame"])))
    for v in out.values():
        v.sort()
    return out


def load_pops(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(int(r["rally_cum"]), []).append(
            (float(r["t_video"]), float(r["z"])))
    for v in out.values():
        v.sort()
    return out


def load_labels(path, windows):
    """rally_cum -> list of (team, type); only fully-coded rallies count."""
    raw, notes = {}, {}
    for r in csv.DictReader(open(path)):
        cum = int(r["rally_cum"])
        raw.setdefault(cum, []).append(
            (int(r["shot_index"]), r["hitter_uuid"], r["shot_type"]))
        if r.get("rally_note", "").strip():
            notes[cum] = True

    # Tripwire on the hand-maintained CODED_RALLIES list. A coded rally
    # always shows one of two marks: a shot type beyond the prefilled
    # serve/return, or a note. Rallies 13 and 15 are genuinely two shots
    # (return went out / return missed) and prove length alone cannot
    # decide, so the list stays authoritative — this only shouts when the
    # file and the list disagree, which is how a future stray click gets
    # caught instead of quietly becoming ground truth.
    for cum, rows in sorted(raw.items()):
        rich = notes.get(cum, False) or any(
            t not in ("serve", "return", "") for _, _, t in rows)
        if cum in CODED_RALLIES and not rich:
            print(f"  ! rally {cum} is in CODED_RALLIES but looks like an "
                  f"untouched prefill stub — verify before trusting it")
        if cum not in CODED_RALLIES and rich:
            print(f"  ! rally {cum} looks coded but is excluded from "
                  f"CODED_RALLIES — was the label set extended?")

    out, skipped = {}, 0
    for cum, rows in raw.items():
        if cum not in windows or cum not in CODED_RALLIES:
            continue
        rows.sort()
        w = windows[cum]
        seq = []
        ok = True
        for _, uuid, typ in rows:
            if not uuid or not typ:
                ok = False
                break
            team = "A" if uuid in w["teamA"] else ("B" if uuid in w["teamB"] else "?")
            if team == "?":
                ok = False
                break
            seq.append((team, typ))
        if ok and len(seq) >= 2:
            out[cum] = seq
        else:
            skipped += 1
    return out, skipped


# ------------------------------------------------------------- contacts


def gate_contacts(swings, pops, tv, tz, coinc=0.18, merge=0.12):
    """contacts per rally: swing peaks above tv with a pop above tz within
    +-coinc, merged within `merge`.

    Two treatments the Chicago smoke data forced (2026-08-12, label-free):
    - NO velocity ceiling.  v > 0.6 box-h/frame is physically impossible
      for a wrist, but cutting those events COLLAPSED alternation (71-77%
      -> 20-45%) — the recall-removal signature.  They are real hard
      swings whose wrist keypoint exploded from motion blur at contact:
      garbage magnitude, real timing.  Keep them.
    - Same-instant opposite-side pairs (a slot-swap artifact fires both
      sides within ~0.06 s) are ONE contact of unknown side: merged to
      side '?', which counts toward rate but never toward alternation or
      team alignment."""
    out = {}
    for cum, sw in swings.items():
        pp = [t for t, z in pops.get(cum, []) if z >= tz]
        cand = []
        for t, side, v in sw:
            if v < tv:
                continue
            if pp and min(abs(t - p) for p in pp) <= coinc:
                cand.append((t, side, v))
        cand.sort()
        merged = []
        for c in cand:
            if merged and c[0] - merged[-1][0] < merge:
                p = merged[-1]
                side = p[1] if p[1] == c[1] else "?"
                merged[-1] = (p[0], side, max(p[2], c[2]))
            else:
                merged.append(c)
        out[cum] = merged
    return out


def alternation(contacts):
    """Side-alternation over contacts of KNOWN side ('?' merges excluded —
    an unknown side can neither confirm nor break parity)."""
    pairs = ok = 0
    for seq in contacts.values():
        known = [c for c in seq if c[1] != "?"]
        for a, b in zip(known, known[1:]):
            pairs += 1
            ok += a[1] != b[1]
    return ok / pairs if pairs else 0.0, pairs


def contact_rate(contacts, windows):
    n = sum(len(v) for v in contacts.values())
    dur = sum(windows[c]["dur"] for c in contacts if c in windows)
    return n / dur if dur else 0.0


def pick_operating_point(swings, pops, windows, coincs=(0.15, 0.20, 0.25)):
    """Label-free grid over (theta_v, theta_z, coinc).  The smoke offset
    histogram showed swing peaks centred on pops (median -1 ms, IQR
    +-0.14 s) with tails to +-0.3 s, so the coincidence window is swept
    too rather than fixed."""
    rows, best = [], None
    for tv in GRID_V:
        for tz in GRID_Z:
            for co in coincs:
                c = gate_contacts(swings, pops, tv, tz, co)
                alt, pairs = alternation(c)
                rate = contact_rate(c, windows)
                feas = RATE_BAND[0] <= rate <= RATE_BAND[1] and pairs >= 50
                rows.append((tv, tz, co, alt, rate, pairs, feas))
                if feas and (best is None or alt > best[3]):
                    best = rows[-1]
    if best is None:                        # nothing plausible: report best,
        best = max(rows, key=lambda r:      # discounting thin pairs AND
                   r[3]                     # off-band rates (both sides)
                   * min(1.0, r[4] / RATE_BAND[0],
                         RATE_BAND[1] / max(r[4], 1e-9))
                   * min(1.0, r[5] / 50.0))
    return best, rows


# ------------------------------------------------------------- mapping


def side_team_maps(contacts, windows):
    """near/far -> A/B per (game, half), voted by the serve anchor: the
    first contact within [t0-0.5, t0+2.5] sits on the serving team's side.
    Log-determined — no labels involved."""
    votes = {}
    for cum, seq in contacts.items():
        w = windows.get(cum)
        known = [c for c in (seq or []) if c[1] != "?"]
        if not w or not known:
            continue
        t, side, _ = known[0]
        if not (w["t0"] - 0.5 <= t <= w["t0"] + 2.5):
            continue
        serving = "A" if w["server"] in w["teamA"] else "B"
        key = (w["game"], w["half"])
        votes.setdefault(key, []).append((side, serving))
    maps = {}
    for key, vs in votes.items():
        agree = sum(1 for s, tm in vs if maps_guess(vs)[s] == tm)
        maps[key] = (maps_guess(vs), agree, len(vs))
    return maps


def maps_guess(vs):
    near_a = sum(1 for s, tm in vs if (s == "near") == (tm == "A"))
    if near_a * 2 >= len(vs):
        return {"near": "A", "far": "B"}
    return {"near": "B", "far": "A"}


def teamify(contacts, windows, maps):
    out = {}
    for cum, seq in contacts.items():
        w = windows.get(cum)
        if not w:
            continue
        m = maps.get((w["game"], w["half"]))
        if not m:                            # fall back to the game's other half
            alt_key = (w["game"], 1 - w["half"])
            m = maps.get(alt_key)
        if not m:
            continue
        out[cum] = [(t, m[0][side], v) for t, side, v in seq if side != "?"]
    return out


# ------------------------------------------------------------ alignment


def align(labeled, detected):
    """Needleman-Wunsch over team sequences. Returns list of matched
    (label_idx, det_idx) where teams agree."""
    n, m = len(labeled), len(detected)
    S = np.zeros((n + 1, m + 1))
    gap = -1.5
    for i in range(1, n + 1):
        S[i][0] = i * gap
    for j in range(1, m + 1):
        S[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sc = 2.0 if labeled[i - 1][0] == detected[j - 1][1] else -2.0
            S[i][j] = max(S[i - 1][j - 1] + sc, S[i - 1][j] + gap,
                          S[i][j - 1] + gap)
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        sc = 2.0 if labeled[i - 1][0] == detected[j - 1][1] else -2.0
        if abs(S[i][j] - (S[i - 1][j - 1] + sc)) < 1e-9:
            if sc > 0:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif abs(S[i][j] - (S[i - 1][j] + gap)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def score_vs_labels(labels, team_contacts):
    per_type = {}
    matched_det = det_total = 0
    for cum, lseq in labels.items():
        dseq = team_contacts.get(cum, [])
        pairs = align(lseq, dseq)
        matched_l = {i for i, _ in pairs}
        for i, (team, typ) in enumerate(lseq):
            d = per_type.setdefault(typ, [0, 0])
            d[1] += 1
            d[0] += i in matched_l
        matched_det += len(pairs)
        det_total += len(dseq)
    n_ok = sum(v[0] for v in per_type.values())
    n_all = sum(v[1] for v in per_type.values())
    fast_ok = sum(v[0] for t, v in per_type.items() if t in FAST_TYPES)
    fast_all = sum(v[1] for t, v in per_type.items() if t in FAST_TYPES)
    return {
        "per_type": {t: (v[0], v[1]) for t, v in sorted(per_type.items())},
        "recall_overall": n_ok / n_all if n_all else 0.0,
        "recall_fast": fast_ok / fast_all if fast_all else float("nan"),
        "n_fast": fast_all,
        "precision": matched_det / det_total if det_total else 0.0,
        "n_labeled_shots": n_all, "n_detected": det_total,
    }


# -------------------------------------------------------------- report


def verdict(alt, s):
    r, q = s["recall_overall"], s["precision"]
    exp = alt_expected(r, q)
    g1_junk = alt < BARS["alt_junk_kill"]
    g1_consistent = abs(alt - exp) <= BARS["alt_tolerance"]
    fast_ok = (not np.isnan(s["recall_fast"])
               and s["recall_fast"] >= BARS["recall_fast"])
    pass_full = (s["recall_overall"] >= BARS["recall_overall"] and fast_ok
                 and q >= BARS["precision"] and not g1_junk)
    kill = g1_junk or s["recall_overall"] < BARS["kill_recall_overall"]
    if pass_full:
        return "PASS — build the swing pipeline (vision MVP v2)"
    if kill:
        return "KILL — below the pre-registered floor; the thread closes"
    if s["recall_overall"] >= BARS["recall_overall"] and q >= BARS["precision"]:
        return ("MIDDLE — overall detection works, fast stratum short: "
                "touch-share-only scope (or label more hands-battle rallies)")
    return "MIDDLE — between floor and bars; read the table and decide"


def run(a):
    windows = load_windows(a.windows)
    swings = load_swings(a.swings)
    pops = load_pops(a.pops)

    n_conf = sum(1 for w in windows.values() if w["conf"])
    print(f"windows: {len(windows)} total, {n_conf} confident — "
          f"approx-flagged rallies are EXCLUDED from all scoring")
    conf_cums = {c for c, w in windows.items() if w["conf"]}
    swings = {c: s for c, s in swings.items() if c in conf_cums}
    pops = {c: s for c, s in pops.items() if c in conf_cums}
    print("== STAGE 1: label-free operating point ==")
    best, rows = pick_operating_point(swings, pops, windows)
    print("   theta_v theta_z coinc   alternation  contacts/s  pairs feasible")
    for tv, tz, co, alt, rate, pairs, feas in rows:
        if not feas and (tv, tz, co) != best[:3]:
            continue                        # keep the print readable
        mark = " <== chosen" if (tv, tz, co) == best[:3] else ""
        print(f"   {tv:7.2f} {tz:7.1f} {co:5.2f}   {alt:11.3f}  {rate:10.2f}  "
              f"{pairs:5d} {'yes' if feas else ' no'}{mark}")
    tv, tz, co, alt, rate, pairs, _ = best
    print(f"\n   frozen: theta_v={tv}, theta_z={tz}, coinc={co}  "
          f"(alternation {alt:.1%}, {rate:.2f} contacts/s)")

    contacts = gate_contacts(swings, pops, tv, tz, co)
    n_amb = sum(1 for s in contacts.values() for c in s if c[1] == "?")
    if n_amb:
        print(f"   ({n_amb} side-ambiguous merged contacts — count toward "
              f"rate, excluded from alternation and alignment)")
    maps = side_team_maps(contacts, windows)
    print("\n   serve-anchored side->team maps (game, half): "
          + (", ".join(f"G{g}H{h}: near={m['near']} ({agr}/{tot})"
                       for (g, h), (m, agr, tot) in sorted(maps.items()))
             or "none (no gated serve-anchor contacts)"))
    team_contacts = teamify(contacts, windows, maps)

    out = {"operating_point": {"theta_v": tv, "theta_z": tz, "coinc_s": co},
           "alternation": alt, "contact_rate": rate}

    if not Path(a.labels).exists():
        print(f"\nno labels at {a.labels} — stage 2 skipped. "
              "Label rallies, then rerun.")
        Path(a.out).write_text(json.dumps(out, indent=1))
        return

    print("\n== STAGE 2: frozen point vs hand labels (touched once) ==")
    labels, skipped = load_labels(a.labels, windows)
    s = score_vs_labels(labels, team_contacts)
    exp = alt_expected(s["recall_overall"], s["precision"])
    print(f"   labeled rallies used: {len(labels)} "
          f"(+{skipped} skipped as incomplete)")
    print(f"   per-type recall:")
    for t, (ok, n) in s["per_type"].items():
        star = "  <- fast stratum" if t in FAST_TYPES else ""
        print(f"     {t:<9} {ok:>3}/{n:<3} = {ok / n if n else 0:6.1%}{star}")
    print(f"   overall recall    {s['recall_overall']:6.1%}  (bar {BARS['recall_overall']:.0%})")
    print(f"   fast recall       {s['recall_fast']:6.1%}  on n={s['n_fast']}"
          f"  (bar {BARS['recall_fast']:.0%})")
    print(f"   precision         {s['precision']:6.1%}  (bar {BARS['precision']:.0%})")
    print(f"   G1 alternation    {alt:6.1%}  vs {exp:.1%} implied by "
          f"(r,q) — consistent within ±{BARS['alt_tolerance']:.0%}: "
          f"{'yes' if abs(alt - exp) <= BARS['alt_tolerance'] else 'NO'}")
    v = verdict(alt, s)
    print(f"\n   VERDICT: {v}")
    out.update(s, alt_expected=exp, verdict=v)
    out["per_type"] = {t: list(v2) for t, v2 in s["per_type"].items()}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"   report -> {a.out}")


# ------------------------------------------------------------ selftest


def selftest():
    rng = np.random.default_rng(11)

    # --- alignment math on a known case --------------------------------
    lab = [("A", "serve"), ("B", "return"), ("A", "drop"), ("B", "dink"),
           ("A", "dink"), ("B", "speed-up"), ("A", "counter"), ("B", "dink")]
    det = [(0.0, "A", 1), (1.2, "B", 1), (2.4, "B", 1),      # miss idx2, keep 3
           (3.0, "A", 1), (4.1, "B", 1), (5.0, "A", 1),      # 4,5,6 ok
           (6.2, "A", 1)]                                    # false extra A
    pairs = align(lab, det)
    matched_lab = {i for i, _ in pairs}
    assert {0, 1, 3, 4, 5, 6} <= matched_lab and 2 not in matched_lab, pairs
    s = score_vs_labels({1: lab}, {1: det})
    assert abs(s["recall_overall"] - 7 / 8) < 1e-9 or \
           abs(s["recall_overall"] - 6 / 8) < 0.13   # NW may trade the tail
    print(f"alignment: recall {s['recall_overall']:.2f}, "
          f"precision {s['precision']:.2f} on the constructed case")

    # --- alternation-vs-(r,q) consistency curve ------------------------
    windows = {}
    contacts = {}
    r_true, q_true = 0.75, 0.90
    for cum in range(1, 41):
        n = rng.integers(6, 16)
        windows[cum] = {"cum": cum, "game": 1, "t0": 0.0, "t1": n * 0.8,
                        "dur": n * 0.8, "half": 0, "server": "u1",
                        "teamA": {"u1", "u2"}, "teamB": {"u3", "u4"}}
        seq = []
        t = 0.0
        for k in range(n):
            t += 0.8
            side = "near" if k % 2 == 0 else "far"
            if rng.random() < r_true:
                seq.append((t, side, 0.3))
            if rng.random() < (1 - q_true) * r_true:   # false, random side
                seq.append((t + 0.3, rng.choice(["near", "far"]), 0.2))
        contacts[cum] = sorted(seq)
    alt, pairs_n = alternation(contacts)
    exp = alt_expected(r_true, q_true)
    print(f"consistency: observed alternation {alt:.3f} vs alt(r,q) {exp:.3f} "
          f"on {pairs_n} pairs")
    assert abs(alt - exp) < 0.06, "consistency curve is off"

    # --- serve-anchor mapping voting ------------------------------------
    maps = side_team_maps(contacts, windows)
    m, agr, tot = maps[(1, 0)]
    assert m["near"] == "A" and agr / tot > 0.8, maps
    print(f"mapping: near=A recovered with {agr}/{tot} serve votes")
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=str(D / "rally_windows_chicago0725.csv"))
    ap.add_argument("--swings", default=str(D / "swing_probe_swings.csv"))
    ap.add_argument("--pops", default=str(D / "swing_probe_pops.csv"))
    ap.add_argument("--labels", default=str(D / "shot_labels_chicago0725.csv"))
    ap.add_argument("--out", default=str(D / "swing_gate_report.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)


if __name__ == "__main__":
    main()
