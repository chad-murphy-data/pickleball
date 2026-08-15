"""Align the scorebug STATE stream to the referee log -> rally windows v3.

The endgame of the windows saga.  scorebug_read v6 emits per-second
row-change events with team attribution (utah_row / chi_row).  The
referee log predicts the exact GRAMMAR of those events, rally by rally,
via the side-out rules the user decoded:

    outcome 'point'   -> the SERVING team's row changes (score +1)
    outcome 'second'  -> the serving team's row changes (dots 1 -> 2)
    outcome 'sideout' -> BOTH rows change (dots switch sides)

So the expected label sequence (U / C / UC per rally, in order) is fully
determined, and alignment is a labeled monotone DP with the same physical
timing bounds as before (play is never trimmed; replays insert time).
Label mismatches are heavily penalized, which is what makes this
over-determined where gap-only alignment was ambiguous: a shifted chain
must also reproduce the wrong label sequence, and dies.

    python vision/scorebug_align.py --states scorebug_states.csv \\
        --windows data/vision/rally_windows_chicago0725.csv \\
        --timeline data/vision/rally_timeline_matchup_20260725_c4e686d1.csv \\
        --out rally_windows_chicago0725_v3.csv

Validation gates printed: the 778 s hand-verified anchor (rally #30),
per-game match rates, label-agreement rate, overlap count, and the
implied condensation vs the known 26.9 min.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CHICAGO_PLAYERS = {"Emma Nelson", "Ting Chieh Wei", "AJ Koller",
                   "John Lucian Goins"}   # Chicago Slice roster this matchup
PAIR_S = 2.0            # U and C changes within this = one UC event
EVENT_LAG_S = 1.0       # bug updates ~1 s after the rally's final ball
COST_LABEL_MISS = 6.0   # observed label != expected label
COST_LABEL_PART = 1.5   # UC observed where U/C expected (merged hidden change)
COST_SKIP_EVENT = 2.5   # junk event (game-start redraws, banners)
COST_SKIP_RALLY = 5.0   # rally end not observed (hidden behind replay)
REPLAY_MAX_S = 90.0
MAXJ = 4


def wall_s(iso):
    hh, mm, ss = iso.split("T")[1].split("+")[0].split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def load_rallies(timeline_csv, windows_csv):
    teams = {}
    for r in csv.DictReader(open(windows_csv)):
        chiA = any(n in CHICAGO_PLAYERS for n in r["teamA_names"].split("|"))
        teams[int(r["rally_cum"])] = ("A" if chiA else "B")
    have = set()
    for r in csv.DictReader(open(windows_csv)):
        have.add(int(r["rally_cum"]))
    out = []
    for r in csv.DictReader(open(timeline_csv)):
        cum = int(r["rally"])
        if cum not in have:
            continue                 # the 2 rallies v1 never timed
        out.append({
            "cum": cum, "match_id": r["match_id"],
            "wall": wall_s(r["t_end"]), "dur": float(r["duration_s"]),
            "outcome": r["outcome"],
            "server": r["server_uuid"].lower(),
        })
    out.sort(key=lambda x: x["cum"])
    # serving TEAM row (U or C) per rally, from the windows roster columns
    rosters = {}
    for r in csv.DictReader(open(windows_csv)):
        rosters[int(r["rally_cum"])] = (
            set(u.lower() for u in r["teamA_uuids"].split("|")),
            any(n in CHICAGO_PLAYERS for n in r["teamA_names"].split("|")))
    for x in out:
        inA, a_is_chi = rosters[x["cum"]]
        serving_chi = (x["server"] in inA) == a_is_chi
        srow = "C" if serving_chi else "U"
        x["label"] = "UC" if x["outcome"] == "sideout" else srow
    return out


def load_events(states_csv):
    raw = []
    for r in csv.DictReader(open(states_csv)):
        t = float(r["t_s"])
        if int(r["utah_row_chg"]):
            raw.append([t, "U"])
        if int(r["chi_row_chg"]):
            raw.append([t, "C"])
    raw.sort()
    ev, used = [], [False] * len(raw)
    for i, (t, lab) in enumerate(raw):
        if used[i]:
            continue
        merged = lab
        for j in range(i + 1, len(raw)):
            if raw[j][0] - t > PAIR_S:
                break
            if not used[j] and raw[j][1] != lab:
                merged = "UC"
                used[j] = True
                break
        used[i] = True
        ev.append((t, merged))
    return ev


def lab_cost(expected, observed):
    if expected == observed:
        return 0.0
    if observed == "UC":
        return COST_LABEL_PART        # a hidden change merged in
    if expected == "UC":
        return COST_LABEL_PART        # one row's change was missed
    return COST_LABEL_MISS


def align(events, rallies):
    m, n = len(events), len(rallies)
    INF = 1e18
    T = [e[0] for e in events]
    import bisect
    best = [dict() for _ in range(n)]
    par = {}
    for j in range(min(n, MAXJ + 1)):
        for i in range(m):
            c = j * COST_SKIP_RALLY + lab_cost(rallies[j]["label"], events[i][1])
            if c < best[j].get(i, INF):
                best[j][i] = c
                par[(i, j)] = None
    for j in range(n):
        for i, c in sorted(best[j].items()):
            for j2 in range(j + 1, min(n, j + 1 + MAXJ + 1)):
                mp = sum(rallies[k]["dur"] for k in range(j + 1, j2 + 1))
                dw = rallies[j2]["wall"] - rallies[j]["wall"]
                lo = bisect.bisect_left(T, T[i] + mp - 2.0, i + 1)
                hi = bisect.bisect_right(T, T[i] + dw + REPLAY_MAX_S, i + 1)
                for i2 in range(lo, hi):
                    dv = T[i2] - T[i]
                    slack = dw - dv
                    cost = (lab_cost(rallies[j2]["label"], events[i2][1])
                            + 0.01 * max(0.0, dv - mp)
                            + 0.15 * max(0.0, -slack)
                            + (i2 - i - 1) * COST_SKIP_EVENT
                            + (j2 - j - 1) * COST_SKIP_RALLY)
                    if c + cost < best[j2].get(i2, INF):
                        best[j2][i2] = c + cost
                        par[(i2, j2)] = (i, j)
    eb, es = INF, None
    for j in range(n):
        for i, c in best[j].items():
            tot = c + (n - 1 - j) * COST_SKIP_RALLY
            if tot < eb:
                eb, es = tot, (i, j)
    match = [None] * n
    s = es
    while s is not None:
        i, j = s
        match[j] = i
        s = par.get(s)
    return match, eb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--out", default="rally_windows_chicago0725_v3.csv")
    a = ap.parse_args()

    rallies = load_rallies(a.timeline, a.windows)
    events = load_events(a.states)
    print(f"{len(events)} labeled events for {len(rallies)} rallies "
          f"(labels expected: "
          f"U {sum(1 for r in rallies if r['label']=='U')}, "
          f"C {sum(1 for r in rallies if r['label']=='C')}, "
          f"UC {sum(1 for r in rallies if r['label']=='UC')}; observed: "
          f"U {sum(1 for _, l in events if l=='U')}, "
          f"C {sum(1 for _, l in events if l=='C')}, "
          f"UC {sum(1 for _, l in events if l=='UC')})")

    match, cost = align(events, rallies)
    n_ok = sum(1 for x in match if x is not None)
    agree = sum(1 for j, x in enumerate(match) if x is not None
                and events[x][1] == rallies[j]["label"])
    print(f"aligned {n_ok}/{len(rallies)} rallies (cost {cost:.0f}); "
          f"label agreement {agree}/{n_ok}")

    # gates
    for j, r in enumerate(rallies):
        if r["cum"] == 30 and match[j] is not None:
            t30 = events[match[j]][0]
            print(f"ANCHOR rally #30: event at {t30:.0f}s vs hand-verified "
                  f"~777-778s  ({'PASS' if abs(t30 - 777.5) < 3 else 'FAIL'})")
    games = {}
    for j, r in enumerate(rallies):
        g = games.setdefault(r["match_id"], [0, 0])
        g[0] += 1
        g[1] += match[j] is not None
    for gi, (mid, (tot, ok)) in enumerate(sorted(games.items(),
            key=lambda kv: min(r["cum"] for r in rallies
                               if r["match_id"] == kv[0]))):
        print(f"  game {gi+1}: {ok}/{tot} matched")

    # windows + interpolation for the missed
    t1v = [events[x][0] - EVENT_LAG_S if x is not None else None
           for x in match]
    for k in range(len(t1v)):
        if t1v[k] is None:
            lo = next((x for x in range(k - 1, -1, -1)
                       if t1v[x] is not None), None)
            hi = next((x for x in range(k + 1, len(t1v))
                       if t1v[x] is not None), None)
            if lo is not None and hi is not None:
                f = ((rallies[k]["wall"] - rallies[lo]["wall"])
                     / max(rallies[hi]["wall"] - rallies[lo]["wall"], 1e-9))
                t1v[k] = t1v[lo] + f * (t1v[hi] - t1v[lo])
            elif lo is not None:
                t1v[k] = t1v[lo] + (rallies[k]["wall"] - rallies[lo]["wall"])
            else:
                t1v[k] = t1v[hi] - (rallies[hi]["wall"] - rallies[k]["wall"])

    ov_set = set()
    for kk in range(1, len(rallies)):
        if rallies[kk]["match_id"] == rallies[kk-1]["match_id"] \
                and match[kk] is not None and match[kk-1] is not None \
                and (t1v[kk] - rallies[kk]["dur"]) - t1v[kk-1] < -1.0:
            ov_set.add(kk); ov_set.add(kk-1)
    print(f"overlapping consecutive matched windows: {len(ov_set)//2} "
          f"(participants flagged approx)")
    cut = (rallies[-1]["wall"] - rallies[0]["wall"]) - (t1v[-1] - t1v[0])
    print(f"implied condensation: {cut/60:.1f} min (known ~26.9 + breaks)")

    old_rows = list(csv.DictReader(open(a.windows)))
    old_map = {int(r["rally_cum"]): r for r in old_rows}
    shifts = []
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = list(old_rows[0].keys())
        w.writerow(hdr)
        for k, r in enumerate(rallies):
            row = dict(old_map[r["cum"]])
            shifts.append(t1v[k] - float(row["t1s"]))
            row["t1s"] = f"{t1v[k]:.1f}"
            row["t0s"] = f"{t1v[k] - r['dur']:.1f}"
            bad_ov = k in ov_set
            row["approx"] = "0" if (match[k] is not None
                                    and events[match[k]][1] == r["label"]
                                    and not bad_ov) else "1"
            w.writerow([row[h] for h in hdr])
    s = np.array(shifts)
    conf = sum(1 for k, r in enumerate(rallies)
               if match[k] is not None and events[match[k]][1] == r["label"])
    print(f"wrote {a.out}: {conf}/{len(rallies)} label-exact confident windows")
    print(f"shift vs v1 windows: median {np.median(s):+.1f}s "
          f"IQR [{np.percentile(s,25):+.1f}, {np.percentile(s,75):+.1f}]")


if __name__ == "__main__":
    main()
