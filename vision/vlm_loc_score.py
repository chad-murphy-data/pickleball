"""Score the blind VLM LOCALIZATION test (2026-08-19).

CALLS were locked in-thread before the key was uploaded. NOT fully
blind: the realized contacts-per-window distribution ({0:7, 1:18, 2:5})
was printed while sizing the draw for power, so I knew the counts
before looking. Disclosed at scoring time; discount accordingly. The
calls diverge sharply from what was known (37 called vs 28 real, 14
doubles called vs 5 real), which argues against anchoring.

    python3 vision/vlm_loc_score.py data/vision/vlm_loc_key_20260819.csv
"""
import csv
import sys

CALLS = {
 1: [0.00, 1.05],  2: [0.60, 1.05],  3: [0.45],        4: [],
 5: [0.00, 1.05],  6: [0.00, 0.75],  7: [],            8: [0.45],
 9: [0.60],       10: [],           11: [0.00],       12: [0.00, 0.45],
13: [0.00, 0.75], 14: [0.15],       15: [0.45],       16: [0.30, 1.05],
17: [0.75],       18: [],           19: [],           20: [0.75],
21: [0.30, 1.20], 22: [0.60, 1.20], 23: [0.00, 0.90], 24: [0.15, 0.60],
25: [0.00, 0.90], 26: [0.75],       27: [0.45, 1.05], 28: [],
29: [],           30: [0.15, 1.05]}

TOL = 0.5     # same tolerance phase_grader uses to match decoded events


def match(calls, truth, tol):
    """Greedy nearest one-to-one; returns [(call, true)] pairs."""
    cand = sorted((abs(c - t), i, j)
                  for i, c in enumerate(calls) for j, t in enumerate(truth)
                  if abs(c - t) <= tol)
    ui, uj, out = set(), set(), []
    for _d, i, j in cand:
        if i not in ui and j not in uj:
            ui.add(i); uj.add(j); out.append((calls[i], truth[j]))
    return out


def main():
    key = {}
    for r in csv.DictReader(open(sys.argv[1])):
        q = int(r["window"].split(".")[0][1:])
        offs = [float(x) for x in r["offsets_s"].split("|") if x]
        key[q] = (offs, int(r["rally_cum"]),
                  [p for p in r["paces"].split("|") if p])

    exact = tp = fp = fn = 0
    empty_ok = empty_n = 0
    errs = []
    print("w    truth offsets       my calls            n  matched")
    for q in sorted(key):
        truth, rally, paces = key[q]
        calls = CALLS[q]
        m = match(calls, truth, TOL)
        exact += len(calls) == len(truth)
        tp += len(m); fp += len(calls) - len(m); fn += len(truth) - len(m)
        errs += [abs(c - t) for c, t in m]
        if not truth:
            empty_n += 1
            empty_ok += not calls
        ts = ",".join(f"{t:.2f}" for t in truth) or "-"
        cs = ",".join(f"{c:.2f}" for c in calls) or "-"
        flag = "OK " if len(calls) == len(truth) else "   "
        print(f"w{q:02d}  {ts:<16} {cs:<18} {flag} {len(m)}/{len(truth)}")

    n = len(key)
    print(f"\nEMPTY windows called empty      {empty_ok}/{empty_n}")
    nonempty_called = sum(1 for q in key if key[q][0] and CALLS[q])
    print(f"NON-EMPTY windows called non-empty "
          f"{nonempty_called}/{n - empty_n}")
    print(f"  => play / no-play decision      "
          f"{empty_ok + nonempty_called}/{n} = "
          f"{(empty_ok + nonempty_called)/n:.0%}")
    print(f"\nexact CONTACT COUNT right       {exact}/{n} = {exact/n:.0%}"
          f"   (registered 50-70%)")
    print(f"placement RECALL  (+/-{TOL}s)      {tp}/{tp+fn} = "
          f"{tp/(tp+fn):.0%}   (registered 45-65%)")
    print(f"placement PRECISION             {tp}/{tp+fp} = "
          f"{tp/(tp+fp):.0%}")
    if errs:
        errs.sort()
        print(f"median |timing error| on matches  "
              f"{errs[len(errs)//2]:.2f}s")
    for t in (0.3, 0.2):
        k = sum(len(match(CALLS[q], key[q][0], t)) for q in key)
        print(f"  recall at +/-{t}s: {k}/{tp+fn} = {k/(tp+fn):.0%}")

    print("\nby true window size:")
    for size in (1, 2):
        ws = [q for q in key if len(key[q][0]) == size]
        ok = sum(1 for q in ws if len(CALLS[q]) == size)
        r = sum(len(match(CALLS[q], key[q][0], TOL)) for q in ws)
        print(f"  {size}-contact windows: count right {ok}/{len(ws)}, "
              f"recall {r}/{size*len(ws)}")
    srv = [q for q in key if "position" in key[q][2]]
    r = sum(len(match(CALLS[q], key[q][0], TOL)) for q in srv)
    print(f"  serves/returns ('position'): recall {r}/"
          f"{sum(len(key[q][0]) for q in srv)} over {len(srv)} windows")


if __name__ == "__main__":
    main()
