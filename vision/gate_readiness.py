"""Temporal-gate readiness — labels-only progress report.

Answers "where am I in the labeling program?" from the exported CSV
alone: no pose, no video, no classifiers. Run it after every export
(Mac or repo). The verdict-side numbers (pre-check bars, closures)
stay in phase_grader.py; this script only tracks the RUN CONDITIONS
of vision/temporal_gate.md and the block coverage of
vision/labeling_protocol.md.

Holdout hygiene: reads holdout rows for PRESENCE and pace-pass
completeness only (the gate itself requires "all panel holdout
rallies labeled") — no times, no content, nothing a tuning script
could exploit.

Usage, from data/vision/ (defaults) or anywhere with flags:
    python3 ../../vision/gate_readiness.py
    python3 vision/gate_readiness.py --labels data/vision/contact_labels_chicago0725.csv \
        --split data/vision/label_split.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastslow_check import classify_type  # noqa: E402

LABELS = "contact_labels_chicago0725.csv"
SPLIT = "label_split.csv"
TRAIN_MIN = 90          # temporal_gate.md run condition
QUARANTINED = (9, 10)   # span anomaly — out of training until the
                        # debug-frame scorebug check clears them
NOT_IN_VOD = (11, 12)   # the tool's ⛔ case (labeling_protocol.md);
                        # override with --missing as more turn up
FULL_CHICAGO_CONTACTS = 3000


def resolve(path):
    p = Path(path)
    if p.exists():
        return p
    alt = Path(__file__).resolve().parent.parent / "data" / "vision" / path
    if alt.exists():
        return alt
    raise SystemExit(f"not found: {path} (also tried {alt})")


def load_split(path):
    """rally_cum -> (game, division, split)"""
    out = {}
    for r in csv.DictReader(open(resolve(path))):
        out[int(r["rally_cum"])] = (int(r["game"]), r["division"],
                                    r["split"])
    return out


def load_labels(path):
    """rally_cum -> {"n": contacts, "untyped": pace-pass debt}.
    Whiffs (contact=0) are skipped by the pace pass, so they never
    count as debt; serves/returns/lunges likewise (POSITION/NONSWING)."""
    out = defaultdict(lambda: {"n": 0, "untyped": 0})
    for r in csv.DictReader(open(resolve(path))):
        d = out[int(r["rally_cum"])]
        d["n"] += 1
        if r.get("contact", "1") == "0":
            continue
        if classify_type(r["shot_type"]) == "untyped":
            d["untyped"] += 1
    return dict(out)


def report(split, labels, missing, train_min=TRAIN_MIN):
    missing = set(missing)
    done = {c for c, d in labels.items() if d["untyped"] == 0}
    stamped = set(labels)
    lines = []

    def block(game, division, sp):
        cums = sorted(c for c, (g, dv, s) in split.items()
                      if g == game and s == sp)
        avail = [c for c in cums if c not in missing]
        lab = [c for c in avail if c in done]
        part = [c for c in avail if c in stamped and c not in done]
        return cums, avail, lab, part

    games = sorted({(g, dv) for g, dv, _ in split.values()})
    lines.append("block coverage (labeled+paced / in-VOD, [range])")
    tot = {"train": [0, 0], "holdout": [0, 0]}
    for g, dv in games:
        row = f"  game {g} ({dv:6s})"
        for sp in ("train", "holdout"):
            cums, avail, lab, part = block(g, dv, sp)
            tot[sp][0] += len(lab)
            tot[sp][1] += len(avail)
            extra = f" +{len(part)} partial" if part else ""
            row += (f"   {sp} {len(lab):3d}/{len(avail):3d} "
                    f"[{cums[0]}-{cums[-1]}]{extra}")
        lines.append(row)
    lines.append(f"  totals: train {tot['train'][0]}/{tot['train'][1]}"
                 f"   holdout {tot['holdout'][0]}/{tot['holdout'][1]}"
                 + (f"   (not-in-VOD excluded: "
                    f"{','.join(map(str, sorted(missing)))})"
                    if missing else ""))

    n_contacts = sum(d["n"] for d in labels.values())
    debt = {c: d["untyped"] for c, d in labels.items() if d["untyped"]}
    quarantine_live = [c for c in QUARANTINED if c in stamped]

    lines.append("")
    lines.append("run conditions (temporal_gate.md)")
    tr, tr_avail = tot["train"]
    ho, ho_avail = tot["holdout"]
    ok = "PASS" if tr >= train_min else "    "
    lines.append(f"  [{'x' if tr >= train_min else ' '}] >= {train_min} "
                 f"train rallies, both passes: {tr}"
                 f"  ({max(0, train_min - tr)} to go)")
    lines.append(f"  [{'x' if ho == ho_avail and ho_avail else ' '}] all "
                 f"holdout rallies labeled: {ho}/{ho_avail}"
                 f"  (pose-extraction not checked here)")
    lines.append(f"  [{'x' if not debt else ' '}] pace-pass debt: "
                 + ("none" if not debt else
                    ", ".join(f"r{c}({n})" for c, n in sorted(debt.items()))))
    lines.append(f"  [{'x' if not quarantine_live else '!'}] r9/r10 "
                 f"quarantine: "
                 + ("no labels present" if not quarantine_live else
                    f"labeled ({','.join('r%d' % c for c in quarantine_live)})"
                    " — OUT of training until the scorebug span check"))

    frontier = None
    order = sorted(c for c in split if c not in missing)
    for c in order:
        if c not in done:
            frontier = c
            break
    lines.append("")
    lines.append(f"archive: {n_contacts} contacts "
                 f"(~{100 * n_contacts // FULL_CHICAGO_CONTACTS}% of the "
                 f"~{FULL_CHICAGO_CONTACTS} full-Chicago estimate)")
    if frontier is not None:
        lines.append(f"next in video order: rally {frontier} "
                     f"(chained seek stays tight labeling in order)")
    else:
        lines.append("every in-VOD rally is labeled and paced.")
    return "\n".join(lines)


def selftest():
    split = {}
    for c in range(1, 11):
        split[c] = (1, "womens", "train" if c <= 6 else "holdout")
    labels = {
        1: {"n": 12, "untyped": 0},
        2: {"n": 8, "untyped": 0},
        3: {"n": 9, "untyped": 2},   # owes the pace pass
        7: {"n": 5, "untyped": 0},   # holdout, presence only
    }
    out = report(split, labels, missing={5}, train_min=4)
    assert "train   2/  5" in out.replace("  ", " ") or "2/5" in out.replace(" ", "").replace("[", "").split("train")[1][:8], out
    assert "r3(2)" in out, out
    assert "2 to go" in out, out
    assert "rally 3" in out, out          # frontier = first unpaced
    assert "not-in-VOD excluded: 5" in out, out
    # classify_type contract this script leans on
    assert classify_type("fast") == "fast"
    assert classify_type("dink") == "slow"
    assert classify_type("other") == "untyped"
    assert classify_type("lunge") == "nonswing"
    print("selftest OK")
    print()
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--missing", default=",".join(map(str, NOT_IN_VOD)),
                    help="comma-separated rally_cum not in the VOD (⛔)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    missing = {int(x) for x in a.missing.split(",") if x.strip()}
    split = load_split(a.split)
    labels = load_labels(a.labels)
    unknown = sorted(set(labels) - set(split))
    if unknown:
        print(f"!! labeled rallies not in the split (check them): "
              f"{unknown}\n")
    print(report(split, labels, missing))


if __name__ == "__main__":
    main()
