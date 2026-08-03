"""Who makes rallies longer? A four-player SRM with DURATION as the outcome.

The question this exists to answer
----------------------------------
Do attacking and defending styles exist as a measurable player trait?

Two earlier attempts were both wrong, for reasons worth keeping:

1. "Ben Johns plays long points" — his rallies median 17s, but so does
   everyone's in his matches. Pure matchup: elite vs elite runs long.
2. Comparing a player's SERVE rallies to other players' serve rallies in the
   same match. This does not isolate the player at all — every rally in the
   match has all four players on court, so the split by server isolates the
   SERVING ROLE, not the person. And serving is largely uninteresting in
   pickleball; the serve is a put-in, not a weapon.

The right design is the same four-player structure used for skill, with rally
duration as the dependent variable:

    duration = mu + L[p1] + L[p2] + L[p3] + L[p4] + eps

Every rally loads all four players, so L is a player's effect on rally length
NET of teammates and opponents. No serve term — presence only.

Ridge selection
---------------
lambda is chosen by HELD-OUT prediction of rally duration, split by match —
NOT by the split-half statistic. Selecting the penalty on the reliability
metric and then reporting that metric is circular, and heavier shrinkage
inflates it almost mechanically. (Both routes happen to land in the same
place here, which is reassuring but was not knowable in advance.)

Result on ~38k rallies / 139 players
------------------------------------
    held-out MSE bottoms at ridge ~1500
    split-half r = +0.453  (Spearman-Brown full-length +0.624)
    sd(presence) 0.33s, intercept 18.3s
    variance explained vs intercept-only: 0.79%

**This is the first per-player trait in the clutch/duration work that
replicates.** Attacking vs defending styles are real and measurable.

Two caveats that travel with it:
  * 0.79% of variance. A real, replicable trait that is a rounding error
    against rally-to-rally noise. Both things are true at once.
  * At ridge 1500 the sd collapses from 1.44s to 0.33s. The ORDERING is what
    replicates; magnitudes are pulled hard toward zero. Read the top of the
    list as "longest-rally players", not as a literal 1.5s effect.

Run: python model/rally_presence.py        # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

import clutch_srm as cs                                       # noqa: E402

MIN_RALLIES = 300
LAMBDAS = (5., 15., 40., 100., 250., 600., 1500., 4000.)


def load():
    blob = cs.fetch()
    roster = {}
    for r in blob["roster"]:
        if r["p1"] and r["p2"]:
            roster.setdefault(r["match_id"], {})[r["side"]] = [r["p1"].lower(),
                                                               r["p2"].lower()]
    v2 = {r["player_id"].lower(): r["full_name"]
          for r in csv.DictReader((ROOT / "data" / "v2_players.csv").open())}
    rows, seen = [], set()
    src = ROOT / "data" / "rally_times.csv"
    for r in csv.DictReader(src.open()):
        k = (r["match_id"], r["game_number"], r["log_index"])
        if k in seen:
            continue
        seen.add(k)
        if not (r["dur_s"] and r["clean"] == "1"
                and 2 <= int(r["dur_s"]) <= 90):
            continue
        rr = roster.get(r["match_id"])
        if not rr or 0 not in rr or 1 not in rr:
            continue
        four = rr[0] + rr[1]
        if all(u in v2 for u in four):
            rows.append((four, int(r["dur_s"]), r["match_id"]))
    return rows, v2


def main():
    rows, v2 = load()
    cnt = defaultdict(int)
    for four, _, _ in rows:
        for u in four:
            cnt[u] += 1
    keep = {u for u, c in cnt.items() if c >= MIN_RALLIES}
    rows = [r for r in rows if all(u in keep for u in r[0])]
    idx = {u: i for i, u in enumerate(sorted(keep))}
    P, n = len(idx), len(rows)
    print(f"{n} rallies, {P} players with >= {MIN_RALLIES} on court, "
          f"{len({r[2] for r in rows})} matches\n")

    X = np.zeros((n, P + 1))
    y = np.array([d for _, d, _ in rows], dtype=float)
    for i, (four, d, _) in enumerate(rows):
        for u in four:
            X[i, idx[u]] += 1.0
        X[i, P] = 1.0
    R = np.eye(P + 1)
    R[P, P] = 0.0                       # never shrink the intercept
    mids = sorted({r[2] for r in rows})
    half = {m: i % 2 for i, m in enumerate(mids)}
    fold = np.array([half[r[2]] for r in rows])

    def fit(Xs, ys, lam):
        return np.linalg.solve(Xs.T @ Xs + lam * R, Xs.T @ ys)

    print("ridge by HELD-OUT prediction (not by the reliability statistic):")
    print(f"  {'ridge':>7}{'held-out MSE':>15}{'split-half r':>14}")
    best = None
    for lam in LAMBDAS:
        mse = []
        for f in (0, 1):
            b = fit(X[fold != f], y[fold != f], lam)
            mse.append(np.mean((X[fold == f] @ b - y[fold == f]) ** 2))
        m = float(np.mean(mse))
        b0 = fit(X[fold == 0], y[fold == 0], lam)
        b1 = fit(X[fold == 1], y[fold == 1], lam)
        r = float(np.corrcoef(b0[:P], b1[:P])[0, 1])
        print(f"  {lam:>7.0f}{m:>15.4f}{r:>+14.3f}")
        if best is None or m < best[0]:
            best = (m, lam, r)
    m, lam, r = best
    b = fit(X, y, lam)
    L = b[:P]
    inv = {i: u for u, i in idx.items()}
    base = float(np.mean((y - y.mean()) ** 2))
    print(f"\nbest ridge {lam:.0f}: split-half {r:+.3f} "
          f"(Spearman-Brown {2 * r / (1 + r):+.3f})")
    print(f"  sd(presence) {L.std():.3f}s  intercept {b[P]:.2f}s  "
          f"variance explained {100 * (1 - m / base):.2f}%")
    o = np.argsort(-L)
    print(f"\n  {'player':<26}{'presence':>10}{'rallies':>9}")
    for i in o[:10]:
        print(f"  {v2[inv[i]]:<26}{L[i]:>+10.2f}{cnt[inv[i]]:>9}")
    print("   ...")
    for i in o[-6:]:
        print(f"  {v2[inv[i]]:<26}{L[i]:>+10.2f}{cnt[inv[i]]:>9}")
    dest = ROOT / "data" / "rally_presence.csv"
    with dest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uuid", "name", "presence_s", "rallies", "rank"])
        for rank, i in enumerate(o, 1):
            w.writerow([inv[i], v2[inv[i]], round(float(L[i]), 4),
                        cnt[inv[i]], rank])
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
