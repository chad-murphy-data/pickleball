"""Serve vs return rally-win rates per player, and who beats the field.

Reads the committed rally-log summary (data/player_serve_rallies.csv, which
carries return columns once the collector has re-summarized with the
return-aware harvest_logs.py) and, for every player above a points
threshold, computes:

    serve win%    = serve_wins / serve_rallies        (individual)
    return win%   = return_wins / return_rallies       (team-attributed
                    in doubles: both partners share the side's return
                    rallies — a valid per-player RATE, never a sum)
    total win%    = (serve_wins + return_wins) / (all rallies)

It then fits the population regression return% ~ serve% (weighted by rally
count) and ranks players by residual: a positive residual means the player
(and their partners) win MORE return rallies than their serving strength
would predict. Field baselines come from the pooled aggregate.

Outputs data/serve_return.csv and prints the leaderboards. Pure stdlib.

    python model/serve_return_report.py [--min-points 300] [--discipline doubles]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_names():
    names, gender = {}, {}
    with open(DATA / "players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            names[u] = r["full_name"]
            gender[u] = r.get("gender", "")
    return names, gender


def wls(xs, ys, ws):
    """Weighted least squares slope/intercept for ys ~ xs."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-points", type=int, default=300,
                    help="minimum serve wins (points scored) to include")
    ap.add_argument("--discipline", default="doubles",
                    choices=["doubles", "singles", "all"])
    args = ap.parse_args()

    src = DATA / "player_serve_rallies.csv"
    with open(src) as fh:
        rows = list(csv.DictReader(fh))
    if "return_rallies" not in (rows[0] if rows else {}):
        raise SystemExit(
            f"{src} has no return columns yet — the collector must re-summarize "
            "with the return-aware harvest_logs.py first (see PR: persist return "
            "rallies). Until then only serve stats are available.")

    names, gender = load_names()
    pl = defaultdict(lambda: [0, 0, 0, 0])   # uuid -> sr, sw, rr, rw
    for r in rows:
        if args.discipline != "all" and r["discipline"] != args.discipline:
            continue
        a = pl[r["player_uuid"]]
        a[0] += int(r["serve_rallies"]); a[1] += int(r["serve_wins"])
        a[2] += int(r["return_rallies"]); a[3] += int(r["return_wins"])

    # field baselines (serve is individual; sum is honest. return is
    # team-shared, so pool the RATE numerator/denominator, not per-player)
    tot = [sum(v[i] for v in pl.values()) for i in range(4)]
    field_serve = tot[1] / tot[0]
    field_return = tot[3] / tot[2]

    recs = []
    for u, (sr, sw, rr, rw) in pl.items():
        if sw < args.min_points or sr < 100 or rr < 100:
            continue
        recs.append({
            "uuid": u, "name": names.get(u, u[:8]), "gender": gender.get(u, ""),
            "serve_rallies": sr, "serve_wins": sw, "serve_pct": sw / sr,
            "return_rallies": rr, "return_wins": rw, "return_pct": rw / rr,
            "total_pct": (sw + rw) / (sr + rr),
        })

    slope, intercept = wls([r["serve_pct"] for r in recs],
                           [r["return_pct"] for r in recs],
                           [r["serve_rallies"] + r["return_rallies"] for r in recs])
    for r in recs:
        r["return_resid"] = r["return_pct"] - (intercept + slope * r["serve_pct"])

    out = DATA / "serve_return.csv"
    cols = ["uuid", "name", "gender", "serve_rallies", "serve_wins", "serve_pct",
            "return_rallies", "return_wins", "return_pct", "total_pct",
            "return_resid"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(recs, key=lambda r: -r["total_pct"]):
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                        for k in cols})

    print(f"players (>{args.min_points} pts, {args.discipline}): {len(recs)}")
    print(f"field: serve {field_serve:.3f}  return {field_return:.3f}")
    print(f"regression  return% = {intercept:.3f} + {slope:+.3f}*serve%\n")

    def show(title, key, rev, n=12):
        print(title)
        for r in sorted(recs, key=lambda r: r[key], reverse=rev)[:n]:
            print(f"  {r['serve_pct']:.3f} serve | {r['return_pct']:.3f} return "
                  f"| {r['total_pct']:.3f} all | resid {r['return_resid']:+.3f} "
                  f"| {r['gender']} {r['name']}")
        print()

    show("Best overall rally win% (serve+return):", "total_pct", True)
    show("Best on serve:", "serve_pct", True)
    show("Most ABOVE the return regression line (return over-performers):",
         "return_resid", True)
    show("Most BELOW the return regression line:", "return_resid", False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
