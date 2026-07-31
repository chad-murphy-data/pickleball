"""B3 part 4 — what, other than wind, distinguishes the top wind bin?

The 16-40 mph bin's skill slope drops in BOTH the outdoor and the indoor
arm (0.4087 vs 0.4085), so whatever causes it is not wind on the court.
This checks the obvious composition candidates:
  * tour mix (top bin is ~all PPA in both arms)
  * match length (median rallies per match-side is higher in the top bin —
    longer matches are closer matches, which mechanically flattens the
    fitted skill slope)
  * which indoor events supply the top bin
Each control re-fits the CALM (0-12 mph) reference restricted to look like
the top bin, and reports how much of the -0.10 gap it eats.

    python model/weather_review/b3_topbin_composition.py <scratch>
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rally_favorites_allmatches as P1  # noqa: E402
from b3_binned_trend import fit2, ci  # noqa: E402

R = 600
SEED = 20260731


def main():
    cells, _ = P1.build_cells()
    geo = {r["event_id"]: r for r in P1.read_csv(ROOT / "data/event_geo.csv")}
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# B3 part 4 — composition of the 16-40 mph bin\n")

    for c in cells:
        c["top"] = c["wind"] >= 16
        c["calm"] = c["wind"] < 12

    def rowsof(sub):
        return [(c["wins"], c["n"], c["adv"]) for c in sub]

    def boot_diff(sub_top, sub_calm, label):
        rt, rc = rowsof(sub_top), rowsof(sub_calm)
        if len(rt) < 6 or len(rc) < 6:
            say(f"{label:<52} too few cells")
            return
        ft, fc = fit2(rt), fit2(rc)
        bev = defaultdict(lambda: ([], []))
        for c in sub_top:
            bev[c["ev"]][0].append((c["wins"], c["n"], c["adv"]))
        for c in sub_calm:
            bev[c["ev"]][1].append((c["wins"], c["n"], c["adv"]))
        ks = sorted(bev)
        rng = random.Random(SEED)
        dd = []
        for _ in range(R):
            pk = [rng.choice(ks) for _ in ks]
            a1 = sum((bev[k][0] for k in pk), [])
            a2 = sum((bev[k][1] for k in pk), [])
            if len(a1) < 6 or len(a2) < 6:
                continue
            f1, f2 = fit2(a1, b0=ft, iters=6), fit2(a2, b0=fc, iters=6)
            if f1 and f2:
                dd.append(f1[1] - f2[1])
        lo, hi = ci(dd)
        say(f"{label:<52} top {ft[1]:+.4f} (n={sum(x[1] for x in rt)})  "
            f"calm {fc[1]:+.4f} (n={sum(x[1] for x in rc)})  "
            f"diff {ft[1]-fc[1]:+.4f} [{lo:+.4f},{hi:+.4f}]")

    say("## A. indoor events supplying the 16-40 mph bin\n")
    ev_n = defaultdict(int)
    for c in cells:
        if c["setting"] == "indoor" and c["top"]:
            ev_n[c["ev"]] += c["n"]
    tot = sum(ev_n.values())
    for ev, n in sorted(ev_n.items(), key=lambda kv: -kv[1]):
        g = geo.get(ev, {})
        say(f"  {n:>6} {n/tot:>6.1%}  {g.get('event_name','?')} | "
            f"{g.get('city','?')}, {g.get('state','?')}")
    say("")

    say("## B. controls: make the calm reference look like the top bin\n")
    for setting in ("outdoor", "indoor"):
        say(f"### {setting}")
        A = [c for c in cells if c["setting"] == setting]
        top = [c for c in A if c["top"]]
        boot_diff(top, [c for c in A if c["calm"]], "  raw")
        boot_diff([c for c in top if c["tour"] != "MLP"],
                  [c for c in A if c["calm"] and c["tour"] != "MLP"],
                  "  PPA only")
        # match-length control: keep only long match-sides in both
        boot_diff([c for c in top if c["n"] >= 40],
                  [c for c in A if c["calm"] and c["n"] >= 40],
                  "  PPA+long (n>=40 serve rallies)"
                  if False else "  long match-sides only (n>=40)")
        boot_diff([c for c in top if c["tour"] != "MLP" and c["n"] >= 40],
                  [c for c in A if c["calm"] and c["tour"] != "MLP"
                   and c["n"] >= 40],
                  "  PPA only AND n>=40")
        boot_diff([c for c in top if c["tour"] != "MLP" and c["n"] < 40],
                  [c for c in A if c["calm"] and c["tour"] != "MLP"
                   and c["n"] < 40],
                  "  PPA only AND n<40 (short match-sides)")
        say("")

    say("## C. is the top bin just late-round matches? (stage mix)\n")
    for setting in ("outdoor", "indoor"):
        A = [c for c in cells if c["setting"] == setting]
        for lab, sub in (("top", [c for c in A if c["top"]]),
                         ("calm", [c for c in A if c["calm"]])):
            n = sum(c["n"] for c in sub)
            ml = sum(c["n"] * c["n"] for c in sub) / n
            say(f"  {setting:>7} {lab:>4}: rallies={n:>7} "
                f"rally-wtd mean match-side length={ml:.1f} "
                f"cells={len(sub)}")
    say("")

    (Path(__file__).parent / "b3_topbin_composition.txt").write_text(
        "\n".join(out) + "\n")
    print("\nwrote model/weather_review/b3_topbin_composition.txt")


if __name__ == "__main__":
    main()
