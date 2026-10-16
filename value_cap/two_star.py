"""Why a man + a woman beats two men: game-by-game decomposition of the
two-star builds against the market-limit field.

The auction room's chasers (auction.md, "What this says") were two-star
rosters, almost always one man and one woman. This script shows why by
playing each two-star build (two stars + floor fill-ins to 3M+3W) against
the nineteen non-Waters rosters of a market-limit league (market_eq.py
cache) and printing the per-game win probabilities, the DreamBreaker
edge and the tie probability.

    python value_cap/two_star.py                    # default cache: equalize seed 1, c 8
    python value_cap/two_star.py --cache value_cap/cache/market_eq_equalize_seed3_c4_n0.json

Needs a market_eq cache (run `python value_cap/market_eq.py` first).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cache", default=os.path.join(HERE, "cache", "market_eq_equalize_seed1_c8_n0.json"))
ap.add_argument("--rep", type=int, default=0, help="which allocation of the cached league to use as the field")
args = ap.parse_args()

sys.argv = ["market_eq.py"]  # market_eq parses argv at import
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import market_eq as M  # noqa: E402

E = M.E
pid = M.pid_named
NAME = M.NAME
cache = json.load(open(args.cache))
pbar = dict(cache["prices"])
field = [tuple(pid(n) for n, _ in ro) for ro in cache["leagues"][args.rep]["rosters"]]
W = pid("Anna Leigh Waters")
others = [ro for ro in field if W not in ro]
print(f"field: {len(others)} non-Waters rosters from {os.path.basename(args.cache)} rep {args.rep}")

# The floor fill-ins the market-limit rosters actually use (the best $30k players by value / singles).
floorM = [pid(n) for n in ["Jonathan Truong", "Gabriel Joseph", "Martin Emmrich"]]
floorW = [pid(n) for n in ["Lina Padegimaite", "Genie Erokhina", "Alexa Schull"]]


def price(u):
    return pbar.get(NAME[u], M.FLOOR)


def show_vals(us):
    return ", ".join(f"{NAME[u]} v={E.v[u]:+.2f} s={E.s[u]:+.2f} ${price(u)/1e3:.0f}k" for u in us)


print("floor men:  ", show_vals(floorM))
print("floor women:", show_vals(floorW))


def build(stars):
    men = [u for u in stars if E.gender[u] == "M"]
    women = [u for u in stars if E.gender[u] == "F"]
    men += floorM[:3 - len(men)]
    women += floorW[:3 - len(women)]
    return tuple(men + women)


def decompose(ro, opps, label):
    wa, ma, xa1, xa2, dba = E.lineup(ro)
    rows = []
    for ob in opps:
        wb, mb, xb1, xb2, dbb = E.lineup(ob)
        p = [E.game(wa, wb, "WD"), E.game(ma, mb, "MD"), E.game(xa1, xb1, "MXD"), E.game(xa2, xb2, "MXD")]
        pdb = M.FT.db_win(round(M.FT.sigmoid(M.FT.K_DB * (dba - dbb)), 4))
        rows.append(p + [pdb, E.tie(ro, ob)])
    r = np.mean(rows, axis=0)

    def S(pr, div):
        return E.S(*pr, div)

    print(f"\n{label}: cost ${sum(price(u) for u in ro)/1e3:.0f}k at market")
    print(f"  WD {NAME[wa[0]]}+{NAME[wa[1]]} S={S(wa,'WD'):+.2f} | MD {NAME[ma[0]]}+{NAME[ma[1]]} S={S(ma,'MD'):+.2f}"
          f" | MXD1 {NAME[xa1[0]]}+{NAME[xa1[1]]} S={S(xa1,'MXD'):+.2f} | MXD2 {NAME[xa2[0]]}+{NAME[xa2[1]]} S={S(xa2,'MXD'):+.2f}"
          f" | DB mean singles {dba:+.2f}")
    print(f"  vs field (mean of {len(opps)}): WD {r[0]*100:.0f}%  MD {r[1]*100:.0f}%  MXD1 {r[2]*100:.0f}%  MXD2 {r[3]*100:.0f}%"
          f"  DB {r[4]*100:.0f}%  -> tie {r[5]*100:.1f}%   (expected games won {sum(r[:4]):.2f} of 4)")
    return r


fs = {"WD": [], "MD": [], "MXD1": [], "MXD2": [], "DB": []}
for ob in others:
    wb, mb, xb1, xb2, dbb = E.lineup(ob)
    fs["WD"].append(E.S(*wb, "WD"))
    fs["MD"].append(E.S(*mb, "MD"))
    fs["MXD1"].append(E.S(*xb1, "MXD"))
    fs["MXD2"].append(E.S(*xb2, "MXD"))
    fs["DB"].append(dbb)
print("field pair strengths (mean +- sd): " + "  ".join(f"{k} {np.mean(v):+.2f}+-{np.std(v):.2f}" for k, v in fs.items()))

STARS = {
    "M+W Patriquin+Rohrabacher": ["Hayden Patriquin", "Rachel Rohrabacher"],
    "M+M Patriquin+Tardio": ["Hayden Patriquin", "Gabriel Tardio"],
    "M+M Patriquin+JW Johnson": ["Hayden Patriquin", "JW Johnson"],
    "F+F Rohrabacher+Fahey": ["Rachel Rohrabacher", "Kate Fahey"],
    "F+F Rohrabacher+Todd": ["Rachel Rohrabacher", "Parris Todd"],
    "M+W Tardio+Fahey": ["Gabriel Tardio", "Kate Fahey"],
    "M+W Johns+Bright": ["Ben Johns", "Anna Bright"],
    "M+M Johns+JW Johnson": ["Ben Johns", "JW Johnson"],
    "F+F Bright+Todd": ["Anna Bright", "Parris Todd"],
}
for label, names in STARS.items():
    decompose(build([pid(n) for n in names]), others, label)

print("\n-- value spread by gender in the priced pool (per-point logit): top-1 / top-5 mean / #30 / #60 / floor fill-ins")
for g in "FM":
    pool = sorted([u for u in M.POOLSET if E.gender[u] == g], key=lambda u: -E.v[u])
    fl = floorW if g == "F" else floorM
    print(f"  {g}: {E.v[pool[0]]:+.2f} / {np.mean([E.v[u] for u in pool[:5]]):+.2f} / {E.v[pool[29]]:+.2f} / {E.v[pool[59]]:+.2f}"
          f" / floor {np.mean([E.v[u] for u in fl]):+.2f}")
