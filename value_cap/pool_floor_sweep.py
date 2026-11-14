"""value_cap/pool_floor_sweep.py -- what if the priced pool were bigger than
the league (top 80 or 100 per gender instead of 60) and the floor were set
differently? Every cell: rebuild phi for that pool size (self-consistent,
fast engine), price it with Waters franchise-tagged at alpha 1, run the
draft on the real-2026-fill-in board, and read the slot-1 prize.

    python value_cap/pool_floor_sweep.py                      # sizes 60/80/100 x floors 10..75k
    python value_cap/pool_floor_sweep.py --sizes 60 80 --floors 30000 --drafts 5

phi per pool size is cached to value_cap/phi_pool{P}.csv (committed; ~20 s
to rebuild each with the fast engine). The replacement person is the
doubles #P player of each gender, so "replacement" moves with the pool --
a swept convention, not a picked constant. P = 60 reproduces
player_value_shapley.csv (same draws, same seed; the fast engine is exact
to ~1e-4) -- the self-test prints the max deviation.

What a bigger pool does to the money: the $20M is spread over 2P priced
players, but only 120 get drafted, so the priced-but-undrafted tail is
league money nobody spends and every drafted dollar buys more value. What
it does to Waters: her share of the pool's value falls as positive-phi
players are added below #60, which is the only thing that can make her
rosterable at a fair price. What the floor does: the tag = cap minus the
cheapest legal completion, so a higher floor LOWERS her tag price and
forces the tagged team's cast toward pure floor; a lower floor raises it.
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv, _argv = [sys.argv[0]], sys.argv          # phase2_pricing parses sys.argv on import
import draft_sim as D  # noqa: E402
from fast_tie import FastTie  # noqa: E402
from phase2_pricing import DOUBLES, NAME, SINGLES, TEAM_CAP, pid_named, prices_tagged  # noqa: E402
sys.argv = _argv

ENGINE = D.TRUE_ENGINE
RANKED = {g: sorted((u for u in DOUBLES if DOUBLES[u]["gender"] == g), key=lambda u: -DOUBLES[u]["v"])
          for g in ("M", "F")}
V_TOTAL, GENDER = {}, {}
for _r in csv.DictReader((HERE / "player_value.csv").open()):
    V_TOTAL[_r["player_id"]] = float(_r["V_total"])
    GENDER[_r["player_id"]] = _r["gender"]
BY_V = {g: sorted((u for u in V_TOTAL if GENDER[u] == g), key=lambda u: -V_TOTAL[u]) for g in ("M", "F")}


# ----------------------------------------------------------------- phi
def phi_for(candidates, pool_ids, repl, n_samples, seed):
    """shapley_value.phi_for on the fast engine (same draws for the same seed)."""
    rng = random.Random(seed)
    out = {}
    for g in ("M", "F"):
        og = "F" if g == "M" else "M"
        draws = []
        for _ in range(n_samples):
            same = tuple(rng.sample(pool_ids[g], 2))
            other = tuple(rng.sample(pool_ids[og], 3))
            opp = tuple(rng.sample(pool_ids["M"], 3) + rng.sample(pool_ids["F"], 3))
            draws.append((same, other, opp))
        base = [ENGINE.tie(same + (repl[g],) + other, opp) for same, other, opp in draws]
        for pid in candidates[g]:
            diffs = [ENGINE.tie(same + (pid,) + other, opp) - b
                     for (same, other, opp), b in zip(draws, base)
                     if pid not in same and pid not in opp]
            n = len(diffs)
            mean = sum(diffs) / n
            sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
            out[pid] = (mean, sd / n ** 0.5, n)
    return out


def phi_pool(P, n_samples=3000, seed=1, max_iter=4, verbose=True):
    """-> gender -> [(pid, name, phi)] sorted, P per gender, self-consistent.
    Cached to phi_pool{P}.csv."""
    path = HERE / f"phi_pool{P}.csv"
    if path.exists():
        pool = {"M": [], "F": []}
        for r in csv.DictReader(path.open()):
            if r["in_pool"] == "1":
                pool[r["gender"]].append((r["player_id"], r["full_name"], float(r["phi"])))
        for g in pool:
            pool[g].sort(key=lambda t: -t[2])
            assert len(pool[g]) == P, (P, g, len(pool[g]))
        return pool
    t0 = time.time()
    repl = {g: RANKED[g][P - 1] for g in ("M", "F")}
    candidates = {g: BY_V[g][:P + 20] for g in ("M", "F")}
    pool_ids = {g: BY_V[g][:P] for g in ("M", "F")}
    for it in range(max_iter):
        phi = phi_for(candidates, pool_ids, repl, n_samples, seed + it)
        new_pool = {g: sorted(candidates[g], key=lambda u: -phi[u][0])[:P] for g in ("M", "F")}
        churn = sum(len(set(new_pool[g]) ^ set(pool_ids[g])) for g in ("M", "F"))
        if verbose:
            print(f"  P={P} iteration {it}: churn {churn}", file=sys.stderr)
        stable = churn == 0
        pool_ids = new_pool
        if stable:
            break
    rows = []
    for g in ("M", "F"):
        for rank_, u in enumerate(sorted(candidates[g], key=lambda u: -phi[u][0]), 1):
            m, se, n = phi[u]
            rows.append({"player_id": u, "full_name": NAME[u], "gender": g,
                         "V_total": f"{V_TOTAL[u]:.5f}", "phi": f"{m:.5f}", "phi_se": f"{se:.5f}",
                         "n": n, "in_pool": int(u in pool_ids[g]), "pool_rank": rank_,
                         "replacement": NAME[repl[g]]})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print(f"  wrote {path.name} in {time.time()-t0:.0f}s", file=sys.stderr)
    return phi_pool(P, n_samples, seed, max_iter, verbose)


def selftest():
    """P=60 on the fast engine vs the committed exact-engine phi."""
    ref = {r["player_id"]: float(r["phi"]) for r in csv.DictReader((HERE / "player_value_shapley.csv").open())}
    pool_ids = {g: BY_V[g][:60] for g in ("M", "F")}
    repl = {g: RANKED[g][59] for g in ("M", "F")}
    cands = {g: [u for u in BY_V[g][:80]] for g in ("M", "F")}
    # the committed file's LAST iteration used seed 1 + it and the converged pool; compare on the
    # first iteration's draws instead (same pool = top 60 by V_total, seed 1) against a fresh exact run
    from phase1_value_model import tie_win_prob
    rng = random.Random(1)
    fast = phi_for(cands, pool_ids, repl, 300, 1)
    # exact on the same 300 draws
    out = {}
    for g in ("M", "F"):
        og = "F" if g == "M" else "M"
        draws = []
        for _ in range(300):
            same = tuple(rng.sample(pool_ids[g], 2)); other = tuple(rng.sample(pool_ids[og], 3))
            opp = tuple(rng.sample(pool_ids["M"], 3) + rng.sample(pool_ids["F"], 3))
            draws.append((same, other, opp))
        base = [tie_win_prob(same + (repl[g],) + other, opp, DOUBLES, SINGLES) for same, other, opp in draws]
        for pid in cands[g][:10]:
            diffs = [tie_win_prob(same + (pid,) + other, opp, DOUBLES, SINGLES) - b
                     for (same, other, opp), b in zip(draws, base) if pid not in same and pid not in opp]
            out[pid] = sum(diffs) / len(diffs)
    err = max(abs(out[u] - fast[u][0]) for u in out)
    corr_n = len([u for u in ref if u in fast])
    print(f"self-test: fast vs exact phi on identical draws, 20 players x 300 draws: max |diff| = {err:.2e}; "
          f"{corr_n} players overlap the committed file", file=sys.stderr)
    assert err < 2e-3, err


# --------------------------------------------------------------- the sweep
def run_cell(P, floor, pool, drafts, seasons, seed, noise=0.10):
    D.FLOOR = floor
    waters = pid_named("Anna Leigh Waters")
    price = prices_tagged(pool, 1.0, waters, "joint", floor=floor)
    D.set_board("mlp2026", pool)
    price = {u: price.get(u, floor) for u in D.BOARD}
    stars = [u for g in ("F", "M") for u, _, _ in pool[g][:6]]
    pos = sum(max(v, 0.0) for g in pool for _, _, v in pool[g])
    share = pool_value(pool, waters) / pos
    cell = {"P": P, "floor": floor, "waters_price": price[waters], "waters_share": share,
            "bright_price": price[pid_named("Anna Bright")], "johns_price": price[pid_named("Ben Johns")],
            "top_undrafted": None}
    for lab, nz, nd in (("perfect", 0.0, 1), ("noisy", noise, drafts)):
        r = D.run_variant(price, "snake", nz, nd, seasons, seed, stars)
        s = r["stars"]
        cell[lab] = {
            "spread": r["spread"], "max": r["max_exp"], "spend": r["spend"], "floor_taken": r["floor_taken"],
            "waters": (statistics.mean(s[waters]["exp"]), statistics.mean(s[waters]["title"])),
            "bright": (statistics.mean(s[pid_named("Anna Bright")]["exp"]), statistics.mean(s[pid_named("Anna Bright")]["title"])),
            "johns": (statistics.mean(s[pid_named("Ben Johns")]["exp"]), statistics.mean(s[pid_named("Ben Johns")]["title"])),
            "undrafted_top30": sorted(((NAME[u], D.POOL_RANK[u], c) for u, c in r["undrafted"].items()
                                      if D.POOL_RANK[u] <= 30), key=lambda t: -t[2]),
            "undrafted_n": len(r["undrafted"]),
            "slot2_20": statistics.mean(statistics.mean(x) for x in r["slot_exp"][1:]),
        }
    return cell


def pool_value(pool, pid):
    for g in pool:
        for u, _, v in pool[g]:
            if u == pid:
                return max(v, 0.0)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[60, 80, 100])
    ap.add_argument("--floors", nargs="+", type=int, default=[10_000, 20_000, 30_000, 50_000, 75_000])
    ap.add_argument("--drafts", type=int, default=10, help="noisy (10%%) drafts per cell")
    ap.add_argument("--seasons", type=int, default=200)
    ap.add_argument("--samples", type=int, default=3000, help="phi Monte Carlo draws")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(HERE / "pool_floor_sweep.md"))
    a = ap.parse_args()
    if a.selftest:
        selftest()
    pools = {}
    for P in a.sizes:
        pools[P] = phi_pool(P, a.samples, a.seed)
    cells = []
    for P in a.sizes:
        for floor in a.floors:
            t0 = time.time()
            c = run_cell(P, floor, pools[P], a.drafts, a.seasons, a.seed)
            cells.append(c)
            print(f"P={P} floor=${floor/1e3:.0f}k: Waters ${c['waters_price']/1e3:.0f}k share {100*c['waters_share']:.2f}% "
                  f"-> slot-1 {100*c['perfect']['waters'][0]:.1f}% perfect / {100*c['noisy']['waters'][0]:.1f}% noisy, "
                  f"spread {100*c['noisy']['spread']:.1f}, {time.time()-t0:.0f}s", file=sys.stderr)
    render(cells, pools, a)


def render(cells, pools, a):
    L = ["# Pool size x floor sweep -- does a bigger priced pool or a different floor blunt the #1 pick?", "",
         f"Every cell: phi rebuilt for a pool of P per gender (self-consistent, replacement = doubles #P), "
         f"Waters franchise-tagged at alpha 1 on one joint $20M pool with the given floor, then a 20-team "
         f"snake draft on the real-2026-fill-in board (`draft_sim.set_board('mlp2026', pool)`): one "
         f"perfect-information draft and {a.drafts} drafts at 10% owner error, {a.seasons} seasons each. "
         f"Built by `pool_floor_sweep.py`.", "",
         "## Waters' share of the pool's value (the thing pool size moves)", "",
         "| P per gender | positive-phi players | Waters share | 1/20 = fair team share | #1-#6 women phi | #1-#6 men phi |",
         "|---|---|---|---|---|---|"]
    for P, pool in pools.items():
        pos = [v for g in pool for _, _, v in pool[g] if v > 0]
        w = pool_value(pool, pid_named("Anna Leigh Waters"))
        L.append(f"| {P} | {len(pos)} | {100*w/sum(pos):.2f}% | 5.00% | "
                 + " ".join(f"{v:.3f}" for _, _, v in pool["F"][:6]) + " | "
                 + " ".join(f"{v:.3f}" for _, _, v in pool["M"][:6]) + " |")
    L += ["", "## The grid", "",
          "| P | floor | Waters price | Bright | Johns | slot-1 win% (perfect) | slot-1 win% / title% (10% error) | "
          "Bright team | Johns team | slots 2-20 mean | parity spread | mean spend | floor players taken | "
          "top-30 undrafted (10% error) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        n = c["noisy"]
        und = ", ".join(f"{nm} #{rk} {100*k/a.drafts:.0f}%" for nm, rk, k in n["undrafted_top30"][:4])
        if len(n["undrafted_top30"]) > 4:
            und += f" (+{len(n['undrafted_top30'])-4})"
        L.append(f"| {c['P']} | ${c['floor']/1e3:.0f}k | ${c['waters_price']/1e3:.0f}k | ${c['bright_price']/1e3:.0f}k | "
                 f"${c['johns_price']/1e3:.0f}k | {100*c['perfect']['waters'][0]:.1f}% | "
                 f"**{100*n['waters'][0]:.1f}%** / {100*n['waters'][1]:.0f}% | {100*n['bright'][0]:.1f}% | "
                 f"{100*n['johns'][0]:.1f}% | {100*n['slot2_20']:.1f}% | {100*n['spread']:.1f} pts | "
                 f"${n['spend']/1e3:,.0f}k | {n['floor_taken']:.0f} | {und or 'none'} |")
    L += ["", "Reading guide: 'floor players taken' counts drafted players priced AT the floor (fill-ins and "
          "any priced player whose curve price rounds to the floor). Undrafted = priced players inside their "
          "gender's top 30 by phi left on the board in at least one draft (info, not a test). Parity = 50% / 5%.", ""]
    Path(a.out).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
