"""When would Team 1 ever choose CROSS-GENDER DreamBreaker matchups?

Abstract game-theory solve on hypothetical players (NOT real rosters; the
cross-gender house rule doesn't apply because nothing here is a claim
about real athletes). Rule: Team 1 sets the four singles matchups, Team 2
sets the slot order, both with perfect information.

Every game outcome is the EXACT full-game win probability from the
db_scenarios backward DP: rally scoring to 21, win by 2, the freeze,
winner-serves-next, 4-point rotation cycling P1->P4->P1 until the game
ends. No expected-points shortcuts anywhere -- sequence and truncation
are fully modelled.

Stage 1 (the simplification): second men's and second women's matchups
are dead neutral (p = 0.5); only the four "top" players matter. Three
knobs, in logits on the singles-value scale (rally p = sigmoid(0.510*d)):

    g_M  = T2 best man   - T1 best man    (T1's same-gender men's deficit)
    g_W  = T1 best woman - T2 best woman  (T1's same-gender women's edge)
    D    = T1 best man   - T1 best woman  (the gender gulf)

T1's two candidate strategies (middles pinned neutral):
    same-gender:  {sig(-k*g_M), sig(k*g_W), .5, .5}
    cross (top-pair swap -- T1's best woman takes T2's best man, T1's
    best man takes T2's best woman):
                  {sig(-k*(D+g_M)), sig(k*(D+g_W)), .5, .5}
Team 2 answers each with its true optimal order (min over all 24 slot
permutations of the exact DP). Cross is rational iff its minimaxed win
prob beats same-gender's. We grid the (g_M, g_W) plane per D and bisect
the exact frontier.

Stage 2 (assumption relaxed): random 8-player configurations (every
woman below every man), full solve -- T1 max over all 24 matchings, T2
exact adversarial order -- to measure how often cross-gender is optimal
without the neutral-middles symmetry, plus an edge-sort-T2 robustness
pass on the flagged configs.

Run: python model/db_crossgender_conditions.py [--step 0.1] [--draws 1500]
Output: stdout tables (the .md report is written by hand from these).
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_scenarios import K_RALLY, sigmoid, win_prob  # noqa: E402


# ---------------------------------------------------------------------------
# game values (exact, full race to 21)
# ---------------------------------------------------------------------------

def adversarial_value(ps) -> float:
    """T1 win prob when T2 picks its true optimal slot order:
    min over all distinct permutations of the exact full-game DP."""
    return min(win_prob(perm) for perm in set(permutations(tuple(ps))))


def edge_sort_value(ps) -> float:
    """T1 win prob under T2's realistic heuristic: biggest T2 edge first
    (= T1 win prob ascending)."""
    return win_prob(tuple(sorted(ps)))


# ---------------------------------------------------------------------------
# stage 1: neutral middles, three knobs
# ---------------------------------------------------------------------------

def stage1_ps(g_m: float, g_w: float, d: float, k: float):
    sg = (sigmoid(-k * g_m), sigmoid(k * g_w), 0.5, 0.5)
    xg = (sigmoid(-k * (d + g_m)), sigmoid(k * (d + g_w)), 0.5, 0.5)
    return sg, xg


def stage1_gain(g_m: float, g_w: float, d: float, k: float) -> float:
    """Cross-gender minimax value minus same-gender minimax value."""
    sg, xg = stage1_ps(g_m, g_w, d, k)
    return adversarial_value(xg) - adversarial_value(sg)


def stage1_frontier(g_w: float, d: float, k: float,
                    lo: float = 0.0, hi: float = 6.0,
                    tol: float = 1e-3) -> float | None:
    """Critical g_M above which cross-gender becomes optimal (bisection;
    gain is monotone increasing in g_M). None if never inside [lo, hi]."""
    if stage1_gain(hi, g_w, d, k) <= 0:
        return None
    if stage1_gain(lo, g_w, d, k) > 0:
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if stage1_gain(mid, g_w, d, k) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def run_stage1(k: float, step: float, d_values):
    print("=" * 72)
    print("STAGE 1 -- neutral middles; exact minimax, full game to 21")
    print(f"k = {k}; grid step {step}; T2 = true optimal order (24 perms)")
    print("=" * 72)

    grid = [round(i * step, 6) for i in range(int(round(2.0 / step)) + 1)]
    for d in d_values:
        print(f"\n--- gender gulf D = {d} logits "
              f"(cross marquee p at g_M=0: "
              f"{sigmoid(-k * d):.3f} for T1's woman) ---")
        # heatmap: rows g_W (down = bigger women's edge), cols g_M
        print("cross-gender optimal region ('#'), by (g_W row, g_M col):")
        header = "  g_W\\g_M " + "".join(
            f"{g:4.1f}" if abs(g * 10 % 5) < 1e-9 else "    " for g in grid)
        print(header)
        for g_w in grid:
            cells = []
            for g_m in grid:
                gain = stage1_gain(g_m, g_w, d, k)
                cells.append("#" if gain > 0 else ".")
            print(f"  {g_w:7.2f} " + "   ".join(cells))
        print("\nexact frontier: critical g_M (bisection, tol 1e-3):")
        print("   g_W    g_M*     same-gender men's rally p at frontier")
        for g_w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            gm_star = stage1_frontier(g_w, d, k)
            if gm_star is None:
                print(f"  {g_w:5.2f}   never (within g_M <= 6)")
            else:
                print(f"  {g_w:5.2f}  {gm_star:6.3f}   "
                      f"{sigmoid(-k * gm_star):.3f}")
        # max attainable gain in the swept box
        best = max(((stage1_gain(g_m, g_w, d, k), g_m, g_w)
                    for g_m in grid for g_w in grid), key=lambda t: t[0])
        print(f"max gain in box: {best[0] * 100:+.2f} pp win prob "
              f"at g_M={best[1]}, g_W={best[2]}")


# ---------------------------------------------------------------------------
# stage 2: full 8-player random configurations
# ---------------------------------------------------------------------------

def solve_full(t1, t2, k: float, t2_policy=adversarial_value):
    """t1/t2 = [('M', v), ('M', v), ('F', v), ('F', v)].
    Returns (best value, best matching is cross-gender,
             best same-gender value, best cross-gender value or None)."""
    best_sg, best_xg = None, None
    for perm in permutations(range(4)):
        ps = tuple(sigmoid(k * (t1[i][1] - t2[perm[i]][1])) for i in range(4))
        cross = any(t1[i][0] != t2[perm[i]][0] for i in range(4))
        v = t2_policy(ps)
        if cross:
            best_xg = v if best_xg is None else max(best_xg, v)
        else:
            best_sg = v if best_sg is None else max(best_sg, v)
    return max(best_sg, best_xg), best_xg > best_sg, best_sg, best_xg


def run_stage2(k: float, draws: int, seed: int = 20260723):
    import random
    rng = random.Random(seed)
    print("\n" + "=" * 72)
    print(f"STAGE 2 -- {draws} random 8-player configs, full 24-matching "
          "solve, T2 = true optimal order")
    print("men ~ U[0, 2], women ~ U[min(men) - 2.5, min(men) - 0.05]")
    print("=" * 72)

    t0 = time.time()
    n_cross, flagged = 0, []
    for i in range(draws):
        men1 = sorted((rng.uniform(0, 2) for _ in range(2)), reverse=True)
        men2 = sorted((rng.uniform(0, 2) for _ in range(2)), reverse=True)
        top = min(men1[1], men2[1])
        wom1 = sorted((rng.uniform(top - 2.5, top - 0.05) for _ in range(2)),
                      reverse=True)
        wom2 = sorted((rng.uniform(top - 2.5, top - 0.05) for _ in range(2)),
                      reverse=True)
        t1 = [("M", men1[0]), ("M", men1[1]), ("F", wom1[0]), ("F", wom1[1])]
        t2 = [("M", men2[0]), ("M", men2[1]), ("F", wom2[0]), ("F", wom2[1])]
        _, is_cross, v_sg, v_xg = solve_full(t1, t2, k)
        if is_cross:
            n_cross += 1
            flagged.append((t1, t2, v_sg, v_xg))
        if i + 1 in (50, 200) or (i + 1) % 500 == 0:
            print(f"  {i + 1}/{draws} done ({time.time() - t0:.0f}s), "
                  f"cross-optimal so far: {n_cross}")

    print(f"\ncross-gender optimal: {n_cross}/{draws} "
          f"({100 * n_cross / draws:.1f}%)")

    if flagged:
        # what separates the flagged configs (stage-1 knobs recomputed)
        print("\nflagged configs -- stage-1 knobs (g_M, g_W, D) and gain:")
        print("   g_M     g_W      D    gain(pp)   sg->xg")
        shown = 0
        for t1, t2, v_sg, v_xg in sorted(
                flagged, key=lambda r: r[3] - r[2], reverse=True):
            g_m = t2[0][1] - t1[0][1]
            g_w = t1[2][1] - t2[2][1]
            d = t1[0][1] - t1[2][1]
            if shown < 15:
                print(f"  {g_m:5.2f}  {g_w:6.2f}  {d:5.2f}  "
                      f"{(v_xg - v_sg) * 100:8.2f}   "
                      f"{v_sg:.3f} -> {v_xg:.3f}")
            shown += 1
        gms = [t2[0][1] - t1[0][1] for t1, t2, _, _ in flagged]
        gws = [t1[2][1] - t2[2][1] for t1, t2, _, _ in flagged]
        ds = [t1[0][1] - t1[2][1] for t1, t2, _, _ in flagged]
        gains = [(v_xg - v_sg) * 100 for _, _, v_sg, v_xg in flagged]
        print(f"\nflagged medians: g_M={sorted(gms)[len(gms) // 2]:.2f}  "
              f"g_W={sorted(gws)[len(gws) // 2]:.2f}  "
              f"D={sorted(ds)[len(ds) // 2]:.2f}  "
              f"gain={sorted(gains)[len(gains) // 2]:.2f} pp "
              f"(max {max(gains):.2f} pp)")

        # robustness: does cross survive if T2 only edge-sorts?
        survive = 0
        for t1, t2, _, _ in flagged:
            _, is_cross_es, _, _ = solve_full(t1, t2, k,
                                              t2_policy=edge_sort_value)
            survive += is_cross_es
        print(f"robustness (T2 edge-sorts instead of true optimal): "
              f"cross still optimal in {survive}/{len(flagged)} flagged")


# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------

def sanity(k: float):
    assert abs(adversarial_value((0.5, 0.5, 0.5, 0.5)) - 0.5) < 1e-12
    # degenerate: huge gulf, no men's deficit -> cross must NOT help
    assert stage1_gain(0.0, 0.5, 3.0, k) < 0
    # monotone spot-check: more men's deficit -> cross relatively better
    g0 = stage1_gain(0.5, 0.5, 0.5, k)
    g1 = stage1_gain(1.5, 0.5, 0.5, k)
    assert g1 > g0
    # symmetry: flipping every p through 0.5 flips the ADVERSARIAL value
    # to the other side's MAXIMIN; check on a symmetric multiset instead
    ps = (0.4, 0.6, 0.5, 0.5)
    best_for_t1 = max(win_prob(p) for p in set(permutations(ps)))
    worst_flip = adversarial_value(tuple(1 - p for p in ps))
    assert abs(best_for_t1 - (1 - worst_flip)) < 1e-9
    print("sanity checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=K_RALLY)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--d-values", type=str, default="0.25,0.5,1.0,1.5,2.0")
    args = ap.parse_args()

    sanity(args.k)
    d_values = [float(x) for x in args.d_values.split(",")]
    t0 = time.time()
    run_stage1(args.k, args.step, d_values)
    print(f"\nstage 1 wall clock: {time.time() - t0:.0f}s")
    if args.draws > 0:
        run_stage2(args.k, args.draws)
    print(f"\ntotal wall clock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
