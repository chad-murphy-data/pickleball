"""Injection power test for the rare-wind-skill battery (wind_rare.py).

    python model/wind_rare_power.py   # prints + writes model/wind_rare_power.md

The battery came back null. Null results are only worth stating with a
detection floor: plant a clutch-shaped minority trait into the REAL panel —
a random PI_INJ of players get a personal wind slope of ±s (sign persistent
across eras, i.e. a real trait) — and ask how often the battery's two
sharpest tests fire:

  * spike-slab LR exceeding the null max observed in wind_rare.py
  * select-then-verify (K=40, best of both directions) clearing z > 2.5

Grid of s in point-share per mph. The result is the sentence "a minority
wind trait of ≥X share per 10 mph in ~13% of players would have been
detected with probability Y" — the honest size of the telescope.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from wind_rare import (ERA_SPLIT, MIN_GAMES, MIN_GAMES_ERA, N_REPS,  # noqa: E402
                       build_arrays, load_panel, slope_se, spike_slab, stats)

PI_INJ = 0.13                      # fraction of players given the trait
SIZES = (0.0, 0.001, 0.002, 0.003, 0.004)   # share per mph
N_SIM = 20
LR_BAR = 7.2                       # null max LR from wind_rare.py run
SEED = 7


def obs_arrays(players, pids, slopes):
    """Observed (b, se, n) per arm after injecting per-player slopes
    (slope * centered wind added to each residual)."""
    arms = ("full", "pre26", "y26")
    obs = {a: {"b": [], "se": [], "n": []} for a in arms}
    for pid in pids:
        gs = players[pid]
        s = slopes.get(pid, 0.0)
        mw = sum(w for w, _, _ in gs) / len(gs)
        gs = [(w, y + s * (w - mw), d) for w, y, d in gs]
        eras = {"pre26": [(w, y) for w, y, d in gs if d < ERA_SPLIT],
                "y26": [(w, y) for w, y, d in gs if d >= ERA_SPLIT]}
        eras["full"] = eras["pre26"] + eras["y26"]
        for a in arms:
            r = slope_se([w for w, _ in eras[a]], [y for _, y in eras[a]])
            if r is None:
                obs[a]["b"].append(np.nan); obs[a]["se"].append(np.nan)
            else:
                obs[a]["b"].append(r[0]); obs[a]["se"].append(r[1])
            obs[a]["n"].append(len(eras[a]))
    return {a: {k: np.array(v, float) for k, v in d.items()}
            for a, d in obs.items()}


def stv_z(obs, rep, sign):
    b1, s1, o1 = stats(obs, rep, "pre26", MIN_GAMES_ERA)
    b2, s2, o2 = stats(obs, rep, "y26", MIN_GAMES_ERA)
    idx = np.where(o1 & o2)[0]
    if len(idx) < 40:
        return np.nan
    order = np.argsort(-sign * (b1 / s1)[idx])
    sel = idx[order[:40]]
    w = 1.0 / s2[sel] ** 2
    return sign * np.sum(w * b2[sel]) / np.sum(w) * math.sqrt(np.sum(w))


def main():
    rng = random.Random(SEED)
    players, n_games = load_panel()
    pids, obs0, rep = build_arrays(players, rng)

    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# Rare wind skill — injection power test\n")
    say(f"Real panel ({len(pids)} players), trait planted in "
        f"{PI_INJ:.0%} of players with slope ±s (random sign, persistent "
        f"across eras), {N_SIM} sims per size. Detection = spike-slab "
        f"LR > {LR_BAR} (wind_rare null max) or STV K=40 z > 2.5 "
        "(either direction).\n")
    say("| s (share/10mph) | LR mean | P(LR fires) | STV z mean | "
        "P(STV fires) | P(either) |")
    say("|---|---|---|---|---|---|")

    for s in SIZES:
        lrs, stvs, hit_lr, hit_stv, hit_any = [], [], 0, 0, 0
        for _ in range(N_SIM):
            slopes = {pid: (s if rng.random() < 0.5 else -s)
                      for pid in pids if rng.random() < PI_INJ}
            obs = obs_arrays(players, pids, slopes)
            b, se, ok = stats(obs, rep, "full", MIN_GAMES)
            f = spike_slab(b[ok], se[ok])
            z = np.nanmax([stv_z(obs, rep, +1), stv_z(obs, rep, -1)])
            lrs.append(f["lr"]); stvs.append(z)
            a = f["lr"] > LR_BAR; b_ = z > 2.5
            hit_lr += a; hit_stv += b_; hit_any += (a or b_)
        say(f"| {s*10:.3f} | {np.mean(lrs):.1f} | {hit_lr/N_SIM:.2f} "
            f"| {np.nanmean(stvs):+.2f} | {hit_stv/N_SIM:.2f} "
            f"| {hit_any/N_SIM:.2f} |")

    say("\n*Sign persistent across eras = a genuine trait; random sign "
        "mirrors the two-sided tail structure. s = 0 row is the false-"
        "positive check. Null replicates are the real-data permutations "
        "(injection slightly widens injected players' true se, so "
        "detection rates are, if anything, optimistic — fine for an "
        "upper bound on the telescope).*")

    (ROOT / "model/wind_rare_power.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/wind_rare_power.md")


if __name__ == "__main__":
    main()
