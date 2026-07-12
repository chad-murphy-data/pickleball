"""Bayesian social-relations-style model for 2026 pro pickleball doubles margins.

Identifiability note (the honest version of "actor vs partner"):
with team-level margins, any additive player-level split of "own skill" vs
"boost to partner" is unidentifiable — only their sum enters the likelihood.
What IS identifiable:

    margin_g = beta_tour[t]                          (team-one bias, e.g. seeding)
             + sum_{i in team1} (v_i + w_{i,ctx})    player value (+ context shift)
             - sum_{i in team2} (v_i + w_{i,ctx})
             + d_{dyad1} - d_{dyad2}                 pair chemistry beyond additivity
             + m_{match}                             correlated games within a match
             + eps_g

v = overall player value (points per game added to the team margin),
w = context-specific deviation (mixed / mens / womens "DIF"),
d = dyad chemistry (zero-mean; how far a pair deviates from the sum of parts),
m = match random intercept (handles Bo3/Bo5 correlation + game-3 selection).

Everything non-centered; HalfNormal priors on scales.

Run:  python model/fit_srm.py            (NUTS, ~10-30 min CPU)
Outputs: data/results_players.csv, data/results_dyads.csv, model/fit_summary.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(2)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent

import os

SEED = 20260712
N_WARMUP = int(os.environ.get("SRM_WARMUP", 700))
N_SAMPLES = int(os.environ.get("SRM_SAMPLES", 700))
N_CHAINS = int(os.environ.get("SRM_CHAINS", 2))
SUFFIX = os.environ.get("SRM_SUFFIX", "")
SD_D_PRIOR = float(os.environ.get("SRM_SD_D_PRIOR", 1.5))
PLAYER_TOUR = os.environ.get("SRM_PLAYER_TOUR", "") == "1"
NEWNESS = os.environ.get("SRM_NEWNESS", "") == "1"
SAVE_DRAWS = os.environ.get("SRM_SAVE_DRAWS", "") == "1"
NEW_GAMES = 6      # a dyad's first N games count as "new"
NEW_MIN_TOTAL = 15 # ...but only for pairs that eventually log >= this many


def load():
    rows = list(csv.DictReader((DATA / f"model_data{SUFFIX}.csv").open()))
    def col(k, dtype=np.int32):
        return np.array([dtype(r[k]) for r in rows])
    dat = dict(
        margin=col("margin", np.float32),
        a=np.stack([col("a1"), col("a2"), col("a3"), col("a4")], axis=1),
        dyad1=col("dyad1"), dyad2=col("dyad2"),
        match=col("match_idx"), ctx=col("ctx_idx"), tour=col("tour_idx"),
    )
    # ramp-up indicator: rows are date-ordered, so a running per-dyad counter
    # marks each pairing's first NEW_GAMES games (established dyads only)
    from collections import Counter
    totals = Counter()
    for r in rows:
        totals[r["dyad1"]] += 1; totals[r["dyad2"]] += 1
    seen = Counter()
    xnew = np.zeros(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        for dcol, sign in (("dyad1", 1.0), ("dyad2", -1.0)):
            k = r[dcol]
            if totals[k] >= NEW_MIN_TOTAL and seen[k] < NEW_GAMES:
                xnew[i] += sign
            seen[k] += 1
    dat["xnew"] = xnew
    players = list(csv.DictReader((DATA / f"model_players{SUFFIX}.csv").open()))
    dyads = list(csv.DictReader((DATA / f"model_dyads{SUFFIX}.csv").open()))
    return dat, players, dyads


def build_pc_index(dat, n_players):
    """Index observed (player, context) combos for the w deviations."""
    combos = {}
    pc = np.zeros((len(dat["margin"]), 4), dtype=np.int32)
    for g in range(len(dat["margin"])):
        c = dat["ctx"][g]
        for slot in range(4):
            key = (dat["a"][g, slot], c)
            pc[g, slot] = combos.setdefault(key, len(combos))
    return pc, combos


def model(dat, n_players, n_dyads, n_matches, n_pc, pc, pt=None, n_pt=0):
    sd_v = numpyro.sample("sd_v", dist.HalfNormal(3.0))
    sd_w = numpyro.sample("sd_w", dist.HalfNormal(1.5))
    sd_d = numpyro.sample("sd_d", dist.HalfNormal(SD_D_PRIOR))
    sd_m = numpyro.sample("sd_m", dist.HalfNormal(3.0))
    sd_e = numpyro.sample("sd_e", dist.HalfNormal(5.0))
    beta_tour = numpyro.sample("beta_tour", dist.Normal(0, 3).expand([2]))

    v = numpyro.sample("v_raw", dist.Normal(0, 1).expand([n_players])) * sd_v
    w = numpyro.sample("w_raw", dist.Normal(0, 1).expand([n_pc])) * sd_w
    d = numpyro.sample("d_raw", dist.Normal(0, 1).expand([n_dyads])) * sd_d
    m = numpyro.sample("m_raw", dist.Normal(0, 1).expand([n_matches])) * sd_m

    val = v[dat["a"]] + w[pc]                     # (games, 4)
    if pt is not None:
        sd_pt = numpyro.sample("sd_pt", dist.HalfNormal(1.0))
        u = numpyro.sample("pt_raw", dist.Normal(0, 1).expand([n_pt])) * sd_pt
        val = val + u[pt]
    team1 = val[:, 0] + val[:, 1]
    team2 = val[:, 2] + val[:, 3]
    mu = (beta_tour[dat["tour"]]
          + team1 - team2
          + d[dat["dyad1"]] - d[dat["dyad2"]]
          + m[dat["match"]])
    if NEWNESS:
        beta_new = numpyro.sample("beta_new", dist.Normal(0, 2))
        mu = mu + beta_new * dat["xnew"]
    numpyro.sample("y", dist.Normal(mu, sd_e), obs=dat["margin"])


def main():
    dat, players, dyads = load()
    n_players, n_dyads = len(players), len(dyads)
    n_matches = int(dat["match"].max()) + 1
    pc, combos = build_pc_index(dat, n_players)
    n_pc = len(combos)
    pt, ptcombos, n_pt = None, {}, 0
    if PLAYER_TOUR:
        pt = np.zeros((len(dat["margin"]), 4), dtype=np.int32)
        for g in range(len(dat["margin"])):
            t = dat["tour"][g]
            for slot in range(4):
                key = (dat["a"][g, slot], t)
                pt[g, slot] = ptcombos.setdefault(key, len(ptcombos))
        n_pt = len(ptcombos)
        print(f"player-tour combos: {n_pt}")
    print(f"games={len(dat['margin'])} players={n_players} dyads={n_dyads} "
          f"matches={n_matches} player-context combos={n_pc}")

    jdat = {k: jnp.asarray(x) for k, x in dat.items()}
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                num_chains=N_CHAINS, chain_method="parallel", progress_bar=True)
    mcmc.run(jax.random.PRNGKey(SEED), jdat, n_players, n_dyads, n_matches,
             n_pc, jnp.asarray(pc),
             pt=None if pt is None else jnp.asarray(pt), n_pt=n_pt)

    samp = mcmc.get_samples(group_by_chain=True)  # (chains, draws, ...)
    extra = mcmc.get_extra_fields()
    n_div = int(np.sum(np.asarray(extra.get("diverging", np.zeros(1)))))

    # scalars + convergence
    from numpyro.diagnostics import summary as nsummary
    extra_scalars = [k for k in ("sd_pt", "beta_new") if k in samp]
    scal = nsummary({k: samp[k] for k in
                     ("sd_v", "sd_w", "sd_d", "sd_m", "sd_e", "beta_tour") + tuple(extra_scalars)})
    rhats = []
    for k in ("v_raw", "d_raw"):
        s = nsummary({k: samp[k]})[k]
        rhats.append(float(np.nanmax(s["r_hat"])))

    sd_v = np.asarray(samp["sd_v"]).reshape(-1)
    sd_w = np.asarray(samp["sd_w"]).reshape(-1)
    sd_d = np.asarray(samp["sd_d"]).reshape(-1)

    v = np.asarray(samp["v_raw"]).reshape(-1, n_players) * sd_v[:, None]
    w = np.asarray(samp["w_raw"]).reshape(-1, n_pc) * sd_w[:, None]
    d = np.asarray(samp["d_raw"]).reshape(-1, n_dyads) * sd_d[:, None]

    # per-player games + per-dyad games for reporting
    pg = np.zeros(n_players, dtype=int)
    for slot in range(4):
        np.add.at(pg, dat["a"][:, slot], 1)
    dg = np.zeros(n_dyads, dtype=int)
    np.add.at(dg, dat["dyad1"], 1)
    np.add.at(dg, dat["dyad2"], 1)

    ctx_names = {0: "mixed", 1: "mens", 2: "womens"}
    w_mean = w.mean(axis=0)
    w_by_player = {}
    for (pi, c), idx in combos.items():
        w_by_player.setdefault(int(pi), {})[ctx_names[int(c)]] = float(w_mean[idx])

    with (DATA / f"results_players{SUFFIX}.csv").open("w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["player_id", "full_name", "gender", "games",
                       "value_mean", "value_sd", "w_mixed", "w_mens", "w_womens"])
        vm, vs = v.mean(axis=0), v.std(axis=0)
        for i, p in enumerate(players):
            wb = w_by_player.get(i, {})
            wcsv.writerow([p["player_id"], p["full_name"], p["gender"], pg[i],
                           round(float(vm[i]), 3), round(float(vs[i]), 3),
                           round(wb.get("mixed", float("nan")), 3),
                           round(wb.get("mens", float("nan")), 3),
                           round(wb.get("womens", float("nan")), 3)])

    dm, ds = d.mean(axis=0), d.std(axis=0)
    p_pos = (d > 0).mean(axis=0)
    order = np.argsort(dm)
    pct = np.empty_like(order, dtype=float)
    pct[order] = np.arange(n_dyads) / (n_dyads - 1) * 100
    with (DATA / f"results_dyads{SUFFIX}.csv").open("w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["p1_name", "p2_name", "context", "games",
                       "chemistry_mean", "chemistry_sd", "p_positive", "percentile"])
        for i, dy in enumerate(dyads):
            wcsv.writerow([dy["p1_name"], dy["p2_name"], dy["context"], dg[i],
                           round(float(dm[i]), 3), round(float(ds[i]), 3),
                           round(float(p_pos[i]), 3), round(float(pct[i]), 1)])

    if SAVE_DRAWS:
        np.savez_compressed(
            OUT / f"draws{SUFFIX}.npz",
            v=v.astype(np.float32), d=d.astype(np.float32),
            sd_m=np.asarray(samp["sd_m"]).reshape(-1).astype(np.float32),
            sd_e=np.asarray(samp["sd_e"]).reshape(-1).astype(np.float32),
            player_ids=np.array([p["player_id"] for p in players]))
    if PLAYER_TOUR:
        sd_pt_s = np.asarray(samp["sd_pt"]).reshape(-1)
        uu = np.asarray(samp["pt_raw"]).reshape(-1, n_pt) * sd_pt_s[:, None]
        um = uu.mean(axis=0); us = uu.std(axis=0)
        with (DATA / f"results_player_tour{SUFFIX}.csv").open("w", newline="") as fh:
            wcsv = csv.writer(fh)
            wcsv.writerow(["player_id", "full_name", "tour", "effect_mean", "effect_sd"])
            for (pi, t), idx in ptcombos.items():
                wcsv.writerow([players[int(pi)]["player_id"], players[int(pi)]["full_name"],
                               "MLP" if t == 0 else "PPA",
                               round(float(um[idx]), 3), round(float(us[idx]), 3)])
    fit_summary = {
        "n_divergences": n_div,
        "max_rhat_v_d": rhats,
        "scalars": {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                        for kk, vv in s.items() if kk in ("mean", "std", "r_hat", "n_eff")}
                    for k, s in scal.items()},
    }
    (OUT / f"fit_summary{SUFFIX}.json").write_text(json.dumps(fit_summary, indent=1, default=str))
    print(json.dumps(fit_summary["scalars"], indent=1, default=str)[:1500])
    print("divergences:", n_div, "| max rhat (v,d):", rhats)


if __name__ == "__main__":
    main()
