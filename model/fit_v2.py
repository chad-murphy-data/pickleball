"""v2 model: dynamic skill + race-to-T likelihood + weakest link.

Everything that survived testing, in one joint 2024-2026 fit:

  points:   s1 ~ Binomial(s1+s2, sigma(eta))          [per-point Bernoulli;
            includes to-15 Challenger games natively — the likelihood only
            sees points, the target only matters for win-prob prediction]
  eta_g   = b_tour + S1 - S2 + d_dyad1 - d_dyad2 + m_match + beta_new*xnew
  S_team  = v_i(t) + v_j(t) + gamma * |v_i(t) - v_j(t)|   [weakest link]
  v_i(t)  = v0_i + tau * cumsum(innovations)  (monthly grid, 2024-01..2026-07)
            for players with >= DYN_MIN career games; static otherwise

Dropped with evidence: context deviations (sd 0.13), player-tour effects
(no sandbagging; slope test flat), gender-role delta in mixed (rejected).
Ramp-up (beta_new) retested here with TRUE pair histories across all years.

Env: SRM2_SUFFIX, SRM2_DATE_BEFORE (holdout training), SRM2_WARMUP/SAMPLES.
Run:  python model/fit_v2.py     (CPU NUTS, expect 1-3 h)
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
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

SEED = 20260712
SUFFIX = os.environ.get("SRM2_SUFFIX", "")
DATE_BEFORE = os.environ.get("SRM2_DATE_BEFORE", "")
N_WARMUP = int(os.environ.get("SRM2_WARMUP", 600))
N_SAMPLES = int(os.environ.get("SRM2_SAMPLES", 600))
DYN_MIN = 60          # career games to earn a dynamic value
NEW_GAMES, NEW_MIN_TOTAL = 6, 15
START = (2024, 1)     # month grid origin


def month_idx(date_str):
    y, m = int(date_str[:4]), int(date_str[5:7])
    return (y - START[0]) * 12 + (m - START[1])


def prep():
    games = [g for g in csv.DictReader((DATA / "games.csv").open())
             if g["is_forfeit"] == "False"
             and g["scoring_format"] in ("sideout_11", "sideout_15")]
    if DATE_BEFORE:
        games = [g for g in games if g["date"] < DATE_BEFORE]
    games.sort(key=lambda g: g["date"])

    pidx, didx, midx = {}, {}, {}
    pgames = Counter()
    for g in games:
        for c in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pgames[g[c]] += 1
    dyn = {u for u, n in pgames.items() if n >= DYN_MIN}

    def p(u): return pidx.setdefault(u, len(pidx))
    def d(a, b): return didx.setdefault(tuple(sorted((a, b))), len(didx))

    n = len(games)
    A = np.zeros((n, 4), np.int32)
    D1 = np.zeros(n, np.int32); D2 = np.zeros(n, np.int32)
    M = np.zeros(n, np.int32); T = np.zeros(n, np.int32)
    MO = np.zeros(n, np.int32)
    S1 = np.zeros(n, np.float32); S2 = np.zeros(n, np.float32)
    dyad_seen = Counter(); dyad_tot = Counter()
    for g in games:
        dyad_tot[tuple(sorted((g["t1_p1"], g["t1_p2"])))] += 1
        dyad_tot[tuple(sorted((g["t2_p1"], g["t2_p2"])))] += 1
    XN = np.zeros(n, np.float32)
    for i, g in enumerate(games):
        A[i] = [p(g["t1_p1"]), p(g["t1_p2"]), p(g["t2_p1"]), p(g["t2_p2"])]
        k1 = tuple(sorted((g["t1_p1"], g["t1_p2"])))
        k2 = tuple(sorted((g["t2_p1"], g["t2_p2"])))
        D1[i] = d(*k1); D2[i] = d(*k2)
        M[i] = midx.setdefault(g["match_id"], len(midx))
        T[i] = 0 if g["tour"] == "MLP" else 1
        MO[i] = month_idx(g["date"])
        S1[i] = int(g["t1_score"]); S2[i] = int(g["t2_score"])
        for k, sign in ((k1, 1.0), (k2, -1.0)):
            if dyad_tot[k] >= NEW_MIN_TOTAL and dyad_seen[k] < NEW_GAMES:
                XN[i] += sign
            dyad_seen[k] += 1

    is_dyn = np.zeros(len(pidx), np.int32)
    dyn_id = np.full(len(pidx), -1, np.int32)
    k = 0
    for u, i in pidx.items():
        if u in dyn:
            is_dyn[i] = 1; dyn_id[i] = k; k += 1
    n_months = int(MO.max()) + 1
    print(f"games={n} players={len(pidx)} (dynamic={k}) dyads={len(didx)} "
          f"matches={len(midx)} months={n_months}")
    dat = dict(A=A, D1=D1, D2=D2, M=M, T=T, MO=MO, S1=S1, S2=S2, XN=XN,
               is_dyn=is_dyn, dyn_id=np.maximum(dyn_id, 0), n_months=n_months)
    return dat, pidx, didx, len(midx), k


def model(dat, n_players, n_dyads, n_matches, n_dyn):
    sd_v = numpyro.sample("sd_v", dist.HalfNormal(0.5))
    sd_d = numpyro.sample("sd_d", dist.HalfNormal(0.15))
    sd_m = numpyro.sample("sd_m", dist.HalfNormal(0.3))
    tau = numpyro.sample("tau", dist.HalfNormal(0.05))
    gamma = numpyro.sample("gamma", dist.Normal(0, 0.3))
    beta_new = numpyro.sample("beta_new", dist.Normal(0, 0.2))
    b_tour = numpyro.sample("b_tour", dist.Normal(0, 0.3).expand([2]))

    v0 = numpyro.sample("v0_raw", dist.Normal(0, 1).expand([n_players])) * sd_v
    d = numpyro.sample("d_raw", dist.Normal(0, 1).expand([n_dyads])) * sd_d
    m = numpyro.sample("m_raw", dist.Normal(0, 1).expand([n_matches])) * sd_m
    innov = numpyro.sample("innov", dist.Normal(0, 1).expand([n_dyn, dat["n_months"]]))
    walk = jnp.cumsum(innov, axis=1) * tau              # (n_dyn, months)

    # player value at each game's month
    base = v0[dat["A"]]                                  # (games, 4)
    drift = walk[dat["dyn_id"][dat["A"]], dat["MO"][:, None]]
    val = base + drift * dat["is_dyn"][dat["A"]]

    g1 = jnp.abs(val[:, 0] - val[:, 1]); g2 = jnp.abs(val[:, 2] - val[:, 3])
    team1 = val[:, 0] + val[:, 1] + gamma * g1
    team2 = val[:, 2] + val[:, 3] + gamma * g2
    eta = (b_tour[dat["T"]] + team1 - team2
           + d[dat["D1"]] - d[dat["D2"]] + m[dat["M"]] + beta_new * dat["XN"])
    total = dat["S1"] + dat["S2"]
    numpyro.sample("y", dist.Binomial(total_count=total, logits=eta), obs=dat["S1"])


def main():
    dat, pidx, didx, n_matches, n_dyn = prep()
    n_players, n_dyads = len(pidx), len(didx)
    jdat = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in dat.items()}

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9),
                num_warmup=N_WARMUP, num_samples=N_SAMPLES, num_chains=2,
                chain_method="parallel", progress_bar=True)
    mcmc.run(jax.random.PRNGKey(SEED), jdat, n_players, n_dyads, n_matches, n_dyn)
    samp = mcmc.get_samples(group_by_chain=True)
    n_div = int(np.sum(np.asarray(mcmc.get_extra_fields().get("diverging", np.zeros(1)))))

    from numpyro.diagnostics import summary as nsummary
    scal = nsummary({k: samp[k] for k in ("sd_v", "sd_d", "sd_m", "tau",
                                          "gamma", "beta_new", "b_tour")})
    rhat_v = float(np.nanmax(nsummary({"v0_raw": samp["v0_raw"]})["v0_raw"]["r_hat"]))

    sd_v = np.asarray(samp["sd_v"]).reshape(-1)
    tau = np.asarray(samp["tau"]).reshape(-1)
    v0 = np.asarray(samp["v0_raw"]).reshape(-1, n_players) * sd_v[:, None]
    innov = np.asarray(samp["innov"]).reshape(-1, n_dyn, dat["n_months"])
    walk = np.cumsum(innov, axis=2) * tau[:, None, None]

    players_meta = {r["player_id"]: r for r in csv.DictReader((DATA / "players.csv").open())}
    inv_p = {i: u for u, i in pidx.items()}
    pg = Counter()
    for row in dat["A"]:
        for i in row:
            pg[int(i)] += 1

    # trajectories for dynamic players + final-month values for everyone
    with (DATA / f"v2_players{SUFFIX}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "full_name", "gender", "games",
                    "value_now_mean", "value_now_sd", "dynamic"])
        last = dat["n_months"] - 1
        for i in range(n_players):
            u = inv_p[i]
            meta = players_meta.get(u, {})
            vi = v0[:, i]
            if dat["is_dyn"][i]:
                vi = vi + walk[:, dat["dyn_id"][i], last]
            w.writerow([u, meta.get("full_name", ""), meta.get("gender", ""),
                        pg[i], round(float(vi.mean()), 4), round(float(vi.std()), 4),
                        int(dat["is_dyn"][i])])
    with (DATA / f"v2_trajectories{SUFFIX}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "full_name", "month", "value_mean", "value_sd"])
        for i in range(n_players):
            if not dat["is_dyn"][i]:
                continue
            u = inv_p[i]
            name = players_meta.get(u, {}).get("full_name", "")
            for t in range(dat["n_months"]):
                vt = v0[:, i] + walk[:, dat["dyn_id"][i], t]
                w.writerow([u, name, f"{START[0] + t // 12}-{t % 12 + 1:02d}",
                            round(float(vt.mean()), 4), round(float(vt.std()), 4)])

    sd_d_s = np.asarray(samp["sd_d"]).reshape(-1)
    dd = np.asarray(samp["d_raw"]).reshape(-1, n_dyads) * sd_d_s[:, None]
    inv_d = {i: k for k, i in didx.items()}
    with (DATA / f"v2_dyads{SUFFIX}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["p1_name", "p2_name", "chem_logit_mean", "chem_logit_sd"])
        dm, ds = dd.mean(axis=0), dd.std(axis=0)
        for i in range(n_dyads):
            u1, u2 = inv_d[i]
            w.writerow([players_meta.get(u1, {}).get("full_name", u1[:8]),
                        players_meta.get(u2, {}).get("full_name", u2[:8]),
                        round(float(dm[i]), 4), round(float(ds[i]), 4)])

    if os.environ.get("SRM2_SAVE_DRAWS", "") == "1":
        np.savez_compressed(OUT / f"v2_draws{SUFFIX}.npz",
                            v0=v0.astype(np.float32),
                            walk_last=(walk[:, :, -1]).astype(np.float32),
                            dyn_id=dat["dyn_id"], is_dyn=dat["is_dyn"],
                            gamma=np.asarray(samp["gamma"]).reshape(-1),
                            player_ids=np.array([inv_p[i] for i in range(n_players)]))

    fit = {"n_divergences": n_div, "max_rhat_v0": rhat_v,
           "scalars": {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                           for kk, vv in s.items() if kk in ("mean", "std", "r_hat")}
                       for k, s in scal.items()}}
    (OUT / f"v2_fit_summary{SUFFIX}.json").write_text(json.dumps(fit, indent=1, default=str))
    print(json.dumps(fit["scalars"], indent=1, default=str)[:1200])
    print("divergences:", n_div, "| max rhat v0:", rhat_v)


if __name__ == "__main__":
    main()
