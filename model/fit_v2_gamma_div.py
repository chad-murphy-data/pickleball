"""v2 variant: per-division weakest-link gamma (mens/womens/mixed).

    python model/fit_v2_gamma_div.py     # CPU NUTS, hours; prints + writes
                                         # model/gamma_division_refit.md

Identical to fit_v2.py in every respect except gamma is a length-3 vector
indexed by game context. Answers "is the 0.42/0.58 better/worse partner
weighting the same across divisions?" WITHOUT the circularity of the
profile-likelihood check (model/gamma_division.py), where values fitted
under the pooled gamma partially absorb any divergence.

Writes only the report + gamma draws (model/gamma_div_draws.npz); does NOT
touch the standard v2 outputs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import fit_v2 as F  # noqa: E402  (reuse prep; must not diverge from it)

DIVS = ("mens", "womens", "mixed")


def div_array():
    """Division index per game, in the exact order fit_v2.prep uses."""
    games = [g for g in csv.DictReader((F.DATA / "games.csv").open())
             if g["is_forfeit"] == "False"
             and g["scoring_format"] in ("sideout_11", "sideout_15")]
    if F.DATE_BEFORE:
        games = [g for g in games if g["date"] < F.DATE_BEFORE]
    games.sort(key=lambda g: g["date"])
    return np.array([DIVS.index(g["context"]) for g in games], np.int32)


def model(dat, n_players, n_dyads, n_matches, n_dyn):
    sd_v = numpyro.sample("sd_v", dist.HalfNormal(0.5))
    sd_d = numpyro.sample("sd_d", dist.HalfNormal(0.15))
    sd_m = numpyro.sample("sd_m", dist.HalfNormal(0.3))
    tau = numpyro.sample("tau", dist.HalfNormal(0.05))
    gamma = numpyro.sample("gamma", dist.Normal(0, 0.3).expand([3]))
    beta_new = numpyro.sample("beta_new", dist.Normal(0, 0.2))
    b_tour = numpyro.sample("b_tour", dist.Normal(0, 0.3).expand([2]))

    v0 = numpyro.sample("v0_raw", dist.Normal(0, 1).expand([n_players])) * sd_v
    d = numpyro.sample("d_raw", dist.Normal(0, 1).expand([n_dyads])) * sd_d
    m = numpyro.sample("m_raw", dist.Normal(0, 1).expand([n_matches])) * sd_m
    innov = numpyro.sample("innov", dist.Normal(0, 1).expand([n_dyn, dat["n_months"]]))
    walk = jnp.cumsum(innov, axis=1) * tau

    base = v0[dat["A"]]
    drift = walk[dat["dyn_id"][dat["A"]], dat["MO"][:, None]]
    val = base + drift * dat["is_dyn"][dat["A"]]

    gam_g = gamma[dat["DIV"]]
    g1 = jnp.abs(val[:, 0] - val[:, 1]); g2 = jnp.abs(val[:, 2] - val[:, 3])
    team1 = val[:, 0] + val[:, 1] + gam_g * g1
    team2 = val[:, 2] + val[:, 3] + gam_g * g2
    eta = (b_tour[dat["T"]] + team1 - team2
           + d[dat["D1"]] - d[dat["D2"]] + m[dat["M"]] + beta_new * dat["XN"])
    total = dat["S1"] + dat["S2"]
    numpyro.sample("y", dist.Binomial(total_count=total, logits=eta), obs=dat["S1"])


def main():
    dat, pidx, didx, n_matches, n_dyn = F.prep()
    dat["DIV"] = div_array()
    assert len(dat["DIV"]) == len(dat["S1"]), "DIV misaligned with prep()"
    jdat = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
            for k, v in dat.items()}

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9),
                num_warmup=F.N_WARMUP, num_samples=F.N_SAMPLES, num_chains=2,
                chain_method="parallel", progress_bar=True)
    mcmc.run(jax.random.PRNGKey(F.SEED), jdat, len(pidx), len(didx),
             n_matches, n_dyn)
    samp = mcmc.get_samples(group_by_chain=True)
    n_div = int(np.sum(np.asarray(
        mcmc.get_extra_fields().get("diverging", np.zeros(1)))))

    from numpyro.diagnostics import summary as nsummary
    scal = nsummary({"gamma": samp["gamma"]})["gamma"]
    gam = np.asarray(samp["gamma"]).reshape(-1, 3)
    np.savez(ROOT / "model/gamma_div_draws.npz", gamma=gam, divs=np.array(DIVS))

    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# Per-division weakest-link gamma — joint v2 refit\n")
    say(f"Same spec as fit_v2 except gamma is per-division. "
        f"{len(dat['DIV'])} games; divergences: {n_div}; "
        f"gamma r_hat max {float(np.max(scal['r_hat'])):.3f}\n")
    say("| division | gamma | 95% CI | better wt | worse wt |")
    say("|---|---|---|---|---|")
    for i, dv in enumerate(DIVS):
        g = gam[:, i]
        lo, hi = np.percentile(g, [2.5, 97.5])
        mu = g.mean()
        say(f"| {dv} | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}] "
            f"| {(1+mu)/2:.3f} | {(1-mu)/2:.3f} |")
    say("\n| contrast | Δgamma | 95% CI | P(Δ>0) |")
    say("|---|---|---|---|")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        d = gam[:, a] - gam[:, b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        say(f"| {DIVS[a]} − {DIVS[b]} | {d.mean():+.3f} "
            f"| [{lo:+.3f}, {hi:+.3f}] | {float((d > 0).mean()):.3f} |")
    say("\n*Posterior from the joint refit — values, dyads, match effects "
        "and the division gammas estimated together, so no "
        "conditional-on-pooled-values circularity.*")
    (ROOT / "model/gamma_division_refit.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/gamma_division_refit.md")


if __name__ == "__main__":
    main()
