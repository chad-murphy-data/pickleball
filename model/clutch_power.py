"""Positive control: if clutch existed at size X, would this pipeline see it?

A null result is only worth something with a power curve behind it. So we
generate synthetic seasons in which clutch is REAL and of known size — each
player gets a true coefficient on standardised leverage, drawn from
N(0, sigma) — run them through exactly the same correction, shrinkage and
reliability machinery as the real archive, and check what comes back.

If the pipeline recovers tau at the injected value and the cross-era
correlation lights up, then failing to find it in the real data means it is
not there at that size. If the pipeline recovers nothing even when clutch is
injected, the real null means nothing at all.

Run:  python model/clutch_power.py [--null clutch_null_model]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))

import clutch_leverage as cl          # noqa: E402
import clutch_null as cn              # noqa: E402
from clutch_leverage import eb, boot_corr  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def sim_with_clutch(games, rng, tabs, model, kappa, lev_sd, gslice, npl, ng):
    """One season where player i's rally-win log-odds carry kappa[i] * Lz.

    Lz is the rally's leverage in global sd units — the same quantity the
    estimator regresses on — so the injected effect is on exactly the scale
    the pipeline reports, up to the logistic derivative p(1-p).
    """
    srv, rcv, gid, lev, won = [], [], [], [], []
    lev_mean_z = 0.1347 / lev_sd
    for (g, disc, T, s0, s1) in games:
        team = (s0, s1)
        cap = T + cn.CAP_EXTRA
        tab = tabs[(disc, T)]
        sc = [0, 0]
        s = int(rng.integers(2))
        doubles = disc == "doubles"
        sn = 2 if doubles else 0
        who = 0
        rpos = [int(rng.integers(2)), int(rng.integers(2))]
        for _ in range(400):
            a, b = sc[s], sc[1 - s]
            if (a >= T and a - b >= 2) or (b >= T and b - a >= 2):
                break
            if a >= cap or b >= cap:
                break
            server = team[s][who] if doubles else team[s][0]
            receiver = team[1 - s][rpos[1 - s]] if doubles else team[1 - s][0]
            lz = tab[(a, b, sn)]
            eta = model[0][server] - model[1][receiver]
            # server pushes on big points; receiver pulls back on them
            eta += (kappa[server] - kappa[receiver]) * (lz - lev_mean_z)
            w = 1 if rng.random() < 1.0 / (1.0 + np.exp(-eta)) else 0
            srv.append(server); rcv.append(receiver); gid.append(g)
            lev.append(lz); won.append(w)
            if w:
                sc[s] += 1
                if doubles:
                    rpos[1 - s] ^= 1
            elif doubles:
                if sn == 1:
                    sn, who = 2, 1 - who
                else:
                    s, sn, who = 1 - s, 1, 0
            else:
                s = 1 - s
    return (np.array(srv), np.array(rcv), np.array(gid),
            np.array(lev, dtype=float), np.array(won, dtype=float))


def pipeline(nz, obs_slices, sl, chans, min_rallies=400):
    """Same correction as clutch_report.stat, on an injected season."""
    U = sum(obs_slices[f"U_{sl}_{c}"] for c in chans)
    S = sum(obs_slices[f"SSL_{sl}_{c}"] for c in chans)
    V = sum(obs_slices[f"V_{sl}_{c}"] for c in chans)
    n = sum(obs_slices[f"n_{sl}_{c}"] for c in chans)
    Ur = sum(nz[f"U_rep_{sl}_{c}"] for c in chans)
    Sr = sum(nz[f"SSL_rep_{sl}_{c}"] for c in chans)
    reps = Ur.shape[0]
    ok = (S > 0) & (V > 0) & (n >= min_rallies) & (Sr.min(axis=0) > 0)
    b = np.where(ok, U / np.where(ok, S, 1), np.nan)
    br = np.where(ok[None, :], Ur / np.where(ok[None, :], Sr, 1), np.nan)
    bm = np.nanmean(br, axis=0)
    sdm = np.nanstd(br, axis=0, ddof=1)
    se = np.sqrt((np.sqrt(V) / np.where(ok, S, 1)) ** 2 + sdm ** 2 / reps)
    return b - bm, se, ok


def main(null="clutch_null_model", levels=(0.0, 0.005, 0.010, 0.020),
         seed=777):
    d = cl.load()
    F = cl.Frame(d)
    npl, ng = F.npl, int(d["gidx"].max()) + 1
    games, gok = cn.rosters(d, F)
    lev_sd = float(d["lev_raw"].std())
    tabs = {}
    for T in (11, 15):
        tabs[("doubles", T)] = {k: v / lev_sd for k, v in
                                cl.doubles_leverage(d["k_doubles"], T).items()}
        tabs[("singles", T)] = {k: v / lev_sd for k, v in
                                cl.singles_leverage(d["k_singles"], T).items()}
    model = cn.fit_serve_model(d, F, npl)
    nz = np.load(DATA / f"{null}.npz", allow_pickle=True)

    year = np.array([s[:4] for s in d["date"]])
    gyear = np.empty(ng, dtype="<U4")
    gyear[d["gidx"]] = year
    gsl = {"all": np.ones(ng, bool), "pre26": gyear != "2026",
           "y26": gyear == "2026"}

    print("=" * 74)
    print(f"POSITIVE CONTROL — injected clutch vs recovered ({null})")
    print("=" * 74)
    print(f"{'injected tau':>13}{'recovered tau':>15}{'95% CI':>22}"
          f"{'era r':>9}{'n sig':>7}")
    rng = np.random.default_rng(seed)
    for sig in levels:
        # b is on the probability scale; the injected logit coefficient is
        # b / p(1-p) with p ~ 0.47, i.e. roughly 4x.
        kappa = rng.normal(0, sig / 0.249, npl)
        obs = {}
        srv, rcv, gid, lev, won = sim_with_clutch(
            games, rng, tabs, model, kappa, lev_sd, gsl, npl, ng)
        for nm, gm in gsl.items():
            m = gm[gid]
            for ch, who, y in (("S", srv, won), ("R", rcv, 1.0 - won)):
                u, s_, v_, n_ = cn.channel_U(who[m], gid[m], lev[m], y[m],
                                             npl, ng)
                obs[f"U_{nm}_{ch}"] = u
                obs[f"SSL_{nm}_{ch}"] = s_
                obs[f"V_{nm}_{ch}"] = v_
                obs[f"n_{nm}_{ch}"] = n_
        b, se, ok = pipeline(nz, obs, "all", ("S", "R"))
        mu, tau, post, psd, shr = eb(b[ok], se[ok])
        lo, hi = cl.tau_ci(b[ok], se[ok])
        b1, s1, o1 = pipeline(nz, obs, "pre26", ("S", "R"))
        b2, s2, o2 = pipeline(nz, obs, "y26", ("S", "R"))
        m2 = o1 & o2
        era = float(np.corrcoef(b1[m2], b2[m2])[0, 1])
        nsig = int((np.abs(post) > 1.96 * psd).sum())
        print(f"{sig:>13.4f}{tau:>15.5f}   [{lo:.5f}, {hi:.5f}]"
              f"{era:>9.3f}{nsig:>7}")
    print()
    print("Injected tau is the sd of TRUE clutch across players, in extra")
    print("rally-win probability per +1 sd of leverage — the same units the")
    print("real answer is reported in.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", default="clutch_null_model")
    main(**vars(ap.parse_args()))
