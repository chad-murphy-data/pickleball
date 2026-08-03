"""Clutch, after the mechanical bias is taken out — the answer file.

Reads the observed leverage-covariance statistics and the real-schedule
no-clutch bootstrap, subtracts the second from the first, and reports what
survives: population spread, reliability across independent slices, and a
shrunk per-player leaderboard.

    python model/clutch_null.py --reps 60     # once, ~40 min
    python model/clutch_report.py

Everything here is a difference against a simulated league that shares the
real schedule, rosters, abilities and rules and differs in exactly one
respect: nobody in it is clutch.
"""
from __future__ import annotations

import csv
import json
import warnings
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "model"))

from clutch_leverage import eb, tau_ci, boot_corr, names, v2_values, wls  # noqa
from clutch_selfcheck import slice_stats as sc_slice  # noqa

warnings.filterwarnings("ignore", category=RuntimeWarning)

LEV_SD = 0.0659          # sd of raw leverage (win-prob swing) in the archive
MIN_RALLIES = 400


NULL = "clutch_null_model"


def load(null=None):
    return np.load(DATA / f"{null or NULL}.npz", allow_pickle=True), \
        np.load(DATA / "clutch_obs_year.npz", allow_pickle=True)


def stat(nz, oz, sl, channels=("S", "R"), min_rallies=MIN_RALLIES):
    """Bias-corrected clutch for one slice.

    b_obs   = U_obs / SSL_obs                    (observed slope)
    b_mech  = mean over replicates of U/SSL      (what no clutch produces)
    b_adj   = b_obs - b_mech
    se      = closed-form permutation se, widened by the Monte-Carlo error
              in b_mech; calibrated against the replicates themselves.
    """
    if sl == "all":
        U = sum(nz[f"U_obs_{c}"] for c in channels)
        SSL = sum(nz[f"SSL_obs_{c}"] for c in channels)
        V = sum(nz[f"V_obs_{c}"] for c in channels)
        n = sum(nz[f"n_obs_{c}"] for c in channels)
    else:
        U = sum(oz[f"U_{sl}_{c}"] for c in channels)
        SSL = sum(oz[f"SSL_{sl}_{c}"] for c in channels)
        V = sum(oz[f"V_{sl}_{c}"] for c in channels)
        n = sum(oz[f"n_{sl}_{c}"] for c in channels)

    Ur = sum(nz[f"U_rep_{sl}_{c}"] for c in channels)
    Sr = sum(nz[f"SSL_rep_{sl}_{c}"] for c in channels)
    Vr = sum(nz[f"V_rep_{sl}_{c}"] for c in channels)
    reps = Ur.shape[0]

    ok = (SSL > 0) & (V > 0) & (n >= min_rallies) & (Sr.min(axis=0) > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b_obs = np.where(ok, U / np.where(ok, SSL, 1), np.nan)
        br = np.where(ok[None, :], Ur / np.where(ok[None, :], Sr, 1), np.nan)
        b_mech = np.nanmean(br, axis=0)
        sd_mech = np.nanstd(br, axis=0, ddof=1)
        se_perm = np.where(ok, np.sqrt(np.where(ok, V, 1)) / np.where(ok, SSL, 1),
                           np.nan)

    # calibration: treat each replicate as a pseudo-observation and check the
    # closed-form se reproduces the spread the simulator actually generates.
    loo = (np.nansum(br, axis=0)[None, :] - br) / (reps - 1)
    se_r = np.sqrt(Vr) / np.where(Sr > 0, Sr, 1)
    zr = (br - loo) / se_r
    cal = float(np.nanvar(zr[:, ok]) / (1.0 + 1.0 / (reps - 1)))

    se = np.sqrt(se_perm ** 2 * cal + sd_mech ** 2 / reps)
    b_adj = b_obs - b_mech
    return {"b": b_adj, "se": se, "ok": ok, "n": n, "SSL": SSL,
            "b_obs": b_obs, "b_mech": b_mech, "cal": cal, "reps": reps,
            "se_perm": se_perm}


def main(null=None):
    nz, oz = load(null)
    uuids = nz["uuids"]
    nm, gd = names()
    val = v2_values()
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("WHO IS ACTUALLY BETTER IN BIG MOMENTS")
    say("clutch = leverage/outcome covariance, minus what a no-clutch league")
    say(f"produces on the same schedule ({int(nz['reps'])} simulated seasons)")
    say("=" * 78)
    say()

    A = stat(nz, oz, "all")
    ok = A["ok"]
    say(f"[0] NULL CALIBRATION   closed-form se reproduces the simulator's own")
    say(f"    spread to within a factor of {A['cal']:.3f} (target 1.000);")
    say(f"    that factor is applied to every se below.")
    say()

    say("[1] HOW BIG IS THE ARTIFACT?")
    say(f"    {'':<26}{'mean':>10}{'sd across players':>20}")
    say(f"    {'observed slope':<26}{np.nanmean(A['b_obs'][ok]):>+10.5f}"
        f"{np.nanstd(A['b_obs'][ok]):>20.5f}")
    say(f"    {'no-clutch simulation':<26}{np.nanmean(A['b_mech'][ok]):>+10.5f}"
        f"{np.nanstd(A['b_mech'][ok]):>20.5f}")
    say(f"    {'corrected':<26}{np.nanmean(A['b'][ok]):>+10.5f}"
        f"{np.nanstd(A['b'][ok]):>20.5f}")
    r_mech = np.corrcoef(A["b_obs"][ok], A["b_mech"][ok])[0, 1]
    say(f"    corr(observed, simulated-artifact) = {r_mech:+.3f}")
    say()

    say("[2] EXISTENCE — after correction, is the field still wider than chance?")
    say(f"    {'arm':<26}{'players':>8}{'var(z)':>9}{'tau':>9}{'tau 95% CI':>20}")
    taus = {}
    arms = [("all", ("S", "R"), "everything"),
            ("dbl", ("S", "R"), "doubles"),
            ("sgl", ("S", "R"), "singles"),
            ("all", ("S",), "serve rallies only"),
            ("all", ("R",), "return rallies only")]
    for sl, ch, lab in arms:
        s = stat(nz, oz, sl, ch)
        m = s["ok"]
        b, se = s["b"][m], s["se"][m]
        mu, tau, post, psd, shr = eb(b, se)
        lo, hi = tau_ci(b, se)
        taus[lab] = (tau, lo, hi)
        say(f"    {lab:<26}{m.sum():>8}{np.var(b/se):>9.3f}{tau:>9.5f}"
            f"   [{lo:.5f}, {hi:.5f}]")
    say("    tau = population sd of TRUE clutch: extra rally-win probability")
    say("    per +1 sd of leverage. var(z) = 1.0 would mean pure noise.")
    say()

    say("[3] RELIABILITY — does one slice predict a disjoint one?")
    say("    'null floor' is the same correlation measured on no-clutch")
    say("    replicates: two slices of one simulated season are not quite")
    say("    independent, so a small positive r is expected even at zero.")
    say(f"    {'split':<32}{'n':>6}{'r':>8}{'95% CI':>18}"
        f"{'null floor':>14}{'calib':>9}")

    def rel(a, b_, label, minr=MIN_RALLIES):
        s1 = stat(nz, oz, a[0], a[1], minr)
        s2 = stat(nz, oz, b_[0], b_[1], minr)
        m = s1["ok"] & s2["ok"]
        if m.sum() < 15:
            say(f"    {label:<32}{m.sum():>6}   (too few players)")
            return
        mu, tau, post, psd, shr = eb(s1["b"][m], s1["se"][m])
        r = float(np.corrcoef(s1["b"][m], s2["b"][m])[0, 1])
        lo, hi = boot_corr(s1["b"][m], s2["b"][m])
        sl_, sle, _ = wls(post, s2["b"][m], 1.0 / s2["se"][m] ** 2)
        fl = []
        for rep in range(0, int(nz["reps"]), max(1, int(nz["reps"]) // 8)):
            x1, e1, o1 = sc_slice(nz, oz, a[0], a[1], rep)
            x2, e2, o2 = sc_slice(nz, oz, b_[0], b_[1], rep)
            mm = o1 & o2
            if mm.sum() > 15:
                fl.append(float(np.corrcoef(x1[mm], x2[mm])[0, 1]))
        floor = f"{np.mean(fl):+.3f}±{np.std(fl):.3f}" if fl else "n/a"
        say(f"    {label:<32}{m.sum():>6}{r:>8.3f}   [{lo:+.3f}, {hi:+.3f}]"
            f"{floor:>14}{sl_:>7.2f}")

    rel(("all", ("S",)), ("all", ("R",)), "serve vs return (disjoint)")
    rel(("pre26", ("S", "R")), ("y26", ("S", "R")), "2024-25 vs 2026")
    rel(("dbl", ("S", "R")), ("sgl", ("S", "R")), "doubles vs singles", 300)
    say("    calib = weighted regression of the held-out slice on the SHRUNK")
    say("    estimate from the other. 1.00 = honestly scaled.")
    say()

    # ---- leaderboard --------------------------------------------------
    b, se = A["b"][ok], A["se"][ok]
    mu, tau, post, psd, shr = eb(b, se)
    uu = uuids[ok]
    nr = A["n"][ok]
    ssl = A["SSL"][ok]
    nam = np.array([nm.get(u, u[:8]) for u in uu])
    gnd = np.array([gd.get(u, "") for u in uu])
    v = np.array([val.get(u, np.nan) for u in uu])
    have = np.isfinite(v)

    say("[4] IS IT JUST BEING GOOD?")
    say(f"    corr(raw slope, v2 rating)       = "
        f"{np.corrcoef(A['b_obs'][ok][have], v[have])[0,1]:+.3f}")
    say(f"    corr(simulated artifact, rating) = "
        f"{np.corrcoef(A['b_mech'][ok][have], v[have])[0,1]:+.3f}")
    say(f"    corr(CORRECTED clutch, rating)   = "
        f"{np.corrcoef(post[have], v[have])[0,1]:+.3f}   (n={have.sum()})")
    say()

    # Win probability actually added. Winning a rally rather than losing it
    # moves the game by exactly that rally's leverage, so the expected win
    # probability a player adds is sum_r (extra P(win r)) * leverage_r
    # = b * sum_r (L_r - Lbar) * L_r = b * SSL, in raw leverage units.
    gz = np.load(DATA / "clutch_games.npz", allow_pickle=True)
    gmap = dict(zip(gz["uuids"], gz["games"]))
    games = np.maximum(np.array([gmap.get(u, np.nan) for u in uu]), 1.0)
    cwa100 = post * ssl * LEV_SD / games * 100.0

    say("[5] LEADERBOARD — shrunk clutch, players with >=3,000 rallies")
    say("    CWA/100 = extra GAMES won per 100 played, purely from winning")
    say("    big points instead of small ones at the same total points.")

    def table(sel, title, top=15, asc=False):
        say()
        say(title)
        say(f"    {'#':>2} {'player':<24}{'rallies':>9}{'clutch':>9}"
            f"{'95% CI':>20}{'CWA/100':>9}")
        idx = np.argsort(post[sel]) if asc else np.argsort(-post[sel])
        for rank, i in enumerate(np.where(sel)[0][idx][:top], 1):
            lo, hi = post[i] - 1.96 * psd[i], post[i] + 1.96 * psd[i]
            say(f"    {rank:>2} {nam[i]:<24}{int(nr[i]):>9,}{post[i]:>+9.4f}"
                f"   [{lo:+.4f},{hi:+.4f}]{cwa100[i]:>+9.1f}")

    big = nr >= 3000
    W = np.isin(gnd, ("F", "W"))
    table(big & (gnd == "M"), "    MEN — most clutch")
    table(big & W, "    WOMEN — most clutch")
    table(big & (gnd == "M"), "    MEN — least clutch", top=5, asc=True)
    table(big & W, "    WOMEN — least clutch", top=5, asc=True)
    say()

    sig = np.abs(post) > 1.96 * psd
    say(f"[6] {int(sig.sum())} of {len(post)} players are distinguishable from "
        f"league-average clutch at 95%.")
    say(f"    Largest |shrunk| effect: {np.max(np.abs(post)):.4f} "
        f"({nam[np.argmax(np.abs(post))]}).")

    with open(DATA / "clutch_leverage.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "name", "gender", "rallies", "b_observed",
                    "b_mechanical", "b_adjusted", "se", "z", "clutch_shrunk",
                    "shrunk_sd", "cwa_per_100_games", "v2_value"])
        for i in np.argsort(-post):
            w.writerow([uu[i], nam[i], gnd[i], int(nr[i]),
                        f"{A['b_obs'][ok][i]:.6f}", f"{A['b_mech'][ok][i]:.6f}",
                        f"{b[i]:.6f}", f"{se[i]:.6f}", f"{b[i]/se[i]:.3f}",
                        f"{post[i]:.6f}", f"{psd[i]:.6f}", f"{cwa100[i]:.2f}",
                        "" if not np.isfinite(v[i]) else f"{v[i]:.4f}"])
    with open(ROOT / "model" / "clutch_leverage_summary.json", "w") as fh:
        json.dump({"tau": {k: {"tau": t[0], "ci": [t[1], t[2]]}
                           for k, t in taus.items()},
                   "calibration": A["cal"], "reps": int(nz["reps"]),
                   "n_players": int(ok.sum()),
                   "corr_obs_vs_artifact": float(r_mech)}, fh, indent=1)
    say()
    say("wrote data/clutch_leverage.csv, model/clutch_leverage_summary.json")
    return lines


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", default=NULL,
                    help="clutch_null (flat rates) or clutch_null_model")
    main(**vars(ap.parse_args()))
