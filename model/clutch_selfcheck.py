"""Does the correction machinery return zero when the truth is zero?

Take one no-clutch replicate and treat it as if it were the real season:
correct it against the other replicates, shrink it, and run the exact same
existence and reliability tests. Every number should come back at its null
value (tau = 0, r = 0). Anything that does not is machinery error, and would
also be present in the real answer.

This checks the estimator, the closed-form se, the empirical-Bayes step and
the reliability tests. It does NOT check whether the simulator resembles pro
pickleball — that is what the --model robustness arm is for.

Run:  python model/clutch_selfcheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from clutch_leverage import eb, boot_corr, wls  # noqa: E402

MIN_RALLIES = 400


def slice_stats(nz, oz, sl, chans, r):
    """Replicate r as pseudo-observed; the other replicates as the null."""
    U = sum(nz[f"U_rep_{sl}_{c}"] for c in chans)
    S = sum(nz[f"SSL_rep_{sl}_{c}"] for c in chans)
    V = sum(nz[f"V_rep_{sl}_{c}"] for c in chans)
    n_obs = sum(oz[f"n_{sl}_{c}"] for c in chans) if sl != "all" \
        else sum(nz[f"n_obs_{c}"] for c in chans)
    reps = U.shape[0]
    others = [i for i in range(reps) if i != r]
    with np.errstate(invalid="ignore", divide="ignore"):
        br = np.where(S > 0, U / np.where(S > 0, S, 1), np.nan)
    ok = (S[r] > 0) & (V[r] > 0) & (n_obs >= MIN_RALLIES) & \
         np.isfinite(br).all(axis=0)
    b_obs = br[r]
    b_mech = np.nanmean(br[others], axis=0)
    sd_mech = np.nanstd(br[others], axis=0, ddof=1)
    se_perm = np.sqrt(V[r]) / np.where(S[r] > 0, S[r], 1)
    se = np.sqrt(se_perm ** 2 + sd_mech ** 2 / len(others))
    return b_obs - b_mech, se, ok


def main(n_trials=8):
    nz = np.load(ROOT / "data" / "clutch_null.npz", allow_pickle=True)
    oz = np.load(ROOT / "data" / "clutch_obs_year.npz", allow_pickle=True)
    reps = int(nz["reps"])
    rng = np.random.default_rng(3)
    trials = rng.choice(reps, size=min(n_trials, reps), replace=False)

    print("=" * 70)
    print("SELF-CHECK — running the pipeline on data with NO clutch in it")
    print("=" * 70)
    print(f"{'rep':>4}{'players':>9}{'var(z)':>9}{'tau':>10}"
          f"{'r(S,R)':>9}{'r(era)':>9}{'r(d,s)':>9}")
    taus, vz, rsr, rer, rds = [], [], [], [], []
    for r in trials:
        b, se, ok = slice_stats(nz, oz, "all", ("S", "R"), r)
        mu, tau, post, psd, shr = eb(b[ok], se[ok])
        z = (b / se)[ok]
        bs, ss, oks = slice_stats(nz, oz, "all", ("S",), r)
        br_, sr_, okr = slice_stats(nz, oz, "all", ("R",), r)
        m = oks & okr
        r1 = float(np.corrcoef(bs[m], br_[m])[0, 1])
        b1, s1, o1 = slice_stats(nz, oz, "pre26", ("S", "R"), r)
        b2, s2, o2 = slice_stats(nz, oz, "y26", ("S", "R"), r)
        m2 = o1 & o2
        r2 = float(np.corrcoef(b1[m2], b2[m2])[0, 1])
        b3, s3, o3 = slice_stats(nz, oz, "dbl", ("S", "R"), r)
        b4, s4, o4 = slice_stats(nz, oz, "sgl", ("S", "R"), r)
        m3 = o3 & o4
        r3 = float(np.corrcoef(b3[m3], b4[m3])[0, 1])
        taus.append(tau); vz.append(np.var(z))
        rsr.append(r1); rer.append(r2); rds.append(r3)
        print(f"{r:>4}{ok.sum():>9}{np.var(z):>9.3f}{tau:>10.5f}"
              f"{r1:>9.3f}{r2:>9.3f}{r3:>9.3f}")

    print("-" * 70)
    print(f"{'mean':>4}{'':>9}{np.mean(vz):>9.3f}{np.mean(taus):>10.5f}"
          f"{np.mean(rsr):>9.3f}{np.mean(rer):>9.3f}{np.mean(rds):>9.3f}")
    print(f"{'sd':>4}{'':>9}{np.std(vz):>9.3f}{np.std(taus):>10.5f}"
          f"{np.std(rsr):>9.3f}{np.std(rer):>9.3f}{np.std(rds):>9.3f}")
    print()
    print("Targets: var(z) = 1.000, tau = 0.00000, all r = 0.000.")
    print("Residual tau here is the floor below which a real tau means nothing.")


if __name__ == "__main__":
    main()
