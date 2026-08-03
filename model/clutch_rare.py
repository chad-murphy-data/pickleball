"""Is clutch a RARE trait? Tests aimed at a minority, not at the field.

The distributional tests elsewhere in this thread (var(z), tau) ask whether
the WHOLE field is wider than chance. If clutch is something a handful of
players have and nobody else does, those tests are the wrong instrument:
800 zero players dilute 40 real ones, and a Gaussian empirical-Bayes prior —
which asserts that everybody has a little — shrinks genuine outliers far too
hard. A population where 5% of players carry +0.03 and 95% carry exactly 0
has tau = sqrt(0.05 * 0.03^2) = 0.0067, which is indistinguishable, as a
single summary number, from everyone carrying a uniform 0.0067.

So ask the rare-trait questions directly.

  1. SPIKE AND SLAB. Fit b_p ~ (1-pi) * delta_0 + pi * N(mu_s, sigma^2),
     observed through known noise se_p. pi is the fraction of players who
     have any clutch at all; the per-player posterior P(slab | data) says
     who. Fitted on the null replicates too, because a misspecified null or
     a heavy-tailed noise distribution can manufacture a nonzero pi.

  2. TAIL COUNT. How many players clear z > 2, against the distribution of
     that same count over no-clutch seasons. This is the "13% are
     left-handed" test: it does not care whether the bulk of the field is
     spread out.

  3. SELECT THEN VERIFY. The only test that establishes individuals: name
     the top K on 2024-25 alone, then measure that fixed group on 2026
     alone, and compare against the identical procedure run on no-clutch
     seasons. Selection plus regression-to-the-mean produces a specific
     signature under the null; beating that signature is what "these
     particular players are clutch" means.

Needs data/clutch_team_raw.npz (written by model/clutch_team.py).
Run:  python model/clutch_rare.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "model"))

from clutch_leverage import names  # noqa: E402

warnings.filterwarnings("ignore")


def load():
    return np.load(DATA / "clutch_team_raw.npz", allow_pickle=True)


def stats(z, arm, min_games, rep=None):
    """Corrected b and se for one arm. rep=None -> the real season;
    rep=i -> replicate i treated as observed, the others as its null."""
    U_r, S_r, V_r = (z[f"rep_{arm}_{k}"] for k in ("U", "SSL", "V"))
    reps = U_r.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        br = np.where(S_r > 0, U_r / np.where(S_r > 0, S_r, 1), np.nan)
    if rep is None:
        U, S, V, G = (z[f"obs_{arm}_{k}"] for k in ("U", "SSL", "V", "games"))
        others = np.arange(reps)
    else:
        U, S, V = U_r[rep], S_r[rep], V_r[rep]
        G = z[f"rep_{arm}_games"][rep]
        others = np.array([i for i in range(reps) if i != rep])
    ok = (S > 0) & (V > 0) & (G >= min_games) & np.isfinite(br).all(axis=0)
    b = np.where(ok, U / np.where(ok, S, 1), np.nan)
    bm = np.nanmean(br[others], axis=0)
    sdm = np.nanstd(br[others], axis=0, ddof=1)
    se = np.sqrt((np.sqrt(V) / np.where(ok, S, 1)) ** 2 + sdm ** 2 / len(others))
    return b - bm, se, ok


# ------------------------------------------------------- spike and slab --


def spike_slab(b, se):
    """ML fit of (pi, mu_s, sigma). Returns params + per-player posterior."""
    def nll(th):
        pi = 1.0 / (1.0 + np.exp(-th[0]))
        mu, sig = th[1], np.exp(th[2])
        f0 = norm.pdf(b, 0.0, se)
        f1 = norm.pdf(b, mu, np.sqrt(se ** 2 + sig ** 2))
        return -np.sum(np.log(np.maximum((1 - pi) * f0 + pi * f1, 1e-300)))

    best, bx = None, None
    for p0 in (-3.0, -1.5, 0.0):
        for s0 in (np.log(0.005), np.log(0.015)):
            r = minimize(nll, [p0, 0.0, s0], method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-8,
                                  "fatol": 1e-8})
            if best is None or r.fun < best:
                best, bx = r.fun, r.x
    pi = 1.0 / (1.0 + np.exp(-bx[0]))
    mu, sig = bx[1], np.exp(bx[2])
    f0 = norm.pdf(b, 0.0, se)
    f1 = norm.pdf(b, mu, np.sqrt(se ** 2 + sig ** 2))
    post = pi * f1 / np.maximum((1 - pi) * f0 + pi * f1, 1e-300)
    # likelihood-ratio vs pi = 0
    ll0 = np.sum(np.log(np.maximum(norm.pdf(b, 0.0, se), 1e-300)))
    return {"pi": pi, "mu": mu, "sigma": sig, "post": post,
            "lr": 2 * (-best - ll0)}


def main(min_games=60, min_games_era=30):
    z = load()
    uu = z["uuids"]
    nm, gd = names()
    reps = int(z["reps"])

    print("=" * 76)
    print("IS CLUTCH A RARE TRAIT? — tests aimed at a minority")
    print("=" * 76)

    # ---- 1. spike and slab ------------------------------------------
    print("\n[1] SPIKE AND SLAB — what FRACTION of players have any clutch?")
    print(f"    {'arm':<12}{'players':>8}{'pi':>8}{'slab mu':>10}"
          f"{'slab sd':>10}{'LR stat':>10}{'null pi':>16}")
    fits = {}
    for arm in ("doubles", "singles"):
        b, se, ok = stats(z, arm, min_games)
        f = spike_slab(b[ok], se[ok])
        fits[arm] = (f, ok)
        npi, nlr = [], []
        for r in range(min(reps, 12)):
            bn, sn, okn = stats(z, arm, min_games, rep=r)
            fn = spike_slab(bn[okn], sn[okn])
            npi.append(fn["pi"])
            nlr.append(fn["lr"])
        print(f"    {arm:<12}{ok.sum():>8}{f['pi']:>8.3f}{f['mu']:>+10.4f}"
              f"{f['sigma']:>10.4f}{f['lr']:>10.1f}"
              f"   {np.mean(npi):.3f}±{np.std(npi):.3f}")
        print(f"    {'':<12}{'':>8}{'':>8}{'':>10}{'':>10}"
              f"{'null LR:':>10}   {np.mean(nlr):.1f}±{np.std(nlr):.1f}"
              f"   (obs {f['lr']:.1f})")

    # ---- 2. tail counts ---------------------------------------------
    print("\n[2] TAIL COUNT — how many players clear the bar, vs no-clutch seasons")
    print(f"    {'arm':<12}{'bar':>8}{'observed':>10}{'null median':>13}"
          f"{'null 95%':>10}{'p':>7}")
    for arm in ("doubles", "singles"):
        b, se, ok = stats(z, arm, min_games)
        zz = (b / se)[ok]
        for bar in (2.0, 2.5, 3.0):
            obs_n = int((zz > bar).sum())
            nulls = []
            for r in range(reps):
                bn, sn, okn = stats(z, arm, min_games, rep=r)
                nulls.append(int(((bn / sn)[okn] > bar).sum()))
            nulls = np.array(nulls)
            p = float((nulls >= obs_n).mean())
            ptxt = f"<{1.0/len(nulls):.3f}" if p == 0 else f"{p:.3f}"
            print(f"    {arm:<12}{'z>'+str(bar):>8}{obs_n:>10}"
                  f"{np.median(nulls):>13.0f}{np.percentile(nulls,95):>10.0f}"
                  f"{ptxt:>7}")

    # ---- 3. select then verify --------------------------------------
    print("\n[3] SELECT THEN VERIFY — name them on 2024-25, test them on 2026")
    print("    (verification statistic = precision-weighted mean of the")
    print("     selected group's 2026 clutch, in sd units of its own noise)")
    print(f"    {'arm':<8}{'K':>4}{'obs z':>9}{'null mean':>11}"
          f"{'null sd':>9}{'p':>7}{'n avail':>9}")
    for arm, tag in (("doubles", "dbl"), ("singles", "sgl")):
        b1, s1, o1 = stats(z, f"{tag}_pre26", min_games_era)
        b2, s2, o2 = stats(z, f"{tag}_y26", min_games_era)
        both = o1 & o2
        idx = np.where(both)[0]
        if len(idx) < 20:
            print(f"    {arm:<8}  too few players ({len(idx)})")
            continue

        def verify(bb1, ss1, bb2, ss2, K):
            order = np.argsort(-(bb1 / ss1))
            sel = order[:K]
            w = 1.0 / ss2[sel] ** 2
            est = np.sum(w * bb2[sel]) / np.sum(w)
            se = 1.0 / np.sqrt(np.sum(w))
            return est / se

        for K in (5, 10, 20, 40):
            if K > len(idx):
                continue
            obs_z = verify(b1[idx], s1[idx], b2[idx], s2[idx], K)
            nl = []
            for r in range(reps):
                n1, m1, k1 = stats(z, f"{tag}_pre26", min_games_era, rep=r)
                n2, m2, k2 = stats(z, f"{tag}_y26", min_games_era, rep=r)
                kk = k1 & k2
                j = np.where(kk)[0]
                if len(j) < K:
                    continue
                nl.append(verify(n1[j], m1[j], n2[j], m2[j], K))
            nl = np.array(nl)
            p = float((nl >= obs_z).mean())
            ptxt = f"<{1.0/len(nl):.3f}" if p == 0 else f"{p:.3f}"
            print(f"    {arm:<8}{K:>4}{obs_z:>9.2f}{np.mean(nl):>11.2f}"
                  f"{np.std(nl):>9.2f}{ptxt:>7}{len(idx):>9}")

    # ---- 4. who ------------------------------------------------------
    print("\n[4] WHO — posterior probability the player has ANY clutch")
    for arm in ("doubles", "singles"):
        f, ok = fits[arm]
        b, se, _ = stats(z, arm, min_games)
        uu_a, post = uu[ok], f["post"]
        g = z[f"obs_{arm}_games"][ok]
        order = np.argsort(-post)
        print(f"\n    {arm}:")
        print(f"    {'player':<24}{'games':>7}{'clutch':>9}{'z':>7}{'P(clutch)':>11}")
        for i in order[:10]:
            print(f"    {nm.get(uu_a[i], uu_a[i][:8]):<24}{int(g[i]):>7}"
                  f"{b[ok][i]:>+9.4f}{(b/se)[ok][i]:>7.2f}{post[i]:>11.2f}")
    print(f"\n    pi implies about {fits['doubles'][0]['pi']*fits['doubles'][1].sum():.0f} "
          f"doubles players with a real effect.")


if __name__ == "__main__":
    main()
