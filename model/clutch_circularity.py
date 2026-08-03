"""Two objections to the null, tested rather than argued.

OBJECTION 1 — "the null is circular". The no-clutch null gives each simulated
player an ability fitted from the same rallies that would contain their
clutch. If a player really is clutch, their fitted rate is already inflated
by it, so the null's twin partly reproduces the thing we are trying to
detect, and we subtract real signal along with the artifact. `clutch_power.py`
does NOT test this: it fits abilities on the real archive and then adds
clutch on top, so its null never sees the contamination.

  Experiment A: inject clutch of known size, then REFIT the ability model on
  the injected season and build the null from those contaminated abilities —
  exactly the circle the real analysis runs in. Compare recovery against the
  clean-ability null. The gap is the cost of the circularity.

OBJECTION 2 — "server/receiver attribution is meaningless in doubles". A
doubles rally is contested by four players; crediting it to whoever served or
returned is close to arbitrary. If clutch is a property of the TEAM (or of
all four players jointly), an estimator that assigns the rally to one of them
sees a diluted version.

  Experiment B: inject clutch as a team-level effect in doubles — the serving
  PAIR's mean coefficient against the receiving PAIR's — and ask how much the
  server/receiver-attributed estimator recovers. Compare with the same
  effect injected through the individual channel, and with singles, where
  attribution is exact by construction.

Run:  python model/clutch_circularity.py
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

import clutch_leverage as cl      # noqa: E402
import clutch_null as cn          # noqa: E402
from clutch_leverage import eb, tau_ci  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
LEV_MEAN_RAW = 0.1347


def sim(games, rng, tabs, model, kappa, lev_mean_z, mode):
    """mode: 'indiv'  -> server's own kappa vs receiver's own
             'team'   -> serving pair's mean kappa vs receiving pair's mean
             'none'   -> no clutch at all
    """
    srv, rcv, gid, lev, won = [], [], [], [], []
    for (g, disc, T, s0, s1) in games:
        team = (s0, s1)
        cap = T + cn.CAP_EXTRA
        tab = tabs[(disc, T)]
        sc_ = [0, 0]
        s = int(rng.integers(2))
        doubles = disc == "doubles"
        sn = 2 if doubles else 0
        who = 0
        rpos = [int(rng.integers(2)), int(rng.integers(2))]
        kt = (kappa[s0].mean(), kappa[s1].mean()) if doubles else (0.0, 0.0)
        for _ in range(400):
            a, b = sc_[s], sc_[1 - s]
            if (a >= T and a - b >= 2) or (b >= T and b - a >= 2):
                break
            if a >= cap or b >= cap:
                break
            server = team[s][who] if doubles else team[s][0]
            receiver = team[1 - s][rpos[1 - s]] if doubles else team[1 - s][0]
            lz = tab[(a, b, sn)]
            eta = model[0][server] - model[1][receiver]
            if mode == "indiv":
                eta += (kappa[server] - kappa[receiver]) * (lz - lev_mean_z)
            elif mode == "team":
                if doubles:
                    eta += (kt[s] - kt[1 - s]) * (lz - lev_mean_z)
                else:
                    eta += (kappa[server] - kappa[receiver]) * (lz - lev_mean_z)
            w = 1 if rng.random() < 1.0 / (1.0 + np.exp(-eta)) else 0
            srv.append(server); rcv.append(receiver); gid.append(g)
            lev.append(lz); won.append(w)
            if w:
                sc_[s] += 1
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


def slices_of(srv, rcv, gid, lev, won, gmask, npl, ng, keys):
    out = {}
    for nm in keys:
        m = gmask[nm][gid]
        for ch, who, y in (("S", srv, won), ("R", rcv, 1.0 - won)):
            u, s_, v_, n_ = cn.channel_U(who[m], gid[m], lev[m], y[m], npl, ng)
            out[f"U_{nm}_{ch}"], out[f"SSL_{nm}_{ch}"] = u, s_
            out[f"V_{nm}_{ch}"], out[f"n_{nm}_{ch}"] = v_, n_
    return out


def recover(obs, nulls, sl, chans, min_rallies):
    U = sum(obs[f"U_{sl}_{c}"] for c in chans)
    S = sum(obs[f"SSL_{sl}_{c}"] for c in chans)
    V = sum(obs[f"V_{sl}_{c}"] for c in chans)
    n = sum(obs[f"n_{sl}_{c}"] for c in chans)
    Ur = np.array([sum(o[f"U_{sl}_{c}"] for c in chans) for o in nulls])
    Sr = np.array([sum(o[f"SSL_{sl}_{c}"] for c in chans) for o in nulls])
    ok = (S > 0) & (V > 0) & (n >= min_rallies) & (Sr.min(axis=0) > 0)
    b = np.where(ok, U / np.where(ok, S, 1), np.nan)
    br = np.where(ok[None, :], Ur / np.where(ok[None, :], Sr, 1), np.nan)
    bm, sdm = np.nanmean(br, axis=0), np.nanstd(br, axis=0, ddof=1)
    se = np.sqrt((np.sqrt(V) / np.where(ok, S, 1)) ** 2 + sdm ** 2 / len(nulls))
    mu, tau, post, psd, shr = eb((b - bm)[ok], se[ok])
    return tau, int((np.abs(post) > 1.96 * psd).sum()), int(ok.sum())


def main(reps=20, tau_inject=0.010, seed=20260803):
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
    lmz = LEV_MEAN_RAW / lev_sd
    base = cn.fit_serve_model(d, F, npl)

    gdisc = np.empty(ng, dtype="<U8")
    gdisc[d["gidx"]] = d["disc"]
    gmask = {"all": np.ones(ng, bool), "dbl": gdisc == "doubles",
             "sgl": gdisc == "singles"}
    keys = list(gmask)
    rng = np.random.default_rng(seed)

    print("=" * 76)
    print(f"OBJECTION TESTS — injected tau = {tau_inject}, {reps} null replicates")
    print("=" * 76)

    def build_null(model, tag):
        out = []
        for r in range(reps):
            s_, rc, g_, l_, w_ = sim(games, rng, tabs, model, np.zeros(npl),
                                     lmz, "none")
            out.append(slices_of(s_, rc, g_, l_, w_, gmask, npl, ng, keys))
            if (r + 1) % 5 == 0:
                print(f"    {tag} null {r+1}/{reps}", flush=True)
        return out

    print("\n[A] CIRCULARITY — does fitting the null's abilities on")
    print("    clutch-contaminated data destroy detection?")
    kappa = rng.normal(0, tau_inject / 0.249, npl)
    srv, rcv, gid, lev, won = sim(games, rng, tabs, base, kappa, lmz, "indiv")
    obs = slices_of(srv, rcv, gid, lev, won, gmask, npl, ng, keys)

    print("    fitting abilities ON the injected (contaminated) season ...")
    dd = dict(d)
    dd["won"] = won.astype(np.int8)
    FF = type("F", (), {})()
    FF.srv_code, FF.rcv_code = srv, rcv
    refit = cn.fit_serve_model({"won": won.astype(np.int8)}, FF, npl)

    clean = build_null(base, "clean")
    dirty = build_null(refit, "refit")
    print()
    print(f"    {'null abilities':<34}{'recovered tau':>15}{'n sig':>8}")
    for lab, nl in (("clean (uncontaminated)", clean),
                    ("REFIT on contaminated season", dirty)):
        t, ns, np_ = recover(obs, nl, "all", ("S", "R"), 400)
        print(f"    {lab:<34}{t:>15.5f}{ns:>8}")
    print(f"    injected                          {tau_inject:>15.5f}")

    print("\n[B] ATTRIBUTION — is team-level clutch visible through a")
    print("    server/receiver-attributed estimator?")
    kappa2 = rng.normal(0, tau_inject / 0.249, npl)
    print(f"    {'injection':<22}{'arm':<10}{'recovered tau':>15}{'n sig':>8}"
          f"{'players':>9}")
    for mode in ("indiv", "team"):
        s_, rc, g_, l_, w_ = sim(games, rng, tabs, base, kappa2, lmz, mode)
        o = slices_of(s_, rc, g_, l_, w_, gmask, npl, ng, keys)
        for arm, mr in (("doubles", 400), ("singles", 300)):
            sl = "dbl" if arm == "doubles" else "sgl"
            t, ns, np_ = recover(o, clean, sl, ("S", "R"), mr)
            print(f"    {mode:<22}{arm:<10}{t:>15.5f}{ns:>8}{np_:>9}")
    print(f"\n    injected tau = {tau_inject}. 'team' means the serving PAIR's")
    print("    mean coefficient acts, so no individual owns the rally.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tau-inject", type=float, default=0.010, dest="tau_inject")
    main(**vars(ap.parse_args()))
