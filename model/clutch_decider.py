"""Do players elevate in DECIDERS? — the between-game question.

Everything else in this thread measures clutch *inside* a game: leverage is
demeaned within each game cell, so a player who lifts their whole level for a
tight match is invisible to it by construction. This file asks the other
question. A PPA best-of-three that reaches 1-1 plays a third game for the
match; 3,895 of them exist in the archive. Does anyone play that game better
than their own baseline?

Statistic. Per game, the team's residual is observed point share minus the v2
expectation (weakest-link team value plus dyad chemistry, no knowledge of how
the match has gone). Per player,

    D = mean residual in their DECIDERS - mean residual in their other games

which is a within-player contrast, so it cannot be moved by a player's level,
by a stale rating, or by which events they enter. Both partners of a side get
the same residual — a game is a team outcome, the same call
`clutch_team.py` makes — so individuals separate only through partner
rotation, and a pair leaderboard is reported alongside.

THE TRAP, and why this needs a simulated null. Deciders are not a random
sample of games: a match only reaches 1-1 if the two teams split, which is
evidence about the match-level random effect the model already knows exists
(sd 0.352 on eta). A heavy favourite that split games 1-2 is having a bad
day, so its third-game residual is negative *on average with no clutch
anywhere*. That selection effect scales with how big a favourite the team
was — i.e. with skill — which is exactly the shape of artifact that has eaten
this analysis twice already.

So the null replays every real match: draw a match effect, simulate each game
as a race to T at the model's per-point probability, stop when a side has two
wins, and compute D the same way. Whatever it finds is selection, not clutch.

Run:  python model/clutch_decider.py [--reps 200]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))

from clutch_leverage import eb, tau_ci, boot_corr  # noqa: E402

warnings.filterwarnings("ignore")

GAMMA = -0.16596178710460663      # weakest-link gamma, v2 logit scale
MIN_DEC = 25                      # minimum deciders to be listed

# The null's variance components are CALIBRATED, not assumed. Feeding it the
# race model's nominal per-match sd of 0.352 reproduces neither the observed
# decider rate (0.219 simulated vs 0.281 real) nor the observed selection
# artifact (-0.213 vs -0.016) — it overstates the trap about thirteenfold,
# and subtracting an artifact that large manufactures clutch out of nothing.
# Two components are needed and they pull opposite ways: a per-MATCH effect
# makes a match's games agree, which LOWERS the decider rate and drives the
# skill-linked selection bias; a per-GAME effect makes them disagree, which
# RAISES the decider rate without biasing anything. Real pickleball needs
# mostly the second. calibrate() grid-searches both against the two observed
# quantities that govern the artifact.
SD_GRID_MATCH = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
SD_GRID_GAME = (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40)


def load():
    players, names = {}, {}
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        players[r["player_id"].lower()] = float(r["value_now_mean"])
        names[r["player_id"].lower()] = r["full_name"]
    chem = {}
    try:
        for r in csv.DictReader((DATA / "v2_dyads.csv").open()):
            chem[frozenset((r["p1_name"], r["p2_name"]))] = \
                float(r["chem_logit_mean"])
    except FileNotFoundError:
        pass

    matches = defaultdict(list)
    for g in csv.DictReader((DATA / "games.csv").open()):
        if g["is_forfeit"] != "False" or g["is_dreambreaker"] == "True":
            continue
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        if g["tour"] != "PPA" or g["best_of"] != "3":
            continue
        us = [g["t1_p1"].lower(), g["t1_p2"].lower(),
              g["t2_p1"].lower(), g["t2_p2"].lower()]
        if any(u not in players for u in us):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 5:
            continue
        matches[g["match_id"].lower()].append(dict(
            date=g["date"], gn=int(g["game_number"]), us=us, s1=s1, s2=s2,
            T=11 if g["scoring_format"] == "sideout_11" else 15))
    out = []
    for mid, gs in matches.items():
        gs.sort(key=lambda x: x["gn"])
        if len(gs) not in (2, 3):
            continue
        if len({tuple(x["us"]) for x in gs}) != 1:
            continue          # roster changed mid-match: not a clean bo3
        out.append((mid, gs))
    return out, players, names, chem


def eta_of(us, players, names, chem):
    v = [players[u] for u in us]
    t1 = v[0] + v[1] + GAMMA * abs(v[0] - v[1])
    t2 = v[2] + v[3] + GAMMA * abs(v[2] - v[3])
    c1 = chem.get(frozenset((names[us[0]], names[us[1]])), 0.0)
    c2 = chem.get(frozenset((names[us[2]], names[us[3]])), 0.0)
    return t1 - t2 + c1 - c2


def sim_game(p, T, rng, cap=40):
    """Race to T, win by 2, per-point probability p. Returns team-1 share."""
    a = b = 0
    while True:
        if rng.random() < p:
            a += 1
        else:
            b += 1
        if (a >= T and a - b >= 2) or (b >= T and b - a >= 2):
            break
        if a >= cap or b >= cap:
            break
    return a / (a + b), a > b


def accumulate(rec, npl, idx):
    """rec: list of (us, resid_team1, is_decider). -> per-player sums."""
    sd_ = np.zeros(npl)
    nd_ = np.zeros(npl)
    so_ = np.zeros(npl)
    no_ = np.zeros(npl)
    for us, r, dec in rec:
        for j, u in enumerate(us):
            i = idx[u]
            v = r if j < 2 else -r
            if dec:
                sd_[i] += v
                nd_[i] += 1
            else:
                so_[i] += v
                no_[i] += 1
    return sd_, nd_, so_, no_


def contrast(sd_, nd_, so_, no_, min_dec):
    ok = (nd_ >= min_dec) & (no_ >= min_dec)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(ok, sd_ / np.where(nd_ > 0, nd_, 1)
                     - so_ / np.where(no_ > 0, no_, 1), np.nan)
    return D, ok


def simulate_season(rows, sdm, sdg, rng):
    """One synthetic season. rows = [(us, eta, T)]. Returns records + stats."""
    rec, dec_flag, dec_res, dec_eta, all_res = [], [], [], [], []
    for us, eta, T in rows:
        m = rng.normal(0.0, sdm) if sdm > 0 else 0.0
        p_ref = 1.0 / (1.0 + np.exp(-eta))
        wins, shares = [0, 0], []
        while max(wins) < 2 and len(shares) < 3:
            gg = rng.normal(0.0, sdg) if sdg > 0 else 0.0
            share, w1 = sim_game(1.0 / (1.0 + np.exp(-(eta + m + gg))), T, rng)
            wins[0 if w1 else 1] += 1
            shares.append(share)
        dec_i = 2 if len(shares) == 3 else None
        for k, share in enumerate(shares):
            rec.append((us, share - p_ref, k == dec_i))
            all_res.append(share - p_ref)
        dec_flag.append(len(shares) == 3)
        if dec_i is not None:
            dec_res.append(shares[2] - p_ref)
            dec_eta.append(eta)
    return (rec, float(np.mean(dec_flag)),
            float(np.corrcoef(dec_res, dec_eta)[0, 1]), float(np.std(all_res)))


def calibrate(rows, targets, rng, passes=3):
    """Pick (sd_match, sd_game) reproducing the observed decider rate and the
    observed corr(decider residual, eta). A null that cannot reproduce the
    data's own selection structure has no business correcting it."""
    t_rate, t_corr, t_sd = targets
    best = None
    for sdm in SD_GRID_MATCH:
        for sdg in SD_GRID_GAME:
            rates, corrs, sds = [], [], []
            for _ in range(passes):
                _, rt, cr, sdv = simulate_season(rows, sdm, sdg, rng)
                rates.append(rt); corrs.append(cr); sds.append(sdv)
            rate, corr, sdv = np.mean(rates), np.mean(corrs), np.mean(sds)
            # both targets matter; scale each by a plausible tolerance
            loss = ((rate - t_rate) / 0.010) ** 2 + ((corr - t_corr) / 0.030) ** 2
            if best is None or loss < best[0]:
                best = (loss, sdm, sdg, rate, corr, sdv)
    return best


def main(reps=200, seed=31337, min_dec=MIN_DEC):
    matches, players, names, chem = load()
    uu = sorted({u for _, gs in matches for u in gs[0]["us"]})
    idx = {u: i for i, u in enumerate(uu)}
    npl = len(uu)
    print(f"{len(matches):,} clean PPA best-of-3 matches, {npl:,} players")
    n3 = sum(1 for _, gs in matches if len(gs) == 3)
    print(f"{n3:,} went to a decider ({100*n3/len(matches):.1f}%)")

    # ---- observed -----------------------------------------------------
    obs, meta = [], []
    for mid, gs in matches:
        eta = eta_of(gs[0]["us"], players, names, chem)
        p = 1.0 / (1.0 + np.exp(-eta))
        dec_i = 2 if len(gs) == 3 else None
        for k, g in enumerate(gs):
            share = g["s1"] / (g["s1"] + g["s2"])
            obs.append((g["us"], share - p, k == dec_i))
        meta.append((gs[0]["us"], eta, gs[0]["T"], gs[0]["date"]))
    sd_, nd_, so_, no_ = accumulate(obs, npl, idx)
    D_obs, ok = contrast(sd_, nd_, so_, no_, min_dec)

    # ---- calibrate the null against the data's own structure ----------
    rows = [(us, eta, T) for us, eta, T, _ in meta]
    o_rate = n3 / len(matches)
    dres = [r for us, r, dec in obs if dec]
    deta = [e for (us, e, T, _), (mid, gs) in zip(meta, matches) if len(gs) == 3]
    o_corr = float(np.corrcoef(dres, deta)[0, 1])
    o_sd = float(np.std([r for _, r, _ in obs]))
    rng = np.random.default_rng(seed)
    loss, sdm, sdg, c_rate, c_corr, c_sd = calibrate(
        rows, (o_rate, o_corr, o_sd), rng)
    print(f"\ncalibrated null: sd_match {sdm:.2f}, sd_game {sdg:.2f}")
    print(f"  decider rate      observed {o_rate:.3f}  simulated {c_rate:.3f}")
    print(f"  corr(resid, eta)  observed {o_corr:+.3f}  simulated {c_corr:+.3f}")
    print(f"  residual sd       observed {o_sd:.4f}  simulated {c_sd:.4f}"
          f"   -> null noise inflated x{o_sd/c_sd:.3f}")
    infl = o_sd / c_sd

    # ---- null ---------------------------------------------------------
    Dn = np.full((reps, npl), np.nan)
    for r in range(reps):
        rec, _, _, _ = simulate_season(rows, sdm, sdg, rng)
        a, b, c, d = accumulate(rec, npl, idx)
        Dn[r], _ = contrast(a, b, c, d, min_dec)
        if (r + 1) % 25 == 0:
            print(f"  replicate {r+1}/{reps}", flush=True)

    base = np.nanmean(Dn, axis=0)
    # the simulator is short on residual spread (imperfect expectations widen
    # the real residuals), so its per-player noise is scaled up to match.
    sdn = np.nanstd(Dn, axis=0, ddof=1) * infl
    adj = D_obs - base
    z = adj / sdn

    print("\n" + "=" * 74)
    print("ELEVATING IN DECIDERS — game 3 of a PPA best-of-three")
    print("=" * 74)
    m = ok & np.isfinite(z)
    print(f"players with >= {min_dec} deciders and >= {min_dec} others: {m.sum()}")
    print(f"\nselection artifact (no-clutch simulation):")
    print(f"  mean D under the null      {np.nanmean(base[m]):+.5f}")
    print(f"  sd of D across players     {np.nanstd(base[m]):.5f}")
    good = m & np.isfinite(base)
    v2v = np.array([players[u] for u in uu])
    print(f"  corr(null artifact, skill) {np.corrcoef(base[good], v2v[good])[0,1]:+.3f}"
          "   <- the trap, quantified")
    print(f"  corr(raw D, skill)         {np.corrcoef(D_obs[good], v2v[good])[0,1]:+.3f}")
    print(f"  corr(corrected D, skill)   {np.corrcoef(adj[good], v2v[good])[0,1]:+.3f}")

    print(f"\nis anyone elevating?")
    print(f"  var(z)  {np.nanvar(z[m]):.3f}   (chance 1.000)")
    mu, tau, post, psd, shr = eb(adj[m], sdn[m])
    lo, hi = tau_ci(adj[m], sdn[m])
    print(f"  tau     {tau:.5f}   95% CI [{lo:.5f}, {hi:.5f}]")
    print(f"  |z|>1.96: {int((np.abs(z[m])>1.96).sum())}  "
          f"(expected {0.05*m.sum():.0f})")
    print(f"  max z   {np.nanmax(z[m]):+.2f} / min {np.nanmin(z[m]):+.2f}")

    order = np.argsort(-z[m])
    ids = np.array(uu)[m]
    print(f"\n{'':<3}{'player':<26}{'deciders':>9}{'D (pts share)':>15}{'z':>8}")
    for r_, i in enumerate(order[:10], 1):
        print(f"{r_:>2}. {names[ids[i]]:<26}{int(nd_[m][i]):>9}"
              f"{adj[m][i]:>+15.4f}{z[m][i]:>8.2f}")
    print("   ...")
    for i in order[-3:]:
        print(f"    {names[ids[i]]:<27}{int(nd_[m][i]):>9}"
              f"{adj[m][i]:>+15.4f}{z[m][i]:>8.2f}")

    with open(DATA / "clutch_decider.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "name", "deciders", "other_games",
                    "D_raw", "D_null", "D_adjusted", "sd", "z", "v2_value"])
        for i in np.argsort(-np.where(m, z, -np.inf)):
            if not m[i]:
                continue
            w.writerow([uu[i], names[uu[i]], int(nd_[i]), int(no_[i]),
                        f"{D_obs[i]:.5f}", f"{base[i]:.5f}", f"{adj[i]:.5f}",
                        f"{sdn[i]:.5f}", f"{z[i]:.3f}", f"{players[uu[i]]:.4f}"])
    print("\nwrote data/clutch_decider.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--min-dec", type=int, default=MIN_DEC, dest="min_dec")
    main(**vars(ap.parse_args()))
