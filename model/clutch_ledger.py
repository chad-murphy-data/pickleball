"""The clutch LEDGER — what happened, not what will happen.

There are two different questions hiding under the word clutch, and they take
different statistics:

  TRAIT   "Is this player clutch?" — a forecast. It has to replicate, it has
          to be shrunk toward the field, and on this archive only Ben Johns
          and Anna Leigh Waters survive it (`clutch_rare.py`).

  LEDGER  "How much did this player actually win in big moments?" — a record.
          It happened. Demanding that a record replicate is a category error:
          nobody asks whether an RBI total is repeatable before printing it.
          Same family as the MLP MVP award (CLAUDE.md finding 9), which is
          deliberately pure outcome accounting.

This file is the ledger. It reports CWPA — Clutch Win Probability Added.

    CWPA = sum over the player's games of  sum_r (L_r - Lbar_g)(o_r - obar_g)

with L in RAW win-probability units (the swing the rally actually put on the
game) and o = 1 if the player's side won the rally. Because a rally's leverage
IS the win probability it moves, this sum is denominated in games: +1.0 means
the player's side banked one full game's worth of win probability purely from
*when* its rally wins landed, at the same total rallies won.

Two deliberate choices, and they are the whole design:

  * NO SHRINKAGE. Shrinking is how you estimate a latent trait from a noisy
    sample. A ledger is not an estimate of anything — it is the thing that
    happened, so it is reported at face value.

  * STILL BASELINE-CORRECTED. Side-out scoring makes every service run end in
    a loss at its highest score, which manufactures big-point covariance for
    free and hands more of it to better servers (`clutch_mechanical.py`).
    Leaving that in would make this a serve-rate leaderboard wearing a clutch
    hat. Subtracting the no-clutch simulation's mean for each player's own
    schedule is not shrinkage toward the field — it sets where zero is.

Run:  python model/clutch_ledger.py       (needs data/clutch_team_raw.npz)
Writes data/clutch_ledger.csv.
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "model"))

from clutch_rare import load                      # noqa: E402
from clutch_leverage import names, v2_values      # noqa: E402

warnings.filterwarnings("ignore")

LEV_SD = 0.0659      # raw leverage sd; the stored statistic is in sd units


def ledger(z, arm, min_games):
    """CWPA in games of win probability, and its per-100-game rate."""
    U = z[f"obs_{arm}_U"]
    G = z[f"obs_{arm}_games"]
    Ur = z[f"rep_{arm}_U"]
    S = z[f"obs_{arm}_SSL"]
    V = z[f"obs_{arm}_V"]
    # Scale each replicate's baseline to the player's OWN leverage
    # sum-of-squares before averaging. U grows with SSL, and a player whose
    # real games ran shorter than their simulated twins' (blowouts: Waters)
    # would otherwise be charged a baseline they never had the chance to earn.
    Sr = z[f"rep_{arm}_SSL"]
    with np.errstate(invalid="ignore", divide="ignore"):
        base = np.nanmean(np.where(Sr > 0, Ur / np.where(Sr > 0, Sr, 1),
                                   np.nan), axis=0) * S
    cwpa = (U - base) * LEV_SD
    # noise scale, for context only -- it does not move the number
    se = np.sqrt(V) * LEV_SD
    ok = (G >= min_games) & (S > 0) & (V > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per100 = np.where(G > 0, cwpa / np.maximum(G, 1) * 100.0, np.nan)
    return cwpa, per100, se, G, ok


def show(z, arm, min_games, label, top=15):
    cwpa, per100, se, G, ok = ledger(z, arm, min_games)
    nm, gd = names()
    uu = z["uuids"]
    idx = np.where(ok)[0]
    order = idx[np.argsort(-cwpa[idx])]
    print(f"\n{label} — {ok.sum()} players with >= {min_games} games")
    print(f"{'':<3}{'player':<24}{'sex':>4}{'games':>7}{'CWPA':>8}"
          f"{'per 100':>9}{'+/- noise':>11}")
    for r, i in enumerate(order[:top], 1):
        sx = {"F": "W"}.get(gd.get(uu[i], ""), gd.get(uu[i], ""))
        print(f"{r:>2}. {nm.get(uu[i], uu[i][:8]):<24}{sx:>4}{int(G[i]):>7}"
              f"{cwpa[i]:>+8.2f}{per100[i]:>+9.2f}{se[i]:>11.2f}")
    return cwpa, per100, se, G, ok


def main():
    z = load()
    nm, gd = names()
    val = v2_values()
    print("=" * 72)
    print("THE CLUTCH LEDGER — win probability banked in big moments")
    print("CWPA is denominated in GAMES. +1.0 = one full game's worth of win")
    print("probability, earned purely from WHEN the rally wins landed.")
    print("Not shrunk, not a forecast: this is the record of 2024-01 to 2026-07.")
    print("=" * 72)

    cd, pd_, sd, Gd, okd = show(z, "doubles", 200, "DOUBLES (career)")
    show(z, "dbl_y26", 60, "DOUBLES (2026 season only)", top=10)
    cs, ps, ss, Gs, oks = show(z, "singles", 150, "SINGLES (career)", top=10)

    print("\nrate leaders, doubles, >= 400 games (per 100 games):")
    m = okd & (Gd >= 400)
    for r, i in enumerate(np.where(m)[0][np.argsort(-pd_[np.where(m)[0]])][:6], 1):
        print(f"  {r}. {nm.get(z['uuids'][i], ''):<24}{pd_[i]:>+7.2f}"
              f"   ({int(Gd[i])} games, {cd[i]:+.2f} total)")

    with open(DATA / "clutch_ledger.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "name", "gender", "discipline", "games",
                    "cwpa_games", "cwpa_per_100_games", "noise_sd",
                    "v2_value"])
        for arm, disc, mg in (("doubles", "doubles", 60),
                              ("singles", "singles", 60)):
            c, p, s, G, ok = ledger(z, arm, mg)
            for i in np.where(ok)[0][np.argsort(-c[np.where(ok)[0]])]:
                u = z["uuids"][i]
                w.writerow([u, nm.get(u, u[:8]), gd.get(u, ""), disc,
                            int(G[i]), f"{c[i]:.3f}", f"{p[i]:.3f}",
                            f"{s[i]:.3f}",
                            f"{val[u]:.4f}" if u in val else ""])
    print("\nwrote data/clutch_ledger.csv")


if __name__ == "__main__":
    main()
