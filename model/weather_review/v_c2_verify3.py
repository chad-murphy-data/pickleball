"""C2 verification, part 3: attenuation and subsample robustness of the
wind coefficient — the brief's rule 5, which the C2 report never addressed.

If the measured wind is a noisy proxy, |b| should GROW when the noisiest
joins are dropped (planned rather than actual start times) and when only
high-confidence outdoor venue labels are used.  Reported so the bound can
be inflated for attenuation instead of quoted on the measured scale.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_c2_verify import (Race, VSEED, eta_of, feats, load_all, newton_fit,  # noqa
                         pred)

OUT = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parent.parent.parent / "data"


def boot_b(nm, eta, F, T, won, ev, race, nb=400, seed=VSEED):
    uniq = np.array(sorted(set(ev)))
    ev_idx = {e: np.flatnonzero(ev == e) for e in uniq}
    xf = newton_fit(nm, eta, F, T, won, race, [1.0, 0.0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nb):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        idx = np.concatenate([ev_idx[e] for e in pick])
        xb = newton_fit(nm, eta[idx], {k: v[idx] for k, v in F.items()},
                        T[idx], won[idx], race, [1.0, 0.0])
        bs.append(float(xb[1]))
    bs = np.array(bs)
    return dict(b=float(xf[1]), s=float(xf[0]), n=int(len(eta)),
                n_events=int(len(uniq)),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                se=float(bs.std()))


def main():
    race = Race()
    games, pl, chem, traj = load_all("override")
    eta = eta_of(games, pl, chem, traj)
    conf = {r["event_id"].lower(): r["confidence"]
            for r in csv.DictReader(open(DATA / "venue_overrides.csv"))}
    out = {}

    variants = {}
    n = len(games)
    allm = np.ones(n, bool)
    variants["full"] = allm
    variants["hour_actual_only"] = np.array([g["src"] == "hour_actual" for g in games])
    variants["outdoor_high_conf"] = np.array(
        [not (g["setting"] == "outdoor" and conf.get(g["event"], "low") != "high")
         for g in games])

    for tag, m in variants.items():
        F = feats([g for g, k in zip(games, m) if k])
        e = eta[m]
        T = np.array([g["T"] for g, k in zip(games, m) if k])
        won = np.array([g["won"] for g, k in zip(games, m) if k])
        ev = np.array([g["event"] for g, k in zip(games, m) if k])
        row = {}
        for nm in ("a", "d"):
            row[nm] = boot_b(nm, e, F, T, won, ev, race, nb=400)
            print(f"{tag:20s} {nm}: n={row[nm]['n']} ev={row[nm]['n_events']} "
                  f"b={row[nm]['b']:+.4f} CI[{row[nm]['ci'][0]:+.4f},"
                  f"{row[nm]['ci'][1]:+.4f}] se={row[nm]['se']:.4f}",
                  flush=True)
        row["n_outdoor"] = int(F["out"].sum())
        variants_out = out.setdefault("variants", {})
        variants_out[tag] = row

    json.dump(out, open(OUT / "v_c2_verify3.json", "w"), indent=1)
    print("wrote v_c2_verify3.json", flush=True)


if __name__ == "__main__":
    main()
