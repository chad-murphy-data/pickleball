"""C3(A) addendum — is the audited/unaudited gap in d real, with a PAIRED CI?

Phase-2 B2a reported the interaction separately on web-verified-outdoor and
unaudited-outdoor events.  Two separately bootstrapped estimates cannot be
differenced honestly, so this fits ONE regression on the outdoor arm-c pool
with a full set of interactions with an `unaudited` dummy, and bootstraps
the gap itself over events:

    y = a + b*skill + c*w + d*sw
        + U*(a2 + b2*skill + c2*w + d2*sw)

d2 is the gap.  Also run with event FE + event x skill absorbed (the
within-event version), where U itself is absorbed but d2 survives.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3a_fixed_effects as fx  # noqa: E402
from c3_lib import absorb, cluster_se, load_frame  # noqa: E402


def run(spec, rows, B=1000, seed=5):
    D = fx.pack(rows)
    n = len(rows)
    U = np.array([0.0 if r["audited"] else 1.0 for r in rows])
    sw = D["skill"] * D["w"]
    cols = [D["skill"], D["w"], sw, U, U * D["skill"], U * D["w"], U * sw]
    M = np.column_stack([D["y"]] + cols)
    if spec != "L0":
        fe = fx.make_fe(spec, D, np.zeros(n, np.int64))
        M, _ = absorb(fe, M)
    X = np.column_stack([np.ones(n)] + [M[:, i] for i in range(1, M.shape[1])])
    beta = np.linalg.solve(X.T @ X, X.T @ M[:, 0])
    cov, G = cluster_se(X, M[:, 0], beta, D["ev"])
    by = defaultdict(list)
    for i, e in enumerate(D["ev"]):
        by[int(e)].append(i)
    idxs = {k: np.array(v) for k, v in by.items()}
    keys = list(idxs)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        sel = np.concatenate([idxs[keys[i]]
                              for i in rng.integers(0, len(keys), len(keys))])
        Xb, yb = X[sel], M[sel, 0]
        try:
            draws.append(np.linalg.solve(Xb.T @ Xb, Xb.T @ yb))
        except np.linalg.LinAlgError:
            pass
    d = np.array(draws)
    lo, hi = np.percentile(d[:, 3], [2.5, 97.5])
    glo, ghi = np.percentile(d[:, 7], [2.5, 97.5])
    print(f"{spec}: n={n} events={G}")
    print(f"   d (audited events)      = {beta[3]:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"   d2 (unaudited - audited)= {beta[7]:+.4f} [{glo:+.4f},{ghi:+.4f}]"
          f"  CRse={np.sqrt(cov[7,7]):.4f}")
    print(f"   implied d on unaudited  = {beta[3]+beta[7]:+.4f}")


if __name__ == "__main__":
    rows = [r for r in load_frame() if r["setting_c"] == "outdoor"]
    for spec in ("L0", "L1", "L2", "L3"):
        run(spec, rows)
