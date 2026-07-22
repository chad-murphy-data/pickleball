"""Empirical doubles->singles imputation shrink from DreamBreaker rallies.

Never-play-singles players get a singles value imputed from doubles
(make_forecast.SINGLES_IMPUTE). db_model.md flagged that imputation is
selection-biased UP: they don't play singles because they're worse at it.
This measures it directly — DreamBreakers ARE singles points — by pooling
every non-singles player's DB rallies and comparing their actual rally wins
to what the imputation predicts.

Method:
- DreamBreakers are excluded from the Supabase rally warehouse (tie-breakers),
  so fetch the referee logs directly (getListLogs, cached in raw/match_logs).
- DB logs are quirky: the score string is the TEAM total (server-team first,
  a+b = rally index) and receiver_uuid is unreliable, so reconstruct each
  rally's winner from the SERVER rotation + team scores, and read matchups off
  the two distinct servers in each 4-rally segment.
- Restrict to SAME-GENDER rallies (cross-gender value gaps aren't identified).
- Compare imputed vs real-singles players; fit a logistic for the shrink.

Internal check: the recovered rally-level k should match the independent
team-level DB fit (~0.42). Run: python model/db_impute.py
"""
from __future__ import annotations
import csv, math, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scraper"))
from pb_api import PBClient                                   # noqa: E402

K = 0.42


def load_values():
    sv, sg, gen, dbl = {}, {}, {}, {}
    with open(ROOT / "data" / "singles_players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            sv[u] = float(r["singles_value"]); sg[u] = int(r["singles_games"])
            gen[u] = r["gender"]
    with open(ROOT / "data" / "v2_players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            dbl[u] = float(r["value_now_mean"]); gen.setdefault(u, r["gender"])

    def info(u):
        g = gen.get(u)
        if u in sv and sg[u] >= 10:
            return sv[u], False, g            # real singles history
        if u in dbl:
            return 0.28 + 1.14 * dbl[u], True, g   # UN-shrunk imputation
        if u in sv:
            return sv[u], True, g
        return None, None, g
    return info


def db_match_ids():
    ids = []
    with open(ROOT / "data" / "dreambreakers.csv") as fh:
        for r in csv.DictReader(fh):
            m = r["match_id"]
            if len(m) == 36 and m.count("-") == 4:
                ids.append(m)
    return sorted(set(ids))


def reconstruct(rows):
    """(server, winner_team) per rally + team_of, from server + team score."""
    seq = []
    for r in rows:
        if r.get("log_type") != 12:
            continue
        s = (r.get("server_uuid") or "").lower()
        st = r.get("start_score_current_game_string")
        if not s or not st:
            continue
        try:
            a, b = (int(x) for x in st.split("-"))
        except ValueError:
            continue
        seq.append((s, a, b))
    other = lambda x: "Y" if x == "X" else "X"
    team_of, cX, cY, stand = {}, 0, 0, []
    for s, a, b in seq:
        if s in team_of:
            t = team_of[s]
        else:                              # a=server-team total, b=opponent's
            if a == cX and b == cY: t = "X"
            elif a == cY and b == cX: t = "Y"
            elif b == cX: t = "Y"
            elif b == cY: t = "X"
            elif a == cX: t = "X"
            elif a == cY: t = "Y"
            else: t = "X"
            team_of[s] = t
        X, Y = (a, b) if t == "X" else (b, a)
        cX, cY = X, Y
        stand.append((s, t, X, Y))
    winners = []
    for i in range(len(stand)):
        s, t, X, Y = stand[i]
        wX = (stand[i + 1][2] > X) if i + 1 < len(stand) else (X + 1 > Y)
        winners.append("X" if wX else "Y")
    return seq, team_of, winners


def main():
    info = load_values()
    c = PBClient()
    ids = db_match_ids()
    print(f"fetching {len(ids)} DreamBreaker logs (cached in raw/match_logs)...")
    rec = collections.defaultdict(lambda: [0, 0.0, 0.0])   # grp -> n,act,exp
    data = []
    sig = lambda x: 1 / (1 + math.exp(-x))
    for mid in ids:
        rows = c.match_logs(mid) or []
        seq, team_of, winners = reconstruct(rows)
        if not seq:
            continue
        posmap = collections.defaultdict(dict)
        for t, (s, a, b) in enumerate(seq):
            posmap[(t // 4) % 4][team_of[s]] = s
        tally = collections.defaultdict(lambda: [0, 0])
        for t, (s, a, b) in enumerate(seq):
            px, py = posmap[(t // 4) % 4].get("X"), posmap[(t // 4) % 4].get("Y")
            if px and py:
                tally[(px, py)][0 if winners[t] == "X" else 1] += 1
        for (px, py), (wx, wy) in tally.items():
            vi, impi, gi = info(px); vj, impj, gj = info(py)
            if vi is None or vj is None or gi != gj:
                continue
            m = wx + wy
            pi = sig(K * (vi - vj))
            for imp, w, pw in ((impi, wx, pi), (impj, wy, 1 - pi)):
                g = "imputed" if imp else "real"
                rec[g][0] += m; rec[g][1] += w; rec[g][2] += m * pw
            data.append((vi - vj, (1 if impi else 0) - (1 if impj else 0), wx, m))

    print(f"\n{'group':8s}{'rallies':>9}{'actual':>9}{'expected':>10}{'resid':>9}")
    for g in ("real", "imputed"):
        n, act, exp = rec[g]
        if n:
            print(f"{g:8s}{n:>9d}{act/n:>9.3f}{exp/n:>10.3f}{(act-exp)/n:>+9.3f}")

    # logistic: sigmoid(a + b1*gap + b2*impdiff); Newton + Hessian SE
    a, b1, b2, H = 0.0, 0.3, 0.0, None
    for _ in range(100):
        g = [0.0] * 3; H = [[0.0] * 3 for _ in range(3)]
        for gap, imd, wi, m in data:
            p = sig(a + b1 * gap + b2 * imd); x = [1.0, gap, float(imd)]
            r = wi - m * p; w = m * p * (1 - p)
            for u in range(3):
                g[u] += r * x[u]
                for v in range(3):
                    H[u][v] += w * x[u] * x[v]
        M = [row[:] + [g[k]] for k, row in enumerate(H)]
        for col in range(3):
            pv = M[col][col]
            for cc in range(col, 4):
                M[col][cc] /= pv
            for rr in range(3):
                if rr != col:
                    f = M[rr][col]
                    for cc in range(col, 4):
                        M[rr][cc] -= f * M[col][cc]
        a += M[0][3]; b1 += M[1][3]; b2 += M[2][3]
    # invert H for SE
    n = 3; A = [row[:] for row in H]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pv = A[col][col]
        for j in range(n):
            A[col][j] /= pv; I[col][j] /= pv
        for rr in range(n):
            if rr != col:
                f = A[rr][col]
                for j in range(n):
                    A[rr][j] -= f * A[col][j]; I[rr][j] -= f * I[col][j]
    se2 = I[2][2] ** 0.5
    delta = -b2 / b1
    print(f"\nlogistic  P = sigmoid({a:+.3f} + {b1:.3f}*gap + {b2:+.3f}*impdiff)")
    print(f"  empirical rally k = {b1:.3f}   (team-level DB fit ~0.42)")
    print(f"  imputed penalty   = {b2:+.3f} +/- {se2:.3f}  (z={b2/se2:.2f})")
    print(f"  => shrink imputed value by ~{delta:.2f} "
          f"(95% CI {-(b2 - 1.96*se2)/b1:+.2f}..{-(b2 + 1.96*se2)/b1:+.2f})")
    print("  Applied: SINGLES_IMPUTE intercept 0.28 -> 0.08 "
          "(rough; directional, not significant).")


if __name__ == "__main__":
    main()
