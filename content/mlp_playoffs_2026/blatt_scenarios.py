#!/usr/bin/env python3
"""Price Brooklyn's COL-BKN matchup under Hannah Blatt substitution scenarios.

Uses web/make_forecast.py's own pricing engine (price_game, matchup_tree,
db_win_prob) so numbers are directly comparable to data/forecasts.json.
No network: values come from the committed model CSVs.

Scenarios, all vs the same Columbus projected lineup:
  0  baseline   Rohrabacher WD + MXD1     (current published projection)
  A  Blatt WD   Blatt in WD, Rohrabacher still MXD1  (3-woman rotation)
  B  Blatt both Blatt in WD + MXD1, Rohrabacher off the card

MLP allows the 3-woman rotation scenario A needs: 117 of 572 team-matchups
in 2026 used a third woman (data/mlp_matchups_2026.csv joined to games.csv).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "web"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scraper"))

from make_forecast import (db_win_prob, load_singles, load_values,  # noqa: E402
                           matchup_tree, price_game)

# Columbus (team1 in the published forecast) — projected best lineup
TODD = "parris-todd"
BLACK = "tyra-black"
DAESCU = "andrei-daescu"
KLINGER = "cj-klinger"
# Brooklyn (team2)
ROHR = "50908953-a944-4a34-898b-16af312bc814"
BLATT = "25a6f570-d617-427a-9fc4-3bf6115a9a84"


def uuid_by_name(vals, name):
    for u, (n, _v, _g) in vals.items():
        if n.lower() == name.lower():
            return u
    raise KeyError(name)


def series(p):
    """Best-of-3 series prob from a per-matchup prob."""
    return p * p * (3 - 2 * p)


def main():
    vals = load_values()
    singles = load_singles()
    U = lambda n: uuid_by_name(vals, n)  # noqa: E731

    todd, black = U("Parris Todd"), U("Tyra Hurricane Black")
    daescu, klinger = U("Andrei Daescu"), U("CJ Klinger")
    jkaw, alshon, newman = U("Jackie Kawamoto"), U("Christian Alshon"), U("Riley Newman")

    scen = {
        "0 baseline (Rohrabacher WD + MXD1)": {
            "WD": ([todd, black], [ROHR, jkaw]),
            "MD": ([daescu, klinger], [alshon, newman]),
            "MXD1": ([todd, daescu], [ROHR, alshon]),
            "MXD2": ([black, klinger], [jkaw, newman]),
            "db": [ROHR, jkaw, alshon, newman],
        },
        "A Blatt WD only (Rohrabacher stays MXD1)": {
            "WD": ([todd, black], [BLATT, jkaw]),
            "MD": ([daescu, klinger], [alshon, newman]),
            "MXD1": ([todd, daescu], [ROHR, alshon]),
            "MXD2": ([black, klinger], [jkaw, newman]),
            # DB four is a choice in a 3-woman rotation; both shown below
            "db": [ROHR, jkaw, alshon, newman],
            "db_alt": [BLATT, jkaw, alshon, newman],
        },
        "B Blatt WD + MXD1 (Rohrabacher off)": {
            "WD": ([todd, black], [BLATT, jkaw]),
            "MD": ([daescu, klinger], [alshon, newman]),
            "MXD1": ([todd, daescu], [BLATT, alshon]),
            "MXD2": ([black, klinger], [jkaw, newman]),
            "db": [BLATT, jkaw, alshon, newman],
        },
    }

    col_db_roster = [todd, black, daescu, klinger]

    def s_of(u):
        """Mirror db_win_prob's singles resolution, for reporting."""
        if u in singles and singles[u][1] >= 10:
            return f"{singles[u][0]:.3f} (real, {singles[u][1]} g)"
        from make_forecast import SINGLES_IMPUTE
        a, b = SINGLES_IMPUTE
        return f"{a + b * vals[u][1]:.3f} (imputed)"

    print("Hannah Blatt   v2 doubles %.4f | singles %s"
          % (vals[BLATT][1], s_of(BLATT)))
    print("R. Rohrabacher v2 doubles %.4f | singles %s"
          % (vals[ROHR][1], s_of(ROHR)))
    print()

    for label, s in scen.items():
        ps, lines = [], []
        for slot in ("WD", "MD", "MXD1", "MXD2"):
            a, b = s[slot]
            g = price_game(a, b, vals)
            ps.append(g["p"])                      # p is COLUMBUS win prob
            lines.append("    %-5s BKN %5.1f%%  (%s)  COL by %+.2f, modal %s"
                         % (slot, 100 * (1 - g["p"]),
                            " / ".join(vals[u][0] for u in b),
                            g["margin"], g["modal"]))
        p_db = db_win_prob(col_db_roster, s["db"], vals, singles)
        tree = matchup_tree(ps, p_db)
        bkn = 1 - tree["p_win"]
        print(label)
        print("\n".join(lines))
        print("    DreamBreaker reached %.1f%% | BKN wins it %.1f%%"
              % (100 * tree["p_db"], 100 * (1 - p_db)))
        if "db_alt" in s:
            p_alt = db_win_prob(col_db_roster, s["db_alt"], vals, singles)
            t_alt = matchup_tree(ps, p_alt)
            print("      (if Blatt takes the DB slot instead: BKN wins DB "
                  "%.1f%% -> matchup %.1f%%)"
                  % (100 * (1 - p_alt), 100 * (1 - t_alt["p_win"])))
        print("    => BKN matchup %.1f%%   series (best-of-3) %.1f%%"
              % (100 * bkn, 100 * series(bkn)))
        print()


if __name__ == "__main__":
    main()
