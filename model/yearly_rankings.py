"""Side-by-side yearly rankings: full pool vs core pool ("casuals removed").

Full pool  = every sideout-11 game that year (results_players_{Y}.csv).
Core pool  = only games where all four players logged >=30 games that year
             (results_players_{Y}core.csv).

Writes model/yearly_rankings.md with, per year x gender, the top N by core
value alongside full-pool value and both ranks.

Run after the per-year fits:  python model/yearly_rankings.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
YEARS = ["2024", "2025", "2026"]
TOP_N = 15
MIN_GAMES = 40  # eligibility within each fit's own games count


def load(suffix):
    rows = list(csv.DictReader((DATA / f"results_players{suffix}.csv").open()))
    for r in rows:
        r["value_mean"] = float(r["value_mean"])
        r["games"] = int(r["games"])
    return {r["player_id"]: r for r in rows}


def ranked(pool, gender, min_games):
    sub = [r for r in pool.values() if r["gender"] == gender and r["games"] >= min_games]
    sub.sort(key=lambda r: -r["value_mean"])
    return sub, {r["player_id"]: i for i, r in enumerate(sub, 1)}


def main():
    L = ["# Yearly rankings: full tour vs core pool (\"casuals removed\")\n",
         "Core pool keeps only games where **all four players on court logged ≥30 games "
         "that year** — Challenger one-weekenders and qualifier cannon fodder drop out, "
         "and values re-center on the stronger pool (so core values run lower; compare "
         "ranks and gaps, not levels, across the two columns).\n"]
    for y in YEARS:
        full = load(f"_{y}")
        core = load(f"_{y}core")
        for gender, glabel in (("M", "Men"), ("F", "Women")):
            f_sub, f_rank = ranked(full, gender, MIN_GAMES)
            c_min = 30 if y == "2026" else MIN_GAMES  # 2026 is a half season
            c_sub, c_rank = ranked(core, gender, c_min)
            L.append(f"## {y} — {glabel}  "
                     f"(full pool: {len(f_sub)} eligible; core: {len(c_sub)})\n")
            L.append("| core rank | player | core value | full value | full rank | core games |")
            L.append("|--:|:--|--:|--:|--:|--:|")
            for i, r in enumerate(c_sub[:TOP_N], 1):
                u = r["player_id"]
                fr = full.get(u)
                fv = f"{fr['value_mean']:+.2f}" if fr else "—"
                L.append(f"| {i} | {r['full_name']} | {r['value_mean']:+.2f} | {fv} | "
                         f"{f_rank.get(u, '—')} | {r['games']} |")
            # biggest rank movers full->core among core top 30
            movers = []
            for i, r in enumerate(c_sub[:30], 1):
                u = r["player_id"]
                if u in f_rank and abs(f_rank[u] - i) >= 5:
                    movers.append(f"{r['full_name']} (full #{f_rank[u]} → core #{i})")
            if movers:
                L.append("\nMovers ≥5 ranks: " + "; ".join(movers))
            L.append("")
    out = ROOT / "model" / "yearly_rankings.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
