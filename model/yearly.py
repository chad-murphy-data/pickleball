"""Per-year SRM fits + player trajectory report.

For each season, builds a year-restricted model table, fits the same SRM,
and assembles data/yearly_values.csv plus a trajectory table for players of
interest (printed and written to model/trajectories.md).

Values are POOL-RELATIVE PER YEAR (zero = that year's average player in pro
draws). Cross-year movement therefore mixes true skill change with pool
drift; within-year ranks and gaps to peers are the robust quantities.

Run after all years are harvested+parsed:  python model/yearly.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
YEARS = ["2024", "2025", "2026"]
MIN_GAMES = 40

TRACK = ["Gabriel Tardio", "Ben Johns", "Anna Leigh Waters", "Anna Bright",
         "Hayden Patriquin", "Federico Staksrud", "Christian Alshon",
         "JW Johnson", "Jorja Johnson", "Tyra Hurricane Black",
         "Catherine Parenteau", "Andrei Daescu"]


def run(cmd, env_extra):
    env = dict(os.environ, **env_extra)
    subprocess.run([sys.executable] + cmd, check=True, env=env, cwd=ROOT)


def main():
    for y in YEARS:
        suffix = f"_{y}"
        if not (DATA / f"results_players{suffix}.csv").exists():
            print(f"=== {y}: build + fit ===")
            run(["scraper/build_model_data.py"],
                {"DATE_FROM": f"{y}-01-01", "DATE_BEFORE": f"{int(y)+1}-01-01",
                 "OUT_SUFFIX": suffix})
            run(["model/fit_srm.py"], {"SRM_SUFFIX": suffix})

    # assemble yearly values
    out_rows = []
    per_year = {}
    for y in YEARS:
        rows = list(csv.DictReader((DATA / f"results_players_{y}.csv").open()))
        for r in rows:
            r["value_mean"] = float(r["value_mean"]); r["value_sd"] = float(r["value_sd"])
            r["games"] = int(r["games"])
        elig = sorted([r for r in rows if r["games"] >= MIN_GAMES],
                      key=lambda r: -r["value_mean"])
        rank_all = {r["player_id"]: i for i, r in enumerate(elig, 1)}
        rank_g = {}
        for gname in ("M", "F"):
            for i, r in enumerate([x for x in elig if x["gender"] == gname], 1):
                rank_g[r["player_id"]] = i
        per_year[y] = {"rows": {r["full_name"]: r for r in rows},
                       "rank_gender": rank_g, "n_elig": len(elig),
                       "top_value": elig[0]["value_mean"] if elig else float("nan")}
        for r in rows:
            out_rows.append({"year": y, "player_id": r["player_id"],
                             "full_name": r["full_name"], "gender": r["gender"],
                             "games": r["games"], "value": r["value_mean"],
                             "value_sd": r["value_sd"],
                             "gender_rank": rank_g.get(r["player_id"], "")})
    with (DATA / "yearly_values.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    L = ["# Player trajectories, 2024–2026 (per-year SRM fits)\n",
         "Values are pool-relative per year; ranks are within gender among "
         f"players with ≥{MIN_GAMES} games that year.\n"]
    L.append("| player | " + " | ".join(f"{y} value (rank, games)" for y in YEARS) + " |")
    L.append("|:--|" + "--:|" * len(YEARS))
    for name in TRACK:
        cells = []
        for y in YEARS:
            r = per_year[y]["rows"].get(name)
            if not r or r["games"] < 15:
                cells.append("—")
                continue
            rk = per_year[y]["rank_gender"].get(r["player_id"], "·")
            cells.append(f'{r["value_mean"]:+.2f} (#{rk}, {r["games"]}g)')
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    md = "\n".join(L)
    (ROOT / "model" / "trajectories.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
