#!/usr/bin/env python3
"""Build the Claude Design handoff for the MLP Playoffs weekend slate.

Reads data/forecasts.json (produced by web/make_forecast.py — re-run that
first if lineups have published) and emits, next to this script:

  design_handoff.md   human-readable numbers at all three levels
  forecast_data.json  carousel-ready data (one object per series)

Series math: MLP playoff rounds are best-of-3 matchups; match 3 is
if-necessary. With per-matchup win prob p (identical across the series
while lineups are projections), the series prob is p^2(3-2p).
"""
import json
import os
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PT = timezone(timedelta(hours=-7))  # PDT (Newport Beach, August)

SLOT_NAMES = {"WD": "Women's Doubles", "MD": "Men's Doubles",
              "MXD1": "Mixed Doubles 1", "MXD2": "Mixed Doubles 2"}
ABBR = {"St. Louis Shock": "STL", "Texas Ranchers": "TEX",
        "Los Angeles Mad Drops": "LA", "Dallas Flash": "DAL",
        "Columbus Sliders": "COL", "Brooklyn Pickleball Team": "BKN",
        "New Jersey 5s": "NJ", "Palm Beach Royals": "PB"}


def pt(iso):
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(PT)
    return dt.strftime("%a %-m/%-d %-I:%M%p PT").replace(":00", "")


def pct(x):
    """House rule: never display 0% or 100%."""
    v = round(100 * x)
    if v >= 100:
        return ">99%"
    if v <= 0:
        return "<1%"
    return f"{v}%"


def main():
    src = json.load(open(os.path.join(ROOT, "data", "forecasts.json")))
    series = {}
    for f in src["forecasts"]:
        key = (f["team1"], f["team2"])
        series.setdefault(key, {"meta": f, "starts": []})
        series[key]["starts"].append(f["start"])

    out, md = [], []
    md.append("# MLP Playoffs, Newport Beach — weekend forecast handoff")
    md.append("")
    md.append(f"Generated {src['generated']} from `data/forecasts.json` "
              "(v2 model, calibrated probabilities).")
    md.append("")
    md.append("**Read this first**")
    md.append("- Four best-of-3 series; match 3 is if-necessary. All times PT.")
    md.append("- Official lineups are NOT published yet (BFF status: waiting "
              "for lineups). Pairings below are each team's projected BEST "
              "lineup from its season roster. Historically ~9/10 matchups run "
              "pairings that differ from projection — the mixed splits are "
              "the usual movers. Re-run `web/make_forecast.py` then this "
              "script once lineups drop.")
    md.append("- Series math assumes the same matchup probability each time "
              "the teams meet (true while lineups are projections).")
    md.append("- Display rule: never show 0% or 100% — use >99% / <1%.")
    md.append("")

    for (t1, t2), s in series.items():
        f = s["meta"]
        tr = f["tree"]
        p = tr["p_win"]
        p20 = p * p
        p21 = 2 * p * p * (1 - p)
        p12 = 2 * p * (1 - p) * (1 - p)
        p02 = (1 - p) * (1 - p)
        pser = p20 + p21
        a1, a2 = ABBR[t1], ABBR[t2]
        fav = t1 if p >= 0.5 else t2

        md.append(f"## {t1} ({a1}) vs {t2} ({a2})")
        md.append("")
        md.append(f"Schedule: {', '.join(pt(x) for x in sorted(s['starts']))} "
                  "(match 3 if-necessary)")
        md.append("")
        md.append(f"**Series winner: {fav} {pct(pser if fav == t1 else 1 - pser)}**  ")
        md.append(f"{a1} 2-0: {pct(p20)} · {a1} 2-1: {pct(p21)} · "
                  f"{a2} 2-1: {pct(p12)} · {a2} 2-0: {pct(p02)}")
        md.append("")
        md.append(f"**Single matchup: {fav} {pct(p if fav == t1 else 1 - p)}**  ")
        md.append(f"{a1} 4-0: {pct(tr['p_40'])} · {a1} 3-1: {pct(tr['p_31'])} · "
                  f"reaches DreamBreaker: {pct(tr['p_db'])} "
                  f"({a1} wins it {pct(tr['p_db_win'])}) · "
                  f"{a2} 3-1: {pct(tr['p_13'])} · {a2} 4-0: {pct(tr['p_04'])}")
        md.append("")
        md.append("| Game | " + a1 + " pair | " + a2 + " pair | Favorite | Win prob | Modal score |")
        md.append("|---|---|---|---|---|---|")
        games_out = []
        for g in f["games"]:
            gp = g["p"]
            gfav = a1 if gp >= 0.5 else a2
            gpf = gp if gp >= 0.5 else 1 - gp
            md.append(f"| {SLOT_NAMES[g['slot']]} | {' / '.join(g['t1_pair'])} | "
                      f"{' / '.join(g['t2_pair'])} | {gfav} | {pct(gpf)} | "
                      f"{g['modal']} |")
            games_out.append({"slot": g["slot"], "name": SLOT_NAMES[g["slot"]],
                              "team1_pair": g["t1_pair"], "team2_pair": g["t2_pair"],
                              "p_team1": gp, "favorite": gfav,
                              "p_favorite": round(gpf, 3), "modal_score": g["modal"],
                              "margin": g["margin"]})
        md.append("")
        md.append("DreamBreaker is rally-to-21 singles priced off mean roster "
                  "singles value — no fixed pairing to show.")
        md.append("")
        out.append({
            "team1": t1, "team2": t2, "abbr1": a1, "abbr2": a2,
            "schedule_pt": [pt(x) for x in sorted(s["starts"])],
            "starts_utc": sorted(s["starts"]),
            "lineup_status": "projected (best lineup); officials not yet published",
            "matchup": {"p_team1": p, "favorite": fav,
                        "dist": {"t1_4_0": tr["p_40"], "t1_3_1": tr["p_31"],
                                 "dreambreaker": tr["p_db"],
                                 "t1_wins_db": tr["p_db_win"],
                                 "t2_3_1": tr["p_13"], "t2_4_0": tr["p_04"]}},
            "series_best_of_3": {"p_team1": round(pser, 4), "favorite": fav,
                                 "dist": {"t1_2_0": round(p20, 4),
                                          "t1_2_1": round(p21, 4),
                                          "t2_2_1": round(p12, 4),
                                          "t2_2_0": round(p02, 4)}},
            "games": games_out,
        })

    md_path = os.path.join(HERE, "design_handoff.md")
    json_path = os.path.join(HERE, "forecast_data.json")
    open(md_path, "w").write("\n".join(md) + "\n")
    json.dump({"generated": src["generated"],
               "event": "Toray MLP Playoffs Newport Beach",
               "display_rule": "never render 0% or 100%; clamp to >99% / <1%",
               "series": out}, open(json_path, "w"), indent=2)
    print("wrote", md_path)
    print("wrote", json_path)


if __name__ == "__main__":
    main()
