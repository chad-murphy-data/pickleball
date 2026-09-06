"""value_cap/simulate_templates.py -- play the template-strategy rosters
against each other for a whole season, many times.

    python value_cap/simulate_templates.py                       # both price lists, 20k seasons each
    python value_cap/simulate_templates.py --results value_cap/draft_strategies_tag.md --seasons 50000

Input = a results file written by draft_strategies.py (the "### name --
desc" sections with "- Starters:" / "- Bench:" lines). Reference-only
rosters (the $500k Cheapskate) are left out of the league. Every pair of
rosters gets its tie probability from the real tie model
(phase2_pricing.win = phase1_value_model.tie_win_prob), then each season is:

  - double round robin: every pair plays twice (7 opponents x 2 = 14 ties)
  - standings by wins, ties in the standings broken by coin flip (no game
    differential is simulated)
  - top 4 make the playoffs: 1 v 4, 2 v 3, winners meet; each playoff tie
    is one draw from the same pairwise probability

What this is: a "which blueprint wins" tournament. The rosters were built
with NO exclusivity (draft_strategies.py's framing), so several of them
share players -- Grayson Goldin is on most of them. That is fine for asking
"is any template a dominant blueprint at these prices?" and wrong for
asking "what does a real 20-team league look like" (that needs a draft
with scarcity, the Phase 3 build). The overlap is printed so nobody
forgets.

Stdlib only, seeded, ~10 s for 20k seasons.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from phase2_pricing import NAME, cost, pid_named, prices, prices_tagged, win
from pool import load_pool  # noqa: F401  (import side effect: same pool as the results)

HERE = Path(__file__).resolve().parent


def parse_results(path):
    """-> (header_line, [(name, desc, roster, is_reference)])"""
    text = Path(path).read_text()
    header = next(l for l in text.splitlines() if l.startswith("Prices:"))
    out = []
    for m in re.finditer(r"^### (.+?)  --  (.+?)\n(.*?)(?=^### |^## |\Z)", text, re.S | re.M):
        name, desc, body = m.group(1).strip(), m.group(2).strip(), m.group(3)
        names = []
        for line in body.splitlines():
            if line.startswith("- Starters: ") or line.startswith("- Bench: "):
                names += [re.sub(r" \(\$.*", "", x).strip()
                          for x in line.split(": ", 1)[1].split("), ")]
        roster = tuple(pid_named(n) for n in names)
        assert len(roster) == 6, (name, names)
        out.append((name, desc, roster, "reference only" in desc))
    return header, out


def price_list_for(header):
    """Rebuild the price dict the results file says it used (for spend + overlap notes)."""
    alpha = float(re.search(r"alpha = ([0-9.]+)", header).group(1))
    tag = re.search(r"\*\*(.+?) franchise-tagged", header)
    if tag:
        return prices_tagged(load_pool("phi"), alpha, pid_named(tag.group(1)))
    return prices(load_pool("phi"), alpha)


def simulate(teams, P, seasons, rng):
    """teams: list of names; P[i][j] = P(i beats j) in one tie."""
    n = len(teams)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    tally = {t: {"wins": 0.0, "reg1": 0, "top4": 0, "title": 0,
                 "rank": [0] * n} for t in range(n)}
    for _ in range(seasons):
        wins = [0] * n
        for i, j in pairs:
            for _leg in range(2):
                if rng.random() < P[i][j]:
                    wins[i] += 1
                else:
                    wins[j] += 1
        order = sorted(range(n), key=lambda t: (-wins[t], rng.random()))
        for r, t in enumerate(order):
            tally[t]["rank"][r] += 1
            tally[t]["wins"] += wins[t]
        tally[order[0]]["reg1"] += 1
        s1, s2, s3, s4 = order[:4]
        for t in order[:4]:
            tally[t]["top4"] += 1
        f1 = s1 if rng.random() < P[s1][s4] else s4
        f2 = s2 if rng.random() < P[s2][s3] else s3
        champ = f1 if rng.random() < P[f1][f2] else f2
        tally[champ]["title"] += 1
    return tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append",
                    default=None, help="draft_strategies*.md file(s); default = both lists")
    ap.add_argument("--seasons", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "template_season.md"))
    args = ap.parse_args()
    results = args.results or [str(HERE / "draft_strategies_tag.md"),
                               str(HERE / "draft_strategies.md")]

    lines = ["# Template rosters, played as a season", "",
             f"Each price list's template rosters (from `draft_strategies*.md`, reference-only "
             f"rosters excluded) play a double round robin ({args.seasons:,} seasons, seed "
             f"{args.seed}), standings ties broken by coin flip, top 4 to a 1v4 / 2v3 / final "
             f"playoff. Tie probabilities come from the real tie model. NOTE: these rosters were "
             f"built without exclusivity and share players, so this is blueprint-vs-blueprint, "
             f"not a league. Built by `simulate_templates.py`.", ""]
    rng = random.Random(args.seed)
    for path in results:
        header, rosters = parse_results(path)
        price = price_list_for(header)
        league = [(n, d, r) for n, d, r, ref in rosters if not ref]
        teams = [n for n, _, _ in league]
        P = [[0.5 if i == j else win(league[i][2], league[j][2])
              for j in range(len(league))] for i in range(len(league))]
        tally = simulate(teams, P, args.seasons, rng)
        n = len(teams)
        exp_wins = [sum(2 * P[i][j] for j in range(n) if j != i) for i in range(n)]

        lines += [f"## {Path(path).name}", "", header, "",
                  f"League = {n} template rosters, {2*(n-1)} ties each per season.", "",
                  "| roster | expected wins | mean wins | P(1st in table) | P(top 4) | **P(title)** | "
                  "P(last) | spend |", "|---|---|---|---|---|---|---|---|"]
        order = sorted(range(n), key=lambda i: -tally[i]["title"])
        for i in order:
            t = tally[i]
            lines.append(f"| {teams[i]} | {exp_wins[i]:.2f} | {t['wins']/args.seasons:.2f} | "
                         f"{100*t['reg1']/args.seasons:.1f}% | {100*t['top4']/args.seasons:.1f}% | "
                         f"**{100*t['title']/args.seasons:.1f}%** | "
                         f"{100*t['rank'][-1]/args.seasons:.1f}% | ${cost(league[i][2], price)/1e3:,.0f}k |")
        lines += ["", f"Parity yardstick: {n} equal teams would each take the title "
                  f"{100/n:.1f}% of the time and average {n-1:.1f} wins.", ""]

        # finish-position distribution
        lines += ["Finish-position distribution (P of finishing 1st .. last):", "",
                  "| roster | " + " | ".join(str(k + 1) for k in range(n)) + " |",
                  "|---|" + "---|" * n]
        for i in order:
            lines.append(f"| {teams[i]} | " + " | ".join(
                f"{100*c/args.seasons:.0f}" for c in tally[i]["rank"]) + " |")
        lines.append("")

        # overlap
        seen = {}
        for name, _, r in league:
            for u in r:
                seen.setdefault(u, []).append(name)
        shared = sorted(((u, ts) for u, ts in seen.items() if len(ts) > 1),
                        key=lambda kv: -len(kv[1]))
        lines += ["Shared players (why this is not a league):", ""]
        for u, ts in shared:
            lines.append(f"- {NAME[u]} (${price[u]/1e3:,.0f}k): {len(ts)} rosters -- {', '.join(ts)}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
