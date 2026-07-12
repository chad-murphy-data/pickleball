"""Go/no-go diagnostics for the SRM dataset → diagnostics.md

Four checks, per the project brief:
  (a) dyad game-count distribution + focal-dyad table
  (b) distinct-partner counts per focal player
  (c) partnership-graph connected components WITHIN each context
  (d) margin distributions by tour and by game number (PPA game-3 compression)

Run after parse.py:  python scraper/diagnostics.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "diagnostics.md"

FOCAL_NAMES = [
    "Ben Johns", "Anna Leigh Waters", "Anna Bright", "Hayden Patriquin",
    "Gabriel Tardio", "Federico Staksrud", "Jade Kawamoto", "Jorja Johnson",
    "Will Howells", "Noe Khlif", "Etta Fahey", "Tyra Black",
]


def load():
    games = list(csv.DictReader((DATA / "games.csv").open()))
    players = {r["player_id"]: r for r in csv.DictReader((DATA / "players.csv").open())}
    for g in games:
        for k in ("t1_score", "t2_score", "margin", "game_number", "best_of"):
            g[k] = int(g[k])
        g["is_forfeit"] = g["is_forfeit"] == "True"
    return games, players


def find_focal(players):
    """Map focal display names -> uuid by containment on canonical + variants.

    Returns (focal: want->uuid|None, notes: list[str]) — every non-exact
    resolution is surfaced in the report rather than silently accepted.
    """
    focal, notes = {}, []
    for want in FOCAL_NAMES:
        tokens = want.lower().split()
        hits = []
        for uuid, rec in players.items():
            hay = (rec["full_name"] + " " + rec["name_variants"]).lower()
            if all(t in hay for t in tokens):
                hits.append(uuid)
        if len(hits) == 1:
            focal[want] = hits[0]
            if players[hits[0]]["full_name"].lower() != want.lower():
                notes.append(f'"{want}" resolved to **{players[hits[0]]["full_name"]}** '
                             f'(`{hits[0]}`) by token match.')
            continue
        if len(hits) > 1:
            focal[want] = None
            notes.append(f'"{want}" is AMBIGUOUS: ' + "; ".join(
                f'{players[h]["full_name"]} (`{h}`)' for h in hits))
            continue
        # last-name-only fallback — always disclosed
        last = tokens[-1]
        hits = [u for u, r in players.items()
                if last in r["full_name"].lower().split()]
        if len(hits) == 1:
            focal[want] = hits[0]
            notes.append(f'⚠️ "{want}" not found; last-name fallback picked '
                         f'**{players[hits[0]]["full_name"]}** (`{hits[0]}`) — verify this '
                         "is the intended person.")
        else:
            focal[want] = None
            if hits:
                notes.append(f'"{want}" not found; last name "{last}" matches ' + "; ".join(
                    f'{players[h]["full_name"]} (`{h}`)' for h in hits))
            else:
                notes.append(f'"{want}" not found at all — not in any 2026 pro doubles draw?')
    return focal, notes


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def hist_ascii(values, bins):
    """values: list of numbers; bins: list of (lo, hi_exclusive, label)."""
    counts = []
    for lo, hi, label in bins:
        n = sum(1 for v in values if lo <= v < hi)
        counts.append((label, n))
    mx = max((n for _, n in counts), default=1) or 1
    lines = []
    for label, n in counts:
        bar = "█" * max(1 if n else 0, round(24 * n / mx))
        lines.append(f"| {label:>7} | {n:>5} | {bar} |")
    return lines


def main():
    games, players = load()
    modeling = [g for g in games if not g["is_forfeit"]]
    focal, focal_notes = find_focal(players)
    name_of = lambda u: players[u]["full_name"] if u in players else u[:8]

    # ---------- (a) dyad game counts ----------
    dyad_games = Counter()          # frozenset(p1,p2) -> n games (any context)
    dyad_games_ctx = defaultdict(Counter)   # context -> dyad -> n
    for g in modeling:
        for a, b in ((g["t1_p1"], g["t1_p2"]), (g["t2_p1"], g["t2_p2"])):
            d = frozenset((a, b))
            dyad_games[d] += 1
            dyad_games_ctx[g["context"]][d] += 1

    # ---------- (b) partners per player ----------
    partners = defaultdict(set)
    partners_ctx = defaultdict(lambda: defaultdict(set))
    player_games = Counter()
    for g in modeling:
        for a, b in ((g["t1_p1"], g["t1_p2"]), (g["t2_p1"], g["t2_p2"])):
            partners[a].add(b); partners[b].add(a)
            partners_ctx[g["context"]][a].add(b)
            partners_ctx[g["context"]][b].add(a)
            player_games[a] += 1; player_games[b] += 1

    # ---------- (c) connectivity per context ----------
    comp_report = {}
    for ctx in ("mixed", "mens", "womens"):
        dsu = DSU()
        nodes = set()
        for g in (x for x in modeling if x["context"] == ctx):
            for a, b in ((g["t1_p1"], g["t1_p2"]), (g["t2_p1"], g["t2_p2"])):
                dsu.union(a, b); nodes.update((a, b))
        comps = defaultdict(set)
        for n in nodes:
            comps[dsu.find(n)].add(n)
        sizes = sorted((len(v) for v in comps.values()), reverse=True)
        # which component holds the focal players?
        focal_comp = {}
        for want, uuid in focal.items():
            if uuid is not None and uuid in nodes:
                focal_comp[want] = len(comps[dsu.find(uuid)])
        # opposition-augmented connectivity (supplementary)
        dsu2 = DSU()
        nodes2 = set()
        for g in (x for x in modeling if x["context"] == ctx):
            ps = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
            for i in range(3):
                dsu2.union(ps[i], ps[i + 1])
            nodes2.update(ps)
        comps2 = defaultdict(set)
        for n in nodes2:
            comps2[dsu2.find(n)].add(n)
        sizes2 = sorted((len(v) for v in comps2.values()), reverse=True)
        comp_report[ctx] = {
            "n_players": len(nodes), "n_components": len(comps), "sizes": sizes[:8],
            "focal_component_sizes": focal_comp,
            "n_components_with_opposition": len(comps2), "sizes_with_opposition": sizes2[:8],
            "components": comps,
        }

    # ---------- (d) margins ----------
    def describe(rows):
        m = [r["margin"] for r in rows]
        am = [abs(x) for x in m]
        return dict(n=len(m), mean=st.mean(m) if m else float("nan"),
                    sd=st.stdev(m) if len(m) > 1 else float("nan"),
                    mean_abs=st.mean(am) if am else float("nan"))

    # margin comparisons on the common format only — to-15/to-21 Challenger
    # rounds are a different margin scale and would muddy the tour comparison
    s11 = [g for g in modeling if g["scoring_format"] == "sideout_11"]
    n_other_fmt = len(modeling) - len(s11)
    by_tour = {t: describe([g for g in s11 if g["tour"] == t]) for t in ("MLP", "PPA")}
    ppa = [g for g in s11 if g["tour"] == "PPA"]
    by_gn = {n: describe([g for g in ppa if g["game_number"] == n]) for n in (1, 2, 3, 4, 5)}

    # ---------------- write report ----------------
    L = []
    A = L.append
    A("# diagnostics.md — GO/NO-GO gate for the SRM dataset\n")
    A(f"Dataset: `data/games.csv` — **{len(games)} games** "
      f"({len(modeling)} after excluding {len(games)-len(modeling)} forfeit-tainted), "
      f"**{len(players)} players**, "
      f"{sum(1 for g in modeling if g['tour']=='MLP')} MLP / "
      f"{sum(1 for g in modeling if g['tour']=='PPA')} PPA. "
      "DreamBreakers are in `data/dreambreakers.csv`, never here.\n")

    A("## Data-quality overview\n")
    A("Scoring formats present (modeling rows):\n")
    A("| tour | scoring_format | best_of | games |")
    A("|:--|:--|--:|--:|")
    fmt_mix = Counter((g["tour"], g["scoring_format"], g["best_of"]) for g in modeling)
    for (tour, fmt, bo), n in sorted(fmt_mix.items()):
        A(f"| {tour} | {fmt} | {bo} | {n} |")
    A("")
    dropped_path = DATA / "dropped.csv"
    if dropped_path.exists():
        drops = list(csv.DictReader(dropped_path.open()))
        A(f"Dropped rows: **{len(drops)}** (full detail in `data/dropped.csv`):\n")
        A("| reason | rows |")
        A("|:--|--:|")
        for r, n in Counter(d["reason"].split("(")[0].strip() for d in drops).most_common():
            A(f"| {r} | {n} |")
        A("")
    flags_path = DATA / "flags.csv"
    if flags_path.exists():
        fl = list(csv.DictReader(flags_path.open()))
        A(f"Flagged-but-kept rows: **{len(fl)}** (full detail in `data/flags.csv`):\n")
        A("| flag | rows |")
        A("|:--|--:|")
        for r, n in Counter(f["reason"].split("(")[0].strip()[:60] for f in fl).most_common(12):
            A(f"| {r} | {n} |")
        A("")

    # (a)
    A("## (a) Dyad game-count distribution\n")
    A(f"Distinct dyads: **{len(dyad_games)}** across all contexts "
      "(a dyad = unordered player pair that appeared on the same side of the net).\n")
    A("| games together | dyads | |")
    A("|---:|---:|:---|")
    bins = [(1, 2, "1"), (2, 5, "2–4"), (5, 10, "5–9"), (10, 15, "10–14"),
            (15, 25, "15–24"), (25, 50, "25–49"), (50, 10**9, "50+")]
    L.extend(hist_ascii(list(dyad_games.values()), bins))
    med = st.median(dyad_games.values())
    A(f"\nMedian games per dyad: **{med:.0f}**. "
      f"Dyads with ≥10 games: **{sum(1 for v in dyad_games.values() if v >= 10)}**; "
      f"with ≥15: **{sum(1 for v in dyad_games.values() if v >= 15)}**.\n")

    A("### Focal dyads (games together, all contexts)\n")
    focal_uuids = {u: w for w, u in focal.items() if u}
    A("| dyad | games | contexts |")
    A("|:--|--:|:--|")
    focal_dyads = [(d, n) for d, n in dyad_games.items() if any(p in focal_uuids for p in d)]
    for d, n in sorted(focal_dyads, key=lambda x: -x[1])[:40]:
        names = " + ".join(sorted(name_of(p) for p in d))
        ctxs = ",".join(sorted(c for c in dyad_games_ctx if d in dyad_games_ctx[c]))
        A(f"| {names} | {n} | {ctxs} |")
    A("")

    # (b)
    A("## (b) Distinct partners per focal player\n")
    A("| player | games | partners (all) | mixed | mens | womens |")
    A("|:--|--:|--:|--:|--:|--:|")
    for want in FOCAL_NAMES:
        u = focal.get(want)
        if not u:
            A(f"| {want} | — | *not resolved — see notes* | | | |")
            continue
        A(f"| {name_of(u)} | {player_games[u]} | {len(partners[u])} | "
          f"{len(partners_ctx['mixed'][u])} | {len(partners_ctx['mens'][u])} | "
          f"{len(partners_ctx['womens'][u])} |")
    if focal_notes:
        A("\n**Focal-name resolution notes (human should confirm):**")
        for n in focal_notes:
            A(f"- {n}")
    A("")

    # (c)
    A("## (c) Partnership-graph connectivity, per context\n")
    A("Edges = played together. (Supplementary: components when opponent edges "
      "are added too, since margins also link the two sides.)\n")
    A("| context | players | components (partner edges) | component sizes | components (+opponent edges) |")
    A("|:--|--:|--:|:--|--:|")
    for ctx, r in comp_report.items():
        A(f"| {ctx} | {r['n_players']} | {r['n_components']} | "
          f"{', '.join(map(str, r['sizes']))}{'…' if r['n_components']>8 else ''} | "
          f"{r['n_components_with_opposition']} ({', '.join(map(str, r['sizes_with_opposition']))}) |")
    A("")
    A("Component membership of focal players (size of their component, partner-edge graph):\n")
    A("| context | " + " | ".join(w for w in FOCAL_NAMES) + " |")
    A("|:--|" + "--:|" * len(FOCAL_NAMES))
    for ctx, r in comp_report.items():
        cells = []
        for w in FOCAL_NAMES:
            cells.append(str(r["focal_component_sizes"].get(w, "—")))
        A(f"| {ctx} | " + " | ".join(cells) + " |")
    A("")

    # (d)
    A("## (d) Margin distributions\n")
    A(f"Computed on `sideout_11` rows only ({len(s11)} games; {n_other_fmt} games in "
      "other formats excluded from this comparison — see format table above).\n")
    A("| slice | n games | mean margin | SD | mean \\|margin\\| |")
    A("|:--|--:|--:|--:|--:|")
    for t, dsc in by_tour.items():
        A(f"| {t} | {dsc['n']} | {dsc['mean']:+.2f} | {dsc['sd']:.2f} | {dsc['mean_abs']:.2f} |")
    for n, dsc in by_gn.items():
        if dsc["n"]:
            A(f"| PPA game {n} | {dsc['n']} | {dsc['mean']:+.2f} | {dsc['sd']:.2f} | {dsc['mean_abs']:.2f} |")
    A("")
    A("Margin histogram (absolute margin, sideout_11 modeling rows):\n")
    for tour in ("MLP", "PPA"):
        vals = [abs(g["margin"]) for g in s11 if g["tour"] == tour]
        A(f"\n**{tour}**\n")
        A("| \\|margin\\| | games | |")
        A("|---:|---:|:---|")
        L.extend(hist_ascii(vals, [(2, 3, "2"), (3, 5, "3–4"), (5, 7, "5–6"),
                                   (7, 9, "7–8"), (9, 11, "9–10"), (11, 22, "11+")]))
    A("")

    # ---- automated interpretation ----
    A("## Interpretation & verdict\n")
    verdicts = []

    focal_dyad_counts = {tuple(sorted(name_of(p) for p in d)): n for d, n in focal_dyads}
    weak = {k: v for k, v in focal_dyad_counts.items() if v < 10}
    strong = {k: v for k, v in focal_dyad_counts.items() if v >= 10}
    verdicts.append(f"- **(a)** {len(strong)} focal dyads have ≥10 games; "
                    f"{len(weak)} are under 10 (heavy shrinkage for those).")

    ok_partners = all(
        len(partners[focal[w]]) >= 2 for w in FOCAL_NAMES if focal.get(w))
    verdicts.append("- **(b)** " + ("every resolved focal player has ≥2 distinct partners — "
                                     "actor/partner effects separable."
                                     if ok_partners else
                                     "⚠️ at least one focal player has <2 partners — their actor "
                                     "effect is confounded with that single partnership."))
    # context-level caveat: a player with exactly one partner IN A CONTEXT
    # cannot have context-specific actor and dyad effects separated within it
    ctx_caveats = []
    for w in FOCAL_NAMES:
        u = focal.get(w)
        if not u:
            continue
        for ctx in ("mixed", "mens", "womens"):
            n_games_ctx = sum(1 for g in modeling if g["context"] == ctx
                              and u in (g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]))
            if n_games_ctx >= 10 and len(partners_ctx[ctx][u]) == 1:
                only = name_of(next(iter(partners_ctx[ctx][u])))
                ctx_caveats.append(f"{name_of(u)} in {ctx} ({n_games_ctx} games, "
                                   f"only partner: {only})")
    if ctx_caveats:
        verdicts.append("- **(b-caveat)** single-partner *within a context*: "
                        + "; ".join(ctx_caveats)
                        + ". Within that context alone, actor and dyad effects for these "
                          "players are confounded — separation leans on their play in other "
                          "contexts (i.e., on the pooled-SRM structure).")

    for ctx, r in comp_report.items():
        main_sz = r["sizes"][0] if r["sizes"] else 0
        frac = main_sz / r["n_players"] if r["n_players"] else 0
        focal_in_main = all(v == main_sz for v in r["focal_component_sizes"].values()) \
            if r["focal_component_sizes"] else False
        verdicts.append(
            f"- **(c/{ctx})** giant component holds {main_sz}/{r['n_players']} players "
            f"({frac:.0%}); focal players "
            + ("all inside it." if focal_in_main else "⚠️ NOT all in the giant component."))

    g12 = [abs(g["margin"]) for g in ppa if g["game_number"] in (1, 2)]
    g3 = [abs(g["margin"]) for g in ppa if g["game_number"] == 3 and g["best_of"] == 3]
    if g3 and g12:
        m12, m3 = st.mean(g12), st.mean(g3)
        verdicts.append(
            f"- **(d)** PPA mean |margin|: games 1–2 = {m12:.2f}, game 3 = {m3:.2f} → "
            + ("compression confirmed (game 3 exists only after a 1–1 split), consistent "
               "with real, correctly-labeled data." if m3 < m12 else
               "⚠️ NO compression in game 3 — investigate before trusting the data."))
    if by_tour["PPA"]["n"] and abs(by_tour["PPA"]["mean"]) > 1:
        verdicts.append(
            f"- **(d-note)** PPA mean *signed* margin is {by_tour['PPA']['mean']:+.2f}: "
            "team one is not random (bracket convention favors the higher seed / "
            "qualifier winner), so signed margins are not zero-centered. Not a bug — "
            "but don't interpret raw team-one margins as symmetric.")
    mlp_sd, ppa_sd = by_tour["MLP"]["sd"], by_tour["PPA"]["sd"]
    if not math.isnan(mlp_sd) and not math.isnan(ppa_sd):
        gap = abs(mlp_sd - ppa_sd) / max(mlp_sd, ppa_sd)
        verdicts.append(
            f"- **(d)** margin SD: MLP {mlp_sd:.2f} vs PPA {ppa_sd:.2f} "
            f"({gap:.0%} relative gap) — "
            + ("comparable across tours, as expected for same-format side-out-to-11 games."
               if gap < 0.25 else "⚠️ large variance gap — check for scoring-format leaks."))
    L.extend(verdicts)
    A("")

    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")
    print("\n".join(verdicts))


if __name__ == "__main__":
    main()
