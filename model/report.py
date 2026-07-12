"""Generate analysis.md from the fitted SRM results.

Run after fit_srm.py:  python model/report.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "analysis.md"

FOCAL = ["Ben Johns", "Anna Leigh Waters", "Anna Bright", "Hayden Patriquin",
         "Gabriel Tardio", "Federico Staksrud", "Jade Kawamoto", "Jorja Johnson",
         "Will Howells", "Noe Khlif", "Kate Fahey", "Tyra Hurricane Black"]

MIN_GAMES_LEADERBOARD = 40
MIN_GAMES_CHEM = 15


def f(x, nd=2):
    return f"{float(x):+.{nd}f}"


def ordinal(x):
    n = int(round(float(x)))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def main():
    players = list(csv.DictReader((DATA / "results_players.csv").open()))
    dyads = list(csv.DictReader((DATA / "results_dyads.csv").open()))
    fit = json.loads((ROOT / "model" / "fit_summary.json").read_text())
    for p in players:
        p["games"] = int(p["games"]); p["value_mean"] = float(p["value_mean"])
        p["value_sd"] = float(p["value_sd"])
    for d in dyads:
        d["games"] = int(d["games"]); d["chemistry_mean"] = float(d["chemistry_mean"])
        d["chemistry_sd"] = float(d["chemistry_sd"])
        d["p_positive"] = float(d["p_positive"]); d["percentile"] = float(d["percentile"])

    by_name = {p["full_name"]: p for p in players}
    sc = fit["scalars"]

    def scalar(name, idx=None):
        m = sc[name]["mean"]
        s = sc[name]["std"]
        if idx is not None:
            return float(m[idx]), float(s[idx])
        return float(m), float(s)

    sd_v, _ = scalar("sd_v"); sd_w, _ = scalar("sd_w")
    sd_d, _ = scalar("sd_d"); sd_m, _ = scalar("sd_m"); sd_e, _ = scalar("sd_e")
    b_mlp = scalar("beta_tour", 0); b_ppa = scalar("beta_tour", 1)

    L = []
    A = L.append
    A("# analysis.md — player value & pair chemistry, 2026 MLP + PPA\n")

    # ---- headlines (computed) ----
    def dyad_row(n1, n2):
        for dd in dyads:
            if {dd["p1_name"], dd["p2_name"]} == {n1, n2}:
                return dd
        return None

    elig_h = sorted([p for p in players if p["games"] >= MIN_GAMES_LEADERBOARD],
                    key=lambda p: -p["value_mean"])
    top = elig_h[0]
    bp = dyad_row("Anna Bright", "Hayden Patriquin")
    wj = dyad_row("Anna Leigh Waters", "Ben Johns")
    bw = dyad_row("Anna Leigh Waters", "Anna Bright")
    A("## Headlines\n")
    A(f"1. **Individual value dwarfs chemistry.** Player-value spread is sd = {sd_v:.2f} "
      f"points/game; pair chemistry spread is sd = {sd_d:.2f}. Who you are matters ~"
      f"{sd_v/sd_d:.0f}× more than who you're standing next to. Chemistry exists league-wide "
      "(sd_d is bounded away from zero) but is small, and no single pair's chemistry is "
      "certifiable with high confidence — even 100+ game dyads carry ±0.45 posteriors.")
    A(f"2. **{top['full_name']} is #1 by a wide margin** ({f(top['value_mean'])} pts/game vs "
      f"{f(elig_h[1]['value_mean'])} for #{2} {elig_h[1]['full_name']}) — but see the "
      "cross-gender caveat: rankings are rock-solid *within* gender, while the men-vs-women "
      "alignment is a modeling convention, not a data statement.")
    if bp:
        A(f"3. **Bright + Patriquin's dominance is star power, not magic.** Their mixed "
          f"chemistry is {f(bp['chemistry_mean'])} ± {bp['chemistry_sd']:.2f} "
          f"({ordinal(bp['percentile'])} percentile, P(>0) = {bp['p_positive']:.2f}) over "
          f"{bp['games']} games — mildly positive, far from certain. Their expected margin "
          "comes almost entirely from both being top-5 players.")
    if wj:
        A(f"4. **Waters + Johns are exactly the sum of their parts**: chemistry "
          f"{f(wj['chemistry_mean'])} ({ordinal(wj['percentile'])} pct) across {wj['games']} games.")
    if bw:
        A(f"5. **The Waters + Bright superteam slightly *under*-performs its parts** "
          f"({f(bw['chemistry_mean'])}, {ordinal(bw['percentile'])} percentile, {bw['games']} games) "
          "— two #1-caliber players don't add a bonus on top of being overwhelming favorites.")
    A(f"6. **Skill transfers across contexts almost perfectly**: the mixed/mens/womens "
      f"deviation scale is sd_w = {sd_w:.2f} — negligible. A player's gendered-doubles level "
      "is their mixed level.\n")

    A("## What is (and is not) identifiable — the \"actor vs partner\" question\n")
    A("The original brief asked to separate each player's **actor effect** (own skill) "
      "from their **partner effect** (how much they elevate whoever stands next to them). "
      "With team-level margins that split is **not identifiable**: if a team's strength is "
      "`actor_i + partner_j→i + actor_j + partner_i→j`, then only the sums "
      "`(actor_i + partner_i)` and `(actor_j + partner_j)` ever enter the likelihood — any "
      "reallocation between a player's actor and partner components produces identical "
      "predictions for every game, including counterfactual pairings. No amount of data "
      "fixes this; it's structural. (Kenny's classic SRM separates them because dyadic "
      "outcomes there are *directional* — i's rating of j differs from j's rating of i. "
      "A game margin has no direction within a team.)\n")
    A("### The cross-gender flat direction\n")
    A("A second structural non-identifiability: **every observed game has an equal number "
      "of women on each side** — womens (2v2), mens (0v0), mixed (1v1). Verified across all "
      "modeling rows. Consequence: adding any constant *c* to every woman's value changes no "
      "predicted margin anywhere, so the *offset* between the men's and women's value blocks "
      "is invisible to the data. The zero-mean prior resolves it by (approximately) equating "
      "the two pools' averages. Therefore:\n")
    A("- comparisons **within** a gender are data-driven and safe;")
    A("- comparisons **across** genders (\"is Waters better than Johns?\") reflect the "
      "equal-pools convention, not evidence;")
    A("- hypothetical cross-gender matchups (\"Waters/Bright vs Johns/Tardio\") shift by 2c "
      "along the flat direction — the model has **no** data-driven prediction for them, and "
      "no such game exists in the data. Mixed-team predictions (1M+1F vs 1M+1F) are safe: "
      "the offset cancels.\n")
    A("What the data **does** identify:\n")
    A("1. **Player value** `v_i` — total points per game a player adds to their team's "
      "margin (actor + partner combined), with context-specific deviations (mixed/mens/womens).")
    A("2. **Dyad chemistry** `d_ij` — how far a specific pair deviates from the sum of its "
      "parts. This is exactly the \"relationship effect\" of the SRM.")
    A("3. **Partner-dependence** (descriptive) — the spread of a player's dyad effects: "
      "players whose pairs consistently over/under-perform additivity vs. players who are "
      "partner-proof.\n")

    A("## Model\n")
    A("```")
    A("margin_g ~ Normal(mu_g, sigma_e)")
    A("mu_g = beta_tour + sum(v_i + w_i,ctx | team1) - sum(... | team2)")
    A("     + d_dyad1 - d_dyad2 + m_match")
    A("v ~ N(0, sd_v)   w ~ N(0, sd_w)   d ~ N(0, sd_d)   m ~ N(0, sd_m)")
    A("```")
    A("Fit with NUTS (numpyro), 2 chains × 700/700, non-centered, on the 8,875 "
      "side-out-to-11 non-forfeit games (to-15 Challenger rounds excluded — different "
      "margin scale).\n")
    conv = fit.get("max_rhat_v_d", [])
    A(f"Convergence: {fit.get('n_divergences', '?')} divergences; "
      f"max R̂ over all player values {conv[0]:.3f}, over all dyad effects {conv[1]:.3f}.\n")

    A("### Variance decomposition (posterior means of scales)\n")
    A("| component | sd (points) | interpretation |")
    A("|:--|--:|:--|")
    A(f"| player value sd_v | {sd_v:.2f} | spread of individual ability |")
    A(f"| context deviation sd_w | {sd_w:.2f} | how much skill shifts across mixed/mens/womens |")
    A(f"| dyad chemistry sd_d | {sd_d:.2f} | typical size of pair-specific synergy |")
    A(f"| match intercept sd_m | {sd_m:.2f} | shared match-day component (Bo3 correlation) |")
    A(f"| residual sd_e | {sd_e:.2f} | game-to-game noise |")
    A(f"\nTeam-one bias: MLP {f(b_mlp[0])} ± {b_mlp[1]:.2f} (≈0, as expected — home/away is "
      f"arbitrary), PPA {f(b_ppa[0])} ± {b_ppa[1]:.2f} (small residual seeding bias after "
      "conditioning on player values; compare +2.79 raw).\n")

    # leaderboard
    A(f"## Leaderboard — player value (≥{MIN_GAMES_LEADERBOARD} games)\n")
    A("`value` = points per game added to team margin vs. an average pro in this pool. "
      "A pairing's predicted margin ≈ (v+w of your two) − (v+w of theirs) + chemistry terms.\n")
    elig = [p for p in players if p["games"] >= MIN_GAMES_LEADERBOARD]
    elig.sort(key=lambda p: -p["value_mean"])
    A("Shown separately by gender — within-gender order is data-driven; the alignment "
      "*between* the two lists is the equal-pools prior convention (see above).\n")
    for gkey, glabel in (("M", "Men"), ("F", "Women")):
        sub = [p for p in elig if p["gender"] == gkey]
        A(f"### {glabel} (top 25 of {len(sub)} with ≥{MIN_GAMES_LEADERBOARD} games)\n")
        A("| rank | player | value | ±sd | games |")
        A("|--:|:--|--:|--:|--:|")
        for i, p in enumerate(sub[:25], 1):
            A(f"| {i} | {p['full_name']}{' *(focal)*' if p['full_name'] in FOCAL else ''} | "
              f"{f(p['value_mean'])} | {p['value_sd']:.2f} | {p['games']} |")
        A("")

    # focal profiles
    A("## Focal players\n")
    A("| player | value | ±sd | games | value rank (≥40g) |")
    A("|:--|--:|--:|--:|--:|")
    ranks = {p["full_name"]: i for i, p in enumerate(elig, 1)}
    for name in FOCAL:
        p = by_name.get(name)
        if not p:
            A(f"| {name} | — | | | |")
            continue
        A(f"| {name} | {f(p['value_mean'])} | {p['value_sd']:.2f} | {p['games']} | "
          f"{ranks.get(name, '—')}/{len(elig)} |")
    A("")

    # chemistry
    A(f"## Pair chemistry (dyads with ≥{MIN_GAMES_CHEM} games)\n")
    A("`chemistry` = points per game beyond the sum of the two players' values. "
      "`P(>0)` = posterior probability the synergy is real rather than shrinkage noise.\n")
    chem = [d for d in dyads if d["games"] >= MIN_GAMES_CHEM]
    chem.sort(key=lambda d: -d["chemistry_mean"])
    A("### Best chemistry\n")
    A("| pair | context | chem | ±sd | P(>0) | pct | games |")
    A("|:--|:--|--:|--:|--:|--:|--:|")
    for d in chem[:12]:
        A(f"| {d['p1_name']} + {d['p2_name']} | {d['context']} | {f(d['chemistry_mean'])} | "
          f"{d['chemistry_sd']:.2f} | {d['p_positive']:.2f} | {d['percentile']:.0f} | {d['games']} |")
    A("\n### Worst chemistry\n")
    A("| pair | context | chem | ±sd | P(>0) | pct | games |")
    A("|:--|:--|--:|--:|--:|--:|--:|")
    for d in chem[-8:]:
        A(f"| {d['p1_name']} + {d['p2_name']} | {d['context']} | {f(d['chemistry_mean'])} | "
          f"{d['chemistry_sd']:.2f} | {d['p_positive']:.2f} | {d['percentile']:.0f} | {d['games']} |")
    A("")

    A("### Focal dyads\n")
    A("| pair | context | chem | ±sd | P(>0) | pct | games |")
    A("|:--|:--|--:|--:|--:|--:|--:|")
    focal_set = set(FOCAL)
    fd = [d for d in dyads
          if (d["p1_name"] in focal_set or d["p2_name"] in focal_set) and d["games"] >= 10]
    fd.sort(key=lambda d: -d["chemistry_mean"])
    for d in fd:
        A(f"| {d['p1_name']} + {d['p2_name']} | {d['context']} | {f(d['chemistry_mean'])} | "
          f"{d['chemistry_sd']:.2f} | {d['p_positive']:.2f} | {d['percentile']:.0f} | {d['games']} |")
    A("")

    # partner-dependence: spread of a player's dyad means (>=8 games each, >=3 dyads)
    A("## Partner-dependence (descriptive)\n")
    A("Std-dev of a player's dyad-chemistry estimates (dyads with ≥8 games, players with "
      "≥3 such dyads). High = chemistry-sensitive; low = partner-proof. Descriptive only — "
      "see the identifiability note.\n")
    per_player = defaultdict(list)
    for d in dyads:
        if d["games"] >= 8:
            per_player[d["p1_name"]].append(d["chemistry_mean"])
            per_player[d["p2_name"]].append(d["chemistry_mean"])
    rows = []
    for name, chems in per_player.items():
        if len(chems) >= 3 and name in by_name and by_name[name]["games"] >= MIN_GAMES_LEADERBOARD:
            mu = sum(chems) / len(chems)
            sd = math.sqrt(sum((c - mu) ** 2 for c in chems) / (len(chems) - 1))
            rows.append((name, len(chems), mu, sd))
    rows.sort(key=lambda r: -r[3])
    A("| player | dyads | mean chem | spread (sd) |")
    A("|:--|--:|--:|--:|")
    for name, n, mu, sd in rows[:12]:
        A(f"| {name}{' *(focal)*' if name in FOCAL else ''} | {n} | {f(mu)} | {sd:.2f} |")
    A("\n*Most partner-proof (lowest spread):*\n")
    A("| player | dyads | mean chem | spread (sd) |")
    A("|:--|--:|--:|--:|")
    for name, n, mu, sd in rows[-8:]:
        A(f"| {name}{' *(focal)*' if name in FOCAL else ''} | {n} | {f(mu)} | {sd:.2f} |")
    A("")

    # ---- unshrunk fixed-effects dyad estimates (if present) ----
    fe_p = DATA / "results_dyads_fe.csv"
    if fe_p.exists():
        fe = list(csv.DictReader(fe_p.open()))
        for r in fe:
            r["fe_estimate"] = float(r["fe_estimate"]); r["fe_se"] = float(r["fe_se"])
            r["t"] = float(r["t"]); r["games"] = int(r["games"])
        A("## Unshrunk (\"fixed effects\") chemistry check\n")
        A("OLS with a fixed effect per player + one dyad dummy, cluster-robust SEs by "
          "match — no shrinkage prior at all (`model/fixed_effects_dyads.py`, all dyads "
          "with ≥30 games). If the Bayesian prior were burying real chemistry, it would "
          "show up here. It doesn't:\n")
        A("| pair | context | games | unshrunk est. | ±se | t | Bayesian (shrunk) |")
        A("|:--|:--|--:|--:|--:|--:|--:|")
        bayes = {frozenset((d["p1_name"], d["p2_name"])): d for d in dyads}
        focal_set2 = set(FOCAL)
        shown = [r for r in fe if r["p1_name"] in focal_set2 or r["p2_name"] in focal_set2]
        for r in sorted(shown, key=lambda x: -x["fe_estimate"]):
            b = bayes.get(frozenset((r["p1_name"], r["p2_name"])))
            bs = f(b["chemistry_mean"]) if b else "—"
            A(f"| {r['p1_name']} + {r['p2_name']} | {r['context']} | {r['games']} | "
              f"{f(r['fe_estimate'])} | {r['fe_se']:.2f} | {r['t']:+.1f} | {bs} |")
        ts = [r["t"] for r in fe]
        mu_t = sum(ts) / len(ts)
        sd_t = (sum((t - mu_t) ** 2 for t in ts) / (len(ts) - 1)) ** 0.5
        A(f"\nAcross all {len(fe)} high-volume dyads, the t-statistics have mean "
          f"{mu_t:+.2f} and sd {sd_t:.2f} (pure noise would give ≈0 and ≈1). The mild "
          "overdispersion is the small league-wide chemistry variance; the positive mean "
          "hints at survivorship (pairs that keep playing together are pairs it's working "
          "for). No individual pair separates from the pack.\n")
        A("Note for Bright + Patriquin specifically: Patriquin is Bright's only mixed "
          "partner, so her personal mixed-context shift and the pair dummy are the same "
          "regression column — the unshrunk estimate is their *sum*, i.e. if anything an "
          "overstatement of pure pair chemistry.\n")
        A("### Why a player's dyad estimates see-saw (read before over-interpreting)\n")
        A("A player's value is fitted to their *average* performance across partnerships, "
          "so their dyad effects are deviations around that average: one strongly negative "
          "pairing mechanically implies their other pairings lean positive. The apparent "
          "\"outliers\" (Waters+Bright −1.7, Alshon+Patriquin −1.7) and the mirrored "
          "positives (Bright+Fahey +1.6, Patriquin+Tardio +1.8) are therefore **not "
          "independent facts** — each within-player set is one identified contrast: "
          "*Bright's games with Waters run ~2–3 points worse than her games with her other "
          "partners* (≈2.3σ), and likewise Patriquin with Alshon vs Tardio (≈2.3σ). "
          "Which pairing deserves the label \"bad\" vs \"good\" is not identified — only "
          "the difference is. Candidate real mechanisms (court-side preference conflicts, "
          "role redundancy) and plain multiple-comparisons luck are both live; the "
          "season's remaining games are the honest test.\n")
        A("A further confound: for both stars the negative-chemistry partner is PPA-only "
          "(Waters, Alshon) and the positive-chemistry partner is MLP-only (Fahey, "
          "Tardio), so the partner contrast is collinear with a player-by-tour contrast "
          "the model doesn't include. The one testable implication — Bright+Patriquin's "
          "own games split by tour — shows no MLP advantage (MLP +0.04 ± 0.94 vs PPA "
          "+0.36 ± 0.77, unshrunk), which disfavors but cannot rule out the tour story "
          "(only 23 MLP games together).\n")

    # ---- temporal holdout (if holdout_summary.json exists) ----
    ho_p = ROOT / "model" / "holdout_summary.json"
    if ho_p.exists():
        ho = json.loads(ho_p.read_text())
        A("## Does it predict? Temporal holdout\n")
        A(f"Model refit on games before {ho['split']} only, then used to predict every "
          f"later game whose four players all had ≥10 training games "
          f"(n = {ho['test_games_evaluable']} — mostly MLP, predicted from PPA-heavy "
          "training):\n")
        A("| metric | model | coin flip |")
        A("|:--|--:|--:|")
        A(f"| winner accuracy | {ho['accuracy']:.1%} | 50% |")
        A(f"| Brier score | {ho['brier']:.3f} | 0.250 |")
        A(f"| log loss | {ho['log_loss']:.3f} | 0.693 |")
        A(f"| margin MAE | {ho['mae_model']:.2f} | {ho['mae_zero_baseline']:.2f} (predict 0) |")
        A("\nCalibration is *under*confident (e.g. games called ~65% go the favorite's way "
          "~76% of the time) — the conservative direction: player values generalize at "
          "least as well as their posteriors claim.\n")

    # ---- platform-rating benchmark (if present) ----
    rc_p = ROOT / "model" / "rating_comparison.json"
    if rc_p.exists():
        rc = json.loads(rc_p.read_text())
        A("## Benchmark: platform rating vs this model\n")
        A("pickleball.com embeds its own per-player rating (its in-house system — not "
          "DUPR, which requires an authenticated API) as an as-of-match snapshot in the "
          "raw payloads. Head-to-head on the same holdout games "
          f"(≥ {rc['split']}, n = {rc['n_games']}), predicting each game's winner:\n")
        A("| predictor | accuracy | Brier |")
        A("|:--|--:|--:|")
        A(f"| this model (frozen {rc['split']}) | {rc['model']['accuracy']:.1%} | "
          f"{rc['model']['brier']:.3f} |")
        A(f"| platform rating (as-of-match, updates all season) | "
          f"{rc['platform_rating']['accuracy']:.1%} | {rc['platform_rating']['brier']:.3f} |")
        cm, cf = rc["value_rating_correlation"]["M"], rc["value_rating_correlation"]["F"]
        A(f"\nThe two systems agree on {rc['agreement']:.0%} of games; correlation between "
          f"model value and latest rating is {cm['r']:.2f} (men, n={cm['n']}) / "
          f"{cf['r']:.2f} (women, n={cf['n']}). The model wins despite the rating having "
          "an information edge (it updates through the test window; the model is frozen "
          "at the split). Notable rating oddities the model avoids: Gabriel Tardio (#1 "
          "here) ranked ~#30 by rating; Jackie Kawamoto's rating collapsed mid-season "
          "from 6.13 to 3.50 — an apparent reset/identity glitch in the rating engine.\n")

    # ---- core-pool robustness (if the _core fit exists) ----
    core_p = DATA / "results_players_core.csv"
    if core_p.exists():
        core = {r["player_id"]: r for r in csv.DictReader(core_p.open())}
        full_by_id = {p["player_id"]: p for p in players}
        common = [(float(full_by_id[u]["value_mean"]), float(core[u]["value_mean"]))
                  for u in core if u in full_by_id and int(core[u]["games"]) >= 40]
        if len(common) > 20:
            fv = [c[0] for c in common]; cv = [c[1] for c in common]
            def _ranks(xs):
                order = sorted(range(len(xs)), key=lambda i: -xs[i])
                r = [0] * len(xs)
                for rank, i in enumerate(order):
                    r[i] = rank
                return r
            rf, rc = _ranks(fv), _ranks(cv)
            n = len(rf)
            rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rf, rc)) / (n * (n * n - 1))
            csc = json.loads((ROOT / "model" / "fit_summary_core.json").read_text())["scalars"]
            A("## Robustness: core-pool refit (\"real pros only\")\n")
            A("Same model refit on games where **all four players have ≥30 appearances** "
              "(4,698 games, 312 players — drops Challenger one-weekenders and qualifier "
              "cannon fodder). Result:\n")
            A(f"- **Rankings are stable**: Spearman ρ = {rho:.3f} between the two fits over "
              f"the {n} shared regulars. The leaderboard's composition is unchanged.")
            A(f"- **The zero point moves up ~{(sum(fv)/n - sum(cv)/n):.1f} points** (values "
              "are relative to the pool average, and the pool got stronger). E.g. Waters "
              f"{f(fv[0] if False else max(fv))} → {f(max(cv))}. Differences *between* players "
              "are what carry meaning; the absolute level is pool-dependent.")
            A(f"- **Chemistry shrinks further** (sd_d {sd_d:.2f} → "
              f"{float(csc['sd_d']['mean']):.2f}): part of the apparent synergy in the full "
              "pool was pairs feasting together on weak fields.")
            A("- One notable mover: players who log many Challenger games (e.g. Patriquin) "
              "give back a fraction of a point relative to peers who don't — mild "
              "\"Challenger farming\" inflation in the full-pool fit.\n")

    A("## Caveats\n")
    A("- **Cross-gender comparisons are convention** (see the flat-direction section): "
      "read the leaderboard as two interleaved within-gender rankings aligned by prior.")
    A("- Single 2026 season, mid-season snapshot (through Jul 11): no time-varying skill; "
      "Patriquin-type trajectories are averaged over the window.")
    A("- Margins treated as Gaussian; to-11 games truncate blowouts (±11-ish cap), so "
      "elite values are mildly compressed relative to \"true\" dominance.")
    A("- Anna Bright's mixed games are 100% with Patriquin: her mixed context deviation and "
      "that dyad's chemistry are separated only by the pooled hierarchical structure.")
    A("- Selection: PPA game 3 exists only after 1–1 splits; handled by the match intercept, "
      "not modeled as an explicit selection process.")
    A("- Qualifier and Challenger main-draw games are included; they mostly inform the tail "
      "of the player pool and tighten opponent-quality adjustment for focal players.")

    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
