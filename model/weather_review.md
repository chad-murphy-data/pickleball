# Weather methods review — second pass, neutral eye

**Commissioned 2026-07-28, completed 2026-07-31.** The 2026-07-28 weather
thread (PR #37) concluded "six hypotheses, six nulls" and the project owner
suspected it had leaned toward rejecting. This is the independent re-check.

**Method.** Eighteen agents in three waves: four code and pipeline audits, a
web-sourced verification of every venue label, a power/inference audit, ten
re-runs and new tests, and eleven adversarial verifiers whose standing brief
was that over-claiming a null and over-claiming a signal are equally errors.
Every load-bearing number was re-derived by an independent route. Scripts are
in `model/weather_review/`; the phase-1 agent reports are preserved verbatim
in `model/weather_review_interim.md`.

**Verdict in one line.** The original code was clean and its bottom line —
that wind does not measurably move pro pickleball outcomes — survives. Its
*reasoning* largely does not: the falsification argument it leaned on was
built on mislabeled venues and a selection artifact, one of its "killed"
results is reproduced out-of-sample at its published magnitude, and "six
nulls" compressed one strong null, several underpowered ones, and one
unresolved question into a single rhetorical unit. The defensible claim is
not "wind does nothing" but **"any wind effect is smaller than ~1.5 pp of
point share, and every well-identified design puts it slightly negative."**

---

## 1. What the number actually is

The single most important output of this review is that the wind-compression
coefficient is **not zero-centred — it is consistently, mildly negative**, in
independent designs that do not share data-processing paths:

| design | d (skill × wind, per +10 mph) | 95% CI |
|---|---|---|
| B2a corrected labels, game level | −0.038 | [−0.104, +0.017] |
| C3 no fixed effects (L0) | −0.038 | [−0.096, +0.014] |
| C3 event FE (L2) | −0.050 | [−0.124, +0.017] |
| C3 player FE (L3, bug-corrected) | −0.048 | [−0.114, +0.014] |
| C3 event × skill FE (L6) | −0.078 | [−0.144, +0.008] |
| B6 de-attenuated | −0.040 | [−0.100, +0.021] |
| B3 rally level, all matches | −0.019 | [−0.048, +0.013] |

Not one interval excludes zero. Every point estimate has the same sign, and
the sign is the one the folklore predicts. Seven designs agreeing in
direction is weak evidence individually and is not significance — but
"consistently −0.04 to −0.08, never separable from zero" is a materially
different statement from the published "d = +0.002, no compression at all."

**The published +0.002 was a labeling artifact.** It does not reproduce once
venues are labeled correctly.

### The bound, which is the publishable part

After de-attenuation (B6), over 24,718 outdoor games at 0–18 mph of
reanalysis wind:

- any true compression larger than **1.5 pp of point share**, or **3.9 pp of
  win probability for an 84% favourite between 5 and 20 mph**, is excluded;
- any inflation of outcome variance beyond **+1.2 to +2.5 pp of upset rate**
  (the range depends on how the z² excess is attributed — the single "+1.5 pp"
  figure was an artifact of mixing two conventions) is excluded.

Measured reliability of the wind regressor is λ = 0.941, so the "the wind
proxy is too noisy to see it" defence mostly fails: timing error can hide at
most 6% of any slope. What remains unbounded is ERA5 grid wind vs true
on-court wind, and at λ_S ≤ 0.7 a genuine threshold effect *would* be
diluted below detection. That caveat is real and should not be dropped.

---

## 2. What the original got wrong

Ranked by how much it matters.

**2.1 The venue labels were wrong on 26% of games, in both directions.**
A web-sourced audit of all 77 label-bearing events (`data/venue_overrides.csv`,
with per-event evidence) found 5,989 games flipped indoor→outdoor, 3,540
outdoor→indoor, 3,231 mixed-venue, 1,442 unknown. Nine of 23 MLP events were
outdoor events stamped indoor; the PPA "indoor" arm was only ~28% genuinely
indoor. Every published weather test used these labels.

**2.2 "Indoor again more negative" was a selection artifact — delete it.**
The published rally-level dismissal rested on indoor (−0.060) being more
negative than outdoor (−0.017). On all matches rather than deciders only,
indoor collapses to −0.013 and outdoor is −0.019. The published contrast was
a property of the decider sample, not of indoor play. Conditioning on
deciders halves the apparent rally-level skill slope (0.574 → 0.289), which
is what a collider does.

**2.3 "Favorites × wind KILLED" overstated a result the review reproduces.**
The published binned finding was a −2.0 pp wind-attributable drift in the
14–20 mph bin. C2 fitted a threshold specification **out of sample, event
cross-fit, with corrected labels** and got **−2.2 pp [−4.04, +0.20]** — the
published magnitude, from an independent and more conservative route. The
"kill" was based on comparing that differential to a *level* (−6.0 pp), which
is not the same quantity. The honest status is "reproduced, still not
significant," not "killed."

**2.4 The 92 °F+ heat claim was never estimable.** Under the published labels
that bin was 62% one event played indoors in a hotel ballroom, and the drift
was −0.010 [−0.034, +0.028] — it never excluded zero on its own labels. The
published sentence should be retracted. (An earlier draft of this review
claimed the sign *reverses* under corrected labels; the verifier is right
that this over-claims a 295-game, 7-event, CI-spanning-zero contrast. Retract,
don't replace.)

**2.5 Stale numbers quoted beside the numbers that replaced them.** The
CLAUDE.md Design-B figures "+42 indoor ≈ +42 calm < +49 mod < +55 windy" come
from a superseded deciders-only table (n=2,830); current code gives
+39.8/+42.3/+44.5/+55.5 on n=4,786 — cited in the same sentence. Same for
"+0.13 z²" vs the correct +0.16. Record hygiene, not substance: the
calm-to-windy gradient is essentially unchanged.

**2.6 Named-player results with no code path.** `wind_skill.py` reports only
Anna Bright; the Waters / Tyra Black / Jack Sock / grinder-cohort numbers in
the narrative and on the public insights page were computed by machinery that
was never committed. Re-implementation reproduces Waters and Black closely and
the grinder cohort's membership exactly (5,134 games), but the pooled value
still differs in the third decimal.

**2.7 The insights page's Jack Sock sentence is loosely worded, not wrong.**
His naive one-sided p is 0.017, which contradicts "not statistically
significant" as written — but the multiplicity-aware instrument the code
actually uses is null (max|z| p = 0.58), Bonferroni p = 1.00, and with a date
control he falls to p = 0.053. He needs one added clause ("neither survives
the fact that I scanned 552 players"), not a correction notice.

---

## 3. What the original got right

- **The code is exact.** All four scripts regenerate their reports
  byte-identically; every RNG is seeded; no unseeded nondeterminism, and no
  hash-ordering dependence. Independent re-derivations reproduce the
  arithmetic essentially everywhere.
- **H1, serve-point rate, is a genuinely strong null** and is unmoved by
  relabeling: outdoor-minus-indoor +0.0007 [−0.0075, +0.0067] on a 0.45 base.
- **Heat does not level the field** — the physiological prediction fails.
  Outdoor skill × heat = +0.0165 [−0.0049, +0.0409] per +10 °F, the *opposite*
  sign, bounding any leveling at ≤0.23 pp for a 90% favourite over 20 °F. No
  match lengthening either.
- **Wind skill does not exist at any size worth naming.** The properly
  powered replacement for the split-half design bounds sd(player wind slope)
  at ≤0.018 point share per 10 mph, and the indoor placebo — where wind
  cannot physically act — reproduces the entire nominal signal (0.0154 vs
  0.0127). Style × wind is null with a tight bound.
- **No predictive value.** Applying the project's own adoption gate: no wind,
  gust, threshold or probability-flattening challenger improves the forecast
  on the frozen holdout or on a 30,982-game event cross-fit.

---

## 4. What this review found that is new

**4.1 The project's own holdout cannot adjudicate weather.** The frozen
post-2026-06-01 holdout is 65% indoor and its outdoor games top out at
11.9 mph. It needs d ≈ −1.65 for 80% power — a physically absurd effect. Any
future weather challenger gated on it will pass by construction. Weather
questions must be gated on the archive cross-fit instead.

**4.2 Temperature, not wind, is the live thread — and it runs the other way.**
The outdoor favourite edge rises monotonically across all six temperature
bins (+0.43 pp below 55 °F → +1.83 pp at 92 °F+), skill × temp = **+0.031 per
10 °F [+0.004, +0.059]**, surviving event fixed effects, leave-one-event-out,
and a horse race against a pure seasonal wave. Two agents reached +0.031
independently in different code. It is post-hoc, fails a Holm correction, is
absent in 2026, and its indoor placebo sits at −0.022 rather than 0. **Not
established — but it is the only thing in this review pointing anywhere, and
it deserves a pre-registered test rather than a paragraph.**

**4.3 The sport absorbs wind, texturally.** At fixed skill gap, wind does not
change side-out structure (−0.0035 side-outs/point per 10 mph), blowout rate
(−0.8 pp), mean margin (−0.09 pts) or residual margin variance (+1.1%). Two
loose ends: deuce rate +1.37 pp [+0.01, +2.73] over unconditional games, and
pace, where games run **faster** in wind (−2.1 s/point per 10 mph; games 2+,
which use two true UTC stamps, give −2.34 [−4.68, 0.00]). The pace effect
lives entirely between days, not within them, so it is probably a day-level
confound — but "the control abstained" is the accurate description, not "the
control refuted."

**4.4 Rally-level, collider-free.** 989,700 rallies, 682k outdoor across 68
events: outdoor d = −0.019 [−0.048, +0.013], CI 60% narrower than the
published decider-only version at the same point estimate. MDE |d| = 0.044,
i.e. −2.5 / −3.9 / −4.5 pp for a 65 / 75 / 90 % favourite at 20 mph.

**4.5 Between-event heterogeneity in d dwarfs the effect under test.**
Different event subsets give d from +0.114 to −0.111 (permutation p = 0.0017).
It is not specific to audit status — comparable gaps appear on latitude and
year cuts — and it is largely *player* composition, not event composition:
under player FE the gap halves to −0.093 [−0.194, +0.045]. Any future
weather estimate on this archive should carry player fixed effects.

---

## 5. Where this review was itself wrong

The verifiers caught this pass making the same class of error it was
commissioned to find. Recorded here so the record is symmetric:

- **The end-effect DiD was a bootstrap bug.** `b2b_did.py` silently
  de-duplicated resampled events, making every interval ~2× too narrow. Fixed,
  **no DiD arm excludes zero** (windy: [−0.113, +0.890]). The placebo was also
  unmatched on the collider; matched, it reads +0.11 to +0.14 rather than
  zero and the DiD halves to p = 0.21–0.26. The court-end-in-wind question is
  **unresolved and underpowered**, not "suggestive evidence."
- **Phase 1's "nominally significant" paired +0.401 was one weighting.** The
  pre-specified fixed-effects weighting gives +0.375 [−0.087, +0.739],
  randomization p = 0.079. No weighting, label arm or filter reaches p < 0.05.
- **"The published spec is the 3rd-percentile closest-to-zero of its own
  864-member family" has no discriminating power.** Under a pure null a draw
  lands that close to zero 4.6% of the time; observed 3.2% is chance, the
  grid's median |r| equals the null expectation to two decimals, and the
  *signed* placement is percentile 59 — mid-curve. The spec curve is a null
  curve. Retracted.
- **"The published binning hid the gust channel" is a base-rate fallacy.**
  95.4% of all outdoor games sit in the calm/moderate sustained bins, so gusty
  games are 8.7× *enriched* in the windy bin; corr(sustained, gust) = 0.927.
  The sustained binning sorted gust exposure nearly perfectly. Retracted.
- **The B3 dose-response death was slightly over-stated.** Properly
  standardised, match-length composition accounts for ~85% of the top-bin
  drop, leaving −0.013 ± 0.042 — a null, not a sign flip. Conclusion
  unchanged; wording was.
- **A C3 fixed-effects bug** (un-residualised intercept in the FWL second
  stage) produced the only CI that excluded zero. Corrected, none do — and
  correcting it also removed the claim that d "wanders" across the ladder. It
  does not; it is stably −0.04 to −0.08.

Ten of eleven verifications came back PARTIAL, one CONFIRMED. Nothing came
back REFUTED, and no verifier rubber-stamped.

---

## 6. Recommended corrections to the record

1. `CLAUDE.md` weather bullet: replace the rally-logit sentence with the
   all-matches version and **delete "indoor again more negative"** wherever it
   appears as a supporting argument.
2. `CLAUDE.md`: drop the stale +42/+49/+0.13 figures; keep +39.8/+42.3/+44.5/
   +55.5 and +0.16.
3. `CLAUDE.md`: strike "FAVORITES×WIND KILLED" and "no compression at all;
   b≈1.04 so v2 shares are near-perfectly calibrated." Replace with the
   corrected-label estimate and the de-attenuated bound.
4. `CLAUDE.md`: retract the 92 °F+ heat sentence.
5. `web/insights/wind/index.html`: add the multiplicity clause to the Jack
   Sock sentence. **Outward-facing — not changed without your say-so.**
6. Consider regenerating `data/event_geo.csv` through
   `scraper/weather.py`, which already consumes `data/venue_overrides.csv`,
   so downstream work inherits the corrected labels by default.
7. The 33 unaudited events (all heuristic-outdoor, 13,468 games) are the
   obvious next curation target.

---

## 7. What would actually settle it

- **Pre-register the temperature interaction** on 2026-07-31-forward data.
  It is the only live lead, it has a clean directional prediction (favourites
  *better* in heat), and prospective data costs nothing but patience.
- **A wind-exposure covariate that is not ERA5.** The unbounded λ_S is the
  one gap the archive cannot close; court-level exposure notes from
  broadcasts would do it.
- **Curate the remaining 33 event labels**, then re-run the corrected-label
  arms with player fixed effects as standard.

*Scripts: `model/weather_review/`. Venue evidence: `data/venue_overrides.csv`.
Phase-1 reports: `model/weather_review_interim.md`.*
