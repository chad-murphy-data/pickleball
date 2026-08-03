# Rally duration — the first DV outside the win/loss closure

*Session 2026-08-03. Sample: 400 matches, 21,614 clean rallies (100% log
fetch success). NOT the full archive — a first look.*

## Why this is different from everything else in the clutch thread

Every prior DV in this project is a function of **who won each rally**:
points, margins, win probability, hazards, hit rates. Skill is *defined* as
what predicts winning, so any second factor competes for residual variance in
a quantity the skill term was fitted to explain. That is why the zero-clutch
simulation (`clutch_endogeneity.py`) could reproduce every clutch index built
today — it generates the binary sequence, and the binary sequence *was* the
data.

Duration sits outside that closure. The simulation cannot reproduce it,
because it never modelled time.

## Where the data comes from

Referee logs carry `date_created` per entry, and `log_type 12` marks *rally
underway*, so the gap to the resolving entry (14 point / 16 side-out /
23 second server) is a per-rally duration. Validated against the explicit
`point_log` payload (`time_started` / `time_ended`), which matches exactly —
but only appears on point rows, so the type-12 gap is what covers all rallies.

Rallies within two log entries of a `timeout_log`, `line_review_log` or
`penalty_log` are dropped; those produced the 3-minute outliers. Kept range
2–90s. **Not in `pb_rally`** — this needs raw logs, so archive-scale work
means a backfill.

Overall: median 17.0s, mean 18.3s, sd 8.1.

## Big points are played differently

| score state | mean |
|---|---|
| early (both ≤ 4) | 18.0s |
| middle | 18.5s |
| someone ≥ 9 | 18.7s |
| **9+ and close** | **19.6s** |
| 9+ but blowout | 17.4s |

Confounded by matchup (close games select evenly-matched pairs, who rally
longer), so the test is **within match** — same four players, big-and-close
rallies against their own early rallies:

| contrast | effect | 95% CI |
|---|---|---|
| **big-and-close − early** | **+1.74s** | **[+1.24, +2.22]** |
| late-but-**decided** − early (control) | −0.27s | [−0.77, +0.26] |

72% of matches show it. The control is the load-bearing half: general
late-game slowdown or fatigue would move the decided arm too. It doesn't.

## No choke signature — and this is the interesting part

Duration separates the two ways to lose a rally: fast = an error, slow = a
battle you were outplayed in. It is the closest thing to an unforced-error
measure available without shot data.

Relative to each match's own average:

| situation | outcome | n | vs match avg |
|---|---|---|---|
| serving for the game | lost | 847 | −0.41s |
| serving for the game | won | 611 | +0.63s |
| staving off game point | lost | 879 | −0.24s |
| staving off game point | won | 466 | **+0.98s** [+0.18, +1.77] |
| ordinary early | lost | 7,518 | −0.38s |
| ordinary early | won | 5,619 | +0.04s |

**Rallies lost under pressure take the same time as ordinary lost rallies**
(−0.03s and +0.14s difference). Players do not tighten up and miss quickly.

The whole effect is in rallies **won**: +0.59s and +0.94s versus ordinary won
rallies. Pressure does not produce errors; it makes points **harder to
finish**. That also explains the +1.74s above — it is won-rally lengthening,
not a general slowdown.

## Limits

* Only *staving off / won* clears zero on its own; the other cells span it.
* **Prep vs rally is unresolved.** Median 17s is long for a pro rally
  (typically ~10s), so the type-12 marker probably fires at point start, not
  serve strike — meaning this may bundle serve preparation. "Players take
  longer *before* big points" and "big-point rallies are longer" are
  different claims and this cannot yet separate them. Resolvable with data
  that exists.
* Timestamps are referee taps, not true rally boundaries. Latency under
  pressure is a rival explanation, partly cancelled since both endpoints
  would shift.
* 400 matches, not the archive.

## Next

Separate prep from rally (compare the resolution→next-type-12 gap, which is
pure between-point time, against the type-12→resolution gap). If the effect
lives in the between-point gap it is a pacing/routine finding; if in the
rally itself it is a play-style finding. Then per-player contrasts — but
those need the within-match design, since who reaches pressure states is
skill-determined.

## "Make them earn it" — Anna Bright's stated strategy, tested

Bright has written about not missing on match point: make the opponent earn
it. That predicts something specific and checkable — when you are defending
game point and lose, the rally should run LONG. A short loss is the gift.

**Design note (the important part).** The first cut used only game-ENDING
rallies: 374 in 400 matches, 64 in tight situations, and a maximum of **2 per
player**. Useless per-player. The fix is to condition on rallies that COULD
have ended the game — every rally with the server at game point — not the
ones that did. A game only ends when the serving side converts, so restricting
to enders silently drops every game point the defender SAVED, which is the
half Bright is actually talking about. Conditioning on the outcome threw away
the informative cases.

That reframe takes per-player counts from max 2 to max 45; 32 players at ≥10,
14 at ≥20.

| | n | vs that game's average |
|---|---|---|
| game point converted | 605 | +0.52s |
| game point saved by defender | 843 | −0.35s |
| ordinary won | 5,591 | +0.07s |
| ordinary lost | 7,475 | −0.39s |

| contrast | effect | 95% CI |
|---|---|---|
| converting a game point vs an ordinary point | +0.44s | [−0.25, +1.14] |
| saving a game point vs an ordinary lost rally | +0.04s | [−0.40, +0.51] |
| (game-enders only) tight vs blowout ending | +1.48s | [−0.62, +3.68] |

**Nothing clears zero.** All three cuts point Bright's way, but they are the
SAME RALLIES sliced differently — three views of one dataset, not three
independent tests. Do not treat the agreement as accumulating evidence.

Honest state: a plausible ~0.5s effect with the right sign and now the right
design, where **the sample is the binding constraint rather than the method**.
That is the opposite of every clutch result in this thread, where more data
would not have helped because the design was broken. At archive scale
(~26,000 games) the game-ender CI alone would shrink from 4.3s wide to ~0.5s.

Per-player "does everyone make them earn it" remains out of reach until the
backfill, but is no longer hopeless: heavily-used pros would have hundreds of
game-point defences across the archive.

## Red / yellow / green — and a correction

Tennis has a red/yellow/green framing: green = swing freely, red = don't
miss. Crucially you return to GREEN when the situation is nearly lost (down
40-0, 5-0) — nothing to lose. That predicts a NON-MONOTONIC relationship
between score margin and rally length: longest when close, short at BOTH
extremes. A naive "pressure → patience" model predicts monotone and gets the
far-behind cell wrong.

Late rallies (someone ≥ 8), duration vs that game's average, by the server's
signed margin:

| margin | vs avg |
|---|---|
| behind 5+ | −0.10s |
| behind 2–4 | +0.43s |
| **within 1** | **+0.90s** |
| ahead 2–4 | +0.82s |
| ahead 5+ | −0.21s |

The shape is there, including the left tail returning to short.

**But the strict test does not support it.** Comparing late-and-close against
late-and-lopsided gives +1.06s [+0.55, +1.58] — and PAIRED WITHIN THE SAME
GAME it falls to **+0.41s [−0.11, +0.93]**, with only 52.4% of games showing
it (n=231 games visiting both states).

The flaw was mine: normalizing each rally against its own game's average does
NOT force the same game into both cells. Close states come overwhelmingly
from close GAMES — evenly matched pairs, who rally longer intrinsically — and
lopsided states from blowout games. Most of the +1.06s was between-game
matchup composition, the exact confound the normalization was supposed to
handle.

**Status of each duration finding after this:**

* **+1.74s** big-and-close vs the same match's early rallies — paired, with a
  flat control arm (late-but-decided −0.27s). **Stands.**
* **No choke signature** — losses under pressure are not faster. Different
  comparison, untouched. **Stands.**
* **Inverted U across margin** — substantially a matchup artefact. Residual
  within-game effect +0.41s, right sign, **not established**.

Red/yellow/green remains the best theory on the table — it makes the correct
non-monotonic prediction and the far-behind cell behaves as it says — but
whether that is players reading the scoreboard or good matchups producing
long rallies is not yet separable at n=231.

## Duration × outcome — the return advantage is a transient

First, the caveat that governs this whole section: **duration is an outcome,
not a choice.** A rally is long because both sides kept it alive, so
conditioning on it is conditioning on a post-treatment variable. "Win rate in
long rallies" is a valid conditional description; it is NOT evidence that
choosing to extend rallies causes wins.

As a description, though, the shape is clean (29,086 clean timed rallies):

| rally length | n | serving side wins |
|---|---|---|
| 2–6s | 494 | 43.9% |
| 7–10s | 2,737 | 41.1% |
| **11–15s** | **9,498** | **40.5%** |
| 16–22s | 10,165 | 42.5% |
| 23–32s | 4,620 | 46.0% |
| 33–90s | 1,572 | **47.8%** |

U-shaped, with the trough in the third-shot phase. The returning team's
structural advantage — they reach the kitchen first — bites hardest 11–15s
in, holding the server to 40.5%. Past ~30s both teams are established at the
net, the rally neutralises, and the serving side recovers to 47.8%, near the
50% you would expect once the serve no longer matters. **The return advantage
decays with rally length.**

## Per-player style: not computable at this sample

"Wins the long ones" as a player trait (win% on 22s+ rallies minus win% on
≤11s) is the natural next question and cannot be answered here:

* 36 players have ≥25 rallies in each bucket
* only **19** have enough in both halves of the data, so **split-half
  reliability is not computable** — and that is the test that decides whether
  this is a trait or noise
* excess-variance ratio 1.43, which means nothing without the reliability

Names do fall out of the ranking (most grinder / most finisher) on 25–60
rallies per cell. **Treat them as noise.** That is precisely the shape of the
Max Freeman z = +2.26 mirage in `clutch.md` — a standout on 650 serving
rallies that evaporated at 20× the sample.

Needs the archive backfill, like everything else here.

## Ben Johns, targeted (556 matches fetched specifically for him)

A random 520-match sample gave him only 334 timed serve rallies. Fetching
every match he appears in gives **9,164 clean timed serve rallies across 495
matches** — the best-sampled player available, so if a style trait is not
measurable for him it is not measurable for anyone.

**"Ben Johns plays long points" is a fact about his opponents.** His rallies
median 17s; everyone in his matches also medians 17s, against a general field
of 16s. Elite vs elite runs long. The raw statistic says nothing about him,
and is reported here only to be dismissed.

Net of that:

| | short (≤11s) | long (≥22s) | contrast |
|---|---|---|---|
| Johns | 46.2% | 49.8% | **+0.036** [+0.004, +0.067] |
| field in his matches | 40.4% | 46.2% | +0.058 |

His edge over the field is **+5.8pp on short rallies vs +3.6pp on long ones**
— concentrated in finishing, not grinding, which is the opposite of the
"grinds you down" story. **But** the gap between his contrast and the field's
is smaller than his own CI width, so "more of a finisher than his peers" is
NOT established. Only that his contrast is small and positive.

The contrast is orthogonal to quality by construction: a player who is
uniformly better scores zero. That is the point — it is a style axis, not a
ranking, and cannot tell you who is good.

**The deciding number does not decide.** Population split-half of the
contrast across servers in his matches: **r = +0.477, 95% CI
[−0.079, +0.843], n = 13 players**. Spearman-Brown full-length +0.646, which
would be a genuine trait — but the CI spans zero and 13 players is not a
basis for anything. Johns's own halves are at least self-consistent (+0.049
and +0.025, gap 0.024 against a CI width of 0.063).

Verdict: **plausible and undecided**, and undecided for the best-sampled
player in the sport. The ranking that falls out (Collin Johns +0.211, Daescu
+0.154, Alshon +0.082, Ben Johns +0.036, Waters +0.029, Tardio +0.025,
Patriquin −0.026) should be ignored until reliability clears — seven players
is a leaderboard of noise.
