# Touch share: what to build next

Written 2026-08-22, at the end of the session that got naming to 88%
and localised everything that remains to a single stage. Read this
before touching `vision/touch_attribute.py`. Every number below is
**train-split, in-sample** — 15 rallies, 148 contacts, one game, one
camera. The holdout in `data/vision/label_split.csv` is frozen and
**unburned**.

Reproduce the current state with:

```
python3 touch_attribute.py --cascade contact,halves,depth \
    --video full_match.mp4.webm --geom-side --passes
```

---

## 1. Where it stands

**Naming is done for this footage.** Who hit each *true* contact:
**88%** (nearest-time join), geometry **91%**, team **100%**. The VLM
managed 44% identity at $2.59 for the same rallies; this costs nothing
and calls no API.

**The product metric is touch share WITHIN TEAM.** Sides alternate
exactly, so each side's total is fixed by the rally structure and
carries no information; the only estimated quantity is how a pair
split its own touches.

| team | player | pipeline | true | error |
|---|---|---|---|---|
| A | Emma Nelson | 48.9% | 49.3% | −0.4pp |
| A | Ting Chieh Wei | 51.1% | 50.7% | +0.4pp |
| B | Allyce Jones | 48.9% | 39.7% | **+9.2pp** |
| B | Etta Tuionetoa | 51.1% | 60.3% | **−9.2pp** |

Grade every change on this, not on the counts and not on the overall
share. Overall share flatters the pipeline because junk spreads across
all four players and cancels in a ratio; within a pair there is no
cancellation and a naming error moves share by *twice* its own size.
Errors are also lumpy — one wrong side bit moves a whole rally's worth
of one side's contacts, ~12 in a 25-contact rally — so team B's 9.2pp
is a couple of bits, not diffuse noise.

**The event list is the binding constraint.** 184 events emitted, 116
matched a true contact, **68 spurious (37%)**, and **32 of 148 true
contacts have no event on them**. Naming costs 26 misattributions
against that.

---

## 2. The one problem, isolated

```
stage                    events   recall    precision
0 candidates               2529   148/148
1 cluster merge             735   147/148
2 window trim               659   144/148
3 chain                     184   122/148   116/184 = 63%
```

**Passes 0–2 are effectively solved.** They cut 2529 candidates to 659
while keeping 144 of 148 true contacts.

**`swing_explore.decode_rally` is where everything is lost.** It
receives candidates covering 144 real contacts and returns 184 events
covering 122, at 63% precision. Fix that stage and the product is
close; nothing else on the page is worth an hour by comparison.

It fails in two separable ways.

### 2a. Trailing junk — the span constraint

> **0 spurious before the first true contact, 59 after the last, 9
> inside the rally.** Median 3.03s past any true contact, p90 7.85s.

This is not drift, it is obedience. `decode_rally`'s span constraint
requires the path to *"reach the neighborhood of the LAST confident
candidate"* — and between-point knocks score exactly as confidently as
real shots, so the DP is **required** to walk out to them.

### 2b. Missed contacts — the objective

> **26 true contacts carry no event. All 26 had a candidate within
> 0.35s, and 24 of those were at or above the 70th percentile the DP
> itself calls confident.**

The DP is walking past strong evidence. That is a mis-specified
objective, not a weak scorer and not a detection failure — placement
recall confirms a scored detection exists within 0.35s of *every* true
contact.

Two specific suspects, both in `decode_rally`:

- **`ref` is the 70th percentile of each rally's OWN candidates**, so
  by construction only ~30% of candidates can ever have positive gain.
  In a rally where the scorer is uniformly weak that floats down and
  admits junk; where it is strong it floats up and locks real contacts
  out. A per-rally quantile is a strange thing to hang a physical
  decision on.
- **The timing model was never fitted.** `min_gap` 0.25, `max_gap` 3.0,
  and a gap bonus of 0 inside 0.45–2.2s, −1.2 in 0.3–0.45, −3.0
  outside. The labels say real gaps are p01 0.37 / p25 0.69 / **med
  0.96** / p95 1.53 / p99 2.07, and same-side gaps p01 0.83 / med 1.92.
  8% of real gaps fall in the penalised <0.45 band.

---

## 3. What to build, in order

### Target 1 — re-specify the DP objective (the whole game)

Replace the hand-tuned gain and the span constraint with a fitted one.
The shape that fits the evidence:

- **Per-event cost from a calibrated score**, not a per-rally
  quantile. Fit P(true contact | score) on the train labels once and
  use log-odds, so a rally full of weak candidates does not have its
  bar lowered to admit them.
- **Gap likelihood from the measured distribution**, replacing the
  piecewise bonus — a log-density over the real inter-contact
  interval, which is what the bands were guessing at.
- **Drop or bound the span constraint.** It exists because v4 quit
  early in weak dink stretches (r9 decoded 1 of 29 events), so it
  cannot simply be deleted — but "must reach the last confident
  candidate" should become "must reach the last candidate consistent
  with continuous play", which the gap likelihood already expresses.

**Gate:** within-team share error for team B below 5pp, with stage-3
recall not falling below the 122 it starts at. Report the funnel every
time — a change that raises precision by destroying recall is a
regression however good the junk number looks.

### Target 2 — reconcile the `--geom-side` regression

`--geom-side` is clearly right (ghosts 8→0, alternation overwrites
7→0, events on an unlabelled track 17→0, missed contacts 44→24,
geometry 89%→91%, `contact` voter 88%→100% including 13/13 disputed)
**and attribution fell 88% → 74%.** Elimination re-assignments went
7→41 at the same time. Something downstream disagrees with the new
sides. A shared-anchor fix was tried and was a null — every digit
identical — so that hypothesis is dead.

Do not ship the flag as default until this is explained.

### Target 3 — log↔video sync (only if 1 and 2 are done)

Would give a rally-end boundary good to ±1–2s and kill 2a outright.
**It does not currently work, measured 2026-08-22** — see §5.

---

## 4. Do NOT rebuild these. All measured, all null or harmful.

| thing | result |
|---|---|
| `ball` voter (who is nearest the ball) | 62%, never decides under any ordering |
| `intent` voter (intrusion/purpose/netward) | 29% |
| `approach` voter | 0/3 |
| `movement` voter (raw displacement) | 59%, 50% on disputes |
| veto rules (leader overturned by dissent) | every variant worse than no-veto |
| diagonal / cross-court serve constraint | **harmful, 91%→74%** |
| voter re-ordering | `contact,halves,depth` is rank 1 of **720** |
| adding a 7th voter | 4 of 6 existing voters never decide anything |
| settled anchor | no effect |
| shared anchor decoder↔naming | null, every digit identical |
| motion-based track selection | −9 points |
| widened `contact` voter | −5 points |
| window trim from last confident candidate | left 62 junk events |
| window trim from candidate gaps | made the window *looser* (687 vs 637) |
| rally end from player motion | fires 15/15, **overshoots +8.97s median** |
| VLM as namer | 44% identity vs our 88%, ~$30/match |
| VLM as junk rejector | pointless — the junk is dead time, not mid-rally |

**Leave-one-out is unambiguous:** removing `depth`, `ball`, `intent`
or `approach` changes the best achievable score by *nothing*. Only
`contact` (−12) and `halves` (−8) carry weight.

---

## 5. Traps

**The referee log's timing does not align to this video.** Measured
2026-08-22 against `rally_windows_chicago0725_v4.csv` and
`contact_labels_chicago0725.csv`:

- log duration minus label span: med **+12.5s**, range −10.8 to +24.7
- serve lead (first contact − `t0s`): med **−19.3s**, range −58.1 to +12.5
- `t1s` minus last contact: med **+30.8s**, range −16.7 to +65.8

The drift *oscillates* (r1/r2 +1.4s, r3–r8 −30s, r9/r10 +1.5s, r13–r16
−58s), so it is not a correctable offset. And **r9 is internally
contradictory** — log duration 17.0s against a label span of 27.76s,
i.e. play outlasting the referee's whole logged rally. The log's
*content* is excellent (receiver prediction 34/34 = 100% comes from
it); its *timestamps* are not usable here without solving sync first.

**Grading joins.** ATTRIBUTION's index join (`called[k]` vs
`truth[k]`) is destroyed by one spurious detection early in a rally —
that is the entire 72% vs 88% gap. Use the nearest-time join for
naming; keep both printed, because the difference between them *is*
the over-counting.

**A diagnostic must never be able to move a result.** The order
sweep's loop once rebound `ok`, the ATTRIBUTION numerator, and the
headline printed 18% instead of 72% with every other number on the
page unchanged — indistinguishable from a regression. `ok`/`tot` are
now snapshot where the grading loop ends, with an assert.

**`--selftest` never enters `run()`.** Three `UnboundLocalError`s
shipped green in one session. Every per-run accumulator is now
declared in one block above all loops.

**Measure the detector, do not tune it.** Four window builds moved the
junk 70→68 and made the product metric worse, none of them ever
compared against the true last contact — which is one line of labels.
The single run that finally measured it answered the question
immediately.

**Refutation by firing rate.** The diagonal fired on 8 of 15 rallies
when only 4 bits were ever wrong. A constraint that fires more often
than there are errors is refuted by its own behaviour before any
accuracy number is needed.

---

## Sweep results (2026-08-22, replay harness — do not re-run these)

All on the train split, full pipeline via `--segs-from dump4.json`
(no video), baseline = `--geom-side --end-rule cross --chain soft`
at 25 missed / 47 junk / 82% nearest-time / shares +4.4 / +3.0.

1. **Fast-gap hypothesis: DEAD.** Missed contacts sit at median
   0.78s from their true neighbours — same as hits (0.80s), with
   FEWER sub-0.45s gaps (8% vs 15%). The DP does not skip firefight
   partners.
2. **The misses are rally-clustered ADJACENT SPANS** (r1 ×7, r17 ×5,
   r5 ×4): skipping two-in-a-row keeps parity across one ~1.3s FREE
   gap, while including the pair costs two tag penalties.
3. **Flat emission bonus (`--event-bonus`): bad trade.** Buys recall
   at ~2 junk per recovered contact (bonus 1.0: 12 missed, 94 junk).
   Lowering `--side-pen` below 0.9 is pure loss at every bonus.
4. **Ball-corroborated bonus (`--ball-bonus`): NULL BY SATURATION.**
   796 segment endpoints / 287s = one per 0.36s, so a ±0.5s gate is
   true everywhere and the knob degenerates to the flat bonus
   (bb 0.4 → 25/65 vs eb 0.4 → 24/66). A sparse ball event
   (crossings, ~7/rally) could still discriminate; endpoints cannot.

Conclusion: the recall/junk frontier does not move by constant
tuning. Defaults stay neutral (the 82% / ±4.4 / ±3.0 build). The
next lever on the event list is the temporal model
(vision/temporal_gate.md), not another DP knob.
