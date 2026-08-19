# Temporal-model gate — pre-registered bars (FROZEN 2026-08-19)

Status: bars frozen while **no temporal-model code exists**, while the
label archive stands at 25 rallies (fingerprint a0d9248a35), and while
pose extraction covers only rallies 1–10. Successor to
`contact_gate.md` (Gate C, closed KILL) under the discipline that
closed it: exploration results are never verdicts; anything believed
graduates only via a pre-registration like this one, scored once on
untouched holdout.

## Context — what the checkpoint measured

The 2026-08-19 post-tagging checkpoint (`swing_explore_notes.md`,
final entry) routed REGIME 2 on pre-registered bands:

- **Level C** (true label times + the frozen GAP pace classifier),
  n=138 paced contacts / 10 rallies: 74.6% per-contact pace, and the
  structure stats clear — has_fast 10/10, first-fast ≤1 s 8/10
  (median 0.00 s), initiating team 8/10, fast-share corr +0.55.
- **Level B** (label-free decoded events, same rallies): per-contact
  match rate 45.7%, first-fast ≤4/10 (medians 0.79–5.04 s),
  init-team ≤4/10, has_fast 7/10 at best.

**Placement binds, pace classification doesn't.** The temporal model
is the proposed fix: a supervised sequence decoder trained on the
timestamped labels that decodes contact times AND pace states
JOINTLY — the pace state sets the expected next-gap distribution
inside the decoder (registered design input: in a firefight, expect
~0.4–0.8 s to the next contact, where today's pace-blind cadence
prior expects 0.45–2.2 s and ghosts through the exchange). Any model
class that beats that design on train is equally admissible; the
design input is a starting point, not a constraint.

## The decision this gate makes (once)

- **PASS** → label-free structure stats are near label quality on
  held-out rallies: the multi-VOD labeling push ("2000 rallies") and
  the decoded-events structure product are licensed.
- **MIDDLE** → one documented amendment cycle; the final shot runs on
  the FIRST FUTURE VOD's holdout block (Chicago holdout is burned).
- **KILL** → temporal decoding is dead on this footage class. The
  product falls back to labeled-matches-only (Level C — already
  justified by the checkpoint), the label archive keeps full value as
  its fuel, and the thread reopens only with materially different
  footage (higher fps/res or uncondensed source).

## Frozen evaluation

- **Instrument**: `vision/phase_grader.py` grading harness as of
  commit 512163a. MATCH_TOL_S = 0.5 s (same-team, one-to-one greedy),
  FF_TOL_S = 1.0 s, truth builder incl. the lunge rule. Grading
  constants may not move.
- **Panel**: the pre-registered Chicago holdout blocks
  (`data/vision/label_split.csv`: rallies 22–34, 55–67, 113–142,
  174–193 — 76 rallies, ~1,200 contacts), fully labeled (both
  passes) and pose-extracted, minus any division excluded by the
  pre-check below. **This run burns the Chicago holdout for the
  entire swing thread** — any later instrument needs fresh holdout
  (future VODs, assigned at acquisition per `labeling_protocol.md`).
- **Windows granted**: holdout rally windows are derived exactly as
  on train (label/serve-pin anchored). Rally identity and window are
  GRANTED, as at Gate C and the checkpoint — the test is contact
  placement and pace structure within a known window.
- **Three systems, one run, same panel**:
  - **C** (ceiling) = true label times + GAP threshold fit on train.
  - **B** (baseline) = the current label-free pipeline
    (swing_explore scorer → decode_rally → phase_grader Level-B
    readout with the GAP classifier of record), parameters refit on
    the final train corpus by its existing procedure, unchanged
    otherwise.
  - **T** (candidate) = the temporal model, frozen before the run:
    code committed, fit deterministic (seeded) from train rows +
    pose, the exact verdict script committed. Run happens on the
    user's Mac; the pasted output + fingerprints are the record,
    as at the checkpoint.

### Pre-check (train-only; burns nothing)

C is only a ceiling where it works, and mixed (games 3–4) has never
been through it. Before the verdict run, compute C's structure stats
per division on TRAIN rallies. A division fails the pre-check if
has_fast < 0.8 or first-fast rate < 0.6 or init-team < 0.6 there
(womens measured 1.0 / 0.8 / 0.8 at the checkpoint). Failing
divisions are EXCLUDED from the verdict panel, logged here — a
ceiling that doesn't exist can't be closed toward, and diagnosing it
is separate work. If every division fails, the verdict is BLOCKED
before any holdout is touched.

### Run conditions

- ≥ 90 train rallies labeled (both passes); all panel holdout
  rallies labeled and pose-extracted.
- r9/r10 stay OUT of every training set until the debug-frame
  scorebug check clears their ~24 s span anomaly (they are train
  rallies; the verdict panel is unaffected either way).
- If the surviving panel is < 40 rallies after exclusions, PASS is
  not available at this gate — the best possible outcome is MIDDLE
  (final shot on future-VOD holdout at full power). Small panels
  don't get to license a 2,000-rally program.

## Frozen bars

Everything measured on the same holdout panel in the same run.
Per-stat **closure** = (T_s − B_s) / (C_s − B_s), capped at 1.

Stats: **S1** has_fast agreement with truth · **S2** first-fast
within 1.0 s (over rallies whose truth has a fast contact) ·
**S3** initiating team correct · **S4** per-contact match rate
(±0.5 s, same team).

Informative-stat guard: a stat with C_s − B_s < 0.15 on the panel,
or with C_s < 0.6, is uninformative for closure — dropped from the
counts and logged. If fewer than 2 of {S1,S2,S3} are informative,
there is no decoder verdict: outcome CEILING-BREAK, the 2000-rally
decision defaults to NO until a ceiling exists on fresh footage.

- **PASS** = S4 ≥ 65% absolute, AND ≥ 2 of {S1,S2,S3} closures
  ≥ 2/3, AND no informative closure < 1/3.
- **KILL** = S4 < 55% absolute, OR median informative closure < 1/3.
- **MIDDLE** = everything else.

The S4 floor inside KILL is deliberate: a model that "fixes"
structure while matching no more contacts than today's decoder is
painting the expected pattern out of the pace prior, not finding the
events — structure clears via S1–S3 only on top of real placement
gains. Conversely S4 alone can never PASS: placement without
structure isn't the product.

Power at full panel (76 rallies / ~1,200 contacts): proportion se
~5–6 pp per-rally, ~1.4 pp per-contact; closure denominators ~0.3–0.4
per the checkpoint gaps ⇒ closure se ~0.15, so the 2/3 vs 1/3 bars
sit ~2 se apart.

**Secondary (reported, never decisive)**: fast-share correlation
(checkpoint C = +0.55), matched-contact pace accuracy, ghost rate,
n_fast/n_slow count error, per-division splits, first-fast median
error.

## What the temporal model MAY and MAY NOT use

MAY: timestamped contact + pace labels (train rows only), the
existing feature stack (pose window features, kitchen/position
features, cadence, track candidates), any model class, any training
procedure, calibration on train, the registered joint-decode design
or better.

MAY NOT: holdout rows in any form (including "just checking
coverage"); hand-inspection of holdout video during development;
grading-constant changes; Level-C pace-classifier iteration (the
MODEL-ITERATION FREEZE stands — C's readout is GAP as frozen; T's
pace calls come out of its own decoder, which is the point).

## Amendment protocol

Amendments (dated, appended below) are allowed only BEFORE the
verdict run, and only for: run conditions, pre-check mechanics, or
adding secondary reports. Bars, closure definitions, the panel, and
the burn rules do not move. After the verdict run, nothing moves;
the outcome and its pre-committed consequence are final for this
footage.
