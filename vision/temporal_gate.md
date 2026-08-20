# Temporal-model gate — pre-registered bars (VERSION 2, frozen 2026-08-19)

**v2 supersedes v1 (same day).** Still true and load-bearing: **no
temporal-model code exists**, the Chicago holdout is untouched, and
every input to this revision came from TRAIN rows only. Changes are a
deliberate supersession, user-approved 2026-08-19, not amendments —
v1's amendment clause did not cover truth-definition changes, and
pretending it did would be worse than re-freezing. v1 is preserved
verbatim in git history (commit 479c7e6).

## Changelog v1 → v2

1. **S1–S3 are redefined on ATTACK-ONSET, not first-fast.** Trigger:
   the user's taxonomy call, confirmed on train ("the meta is 3:
   drive, 4: counter, 5: dink" — 4/6 fast-shot-3 rallies follow it
   exactly; zero drives are answered slow; full record in
   swing_explore_notes.md 2026-08-19). The drive exchange is
   transition pace, not the rally's break; grading first-fast was
   grading the wrong event (flagship case: r2's "4.4 s miss" was the
   truth definition — the classifier had found the real shot-8
   speed-up all along). The product stat is "when did the rally
   break"; that is attack-onset.
2. **C's readout = the best of the three already-graded classifiers
   (gap / full / seq), selected by per-contact pace accuracy on
   TRAIN at verdict time.** Deterministic, train-only selection;
   currently GAP (73.8% vs SEQ 73.3%, FULL pending pose). Replaces
   v1's GAP-only readout; resolves the C-readout question flagged
   2026-08-19 without a later definition change.
3. **Pre-check refined from per-division wholesale to ≥2-of-3.** A
   division is excluded only if C fails at least two of the three
   pre-check bars there; a single soft stat is instead handled by
   the verdict's informative-stat guard (which drops that stat's
   closure). Rationale: the wholesale exclusion was designed for a
   collapsed ceiling (untested mixed), and the doubled womens panel
   showed the other case — one soft stat (onset ~0.57 at GAP) with
   the rest healthy — which should narrow the verdict, not evict
   the division.

Everything else — bars, closure definition, panel, burn rules, run
conditions, MAY/MAY NOT, consequences, power clause — carries over
from v1 unchanged.

## The frozen ATTACK-ONSET definition

Implemented in `vision/phase_grader.py::attack_onset` (selftested;
applied SYMMETRICALLY to truth and to every prediction stream).
Over a rally's time-ordered PACED contacts (fast/slow; ordinals
count every contact with serve = 1; lunges and whiffs are absent
from the sequence, so fast runs bridge them), a maximal fast run is
an **attack** iff any of:

- it **ends the rally** (its last contact is the rally's last paced
  contact — a trailing lunge does not break enderhood), or
- **length ≥ 2 starting at ordinal ≥ 4**, or
- **length ≥ 3 starting at ordinal 3** (an escalated drive exchange
  — the drive got countered and it stayed hot; data support n=1,
  r3 — thin, frozen anyway as the best current guess).

Attack onset = (t, team) of the first attack run's first contact.
Consequences of the definition, by design: a lone drive–counter
(len-2 at ordinal 3) is NOT an attack; a lone mid-rally fast that
gets reset is NOT an attack; a dink marathon has NO attack; a
third-shot putaway IS one. Truth-side uncertainty: an unjudged
contact before the found onset (or anywhere, when none was found)
makes the rally boundary-uncertain; lunges never do.

## Context — what the checkpoint and the 25-rally panel measured

The 2026-08-19 checkpoint (10 pose rallies) routed REGIME 2:
placement binds, pace classification doesn't (Level B match rate
45.7% vs Level C GAP 74.6%). The labels-only extension to 19 train
rallies (fingerprint 1165ddb642) held per-contact accuracy (73.8%)
and exposed the classifier's two blind spots — rally-enders (no
produced gap) and long-produced-gap fasts (drives; half the fast
population) — plus the semantics problem v2 fixes. Under
attack-onset at GAP: has_attack 15/19, onset ≤1 s 8/14, init-team
10/14 (SEQ: 16/19, 7/14, 10/14). The temporal model's design brief,
registered: joint decode of contact times and pace states
(pace-conditional gap durations), plus the non-tempo channels the
blind spots demand — position (kitchen features) and rally-end
context from the logs.

## The decision this gate makes (once)

- **PASS** → label-free structure stats are near label quality on
  held-out rallies: the multi-VOD labeling push ("2000 rallies") and
  the decoded-events structure product are licensed.
- **MIDDLE** → one documented amendment cycle; the final shot runs on
  the FIRST FUTURE VOD's holdout (Chicago holdout is burned).
- **KILL** → temporal decoding is dead on this footage class; the
  product is labeled-matches-only (Level C), the label archive keeps
  full value as its fuel, and the thread reopens only with
  materially different footage (higher fps/res or uncondensed).

## Frozen evaluation

- **Instrument**: `vision/phase_grader.py` grading harness (attack
  rows) as of the v2 commit. MATCH_TOL_S = 0.5 s (same-team,
  one-to-one greedy), onset tolerance 1.0 s (FF_TOL_S), truth
  builder incl. the lunge rule. Grading constants may not move.
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
  placement and attack structure within a known window.
- **Three systems, one run, same panel**:
  - **C** (ceiling) = true label times + the best-on-train
    classifier (changelog item 2).
  - **B** (baseline) = the current label-free pipeline
    (swing_explore scorer → decode_rally → phase_grader Level-B
    attack readout with the same classifier), parameters refit on
    the final train corpus by its existing procedure, unchanged
    otherwise.
  - **T** (candidate) = the temporal model, frozen before the run:
    code committed, fit deterministic (seeded) from train rows +
    pose, the exact verdict script committed. Run happens on the
    user's Mac; the pasted output + fingerprints are the record.

### Pre-check (train-only; burns nothing)

Per division on TRAIN rallies, C's attack stats against three bars:
has_attack agreement ≥ 0.8, onset-rate (≤1 s) ≥ 0.6, attack-team
≥ 0.6. A division failing **at least two** bars is EXCLUDED from
the verdict panel (logged here) — that is a collapsed ceiling. A
division failing one bar stays in; the soft stat is expected to be
dropped by the informative-stat guard at verdict time. Current
womens reading at GAP: 0.79 / 0.57 / 0.71 — one bar soft, stays in.
If every division is excluded, the verdict is BLOCKED before any
holdout is touched.

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

Stats: **S1** has_attack agreement with truth · **S2** attack onset
within 1.0 s (over rallies whose truth has an attack) · **S3**
attacking team correct · **S4** per-contact match rate (±0.5 s,
same team).

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
painting the expected pattern out of the pace prior, not finding
the events — structure clears via S1–S3 only on top of real
placement gains. Conversely S4 alone can never PASS: placement
without structure isn't the product.

Power at full panel (76 rallies / ~1,200 contacts): proportion se
~5–6 pp per-rally, ~1.4 pp per-contact; closure denominators ~0.3–0.4
⇒ closure se ~0.15, so the 2/3 vs 1/3 bars sit ~2 se apart.

**Secondary (reported, never decisive)**: first-fast rows (v1
semantics, for continuity with the checkpoint record), fast-share
correlation, matched-contact pace accuracy, ghost rate,
n_fast/n_slow count error, per-division splits, onset median error.

## What the temporal model MAY and MAY NOT use

MAY: timestamped contact + pace labels (train rows only), the
existing feature stack (pose window features, kitchen/position
features, cadence, track candidates), rally-end times from the
referee logs (deployment-available; the ender blind spot's
channel), any model class, any training procedure, calibration on
train, the registered joint-decode design or better.

MAY NOT: holdout rows in any form (including "just checking
coverage"); hand-inspection of holdout video during development;
grading-constant changes; pace-classifier iteration at Level C
beyond the frozen best-of-three selection (the MODEL-ITERATION
FREEZE stands; T's pace calls come out of its own decoder, which is
the point).

## Amendment protocol

Amendments (dated, appended below) are allowed only BEFORE the
verdict run, and only for: run conditions, pre-check mechanics, or
adding secondary reports. Bars, closure definitions, the
attack-onset definition, the panel, and the burn rules do not move.
After the verdict run, nothing moves; the outcome and its
pre-committed consequence are final for this footage. (A further
truth-definition change would require a v3 supersession — legal
only while no temporal code exists and no holdout is touched, and
only with user approval; once temporal code exists, definitions are
locked for good.)
