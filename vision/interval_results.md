# The interval histogram — first full-match result (Chicago 2026-07-25)

*2026-08-11. The vision POC's core question, answered on real data: do
inter-contact intervals from broadcast video separate the fast attacking
band from the dink band? **Yes.** Data:
`data/vision/chicago0725_contact_intervals.csv` (1,538 intervals),
`chicago0725_rally_contacts.csv` (per-rally counts joined to the log).*

## Pipeline (all zero-hand-labels)

691,884 ball candidates (60 fps, 720p, chroma+motion detector) →
court-region + area filter → **one gated greedy chain per rally window**
(windows from the cheer↔log alignment; 82% median coverage of rally time)
→ contacts = chain reappearance starts + vertical-direction reversals with
≥22 px displacement on both sides, **excluding reversals inside the
kitchen band (y 260–400), which are bounces** → 0.12 s merge.

## The histogram

2,253 contacts, 1,538 intervals. A distinct fast mode at **0.15–0.25 s**
(peak 0.23 s), separated by a dip at 0.25–0.45, then a broad slow mass
from 0.45–1.0 s. The fast mode is the speed-up/hands-battle band, at
exactly the ~200 ms flight time the geometry predicted before any video
existed.

## Three validations, none using labels

1. **Contacts track the log**: corr(contacts, logged rally duration) =
   **+0.70**; median 12 contacts/rally; 0 empty rallies in 180.
2. **The fast mode is bursty, like real hands battles**: 61% of rallies
   contain ≥1 fast interval, with counts decaying 50/30/16/5/5 for 1–5 and
   four rallies at 6+ (extended exchanges). An artifact sprinkles
   uniformly.
3. **The gender split — no tracking artifact knows which game is which**:
   fast-interval share is **women's 10.4% < mixed 14.1/17.2% ≈ men's
   16.9%**. The known style difference shows up in the measurement,
   blind.

## Known artifacts, quantified (the honest ledger)

- **Recall ~50–65%**: slope of contacts on duration is 0.46/s vs ~0.7–1.5
  expected, so roughly a third of contacts are missed; each miss doubles
  an interval, which inflates the 1.0–1.5 s tail and smears the slow mode.
  Better chain recall (or a learned tracker) sharpens the dink peak.
- **Bounce separation is a crude band**: kitchen-band reversals (475, 17%
  of events) are excluded as bounces, but the band is fixed pixels, not
  homography. Some dink contacts near the NVZ line are discarded, some
  mid-court bounces survive. Homography turns this heuristic into
  geometry.
- **Two self-inflicted lessons recorded**: a 0.25 s merge window *erased
  the fast mode by construction* in an earlier pass (empty bins below the
  merge are a tautology, not a finding — same family as the audio
  refractory bug), and counting every track fragment's endpoints as
  contacts inflated counts 4×. Both are why the per-stage validation
  gates exist.

## What this unlocks

The instrument works. Next in order: attribution (side-of-net is
geometry; player-level rides the log's server/receiver anchors), then the
speed-up roles and the selection ledger already specced in the README —
and the freeze-out final (New Jersey 5s v St. Louis Shock 2026-07-12, all
three games vetted start-marked) becomes a one-command rerun for the
Waters ball-share measurement.

## Post-script: two rejected recall improvements (2026-08-11)

A bidirectional (forward+backward) chain with gap-contact inference was
attempted twice — segment-level dedupe, then point-level fusion — and
**rejected by the blind gender gate both times**: fast share flattened to
a uniform ~26-30% including the women's game, the signature of injected
artifact. Root cause: the backward pass locks onto different objects
(shoes, clutter), and any union of two independent greedy chains
interleaves ball with distractor, manufacturing reversals. The real fix is
a single-best-path tracker (Viterbi over candidates), not chain unioning.
Recorded so the next session does not re-walk it. v3 (forward-only) is the
standing result. A blind 10-rally human count is specced in
`vision/recall_audit.md` to convert the physics-guessed recall into a
measurement.
