# Labeling protocol — timestamped contacts at scale (2026-08-17)

User decision: labeling is cheap for them ("painless and easy when I'm
bored/need a break"), so the label archive grows as a background habit
rather than a one-week push. Labels are the project's binding constraint
for the swing thread (see `swing_explore_notes.md`) and are
instrument-agnostic: the same CSV feeds evaluation CIs, selector
training, ViTPose A/Bs, and any future temporal model. Nothing labeled
is ever wasted by a model change.

## The split is pre-registered — frozen 2026-08-17

`data/vision/label_split.csv` (rally_cum, game, division, split),
generated BEFORE any label beyond rallies 1–10 existed. Rule: within
each Chicago game, the first 60% of rallies in video order = **train**,
the rest = **holdout**. Core rallies 1–16 are train/dev forever
(contact_gate.md). Boundaries:

    game 1 (womens): train 1–21,    holdout 22–34
    game 2 (mens):   train 35–54,   holdout 55–67
    game 3 (mixed):  train 68–112,  holdout 113–142
    game 4 (mixed):  train 143–173, holdout 174–193
    → 117 train / 76 holdout, every division on both sides

Hard rules:

- **Label everything identically.** The split is deliberately NOT shown
  in the tool — no observer effect on how holdout rallies get clicked.
- **Exploration and tuning scripts read train rows ONLY.** Holdout rows
  are quarantined — no script that gets iterated on may load them, even
  "just to look". This is the discipline that makes a future verdict
  mean something (see the circularity traps in CLAUDE.md findings 1, 10).
- A holdout block is **burned once used**: after a pre-registered
  evaluation scores on it, subsequent verdicts need fresh holdout
  (future VODs). The Chicago holdout's registered consumer is the
  temporal-model verdict (`vision/temporal_gate.md`, frozen
  2026-08-19) — one run, all four blocks.
- **Future VODs**: assign each whole match to train or holdout at
  acquisition time, before labeling, by appending to this protocol.

## Workflow (tool build 2026-08-18a — two passes per rally)

Open `contact_audit_chicago0725.html`, load the video
(`full_match.mp4.webm` — the tool verifies the file), pick a rally:

1. **Verify the scorebug** matches the rally's start score — this is
   rally identity, the one check that catches every alignment trap.
2. Stamp the serve at contact (⏎), then each contact in order — number
   keys pick the hitter, prefill flow handles core rallies.
3. **W = whiff** (a swing that missed). Convention (user, 2026-08-16):
   when both players went for a ball and only one hit it, the whiff is
   recorded for the non-hitter — a whiff on the same team as the hitter
   means exactly that.
4. ⛔ if the rally isn't in the video; ✕ drops a wrong prefill.
5. **Pace pass (P)** — new in build 2026-08-18a: after stamping, press
   **P**; the rally rewinds and replays (1× is fine) while **F**/**S**
   tag each highlighted contact fast or slow in order. Serves, returns,
   whiffs, and shots already carrying a granular type are skipped, so
   on old rallies the pass is exactly the "other" backlog. Coarse is
   the contract (user call 2026-08-18): fast = attacked ball, slow =
   soft ball — never agonize over smash vs counter vs speed-up (they
   were coded interchangeably anyway, and the analysis pools them).
   Tags are written into the same shot_type column as literal
   `fast`/`slow`; the CSV format and every consumer are unchanged.
   ⌫ in pace mode un-tags; the orange **F/S** badge in the rally list
   marks rallies still owing the pass.
   **Ambiguous defensive contacts — the tie-break rule** (user
   practice, made explicit 2026-08-19): on a stretched, low or
   off-balance reach the action looks identical either way, so code
   the OUTCOME — if it successfully took the pace off, **slow**; if it
   popped up and invited the attack, **fast**. Keep doing this. It is
   already what the analysis means by pace (a contact's pace is the
   state of the gap it PRODUCES), so the rule agrees with every
   downstream consumer. Two things follow, both recorded in
   swing_explore_notes 2026-08-19: the label for this subclass is
   genuinely NOT visible at the contact instant — a finding about the
   sport, not a flaw in the labeling — and it is the one place where
   the label and a tempo feature share a cause, so pace accuracy on
   these rows is never independent validation of a tempo classifier.
6. **Lunge rule** (user policy 2026-08-18, given its own word rather
   than overloading "other"): a contact WITHOUT a real swing —
   desperate lunge/stretch/stab, usually a forced error — is typed
   **lunge** (**X** during the pace pass, or the dropdown/X button in
   pass 1). Still stamp it: it is a real contact and shot counts /
   alternation depend on it. It is excluded from fast/slow so weird
   lunges can't contaminate the pace classes, and it can be filtered
   from any future swing-kinematics training. Unlike "other" it is a
   judgment, not a backlog — no badge nag, and in phase_grader it
   never makes a rally boundary-uncertain (a lunge can't be the
   rally's first attack). "other" keeps meaning "haven't judged yet".

**Chained seek** (new in build 2026-08-17a): rallies without a hand pin
now auto-seek off YOUR previous labels — the previous rally's last
contact plus a gap learned from your own labeling (~±8 s, vs ~±20 s for
the old machine windows; rallies 17–19 had no window at all). A toast
says which rally it predicted from and what the scorebug should read —
confirm it every time. Labeling in video order keeps the chain tight;
after a skipped stretch the first rally may need a short hunt, then the
chain re-anchors.

## Order of work

1. **Rallies 11–16** — completes the pre-registered core-16
   (11–12 are the fast-heavy ones the dev set is short on).
   *2026-08-19: rallies 11–12 turned out not to be in the VOD (the ⛔
   case) — the core set tops out at 14 of 16 in practice.*
2. **Forward from 17 in video order.** Sequential order is what makes
   the chained seek accurate; it also fills train blocks before holdout
   blocks in games 1–2, which is the useful order anyway.

Full Chicago ≈ 3,000 contacts (≈16/rally × 188 windowed rallies) —
roughly 20× the current archive, enough to train a small temporal model
and evaluate with ±2–3 pp CIs.

## Export hygiene

- **Export the CSV at the end of every sitting** — the export is both
  the deliverable and the backup (⬆ import restores it exactly;
  localStorage is the working copy but is not durable).
- Keep it as `contact_labels_chicago0725.csv` on the Mac (Colab and
  swing_explore consume that name), and share it in-thread so it lands
  in `data/vision/` — timestamps and names only, no imagery. The
  broadcast video itself is never committed (house rule).

## Addendum 2026-08-29 — the state-audit pilot (hitter episodes, gold subset)

User go: "I'm not mad at doing a massive coding set myself… especially
if we just start with the 5-10 training rallies." Scoped per the
economics in swing_explore_notes.md 2026-08-29: step every frame, tap
only BOUNDARIES — the interval inherits, so per-frame states derive
complete at ~10x fewer judgments than literal per-frame coding.

- Tool: `python3 vision/make_state_audit.py` →
  `data/vision/state_audit_chicago0725.html`. Pilot rallies
  PRE-SPECIFIED: 1-8, 13, 14 (train only, r9/r10 quarantined, 124
  contacts, fast-heavy r1/r3 included).
- Per player episode: **B** start ("looks like about to hit"), **I**
  frame-exact impact (the old tap shows as a tick — navigate by it,
  never snap to it), **E** end (follow-through done), **X** =
  no-contact episode (fake / both-went-for-it / aborted — the hard
  negatives). Rally-level: **R** service routine starts, **D** point
  dead (user addition, same day).
- Export every sitting as `state_labels_chicago0725.csv` into
  data/vision/ — same hygiene as the contact CSV.
- What the pilot buys before any scale-up decision: episode onset/
  offset distributions vs impact; the user's own tap-jitter
  distribution (frame-exact I vs old tap); no-contact episode rate;
  R/D vs referee-log lead/end times; and measured minutes-per-rally,
  so "massive set: worth it?" is answered with a number, not a mood.
- These are NEW LABEL TYPES: before any temporal-model code exists,
  temporal_gate.md needs a dated amendment recording them as permitted
  T inputs (draft in swing_explore_notes.md; user freezes it
  explicitly). Holdout rules unchanged: pilot is train-only; if state
  labeling ever extends to holdout rallies, label identically and keep
  tuning scripts out, exactly as with contacts.

### Vocabulary additions, user-approved 2026-08-29 (state audit v2)

- **F/H at impact = forehand/backhand** — fills the fh/bh label gap
  the record flagged (dink-rotation mechanics was blocked on it).
  Press again to clear; wrong one just press the other.
- **P = poach flag** on a hitting episode — took a ball on the
  partner's side. Direct observation of the channel the retracted
  intent measures needed the ball for; the aggressive twin of C.
- Shot DIRECTION (cross/line/middle) considered and DEFERRED to its
  own dedicated pass — it means watching the ball after impact, not
  the player, and stacking it here would split attention.
- Vocabulary freezes after ~2 pilot rallies of practice (same
  convention path as the lunge and pace-tie-break rules).
