# Why the tracker over-calls bounces

**2026-09-04**, owner-approved follow-on to `bounce_replication.md`
(bounce POSITION 1.9 ft, but only 13 of 34 real bounces matched, and the
tracker emits more bounces than exist).

Reproduce (0.2 s a rally — it reads the c3 cache, no decode):

```bash
python3 vision/turn_audit.py --rally 9        # + 7, 10, 17
python3 vision/render_turns.py --rally 9      # the graded video
```

## The mechanism

One line in `ball_replicate.tracked_side`:

```python
bounce_evs = [e for e in turns if e not in claimed]
```

Every direction change in the tracked path is offered to the hitter-chain
anchors. Whatever an anchor claims is a CONTACT. **Everything left over is
declared a BOUNCE** — the residual category. Nothing ever asks whether the
turn looks like a bounce, so a tracking wobble and a contact the anchors
missed both come out as bounce markers.

## What the leftovers actually are

128 turns over r7 / r9 / r10 / r17. 98 claimed as contacts, **30 emitted
as bounces** — and only 6 of the 30 are bounces:

| the 30 emitted bounces | n |
|---|---|
| really a bounce | 6 |
| tracking junk (path wobble) | 12 |
| **really a CONTACT the anchors missed** | **12** |

So the over-calling is two separate faults of about equal size, and the
bigger surprise is the second: 12 real contacts are being drawn as
bounces because no anchor claimed them. That is anchor *recall*, not
bounce logic, and it also explains part of the missing bounce recall —
several true bounces get claimed as contacts in the same pass.

## The fix for the junk half is physics, not a threshold

Image y grows downward, so a ball that bounces is **falling before and
rising after**: `dy_pre > 0 > dy_post`. It is a sign test — nothing to
tune.

| | falling → rising |
|---|---|
| real bounces | **6 / 6 = 100%** |
| tracking junk | **2 / 12 = 17%** |
| missed contacts | 6 / 12 = 50% |

Applied to the emitted bounces it keeps every real bounce and kills 10 of
the 12 junk markers.

The feature the claiming gate uses **today** — the 2D turn angle — does
not separate them at all: real bounces median 91.5° (range 63–150), junk
median 66.5° (range 0–144). The gate is reading the one channel that
carries no signal here while ignoring the one that carries all of it.

## Watch it

`render_turns.py` draws the graded call on the clip at 0.4×, markers
holding 1.2 s: green contact, blue correct bounce, **red tracking junk**,
**orange missed contact**, with each turn's angle and whether it has the
falling→rising signature. r7 / r9 / r10 / r17 rendered.

## Next

1. Gate the bounce branch on falling→rising (cheap, no new labels).
2. The missed-contact half is the anchor chain, and it is the same
   machinery as the paddle gap — a contact is a turn an anchor claims,
   so anchor recall is the shared bottleneck for both open questions.
