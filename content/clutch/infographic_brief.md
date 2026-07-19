# Infographic brief — "Clutch" / crunch-time

**Concept:** A broadcaster said the best players win crunch time. We measured it across 163k rallies. He's right — and here's the leaderboard.

Two hero elements: (1) the **clutch leaderboard**, (2) the **47% biggest-point** stat. Everything else is supporting.

---

## HERO 1 — the clutch leaderboard

*Top players ranked by how often they win the biggest points, above their own baseline.*

| rank | player | clutch score (z)¹ | normalized² |
|---|---|---|---|
| 1 | Anna Leigh Waters | 7.9 | 100 |
| 2 | Ben Johns | 6.8 | 86 |
| 3 | Anna Bright | 6.7 | 85 |
| 4 | Gabriel Tardio | 5.5 | 70 |
| 5 | Christian Alshon | 4.2 | 53 |
| 6 | Jorja Johnson | 4.1 | 52 |

¹ *z = standard deviations above random chance. Higher = more clutch, and more certain it's real.*
² *Normalized to Waters = 100 — use these for bar lengths if cleaner.*

**Point for the designer:** the list is, in order, basically the best players in the sport. That's the whole story — don't over-decorate it.

---

## HERO 2 — the biggest point in pickleball

> **Down 9-10, receiving, on the opponent's 2nd server = a 47% swing in win probability.**

- Exact figure: **0.467** (46.7%). Round to **47%** for display.
- Plain-language: *nearly half the game rides on that one rally.*
- Mirror images (equally big, if useful): **10-9** and **11-10** on your *own* 2nd server.

---

## Copy blocks (ready to set)

- **Eyebrow:** PICKLES · pro pickleball, with receipts
- **Headline:** The best players don't give it away in crunch time.
- **Attribution (small, under headline):** — Dave Fleming, on the broadcast. So we checked.
- **Subhead / method line:** We reconstructed 162,942 rallies across 2,402 pro matches and measured exactly how much each point swung the game — then ranked who wins the big ones above their own baseline.
- **The honest twist (caption):** There's no separate "clutch gene." A player's clutch score tracks their overall skill (correlation 0.58) — the best players don't flip a switch at 9-10, they're already there while everyone else drops.
- **Footer / source:** v2 model · every MLP + PPA pro game since 2024 · 77% accuracy (DUPR 65%)

---

## Optional secondary panel — "and it's not just single points"

*The two players who raise their level MOST against the strongest opponents:*

- **Ben Johns** (+3.2) · **Anna Leigh Waters** (+2.2)

Same two names, twice. Reinforces the headline.

---

## Guardrails (please honor)

- **Do NOT show a "least clutch" / choker list.** Those are lower-profile players on small samples — naming them is punching down and off-brand.
- The z-scores are real; don't invent a "clutch %" that implies a precision we don't have. "Clutch score" or a relative bar is fine; a fake percentage is not.
- Keep the honest-twist caption in. It's what separates this from a lazy stat graphic — the credibility is the point.

## Brand palette (PICKLES — matches the ice-out card)

- ground / dark green `#16321e` · panel `#0f2617`
- accent lime `#d9f154`
- cream text `#fbfdf3` / `#edf4e0` · muted sage `#a9bc8c`
- red (use sparingly, e.g. the biggest-point callout) `#e66767`

## Source of numbers

`model/big_points.py` → `model/big_points_summary.json` (clutch: `top_clutch`; stat: `biggest_points`; durability: `durability.summary.top_biggame`).
