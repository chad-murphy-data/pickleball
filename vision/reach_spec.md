# Reachability voter — spec (UNBUILT)

Purpose: given a contact at court position X, score which of two
partners moved into the region where the ball is going. Feeds the
`intent` family of voters in `touch_attribute.py` as a prior on the
one thing still in doubt after alternation — which of the two players
on a side struck the ball.

## THE TARGET IS THE NEXT CONTACT, NOT THE LANDING POINT

User's constraint, 2026-08-21, and it is the whole design:
**drives don't land.** A drive, a speed-up, a passing shot and most
attackable balls are struck OUT OF THE AIR by the receiving team; the
point where they would have bounced is a counterfactual that never
happens and that no instrument in this project can observe. A model
conditioned on "where does the ball land" would therefore:

  * be fit on the subset of shots that DO land — dinks, drops, resets,
    lobs and serves/returns, which are the slow, soft, near-kitchen
    shots. That is a biased sample in exactly the direction that
    matters, because
  * the shots whose attribution is hardest are the fast ones, where
    both partners are close together at the kitchen and a poach is
    live. Those are precisely the shots that get volleyed.

So the conditional to estimate is

    P(position of contact n+1 | position of contact n, pace)

with position measured AT CONTACT on both ends. Bounce is not in the
model, is not needed, and must not be introduced as an intermediate
variable — "would have landed" is unobserved and unfalsifiable, the
same failure that poisoned the auto-label fine-tune (ground truth only
where a human can see it).

Corollary for labelling: a hand-coding pass marks WHERE EACH CONTACT
HAPPENED, one court position per contact. It never marks a bounce.
That also keeps the labelling unit small and fully observable — the
labeller is pointing at a player striking a ball, not judging a
hypothetical.

Corollary for the ball tracker: `ball_voter.flight_segments` already
returns inter-contact flights and `ball_at_contact` reads their
endpoints. Endpoints are contacts by construction. Nothing there needs
changing; this note exists so the reach model is fit on the same
quantity rather than on a bounce inferred from a trajectory.

## Pace is the conditioning variable, not shot type

Shot type (dink/drive/speed-up) is a label we do not have at scale —
the 203 hand-typed shots in
`data/vision/shot_labels_chicago0725.csv` are the only ones, and their
TYPES carry residual pop-era risk (see CLAUDE.md, vision entry).
Flight speed between contacts is measurable from the tracker's own
segments and is the physical variable that actually drives how far the
next contact is from the last one. Condition on it directly.

## Status

Not built. Gated on the `ball` and `intent` voters showing real
accuracy on the DISPUTED column of `touch_attribute.py`'s VOTER
ACCURACY vs TRUTH table — a reach model inherits their signal, so if
they sit at chance there it is premature.
