"""Per-player court occupancy heat maps — WHERE each player stands.

User request 2026-08-19: "a 'heat map' of each player's position.  We'd
have to control for side of the court or something...but would be
interesting."  Controlling for side is the whole design problem, so two
frames are rendered and BOTH are reported; neither is the "real" one.

PRE-REGISTERED 2026-08-20 BEFORE any density was computed.  Definitions
below are the ones the numbers were produced under, unchanged.

FRAME A — COURT FRAME ("own perspective", ends folded)
  Depth d = |y - NET_Y| in [0, 22+]: 0 at the net, 7 at the kitchen
  line, 22 at the baseline.  Width x' = x for a near-end rally and
  W_FT - x for a far end.  The mirror is REQUIRED, not cosmetic: the
  two ends face opposite directions, so a fixed world x is the player's
  left at one end and their right at the other.  Folding ends without
  mirroring would average a player's forehand side against their
  backhand side.  Frame A is the raw picture: it still mixes the two
  service halves a player rotates through, so a player who plays both
  halves reads bimodal.  That is a true fact about the match, not an
  artifact to be tuned away.

FRAME B — RALLY-RELATIVE FRAME (the "controlled for side" one)
  Per rally, x_ref = median x' over the player's retained rally-phase
  frames in the first START_S = 1.0 s.  The rally is mirrored
  (x'' = W_FT - x') iff x_ref > W_FT/2.  So x'' < 10 is always the half
  the player LINED UP in and x'' > 10 is always the half their partner
  lined up in, whichever physical side that was.  Rallies with fewer
  than START_MIN_FRAMES = 3 frames in that first second are DROPPED
  from frame B and ledgered (drop-don't-guess: an unknown starting half
  is not a coin flip).  Frame B answers "relative to where I set up,
  where do I go" and is the frame in which crossing x = 10 means
  entering the half the partner started in.

  Frame B measures SPACE, not intent, and inherits the retraction in
  coverage_spec.md: a player standing in the partner's half may have
  poached, may have switched, may have been pulled there.  Nothing in a
  position density can separate those without shot data.

  *** STACKING CONFOUND (user, 2026-08-20) — read before quoting any
  frame-B crossing number. ***  "Where they lined up" is NOT a stable
  reference when a team STACKS: stacking deliberately puts both
  partners on one side before the serve, and they cross to their
  preferred sides immediately after it.  Under a start-anchor that
  scripted unwind reads as "crossed into the partner's half", and it
  inflates the crossing statistic MOST for whichever pair stacks most.
  Partial mitigation only: rally-phase frames already begin
  SERVE_PHASE_S = 2.0 s after serve contact, so a fast unwind may
  already be excluded -- but that is not guaranteed, and for anchor-free
  rallies the serve instant is unknown so the 2 s runs from the first
  retained frame instead.

FRAME B-SETTLED — the stacking-robust anchor, PRE-REGISTERED 2026-08-20
BEFORE computing any of it:
  x_settled = median x' over ALL of that player's retained rally-phase
  frames in the rally; mirror iff x_settled > W_FT/2.  A stack unwind
  occupies the opening of a rally, so the median over the whole rally
  reports the side actually played rather than the side lined up on.
  This is an ORIENTATION choice only -- it decides which way to flip a
  rally, never which frames count -- so the crossing statistic it feeds
  (fraction of frames past x = 10) stays a statement about the court
  centreline, not about the player's own average.
  Pre-registered diagnostics, all reported whatever they show:
    D1 DISAGREEMENT RATE: share of (rally, player) whose start-anchor
       and settled-anchor disagree.  This IS the unwind rate; if the
       start-anchor were safe it would be ~0.
    D2 the same rate PER TEAM -- a stacking pair should separate.
    D3 crossing fraction under BOTH anchors, per player, so the size of
       the contamination is visible rather than argued.
    D4 SAME-SIDE-AT-START: share of rallies whose first retained second
       has both partners on one side of the centreline (the geometric
       signature of a stack that has not finished unwinding).
  AMBIGUOUS-ANCHOR LEDGER: |x_settled - 10| < SETTLE_DEAD_FT = 1.0 means
  the player genuinely lives on the centreline and the mirror is a coin
  flip; those are counted and reported, never silently oriented.

CONTRAST PANELS (frame B, difference of normalised densities)
  Same-gender, opposite-team: Alshon - Patriquin and Black - Bright.
  Pre-registered as the only contrasts drawn, because cross-gender
  comparison is likelihood-flat house-wide and men and women play
  different roles in mixed, so a men-vs-women difference map would
  render a role difference as if it were a player difference.

GRID / SMOOTHING (fixed before computing)
  0.5 ft cells over x in [-3, 23] and d in [0, 26] (the margins hold
  off-court and behind-baseline frames).  Separable Gaussian smoothing,
  SIGMA_FT = 1.0.  Each panel is normalised by its own 99.5th
  percentile of non-zero density, not its max, so one hot cell cannot
  crush the map.  Frames are unweighted: at a uniform sample rate each
  retained frame is an equal slice of time, which is what "occupancy"
  should mean.

Frames come from coverage.run(collect=...), so this renders EXACTLY the
frame set the shipped metrics used -- same identity chain, same gates,
same drops.  Writes no committed CSV: the deliverable is the picture,
and adding a numeric table here would invite reading new claims out of
a descriptive visualisation.

  python3 vision/coverage_heatmap.py --pose-dir ... --court ... \
      --windows ... --lineup ... --cam ... [--anchor-free ...] \
      --out heatmap.svg
  python3 vision/coverage_heatmap.py --selftest
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage as C                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

GRID_FT = 0.5
X_LO, X_HI = -3.0, 23.0
D_LO, D_HI = 0.0, 26.0
SIGMA_FT = 1.0
START_S = 1.0
START_MIN_FRAMES = 3
SETTLE_DEAD_FT = 1.0
NORM_PCT = 99.5

NX = int(round((X_HI - X_LO) / GRID_FT))
ND = int(round((D_HI - D_LO) / GRID_FT))

# sequential blue, light->dark (dataviz reference ramp, steps 100..700)
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
GREY_MID = "#f0efec"
# diverging red arm: equal step count to the blue arm used for the
# positive side, gray midpoint (dataviz: blue<->red, never a rainbow)
RED = ["#f6d3d3", "#f2bcbc", "#eda2a2", "#e88787", "#e46b6b", "#e34948",
        "#d03b3b"]
BLUE_ARM = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#5598e7", "#3987e5",
            "#2a78d6", "#1c5cab"]


def own_frame(xy, end):
    """(x, y) court feet -> (x' own-perspective width, d depth from net)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    d = np.abs(xy[:, 1] - C.NET_Y)
    x = xy[:, 0] if end == "near" else C.W_FT - xy[:, 0]
    return np.column_stack([x, d])


def start_half(ts, xp):
    """Median x' over the first START_S; None if too few frames."""
    if len(ts) == 0:
        return None
    m = np.asarray(ts) <= np.asarray(ts)[0] + START_S
    if int(m.sum()) < START_MIN_FRAMES:
        return None
    return float(np.median(np.asarray(xp)[m]))


def hist2d(pts):
    """(N,2) of (x', d) -> occupancy counts on the fixed grid."""
    H = np.zeros((ND, NX), float)
    if len(pts) == 0:
        return H
    p = np.asarray(pts, float)
    ix = np.floor((p[:, 0] - X_LO) / GRID_FT).astype(int)
    iy = np.floor((p[:, 1] - D_LO) / GRID_FT).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < ND)
    np.add.at(H, (iy[ok], ix[ok]), 1.0)
    return H


def _kernel(sigma_cells):
    r = int(np.ceil(3 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    return k / k.sum()


def smooth(H, sigma_ft=SIGMA_FT):
    """Separable Gaussian, ZERO-padded.

    Zero padding is the honest boundary: there were no observations
    outside the grid, so the density should fade there.  An earlier
    edge-padded version replicated the rim values outward and drew
    visible uniform bands down the panel edges -- structure that came
    from the padding, not the match.
    """
    k = _kernel(sigma_ft / GRID_FT)
    r = (len(k) - 1) // 2
    A = np.pad(H, ((0, 0), (r, r)), mode="constant")
    A = np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 1, A)
    A = np.pad(A, ((r, r), (0, 0)), mode="constant")
    A = np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 0, A)
    return A


def norm_level(H):
    """The p99.5 of a panel's own non-zero density."""
    nz = H[H > 0]
    return float(np.percentile(nz, NORM_PCT)) if nz.size else 0.0


def normalise(H, hi=None):
    """Scale to a reference density.

    hi=None keeps each panel on its OWN p99.5, which is only valid for
    reading one panel at a time.  Small multiples get compared across
    panels -- that is what they are for -- so render() passes a SHARED
    hi per row.  Measured on the mixed final, the per-panel normalisers
    ranged 14.3 (Alshon) to 28.7 (Bright), a 2.0x spread: under
    per-panel scaling the same amount of blue meant twice as much time
    in one panel as the next, which silently exaggerates whoever has
    the flattest distribution.
    """
    if hi is None:
        hi = norm_level(H)
    return H / hi if hi > 0 else H


def accumulate(rally_tracks_by_game):
    """-> (per_player_frameA, per_player_frameB, ledger, per_rally).

    A/B are keyed by player uuid and hold (N,2) point stacks.  per_rally
    keeps the (rally, player) structure that BOTH frames are derived
    from, so an alternative anchor (see FRAME B-SETTLED) can be tested
    without re-running the ~12-minute identity chain.
    """
    A = defaultdict(list)
    B = defaultdict(list)
    led = defaultdict(int)
    per = []                       # (game, cum, uuid, ts, x_own, depth)
    for game, rallies in sorted(rally_tracks_by_game.items()):
        for cum, rd, lin in rallies:
            for u, (ts, xy, end) in rd.items():
                if len(ts) == 0:
                    led["empty"] += 1
                    continue
                p = own_frame(xy, end)
                A[u].append(p)
                led["rallies_A"] += 1
                per.append((int(game), int(cum), u,
                            np.asarray(ts, float), p[:, 0].copy(),
                            p[:, 1].copy()))
                ref = start_half(ts, p[:, 0])
                if ref is None:
                    led["dropped_no_start"] += 1
                    continue
                q = p.copy()
                if ref > C.W_FT / 2:
                    q[:, 0] = C.W_FT - q[:, 0]
                    led["mirrored"] += 1
                B[u].append(q)
                led["rallies_B"] += 1
    A = {u: np.vstack(v) for u, v in A.items()}
    B = {u: np.vstack(v) for u, v in B.items()}
    return A, B, led, per


def settled_half(x):
    """Median x' over the WHOLE rally -- the side actually played."""
    return float(np.median(np.asarray(x))) if len(x) else None


def frame_settled(per):
    """per_rally -> (per_player points, ledger) under the settled anchor."""
    out = defaultdict(list)
    led = defaultdict(int)
    for game, cum, u, ts, x, d in per:
        ref = settled_half(x)
        if ref is None:
            led["empty"] += 1
            continue
        if abs(ref - C.W_FT / 2) < SETTLE_DEAD_FT:
            led["ambiguous_centre"] += 1      # counted, still oriented
        q = np.column_stack([x, d])
        if ref > C.W_FT / 2:
            q[:, 0] = C.W_FT - q[:, 0]
            led["mirrored"] += 1
        out[u].append(q)
        led["rallies"] += 1
    return {u: np.vstack(v) for u, v in out.items()}, led


# ---------------------------------------------------------------- svg

PX_FT = 7.0
PAD_L, PAD_T = 60.0, 28.0
GAP_X, GAP_Y = 34.0, 62.0
# text baselines, measured up from a panel's top edge
ROW_HEAD_DY, ROW_SUB_DY = 44.0, 31.0
PANEL_TITLE_DY, PANEL_SUB_DY = 17.0, 5.0
SURFACE = "#fcfcfb"          # committed to a single light look, painted
INK = "#0b0b0b"              # explicitly (no unvalidated dark ramp)
INK2 = "#52514e"
LINE = "#8d8b85"


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _levels(H, ramp):
    """Density -> integer ramp index per cell (0 = draw nothing)."""
    v = np.clip(H, 0.0, 1.0)
    return np.rint(v * (len(ramp) - 1)).astype(int)


def _div_levels(H, n):
    v = np.clip(H, -1.0, 1.0)
    return np.rint(v * n).astype(int)


def _cells(idx, ox, oy, colour_of):
    """Run-length merged rects; keeps the file small enough to commit."""
    out = []
    for r in range(idx.shape[0]):
        c = 0
        row = idx[r]
        while c < len(row):
            lv = row[c]
            c2 = c
            while c2 + 1 < len(row) and row[c2 + 1] == lv:
                c2 += 1
            col = colour_of(lv)
            if col is not None:
                x = ox + (c * GRID_FT) * PX_FT
                # depth runs DOWNWARD from the net at the panel top, so
                # row r (depth D_LO + r*GRID_FT) maps straight down
                y = oy + (r * GRID_FT) * PX_FT
                w = ((c2 - c + 1) * GRID_FT) * PX_FT
                h = GRID_FT * PX_FT
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" '
                           f'width="{w:.1f}" height="{h:.1f}" '
                           f'fill="{col}"/>')
            c = c2 + 1
    return out


def _court(ox, oy):
    """Net / kitchen / baseline / sidelines / half divider, recessive."""
    def px(x, d):
        # net (d = 0) at the panel TOP; must match _cells' row mapping
        return (ox + (x - X_LO) * PX_FT,
                oy + (d - D_LO) * PX_FT)
    g = []
    x0, _ = px(0.0, 0.0)
    x1, _ = px(C.W_FT, 0.0)
    for d, wdt, dash in ((0.0, 1.8, ""), (7.0, 1.2, "4 3"), (22.0, 1.2, "")):
        _, y = px(0.0, d)
        g.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}"'
                 f' y2="{y:.1f}" stroke="{LINE}" stroke-width="{wdt}"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
    _, ytop = px(0.0, 22.0)
    _, ybot = px(0.0, 0.0)
    for x in (0.0, C.W_FT):
        xx, _ = px(x, 0.0)
        g.append(f'<line x1="{xx:.1f}" y1="{ytop:.1f}" x2="{xx:.1f}"'
                 f' y2="{ybot:.1f}" stroke="{LINE}" stroke-width="1.2"/>')
    xc, _ = px(C.W_FT / 2, 0.0)
    g.append(f'<line x1="{xc:.1f}" y1="{ytop:.1f}" x2="{xc:.1f}"'
             f' y2="{ybot:.1f}" stroke="{LINE}" stroke-width="1"'
             f' stroke-dasharray="2 4"/>')
    return g


def panel(H, ox, oy, title, sub, diverging=False):
    g = [f'<rect x="{ox:.1f}" y="{oy:.1f}" '
         f'width="{NX * GRID_FT * PX_FT:.1f}" '
         f'height="{ND * GRID_FT * PX_FT:.1f}" fill="{SURFACE}"/>']
    if diverging:
        n = len(BLUE_ARM)
        idx = _div_levels(H, n)

        def colour_of(lv):
            if lv == 0:
                return None
            return BLUE_ARM[lv - 1] if lv > 0 else RED[-lv - 1]
    else:
        idx = _levels(H, BLUE)

        def colour_of(lv):
            return None if lv == 0 else BLUE[lv]
    g += _cells(idx, ox, oy, colour_of)
    g += _court(ox, oy)
    g.append(f'<text x="{ox:.1f}" y="{oy - PANEL_TITLE_DY:.1f}"'
             f' fill="{INK}" font-size="12.5"'
             f' font-weight="700">{_esc(title)}</text>')
    g.append(f'<text x="{ox:.1f}" y="{oy - PANEL_SUB_DY:.1f}"'
             f' fill="{INK2}" font-size="9.5">{_esc(sub)}</text>')
    return g


def legend(ox, oy, w=150.0):
    g = [f'<text x="{ox:.1f}" y="{oy - 6:.1f}" fill="{INK2}"'
         f' font-size="10.5">time spent — ONE scale across the row</text>']
    step = w / len(BLUE)
    for i, c in enumerate(BLUE):
        g.append(f'<rect x="{ox + i * step:.1f}" y="{oy:.1f}"'
                 f' width="{step:.1f}" height="9" fill="{c}"/>')
    g.append(f'<text x="{ox:.1f}" y="{oy + 21:.1f}" fill="{INK2}"'
             f' font-size="9.5">less</text>')
    g.append(f'<text x="{ox + w:.1f}" y="{oy + 21:.1f}" fill="{INK2}"'
             f' font-size="9.5" text-anchor="end">more</text>')
    return g


def div_legend(ox, oy, left, right, w=150.0):
    g = []
    ramp = [RED[len(RED) - 1 - i] for i in range(len(RED))] + [GREY_MID] \
        + BLUE_ARM
    step = w / len(ramp)
    for i, c in enumerate(ramp):
        g.append(f'<rect x="{ox + i * step:.1f}" y="{oy:.1f}"'
                 f' width="{step + 0.4:.1f}" height="9" fill="{c}"/>')
    g.append(f'<text x="{ox:.1f}" y="{oy + 21:.1f}" fill="{INK2}"'
             f' font-size="9.5">{_esc(left)}</text>')
    g.append(f'<text x="{ox + w:.1f}" y="{oy + 21:.1f}" fill="{INK2}"'
             f' font-size="9.5" text-anchor="end">{_esc(right)}</text>')
    return g


PW = NX * GRID_FT * PX_FT
PH = ND * GRID_FT * PX_FT


def _depth_axis(ox, oy):
    g = []
    for d, lab in ((0.0, "net"), (7.0, "kitchen"), (22.0, "baseline")):
        y = oy + (d - D_LO) * PX_FT
        g.append(f'<text x="{ox - 6:.1f}" y="{y + 3:.1f}" fill="{INK2}"'
                 f' font-size="9.5" text-anchor="end">{lab}</text>')
    return g


def short_name(n):
    """Last token -- difference-panel titles pair two names and the full
    ones overflow the panel width."""
    return n.split()[-1] if n else n


def render(A, B, order, names, led, title):
    cols = len(order)
    W = PAD_L + cols * PW + (cols - 1) * GAP_X + 24
    rows_y = [PAD_T + 78]
    rows_y.append(rows_y[0] + PH + GAP_Y)
    contrast_y = rows_y[1] + PH + GAP_Y
    H_total = contrast_y + PH + 34

    g = [f'<rect width="{W:.0f}" height="{H_total:.0f}" fill="{SURFACE}"/>']
    g.append(f'<text x="{PAD_L:.1f}" y="{PAD_T:.1f}" fill="{INK}"'
             f' font-size="17" font-weight="700">{_esc(title)}</text>')
    g.append(f'<text x="{PAD_L:.1f}" y="{PAD_T + 18:.1f}" fill="{INK2}"'
             f' font-size="11">Each panel is one player’s own half, net '
             f'at the top. Ends are folded and the far end mirrored, so '
             f'left/right is the player’s own.</text>')

    ROWS = ((A, "COURT FRAME — where they actually stood",
             "The physical half of the court, ends folded. A player who "
             "rotates through both service halves reads double-humped "
             "here; that is real, not an artefact."),
            (B, "RALLY-RELATIVE — controlled for which half they lined "
             "up in",
             "Each rally mirrored so the half the player STARTED in is "
             "always left of the dotted line. Right of it is the half "
             "the partner started in."))
    for ri, (dat, lab, sub) in enumerate(ROWS):
        oy = rows_y[ri]
        g.append(f'<text x="{PAD_L:.1f}" y="{oy - ROW_HEAD_DY:.1f}"'
                 f' fill="{INK}" font-size="13"'
                 f' font-weight="700">{_esc(lab)}</text>')
        g.append(f'<text x="{PAD_L:.1f}" y="{oy - ROW_SUB_DY:.1f}"'
                 f' fill="{INK2}" font-size="10.5">{_esc(sub)}</text>')
        dens = {u: smooth(hist2d(dat.get(u, np.zeros((0, 2)))))
                for u in order}
        # ONE scale for the whole row: these are small multiples and get
        # read against each other, so a colour must mean the same time
        # in every panel
        hi = max(norm_level(D) for D in dens.values()) or None
        for ci, u in enumerate(order):
            ox = PAD_L + ci * (PW + GAP_X)
            pts = dat.get(u, np.zeros((0, 2)))
            area = C.ellipse_area(pts, np.ones(len(pts))) if len(pts) else 0
            g += panel(normalise(dens[u], hi), ox, oy,
                       names.get(u, u[:8]),
                       f"{len(pts):,} frames · {area:,.0f} ft² (90%)")
            if ci == 0:
                g += _depth_axis(ox + (0 - X_LO) * PX_FT, oy)

    g.append(f'<text x="{PAD_L:.1f}" y="{contrast_y - ROW_HEAD_DY:.1f}"'
             f' fill="{INK}" font-size="13" font-weight="700">DIFFERENCE '
             f'— same gender, opposite team (rally-relative)</text>')
    g.append(f'<text x="{PAD_L:.1f}" y="{contrast_y - ROW_SUB_DY:.1f}"'
             f' fill="{INK2}" font-size="10.5">Men vs men and women vs '
             f'women only: a men-vs-women map would show a ROLE '
             f'difference in mixed, not a player difference.</text>')
    pairs = [(order[0], order[2]), (order[1], order[3])]
    densB = {u: smooth(hist2d(B.get(u, np.zeros((0, 2))))) for u in order}
    hiB = max(norm_level(D) for D in densB.values()) or None
    for ci, (ua, ub) in enumerate(pairs):
        ox = PAD_L + ci * (PW + GAP_X)
        Ha = normalise(densB[ua], hiB)
        Hb = normalise(densB[ub], hiB)
        g += panel(Ha - Hb, ox, contrast_y,
                   f"{short_name(names.get(ua, ua[:8]))} − "
                   f"{short_name(names.get(ub, ub[:8]))}",
                   "first minus second", diverging=True)
        if ci == 0:
            g += _depth_axis(ox + (0 - X_LO) * PX_FT, contrast_y)

    # legends live in the difference row's two empty columns
    lx = PAD_L + 2 * (PW + GAP_X)
    g += legend(lx, contrast_y + 24)
    g += div_legend(lx, contrast_y + 92, "second player", "first player")
    note = (f'{led["rallies_A"]} player-rallies mapped · '
            f'{led["dropped_no_start"]} dropped from the relative frame '
            f'(start half unreadable)')
    g.append(f'<text x="{lx:.1f}" y="{contrast_y + 148:.1f}"'
             f' fill="{INK2}" font-size="9.5">{_esc(note)}</text>')
    g.append(f'<text x="{lx:.1f}" y="{contrast_y + 163:.1f}"'
             f' fill="{INK2}" font-size="9.5">Position density measures '
             f'SPACE, never intent.</text>')

    body = "\n".join(g)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" '
            f'height="{H_total:.0f}" viewBox="0 0 {W:.0f} {H_total:.0f}" '
            f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">'
            f"\n{body}\n</svg>\n")


def order_players():
    """Partners adjacent, same-gender pairs in columns 0/2 and 1/3."""
    rows = list(csv.DictReader(
        open(ROOT / "data/coverage_players.csv")))
    seen, by = [], {}
    for r in rows:
        u = r["player_uuid"]
        if u not in by:
            by[u] = r
            seen.append(u)
    men = [u for u in seen if by[u]["gender"] == "M"]
    women = [u for u in seen if by[u]["gender"] == "F"]
    if len(men) != 2 or len(women) != 2:
        return seen[:4], {u: by[u]["player"] for u in seen}
    # partners adjacent: man 0 with the woman who is his partner
    p0 = by[men[0]]["partner_uuid"]
    w0 = p0 if p0 in women else women[0]
    w1 = [w for w in women if w != w0][0]
    return ([men[0], w0, men[1], w1],
            {u: by[u]["player"] for u in seen})


def stack_report(per, names, partner):
    """The pre-registered D1-D4 stacking diagnostics. Prints, returns rows."""
    from collections import defaultdict as dd
    by_rp = {(c, u): (t, x, d) for (g, c, u, t, x, d) in per}
    dis = dd(lambda: [0, 0])          # uuid -> [disagree, total]
    cross = dd(lambda: [0, 0, 0])     # uuid -> [n, start_cross, settled_cross]
    amb = dd(int)
    for (g, c, u, t, x, d) in per:
        sref = start_half(t, x)
        tref = settled_half(x)
        if sref is None or tref is None:
            continue
        s_side, t_side = sref > C.W_FT / 2, tref > C.W_FT / 2
        dis[u][1] += 1
        dis[u][0] += int(s_side != t_side)
        if abs(tref - C.W_FT / 2) < SETTLE_DEAD_FT:
            amb[u] += 1
        xs = C.W_FT - x if s_side else x
        xt = C.W_FT - x if t_side else x
        cross[u][0] += len(x)
        cross[u][1] += int(np.sum(xs > C.W_FT / 2))
        cross[u][2] += int(np.sum(xt > C.W_FT / 2))

    # D4: both partners on one side during the first retained second
    same = dd(lambda: [0, 0])
    rallies = dd(list)
    for (g, c, u, t, x, d) in per:
        rallies[c].append(u)
    for c, us in rallies.items():
        for u in us:
            pu = partner.get(u)
            if pu is None or pu not in us:
                continue
            t1, x1, _ = by_rp[(c, u)]
            t2, x2, _ = by_rp[(c, pu)]
            a, b = start_half(t1, x1), start_half(t2, x2)
            if a is None or b is None:
                continue
            same[u][1] += 1
            same[u][0] += int((a > C.W_FT / 2) == (b > C.W_FT / 2))

    print("D1/D2  ANCHOR DISAGREEMENT  (start-anchor vs settled-anchor)")
    print(f"  {'player':<22}{'disagree':>10}{'of':>6}{'rate':>9}"
          f"{'ambiguous':>11}")
    rows = []
    for u in sorted(dis, key=lambda k: -dis[k][0] / max(dis[k][1], 1)):
        dsg, tot = dis[u]
        print(f"  {names.get(u, u[:8]):<22}{dsg:>10}{tot:>6}"
              f"{dsg / max(tot, 1):>9.1%}{amb[u]:>11}")
        rows.append((names.get(u, u[:8]), dsg, tot, amb[u]))
    print()
    print("D3  CROSSING FRACTION under each anchor (frames past x=10)")
    print(f"  {'player':<22}{'start':>9}{'settled':>10}{'change':>10}")
    for u in cross:
        n, cs, ct = cross[u]
        print(f"  {names.get(u, u[:8]):<22}{cs / n:>9.2%}{ct / n:>10.2%}"
              f"{(ct - cs) / n:>+10.2f}pp".replace("pp", " pp"))
    print()
    print("D4  BOTH PARTNERS ONE SIDE at the start of retained frames")
    print("    (geometric signature of a stack still unwinding)")
    for u in same:
        a, b = same[u]
        print(f"  {names.get(u, u[:8]):<22}{a}/{b} = {a / max(b, 1):.1%}")
    return rows


def save_cache(path, A, B, led, per):
    d = {f"A:{u}": v for u, v in A.items()}
    d.update({f"B:{u}": v for u, v in B.items()})
    d["__led__"] = np.array(sorted(led.items()), dtype=object)
    # per-rally table, flattened: one row per detection plus a rally
    # index, so any anchor can be recomputed from the cache alone
    if per:
        idx, game, cum, uu, ts, xs, ds = [], [], [], [], [], [], []
        for i, (g, c, u, t, x, dd) in enumerate(per):
            idx.append(np.full(len(t), i))
            game.append(g), cum.append(c), uu.append(u)
            ts.append(t), xs.append(x), ds.append(dd)
        d["__idx__"] = np.concatenate(idx)
        d["__t__"] = np.concatenate(ts)
        d["__x__"] = np.concatenate(xs)
        d["__d__"] = np.concatenate(ds)
        d["__meta__"] = np.array(list(zip(game, cum, uu)), dtype=object)
    np.savez_compressed(path, **d)


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    A = {k[2:]: z[k] for k in z.files if k.startswith("A:")}
    B = {k[2:]: z[k] for k in z.files if k.startswith("B:")}
    led = defaultdict(int)
    for k, v in z["__led__"]:
        led[k] = int(v)
    per = []
    if "__idx__" in z.files:
        idx, t, x, d = z["__idx__"], z["__t__"], z["__x__"], z["__d__"]
        meta = z["__meta__"]
        order = np.argsort(idx, kind="stable")
        idx, t, x, d = idx[order], t[order], x[order], d[order]
        bounds = np.searchsorted(idx, np.arange(len(meta) + 1))
        for i, (g, c, u) in enumerate(meta):
            lo, hi = bounds[i], bounds[i + 1]
            per.append((int(g), int(c), u, t[lo:hi], x[lo:hi], d[lo:hi]))
    return A, B, led, per


def selftest():
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok = ok and bool(cond)

    print("own_frame: ends fold, far end mirrors")
    a = own_frame([[5.0, 30.0]], "near")[0]
    b = own_frame([[5.0, 14.0]], "far")[0]
    chk(abs(a[0] - 5.0) < 1e-9 and abs(a[1] - 8.0) < 1e-9,
        f"near (5,30) -> x'=5 d=8 (got {a[0]:.2f},{a[1]:.2f})")
    chk(abs(b[0] - 15.0) < 1e-9 and abs(b[1] - 8.0) < 1e-9,
        f"far  (5,14) -> x'=15 d=8, mirrored (got {b[0]:.2f},{b[1]:.2f})")
    n = own_frame([[3.0, C.NET_Y + 7.0]], "near")[0]
    chk(abs(n[1] - 7.0) < 1e-9, "kitchen line lands at d=7")

    print("start_half: first second only, min-frame floor")
    ts = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    xp = np.array([15.0, 15.0, 15.0, 2.0, 2.0])
    chk(abs(start_half(ts, xp) - 15.0) < 1e-9,
        "late frames do not move the start half")
    chk(start_half(np.array([0.0, 0.05]), np.array([1.0, 2.0])) is None,
        "2 frames in the first second -> None (drop, not guess)")

    print("accumulate: mirror rule + drop ledger")
    ts_ok = np.array([0.0, 0.2, 0.4, 1.5])
    xy_hi = np.column_stack([np.full(4, 15.0),
                             np.full(4, C.NET_Y + 8.0)])   # near, x'=15
    xy_lo = np.column_stack([np.full(4, 5.0),
                             np.full(4, C.NET_Y + 8.0)])   # near, x'=5
    rt = {1: [(1, {"hi": (ts_ok, xy_hi, "near"),
                   "lo": (ts_ok, xy_lo, "near")}, None)]}
    A, B, led, per = accumulate(rt)
    chk(abs(A["hi"][:, 0].mean() - 15.0) < 1e-9,
        "frame A keeps the physical half (15 stays 15)")
    chk(abs(B["hi"][:, 0].mean() - 5.0) < 1e-9,
        "frame B mirrors a right-half start onto the left (15 -> 5)")
    chk(abs(B["lo"][:, 0].mean() - 5.0) < 1e-9,
        "frame B leaves a left-half start alone (5 stays 5)")
    chk(led["mirrored"] == 1 and led["rallies_B"] == 2,
        f"ledger: 1 mirrored, 2 kept (got {led['mirrored']},"
        f"{led['rallies_B']})")
    short = np.array([0.0, 0.05])
    rt2 = {1: [(1, {"hi": (short, xy_hi[:2], "near")}, None)]}
    A2, B2, led2, _ = accumulate(rt2)
    chk("hi" in A2 and "hi" not in B2 and led2["dropped_no_start"] == 1,
        "short-start rally: in frame A, dropped from frame B, ledgered")

    print("hist2d / smooth")
    H = hist2d([[0.0, 0.0], [0.0, 0.0], [19.9, 21.9]])
    chk(H.sum() == 3, f"all 3 points binned (got {H.sum():.0f})")
    chk(H[int((0 - D_LO) / GRID_FT), int((0 - X_LO) / GRID_FT)] == 2,
        "duplicate points stack in one cell")
    chk(hist2d([[99.0, 99.0]]).sum() == 0, "out-of-grid point dropped")
    S = smooth(H)
    chk(S.min() >= 0 and S.sum() <= H.sum() + 1e-9,
        f"smoothing stays non-negative and never ADDS mass "
        f"({S.sum():.2f} vs {H.sum():.0f})")
    mid = hist2d([[10.0, 11.0]])            # far from every boundary
    chk(abs(smooth(mid).sum() - 1.0) < 1e-6,
        "an interior point keeps all its mass under zero padding")
    rim = smooth(hist2d([[10.0, 0.0]]))     # on the net edge
    chk(rim[:, 0].sum() == 0.0 and rim.sum() < 1.0,
        "a rim point fades outward instead of banding along the edge")
    off = hist2d([[-1.0, 3.0]])
    chk(off.sum() == 1, "off-court x=-1 is inside the padded grid")

    print("normalise / levels")
    N = normalise(np.array([[0.0, 1.0], [2.0, 100.0]]))
    chk(N.max() > 1.0 and N.min() == 0.0,
        "p99.5 normalisation lets a single hot cell exceed 1 (clipped later)")
    idx = _levels(np.array([[0.0, 0.5, 1.0, 3.0]]), BLUE)
    chk(idx[0, 0] == 0 and idx[0, -1] == len(BLUE) - 1,
        "zero -> blank, over-max clipped to the darkest step")
    d = _div_levels(np.array([[-2.0, 0.0, 2.0]]), len(BLUE_ARM))
    chk(d[0, 0] == -len(BLUE_ARM) and d[0, 1] == 0
        and d[0, 2] == len(BLUE_ARM), "diverging clips both arms, 0 neutral")

    print("shared normalisation makes panels comparable")
    Hbig = smooth(hist2d(np.repeat([[10.0, 11.0]], 40, axis=0)))
    Hsml = smooth(hist2d(np.repeat([[10.0, 11.0]], 10, axis=0)))
    hi = max(norm_level(Hbig), norm_level(Hsml))
    la = _levels(normalise(Hbig, hi), BLUE).max()
    lb = _levels(normalise(Hsml, hi), BLUE).max()
    chk(lb < la, f"a 4x-sparser panel reads LIGHTER on a shared scale "
                 f"(levels {lb} vs {la})")
    la2 = _levels(normalise(Hbig), BLUE).max()
    lb2 = _levels(normalise(Hsml), BLUE).max()
    chk(la2 == lb2, f"...whereas per-panel scaling makes them identical "
                    f"({lb2} vs {la2}) -- the bug shared scaling fixes")

    print("cache round-trip")
    import tempfile, os
    fd, cp = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    save_cache(cp, A, B, led, per)
    A2, B2, led2, per2 = load_cache(cp)
    os.unlink(cp)
    chk(set(A2) == set(A) and np.allclose(A2["hi"], A["hi"]),
        "frame A survives the cache unchanged")
    chk(led2["mirrored"] == led["mirrored"]
        and led2["rallies_B"] == led["rallies_B"],
        "the drop ledger survives the cache")
    chk(len(per2) == len(per)
        and all(np.allclose(a[4], b[4]) and a[2] == b[2]
                for a, b in zip(sorted(per, key=lambda r: (r[1], r[2])),
                                sorted(per2, key=lambda r: (r[1], r[2])))),
        "the per-rally table survives the cache (any anchor recomputable)")

    print("settled anchor vs start anchor")
    # a STACK: starts on the right, plays the whole rally on the left
    ts_s = np.arange(0.0, 4.0, 0.2)
    x_s = np.where(ts_s < 1.0, 15.0, 5.0)
    d_s = np.full(len(ts_s), 8.0)
    per_stack = [(1, 1, "p", ts_s, x_s, d_s)]
    Bs, led_s = frame_settled(per_stack)
    start_ref = start_half(ts_s, x_s)
    chk(start_ref > 10 and settled_half(x_s) < 10,
        f"the two anchors DISAGREE on a stack "
        f"(start {start_ref:.0f}, settled {settled_half(x_s):.0f})")
    chk(abs(np.median(Bs["p"][:, 0]) - 5.0) < 1e-9,
        "settled anchor leaves the played side alone (no phantom crossing)")
    q = np.column_stack([x_s, d_s]).copy()
    q[:, 0] = C.W_FT - q[:, 0]          # what the START anchor would do
    c_start = float(np.mean(q[:, 0] > 10))
    c_settled = float(np.mean(Bs["p"][:, 0] > 10))
    # NOTE: settled REDUCES the phantom crossing but cannot remove it --
    # during the unwind the player really is past the centreline.  Only
    # dropping the unwind frames would zero it, and that is a different
    # (data-losing) choice, deliberately not made here.
    chk(c_start > 0.7 and 0.2 < c_settled < 0.3 and c_settled < c_start / 2,
        f"start anchor reports {c_start:.0%} crossing on this stack vs "
        f"settled {c_settled:.0%} -- reduced 3x, not eliminated")
    per_mid = [(1, 1, "p", ts_s, np.full(len(ts_s), 10.2), d_s)]
    _, led_m = frame_settled(per_mid)
    chk(led_m["ambiguous_centre"] == 1,
        "a centreline-dweller is ledgered as an ambiguous anchor")

    print("orientation: net at the panel top")

    def _y_of(rects):
        return [float(r.split('y="')[1].split('"')[0]) for r in rects]
    at_net = hist2d([[10.0, 0.0]])
    at_base = hist2d([[10.0, 22.0]])
    y_net = _y_of(_cells(_levels(at_net, BLUE), 0.0, 0.0,
                         lambda lv: None if lv == 0 else BLUE[lv]))
    y_base = _y_of(_cells(_levels(at_base, BLUE), 0.0, 0.0,
                          lambda lv: None if lv == 0 else BLUE[lv]))
    chk(y_net and y_base and min(y_net) < min(y_base),
        "a frame at the net draws ABOVE a frame at the baseline")
    net_line = [l for l in _court(0.0, 0.0) if 'stroke-width="1.8"' in l]
    chk(len(net_line) == 1, "one net line drawn")
    y_line = float(net_line[0].split('y1="')[1].split('"')[0])
    chk(abs(y_line - min(y_net)) <= GRID_FT * PX_FT,
        f"the net COURT LINE sits on the net cells "
        f"(line {y_line:.1f} vs cell {min(y_net):.1f}) -- caption and "
        f"geometry agree")

    print("run-length merge covers the same area")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(BLUE), size=(ND, NX))
    rects = _cells(idx, 0.0, 0.0, lambda lv: None if lv == 0 else BLUE[lv])
    area = 0.0
    for r in rects:
        w = float(r.split('width="')[1].split('"')[0])
        h = float(r.split('height="')[1].split('"')[0])
        area += w * h
    want = float((idx > 0).sum()) * (GRID_FT * PX_FT) ** 2
    chk(abs(area - want) < 1e-6,
        f"merged rect area == unmerged ({area:.0f} vs {want:.0f})")

    print("\nSELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--no-cam-gate", action="store_true")
    ap.add_argument("--spotcheck", default="/nonexistent.csv")
    ap.add_argument("--swaps", default="")
    ap.add_argument("--track-map", default="")
    ap.add_argument("--anchor-free", default="")
    ap.add_argument("--vod", default="")
    ap.add_argument("--event", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--match-id", default="")
    ap.add_argument("--title", default="Court occupancy")
    ap.add_argument("--out", default="")
    # the identity chain takes ~12 min on a full match; the renderer is
    # instant.  Cache the accumulated points so layout work does not pay
    # for the pipeline every time.
    ap.add_argument("--cache", default="",
                    help="npz of accumulated points; read if it exists, "
                         "else written after the run")
    ap.add_argument("--stack-report", action="store_true",
                    help="pre-registered D1-D4 stacking diagnostics")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.cache and Path(a.cache).exists():
        A, B, led, per = load_cache(a.cache)
        print(f"cache HIT {a.cache} — pipeline skipped")
    else:
        for req in ("pose_dir", "court", "windows", "lineup"):
            if not getattr(a, req):
                ap.error(f"--{req} required")
        got = {}
        # collect-only: this instrument renders a picture and must never
        # touch the committed coverage tables
        C.run(a, collect=lambda rt: got.update(rt), write=False)
        A, B, led, per = accumulate(got)
        if a.cache:
            save_cache(a.cache, A, B, led, per)
            print(f"cache WRITTEN {a.cache}")
    order, names = order_players()
    if a.stack_report:
        partner = {}
        import csv as _csv
        for r in _csv.DictReader(open(ROOT / "data/coverage_players.csv")):
            partner[r["player_uuid"]] = r["partner_uuid"]
        stack_report(per, names, partner)
        return
    svg = render(A, B, order, names, led, a.title)
    out = a.out or str(ROOT / "data/vision/coverage_heatmap.svg")
    Path(out).write_text(svg)
    print(f"\nwrote {out}  ({len(svg) / 1024:.0f} KB)")
    for u in order:
        na, nb = len(A.get(u, ())), len(B.get(u, ()))
        print(f"  {names.get(u, u[:8]):<24} frame A {na:>6,} frames"
              f"   frame B {nb:>6,}")
    print(f"  ledger: {dict(led)}")


if __name__ == "__main__":
    main()
