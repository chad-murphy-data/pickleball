"""Render a slate into carousel slides (PNG) from social/templates/slide.html.

    python social/render.py --date 2026-09-06          # social/out/<date>/slide-NN.png
    python social/render.py --date 2026-09-06 --html   # also keep the HTML per slide

Deck = cover + one slide per bracket (Men's Doubles, Women's Doubles, Mixed
Doubles, Men's Singles, Women's Singles; MLP days get an MLP Matchups slide
too) + the fixed methodology slide.  Headlines are the plain bracket names
— no editorial voice by design.  Layout follows the Claude Design handoff
(design_handoff_pickles_carousel): match cards with the close-match rule
(favorite under 65% -> off-white number and bar, dashed 50% mark), a
score-distribution block under a lone match, SCENARIO A/B cards when a
finalist's opponent is still on court.  A bracket with more than nine
matches is paginated; density (3 big cards / 6 compact / 9 list rows)
follows the row count so a Friday round-of-32 still fits.

Rendering: headless Chromium via Playwright, 1080x1350 (4:5), the vendored
Anton + Space Mono + Space Grotesk fonts inlined as data URIs so output is
identical offline, in CI, and on the droplet.  PW_CHROMIUM=/path/to/chrome
overrides the browser binary (this environment's Playwright build is newer
than its bundled browser; the workflow runs `playwright install chromium`).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "social"))
from slate import EPS, OUT, parse_date          # noqa: E402

TPL_DIR = ROOT / "social" / "templates"
SITE_URL = os.environ.get("SOCIAL_SITE_URL", "chad-murphy-data.github.io/pickleball")
W, H = 1080, 1350
PAGE = 9                 # matches per bracket slide before paginating
CLOSE = 0.65             # design rule: favorite under this -> off-white number + 50% mark
FLOOR_HI = 1 - EPS / 2 - 1e-6

FONTS = [("Anton", "400", "Anton-Regular.ttf"),
         ("Space Mono", "400", "SpaceMono-Regular.ttf"),
         ("Space Mono", "700", "SpaceMono-Bold.ttf"),
         ("Space Grotesk", "300 700", "SpaceGrotesk-Variable.ttf")]

ROUND_WORD = {"finals": "Finals", "final": "Finals", "semi-finals": "Semifinals",
              "semifinals": "Semifinals", "quarter-finals": "Quarterfinals",
              "quarterfinals": "Quarterfinals", "round of 16": "Round of 16",
              "round of 32": "Round of 32", "round of 64": "Round of 64",
              "bronze": "Bronze", "gold": "Gold"}
ROUND_ABBR = {"Round of 64": "R64", "Round of 32": "R32", "Round of 16": "R16",
              "Quarterfinals": "QF", "Semifinals": "SF"}
CARD_LABEL = {"Finals": "Gold medal match", "Bronze": "Bronze medal match"}
COVER_WORD = {"Finals": "Championship", "Semifinals": "Semifinal",
              "Quarterfinals": "Quarterfinal"}


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def font_css() -> str:
    parts = []
    for fam, wt, fn in FONTS:
        p = TPL_DIR / "fonts" / fn
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        parts.append(f"@font-face{{font-family:\"{fam}\";font-weight:{wt};"
                     f"src:url(data:font/ttf;base64,{b64}) format(\"truetype\");}}")
    return "\n".join(parts)


# ---- number + name formatting ---------------------------------------------------
def fmt_pct(p: float) -> str:
    """Whole numbers except at the cap, where the floor is the story
    (98.95 -> '98.9%', never '99%' rounding into a number we don't mean)."""
    x = 100 * p
    if x >= 98.5 or x <= 1.5:
        return f"{int(x * 10) / 10:.1f}%"
    return f"{round(x)}%"


def surname(full: str) -> str:
    parts = full.replace("  ", " ").strip().split(" ")
    if len(parts) == 1:
        return parts[0]
    first, last = parts[0], parts[-1]
    if len(first) <= 2 and first.isupper():      # "JW Johnson" reads better whole
        return f"{first} {last}"
    return last


def side_short(side: dict, singles: bool) -> str:
    if side.get("tbd"):
        return "TBD"
    names = side["names"]
    if side.get("is_team"):                 # MLP matchup rows: the franchise name whole
        return names[0]
    shorts = [surname(n) for n in names]
    lasts = [n.split()[-1] for n in names]  # two Johnsons on one side -> "J. Johnson / JW Johnson"
    if len(lasts) == 2 and lasts[0] == lasts[1]:
        shorts = [sh if " " in sh else f"{n.split()[0][0]}. {sh}" for n, sh in zip(names, shorts)]
    return " / ".join(shorts)


def side_long(side: dict) -> str:
    if side.get("tbd"):
        return "TBD"
    return " / ".join(side["names"]) if side["names"] else side.get("team", "TBD")


def oriented(row: dict) -> tuple[dict, dict, float | None]:
    """(favorite, underdog, favorite's prob) — the underdog's probability
    is implied; the bar is drawn once, from the favorite's side."""
    p = row.get("p1")
    if p is None:
        if row["t1"].get("tbd") and not row["t2"].get("tbd"):
            return row["t2"], row["t1"], None
        return row["t1"], row["t2"], None
    return (row["t1"], row["t2"], p) if p >= 0.5 else (row["t2"], row["t1"], 1 - p)


def modal_for(modal: str | None, flip: bool) -> str | None:
    if not modal:
        return None
    a, b = modal.split("-")
    return f"{b}–{a}" if flip else f"{a}–{b}"


def round_word(text: str) -> str:
    return ROUND_WORD.get((text or "").strip().lower(), (text or "").strip())


def deck_round(slate: dict) -> str | None:
    """The one round every scheduled match shares, else None."""
    rs = {round_word(r["round"]) for s in slate["slates"] if s["tour"] == "PPA"
          for r in s["rows"] if not r["consolation"]}
    return rs.pop() if len(rs) == 1 else None


def n_games_str() -> str:
    try:
        n = sum(1 for _ in (ROOT / "data" / "games.csv").open()) - 1
        return f"{(n // 1000) * 1000:,}"
    except OSError:
        return "36,000"


def pretty_date(iso: str) -> str:
    return dt.date.fromisoformat(iso).strftime("%b %-d %Y")


def short_event(title: str) -> str:
    """'PPA Tour: Veolia Pickleball National Championships' -> 'PPA National Championships'."""
    t = title.strip()
    ppa = t.lower().startswith("ppa")
    t = t.split(":", 1)[1] if ppa and ":" in t else t
    t = t.replace("Pickleball National Championships", "National Championships")
    for junk in ("Veolia", "Carvana", "powered by"):
        t = t.replace(junk, "")
    return " ".join((["PPA"] if ppa else []) + t.split())


# ---- slide builders ------------------------------------------------------------------
def meter_html(p: float, close: bool) -> str:
    mid = '<span class="mid"></span>' if close else ""
    return f'<div class="meter"><span class="fill" style="width:{round(100 * p, 1)}%"></span>{mid}</div>'


def one_card(label: str, fav_txt: str, dog_txt: str, p: float | None, notes: list[str]) -> str:
    close = p is not None and p < CLOSE
    pct = f'<span class="pct">{fmt_pct(p)}</span>' if p is not None else '<span class="pct na">—</span>'
    meter = meter_html(p, close) if p is not None else meter_html(0, False)
    return (f'<div class="card{" close" if close else ""}"><div class="lbl">{esc(label)}</div>'
            f'<div class="top"><span class="fav">{esc(fav_txt)}</span>{pct}</div>{meter}'
            f'<div class="bot"><span>vs <span class="dog">{esc(dog_txt)}</span></span>'
            f'<span class="meta">{esc(" · ".join(n for n in notes if n))}</span></div></div>')


def notes_for(row: dict, fav_is_t1: bool, p: float | None, modal: str | None) -> list[str]:
    notes = []
    if p is not None and p >= FLOOR_HI:
        notes.append("at the floor — the cap, not the model")
    elif modal and p is not None:
        notes.append(f"modal game {modal}")
    if p is not None and abs(p - 0.5) < 0.1:
        notes.append("nearly even")
    if row.get("p_db") is not None and p is not None:
        pdb = row["p_db"] if fav_is_t1 else 1 - row["p_db"]
        notes.append(f"DreamBreaker {fmt_pct(pdb)}")
    if row.get("note") and p is None and not row.get("branches"):
        notes.append(row["note"])
    if row.get("start"):
        notes.append(row["start"])
    return notes


def cards_for_row(row: dict, singles: bool, label: str) -> list[str]:
    """A priced match -> one card.  A finalist whose opponent is still on
    court -> one SCENARIO card per possible opponent, favorite-first each."""
    if row.get("branches"):
        known = row["t2"] if row["t1"].get("tbd") else row["t1"]
        cards = []
        for i, br in enumerate(row["branches"]):
            p_known = br["p_known"]
            fav, dog, p = (known, br, p_known) if p_known >= 0.5 else (br, known, 1 - p_known)
            modal = modal_for(br.get("modal"), flip=fav is br)
            lab = f"Scenario {chr(65 + i)} — if {side_short(br, singles)} advance{'s' if singles else ''}"
            notes = [f"modal game {modal}" if modal else "", "nearly even" if abs(p - 0.5) < 0.1 else ""]
            cards.append(one_card(lab, side_short(fav, singles), side_short(dog, singles), p, notes))
        return cards
    fav, dog, p = oriented(row)
    modal = modal_for(row.get("modal"), flip=fav is row["t2"])
    lab = label + (f" · opponent TBD" if p is None and (row["t1"].get("tbd") or row["t2"].get("tbd")) else "")
    return [one_card(lab, side_short(fav, singles), side_short(dog, singles), p,
                     notes_for(row, fav is row["t1"], p, modal))]


def dist_html(row: dict, singles: bool) -> str:
    """Score-distribution block from the favorite's perspective (design
    handoff: fills a single-match slide; winning lines chartreuse, losing sage)."""
    ser = row.get("series")
    if not ser:
        return ""
    fav, _, _ = oriented(row)
    flip = fav is row["t2"]
    rows = []
    for k, v in ser.items():
        a, b = k.split("-")
        if flip:
            a, b = b, a
        rows.append((int(a), int(b), v))
    rows.sort(key=lambda t: (-t[0], t[1]))
    cells = ""
    for a, b, v in rows:
        lose = " lose" if a < b else ""
        cells += (f'<span class="k{lose}">{a}–{b}</span><div class="bar{lose}"><span style="width:{round(100 * v, 1)}%"></span></div>'
                  f'<span class="v{lose}">{fmt_pct(v)}</span>')
    return (f'<div class="dist"><div class="lbl">Score distribution — {esc(side_short(fav, singles))} perspective</div>'
            f'<div class="grid">{cells}</div></div>')


def row_label(row: dict, i: int, n_in_round: dict) -> str:
    r = round_word(row["round"]) or "Match"
    if row["t1"].get("team"):
        return f"{row['t1']['team']} vs {row['t2']['team']} · {row['round']}"
    if n_in_round.get(r, 0) > 1:
        return f"{ROUND_ABBR.get(r, r)} {i}"
    return CARD_LABEL.get(r, r)


def bracket_slides(slate: dict, s: dict, idx_start: int) -> list[dict]:
    rows = s["rows"]
    if not rows:
        return []
    singles = s["bracket"] in ("MS", "WS")
    pages = [rows[i:i + PAGE] for i in range(0, len(rows), PAGE)]
    n_in_round: dict[str, int] = {}
    for r in rows:
        k = round_word(r["round"]) or "Match"
        n_in_round[k] = n_in_round.get(k, 0) + 1
    counters: dict[str, int] = {}
    out = []
    for pi, page in enumerate(pages):
        cards = []
        for r in page:
            k = round_word(r["round"]) or "Match"
            counters[k] = counters.get(k, 0) + 1
            cards += cards_for_row(r, singles, row_label(r, counters[k], n_in_round))
        density = "" if len(cards) <= 3 else "compact" if len(cards) <= 6 else "list"
        rounds = sorted({round_word(r["round"]) for r in page if r["round"]}, key=str.lower)
        kick = f"{idx_start + pi:02d} — " + (rounds[0] if len(rounds) == 1 else " / ".join(rounds) if rounds else s["title"])
        if any(r.get("branches") for r in page):
            kick += " · opponent TBD"
        if len(pages) > 1:
            kick += f" ({pi + 1}/{len(pages)})"
        extra = dist_html(page[0], singles) if len(page) == 1 and len(cards) == 1 else ""
        bo = next((r.get("best_of") for r in page if r.get("best_of")), None)
        fmt = f"best-of-{bo}" if bo else next((r["format"] for r in page if r.get("format")), "")
        out.append({
            "kind": "bracket", "title": s["title"],
            "hdr_right": f"{s.get('venue') or short_event(s['event'])} · {pretty_date(slate['date'])}",
            "body": (f'<div class="kicker">{esc(kick)}</div><h1>{esc(s["title"])}</h1>'
                     f'<div class="cards {density}">{"".join(cards)}</div>{extra}'),
            "ftr_left": "Win probability" + (f", {fmt}" if fmt else "") + " · error bars honest",
            "caption": f"{s['title']} — {short_event(s['event'])}, {pretty_date(slate['date'])}",
        })
    return out


def cover_slide(slate: dict) -> dict:
    d = dt.date.fromisoformat(slate["date"])
    ev = slate["slates"][0] if slate["slates"] else {}
    rw = deck_round(slate)
    weekday = d.strftime("%A")
    word = COVER_WORD.get(rw or "", rw)
    lines = ([esc(word)] if word else []) + [f"{weekday},", '<span class="hl">forecast.</span>']
    what = f"Every {rw.lower().rstrip('s')}" if rw else "Every scheduled match"
    kick = " · ".join(x for x in (short_event(ev.get("event", "")), ev.get("venue")) if x)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    return {
        "kind": "cover", "title": "Forecast",
        "hdr_right": "Pro pickleball, probabilistically",
        "body": (f'<div class="kicker">{esc(kick)}</div><h1>{"<br>".join(lines)}</h1>'
                 f'<div class="sub">{what}, every division. Win probabilities from one Bayesian '
                 f'model of {n_games_str()} games — committed before first serve.</div>'
                 f'<div class="chips"><span class="chip">run {stamp} :: {slate["n_priced"]} of '
                 f'{slate["n_matches"]} priced</span><span class="chip ok">committed pre-serve ... [ok]</span></div>'),
        "ftr_left": SITE_URL, "ftr_class": "url",
        "caption": f"{short_event(ev.get('event', 'Pro pickleball'))} — {weekday} {pretty_date(slate['date'])} forecast",
    }


# methodology copy from the design handoff (the DUPR comparison line was
# dropped: CLAUDE.md house rule, no "we beat DUPR" scoreboard)
METHODS = [
    ("The data", "Every pro doubles and singles game since January 2024 — both tours, "
                 "~{games} games, ~3,600 players, permanent IDs."),
    ("The model", "Bayesian, fit jointly: every game's point margin = your players' value − "
                  "their players' value + luck. Every rating adjusts for every partner and "
                  "opponent automatically. Skeptical by design — hot streaks must earn their way in."),
    ("The weakest link", "A team is not the sum of its players: every point of skill gap between "
                         "partners costs about half a point of team strength (γ ≈ −0.18)."),
    ("The test", "Frozen on pre-June data, scored on 884 unseen games: 77.4% of winners "
                 "called correctly."),
    ("The honesty rules", "No probability is ever 0% or 100% — ~1 in 100 sure things still lose, "
                          "so 98.9% is the ceiling. No chemistry bonuses. No cross-gender rankings "
                          "as fact. Forecasts committed before first serve, graded in public."),
]


def methods_slide(n: int) -> dict:
    games = n_games_str()
    blocks = "".join(f'<div class="block"><div class="k">{esc(k)}</div><div class="t">{esc(t.format(games=games))}</div></div>'
                     for k, t in METHODS)
    return {"kind": "methods", "title": "Methodology",
            "hdr_right": "How the numbers are made",
            "body": (f'<div class="kicker">{n:02d} — Methodology</div>'
                     f'<h1>One model, {games} games, receipts kept</h1>'
                     f'<div class="blocks">{blocks}</div>'),
            "ftr_left": f"Full writeup: {SITE_URL}", "ftr_class": "url",
            "caption": "How the numbers are made (same every post)"}


def build_deck(slate: dict, cover: bool = True) -> list[dict]:
    slides = [cover_slide(slate)] if cover else []
    for s in slate["slates"]:
        slides += bracket_slides(slate, s, len(slides) + 1)
    slides.append(methods_slide(len(slides) + 1))
    return slides


def slide_html(sl: dict, i: int, n: int, fonts: str) -> str:
    tpl = (TPL_DIR / "slide.html").read_text()
    fields = {"TITLE": esc(sl["title"]), "FONT_CSS": fonts, "KIND": sl["kind"],
              "HDR_RIGHT": esc(sl["hdr_right"]), "BODY": sl["body"],
              "FTR_LEFT": esc(sl["ftr_left"]), "FTR_CLASS": sl.get("ftr_class", ""),
              "FTR_RIGHT": f"{i} / {n}" + (" →" if i < n else "")}
    for k, v in fields.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    return tpl


def render_pngs(htmls: list[str], out_dir: Path) -> list[Path]:
    from playwright.sync_api import sync_playwright
    paths = []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=os.environ.get("PW_CHROMIUM") or None)
        except Exception:                                   # noqa: BLE001
            cands = glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")
            if not cands:
                raise
            browser = pw.chromium.launch(executable_path=cands[-1])
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        for i, h in enumerate(htmls, 1):
            page.set_content(h, wait_until="load")
            page.evaluate("document.fonts.ready")
            p = out_dir / f"slide-{i:02d}.png"
            page.screenshot(path=str(p), clip={"x": 0, "y": 0, "width": W, "height": H})
            paths.append(p)
        browser.close()
    return paths


def render(date_iso: str, keep_html: bool = False, cover: bool = True) -> dict:
    out_dir = OUT / date_iso
    slate = json.loads((out_dir / "slate.json").read_text())
    deck = build_deck(slate, cover=cover)
    fonts = font_css()
    htmls = [slide_html(sl, i, len(deck), fonts) for i, sl in enumerate(deck, 1)]
    for old in out_dir.glob("slide-*.png"):
        old.unlink()
    paths = render_pngs(htmls, out_dir)
    if keep_html:
        for i, h in enumerate(htmls, 1):
            (out_dir / f"slide-{i:02d}.html").write_text(h)
    manifest = {"date": date_iso, "slides": [
        {"png": p.name, "kind": sl["kind"], "caption": sl["caption"][:180]}
        for p, sl in zip(paths, deck)]}
    (out_dir / "deck.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="tomorrow")
    ap.add_argument("--html", action="store_true", help="keep per-slide HTML next to the PNGs")
    ap.add_argument("--no-cover", action="store_true")
    args = ap.parse_args()
    d = str(parse_date(args.date))
    m = render(d, keep_html=args.html, cover=not args.no_cover)
    for s in m["slides"]:
        print(f"{s['png']}  [{s['kind']}]  {s['caption']}")


if __name__ == "__main__":
    main()
