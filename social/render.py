"""Render a slate into carousel slides (PNG) from social/templates/slide.html.

    python social/render.py --date 2026-09-06          # social/out/<date>/slide-NN.png
    python social/render.py --date 2026-09-06 --html   # also keep the HTML per slide

Deck = cover + one slide per bracket (Men's Doubles, Women's Doubles, Mixed
Doubles, Men's Singles, Women's Singles; MLP days get an MLP Matchups slide
too) + the fixed methods slide.  Headlines are the plain bracket names —
no editorial voice by design.  A bracket with more than ten matches is
paginated; density (3 big cards / 6 compact / 9 list rows) follows the
row count so a Friday round-of-32 still fits.

Rendering: headless Chromium via Playwright, 1080x1350 (4:5), the vendored
Anton + Space Mono fonts inlined as data URIs so output is identical
offline, in CI, and on the droplet.  PW_CHROMIUM=/path/to/chrome overrides
the browser binary (this environment's Playwright build is newer than its
bundled browser; the workflow runs `playwright install chromium`).
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
from slate import BRACKETS, EPS, OUT, parse_date          # noqa: E402

TPL_DIR = ROOT / "social" / "templates"
SITE_URL = os.environ.get("SOCIAL_SITE_URL", "chad-murphy-data.github.io/pickleball")
W, H = 1080, 1350
PAGE = 9                 # matches per bracket slide before paginating
FLOOR_HI = 1 - EPS / 2 - 1e-6

FONTS = [("Anton", 400, "Anton-Regular.ttf"),
         ("Space Mono", 400, "SpaceMono-Regular.ttf"),
         ("Space Mono", 700, "SpaceMono-Bold.ttf")]

ROUND_WORD = {"finals": "Finals", "final": "Finals", "semi-finals": "Semifinals",
              "semifinals": "Semifinals", "quarter-finals": "Quarterfinals",
              "quarterfinals": "Quarterfinals", "round of 16": "Round of 16",
              "round of 32": "Round of 32", "round of 64": "Round of 64",
              "bronze": "Bronze", "gold": "Gold"}


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
    """Integers except at the extremes, where the floor is the story
    (98.95 -> '98.9%', never '99%' rounding into a number we don't mean)."""
    x = 100 * p
    if x >= 98.5 or x <= 1.5:
        return f"{int(x * 10) / 10:.1f}%"
    return f"{round(x)}%"


def surname(full: str) -> str:
    parts = full.replace("  ", " ").strip().split(" ")
    if len(parts) == 1:
        return parts[0]
    # "Tyra Hurricane Black" -> Black; "JW Johnson" -> JW Johnson (initial-style first names read better whole)
    first, last = parts[0], parts[-1]
    if len(first) <= 2 and first.isupper():
        return f"{first} {last}"
    return last


def side_short(side: dict, singles: bool) -> str:
    if side.get("tbd"):
        return "TBD"
    names = side["names"]
    if side.get("is_team"):                 # MLP matchup rows: the franchise name whole
        return names[0]
    shorts = [surname(n) for n in names]
    # two Johnsons on one side: initial both ("J. Johnson / JW Johnson")
    lasts = [n.split()[-1] for n in names]
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


# ---- slide builders ------------------------------------------------------------------
def card_html(row: dict, singles: bool, label: str) -> str:
    fav, dog, p = oriented(row)
    if p is None:
        pct = '<span class="pct na">—</span>'
        fill = 0
    else:
        pct = f'<span class="pct">{fmt_pct(p)}</span>'
        fill = round(100 * p, 1)
    mid = '<span class="mid"></span>' if p is not None and abs(p - 0.5) < 0.1 else ""
    meta = []
    if row.get("modal") and p is not None:
        a, b = row["modal"].split("-")
        meta.append(f"modal game {a}-{b}" if fav is row["t1"] else f"modal game {b}-{a}")
    if p is not None and p >= FLOOR_HI:
        meta.append("at the floor")
    elif p is not None and abs(p - 0.5) < 0.1:
        meta.append("nearly even")
    if row.get("p_db") is not None and p is not None:
        pdb = row["p_db"] if fav is row["t1"] else 1 - row["p_db"]
        meta.append(f"DreamBreaker {fmt_pct(pdb)}")
    if row.get("note") and not row.get("branches"):
        meta.append(row["note"])
    if row.get("start"):
        meta.append(row["start"])
    if row.get("branches"):
        # the opponent comes out of a still-pending match: one meter per possibility
        when = f" · {row['start']}" if row.get("start") else ""
        body = "".join(
            f'<div class="branch"><span><span class="vs">vs</span><span class="dog">'
            f'{esc(side_short(br, singles))}</span></span><span class="bp">{fmt_pct(br["p_known"])}</span></div>'
            f'<div class="meter"><span class="fill" style="width:{round(100 * br["p_known"], 1)}%"></span></div>'
            for br in row["branches"])
        return (f'<div class="card"><div class="lbl">{esc(label)} · opponent TBD{esc(when)}</div>'
                f'<div class="top"><span class="fav">{esc(side_short(fav, singles))}</span></div>{body}</div>')
    return (f'<div class="card"><div class="lbl">{esc(label)}</div>'
            f'<div class="top"><span class="fav">{esc(side_short(fav, singles))}</span>{pct}</div>'
            f'<div class="meter"><span class="fill" style="width:{fill}%"></span>{mid}</div>'
            f'<div class="bot"><span><span class="vs">vs</span><span class="dog">{esc(side_short(dog, singles))}</span></span>'
            f'<span class="meta">{esc(" · ".join(meta))}</span></div></div>')


ROUND_ABBR = {"Round of 64": "R64", "Round of 32": "R32", "Round of 16": "R16",
              "Quarterfinals": "QF", "Semifinals": "SF"}


def row_label(row: dict, i: int, n_in_round: dict) -> str:
    r = round_word(row["round"]) or "Match"
    if row["t1"].get("team"):
        return f"{row['t1']['team']} vs {row['t2']['team']} · {row['round']}"
    if n_in_round.get(r, 0) > 1:
        return f"{ROUND_ABBR.get(r, r)} {i}"
    return r


def bracket_slides(slate: dict, s: dict, idx_start: int) -> list[dict]:
    rows = s["rows"]
    if not rows:
        return []
    singles = s["bracket"] in ("MS", "WS")
    pages = [rows[i:i + PAGE] for i in range(0, len(rows), PAGE)]
    n_in_round: dict[str, int] = {}
    for r in rows:
        n_in_round[round_word(r["round"]) or "Match"] = n_in_round.get(round_word(r["round"]) or "Match", 0) + 1
    counters: dict[str, int] = {}
    out = []
    for pi, page in enumerate(pages):
        density = "" if len(page) <= 3 else "compact" if len(page) <= 6 else "list"
        cards = []
        for r in page:
            k = round_word(r["round"]) or "Match"
            counters[k] = counters.get(k, 0) + 1
            cards.append(card_html(r, singles, row_label(r, counters[k], n_in_round)))
        rounds = sorted({round_word(r["round"]) for r in page if r["round"]}, key=str.lower)
        kick = f"{idx_start + pi:02d} — {s['title']}"
        if rounds:
            kick += " · " + (rounds[0] if len(rounds) == 1 else " / ".join(rounds))
        if len(pages) > 1:
            kick += f" ({pi + 1}/{len(pages)})"
        bo = next((r.get("best_of") for r in page if r.get("best_of")), None)
        fmt = f"best-of-{bo}" if bo else next((r["format"] for r in page if r.get("format")), "")
        out.append({
            "kind": "bracket", "title": s["title"],
            "hdr_right": f"{s.get('venue') or short_event(s['event'])} · {pretty_date(slate['date'])}",
            "body": (f'<div class="kicker">{esc(kick)}</div><h1>{esc(s["title"])}</h1>'
                     f'<div class="cards {density}">{"".join(cards)}</div>'),
            "ftr_left": "Win probability" + (f", {fmt}" if fmt else "") + " · pre-match, no updates",
            "caption": f"{s['title']} — {short_event(s['event'])}, {pretty_date(slate['date'])}",
        })
    return out


def pretty_date(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return d.strftime("%b %-d %Y")


def short_event(title: str) -> str:
    """'PPA Tour: Veolia Pickleball National Championships' -> 'PPA National Championships'."""
    t = title.strip()
    ppa = t.lower().startswith("ppa")
    t = t.split(":", 1)[1] if ppa and ":" in t else t
    t = t.replace("Pickleball National Championships", "National Championships")
    for junk in ("Veolia", "Carvana", "powered by"):
        t = t.replace(junk, "")
    return " ".join((["PPA"] if ppa else []) + t.split())


def cover_slide(slate: dict) -> dict:
    d = dt.date.fromisoformat(slate["date"])
    ev = slate["slates"][0] if slate["slates"] else {}
    rw = deck_round(slate)
    weekday = d.strftime("%A")
    lines = ([esc(rw.upper())] if rw else []) + [f"{weekday.upper()},", '<span class="hl">FORECAST.</span>']
    what = f"Every {rw.lower()} match" if rw else "Every scheduled match"
    kick = " · ".join(x for x in (short_event(ev.get("event", "")), ev.get("venue")) if x)
    now = dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y-%m-%d")
    return {
        "kind": "cover", "title": "Forecast",
        "hdr_right": "Pro pickleball, probabilistically",
        "body": (f'<div class="kicker">{esc(kick)}</div><h1>{"<br>".join(lines)}</h1>'
                 f'<div class="sub">{what}, every division. Win probabilities from one Bayesian '
                 f'model of {n_games_str()} pro games — committed before first serve.</div>'
                 f'<div class="chips"><span class="chip">run {now} :: {slate["n_priced"]} of '
                 f'{slate["n_matches"]} priced</span><span class="chip ok">committed pre-serve ... [ok]</span></div>'),
        "ftr_left": SITE_URL, "ftr_class": "url",
        "caption": f"{short_event(ev.get('event', 'Pro pickleball'))} — {weekday} {pretty_date(slate['date'])} forecast",
    }


METHODS = [
    ("Ratings", "One Bayesian model fit to every pro game since 2024 — per-point skill "
                "for every player, allowed to drift month to month."),
    ("Teams", "A pair is the sum of its players minus a weakest-link penalty on the gap "
              "between them. Measured from results, not assumed."),
    ("Matches", "Per-point edge → exact race-to-11 math → the best-of series. Singles "
                "runs on its own model, integrated over rating uncertainty."),
    ("Honesty", "Calibrated on games the model never saw. Nothing is shown as 0% or 100%: "
                "about 1 in 100 favorites at 99% still lose, so 98.9% is the ceiling."),
    ("Receipts", "Posted before first serve, graded after. Ratings, the live board and "
                 "the full ledger are on the site below."),
]


def methods_slide(n: int) -> dict:
    steps = "".join(f'<div class="step"><span class="k">{esc(k)}</span><span class="t">{esc(t)}</span></div>'
                    for k, t in METHODS)
    return {"kind": "methods", "title": "Methods",
            "hdr_right": "Same every post",
            "body": (f'<div class="kicker">{n:02d} — Methods</div><h1>How the numbers<br>are made.</h1>'
                     f'<div class="steps">{steps}</div>'),
            "ftr_left": SITE_URL, "ftr_class": "url",
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
              "FTR_RIGHT": f"{i} / {n} →"}
    for k, v in fields.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    return tpl


def chromium_path() -> str | None:
    if os.environ.get("PW_CHROMIUM"):
        return os.environ["PW_CHROMIUM"]
    return None


def render_pngs(htmls: list[str], out_dir: Path) -> list[Path]:
    from playwright.sync_api import sync_playwright
    paths = []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=chromium_path())
        except Exception:                                   # noqa: BLE001
            # fall back to any chromium the sandbox ships (see module doc)
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
