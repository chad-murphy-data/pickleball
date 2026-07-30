"""Official MLP rosters from majorleaguepickleball.co -> data/mlp_rosters.csv.

    python scraper/mlp_rosters.py

MLP's team pages (WordPress + the fau-scores-and-stats plugin) embed the
CURRENT official roster in a `teamPageData` JSON blob — `team_players` is
the league's own list, which leads our appearance-derived rosters by a full
trade window (found 2026-07-30: Staksrud/Rane already on New Jersey,
Hunter Johnson on St. Louis, Navratil on Chicago — none visible yet from
played lineups).

CAVEAT — the site uses its OWN player uuid space (WP), not pickleball.com's,
so the join back to our identity is BY NAME against data/players.csv +
name_variants.  House rule says names are never identity: every row keeps
the site name + site uuid for audit, unmatched names are written with an
empty player_id and logged loudly, and an ambiguous name (two candidate
uuids) is treated as unmatched rather than guessed.  2026-07-30 baseline:
121/121 players matched cleanly.

Consumers: web/make_forecast.py's roster ladder prefers this file per-team
when present (falling back to appearance-derived otherwise), which flows to
make_event_forecast and tournament_state unchanged.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
import unicodedata
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "mlp_rosters.csv"
log = logging.getLogger("mlp_rosters")

BASE = "https://majorleaguepickleball.co"
UA = ("Mozilla/5.0 (compatible; pickles-bot/1.0; "
      "+https://chad-murphy-data.github.io/pickleball/methods.html)")


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("-", " ").split())


def load_name_index():
    """normalized name -> {player_id: (player_id, full_name, gender)}."""
    idx = defaultdict(dict)
    with (DATA / "players.csv").open() as f:
        for r in csv.DictReader(f):
            names = {r["full_name"]}
            try:
                names |= set(json.loads(r["name_variants"]))
            except (ValueError, KeyError):
                pass
            for n in names:
                idx[norm(n)][r["player_id"]] = (
                    r["player_id"], r["full_name"], r["gender"])
    return idx


def match(name: str, idx) -> tuple[str, str, str] | None:
    cands = idx.get(norm(name), {})
    if not cands:                       # retry dropping middle tokens
        toks = norm(name).split()
        if len(toks) > 2:
            cands = idx.get(f"{toks[0]} {toks[-1]}", {})
    if len(cands) == 1:
        return next(iter(cands.values()))
    if cands:
        log.warning("AMBIGUOUS name %r -> %s — leaving unmatched",
                    name, sorted(cands))
    return None


def team_slugs() -> list[str]:
    # /mlp-teams/ is the roster index; the homepage lists the same 20 links
    # and serves as a fallback if the index page ever moves again
    for path in ("/mlp-teams/", "/"):
        slugs = sorted(set(re.findall(r"/team/([a-z0-9-]+)/", get(BASE + path))))
        if slugs:
            return slugs
    raise ValueError("no /team/<slug>/ links found on the MLP site")


def team_roster(slug: str):
    """(team_title, [{mlp_uuid, name}]) from the page's teamPageData blob."""
    h = get(f"{BASE}/team/{slug}/")
    i = h.find("teamPageData")
    if i < 0:
        raise ValueError(f"no teamPageData on /team/{slug}/")
    obj, _ = json.JSONDecoder().raw_decode(h[h.find("=", i) + 1:].lstrip())
    td = obj["team_data"]
    players = []
    for p in td.get("team_players") or []:
        full = " ".join(x for x in (p.get("first_name"), p.get("middle_name"),
                                    p.get("last_name")) if x).strip()
        players.append({"mlp_uuid": p["uuid"].lower(), "name": full})
    return td["team_info"]["title"], players


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    idx = load_name_index()
    rows, n_unmatched = [], 0
    today = str(date.today())
    for slug in team_slugs():
        title, players = team_roster(slug)
        for p in players:
            m = match(p["name"], idx)
            if m is None:
                n_unmatched += 1
                log.warning("UNMATCHED %s | %r — row kept with empty player_id",
                            title, p["name"])
                rows.append((title, "", "", "", p["name"], p["mlp_uuid"], today))
            else:
                rows.append((title, m[0], m[1], m[2], p["name"],
                             p["mlp_uuid"], today))
        log.info("%-28s %d players", title, len(players))
        time.sleep(1.0)                                # polite
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "player_id", "full_name", "gender",
                    "site_name", "mlp_uuid", "fetched"])
        w.writerows(sorted(rows))
    log.info("wrote %s: %d rows, %d unmatched", OUT, len(rows), n_unmatched)
    if n_unmatched:
        log.warning("%d unmatched names — those players are INVISIBLE to "
                    "best-lineup projections until matched", n_unmatched)


if __name__ == "__main__":
    main()
