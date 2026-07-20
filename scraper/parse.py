"""Parse the raw/ cache into the modeling dataset.

Outputs (data/):
    games.csv          one row per GAME (the modeling unit)
    players.csv        canonical player table keyed by source UUID
    dreambreakers.csv  MLP DreamBreakers, kept separate (never in games.csv)
    dropped.csv        every excluded match/game with a reason (no silent drops)
    flags.csv          rows kept in games.csv but worth human eyeballs
    name_variants.json audit of every name string seen per player uuid

Score formats are resolved per match (see enrich_formats.py): PPA Challengers
mix "1 game to 15" rounds into otherwise best-of-3-to-11 events, so the
format label is data, not an assumption.

Run after harvest.py + enrich_formats.py:  python scraper/parse.py
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DATA = ROOT / "data"

log = logging.getLogger("parse")

GAME_ORDINALS = ["One", "Two", "Three", "Four", "Five"]

MATCH_COMPLETED_STATUS = 4   # matchStatus / match_status
NORMAL_COMPLETION = 5        # matchCompletedType; anything else → is_forfeit


def clean_name(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


class PlayerBook:
    """Accumulates identities + name variants + inferred gender per UUID."""

    def __init__(self):
        self.names = defaultdict(Counter)   # uuid -> Counter of full-name strings
        self.gender = {}                    # uuid -> "M"/"F"
        self.gender_conflicts = []

    def see(self, uuid: str, name: str, gender: str | None):
        if not uuid:
            return
        name = clean_name(name)
        if name:
            self.names[uuid][name] += 1
        if gender:
            prev = self.gender.get(uuid)
            if prev and prev != gender:
                self.gender_conflicts.append((uuid, name, prev, gender))
            else:
                self.gender[uuid] = gender

    def canonical(self, uuid: str) -> str:
        if uuid not in self.names or not self.names[uuid]:
            return ""
        return self.names[uuid].most_common(1)[0][0]


def context_from_title(title: str) -> str | None:
    t = title.lower()
    if "mixed" in t:
        return "mixed"
    if "women" in t:
        return "womens"
    if "men" in t:
        return "mens"
    return None


CONTEXT_GENDERS = {"womens": "F", "mens": "M"}


def fmt_label(is_rally: bool, game_max: int) -> str:
    return f"{'rally' if is_rally else 'sideout'}_{game_max}"


def load_enriched_formats() -> dict[str, dict]:
    """match_uuid(lower) -> format info from getResultMatchInfos."""
    out = {}
    fdir = RAW / "result_match_infos"
    if not fdir.exists():
        return out
    for f in fdir.glob("*.json"):
        body = json.loads(f.read_text())
        d = body.get("data")
        m = d[0] if isinstance(d, list) and d else d
        if not isinstance(m, dict):
            continue
        muid = (m.get("match_uuid") or f.stem).lower()
        maxes = {}
        for i, o in enumerate(["one", "two", "three", "four", "five"], start=1):
            maxes[i] = m.get(f"score_format_game_{o}_max") or 0
        out[muid] = {
            "is_rally": bool(m.get("is_rally_scoring")),
            "game_max": maxes,
            "win_by": m.get("score_format_game_one_win_by") or 2,
            "title": m.get("score_format_title") or "",
        }
    return out


def game_rows_from_match(m: dict, *, keys: dict, meta: dict, fmt: dict | None,
                         book: PlayerBook, dropped: list, flags: list) -> list[dict]:
    """Explode one match record into per-game rows. Handles both the
    camelCase (MLP matchup) and snake_case (PPA match_infos_short) payloads
    via the `keys` mapping. `fmt` carries per-game score targets when known."""
    k = keys
    match_id = (m[k["match_uuid"]] or "").lower()

    def drop(reason, game=None):
        dropped.append({"match_id": match_id, "game": game, "reason": reason,
                        **{kk: meta.get(kk, "") for kk in
                           ("event_id", "event_name", "date", "tour", "context", "stage")}})

    def flag(game_id, reason, score=""):
        flags.append({"game_id": game_id, "reason": reason, "score": score,
                      **{kk: meta.get(kk, "") for kk in
                         ("event_id", "event_name", "date", "tour", "context", "stage")}})

    status = m.get(k["match_status"])
    if status != MATCH_COMPLETED_STATUS:
        drop(f"match not completed (status={status})")
        return []

    ctype = m.get(k["match_completed_type"])
    is_forfeit = (ctype != NORMAL_COMPLETION) or bool(m.get(k.get("did_cancel", ""), False))

    players = {}
    for side, pos in (("t1", "p1"), ("t1", "p2"), ("t2", "p1"), ("t2", "p2")):
        # normalize case: the API mixes upper/lowercase UUIDs, and a player
        # split across two spellings would silently break the partner graph
        uuid = (m.get(k[f"{side}_{pos}_uuid"]) or "").lower()
        name = clean_name(m.get(k[f"{side}_{pos}_name"]) or "")
        players[f"{side}_{pos}"] = (uuid, name)

    context = meta["context"]
    if not context:
        drop("cannot determine context")
        return []

    gender = CONTEXT_GENDERS.get(context)
    for slot, (uuid, name) in players.items():
        book.see(uuid, name, gender)

    missing = [slot for slot, (uuid, name) in players.items() if not uuid]
    if missing:
        has_scores = False
        for o in ("One", "one", "Two", "two", "Three", "three", "Four", "four", "Five", "five"):
            try:
                if (m.get(k["t1_game"].format(o)) or 0) or (m.get(k["t2_game"].format(o)) or 0):
                    has_scores = True
                    break
            except (KeyError, IndexError):
                continue
        if not has_scores:
            drop("bracket bye/shell (empty side, no scores)")
        else:
            drop(f"missing player uuid(s) WITH real scores — data loss: {','.join(missing)}")
        return []

    rows = []
    for i, ordinal in enumerate(GAME_ORDINALS, start=1):
        # camelCase payloads use "One", snake_case use "one"
        o = ordinal.lower() if "_" in k["t1_game"] else ordinal
        s1 = m.get(k["t1_game"].format(o)) or 0
        s2 = m.get(k["t2_game"].format(o)) or 0
        if s1 == 0 and s2 == 0:
            continue  # game not played (bo3 that ended 2-0, etc.)
        margin = s1 - s2
        winner, win_margin = max(s1, s2), abs(margin)
        game_id = f"{match_id}:g{i}"

        gmax = (fmt or {}).get("game_max", {}).get(i) or 0
        win_by = (fmt or {}).get("win_by", 2)
        if not is_forfeit:
            if gmax:
                if winner < gmax:
                    drop(f"completed game below target {s1}-{s2} (to {gmax})", game=i)
                    continue
                if winner == gmax and win_margin < win_by:
                    flag(game_id, f"reached {gmax} but margin {win_margin} < win-by {win_by}",
                         f"{s1}-{s2}")
                if winner > gmax and win_margin != win_by:
                    flag(game_id, f"deuce game should end exactly {win_by} apart "
                                  f"(to {gmax}, got {s1}-{s2})", f"{s1}-{s2}")
            else:
                # format unknown — sanity floor for side-out pro play
                if winner < 11 or win_margin < 2:
                    drop(f"implausible completed score {s1}-{s2} (format unknown)", game=i)
                    continue
                if winner > 15:
                    flag(game_id, f"winning score {winner} with unresolved format "
                                  "(check for to-15/to-21 leak)", f"{s1}-{s2}")

        rows.append({
            "game_id": game_id,
            "match_id": match_id,
            "event_id": meta["event_id"],
            "event_name": meta["event_name"],
            "date": meta["date"],
            "tour": meta["tour"],
            "scoring_format": meta["scoring_format"],
            "best_of": meta["best_of"],
            "context": context,
            "stage": meta["stage"],
            "t1_p1": players["t1_p1"][0], "t1_p2": players["t1_p2"][0],
            "t2_p1": players["t2_p1"][0], "t2_p2": players["t2_p2"][0],
            "t1_score": s1, "t2_score": s2,
            "margin": margin,
            "game_number": i,
            "is_dreambreaker": False,
            "is_forfeit": is_forfeit,
        })
    if not rows:
        drop("no played games" + (" (forfeit/walkover)" if is_forfeit else ""))
    return rows


MLP_KEYS = dict(
    match_uuid="matchUuid", match_status="matchStatus",
    match_completed_type="matchCompletedType", did_cancel="didCancelMatch",
    t1_p1_uuid="teamOnePlayerOneUuid", t1_p2_uuid="teamOnePlayerTwoUuid",
    t2_p1_uuid="teamTwoPlayerOneUuid", t2_p2_uuid="teamTwoPlayerTwoUuid",
    t1_p1_name="teamOnePlayerOneName", t1_p2_name="teamOnePlayerTwoName",
    t2_p1_name="teamTwoPlayerOneName", t2_p2_name="teamTwoPlayerTwoName",
    t1_game="teamOneGame{}Score", t2_game="teamTwoGame{}Score",
)

PPA_KEYS = dict(
    match_uuid="match_uuid", match_status="match_status",
    match_completed_type="match_completed_type", did_cancel="did_cancel_match",
    t1_p1_uuid="team_one_player_one_uuid", t1_p2_uuid="team_one_player_two_uuid",
    t2_p1_uuid="team_two_player_one_uuid", t2_p2_uuid="team_two_player_two_uuid",
    t1_p1_name="team_one_player_one_name", t1_p2_name="team_one_player_two_name",
    t2_p1_name="team_two_player_one_name", t2_p2_name="team_two_player_two_name",
    t1_game="team_one_game_{}_score", t2_game="team_two_game_{}_score",
)


def mlp_inline_fmt(m: dict) -> dict:
    maxes = {i: m.get(f"scoreFormatGame{o}Max") or 0
             for i, o in enumerate(["1", "2", "3", "4", "5"], start=1)}
    return {
        "is_rally": bool(m.get("isRallyScoring")),
        "game_max": maxes,
        "win_by": m.get("scoreFormatGame1WinBy") or 2,
        "title": m.get("scoreFormatTitle") or "",
    }


def parse_mlp(book: PlayerBook, dropped: list, flags: list):
    games, dreambreakers = [], []
    mdir = RAW / "matchup_data"
    if not mdir.exists():
        return games, dreambreakers
    fixture_game_counts = Counter()
    for f in sorted(mdir.glob("*.json")):
        mu = json.loads(f.read_text()).get("data") or {}
        if not mu:
            continue
        if mu.get("matchupStatus") == "BYE_MATCHUP_STATUS":
            dropped.append({"match_id": (mu.get("uuid") or "").lower(), "game": None,
                            "reason": "bye matchup (no opponent)",
                            "event_id": (mu.get("matchupGroupUuid") or "").lower(),
                            "event_name": mu.get("matchupGroupTitle") or "",
                            "date": (mu.get("plannedStartDate") or "")[:10],
                            "tour": "MLP", "context": "", "stage": ""})
            continue
        # matchupGroupTitle is the city stop ("MLP Dallas"); teamLeagueTitle can
        # be the generic league name
        event_name = (mu.get("matchupGroupTitle") or mu.get("teamLeagueTitle") or "").strip()
        stage_bits = [mu.get("roundText") or "", mu.get("inBracketType") or ""]
        stage = " ".join(b for b in stage_bits if b).strip() or "RR"
        matches = mu.get("matches") or []
        # DreamBreaker rosters: the DB is 1v1, so the full MLP squads are not
        # on the DB match — they are the union of players across the matchup's
        # doubles matches (teamOne/teamTwo are consistent within a matchup).
        # Emitted so the DreamBreaker model can be fit reproducibly from
        # committed data (mean roster SINGLES value per side) instead of a
        # hardcoded coefficient.
        roster1, roster2 = set(), set()
        for m in matches:
            if bool(m.get("isTieBreaker")) or \
                    (m.get("matchAbbreviation") or "").upper() in ("DB", "TB"):
                continue
            for k in ("teamOnePlayerOneUuid", "teamOnePlayerTwoUuid"):
                u = (m.get(k) or "").lower()
                if u:
                    roster1.add(u)
            for k in ("teamTwoPlayerOneUuid", "teamTwoPlayerTwoUuid"):
                u = (m.get(k) or "").lower()
                if u:
                    roster2.add(u)
        for m in matches:
            abbrev = (m.get("matchAbbreviation") or "").upper()
            is_db = bool(m.get("isTieBreaker")) or abbrev in ("DB", "TB")
            date_str = (m.get("matchStart") or m.get("matchPlannedStart") or "")[:10]
            # WD/MD/MXD1/MXD2 in the regular season; Gold/Bronze finals use
            # WDG/MDG/WDB/MDB ("Womens Doubles Gold", …)
            if abbrev.startswith("MXD"):
                context = "mixed"
            elif abbrev.startswith("WD"):
                context = "womens"
            elif abbrev.startswith("MD"):
                context = "mens"
            else:
                context = None
            if context is None and not is_db:
                context = context_from_title(m.get("roundText") or "")
            fmt = mlp_inline_fmt(m)
            meta = {
                "event_id": (mu.get("matchupGroupUuid") or "").lower(),
                "event_name": event_name or (m.get("moduleSubTitle") or "").split(" - ")[0],
                "date": date_str,
                "tour": "MLP",
                "scoring_format": fmt_label(fmt["is_rally"], fmt["game_max"].get(1) or 11),
                "best_of": m.get("scoreFormatGameBestOutOf") or 1,
                "context": context,
                "stage": stage,
            }
            if is_db:
                s1 = m.get("teamOneGameOneScore") or 0
                s2 = m.get("teamTwoGameOneScore") or 0
                if m.get("matchStatus") != MATCH_COMPLETED_STATUS or (s1 == 0 and s2 == 0):
                    continue  # unplayed DB slot (fixture wasn't 2-2) — not data
                dreambreakers.append({
                    "match_id": (m.get("matchUuid") or "").lower(),
                    "matchup_id": (mu.get("uuid") or "").lower(),
                    "event_name": meta["event_name"],
                    "date": date_str,
                    "t1_score": s1, "t2_score": s2,
                    "scoring_format": meta["scoring_format"],
                    "stage": stage,
                    "roster1": "|".join(sorted(roster1)),
                    "roster2": "|".join(sorted(roster2)),
                })
                continue
            rows = game_rows_from_match(m, keys=MLP_KEYS, meta=meta, fmt=fmt,
                                        book=book, dropped=dropped, flags=flags)
            swapped = [k for k in m if k.startswith("swappedOut") and m.get(k)]
            if swapped and rows:
                flags.append({"game_id": rows[0]["game_id"], "score": "",
                              "reason": "mid-match player swap recorded — listed lineup "
                                        "may not have played the whole game",
                              "event_id": meta["event_id"], "event_name": meta["event_name"],
                              "date": meta["date"], "tour": "MLP",
                              "context": meta["context"] or "", "stage": meta["stage"]})
            games.extend(rows)
            if rows:
                fixture_game_counts[mu.get("uuid")] += len(rows)
    # invariant: completed MLP fixtures have exactly 4 non-DB games
    for muid, n in fixture_game_counts.items():
        if n != 4:
            flags.append({"game_id": "", "score": "",
                          "reason": f"MLP fixture {muid} has {n} non-DB games (expect 4)",
                          "event_id": "", "event_name": "", "date": "", "tour": "MLP",
                          "context": "", "stage": ""})
    return games, dreambreakers


def parse_ppa(book: PlayerBook, dropped: list, flags: list, formats: dict):
    games = []
    mdir = RAW / "match_infos_short"
    if not mdir.exists():
        return games
    titles = {}
    tdir = RAW / "tournaments_on_date"
    for f in sorted(tdir.glob("*.json")) if tdir.exists() else []:
        for t in (json.loads(f.read_text()).get("data") or []):
            titles[t["TournamentID"]] = t["Title"]
    seen_match_dates = {}
    unresolved_bo1 = 0
    for tdir_ in sorted(mdir.iterdir()):
        tid = tdir_.name
        for f in sorted(tdir_.glob("*.json")):
            day = f.stem
            for m in json.loads(f.read_text()).get("data") or []:
                muid = (m.get("match_uuid") or "").lower()
                if muid in seen_match_dates:
                    continue  # same match can echo across adjacent day files
                seen_match_dates[muid] = day
                event_title = m.get("event_title") or ""
                if "doubles" not in event_title.lower():
                    dropped.append({"match_id": muid, "game": None,
                                    "reason": f"non-doubles event leaked: {event_title}",
                                    "event_id": tid, "event_name": titles.get(tid, tid),
                                    "date": day, "tour": "PPA", "context": "", "stage": ""})
                    continue
                best_of = m.get("score_format_game_best_out_of") or 3
                fmt = formats.get(muid)
                if fmt:
                    label = fmt_label(fmt["is_rally"], fmt["game_max"].get(1) or 11)
                elif best_of in (3, 5):
                    # unenriched standard match: verified-by-sample default
                    label = "sideout_11"
                    fmt = {"is_rally": False, "game_max": {i: 11 for i in range(1, 6)},
                           "win_by": 2, "title": "assumed"}
                else:
                    label = "unknown"
                    unresolved_bo1 += 1
                # keep the draw type: "Semi-Finals" of a Qualifier must not
                # look like a main-draw semi
                draw = re.sub(r".*Doubles Pro\s*", "", event_title).strip()
                stage = (m.get("round_text") or "").strip()
                if draw and draw != "Main Draw":
                    stage = f"{draw} — {stage}" if stage else draw
                meta = {
                    "event_id": (m.get("tournament_uuid") or tid).lower(),
                    "event_name": titles.get(tid, m.get("tournament_title") or tid),
                    "date": (m.get("match_start") or m.get("match_planned_start") or day)[:10],
                    "tour": "PPA",
                    "scoring_format": label,
                    "best_of": best_of,
                    "context": context_from_title(event_title),
                    "stage": stage,
                }
                rows = game_rows_from_match(m, keys=PPA_KEYS, meta=meta, fmt=fmt,
                                            book=book, dropped=dropped, flags=flags)
                games.extend(rows)
    if unresolved_bo1:
        log.warning("PPA: %d best-of-1 matches lack format enrichment "
                    "(run enrich_formats.py)", unresolved_bo1)
    return games


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DATA.mkdir(exist_ok=True)
    book = PlayerBook()
    dropped, flags = [], []
    formats = load_enriched_formats()
    log.info("enriched formats loaded: %d", len(formats))

    mlp_games, dreambreakers = parse_mlp(book, dropped, flags)
    ppa_games = parse_ppa(book, dropped, flags, formats)
    games = mlp_games + ppa_games

    # context validation against inferred genders (mixed = 1M + 1F per side)
    for row in games:
        if row["context"] != "mixed":
            continue
        gs = [book.gender.get(row[c]) for c in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        for pair in ((gs[0], gs[1]), (gs[2], gs[3])):
            if None not in pair and pair[0] == pair[1]:
                flags.append({"game_id": row["game_id"],
                              "reason": f"mixed game but same-gender pair inferred {pair}",
                              "score": f"{row['t1_score']}-{row['t2_score']}",
                              "event_id": row["event_id"], "event_name": row["event_name"],
                              "date": row["date"], "tour": row["tour"],
                              "context": row["context"], "stage": row["stage"]})
                break

    games.sort(key=lambda r: (r["date"], r["event_name"], r["match_id"], r["game_number"]))

    cols = ["game_id", "match_id", "event_id", "event_name", "date", "tour",
            "scoring_format", "best_of", "context", "stage",
            "t1_p1", "t1_p2", "t2_p1", "t2_p2",
            "t1_score", "t2_score", "margin", "game_number",
            "is_dreambreaker", "is_forfeit"]
    with (DATA / "games.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(games)

    variants_audit = {}
    with (DATA / "players.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "full_name", "gender", "name_variants"])
        for uuid in sorted(book.names):
            variants = book.names[uuid]
            rare = [n for n, c in variants.items() if c < 3]
            variants_audit[uuid] = {
                "canonical": book.canonical(uuid),
                "variants": dict(variants.most_common()),
                "needs_review": bool(len(variants) > 1 or rare),
            }
            w.writerow([uuid, book.canonical(uuid), book.gender.get(uuid, ""),
                        json.dumps(sorted(variants))])
    (DATA / "name_variants.json").write_text(json.dumps(variants_audit, indent=1))

    with (DATA / "dreambreakers.csv").open("w", newline="") as fh:
        if dreambreakers:
            w = csv.DictWriter(fh, fieldnames=list(dreambreakers[0].keys()))
            w.writeheader(); w.writerows(dreambreakers)

    for name, rows in (("dropped.csv", dropped), ("flags.csv", flags)):
        with (DATA / name).open("w", newline="") as fh:
            if rows:
                allkeys = sorted({k for r in rows for k in r})
                w = csv.DictWriter(fh, fieldnames=allkeys)
                w.writeheader(); w.writerows(rows)

    n_multi = sum(1 for v in variants_audit.values() if len(v["variants"]) > 1)
    log.info("games: %d (MLP %d, PPA %d) | dreambreakers: %d | players: %d "
             "(%d with >1 name variant) | dropped: %d | flags: %d | gender conflicts: %d",
             len(games), len(mlp_games), len(ppa_games), len(dreambreakers),
             len(book.names), n_multi, len(dropped), len(flags), len(book.gender_conflicts))


if __name__ == "__main__":
    main()
