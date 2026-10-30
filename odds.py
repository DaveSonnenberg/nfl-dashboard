"""The Odds API (v4) client for NFL player props + helpers.

Credit use: each event-odds call costs (markets returned) x (regions). Five
markets, one region -> ~5 credits per game, ~70-80 per full-week pull. Event
lists are free. Pulls only happen when you press the button; results are saved
to disk and reused for 12 hours, so app restarts don't burn credits.
"""
from __future__ import annotations

import difflib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
CACHE = Path(__file__).parent / ".odds_cache.json"
MAX_AGE_H = 12
MARKET_KEYS = {
    "receptions": "player_receptions",
    "rec_yards": "player_reception_yds",
    "carries": "player_rush_attempts",
    "rush_yards": "player_rush_yds",
    "pass_yards": "player_pass_yds",
}
KEY_TO_MARKET = {v: k for k, v in MARKET_KEYS.items()}
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------
def _get(url: str, params: dict) -> tuple[object, dict]:
    r = requests.get(url, params=params, timeout=20)
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key (401). Check ODDS_API_KEY in the app's secrets.")
    if r.status_code == 422:
        raise RuntimeError(f"The Odds API refused the request (422): {r.text[:200]}")
    if r.status_code == 429:
        raise RuntimeError("Out of Odds API credits or rate-limited (429).")
    r.raise_for_status()
    usage = {k: r.headers.get(k) for k in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
    return r.json(), usage


def fetch(api_key: str, team_games: set[tuple[str, str]], regions: str = "us") -> dict:
    """Pull props for the given (home_abbr, away_abbr) games. Returns raw payload + usage."""
    events, usage = _get(f"{BASE}/events", {"apiKey": api_key})
    wanted = [e for e in events
              if (TEAM_ABBR.get(e["home_team"]), TEAM_ABBR.get(e["away_team"])) in team_games]
    out = []
    for e in wanted:
        data, usage = _get(f"{BASE}/events/{e['id']}/odds",
                           {"apiKey": api_key, "regions": regions, "markets": ",".join(MARKET_KEYS.values()),
                            "oddsFormat": "american"})
        out.append(data)
    payload = {"pulled_at": time.time(), "events": out, "usage": usage, "n_events": len(wanted)}
    try:
        CACHE.write_text(json.dumps(payload))
    except OSError:
        pass
    return payload


def cached() -> dict | None:
    try:
        p = json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return None
    if time.time() - p.get("pulled_at", 0) > MAX_AGE_H * 3600:
        return None
    return p


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse(payload: dict) -> pd.DataFrame:
    """Long table: one row per event x book x market x player x line with over/under prices."""
    rows = []
    for ev in payload.get("events", []):
        home, away = TEAM_ABBR.get(ev.get("home_team")), TEAM_ABBR.get(ev.get("away_team"))
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                market = KEY_TO_MARKET.get(mk.get("key"))
                if not market:
                    continue
                for oc in mk.get("outcomes", []):
                    side = str(oc.get("name", "")).lower()
                    if side not in ("over", "under") or oc.get("point") is None:
                        continue
                    rows.append({"home": home, "away": away, "book": bk.get("title", bk.get("key")),
                                 "market": market, "player_raw": oc.get("description", ""),
                                 "line": float(oc["point"]), "side": side, "price": oc.get("price")})
    if not rows:
        return pd.DataFrame(columns=["home", "away", "book", "market", "player_raw", "line", "over", "under"])
    d = pd.DataFrame(rows)
    w = d.pivot_table(index=["home", "away", "book", "market", "player_raw", "line"], columns="side",
                      values="price", aggfunc="first").reset_index()
    for c in ("over", "under"):
        if c not in w:
            w[c] = np.nan
    return w


def consensus(w: pd.DataFrame, book: str | None = None) -> pd.DataFrame:
    """One line per player-market. Expects `player_id` (run match_players first).
    A chosen book: its main line (priced closest to even). Consensus: the most
    common line across books, with prices/probabilities combined at that line."""
    w = w.dropna(subset=["player_id"]).copy()
    if w.empty:
        return w.assign(n_books=0, book_p_over=np.nan, book_p_under=np.nan)
    w["dec_o"], w["dec_u"] = american_to_decimal(w["over"]), american_to_decimal(w["under"])
    po, pu = implied_no_vig(w["over"], w["under"])
    w["book_p_over"], w["book_p_under"] = po, pu
    key = ["home", "away", "market", "player_id"]
    if book and book != "Consensus":
        b = w[w["book"] == book].copy()
        b["dist"] = (b["book_p_over"] - 0.5).abs()
        b = b.sort_values("dist").drop_duplicates(key).drop(columns="dist")
        return b.assign(n_books=1)
    mode_line = (w.groupby(key + ["line"]).size().rename("n").reset_index()
                 .sort_values("n", ascending=False).drop_duplicates(key))
    m = w.merge(mode_line[key + ["line"]], on=key + ["line"])
    agg = m.groupby(key + ["line"]).agg(dec_o=("dec_o", "median"), dec_u=("dec_u", "median"),
                                         book_p_over=("book_p_over", "median"), book_p_under=("book_p_under", "median"),
                                         n_books=("book", "nunique"), player_raw=("player_raw", "first")).reset_index()
    agg["over"], agg["under"] = decimal_to_american(agg["dec_o"]), decimal_to_american(agg["dec_u"])
    return agg


# --------------------------------------------------------------------------
# odds math
# --------------------------------------------------------------------------
def american_to_decimal(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def decimal_to_american(d):
    d = np.asarray(d, float)
    return np.where(d >= 2, (d - 1) * 100, -100 / (d - 1)).round(0)


def implied_no_vig(over, under):
    """Book's probabilities with the margin removed (proportional)."""
    ro, ru = 1 / american_to_decimal(over), 1 / american_to_decimal(under)
    s = ro + ru
    return ro / s, ru / s


# --------------------------------------------------------------------------
# name matching
# --------------------------------------------------------------------------
def match_players(lines: pd.DataFrame, proj: pd.DataFrame, norm) -> pd.DataFrame:
    """Attach player_id to each odds row. Exact normalized-name match within the
    game's two teams first, then a close fuzzy match (>=0.88)."""
    if lines.empty:
        return lines.assign(player_id=pd.Series(dtype=str))
    proj = proj.assign(key=proj["name"].map(norm))
    out = []
    for (h, a), g in lines.groupby(["home", "away"]):
        cand = proj[proj["team"].isin([h, a])]
        lookup = dict(zip(cand["key"], cand["player_id"]))
        keys = list(lookup)
        for r in g.itertuples(index=False):
            k = norm(r.player_raw)
            pid = lookup.get(k)
            if pid is None and keys:
                close = difflib.get_close_matches(k, keys, n=1, cutoff=0.88)
                pid = lookup[close[0]] if close else None
            out.append(pid)
    lines = pd.concat([g for _, g in lines.groupby(["home", "away"])])
    return lines.assign(player_id=out)
