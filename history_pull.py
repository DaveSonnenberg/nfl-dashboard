"""One-time pull of historical player-prop lines from The Odds API (paid plans only).

One snapshot per game, PRE_KICK minutes before kickoff. Cost: 10 credits x markets x 1 region
per game (30 for 3 markets) + ~1 credit per kickoff slot to look up event ids.
Results are saved per game (so a stopped run resumes without paying twice) and combined into
history/props_lines.csv.gz.

Run from a terminal:
    python history_pull.py --key YOUR_KEY --seasons 2025 2026            # estimate only
    python history_pull.py --key YOUR_KEY --seasons 2025 --weeks 1 --go   # one-week test
    python history_pull.py --key YOUR_KEY --seasons 2025 2026 --go        # full pull
Or from the app: open the app URL with ?admin=pull added.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

import odds as od

BASE = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl"
MARKETS = ["player_reception_yds", "player_rush_yds", "player_pass_yds"]
PRE_KICK = 15            # minutes before kickoff
CREDIT_FLOOR = 3000      # stop before remaining credits drop below this
OUT = Path(__file__).parent / "history"
RAW = OUT / "raw"
CSV = OUT / "props_lines.csv.gz"


class Stop(Exception):
    pass


def games(seasons: list[int], weeks: list[int] | None = None) -> pd.DataFrame:
    """Completed games (incl. playoffs) with kickoff and snapshot times in UTC."""
    import nflreadpy as nfl
    s = nfl.load_schedules(seasons).to_pandas()
    s = s[s["result"].notna()].copy()
    if weeks:
        s = s[s["week"].isin(weeks)]
    kick = pd.to_datetime(s["gameday"] + " " + s["gametime"]).dt.tz_localize("America/New_York")
    s["kickoff"] = kick.dt.tz_convert("UTC")
    s["snap"] = s["kickoff"] - pd.Timedelta(minutes=PRE_KICK)
    return s[["season", "week", "game_type", "game_id", "home_team", "away_team", "kickoff", "snap"]] \
        .sort_values("kickoff").reset_index(drop=True)


_SAVED: set[str] | None = None


def _saved_ids() -> set[str]:
    """Games already in the committed CSV (so the repo copy counts as 'done')."""
    global _SAVED
    if _SAVED is None:
        try:
            _SAVED = set(pd.read_csv(CSV, usecols=["game_id"])["game_id"])
        except (OSError, ValueError):
            _SAVED = set()
    return _SAVED


def _done(gid: str, season: int) -> bool:
    return (RAW / str(season) / f"{gid}.json").exists() or gid in _saved_ids()


def estimate(G: pd.DataFrame) -> dict:
    todo = G[[not _done(g, s) for g, s in zip(G["game_id"], G["season"])]]
    slots = todo["snap"].nunique()
    return {"games": len(G), "already_saved": len(G) - len(todo), "to_pull": len(todo),
            "slots": slots, "credits": len(todo) * 10 * len(MARKETS) + slots}


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(url: str, params: dict, state: dict):
    r = requests.get(url, params=params, timeout=30)
    if r.status_code >= 400:
        try:
            b = r.json()
            msg = f"{b.get('error_code', '')}: {b.get('message', '')}"
        except ValueError:
            msg = r.text[:200]
        raise Stop(f"Odds API error {r.status_code} {msg}")
    rem = r.headers.get("x-requests-remaining")
    if rem is not None:
        state["remaining"] = float(rem)
    state["used"] += float(r.headers.get("x-requests-last") or 0)
    if state.get("remaining") is not None and state["remaining"] < CREDIT_FLOOR:
        state["stop_reason"] = f"remaining credits {state['remaining']:.0f} < floor {CREDIT_FLOOR}"
    return r.json()


def pull(key: str, G: pd.DataFrame, progress=print) -> dict:
    """Pull every not-yet-saved game in G. Returns run stats. Safe to stop and rerun."""
    state = {"used": 0.0, "remaining": None, "saved": 0, "empty": 0, "unmatched": [], "stop_reason": None}
    todo = G[[not _done(g, s) for g, s in zip(G["game_id"], G["season"])]]
    n = len(todo)
    for i, (snap, slot) in enumerate(todo.groupby("snap", sort=True)):
        if state["stop_reason"]:
            break
        try:
            ev = _get(f"{BASE}/events", {"apiKey": key, "date": _iso(snap),
                                         "commenceTimeFrom": _iso(snap - pd.Timedelta(hours=1)),
                                         "commenceTimeTo": _iso(snap + pd.Timedelta(hours=4))}, state)
        except Stop as e:
            state["stop_reason"] = str(e)
            break
        by_teams = {(od.TEAM_ABBR.get(e["home_team"]), od.TEAM_ABBR.get(e["away_team"])): e
                    for e in ev.get("data", [])}
        for _, g in slot.iterrows():
            if state["stop_reason"]:
                break
            e = by_teams.get((g["home_team"], g["away_team"]))
            if e is None:  # neutral-site listings sometimes flip home/away
                e = by_teams.get((g["away_team"], g["home_team"]))
            if e is None:
                state["unmatched"].append(g["game_id"])
                continue
            try:
                data = _get(f"{BASE}/events/{e['id']}/odds",
                            {"apiKey": key, "date": _iso(snap), "regions": "us",
                             "markets": ",".join(MARKETS), "oddsFormat": "american"}, state)
            except Stop as err:
                state["stop_reason"] = str(err)
                break
            body = data.get("data") or {}
            if not body.get("bookmakers"):
                state["empty"] += 1
            path = RAW / str(g["season"]) / f"{g['game_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"game_id": g["game_id"], "season": int(g["season"]),
                                        "week": int(g["week"]), "game_type": g["game_type"],
                                        "kickoff": _iso(g["kickoff"]), "snapshot": data.get("timestamp"),
                                        "event": body}))
            state["saved"] += 1
            progress(state["saved"], n, g["game_id"], state)
            time.sleep(0.2)
    return state


def combine() -> pd.DataFrame:
    """All saved games -> one long table (book x market x player x line, over/under prices)."""
    frames = []
    for f in sorted(RAW.glob("*/*.json")):
        j = json.loads(f.read_text())
        w = od.parse({"events": [j["event"]]}) if j["event"] else pd.DataFrame()
        if len(w):
            frames.append(w.assign(season=j["season"], week=j["week"], game_type=j["game_type"],
                                   game_id=j["game_id"], kickoff=j["kickoff"], snapshot=j["snapshot"]))
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    try:
        old = pd.read_csv(CSV)
        if len(new):
            old = old[~old["game_id"].isin(new["game_id"])]
    except (OSError, ValueError):
        old = pd.DataFrame()
    d = pd.concat([old, new], ignore_index=True)
    if d.empty:
        return d
    cols = ["season", "week", "game_type", "game_id", "kickoff", "snapshot", "home", "away",
            "book", "market", "player_raw", "line", "over", "under"]
    d = d[cols]
    OUT.mkdir(exist_ok=True)
    d.sort_values(["kickoff", "game_id", "market", "player_raw", "book"]).to_csv(CSV, index=False, compression="gzip")
    global _SAVED
    _SAVED = None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--weeks", type=int, nargs="*")
    ap.add_argument("--go", action="store_true", help="actually pull (otherwise estimate only)")
    a = ap.parse_args()
    G = games(a.seasons, a.weeks)
    est = estimate(G)
    print(f"Games: {est['games']} | already saved: {est['already_saved']} | to pull: {est['to_pull']} "
          f"| est. credits: {est['credits']:,}")
    if not a.go:
        print("Estimate only. Add --go to pull.")
        return
    st = pull(a.key, G, lambda k, n, gid, s: print(
        f"{k}/{n} {gid}  credits used {s['used']:.0f}  remaining {s['remaining']}"))
    print(st)
    d = combine()
    print(f"Saved {len(d):,} rows to {CSV}")


if __name__ == "__main__":
    main()
