"""Missing starters by unit, for every team-game (played and upcoming).

Units - offense: T (tackles), IOL (guards/centers); defense: EDGE, IDL, LB, CB, S.
A starter (as of a game) played heavy snaps (unit threshold) in at least half of his team's earlier
games this season (minimum 2). Missing = for played games, under 20% of snaps; for upcoming games,
Out / Doubtful on the injury report or on a reserve list (IR etc.) in the weekly roster.
Unit comes from the OverTheCap contract position when available (it separates 3-4 OLB edge rushers
from off-ball LBs), else the snap-count position.
Output columns per (game_id, team): own_T, own_IOL (the team's offense) and opp_EDGE, opp_IDL, opp_LB,
opp_CB, opp_S (the opposing defense), each capped at 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OFF_UNITS = ["T", "IOL"]
DEF_UNITS = ["EDGE", "IDL", "LB", "CB", "S"]
HEAVY = {"T": 0.85, "IOL": 0.85, "EDGE": 0.55, "IDL": 0.50, "LB": 0.70, "CB": 0.80, "S": 0.80}
CONTRACT_UNIT = {"ED": "EDGE", "IDL": "IDL", "LB": "LB", "CB": "CB", "S": "S", "LT": "T", "RT": "T", "T": "T",
                 "LG": "IOL", "RG": "IOL", "G": "IOL", "C": "IOL", "IOL": "IOL"}
SNAP_UNIT = {"T": "T", "G": "IOL", "C": "IOL", "DE": "EDGE", "OLB": "EDGE", "DT": "IDL", "NT": "IDL", "DL": "IDL",
             "LB": "LB", "ILB": "LB", "MLB": "LB", "CB": "CB", "DB": "CB", "S": "S", "FS": "S", "SS": "S"}
FEATURES = ["own_T", "own_IOL"] + [f"opp_{u}" for u in DEF_UNITS]


def units(sn: pd.DataFrame, players: pd.DataFrame, contracts: pd.DataFrame | None) -> pd.DataFrame:
    """Snap rows for OL and defenders with unit, side and gsis id."""
    s = sn.copy()
    gs = players.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    s["gsis_id"] = s["pfr_player_id"].map(gs)
    # generic "OL": use the player's most common specific OL position in any season
    ol_mode = (s[s["position"].isin(["T", "G", "C"])].groupby("pfr_player_id")["position"]
               .agg(lambda x: x.mode().iloc[0]))
    pos = np.where(s["position"].eq("OL"), s["pfr_player_id"].map(ol_mode), s["position"])
    s["unit"] = pd.Series(pos, index=s.index).map(SNAP_UNIT)
    if contracts is not None and len(contracts):
        cp = (contracts.dropna(subset=["gsis_id"]).sort_values("year_signed").drop_duplicates("gsis_id", keep="last")
              .set_index("gsis_id")["position"].map(CONTRACT_UNIT))
        cu = s["gsis_id"].map(cp)
        # contract position only re-labels within the same side (keeps OL as OL, defenders as defenders)
        same_side = cu.isin(DEF_UNITS) & s["unit"].isin(DEF_UNITS) | cu.isin(OFF_UNITS) & s["unit"].isin(OFF_UNITS)
        s.loc[same_side, "unit"] = cu[same_side]
    s = s[s["unit"].notna()].copy()
    s["pct"] = np.where(s["unit"].isin(OFF_UNITS), s["offense_pct"], s["defense_pct"]).astype(float)
    return s[["season", "week", "game_id", "team", "pfr_player_id", "gsis_id", "unit", "pct"]]


def _starters(u: pd.DataFrame) -> pd.DataFrame:
    """For each (season, team, week): starters as of that week (from earlier games). One row per starter."""
    u = u.assign(heavy=u["pct"] >= u["unit"].map(HEAVY))
    out = []
    for (s, t), d in u.groupby(["season", "team"]):
        weeks = sorted(d["week"].unique())
        nxt = weeks + [max(weeks) + 1, max(weeks) + 2]  # also the next game(s) after the last played one
        for i, w in enumerate(sorted(set(nxt))):
            prev = d[d["week"] < w]
            ng = prev["week"].nunique()
            if ng < 2:
                continue
            h = prev[prev["heavy"]].groupby(["pfr_player_id", "gsis_id", "unit"])["week"].nunique()
            st = h[h >= 0.5 * ng].reset_index()[["pfr_player_id", "gsis_id", "unit"]]
            out.append(st.assign(season=s, team=t, week=w))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["pfr_player_id", "gsis_id", "unit", "season", "team", "week"])


def _counts(miss: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """miss: rows (game_id, team, unit) of missing starters -> own/opp counts per (game_id, team)."""
    c = miss.groupby(["game_id", "team", "unit"]).size().unstack("unit").reindex(
        columns=OFF_UNITS + DEF_UNITS, fill_value=0).fillna(0)
    g = games[["game_id", "team", "opp"]].merge(c, left_on=["game_id", "team"], right_index=True, how="left").fillna(0)
    d = g[["game_id", "team"] + DEF_UNITS].rename(columns={"team": "opp", **{x: f"opp_{x}" for x in DEF_UNITS}})
    g = g.merge(d, on=["game_id", "opp"], how="left").fillna(0)
    g = g.rename(columns={"T": "own_T", "IOL": "own_IOL"})
    for f in FEATURES:
        g[f] = g[f].clip(upper=2)
    return g[["game_id", "team"] + FEATURES]


def missing(raw: dict, games: pd.DataFrame) -> pd.DataFrame:
    """Counts for every team-game in `games` (columns game_id, team, opp, season, week, played)."""
    u = units(raw["sn"], raw["players"], raw.get("contracts"))
    st = _starters(u)
    # played games: starter with < 20% snaps (or absent)
    played = games[games["played"]]
    sp = st.merge(played[["game_id", "season", "week", "team"]], on=["season", "week", "team"])
    pct = u.set_index(["game_id", "pfr_player_id"])["pct"]
    sp["pct"] = [pct.get((g, p), 0.0) for g, p in zip(sp["game_id"], sp["pfr_player_id"])]
    miss = [sp[sp["pct"] < 0.2][["game_id", "team", "unit"]]]
    # upcoming games: injury report Out/Doubtful, or reserve list on the weekly roster
    up = games[~games["played"]]
    if len(up):
        su = st.merge(up[["game_id", "season", "week", "team"]], on=["season", "week", "team"], how="inner")
        if su.empty:  # starters as of the latest computed week for each team
            last = st.sort_values("week").groupby(["season", "team"]).tail(10_000)
            mw = last.groupby(["season", "team"])["week"].max().rename("mw")
            last = last.join(mw, on=["season", "team"])
            last = last[last["week"] == last["mw"]].drop(columns=["week", "mw"])
            su = last.merge(up[["game_id", "season", "team"]], on=["season", "team"])
        out = set()
        inj = raw.get("inj")
        if inj is not None and len(inj):
            for (s, w, t), _ in up.groupby(["season", "week", "team"]):
                r = inj[(inj["season"] == s) & (inj["week"] == w) & (inj["team"] == t)
                        & inj["report_status"].isin(["Out", "Doubtful"])]
                out |= set(zip([t] * len(r), r["gsis_id"]))
        rost = raw.get("rost")
        if rost is not None and len(rost):
            last = rost.sort_values("week").drop_duplicates("gsis_id", keep="last")
            res = last[last["status"].isin(["RES", "CUT", "RET", "EXE"])]
            out |= set(zip(res["team"], res["gsis_id"]))
        mu = su[[(t, g) in out for t, g in zip(su["team"], su["gsis_id"])]]
        miss.append(mu[["game_id", "team", "unit"]])
    return _counts(pd.concat(miss, ignore_index=True), games)

