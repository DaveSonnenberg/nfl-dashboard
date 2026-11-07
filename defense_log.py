"""Defense game logs: rushing, passing and receiving-by-role allowed, one row per game.

Used by Matchups -> Defense game log and by the Props tab's 'Opp allows' click."""
import numpy as np
import pandas as pd
import nflreadpy as nfl
import polars as pl
import streamlit as st

ROLES = ["WR1", "WR2", "WR3", "WR4", "TE1", "TE2", "RB"]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_stats(season: int) -> pd.DataFrame:
    cols = ["player_display_name", "position", "team", "opponent_team", "week", "game_id", "season_type",
            "carries", "rushing_yards", "rushing_tds", "attempts", "completions", "passing_yards", "passing_tds",
            "passing_interceptions", "sacks_suffered"]
    p = nfl.load_player_stats([season]).filter(pl.col("season_type") == "REG").select(cols).to_pandas()
    return p.drop(columns="season_type")


def _weeks(d, weeks):
    return d if weeks is None else d[d["week"].between(weeks[0], weeks[1])]


def rushing(season: int, defteam: str, weeks=None) -> tuple[pd.DataFrame, dict]:
    p = _weeks(load_stats(season), weeks)
    p = p[(p["opponent_team"] == defteam) & (p["carries"] > 0)]
    rows = []
    for (wk, opp), g in p.groupby(["week", "team"]):
        nq, q = g[g["position"] != "QB"], g[g["position"] == "QB"]
        backs = nq.sort_values("carries", ascending=False)
        rows.append({
            "Wk": wk, "Opp": opp,
            "Ball carriers": "; ".join(f"{r.player_display_name} {int(r.carries)}-{int(r.rushing_yards)}"
                                       + (f", {int(r.rushing_tds)} TD" if r.rushing_tds else "")
                                       for r in backs.itertuples()),
            "Car": nq["carries"].sum(), "Yds": nq["rushing_yards"].sum(),
            "YPC": nq["rushing_yards"].sum() / max(nq["carries"].sum(), 1),
            "QB car": q["carries"].sum(), "QB yds": q["rushing_yards"].sum(),
            "Rush TD": g["rushing_tds"].sum(),
        })
    log = pd.DataFrame(rows)
    if log.empty:
        return log, {}
    log = log.sort_values("Wk")
    summ = {"games": len(log), "Car": log["Car"].mean(), "Yds": log["Yds"].mean(),
            "YPC": log["Yds"].sum() / max(log["Car"].sum(), 1), "QB yds": log["QB yds"].mean()}
    return log, summ


def passing(season: int, defteam: str, weeks=None) -> tuple[pd.DataFrame, dict]:
    p = _weeks(load_stats(season), weeks)
    p = p[(p["opponent_team"] == defteam) & (p["attempts"] > 0)]
    rows = []
    for (wk, opp), g in p.groupby(["week", "team"]):
        qbs = g.sort_values("attempts", ascending=False)
        att, yds = g["attempts"].sum(), g["passing_yards"].sum()
        rows.append({
            "Wk": wk, "Opp": opp,
            "Passers": "; ".join(f"{r.player_display_name} {int(r.completions)}/{int(r.attempts)}, {int(r.passing_yards)}"
                                 + (f", {int(r.passing_tds)} TD" if r.passing_tds else "")
                                 + (f", {int(r.passing_interceptions)} INT" if r.passing_interceptions else "")
                                 for r in qbs.itertuples()),
            "Att": att, "Cmp%": g["completions"].sum() / max(att, 1), "Yds": yds, "YPA": yds / max(att, 1),
            "TD": g["passing_tds"].sum(), "INT": g["passing_interceptions"].sum(), "Sacks": g["sacks_suffered"].sum(),
        })
    log = pd.DataFrame(rows)
    if log.empty:
        return log, {}
    log = log.sort_values("Wk")
    summ = {"games": len(log), "Att": log["Att"].mean(), "Yds": log["Yds"].mean(),
            "YPA": log["Yds"].sum() / max(log["Att"].sum(), 1), "Sacks": log["Sacks"].mean()}
    return log, summ


def receiving(history: pd.DataFrame, season: int, defteam: str, stat: str) -> pd.DataFrame:
    """By role, from the props model's player-game history (same roles as the Props table)."""
    h = history[(~history["upcoming"]) & (history["season"] == season) & (history["opp"] == defteam)].copy()
    h["drole"] = np.where(h["role"].astype(str).str.startswith("RB"), "RB", h["role"])
    h = h[h["targets"] > 0]
    out = []
    for (wk, team), g in h.groupby(["week", "team"]):
        row = {"Wk": wk, "Opp": team}
        for role in ROLES:
            r = g[g["drole"] == role].sort_values("targets", ascending=False)
            row[role] = "; ".join(f"{x.name} {int(x.receptions)}/{int(x.targets)}, {int(x.receiving_yards)}"
                                  for x in r.itertuples()) or "—"
        out.append(row)
    return pd.DataFrame(out).sort_values("Wk") if out else pd.DataFrame()


NUM_FMT = {"Car": "{:.0f}", "Yds": "{:.0f}", "YPC": "{:.2f}", "QB car": "{:.0f}", "QB yds": "{:.0f}",
           "Rush TD": "{:.0f}", "Att": "{:.0f}", "Cmp%": "{:.0%}", "YPA": "{:.2f}", "TD": "{:.0f}",
           "INT": "{:.0f}", "Sacks": "{:.0f}"}


def show(kind: str, season: int, defteam: str, weeks=None, history=None, stat="receptions", key=""):
    """Render one defense log (kind = rushing | passing | receiving)."""
    if kind == "rushing":
        log, s = rushing(season, defteam, weeks)
        if log.empty:
            st.info("No games yet.")
            return
        c = st.columns(4)
        c[0].metric("Non-QB rush yds / g", f"{s['Yds']:.1f}")
        c[1].metric("Carries / g", f"{s['Car']:.1f}")
        c[2].metric("YPC", f"{s['YPC']:.2f}")
        c[3].metric("QB rush yds / g", f"{s['QB yds']:.1f}")
        st.caption("Ball carriers read carries-yards. Car/Yds/YPC = non-QB rushing.")
    elif kind == "passing":
        log, s = passing(season, defteam, weeks)
        if log.empty:
            st.info("No games yet.")
            return
        c = st.columns(4)
        c[0].metric("Pass yds / g", f"{s['Yds']:.1f}")
        c[1].metric("Attempts / g", f"{s['Att']:.1f}")
        c[2].metric("YPA", f"{s['YPA']:.2f}")
        c[3].metric("Sacks / g", f"{s['Sacks']:.1f}")
        st.caption("Passers read completions/attempts, yards. Yards are gross passing yards (sacks not deducted).")
    else:
        log = receiving(history, season, defteam, stat)
        if log.empty:
            st.info("No games yet.")
            return
        st.caption("Each entry reads receptions/targets, yards. Roles as in the Props table (by target share entering the game).")
        st.dataframe(log, hide_index=True, width="stretch", key=f"dl_{key}")
        return
    fmt = {k: v for k, v in NUM_FMT.items() if k in log.columns}
    st.dataframe(log.style.format(fmt), hide_index=True, width="stretch", key=f"dl_{key}",
                 height=min(35 * (len(log) + 1) + 3, 640))
