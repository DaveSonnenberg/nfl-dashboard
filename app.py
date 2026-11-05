"""NFL props & matchups dashboard.

Run locally:  streamlit run app.py
Data: nflverse (refreshed nightly in season), cached here for 6 hours.
Tabs: Props (projections vs lines) · Matchups (defense vs role) · QB rushing.
Each tab has its own filters at the top; there is no sidebar.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import nfl_data as d
import game_ratings
import props_tab
import qb_tab

st.set_page_config(page_title="NFL Props", page_icon="🏈", layout="wide",
                   initial_sidebar_state="collapsed")
st.html("""<style>
.block-container {padding-top: 3.8rem; padding-bottom: 0.5rem;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {margin-bottom: -0.6rem;}
/* Props table: fill the window below the control row */
div[class*="st-key-pm_table_"], div[class*="st-key-pm_table_"] [data-testid="stDataFrame"],
div[class*="st-key-pm_table_"] [data-testid="stDataFrameResizable"] {height: calc(100vh - 258px) !important;}
</style>""")

OVERRIDES_PATH = Path(__file__).parent / "role_overrides.csv"
STAT_LABELS = {
    "ppr": "PPR pts (receiving)",
    "yards": "Receiving yards",
    "receptions": "Receptions",
    "targets": "Targets",
    "tds": "Receiving TDs",
    "air_yards": "Air yards",
}
GOOD = "Greens"  # more allowed = better matchup for the offense


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling nflverse data…")
def get_raw(season: int, include_post: bool):
    return d.load_season(season, include_post)


@st.cache_data(ttl=6 * 3600, show_spinner="Assigning roles…")
def get_player_games(season: int, include_post: bool, overrides: pd.DataFrame):
    return d.build_player_games(get_raw(season, include_post), overrides)


def current_season() -> int:
    t = date.today()
    return t.year if t.month >= 8 else t.year - 1


CS = current_season()
SEASONS = [CS, CS - 1, CS - 2, CS - 3]


# ==========================================================================
# Matchups (defense vs role)
# ==========================================================================
def render_matchups():
    f1, f2, _ = st.columns([1, 1, 7], vertical_alignment="bottom")
    season = f1.selectbox("Season", SEASONS, key="mx_season")
    include_post = f2.toggle("Playoffs", value=False, key="mx_post")

    overrides = d.load_overrides(OVERRIDES_PATH)
    season_ov = overrides[overrides["season"].isin(["", str(season)])] if len(overrides) else overrides
    raw = get_raw(season, include_post)
    pg = get_player_games(season, include_post, season_ov)

    tab_games, tab_grid, tab_log, tab_roles = st.tabs(
        ["Games", "Defense grid", "Defense game log", "Team roles (audit)"])
    with tab_games:
        game_ratings.render(season)
    if pg.empty:
        st.warning("No games yet for this season.")
        return
    max_week = int(pg["week"].max())
    with tab_grid:
        f3, f4, f5 = st.columns([2, 2, 3])
        stat = f3.selectbox("Stat", list(STAT_LABELS), format_func=STAT_LABELS.get, key="mx_stat")
        mode = f4.selectbox("Show", ["Per game", "vs league avg", "vs league avg (%)"], key="mx_mode")
        weeks = f5.slider("Weeks", 1, max_week, (1, max_week), key="mx_weeks",
                          help="Drag the left handle to see only recent form (e.g. last 4 weeks).")

    dvr = d.defense_vs_role(pg, weeks)
    lg = d.league_avg(dvr)
    col = f"{stat}_pg"
    dvr["avg"] = dvr["role"].map(lg[col])
    dvr["diff"] = dvr[col] - dvr["avg"]
    dvr["pct"] = dvr[col] / dvr["avg"] - 1
    value = {"Per game": col, "vs league avg": "diff", "vs league avg (%)": "pct"}[mode]
    nxt = d.next_opponents(raw["schedules"])
    caption = (
        f"{season} · weeks {weeks[0]}–{weeks[1]} · {STAT_LABELS[stat]} allowed per game. "
        "WR1/2/3 = each offense's receivers ranked by target share entering that game. "
        "Darker green = defense allows more. Always check **G** (games): under ~4, treat colors as noise.")

    def fmt_value(v):
        if pd.isna(v):
            return ""
        if mode == "vs league avg (%)":
            return f"{v:+.0%}"
        if mode == "vs league avg":
            return f"{v:+.1f}"
        return f"{v:.1f}"

    def style_grid(df, cols):
        sty = df.style.format({c: fmt_value for c in cols})
        if mode == "Per game":
            return sty.background_gradient(cmap=GOOD, subset=cols, axis=0)
        for c in cols:
            lim = df[c].abs().max() or 1
            sty = sty.background_gradient(cmap="RdYlGn", subset=[c], vmin=-lim, vmax=lim)
        return sty

    with tab_grid:
        st.caption(caption)
        grid = dvr.pivot(index="defteam", columns="role", values=value)[d.ROLE_ORDER]
        grid.insert(0, "G", dvr.groupby("defteam")["games"].first())
        grid = grid.join(nxt.set_index("team")[["next_opp"]].rename(columns={"next_opp": "Next"}))
        sort_by = st.selectbox("Sort by", d.ROLE_ORDER, index=1, key="mx_sort")
        grid = grid.sort_values(sort_by, ascending=False).reset_index().rename(columns={"defteam": "Defense"})
        st.dataframe(style_grid(grid, d.ROLE_ORDER), hide_index=True, width="stretch", height=1180)
        lg_row = pd.DataFrame([lg[col].reindex(d.ROLE_ORDER)]).round(1)
        lg_row.index = ["League avg / game"]
        st.caption("League average per game (for reference):")
        st.dataframe(lg_row, width="stretch")

    with tab_log:
        dteam = st.selectbox("Defense", sorted(dvr["defteam"].unique()), key="log_team")
        sub = dvr[dvr["defteam"] == dteam].set_index("role").reindex(d.ROLE_ORDER)
        cols = st.columns(len(d.ROLE_ORDER))
        for c, role in zip(cols, d.ROLE_ORDER):
            r = sub.loc[role]
            c.metric(role, f"{r[col]:.1f}", f"{r['diff']:+.1f} vs avg")
        st.caption(f"{STAT_LABELS[stat]} per game allowed · {int(sub['games'].iloc[0])} games "
                   f"(stat and weeks follow the Defense grid filters). "
                   "Each line reads Rec/Targets, yards.")
        st.dataframe(d.defense_game_log(pg, dteam, weeks), hide_index=True, width="stretch")

    with tab_roles:
        oteam = st.selectbox("Offense", sorted(pg["team"].unique()), key="role_team")
        st.dataframe(d.team_roles(pg, oteam), hide_index=True, width="stretch")
        st.caption("% = target share entering that game (early weeks blended with last season). "
                   "'prior' = based only on last season; 'snaps' = no target history; ✎ = set by your override file.")
        with st.expander("How to fix a wrong role"):
            st.markdown(
                "Edit `role_overrides.csv` in your GitHub repo (pencil icon → commit). One row per fix:\n\n"
                "```\nseason,week,team,player,role\n2026,,CIN,Tee Higgins,WR1\n2026,6,CIN,Andrei Iosivas,WR2\n```\n"
                "- Leave **week** blank to apply all season; fill it for one game.\n"
                "- **role** is WR1, WR2, WR3, or WR4 (to push someone out of the top 3).\n"
                "- **player** is the full name as shown in this table.")
        if len(season_ov):
            st.caption("Active overrides:")
            st.dataframe(season_ov, hide_index=True)


# ==========================================================================
# QB rushing
# ==========================================================================
def render_qb():
    f1, f2, f3 = st.columns([1, 1, 4])
    season = f1.selectbox("Season", SEASONS, key="qb_season")
    include_post = f2.toggle("Playoffs", value=False, key="qb_post")
    raw = get_raw(season, include_post)
    sch = raw["schedules"].to_pandas() if hasattr(raw["schedules"], "to_pandas") else raw["schedules"]
    played = sch[sch["result"].notna()]
    if played.empty:
        st.warning("No games yet for this season.")
        return
    max_week = int(played["week"].max())
    weeks = f3.slider("Weeks", 1, max_week, (1, max_week), key="qb_weeks")
    qb_tab.render(season, include_post, weeks, d.next_opponents(raw["schedules"]))


tab_props, tab_match, tab_qb = st.tabs(["Props", "Matchups", "QB rushing"])
with tab_props:
    props_tab.render()
with tab_match:
    render_matchups()
with tab_qb:
    render_qb()

st.caption("Data: nflverse (CC-BY 4.0) + FTN charting (CC-BY-SA 4.0) via nflreadpy.")
