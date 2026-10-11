"""NFL defense-vs-role dashboard (v1.0).

Run locally:  streamlit run app.py
Data: nflverse (refreshed nightly in season), cached here for 6 hours.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import nfl_data as d

st.set_page_config(page_title="Defense vs Role", page_icon="🏈", layout="wide")

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


# --------------------------------------------------------------------------
# Cached data
# --------------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner="Pulling nflverse data…")
def get_raw(season: int, include_post: bool):
    return d.load_season(season, include_post)


@st.cache_data(ttl=6 * 3600, show_spinner="Assigning roles…")
def get_player_games(season: int, include_post: bool, overrides: pd.DataFrame):
    return d.build_player_games(get_raw(season, include_post), overrides)


def current_season() -> int:
    t = date.today()
    return t.year if t.month >= 8 else t.year - 1


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    cs = current_season()
    season = st.selectbox("Season", [cs, cs - 1, cs - 2, cs - 3])
    include_post = st.checkbox("Include playoffs", value=False)
    stat = st.selectbox("Stat", list(STAT_LABELS), format_func=STAT_LABELS.get)
    mode = st.radio("Show", ["Per game", "vs league avg", "vs league avg (%)"])

overrides = d.load_overrides(OVERRIDES_PATH)
season_ov = overrides[overrides["season"].isin(["", str(season)])] if len(overrides) else overrides

raw = get_raw(season, include_post)
pg = get_player_games(season, include_post, season_ov)
if pg.empty:
    st.warning("No games yet for this season.")
    st.stop()

max_week = int(pg["week"].max())
with st.sidebar:
    weeks = st.slider("Weeks", 1, max_week, (1, max_week))
    st.caption("Tip: drag the left handle to see only recent form "
               "(e.g. last 4 weeks).")

dvr = d.defense_vs_role(pg, weeks)
lg = d.league_avg(dvr)
col = f"{stat}_pg"
dvr["avg"] = dvr["role"].map(lg[col])
dvr["diff"] = dvr[col] - dvr["avg"]
dvr["pct"] = dvr[col] / dvr["avg"] - 1
value = {"Per game": col, "vs league avg": "diff", "vs league avg (%)": "pct"}[mode]

nxt = d.next_opponents(raw["schedules"])
next_week = dict(zip(nxt["team"], nxt["next_week"]))

st.title("Defense vs Role")
st.caption(
    f"{season} · weeks {weeks[0]}–{weeks[1]} · {STAT_LABELS[stat]} allowed per game. "
    "WR1/2/3 = each offense's receivers ranked by target share entering that game. "
    "Darker green = defense allows more = better matchup. "
    "Always check **G** (games): under ~4, treat colors as noise."
)


def fmt_value(v):
    if pd.isna(v):
        return ""
    if mode == "vs league avg (%)":
        return f"{v:+.0%}"
    if mode == "vs league avg":
        return f"{v:+.1f}"
    return f"{v:.1f}"


def style_grid(df: pd.DataFrame, cols: list[str]):
    sty = df.style.format({c: fmt_value for c in cols})
    if mode == "Per game":
        sty = sty.background_gradient(cmap=GOOD, subset=cols, axis=0)
    else:
        for c in cols:
            lim = df[c].abs().max() or 1
            sty = sty.background_gradient(cmap="RdYlGn", subset=[c], vmin=-lim, vmax=lim)
    return sty


tab_week, tab_grid, tab_log, tab_roles = st.tabs(
    ["This week", "Defense grid", "Defense game log", "Team roles (audit)"])

# --------------------------------------------------------------------------
# Tab 1: this week's matchups, player by player
# --------------------------------------------------------------------------
with tab_week:
    if nxt.empty:
        st.info("No upcoming regular-season games.")
    else:
        roles_now = d.current_roles(pg, season_ov, next_week)
        m = roles_now.merge(nxt, on="team", how="inner")
        m["opp_def"] = m["next_opp"].str.lstrip("@")
        m = m.merge(dvr[["defteam", "role", col, "avg", "diff", "pct", "games"]],
                    left_on=["opp_def", "role"], right_on=["defteam", "role"], how="left")
        wk = int(m["next_week"].min())
        m = m[m["next_week"] == wk]

        c1, c2 = st.columns([2, 3])
        role_pick = c1.multiselect("Roles", d.ROLE_ORDER, default=d.ROLE_ORDER)
        team_pick = c2.multiselect("Teams (blank = all)", sorted(m["team"].unique()))
        m = m[m["role"].isin(role_pick)]
        if team_pick:
            m = m[m["team"].isin(team_pick)]

        out = pd.DataFrame({
            "Team": m["team"], "Opp": m["next_opp"], "Role": m["role"],
            "Player": m["player"], "Tgt share": m["share"],
            "Opp allows": m[col], "Lg avg": m["avg"],
            "Edge": m[value] if mode != "Per game" else m["diff"],
            "G": m["games"],
        }).sort_values("Edge", ascending=False)

        st.subheader(f"Week {wk}: who has the soft matchup")
        st.caption("Roles are projected from target share through each team's last game. "
                   "Anyone who missed that game is not listed, so check injury news. "
                   "Edge = what the opponent allows to this role minus league average.")
        edge_fmt = (lambda v: f"{v:+.0%}") if mode == "vs league avg (%)" else (lambda v: f"{v:+.1f}")
        lim = out["Edge"].abs().max() or 1
        st.dataframe(
            out.style.format({"Tgt share": "{:.0%}", "Opp allows": "{:.1f}",
                              "Lg avg": "{:.1f}", "Edge": edge_fmt, "G": "{:.0f}"},
                             na_rep="")
               .background_gradient(cmap="RdYlGn", subset=["Edge"], vmin=-lim, vmax=lim),
            hide_index=True, width="stretch", height=620)

# --------------------------------------------------------------------------
# Tab 2: 32-defense grid
# --------------------------------------------------------------------------
with tab_grid:
    grid = dvr.pivot(index="defteam", columns="role", values=value)[d.ROLE_ORDER]
    games = dvr.groupby("defteam")["games"].first()
    grid.insert(0, "G", games)
    grid = grid.join(nxt.set_index("team")[["next_opp"]].rename(columns={"next_opp": "Next"}))
    sort_by = st.selectbox("Sort by", d.ROLE_ORDER, index=1)
    grid = grid.sort_values(sort_by, ascending=False).reset_index().rename(
        columns={"defteam": "Defense"})
    st.dataframe(style_grid(grid, d.ROLE_ORDER), hide_index=True,
                 width="stretch", height=1180)

    lg_row = pd.DataFrame([lg[col].reindex(d.ROLE_ORDER)]).round(1)
    lg_row.index = ["League avg / game"]
    st.caption("League average per game (for reference):")
    st.dataframe(lg_row, width="stretch")

# --------------------------------------------------------------------------
# Tab 3: one defense, game by game, with names
# --------------------------------------------------------------------------
with tab_log:
    teams = sorted(dvr["defteam"].unique())
    dteam = st.selectbox("Defense", teams, key="log_team")
    sub = dvr[dvr["defteam"] == dteam].set_index("role").reindex(d.ROLE_ORDER)
    cols = st.columns(len(d.ROLE_ORDER))
    for c, role in zip(cols, d.ROLE_ORDER):
        r = sub.loc[role]
        c.metric(role, f"{r[col]:.1f}", f"{r['diff']:+.1f} vs avg")
    st.caption(f"{STAT_LABELS[stat]} per game allowed · {int(sub['games'].iloc[0])} games. "
               "Each line reads Rec/Targets, yards.")
    st.dataframe(d.defense_game_log(pg, dteam, weeks), hide_index=True,
                 width="stretch")

# --------------------------------------------------------------------------
# Tab 4: role audit + overrides
# --------------------------------------------------------------------------
with tab_roles:
    teams = sorted(pg["team"].unique())
    oteam = st.selectbox("Offense", teams, key="role_team")
    st.dataframe(d.team_roles(pg, oteam), hide_index=True, width="stretch")
    st.caption(
        "% = target share entering that game (early weeks blended with last season). "
        "'prior' = based only on last season; 'snaps' = no target history; "
        "✎ = set by your override file.")
    with st.expander("How to fix a wrong role"):
        st.markdown(
            "Edit `role_overrides.csv` in your GitHub repo (pencil icon → commit). "
            "One row per fix:\n\n"
            "```\nseason,week,team,player,role\n2026,,CIN,Tee Higgins,WR1\n"
            "2026,6,CIN,Andrei Iosivas,WR2\n```\n"
            "- Leave **week** blank to apply all season; fill it for one game.\n"
            "- **role** is WR1, WR2, WR3, or WR4 (to push someone out of the top 3).\n"
            "- **player** is the full name as shown in this table.\n\n"
            "The app redeploys on commit; changes show within a minute or two.")
    if len(season_ov):
        st.caption("Active overrides:")
        st.dataframe(season_ov, hide_index=True)

st.divider()
st.caption("Data: nflverse (CC-BY 4.0) via nflreadpy. Receiving only; "
           "PPR = rec + yds/10 + 6×TD. Sacks, scrambles and 2-pt tries excluded.")
