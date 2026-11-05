"""Opponent-adjusted rush/pass ratings and the weekly game-matchup view.

Every play: yards = league avg + offense effect + defense effect + home field, fit jointly by ridge
regression (current season only, plays before the selected week). Ridge pulls small samples toward
average; LAMBDA = 400 plays was best in the 2023-25 backtest (model/def_adjust.py).
Rush = designed runs (QB designed runs included, kneels excluded). Pass = dropbacks (sacks and
scrambles included). Garbage time (win prob <10% or >90%) and 2-pt tries are dropped.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
import streamlit as st

LAMBDA = 400
KINDS = {"rush": "Run", "pass": "Pass"}
UNIT = {"rush": "yds/carry", "pass": "yds/dropback"}


@st.cache_data(ttl=6 * 3600, show_spinner="Loading play-by-play…")
def load_plays(season: int) -> pd.DataFrame:
    cols = ["season", "season_type", "week", "game_id", "posteam", "defteam", "home_team", "play_type",
            "pass", "rush", "two_point_attempt", "qb_kneel", "qb_spike", "yards_gained", "wp"]
    p = nfl.load_pbp([season]).select(cols).to_pandas()
    p = p[(p["season_type"] == "REG") & p["play_type"].isin(["run", "pass"]) & (p["two_point_attempt"] != 1)
          & (p["qb_kneel"] != 1) & (p["qb_spike"] != 1) & p["yards_gained"].notna() & p["posteam"].notna()]
    p["kind"] = np.where(p["pass"] == 1, "pass", np.where(p["rush"] == 1, "rush", ""))
    p = p[(p["kind"] != "") & p["wp"].between(0.10, 0.90)]
    sch = load_schedule(season)
    neutral = set(sch.loc[sch["location"] == "Neutral", "game_id"])
    p["home"] = np.where(p["game_id"].isin(neutral), 0.0,
                         np.where(p["posteam"] == p["home_team"], 1.0, -1.0))
    return p[["week", "game_id", "posteam", "defteam", "home", "kind", "yards_gained"]]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_schedule(season: int) -> pd.DataFrame:
    s = nfl.load_schedules([season]).to_pandas()
    return s[s["game_type"] == "REG"]


def _fit(d: pd.DataFrame):
    """Ridge with offense, defense and home terms. Returns (mu, off, def, home)."""
    teams = sorted(set(d["posteam"]) | set(d["defteam"]))
    ix = {t: i for i, t in enumerate(teams)}
    k, n = len(teams), len(d)
    X = np.zeros((n, 2 * k + 1))
    r = np.arange(n)
    X[r, d["posteam"].map(ix).to_numpy()] = 1
    X[r, k + d["defteam"].map(ix).to_numpy()] = 1
    X[:, -1] = d["home"].to_numpy()
    y = d["yards_gained"].to_numpy(float)
    mu = y.mean()
    b = np.linalg.solve(X.T @ X + LAMBDA * np.eye(2 * k + 1), X.T @ (y - mu))
    return mu, pd.Series(b[:k], teams), pd.Series(b[k:2 * k], teams), b[-1]


@st.cache_data(ttl=6 * 3600, show_spinner="Rating offenses and defenses…")
def ratings(season: int, before_week: int) -> dict:
    """Per kind: league avg, home edge, and a team table (raw + adjusted, offense and defense)."""
    P = load_plays(season)
    P = P[P["week"] < before_week]
    out = {}
    for kind in KINDS:
        d = P[P["kind"] == kind]
        if d.empty:
            continue
        mu, off, dfn, home = _fit(d)
        o = d.groupby("posteam")["yards_gained"].agg(["mean", "size"])
        a = d.groupby("defteam")["yards_gained"].agg(["mean", "size"])
        T = pd.DataFrame({"off_raw": o["mean"], "off_n": o["size"], "off_adj": mu + off,
                          "def_raw": a["mean"], "def_n": a["size"], "def_adj": mu + dfn})
        T["off_rank"] = T["off_adj"].rank(ascending=False, method="min")   # 1 = best offense
        T["def_rank"] = T["def_adj"].rank(ascending=True, method="min")    # 1 = toughest defense
        T["off_rank_raw"] = T["off_raw"].rank(ascending=False, method="min")
        T["def_rank_raw"] = T["def_raw"].rank(ascending=True, method="min")
        out[kind] = {"mu": mu, "home": home, "T": T}
    return out


def _ord(n):
    n = int(n)
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _games(season: int, week: int, R: dict) -> pd.DataFrame:
    s = load_schedule(season)
    g = s[s["week"] == week].sort_values(["gameday", "gametime"]).copy()
    rows = []
    for _, x in g.iterrows():
        a, h = x["away_team"], x["home_team"]
        hs = 0.0 if x["location"] == "Neutral" else 1.0
        sp = x["spread_line"]  # nflverse: home margin expected by the market (+ = home favored)
        if pd.isna(sp):
            spread = "—"
        elif sp == 0:
            spread = "PK"
        else:
            spread = f"{h} -{abs(sp):g}" if sp > 0 else f"{a} -{abs(sp):g}"
        row = {"game_id": x["game_id"], "Game": f"{a} @ {h}" if hs else f"{a} v {h}",
               "Kickoff": f"{pd.to_datetime(x['gameday']):%a} {x['gametime']}", "Spread": spread,
               "Total": x["total_line"], "away": a, "home": h, "hs": hs}
        for kind in KINDS:
            if kind not in R:
                continue
            k = R[kind]
            T = k["T"]
            for side, team, opp, sign in [("Away", a, h, -1), ("Home", h, a, 1)]:
                if team in T.index and opp in T.index:
                    exp = T.at[team, "off_adj"] + T.at[opp, "def_adj"] - k["mu"] + sign * hs * k["home"]
                else:
                    exp = np.nan
                row[f"{side} {KINDS[kind].lower()}"] = exp
                row[f"{side} {KINDS[kind].lower()} d"] = exp - k["mu"]
        rows.append(row)
    return pd.DataFrame(rows)


def _detail(R: dict, kind: str, team: str, opp: str, exp: float) -> pd.DataFrame:
    T = R[kind]["T"]
    o, d = T.loc[team], T.loc[opp]
    u = UNIT[kind]
    return pd.DataFrame({
        "": [f"{team} {KINDS[kind].lower()} offense", f"{opp} {KINDS[kind].lower()} defense", "Expected this game"],
        "Raw": [f"{o['off_raw']:.2f}", f"{d['def_raw']:.2f}", ""],
        "Adjusted": [f"{o['off_adj']:.2f}", f"{d['def_adj']:.2f}", f"{exp:.2f}"],
        "Rank (adj)": [_ord(o["off_rank"]), _ord(d["def_rank"]), ""],
        "Rank (raw)": [_ord(o["off_rank_raw"]), _ord(d["def_rank_raw"]), ""],
        "Plays": [f"{o['off_n']:.0f}", f"{d['def_n']:.0f}", ""],
    }).set_index("")


def render(season: int):
    s = load_schedule(season)
    if s.empty:
        st.info("No schedule for this season yet.")
        return
    played = s[s["result"].notna()]
    upcoming = s[s["result"].isna()]
    default_wk = int(upcoming["week"].min()) if len(upcoming) else int(s["week"].max())
    weeks = sorted(s["week"].unique())
    c1, c2 = st.columns([1, 6], vertical_alignment="bottom")
    week = c1.selectbox("Week", weeks, index=weeks.index(default_wk), key="gm_week")
    if week < 2 or played.empty:
        c2.caption("Ratings need at least one week of games.")
        return
    R = ratings(season, int(week))
    G = _games(season, int(week), R)
    if G.empty:
        st.info("No games this week.")
        return
    c2.caption(f"Expected yards per play for each unit this game, from offense and defense ratings adjusted for "
               f"opponents (weeks 1–{week - 1}, garbage time removed). Run = yds/carry, pass = yds/dropback. "
               f"(±) = vs league average: run {R['rush']['mu']:.2f}, pass {R['pass']['mu']:.2f}. "
               "Select a game for the build-up.")

    cells = ["Away run", "Home run", "Away pass", "Home pass"]
    show = G[["Game", "Kickoff", "Spread", "Total"]].copy()
    for c in cells:
        show[c] = [f"{v:.2f} ({dv:+.2f})" if pd.notna(v) else "—" for v, dv in zip(G[c], G[c + " d"])]

    def shade(col):
        dv = G[col.name + " d"]
        lim = 0.6 if "run" in col.name else 0.9
        out = []
        for v in dv:
            if pd.isna(v):
                out.append("")
                continue
            a = min(abs(v) / lim, 1) * 0.55
            out.append(f"background-color: rgba({'46,160,67' if v > 0 else '218,54,51'},{a:.2f})")
        return out

    sty = show.style.apply(shade, subset=cells).format({"Total": "{:.1f}"}, na_rep="—")
    ev = st.dataframe(sty, hide_index=True, width="stretch", key=f"gm_table_{week}",
                      on_select="rerun", selection_mode="single-row",
                      height=35 * (len(show) + 1) + 3,
                      column_config={c: st.column_config.TextColumn(
                          help=f"{c.split()[0]} offense's expected {UNIT['rush' if 'run' in c else 'pass']} "
                               "(vs league average)") for c in cells})
    sel = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
    if not sel:
        return
    g = G.iloc[sel[0]]
    st.markdown(f"**{g['Game']}** · {g['Spread']} · total {g['Total']:.1f}" if pd.notna(g["Total"])
                else f"**{g['Game']}**")
    cols = st.columns(2)
    for col, kind in zip(cols, ["rush", "pass"]):
        with col:
            for side, team, opp in [("Away", g["away"], g["home"]), ("Home", g["home"], g["away"])]:
                exp = g[f"{side} {KINDS[kind].lower()}"]
                if pd.isna(exp):
                    continue
                st.caption(f"{team} {KINDS[kind].lower()} vs {opp} ({UNIT[kind]})")
                st.dataframe(_detail(R, kind, team, opp, exp), width="stretch")
