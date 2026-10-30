"""Props tab: projections vs sportsbook lines for receptions, receiving yards,
rush attempts, rushing yards and passing yards."""
from __future__ import annotations

import os
import time
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import odds as od
import props_model as pm

EDGE_MIN = 0.05  # below ~5 pts the model can't separate itself from season-to-season drift
COUNT = {"receptions": True, "rec_yards": False, "carries": True, "rush_yards": False, "pass_yards": False}
STAT_COL = {"receptions": "receptions", "rec_yards": "receiving_yards", "carries": "carries",
            "rush_yards": "rushing_yards", "pass_yards": "pass_yds"}


def _season() -> int:
    t = date.today()
    return t.year if t.month >= 8 else t.year - 1


@st.cache_resource(ttl=6 * 3600, show_spinner="Loading nflverse data…")
def _raw(season: int):
    return pm.load_raw([season - 2, season - 1, season], current=season)


@st.cache_data(ttl=6 * 3600, show_spinner="Building this week's projections…")
def _projections(season: int, manual_out: tuple = ()):
    return pm.project([season - 2, season - 1, season], manual_out=manual_out, raw=_raw(season))


@st.cache_data(ttl=6 * 3600)
def _out_options(season: int) -> pd.DataFrame:
    """Players with real volume who could be ruled out (shown in the picker)."""
    R = _projections(season, ())
    p = R["players"]
    p = p[(p["E_tgt"] >= 1.5) | (p["E_car0"] >= 3)]
    p = pd.concat([p, R["out"].assign(pos=R["out"]["position"])], ignore_index=True)
    p = p.drop_duplicates("player_id").sort_values(["team", "name"])
    return p[["player_id", "name", "team", "pos"]]


def _api_key() -> str | None:
    try:
        k = st.secrets.get("ODDS_API_KEY")
    except Exception:
        k = None
    return k or os.environ.get("ODDS_API_KEY")


def _ordinal(n) -> str:
    if pd.isna(n):
        return ""
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# --------------------------------------------------------------------------
def _market_frame(R: dict, market: str, force_ids: set | None = None) -> pd.DataFrame:
    """Every relevant player for this market with stats, opponent context, projection."""
    T = R["T"]
    if market == "pass_yards":
        q = R["qbs"].copy()
        hist = R["qb_history"]
        df = pd.DataFrame({
            "player_id": q["qb_id"], "Player": q["name"], "Pos": "QB", "team": q["team"], "opp": q["opp"],
            "game_id": q["game_id"], "season_pg": q["ypg_season"],
            "l4_pg": q["l4_pass_yds"],
            "Usage": q["att_pg_season"].map(lambda x: f"{x:.0f} att/g" if pd.notna(x) else "—")
                     + " · proj " + (q["est_team_att"] * q["est_share"]).map(lambda x: f"{x:.0f} att")
                     + " · " + q["est_ypa"].map(lambda x: f"{x:.1f} YPA"),
            "opp_val": q["opp_pass_yds_pg"], "opp_label": "pass yds/g",
            "mu": q["mu_pass_yards"], "games": q["c_g"], "inj": None, "new_team": False,
            "team_spread": q["team_spread"], "total_line": q["total_line"],
        })
        df["_detail"] = [dict(r) for r in q.to_dict("records")]
        return df

    up = R["players"].copy()
    force = up["player_id"].isin(force_ids or set())
    if market in ("receptions", "rec_yards"):
        up = up[((up["E_tgt"] >= 3.0) & up["pos"].isin(["WR", "TE", "RB"])) | force]
        stat = "receptions" if market == "receptions" else "receiving_yards"
        opp_col = "opp_targets_pg" if market == "receptions" else "opp_pass_yds_pg"
        opp_lab = "targets/g" if market == "receptions" else "pass yds/g"
        usage = (up["tshare_adj"].map(lambda x: f"{x:.0%} tgt share") + " · proj "
                 + up["E_tgt"].map(lambda x: f"{x:.1f} tgts"))
    else:
        up = up[((up["E_car0"] >= 6.0) & (up["pos"] == "RB")) | force]
        stat = "carries" if market == "carries" else "rushing_yards"
        opp_col = "opp_carries_pg" if market == "carries" else "opp_rush_yds_pg"
        opp_lab = "RB carries/g" if market == "carries" else "rush yds/g"
        usage = (up["cshare_adj"].map(lambda x: f"{x:.0%} carry share") + " · "
                 + up["est_ypc"].map(lambda x: f"{x:.1f} YPC"))
    df = pd.DataFrame({
        "player_id": up["player_id"], "Player": up["name"], "Pos": up["pos"], "team": up["team"],
        "opp": up["opp"], "game_id": up["game_id"],
        "season_pg": up["c_" + stat] / up["c_g"].replace(0, np.nan), "l4_pg": up["l4_" + stat],
        "Usage": usage, "opp_val": up[opp_col],
        "opp_label": opp_lab, "mu": up["mu_" + market], "games": up["c_g"], "inj": up["inj_status"],
        "new_team": up["new_team"], "team_spread": up["team_spread"], "total_line": up["total_line"],
    })
    df["_detail"] = [dict(r) for r in up.to_dict("records")]
    return df


def _attach_lines(df: pd.DataFrame, lines: pd.DataFrame, market: str) -> pd.DataFrame:
    if lines is None or lines.empty:
        return df.assign(line=np.nan, over=np.nan, under=np.nan, book_p_over=np.nan, n_books=0)
    l = lines[lines["market"] == market][["player_id", "line", "over", "under", "book_p_over", "n_books"]]
    return df.merge(l.drop_duplicates("player_id"), on="player_id", how="left")


def _flags(r, out_by_team: dict) -> str:
    f = []
    if isinstance(r["inj"], str) and r["inj"]:
        f.append(r["inj"])
    if pd.notna(r["games"]) and r["games"] < 4:
        f.append(f"{int(r['games'])} g")
    if r.get("new_team"):
        f.append("new team")
    outs = out_by_team.get(r["team"], [])
    if outs:
        f.append("out: " + ", ".join(outs))
    return " · ".join(f)


# --------------------------------------------------------------------------
def render():
    season = _season()
    opts = _out_options(season)
    label = {r.player_id: f"{r.name} · {r.team} {r.pos}" for r in opts.itertuples(index=False)}
    auto = _projections(season, ())["out"]
    auto = auto[auto["inj_status"] == "Out"]
    picked = st.multiselect(
        "Ruled out this week", options=list(label), format_func=label.get, key="pm_out",
        placeholder="Add players who won't play (official Out designations are applied automatically)",
        help="Removes the player and shifts his usage to teammates (mostly the same position). "
             "Only mark players who are ruled out — questionable players stay in.")
    st.caption("From the official injury report: " + (", ".join(f"{n} ({t})" for n, t in zip(auto["name"], auto["team"]))
               if len(auto) else "no Out designations in nflverse yet for this week (they usually appear Friday night/Saturday)."))
    R = _projections(season, tuple(sorted(picked)))
    dist = R["dist"]
    out_by_team = R["out"].groupby("team")["name"].apply(list).to_dict() if len(R["out"]) else {}

    # ---------------- odds
    key = _api_key()
    top = st.columns([2, 2, 3])
    payload = od.cached()
    if key:
        if top[0].button("Pull latest odds", help="Uses ~5 Odds API credits per game (~70–80 for a full week)."):
            games = R["games"]
            nxt = games[games["game_id"].isin(R["players"]["game_id"]) & games["home"]]
            with st.spinner("Pulling player props…"):
                try:
                    payload = od.fetch(key, set(zip(nxt["team"], nxt["opp"])))
                except Exception as e:
                    st.error(f"Odds pull failed: {e}")
        if payload:
            u = payload.get("usage", {})
            top[1].caption(f"Odds pulled {time.strftime('%a %I:%M %p', time.localtime(payload['pulled_at']))} · "
                           f"{payload.get('n_events', 0)} games · credits left: {u.get('x-requests-remaining', '?')}")
        else:
            top[1].caption("No odds pulled in the last 12 hours.")
    else:
        top[0].info("No Odds API key set. Add `ODDS_API_KEY` in the app's Secrets to pull lines "
                    "automatically. You can still enter lines by hand below.")

    wide = od.parse(payload) if payload else pd.DataFrame()
    books = ["Consensus"] + (sorted(wide["book"].dropna().unique()) if len(wide) else [])
    every = pd.concat([R["players"][["player_id", "name", "team"]],
                       R["qbs"].rename(columns={"qb_id": "player_id"})[["player_id", "name", "team"]]])
    if len(wide):
        wide = od.match_players(wide, every, pm.norm_name)

    # ---------------- controls
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    market = c1.radio("Market", list(pm.MARKETS), format_func=pm.MARKETS.get, horizontal=True, key="pm_market")
    book = c2.selectbox("Book", books, key="pm_book")
    teams = c3.multiselect("Teams", sorted(R["players"]["team"].unique()), key="pm_teams")
    only_lines = c4.toggle("Only players with a line", value=False, key="pm_only")
    lines = od.consensus(wide, book) if len(wide) else pd.DataFrame()

    force = set(lines.loc[lines["market"] == market, "player_id"]) if len(lines) else set()
    df = _market_frame(R, market, force)
    df = _attach_lines(df, lines, market)

    # manual lines (session only) override pulled lines
    man_key = f"manual_{market}"
    manual = st.session_state.get(man_key)
    if manual is not None and len(manual):
        m = manual.dropna(subset=["Line"])
        for r in m.itertuples(index=False):
            i = df.index[df["Player"] == r.Player]
            if len(i):
                df.loc[i, "line"] = r.Line
                df.loc[i, "over"] = r.Over if pd.notna(r.Over) else -110
                df.loc[i, "under"] = r.Under if pd.notna(r.Under) else -110
                po, _ = od.implied_no_vig(df.loc[i, "over"], df.loc[i, "under"])
                df.loc[i, "book_p_over"] = po
                df.loc[i, "n_books"] = 0

    if teams:
        df = df[df["team"].isin(teams)]
    if only_lines:
        df = df[df["line"].notna()]

    # ---------------- probabilities
    spec = dist[market]; isc = COUNT[market]
    df["fair"] = [pm.fair_line(spec, m, isc) for m in df["mu"]]
    has = df["line"].notna()
    df["p_over"] = np.nan
    if has.any():
        df.loc[has, "p_over"] = pm.p_over(spec, df.loc[has, "mu"], df.loc[has, "line"])
    df["p_under"] = 1 - df["p_over"]
    df["edge_over"] = df["p_over"] - df["book_p_over"]
    df["edge_under"] = df["p_under"] - (1 - df["book_p_over"])
    df["side"] = np.where(df["edge_over"] >= df["edge_under"], "Over", "Under")
    df["edge"] = df[["edge_over", "edge_under"]].max(axis=1)
    dec_o, dec_u = od.american_to_decimal(df["over"].fillna(-110)), od.american_to_decimal(df["under"].fillna(-110))
    df["ev"] = np.where(df["side"] == "Over", df["p_over"] * dec_o - 1, df["p_under"] * dec_u - 1)
    df.loc[~has, ["edge", "ev", "side"]] = [np.nan, np.nan, ""]

    # opponent rank among this week's opponents (1 = allows the most)
    T = R["T"]
    opp_vals = df.groupby("opp")["opp_val"].first()
    rank = opp_vals.rank(ascending=False, method="min")
    df["opp_rank"] = df["opp"].map(rank)
    games = R["games"]
    home = games.set_index(["game_id", "team"])["home"]
    df["Opp"] = [("" if home.get((g, t), True) else "@") + o for g, t, o in zip(df["game_id"], df["team"], df["opp"])]
    df["Flags"] = [_flags(r, out_by_team) for r in df.to_dict("records")]

    show = pd.DataFrame({
        "Player": df["Player"], "Pos": df["Pos"], "Team": df["team"], "Opp": df["Opp"],
        "Season /g": df["season_pg"], "Last 4 /g": df["l4_pg"], "Usage": df["Usage"],
        "Opp allows": df["opp_val"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—") + " "
                      + df["opp_label"] + " (" + df["opp_rank"].map(_ordinal) + ")",
        "Projection": df["mu"], "Fair line": df["fair"], "Line": df["line"],
        "Over": df["over"], "Under": df["under"], "P(over)": df["p_over"], "Book P(over)": df["book_p_over"],
        "Pick": df["side"], "Edge": df["edge"], "EV": df["ev"], "Flags": df["Flags"],
    })
    show = show.sort_values(["Edge", "Projection"], ascending=[False, False], na_position="last")

    def hl(v):
        if pd.isna(v):
            return ""
        return "background-color: rgba(34,197,94,.35); font-weight:600" if v >= EDGE_MIN else ""
    st.dataframe(
        show.style.format({"Season /g": "{:.1f}", "Last 4 /g": "{:.1f}", "Projection": "{:.1f}",
                           "Fair line": "{:.1f}", "Line": "{:.1f}", "Over": "{:+.0f}", "Under": "{:+.0f}",
                           "P(over)": "{:.0%}", "Book P(over)": "{:.0%}", "Edge": "{:+.1%}", "EV": "{:+.1%}"},
                          na_rep="—").map(hl, subset=["Edge"]),
        hide_index=True, width="stretch", height=640)
    st.caption(
        "**Projection** = model average. **Fair line** = the line where the model sees 50/50; for yards it sits "
        "below the average because a few big games pull the average up. **Book P(over)** = the book's price "
        "with its margin removed. **Edge** = model probability minus book probability on the better side; "
        f"green when ≥ {EDGE_MIN:.0%} (smaller gaps are within the model's season-to-season drift). "
        "**EV** = expected return per $1 at the listed price. Opp rank is among this week's opponents "
        "(1st = allows the most). Flags: injury status, games played this season, new team, "
        "teammates ruled out.")

    # ---------------- manual entry
    with st.expander("Enter or override lines by hand (kept until you close the tab)"):
        base = st.session_state.get(man_key)
        if base is None:
            base = pd.DataFrame({"Player": df["Player"].sort_values().tolist(), "Line": np.nan, "Over": np.nan, "Under": np.nan})
        ed = st.data_editor(base, hide_index=True, width="stretch", key=f"ed_{market}",
                            column_config={"Player": st.column_config.TextColumn(disabled=True),
                                           "Line": st.column_config.NumberColumn(step=0.5),
                                           "Over": st.column_config.NumberColumn(help="American odds, e.g. -115"),
                                           "Under": st.column_config.NumberColumn(help="American odds, e.g. -105")})
        if st.button("Apply lines", key=f"apply_{market}"):
            st.session_state[man_key] = ed
            st.rerun()

    if len(wide):
        unmatched = wide[wide["player_id"].isna()]["player_raw"].unique()
        if len(unmatched):
            with st.expander(f"{len(unmatched)} players in the odds feed couldn't be matched"):
                st.write(", ".join(sorted(unmatched)))

    # ---------------- detail
    st.divider()
    if show.empty:
        return
    pick = st.selectbox("Player detail", show["Player"].tolist(), key="pm_pick")
    r = df[df["Player"] == pick].iloc[0]
    d = r["_detail"]
    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"**{pick}** ({r['Pos']}, {r['team']}) vs **{r['Opp']}** · spread {r['team_spread']:+.1f}, total {r['total_line']:.1f}")
        st.markdown(_buildup(market, d, r["mu"]))
        if pd.notna(r["line"]):
            st.markdown(f"Line **{r['line']}** · model P(over) **{r['p_over']:.0%}** vs book {r['book_p_over']:.0%} "
                        f"→ **{r['side']} {r['edge']:+.1%}**, EV {r['ev']:+.1%}")
        st.caption(f"Fair line {r['fair']:.1f}. " + (r["Flags"] or ""))
    with right:
        x, p = pm.distribution(spec, float(r["mu"]), isc)
        line = r["line"] if pd.notna(r["line"]) else r["fair"]
        ch = pd.DataFrame({"x": x, "p": p, "side": np.where(x + (0 if isc else 2.5) > line, "Over", "Under")})
        bars = alt.Chart(ch).mark_bar().encode(
            x=alt.X("x:Q", title=pm.MARKETS[market]), y=alt.Y("p:Q", title="Probability", axis=alt.Axis(format="%")),
            color=alt.Color("side:N", scale=alt.Scale(domain=["Under", "Over"], range=["#94a3b8", "#22c55e"]), legend=None),
            tooltip=[alt.Tooltip("x:Q"), alt.Tooltip("p:Q", format=".1%")])
        rule = alt.Chart(pd.DataFrame({"x": [line]})).mark_rule(color="#ef4444", strokeWidth=2).encode(x="x:Q")
        st.altair_chart(bars + rule, width="stretch")
        st.caption("Model's outcome distribution. Red line = the book line (or the fair line if none).")
    _game_log(R, market, r)


def _buildup(market: str, d: dict, mu: float) -> str:
    if market in ("receptions", "rec_yards"):
        eff = d["est_cr"] if market == "receptions" else d["est_ypt"]
        base = d["E_tgt"] * eff
        effs = f"× catch rate {eff:.0%}" if market == "receptions" else f"× {eff:.1f} yds/target"
        return (f"Team targets/g **{d['est_team_tpg']:.1f}** × his share **{d['tshare_adj']:.1%}** "
                f"= **{d['E_tgt']:.1f}** targets {effs} = {base:.1f}  \n"
                f"Game adjustments ×{mu / base:.2f} (spread, total, opponent targets allowed ×{d['opp_tgt_ratio']:.2f}) "
                f"→ **{mu:.1f}**")
    if market in ("carries", "rush_yards"):
        base = d["E_car0"] * (d["est_ypc"] if market == "rush_yards" else 1)
        effs = f"× {d['est_ypc']:.2f} YPC = {base:.1f}" if market == "rush_yards" else ""
        return (f"Team RB carries/g **{d['est_team_cpg']:.1f}** × his share **{d['cshare_adj']:.0%}** "
                f"= **{d['E_car0']:.1f}** carries {effs}  \n"
                f"Game adjustments ×{mu / base:.2f} (spread, total, opponent carries allowed ×{d['opp_car_ratio']:.2f}"
                + (f", YPC allowed ×{d['opp_ypc_ratio']:.2f}" if market == "rush_yards" else "") + f") → **{mu:.1f}**")
    base = d["E_pass0"]
    return (f"Team attempts/g **{d['est_team_att']:.1f}** × starter share {d['est_share']:.0%} × "
            f"**{d['est_ypa']:.2f}** yds/att = {base:.0f}  \n"
            f"Game adjustments ×{mu / base:.2f} (spread, total, opp attempts allowed ×{d['opp_att_ratio']:.2f}, "
            f"YPA allowed ×{d['opp_ypa_ratio']:.2f}) → **{mu:.0f}**")


def _game_log(R: dict, market: str, r) -> None:
    season = R["players"]["season"].max()
    if market == "pass_yards":
        h = R["qb_history"]
        h = h[(h["qb_id"] == r["player_id"]) & h["played"] & (h["season"] == season)]
        log = pd.DataFrame({"Wk": h["week"], "Opp": h["opp"], "Att": h["attempts"], "Pass yds": h["passing_yards"]})
    else:
        h = R["history"]
        h = h[(h["player_id"] == r["player_id"]) & (~h["upcoming"]) & (h["season"] == season)]
        log = pd.DataFrame({"Wk": h["week"], "Opp": h["opp"], "Snap %": h["offense_pct"], "Tgt": h["targets"],
                            "Rec": h["receptions"], "Rec yds": h["receiving_yards"], "Car": h["carries"],
                            "Rush yds": h["rushing_yards"]})
    if len(log):
        st.markdown("**This season**")
        st.dataframe(log.sort_values("Wk").style.format({"Snap %": "{:.0%}"}, na_rep="—"), hide_index=True, width="stretch")
