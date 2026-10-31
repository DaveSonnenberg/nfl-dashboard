"""Props tab: projections vs sportsbook lines for receptions, receiving yards,
rush attempts, rushing yards (RBs + QBs) and passing yards."""
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
COUNT_KEYS = {"receptions", "carries", "qb_carries"}
ROLES = pm.ROLE_ORDER  # WR1-WR4, TE1, TE2, RB


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


def _drole(role: str) -> str:
    return "RB" if str(role).startswith("RB") else str(role)


# --------------------------------------------------------------------------
# table construction
# --------------------------------------------------------------------------
def _market_frame(R: dict, market: str, force_ids: set | None = None) -> pd.DataFrame:
    """Every relevant player for this market with stats, opponent context, projection."""
    force_ids = force_ids or set()
    q = R["qbs"]
    if market == "pass_yards":
        df = pd.DataFrame({
            "player_id": q["qb_id"], "Player": q["name"], "Pos": "QB", "team": q["team"], "opp": q["opp"],
            "game_id": q["game_id"], "season_pg": q["ypg_season"], "l4_pg": q["l4_pass_yds"],
            "Usage": q["att_pg_season"].map(lambda x: f"{x:.0f} att/g" if pd.notna(x) else "—")
                     + " · proj " + (q["est_team_att"] * q["est_share"]).map(lambda x: f"{x:.0f} att")
                     + " · " + q["est_ypa"].map(lambda x: f"{x:.1f} YPA"),
            "opp_val": q["opp_pass_yds_pg"], "opp_label": "pass yds/g", "opp_group": "pass",
            "mu": q["mu_pass_yards"], "games": q["c_g"], "inj": None, "new_team": False,
            "team_spread": q["team_spread"], "total_line": q["total_line"], "dist_key": "pass_yards", "kind": "qb",
        })
        df["_detail"] = q.to_dict("records")
        return df

    up = R["players"]
    force = up["player_id"].isin(force_ids)
    if market in ("receptions", "rec_yards"):
        up = up[((up["E_tgt"] >= 3.0) & up["pos"].isin(["WR", "TE", "RB"])) | force]
        stat = "receptions" if market == "receptions" else "receiving_yards"
        dr = R["def_role"]
        roles = up["role"].map(_drole)
        vals = [dr[stat].loc[o, r] if (o in dr.index and r in ROLES) else np.nan for o, r in zip(up["opp"], roles)]
        rnks = [dr["rank_" + stat].loc[o, r] if (o in dr.index and r in ROLES) else np.nan for o, r in zip(up["opp"], roles)]
        unit = "rec/g" if stat == "receptions" else "yds/g"
        opp_text = [f"{v:.1f} {unit} to {r}s ({_ordinal(k)})" if pd.notna(v) else "—"
                    for v, r, k in zip(vals, roles, rnks)]
        df = pd.DataFrame({
            "player_id": up["player_id"], "Player": up["name"], "Pos": up["role"], "team": up["team"],
            "opp": up["opp"], "game_id": up["game_id"],
            "season_pg": up["c_" + stat] / up["c_g"].replace(0, np.nan), "l4_pg": up["l4_" + stat],
            "Usage": up["tshare_adj"].map(lambda x: f"{x:.0%} tgt share") + " · proj "
                     + up["E_tgt"].map(lambda x: f"{x:.1f} tgts"),
            "opp_text": opp_text, "opp_val": vals, "opp_rank": rnks,
            "mu": up["mu_" + market], "games": up["c_g"], "inj": up["inj_status"], "new_team": up["new_team"],
            "team_spread": up["team_spread"], "total_line": up["total_line"], "dist_key": market, "kind": "skill",
        })
        df["_detail"] = up.to_dict("records")
        return df

    # rushing: RBs + starting QBs
    stat = "carries" if market == "carries" else "rushing_yards"
    rb = up[((up["E_car0"] >= 6.0) & (up["pos"] == "RB")) | (force & (up["pos"] == "RB"))]
    rb_df = pd.DataFrame({
        "player_id": rb["player_id"], "Player": rb["name"], "Pos": rb["role"], "team": rb["team"], "opp": rb["opp"],
        "game_id": rb["game_id"], "season_pg": rb["c_" + stat] / rb["c_g"].replace(0, np.nan), "l4_pg": rb["l4_" + stat],
        "Usage": rb["cshare_adj"].map(lambda x: f"{x:.0%} carry share") + " · " + rb["est_ypc"].map(lambda x: f"{x:.1f} YPC"),
        "opp_val": rb["opp_carries_pg" if stat == "carries" else "opp_rush_yds_pg"],
        "opp_label": "RB carries/g" if stat == "carries" else "rush yds/g", "opp_group": "rb",
        "mu": rb["mu_" + market], "games": rb["c_g"], "inj": rb["inj_status"], "new_team": rb["new_team"],
        "team_spread": rb["team_spread"], "total_line": rb["total_line"], "dist_key": market, "kind": "skill",
    })
    rb_df["_detail"] = rb.to_dict("records")
    qq = q[(q["mu_carries"] >= 2.5) | q["qb_id"].isin(force_ids)]
    qd = R["qb_def"]
    qb_df = pd.DataFrame({
        "player_id": qq["qb_id"], "Player": qq["name"], "Pos": "QB", "team": qq["team"], "opp": qq["opp"],
        "game_id": qq["game_id"],
        "season_pg": qq["qcar_pg_season"] if stat == "carries" else qq["qrush_pg_season"],
        "l4_pg": qq["l4_qcar"] if stat == "carries" else qq["l4_qrush"],
        "Usage": qq["qcar_pg_season"].map(lambda x: f"{x:.1f} car/start" if pd.notna(x) else "—")
                 + " · " + qq["est_qypc"].map(lambda x: f"{x:.1f} YPC"),
        "opp_val": qq["opp"].map(qd["qb_car_pg" if stat == "carries" else "qb_rush_pg"]),
        "opp_label": "QB carries/g" if stat == "carries" else "QB rush yds/g", "opp_group": "qb",
        "mu": qq["mu_" + market], "games": qq["c_qg"], "inj": None, "new_team": False,
        "team_spread": qq["team_spread"], "total_line": qq["total_line"],
        "dist_key": "qb_carries" if stat == "carries" else "qb_rush_yards", "kind": "qb",
    })
    qb_df["_detail"] = qq.to_dict("records")
    return pd.concat([rb_df, qb_df], ignore_index=True)


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


def _probabilities(df: pd.DataFrame, dist: dict) -> pd.DataFrame:
    df = df.copy()
    df["fair"] = np.nan
    df["p_over"] = np.nan
    fallback = {"qb_carries": "carries", "qb_rush_yards": "rush_yards"}
    for key, idx in df.groupby("dist_key").groups.items():
        spec = dist.get(key) or dist[fallback.get(key, key)]  # older dist_specs.json lacks the QB entries
        isc = key in COUNT_KEYS
        df.loc[idx, "fair"] = [pm.fair_line(spec, m, isc) for m in df.loc[idx, "mu"]]
        h = df.loc[idx, "line"].notna()
        hi = h[h].index
        if len(hi):
            df.loc[hi, "p_over"] = pm.p_over(spec, df.loc[hi, "mu"], df.loc[hi, "line"])
    has = df["line"].notna()
    df["p_under"] = 1 - df["p_over"]
    df["edge_over"] = df["p_over"] - df["book_p_over"]
    df["edge_under"] = df["p_under"] - (1 - df["book_p_over"])
    df["side"] = np.where(df["edge_over"] >= df["edge_under"], "Over", "Under")
    df["edge"] = df[["edge_over", "edge_under"]].max(axis=1)
    dec_o = od.american_to_decimal(df["over"].fillna(-110))
    dec_u = od.american_to_decimal(df["under"].fillna(-110))
    df["ev"] = np.where(df["side"] == "Over", df["p_over"] * dec_o - 1, df["p_under"] * dec_u - 1)
    df.loc[~has, ["edge", "ev"]] = np.nan
    df.loc[~has, "side"] = ""
    return df


# --------------------------------------------------------------------------
# player window
# --------------------------------------------------------------------------
@st.dialog("Player", width="large")
def _player_window(R: dict, market: str, r: dict):
    d = r["_detail"]
    fb = {"qb_carries": "carries", "qb_rush_yards": "rush_yards"}
    spec = R["dist"].get(r["dist_key"]) or R["dist"][fb.get(r["dist_key"], r["dist_key"])]
    isc = r["dist_key"] in COUNT_KEYS
    st.markdown(f"### {r['Player']}  \n{r['Pos']} · {r['team']} vs **{r['Opp']}** · spread {r['team_spread']:+.1f}, "
                f"total {r['total_line']:.1f}")
    st.markdown(_buildup(market, r["kind"], d, r["mu"]))
    if pd.notna(r["line"]):
        st.markdown(f"Line **{r['line']}** ({r['over']:+.0f}/{r['under']:+.0f}) · model P(over) **{r['p_over']:.0%}** "
                    f"vs book {r['book_p_over']:.0%} → **{r['side']} {r['edge']:+.1%}**, EV {r['ev']:+.1%}")
    st.caption(f"Projection {r['mu']:.1f} · fair line {r['fair']:.1f}" + (f" · {r['Flags']}" if r["Flags"] else ""))

    x, p = pm.distribution(spec, float(r["mu"]), isc)
    line = r["line"] if pd.notna(r["line"]) else r["fair"]
    ch = pd.DataFrame({"x": x, "p": p, "side": np.where(x + (0 if isc else 2.5) > line, "Over", "Under")})
    bars = alt.Chart(ch).mark_bar().encode(
        x=alt.X("x:Q", title=pm.MARKETS[market]), y=alt.Y("p:Q", title="Probability", axis=alt.Axis(format="%")),
        color=alt.Color("side:N", scale=alt.Scale(domain=["Under", "Over"], range=["#94a3b8", "#22c55e"]), legend=None),
        tooltip=[alt.Tooltip("x:Q"), alt.Tooltip("p:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"x": [line]})).mark_rule(color="#ef4444", strokeWidth=2).encode(x="x:Q")
    st.altair_chart((bars + rule).properties(height=220), width="stretch")

    if market in ("receptions", "rec_yards") and r["opp"] in R["def_role"].index:
        dr = R["def_role"]
        stat = "receptions" if market == "receptions" else "receiving_yards"
        row = pd.DataFrame({"Per game": dr[stat].loc[r["opp"]].map(lambda v: f"{v:.1f}"),
                            "Rank": dr["rank_" + stat].loc[r["opp"]].map(_ordinal)}).T
        st.markdown(f"**{r['opp']} allows by role** ({'receptions' if stat == 'receptions' else 'receiving yards'}; "
                    "rank 1st = allows the most)")
        st.dataframe(row, width="stretch")

    st.markdown("**This season**")
    _game_log(R, r)


def _buildup(market: str, kind: str, d: dict, mu: float) -> str:
    if market in ("receptions", "rec_yards"):
        eff = d["est_cr"] if market == "receptions" else d["est_ypt"]
        base = d["E_tgt"] * eff
        effs = f"× catch rate {eff:.0%}" if market == "receptions" else f"× {eff:.1f} yds/target"
        return (f"Team targets/g **{d['est_team_tpg']:.1f}** × his share **{d['tshare_adj']:.1%}** "
                f"= **{d['E_tgt']:.1f}** targets {effs} = {base:.1f}  \n"
                f"Game adjustments ×{mu / base:.2f} (spread, total, opponent targets allowed ×{d['opp_tgt_ratio']:.2f}) "
                f"→ **{mu:.1f}**")
    if market in ("carries", "rush_yards") and kind == "qb":
        base = d["E_qcar"] * (d["est_qypc"] if market == "rush_yards" else 1)
        effs = f" × {d['est_qypc']:.2f} YPC = {base:.1f}" if market == "rush_yards" else ""
        return (f"His carries per start (this season, blended with his last two) **{d['E_qcar']:.1f}**{effs}  \n"
                f"Game adjustments ×{mu / base:.2f} (spread, total; underdog QBs run a bit more) → **{mu:.1f}**")
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


def _game_log(R: dict, r: dict) -> None:
    season = R["players"]["season"].max()
    if r["kind"] == "qb":
        h = R["qb_history"]
        h = h[(h["qb_id"] == r["player_id"]) & h["played"] & (h["season"] == season)]
        log = pd.DataFrame({"Wk": h["week"], "Opp": h["opp"], "Att": h["attempts"], "Pass yds": h["passing_yards"],
                            "Car": h["carries"], "Rush yds": h["rushing_yards"]})
    else:
        h = R["history"]
        h = h[(h["player_id"] == r["player_id"]) & (~h["upcoming"]) & (h["season"] == season)]
        log = pd.DataFrame({"Wk": h["week"], "Opp": h["opp"], "Role": h["role"], "Snap %": h["offense_pct"],
                            "Tgt": h["targets"], "Rec": h["receptions"], "Rec yds": h["receiving_yards"],
                            "Car": h["carries"], "Rush yds": h["rushing_yards"]})
    if len(log):
        num = [c for c in log.columns if c not in ("Wk", "Opp", "Role", "Snap %")]
        avg = pd.DataFrame([{"Wk": "Avg", "Opp": "", **{c: log[c].mean() for c in num}}])
        out = pd.concat([log.sort_values("Wk").astype({"Wk": str}), avg], ignore_index=True)
        fmt = {c: "{:.0f}" for c in num}
        fmt.update({"Snap %": "{:.0%}"})
        st.dataframe(out.style.format(fmt, na_rep="—").format({c: "{:.1f}" for c in num}, subset=pd.IndexSlice[out.index[-1:], num]),
                     hide_index=True, width="stretch")
    else:
        st.caption("No games yet this season.")


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
        help="Removes the player and shifts his usage to teammates (mostly the same position); "
             "roles (WR1, WR2…) re-rank. Only mark players who are ruled out — questionable players stay in.")
    st.caption("From the official injury report: " + (", ".join(f"{n} ({t})" for n, t in zip(auto["name"], auto["team"]))
               if len(auto) else "no Out designations in nflverse yet for this week (they usually appear Friday night/Saturday)."))
    R = _projections(season, tuple(sorted(picked)))
    out_by_team = R["out"].groupby("team")["name"].apply(list).to_dict() if len(R["out"]) else {}

    # ---------------- odds
    key = _api_key()
    top = st.columns([2, 3])
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
    c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
    market = c1.radio("Market", list(pm.MARKETS), format_func=pm.MARKETS.get, horizontal=True, key="pm_market")
    book = c2.selectbox("Book", books, key="pm_book")
    teams = c3.multiselect("Teams", sorted(R["players"]["team"].unique()), key="pm_teams")
    only_lines = c4.toggle("Only players with a line", value=False, key="pm_only")
    breakdown = False
    if market in ("receptions", "rec_yards"):
        breakdown = c4.toggle("Opponent by role", value=False, key="pm_breakdown",
                              help="Add what each opponent allows per game to WR1–WR4, TE1–TE2 and RBs.")
    lines = od.consensus(wide, book) if len(wide) else pd.DataFrame()

    force = set(lines.loc[lines["market"] == market, "player_id"]) if len(lines) else set()
    df = _market_frame(R, market, force)
    df = _attach_lines(df, lines, market)

    # manual lines (session only) override pulled lines
    man_key = f"manual_{market}"
    manual = st.session_state.get(man_key)
    if manual is not None and len(manual):
        for r in manual.dropna(subset=["Line"]).itertuples(index=False):
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
    df = _probabilities(df, R["dist"])

    # opponent text + rank for non-receiving markets (rank within this week's opponents)
    if "opp_text" not in df:
        df["opp_text"] = ""
    nr = df["opp_text"].isna() | (df["opp_text"] == "")
    if nr.any():
        for grp, idx in df[nr].groupby("opp_group").groups.items():
            vals = df.loc[idx].groupby("opp")["opp_val"].first()
            rk = df.loc[idx, "opp"].map(vals.rank(ascending=False, method="min"))
            df.loc[idx, "opp_text"] = [f"{v:.1f} {lab} ({_ordinal(k)})" if pd.notna(v) else "—"
                                       for v, lab, k in zip(df.loc[idx, "opp_val"], df.loc[idx, "opp_label"], rk)]
    home = R["games"].set_index(["game_id", "team"])["home"]
    df["Opp"] = [("" if home.get((g, t), True) else "@") + o for g, t, o in zip(df["game_id"], df["team"], df["opp"])]
    df["Flags"] = [_flags(r, out_by_team) for r in df.to_dict("records")]
    df = df.sort_values(["edge", "mu"], ascending=[False, False], na_position="last").reset_index(drop=True)

    show = pd.DataFrame({
        "Player": df["Player"], "Pos": df["Pos"], "Team": df["team"], "Opp": df["Opp"],
        "Season /g": df["season_pg"], "Last 4 /g": df["l4_pg"], "Usage": df["Usage"], "Opp allows": df["opp_text"],
        "Projection": df["mu"], "Fair line": df["fair"], "Line": df["line"],
        "Over": df["over"], "Under": df["under"], "P(over)": df["p_over"], "Book P(over)": df["book_p_over"],
        "Pick": df["side"], "Edge": df["edge"], "EV": df["ev"], "Flags": df["Flags"],
    })
    if breakdown:
        dr = R["def_role"]
        stat = "receptions" if market == "receptions" else "receiving_yards"
        for role in ROLES:
            show[f"vs {role}"] = df["opp"].map(dr[stat][role]) if role in dr[stat] else np.nan
        cols = list(show.columns)
        # put the breakdown right after "Opp allows"
        bcols = [f"vs {r}" for r in ROLES]
        rest = [c for c in cols if c not in bcols]
        i = rest.index("Opp allows") + 1
        show = show[rest[:i] + bcols + rest[i:]]

    def hl(v):
        if pd.isna(v):
            return ""
        return "background-color: rgba(34,197,94,.35); font-weight:600" if v >= EDGE_MIN else ""
    fmt = {"Season /g": "{:.1f}", "Last 4 /g": "{:.1f}", "Projection": "{:.1f}", "Fair line": "{:.1f}",
           "Line": "{:.1f}", "Over": "{:+.0f}", "Under": "{:+.0f}", "P(over)": "{:.0%}", "Book P(over)": "{:.0%}",
           "Edge": "{:+.1%}", "EV": "{:+.1%}"}
    sty = show.style.format(fmt, na_rep="—").map(hl, subset=["Edge"])
    if breakdown:
        bcols = [f"vs {r}" for r in ROLES]
        sty = sty.format({c: "{:.1f}" for c in bcols}, na_rep="—").background_gradient(cmap="Greens", subset=bcols, axis=0)
    event = st.dataframe(sty, hide_index=True, width="stretch", height=640, key=f"pm_table_{market}",
                         on_select="rerun", selection_mode="single-row")
    st.caption(
        "To open a player's window (projection build-up, outcome chart, season game log), tick the checkbox at the "
        "far left of his row or pick him under the table. "
        "**Pos** = role this week among teammates expected to play (re-ranks when you rule players out). "
        "**Opp allows** = per game to his role this season, rank among 32 defenses (1st = allows the most); "
        "for rushing, rank among this week's opponents. **Fair line** = where the model sees 50/50 (below the "
        "average for yards). **Book P(over)** = book price with its margin removed. **Edge** = model minus book "
        f"on the better side, green at ≥ {EDGE_MIN:.0%}. **EV** = expected return per $1. "
        "Opponent-by-role numbers are context: in backtests they didn't improve projections.")

    # Open the player window: (a) tick the checkbox at the left edge of a row, or
    # (b) pick a name below. Either way it opens once per new pick.
    sel = event.selection.rows if event is not None and hasattr(event, "selection") else []
    names = df["Player"].tolist()
    pc1, pc2 = st.columns([3, 5])
    picked_name = pc1.selectbox("Open player window", ["—"] + names, key=f"pm_pick_{market}",
                                help="Or tick the checkbox at the far left of any row in the table.")
    if pc2.button("Open", key=f"pm_open_{market}") and picked_name != "—":
        st.session_state["pm_last_open"] = None  # force re-open of the same player
    target = None
    if sel:
        target = ("row", sel[0])
    elif picked_name != "—":
        target = ("pick", names.index(picked_name))
    if target is not None:
        i = target[1]
        token = (market, target[0], df.loc[i, "player_id"])
        if st.session_state.get("pm_last_open") != token:
            st.session_state["pm_last_open"] = token
            _player_window(R, market, df.loc[i].to_dict())
    else:
        st.session_state["pm_last_open"] = None

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
