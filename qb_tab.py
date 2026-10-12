"""Streamlit UI for the QB rushing tab. Data logic lives in qb_data.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import qb_data as q

PCT = "{:.1%}"
F1 = "{:.1f}"
F2 = "{:.2f}"


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling QB play-by-play…")
def _load(season: int, include_post: bool):
    p, players = q.load_plays(season, include_post)
    qg = q.qb_games(p, players)
    return p, players, qg


def _fmt(df: pd.DataFrame) -> dict:
    f = {}
    for c in df.columns:
        if c in ("G", "Sneaks", "adj_games", "Charted DB", "norm_games"):
            f[c] = "{:.0f}"
        elif "%" in c:
            f[c] = PCT
        elif "EPA" in c:
            f[c] = "{:+.3f}"
        elif "vs norm" in c:
            f[c] = "{:+.1f}"
        elif df[c].dtype.kind == "f":
            f[c] = F1
    return f


def _centered(sty, df, cols):
    for c in cols:
        lim = float(np.nanmax(np.abs(df[c].to_numpy(dtype=float)))) if len(df) else 1.0
        sty = sty.background_gradient(cmap="RdYlGn", subset=[c], vmin=-(lim or 1), vmax=lim or 1)
    return sty


def render(season: int, include_post: bool, weeks: tuple[int, int], nxt: pd.DataFrame):
    p_all, players, qg_all = _load(season, include_post)
    p = p_all[(p_all["week"] >= weeks[0]) & (p_all["week"] <= weeks[1])]
    qg = qg_all[(qg_all["week"] >= weeks[0]) & (qg_all["week"] <= weeks[1])]
    if qg.empty:
        st.info("No QB games in this window.")
        return

    dp = q.defense_profile(p, qg).join(q.rb_run_defense(p, players))
    traits = q.add_ranks(dp)
    qt = q.with_traits(qg, traits)
    next_opp = dict(zip(nxt["team"], nxt["next_opp"])) if len(nxt) else {}

    c1, c2, _ = st.columns([1, 1, 3])
    line = c1.number_input("Prop line (rush yds)", 0.0, 150.0, 24.5, 1.0)
    min_g = c2.number_input("Min starts", 1, 17, 2)
    st.caption(
        f"Weeks {weeks[0]}–{weeks[1]} (sidebar). Starters only (most dropbacks in the game). "
        "**Official** yards include kneels, which is how props settle; designed and "
        "scramble numbers exclude them. Splits put opponents in top/bottom half of the league "
        "on each trait over the same window.")

    sub_qb, sub_def, sub_league = st.tabs(["QB profiles", "Defense profiles", "League-wide splits"])

    # ------------------------------------------------------------------ QBs
    with sub_qb:
        tbl = q.qb_table(qg, line, int(min_g))
        if tbl.empty:
            st.info("No QBs meet the minimum starts.")
        else:
            tbl.insert(2, "Next", tbl["Team"].map(next_opp))
            tbl = tbl.sort_values("Median official", ascending=False)
            st.dataframe(tbl.style.format(_fmt(tbl), na_rep="—"), hide_index=True,
                         width="stretch", height=420)
            st.caption("L4 = his last 4 starts in the window. Sneaks are totals, not per game. "
                       "YPC ex-sneak = designed runs with sneaks removed.")

            st.divider()
            pick = st.selectbox("QB detail", tbl["QB"].tolist(), key="qb_pick")
            g = qt[qt["qb"] == pick].sort_values("week")
            team = g["posteam"].iloc[-1]
            season_s, l4_s = q.summarize(g, line), q.summarize(g.tail(4), line)

            keys = ["Designed /g", "Sneaks", "Scramble %", "Yds/scramble",
                    "Median official", f"% over {line:g}"]
            cols = st.columns(len(keys))
            for c, k in zip(cols, keys):
                v, v4 = season_s[k], l4_s[k]
                fmt = (lambda x: f"{x:.0%}") if "%" in k else (lambda x: f"{x:.1f}")
                c.metric(k, "—" if pd.isna(v) else fmt(v),
                         None if pd.isna(v4) or pd.isna(v) else f"L4: {fmt(v4)}",
                         delta_color="off")

            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**{pick}: splits** (sample size in G)")
                sp = q.splits(g, line)
                show = ["Split", "Bucket", "G", "Designed /g", "Scramble %", "Yds/scramble",
                        "Avg official", "Median official", f"% over {line:g}"]
                sp = sp[show]
                st.dataframe(sp.style.format(_fmt(sp), na_rep="—"), hide_index=True,
                             width="stretch", height=460)
            with right:
                opp = next_opp.get(team)
                if opp:
                    od = opp.lstrip("@")
                    st.markdown(f"**Next: {opp}** — {od} defense vs league")
                    if od in dp.index:
                        ranks = dp.rank(ascending=False)
                        card_cols = ["G", "QB rush yds /g", "Rush yds vs norm /g",
                                     "Designed vs norm /g", "Scrambles vs norm /g", "Scramble %",
                                     "Blitz %", "Sack+hit %", "Pass EPA/db", "RB YPC allowed",
                                     "RB rush EPA"]
                        card = pd.DataFrame({
                            "Value": dp.loc[od, card_cols],
                            "Rank (1 = most)": ranks.loc[od, card_cols],
                            "Lg median": dp[card_cols].median(),
                        })
                        disp = card.copy().astype(object)
                        for r in card.index:
                            v, m = card.loc[r, "Value"], card.loc[r, "Lg median"]
                            fmt = ("{:.0%}" if "%" in r else "{:+.3f}" if "EPA" in r
                                   else "{:+.1f}" if "vs norm" in r else "{:.1f}")
                            disp.loc[r, "Value"] = "—" if pd.isna(v) else fmt.format(v)
                            disp.loc[r, "Lg median"] = "—" if pd.isna(m) else fmt.format(m)
                            rk = card.loc[r, "Rank (1 = most)"]
                            disp.loc[r, "Rank (1 = most)"] = "" if r == "G" or pd.isna(rk) else f"{int(rk)}"
                        st.dataframe(disp, width="stretch", height=430)
                        st.caption("Rank 1 = allows the most / blitzes the most / "
                                   "worst EPA allowed (highest).")
                else:
                    st.caption("No upcoming game found.")

            st.markdown(f"**{pick}: game log**")
            log = pd.DataFrame({
                "Wk": g["week"], "Opp": np.where(g["home"], "", "@") + g["defteam"],
                "Spread": g["team_spread"], "DB": g["dropbacks"],
                "Designed": g["des_att"].astype(int).astype(str) + "/" + g["des_yds"].astype(int).astype(str),
                "Sneaks": g["sneak_att"].astype(int),
                "Scrambles": g["scr_att"].astype(int).astype(str) + "/" + g["scr_yds"].astype(int).astype(str),
                "Kneels": g["kneel_att"].astype(int).astype(str) + "/" + g["kneel_yds"].astype(int).astype(str),
                "Official": g["official_att"].astype(int).astype(str) + "/" + g["official_yds"].astype(int).astype(str),
                "Rush TD": g["rush_td"].astype(int),
                "Sack+hit": g["sack_or_hit"],
                "Blitzed": np.where(g["charted_db"] > 0,
                                    (g["blitzed"] / g["charted_db"].replace(0, np.nan)).map(
                                        lambda x: f"{x:.0%}" if pd.notna(x) else "—"), "not charted"),
                "Opp pass D": g["pass_d"], "Opp blitz": g["blitz"],
            })
            over = g["official_yds"] > line
            st.dataframe(
                log.style.apply(lambda _: np.where(over, "background-color: rgba(34,197,94,.25)", ""),
                                subset=["Official"]).format({"Spread": "{:+.1f}"}, na_rep="—"),
                hide_index=True, width="stretch")
            st.caption(f"Att/yards. Green = official yards over {line:g}. "
                       "Spread is from his team's view (negative = favored).")

    # ------------------------------------------------------------- Defenses
    with sub_def:
        show = dp.copy()
        show.insert(1, "Next", show.index.map(lambda t: next_opp.get(t, "")))
        show = show.rename(columns={"adj_games": "Adj G"})
        sort_by = st.selectbox("Sort by", ["Rush yds vs norm /g", "Scrambles vs norm /g",
                                           "Designed vs norm /g", "QB rush yds /g",
                                           "Blitz %", "Sack+hit %", "Pass EPA/db",
                                           "RB rush EPA"], key="def_sort")
        show = show.sort_values(sort_by, ascending=False).reset_index().rename(
            columns={"defteam": "Defense"})
        sty = _centered(show.style.format(_fmt(show), na_rep="—"), show,
                        ["Rush yds vs norm /g", "Scrambles vs norm /g", "Designed vs norm /g"])
        st.dataframe(sty, hide_index=True, width="stretch", height=1180)
        st.caption(
            "**vs norm** = what QBs did against this defense minus what those same QBs did in "
            "their other starts (per game). Positive/green = defense allowed more than normal. "
            "Adj G = games where the QB had another start to compare against. "
            "Scramble % is per dropback faced. Blitz % uses FTN-charted dropbacks only "
            "(the latest week can lag a few days). Sack+hit % is a pressure proxy; "
            "true pressure rate isn't in free data.")

        st.divider()
        dsel = st.selectbox("Defense detail: QBs faced", sorted(dp.index), key="def_pick")
        dg = q.add_norms(qg)
        dg = dg[dg["defteam"] == dsel].sort_values("week")
        dlog = pd.DataFrame({
            "Wk": dg["week"], "QB": dg["qb"], "Team": dg["posteam"], "DB": dg["dropbacks"],
            "Designed": dg["des_att"], "His norm (des)": dg["exp_des_att"],
            "Scrambles": dg["scr_att"], "Expected scr": dg["exp_scr_att"],
            "Rush yds ex-kneel": dg["rush_yds"], "His norm (yds)": dg["exp_rush_yds"],
            "Diff": dg["rush_yds"] - dg["exp_rush_yds"], "Official": dg["official_yds"],
            "Norm based on G": dg["norm_games"],
        })
        st.dataframe(
            _centered(dlog.style.format({"His norm (des)": F1, "Expected scr": F1,
                                         "His norm (yds)": F1, "Diff": "{:+.1f}",
                                         "Rush yds ex-kneel": "{:.0f}", "Official": "{:.0f}",
                                         "Designed": "{:.0f}", "Scrambles": "{:.0f}"},
                                        na_rep="—"), dlog, ["Diff"]),
            hide_index=True, width="stretch")
        st.caption("Expected scr = his scramble rate in other starts × dropbacks in this game. "
                   "Norms built on 1–2 other games are weak; check the last column.")

    # --------------------------------------------------------- League-wide
    with sub_league:
        st.markdown("**All starters pooled:** do these traits move QB rushing at all?")
        lw = q.splits(qt, line)
        show = ["Split", "Bucket", "G", "Designed /g", "Sneaks", "YPC ex-sneak", "Dropbacks /g",
                "Scramble %", "Yds/scramble", "Avg official", "Median official", f"% over {line:g}"]
        lw = lw[show]
        st.dataframe(lw.style.format(_fmt(lw), na_rep="—"), hide_index=True, width="stretch",
                     height=470)
        st.caption("Pooled across all QBs, so differences mix QB style with opponent effect. "
                   "Use it as a sanity check on a hunch, not proof. Treat gaps smaller than "
                   "~1 percentage point of scramble rate as noise at this sample size.")
