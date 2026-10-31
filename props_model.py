"""Production projections for player props: receptions, receiving yards, carries,
rushing yards, passing yards — for every played game (history) and each team's
next unplayed game.

Same structure and tuned constants as the backtests in model/:
  receiving : team targets/g x target share (recency 0.8, 50-target prior, half
              of a missing teammate's share redistributed) x catch rate / yds per
              target (heavily shrunk), x spread/total/opponent-volume multipliers
  rushing   : team non-QB carries/g x carry share (recency 0.7, 10-carry prior,
              half redistribution) x shrunk YPC, x spread/total/opp carries/opp YPC
  passing   : team attempts/g x starter share x QB yards/attempt (whole season +
              half of last season + quarter of the one before), x spread/total/
              opp attempts allowed/opp YPA allowed
Every feature for a game uses only earlier games, the betting lines and who is
expected to play. Coefficients come from model/coefs.json (fit on 2023-2025).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

HERE = Path(__file__).parent
COEFS = HERE / "model" / "coefs.json"
DIST = HERE / "model" / "dist_specs.json"

# tuned constants (see model/*.py)
REC = dict(decay=0.8, k_share=50, k_cr=80, k_ypt=120, w_prior=0.5, alpha=0.5)
RUSH = dict(decay=0.7, k_share=10, k_ypc=40, w_prior=0.5, alpha=0.5)
PASS = dict(decay_team=0.8, k_team=4, k_ypa=100, w_prior=0.5)
QBR = dict(k_games=1, k_ypc=20)
# league constants from 2021-2022 (pre-backtest), used as shrinkage targets
POS_CR = {"WR": 0.632, "TE": 0.688, "RB": 0.772}
POS_YPT = {"WR": 7.91, "TE": 7.26, "RB": 5.74}
RB_YPC = 4.324
POS = ["WR", "TE", "RB", "FB"]
MARKETS = {
    "receptions": "Receptions", "rec_yards": "Receiving yards", "carries": "Rush attempts",
    "rush_yards": "Rushing yards", "pass_yards": "Passing yards",
}


# ==========================================================================
# helpers
# ==========================================================================
def _pos(p):
    return "RB" if p in ("RB", "FB") else p


def _decayed(df: pd.DataFrame, col: str, lam: float, keys: list[str]) -> np.ndarray:
    """Sum of earlier values in the group, each older game weighted lam^k."""
    out = np.zeros(len(df)); v = df[col].to_numpy(float)
    for idx in df.groupby(keys, sort=False).indices.values():
        a = 0.0
        for i in idx:
            out[i] = a; a = a * lam + v[i]
    return out


def norm_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


# ==========================================================================
# raw data
# ==========================================================================
def load_raw(seasons: list[int], current: int | None = None) -> dict:
    ps = (nfl.load_player_stats(seasons).filter(pl.col("season_type") == "REG")
          .select(["player_id", "player_display_name", "position", "season", "week", "game_id", "team",
                   "targets", "receptions", "receiving_yards", "carries", "rushing_yards",
                   "attempts", "passing_yards"]).to_pandas())
    sn = nfl.load_snap_counts([s for s in seasons if s >= min(seasons)]).filter(pl.col("game_type") == "REG").to_pandas()
    sch = nfl.load_schedules(seasons).filter(pl.col("game_type") == "REG").to_pandas()
    players = nfl.load_players().select(["gsis_id", "pfr_id", "display_name", "position"]).to_pandas()
    inj = None
    if current is not None:
        try:
            inj = nfl.load_injuries([current]).to_pandas()
        except Exception:
            inj = None
    return {"ps": ps, "sn": sn, "sch": sch, "players": players, "inj": inj}


def _games(sch: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with lines from the team's view."""
    a = sch[["game_id", "season", "week", "gameday", "home_team", "away_team", "spread_line", "total_line",
             "result", "home_qb_id", "away_qb_id"]]
    h = a.rename(columns={"home_team": "team", "away_team": "opp", "home_qb_id": "qb_id"}).assign(home=True)
    w = a.rename(columns={"away_team": "team", "home_team": "opp", "away_qb_id": "qb_id"}).assign(home=False)
    g = pd.concat([h, w], ignore_index=True).drop(columns=["away_qb_id", "home_qb_id"], errors="ignore")
    g["team_spread"] = np.where(g["home"], -g["spread_line"], g["spread_line"])  # negative = favored
    g["played"] = g["result"].notna()
    return g


def next_games(sch: pd.DataFrame) -> pd.DataFrame:
    g = _games(sch)
    up = g[~g["played"]].sort_values(["gameday", "week"])
    return up.drop_duplicates("team")


# ==========================================================================
# player-game universe (history + upcoming)
# ==========================================================================
def candidates(raw: dict) -> pd.DataFrame:
    """Players the upcoming projections would include (before any Out removals)."""
    raw2 = {**raw, "manual_out": (), "inj": None}
    u, _ = _universe(raw2, True)
    return u[u["upcoming"]][["player_id", "name", "position", "team"]]


def _universe(raw: dict, include_upcoming: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ps, sn, players = raw["ps"], raw["sn"], raw["players"]
    pfr2gsis = players.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    sn = sn[sn["position"].isin(POS) & (sn["offense_snaps"] > 0)].copy()
    sn["player_id"] = sn["pfr_player_id"].map(pfr2gsis)
    sn = sn.dropna(subset=["player_id"])
    u = sn[["game_id", "season", "week", "team", "player_id", "player", "position", "offense_pct"]].copy()
    stats_cols = ["targets", "receptions", "receiving_yards", "carries", "rushing_yards"]
    u = u.merge(ps[["player_id", "game_id", "player_display_name"] + stats_cols], on=["player_id", "game_id"], how="left")
    u[stats_cols] = u[stats_cols].fillna(0)
    u["name"] = u["player_display_name"].fillna(u["player"])
    u = u.drop(columns=["player_display_name", "player"])
    u["upcoming"] = False
    u["inj_status"] = None

    # team-game totals (all positions; non-QB carries)
    tt = ps.groupby(["game_id", "team"]).agg(team_targets=("targets", "sum"), team_att=("attempts", "sum"),
                                             team_pyds=("passing_yards", "sum")).reset_index()
    nq = (ps[ps["position"] != "QB"].groupby(["game_id", "team"])
          .agg(team_carries=("carries", "sum"), team_rush_yds=("rushing_yards", "sum")).reset_index())
    G = _games(raw["sch"]).merge(tt, on=["game_id", "team"], how="left").merge(nq, on=["game_id", "team"], how="left")
    for c in ["team_targets", "team_att", "team_pyds", "team_carries", "team_rush_yds"]:
        G[c] = G[c].fillna(0)

    if include_upcoming:
        cur = raw["sch"]["season"].max()
        up = next_games(raw["sch"][raw["sch"]["season"] == cur])
        rows, excluded = [], []
        inj = raw.get("inj")
        for r in up.itertuples(index=False):
            hist = u[(u["team"] == r.team) & (u["season"] == cur)]
            if hist.empty:  # no games yet this season: last season's regulars on this team
                hist = u[(u["team"] == r.team) & (u["season"] == cur - 1)]
                recent = hist.groupby("player_id").size()
                cand = recent[recent >= 4].index
                base = hist.drop_duplicates("player_id", keep="last")
            else:
                last_weeks = sorted(hist["week"].unique())[-3:]
                cand = hist[hist["week"].isin(last_weeks)]["player_id"].unique()
                base = hist.drop_duplicates("player_id", keep="last")
            b = base[base["player_id"].isin(cand)][["player_id", "name", "position"]].copy()
            status = {}
            if inj is not None and len(inj):
                ii = inj[(inj["season"] == cur) & (inj["week"] == r.week) & (inj["team"] == r.team)]
                status = dict(zip(ii["gsis_id"], ii["report_status"]))
            b["inj_status"] = b["player_id"].map(status).astype(object)
            manual = b["player_id"].isin(raw.get("manual_out", ()))
            b.loc[manual, "inj_status"] = "Out (you)"
            is_out = b["inj_status"].isin(["Out", "Out (you)"])
            excluded.append(b[is_out].assign(team=r.team))
            b = b[~is_out]
            b = b.assign(game_id=r.game_id, season=cur, week=r.week, team=r.team, offense_pct=np.nan,
                         upcoming=True, **{c: 0.0 for c in stats_cols})
            rows.append(b)
        if rows:
            u = pd.concat([u, pd.concat(rows, ignore_index=True)], ignore_index=True)
        raw["excluded"] = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame()
    G = G[G["played"] | G["game_id"].isin(u.loc[u["upcoming"], "game_id"])]
    u = u.merge(G[["game_id", "team", "opp", "home", "team_spread", "total_line", "team_targets", "team_carries"]],
                on=["game_id", "team"], how="left")
    u["pos"] = u["position"].map(_pos)
    return u, G


# ==========================================================================
# team & opponent context
# ==========================================================================
def _team_context(G: pd.DataFrame) -> pd.DataFrame:
    G = G.sort_values(["team", "season", "week"]).reset_index(drop=True)
    g = G.groupby(["team", "season"])
    out = G[["game_id", "team", "opp", "season", "week", "played"]].copy()
    for col, key, k in [("team_targets", "tpg", 4), ("team_carries", "cpg", 4)]:
        played = G[col].where(G["played"], 0)
        cum = played.groupby([G["team"], G["season"]]).cumsum() - played
        n = G["played"].astype(int).groupby([G["team"], G["season"]]).cumsum() - G["played"].astype(int)
        pm = G[G["played"]].groupby(["team", "season"])[col].mean().rename("p").reset_index(); pm["season"] += 1
        p = G[["team", "season"]].merge(pm, on=["team", "season"], how="left")["p"]
        lg = G[G["played"]].groupby("season")[col].mean()
        lgv = G["season"].map(lg.shift(1)).fillna(lg.mean())
        prior = 0.5 * p.fillna(lgv).to_numpy() + 0.5 * lgv.to_numpy()
        out["est_team_" + key] = (cum.to_numpy() + k * prior) / (n.to_numpy() + k)
        out["lg_" + key] = lgv.to_numpy()
    # passing attempts: recency-weighted
    Gp = G.assign(att=G["team_att"].where(G["played"], 0), one=G["played"].astype(float))
    d_att = _decayed(Gp, "att", PASS["decay_team"], ["team", "season"])
    d_n = _decayed(Gp, "one", PASS["decay_team"], ["team", "season"])
    pm = G[G["played"]].groupby(["team", "season"])["team_att"].mean().rename("p").reset_index(); pm["season"] += 1
    p = G[["team", "season"]].merge(pm, on=["team", "season"], how="left")["p"]
    lg = G[G["played"]].groupby("season")["team_att"].mean(); lga = G["season"].map(lg.shift(1)).fillna(lg.mean())
    out["est_team_att"] = (d_att + PASS["k_team"] * (0.5 * p.fillna(lga) + 0.5 * lga).to_numpy()) / (d_n + PASS["k_team"])
    out["lg_att"] = lga.to_numpy()
    lgy = G[G["played"]].groupby("season")["team_pyds"].sum() / G[G["played"]].groupby("season")["team_att"].sum()
    out["lg_ypa"] = G["season"].map(lgy.shift(1)).fillna(lgy.mean()).to_numpy()
    lgr = G[G["played"]].groupby("season")["team_rush_yds"].sum() / G[G["played"]].groupby("season")["team_carries"].sum()
    out["lg_ypc"] = G["season"].map(lgr.shift(1)).fillna(lgr.mean()).to_numpy()

    # what each DEFENSE allowed in earlier games (offense rows where opp == defense)
    Dd = G.rename(columns={"opp": "defteam"}).sort_values(["defteam", "season", "week"]).reset_index(drop=True)
    pl_ = Dd["played"].astype(int)
    grp = [Dd["defteam"], Dd["season"]]
    cnt = pl_.groupby(grp).cumsum() - pl_
    dd = Dd[["game_id", "defteam", "season"]].copy()
    for c in ["team_targets", "team_carries", "team_rush_yds", "team_att", "team_pyds"]:
        v = Dd[c].where(Dd["played"], 0)
        dd["d_" + c] = v.groupby(grp).cumsum() - v
    dd["d_n"] = cnt
    pv = Dd[Dd["played"]].groupby(["defteam", "season"]).agg(pa=("team_att", "sum"), pn=("team_att", "size"),
                                                            py=("team_pyds", "sum")).reset_index(); pv["season"] += 1
    dd = dd.merge(pv, on=["defteam", "season"], how="left").fillna({"pa": 0, "pn": 0, "py": 0})
    out = out.merge(dd.drop(columns="season"), left_on=["game_id", "opp"], right_on=["game_id", "defteam"], how="left")
    out["opp_tgt_ratio"] = ((out["d_team_targets"] + 6 * out["lg_tpg"]) / (out["d_n"] + 6)) / out["lg_tpg"]
    out["opp_car_ratio"] = ((out["d_team_carries"] + 6 * out["lg_cpg"]) / (out["d_n"] + 6)) / out["lg_cpg"]
    out["opp_ypc_ratio"] = ((out["d_team_rush_yds"] + 150 * out["lg_ypc"]) / (out["d_team_carries"] + 150)) / out["lg_ypc"]
    out["opp_att_ratio"] = ((out["d_team_att"] + out["pa"] / 3 + 6 * out["lg_att"]) / (out["d_n"] + out["pn"] / 3 + 6)) / out["lg_att"]
    out["opp_ypa_ratio"] = ((out["d_team_pyds"] + out["py"] / 3 + 250 * out["lg_ypa"])
                            / (out["d_team_att"] + out["pa"] / 3 + 250)) / out["lg_ypa"]
    # plain per-game numbers for display
    out["opp_pass_yds_pg"] = out["d_team_pyds"] / out["d_n"].replace(0, np.nan)
    out["opp_rush_yds_pg"] = out["d_team_rush_yds"] / out["d_n"].replace(0, np.nan)
    out["opp_targets_pg"] = out["d_team_targets"] / out["d_n"].replace(0, np.nan)
    out["opp_carries_pg"] = out["d_team_carries"] / out["d_n"].replace(0, np.nan)
    out["opp_ypc"] = out["d_team_rush_yds"] / out["d_team_carries"].replace(0, np.nan)
    out["opp_ypa"] = out["d_team_pyds"] / out["d_team_att"].replace(0, np.nan)
    return out.drop(columns=["defteam", "opp", "season", "week", "played"])


# ==========================================================================
# receiving + rushing features
# ==========================================================================
def _redistribute(u: pd.DataFrame, k: float = 60) -> np.ndarray:
    """Target share after accounting for who is missing.

    For each team-game: the team's usual target share by position (earlier games
    this season, shrunk toward league) minus what the active players at that
    position normally get = targets 'missing' at that position. Half of that
    (alpha) goes to active teammates at the same position, in proportion to their
    shares; half of any remaining shortfall is spread across everyone. If the
    actives already add up to more than the team total, all are scaled down.
    Backtest (model/redistribute.py): beats proportional-to-everyone by ~2% in
    games with a teammate out."""
    a = REC["alpha"]
    tg = (u.assign(pos=u["pos"].fillna("RB"))
          .groupby(["season", "week", "game_id", "team", "pos"])["targets"].sum().unstack("pos").fillna(0))
    tg = tg.join(u.groupby(["season", "week", "game_id", "team"])["team_targets"].first()).reset_index()
    tg = tg.sort_values(["team", "season", "week"])
    for p in ["WR", "TE", "RB"]:
        if p not in tg:
            tg[p] = 0.0
    lg = {p: tg[p].sum() / max(tg["team_targets"].sum(), 1) for p in ["WR", "TE", "RB"]}
    gg = tg.groupby(["team", "season"])
    ctt = gg["team_targets"].cumsum() - tg["team_targets"]
    S = []
    for p in ["WR", "TE", "RB"]:
        cum = gg[p].cumsum() - tg[p]
        S.append(tg[["game_id", "team"]].assign(pos=p, S_pos=((cum + k * lg[p]) / (ctt + k)).to_numpy()))
    S = pd.concat(S)
    x = u[["game_id", "team", "pos", "est_tshare"]].merge(S, on=["game_id", "team", "pos"], how="left")
    x["S_pos"] = x["S_pos"].fillna(0)
    x["act_pos"] = x.groupby(["game_id", "team", "pos"])["est_tshare"].transform("sum")
    x["act_tot"] = x.groupby(["game_id", "team"])["est_tshare"].transform("sum")
    x["miss_pos"] = (x["S_pos"] - x["act_pos"]).clip(lower=0)
    per_pos = x.drop_duplicates(["game_id", "team", "pos"]).groupby(["game_id", "team"])["miss_pos"].sum()
    x = x.merge(per_pos.rename("miss_pos_sum").reset_index(), on=["game_id", "team"], how="left")
    miss_tot = (0.995 - x["act_tot"]).clip(lower=0)
    rest = (miss_tot - x["miss_pos_sum"]).clip(lower=0)
    over = (x["act_tot"] - 0.995).clip(lower=0)
    sh = x["est_tshare"]
    out = (sh + a * (x["miss_pos"] * sh / x["act_pos"].replace(0, np.nan)).fillna(0)
           + a * rest * sh / x["act_tot"] - a * sh * over / x["act_tot"])
    return out.to_numpy()


def _player_features(u: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    u = u.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = u.groupby(["player_id", "season"])
    for c in ["targets", "receptions", "receiving_yards", "carries", "rushing_yards", "team_targets", "team_carries"]:
        u["c_" + c] = g[c].cumsum() - u[c]
    u["c_g"] = g.cumcount()
    u["l4_targets"] = g["targets"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    u["l4_receptions"] = g["receptions"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    u["l4_receiving_yards"] = g["receiving_yards"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    u["l4_carries"] = g["carries"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    u["l4_rushing_yards"] = g["rushing_yards"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    u["d_tgt"] = _decayed(u, "targets", REC["decay"], ["player_id", "season"])
    u["d_tt"] = _decayed(u, "team_targets", REC["decay"], ["player_id", "season"])
    u["d_car"] = _decayed(u, "carries", RUSH["decay"], ["player_id", "season"])
    u["d_tc"] = _decayed(u, "team_carries", RUSH["decay"], ["player_id", "season"])

    hist = u[~u["upcoming"]]
    ps = hist.groupby(["player_id", "season"]).agg(
        p_g=("game_id", "size"), p_tgt=("targets", "sum"), p_rec=("receptions", "sum"),
        p_yds=("receiving_yards", "sum"), p_tt=("team_targets", "sum"), p_car=("carries", "sum"),
        p_ryds=("rushing_yards", "sum"), p_tc=("team_carries", "sum"), p_team=("team", "last")).reset_index()
    ps["season"] += 1
    u = u.merge(ps, on=["player_id", "season"], how="left")
    for c in ["p_g", "p_tgt", "p_rec", "p_yds", "p_tt", "p_car", "p_ryds", "p_tc"]:
        u[c] = u[c].fillna(0)
    has = u["p_g"] >= 4

    # ---- receiving
    p_share = (u["p_tgt"] / u["p_tt"].replace(0, np.nan)).where(has)
    fb = np.where(u["pos"] == "WR", 0.08, 0.06)
    prior = p_share.fillna(pd.Series(fb, index=u.index))
    kp = np.where(has, REC["k_share"], 25)
    u["est_tshare"] = (u["d_tgt"] + kp * prior) / (u["d_tt"] + kp)
    u["tshare_adj"] = _redistribute(u)
    w = REC["w_prior"]
    n = u["c_targets"] + w * u["p_tgt"]
    u["est_cr"] = (u["c_receptions"] + w * u["p_rec"] + REC["k_cr"] * u["pos"].map(POS_CR)) / (n + REC["k_cr"])
    u["est_ypt"] = (u["c_receiving_yards"] + w * u["p_yds"] + REC["k_ypt"] * u["pos"].map(POS_YPT)) / (n + REC["k_ypt"])

    # ---- rushing
    c_share = (u["p_car"] / u["p_tc"].replace(0, np.nan)).where(has)
    fbr = np.where(u["pos"] == "RB", 0.10, 0.005)
    prior_r = c_share.fillna(pd.Series(fbr, index=u.index))
    kr = np.where(has, RUSH["k_share"], 20)
    u["est_cshare"] = (u["d_car"] + kr * prior_r) / (u["d_tc"] + kr)
    totr = u.groupby(["game_id", "team"])["est_cshare"].transform("sum")
    u["cshare_adj"] = u["est_cshare"] * (1 + RUSH["alpha"] * (1.0 / totr - 1))
    nr = u["c_carries"] + RUSH["w_prior"] * u["p_car"]
    u["est_ypc"] = (u["c_rushing_yards"] + RUSH["w_prior"] * u["p_ryds"] + RUSH["k_ypc"] * RB_YPC) / (nr + RUSH["k_ypc"])

    # teammate-out signal (share of this team's season-to-date targets/carries not on the field)
    act_t = u.groupby(["game_id", "team"])["est_tshare"].transform("sum")
    u["vacated_tgt"] = (1 - act_t).clip(lower=0)

    u = u.merge(T, on=["game_id", "team"], how="left")
    u["E_tgt"] = u["tshare_adj"] * u["est_team_tpg"]
    u["E_car0"] = u["cshare_adj"] * u["est_team_cpg"]
    u["tot_c"] = u["total_line"] - 44.5
    u["new_team"] = u["p_team"].notna() & (u["p_team"] != u["team"]) & (u["c_g"] < 4)
    u["role"] = assign_roles(u)
    return u


ROLE_ORDER = ["WR1", "WR2", "WR3", "WR4", "TE1", "TE2", "RB"]


def assign_roles(u: pd.DataFrame) -> pd.Series:
    """Pre-game role within each team-game among players expected to play:
    WR1-WR4 (4th+ = WR4) and TE1-TE2 by target share; RB1/RB2 by carry share.
    Upcoming games use shares after ruled-out players are removed, so roles move up."""
    rk_t = u.groupby(["game_id", "team", "pos"])["tshare_adj"].rank(ascending=False, method="first")
    rk_c = u.groupby(["game_id", "team", "pos"])["cshare_adj"].rank(ascending=False, method="first")
    return pd.Series(np.select(
        [u["pos"] == "WR", u["pos"] == "TE", u["pos"] == "RB"],
        ["WR" + rk_t.clip(upper=4).fillna(4).astype(int).astype(str),
         "TE" + rk_t.clip(upper=2).fillna(2).astype(int).astype(str),
         "RB" + rk_c.clip(upper=2).fillna(2).astype(int).astype(str)], default=u["pos"].fillna("")), index=u.index)


def defense_by_role(P: pd.DataFrame, G: pd.DataFrame, season: int) -> pd.DataFrame:
    """Per defense, this season so far: targets / receptions / yards allowed per game
    to each role (WR1-4, TE1-2, RB = all backs), plus rank (1 = allows the most)."""
    h = P[(~P["upcoming"]) & (P["season"] == season)].copy()
    h["drole"] = np.where(h["role"].str.startswith("RB"), "RB", h["role"])
    agg = h.groupby(["opp", "drole"])[["targets", "receptions", "receiving_yards"]].sum()
    n = G[G["played"] & (G["season"] == season)].groupby("opp").size()  # games each defense has played
    agg = agg.div(n, level="opp", axis=0).unstack("drole").fillna(0)
    agg = agg.reindex(columns=pd.MultiIndex.from_product([["targets", "receptions", "receiving_yards"], ROLE_ORDER]),
                      fill_value=0)
    ranks = agg.rank(ascending=False, method="min")
    ranks.columns = pd.MultiIndex.from_tuples([(f"rank_{a}", b) for a, b in ranks.columns])
    return pd.concat([agg, ranks], axis=1)


# ==========================================================================
# passing features
# ==========================================================================
def _qb_features(raw: dict, G: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    ps = raw["ps"]
    qb_all = ps[ps["attempts"] > 0][["player_id", "player_display_name", "game_id", "season", "week", "team",
                                     "attempts", "passing_yards"]]
    Q = G[["game_id", "season", "week", "team", "opp", "qb_id", "played", "team_spread", "total_line", "team_att"]].copy()
    nxt = next_games(raw["sch"][raw["sch"]["season"] == raw["sch"]["season"].max()])
    keep_next = set(zip(nxt["team"], nxt["game_id"]))
    Q = Q[Q["played"].to_numpy() | np.array([(t, g) in keep_next for t, g in zip(Q["team"], Q["game_id"])], dtype=bool)]
    # upcoming starter missing in schedule -> last game's starter
    last = (Q[Q["played"]].sort_values(["season", "week"]).groupby("team")["qb_id"].last())
    Q["qb_id"] = Q["qb_id"].fillna(Q["team"].map(last))
    Q = Q.merge(qb_all[["player_id", "game_id", "attempts", "passing_yards"]].rename(columns={"player_id": "qb_id"}),
                on=["qb_id", "game_id"], how="left")
    Q[["attempts", "passing_yards"]] = Q[["attempts", "passing_yards"]].fillna(0)
    Q = Q[(~Q["played"]) | (Q["attempts"] > 0)].copy()  # listed starter who never threw = voided prop
    names = qb_all.drop_duplicates("player_id", keep="last").set_index("player_id")["player_display_name"]
    pnames = raw["players"].drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]
    Q["name"] = Q["qb_id"].map(names).fillna(Q["qb_id"].map(pnames))

    # QB history: all his passing before this game (this season) + last two seasons
    A = qb_all.sort_values(["player_id", "season", "week"])
    cum = A.groupby(["player_id", "season"])[["attempts", "passing_yards"]].cumsum()
    A = A.assign(c_att=cum["attempts"] - A["attempts"], c_yds=cum["passing_yards"] - A["passing_yards"],
                 c_g=A.groupby(["player_id", "season"]).cumcount())
    tot = A.groupby(["player_id", "season"]).agg(s_att=("attempts", "sum"), s_yds=("passing_yards", "sum"),
                                                  s_g=("attempts", "size")).reset_index()
    Q = Q.merge(A[["player_id", "game_id", "c_att", "c_yds", "c_g"]].rename(columns={"player_id": "qb_id"}),
                on=["qb_id", "game_id"], how="left")
    # upcoming game: season-to-date = full season so far
    cur_tot = tot.rename(columns={"player_id": "qb_id"})
    Q = Q.merge(cur_tot.rename(columns={"s_att": "cs_att", "s_yds": "cs_yds", "s_g": "cs_g"}), on=["qb_id", "season"], how="left")
    up = ~Q["played"]
    for a, b in [("c_att", "cs_att"), ("c_yds", "cs_yds"), ("c_g", "cs_g")]:
        Q.loc[up, a] = Q.loc[up, b]
    for k, sfx in [(1, "1"), (2, "2")]:
        t = cur_tot.assign(season=cur_tot["season"] + k).rename(columns={"s_att": "a" + sfx, "s_yds": "y" + sfx, "s_g": "g" + sfx})
        Q = Q.merge(t, on=["qb_id", "season"], how="left")
    for c in ["c_att", "c_yds", "c_g", "a1", "y1", "g1", "a2", "y2", "g2"]:
        Q[c] = Q[c].fillna(0)
    Q = Q.merge(T[["game_id", "team", "est_team_att", "lg_ypa", "opp_att_ratio", "opp_ypa_ratio",
                   "opp_pass_yds_pg", "opp_ypa"]], on=["game_id", "team"], how="left")
    hist_att = Q["c_att"] + Q["a1"] + Q["a2"]
    prior = Q["lg_ypa"] * np.where(hist_att >= 300, 1.0, 0.93)
    w = PASS["w_prior"]
    Q["est_ypa"] = ((Q["c_yds"] + w * Q["y1"] + w / 2 * Q["y2"] + PASS["k_ypa"] * prior)
                    / (Q["c_att"] + w * Q["a1"] + w / 2 * Q["a2"] + PASS["k_ypa"]))
    sh = (Q[Q["played"]]["attempts"] / Q[Q["played"]]["team_att"].replace(0, np.nan)).groupby(Q[Q["played"]]["season"]).mean()
    Q["est_share"] = Q["season"].map(sh.shift(1)).fillna(sh.mean() if len(sh) else 0.945)
    Q["E_pass0"] = Q["est_team_att"] * Q["est_share"] * Q["est_ypa"]
    Q["tot_c"] = Q["total_line"] - 44.5
    Q["ypg_season"] = Q["c_yds"] / Q["c_g"].replace(0, np.nan)
    Q["att_pg_season"] = Q["c_att"] / Q["c_g"].replace(0, np.nan)
    Q["hist_att"] = hist_att
    Q = Q.sort_values(["qb_id", "season", "week"]).reset_index(drop=True)
    Q["l4_pass_yds"] = Q.groupby(["qb_id", "season"])["passing_yards"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean())

    # ---- QB rushing (backtest model/qb_rush.py): his carries per start this season,
    # shrunk toward his last two seasons (worth up to 8 games), x shrunk QB YPC
    qr = ps[ps["position"] == "QB"][["player_id", "game_id", "season", "carries", "rushing_yards"]]
    Q = Q.merge(qr.drop(columns="season").rename(columns={"player_id": "qb_id"}), on=["qb_id", "game_id"], how="left")
    Q[["carries", "rushing_yards"]] = Q[["carries", "rushing_yards"]].fillna(0)
    gq = Q.groupby(["qb_id", "season"])
    Q["c_qcar"] = gq["carries"].cumsum() - Q["carries"]
    Q["c_qyds"] = gq["rushing_yards"].cumsum() - Q["rushing_yards"]
    Q["c_qg"] = gq.cumcount()
    tq = qr.groupby(["player_id", "season"]).agg(g=("game_id", "size"), car=("carries", "sum"),
                                                 yds=("rushing_yards", "sum")).reset_index()
    for k in (1, 2):
        t = tq.assign(season=tq["season"] + k).rename(columns={"player_id": "qb_id", "g": f"qg{k}", "car": f"qc{k}", "yds": f"qy{k}"})
        Q = Q.merge(t, on=["qb_id", "season"], how="left")
    for c in ["qg1", "qc1", "qy1", "qg2", "qc2", "qy2"]:
        Q[c] = Q[c].fillna(0)
    pl_ = Q[Q["played"]]
    lg_cpg = pl_.groupby("season")["carries"].mean()
    lg_ypc = pl_.groupby("season")["rushing_yards"].sum() / pl_.groupby("season")["carries"].sum()
    lgc = Q["season"].map(lg_cpg.shift(1)).fillna(lg_cpg.mean() if len(lg_cpg) else 4.0)
    lgy = Q["season"].map(lg_ypc.shift(1)).fillna(lg_ypc.mean() if len(lg_ypc) else 4.5)
    pg_ = Q["qg1"] + 0.5 * Q["qg2"]
    prior = ((Q["qc1"] + 0.5 * Q["qc2"]) / pg_.replace(0, np.nan)).where(Q["qg1"] + Q["qg2"] >= 6).fillna(lgc)
    pw = pg_.clip(upper=8)
    Q["E_qcar"] = (Q["c_qcar"] + (QBR["k_games"] + pw) * prior) / (Q["c_qg"] + QBR["k_games"] + pw)
    Q["est_qypc"] = ((Q["c_qyds"] + Q["qy1"] + 0.5 * Q["qy2"] + QBR["k_ypc"] * lgy)
                     / (Q["c_qcar"] + Q["qc1"] + 0.5 * Q["qc2"] + QBR["k_ypc"]))
    Q["qcar_pg_season"] = Q["c_qcar"] / Q["c_qg"].replace(0, np.nan)
    Q["qrush_pg_season"] = Q["c_qyds"] / Q["c_qg"].replace(0, np.nan)
    gq = Q.groupby(["qb_id", "season"])
    Q["l4_qcar"] = gq["carries"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    Q["l4_qrush"] = gq["rushing_yards"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    return Q


# ==========================================================================
# apply coefficients
# ==========================================================================
def _mult(df, coefs, cols):
    """exp(sum b * x), with *_ratio columns entering as logs."""
    z = np.zeros(len(df))
    for c, b in zip(cols, coefs):
        x = df[c].to_numpy(float)
        z += b * (np.log(x) if c.endswith("_ratio") else x)
    return np.exp(z)


SPEC = {  # market: (base column(s), multiplier columns)
    "receptions": (["E_tgt", "est_cr"], ["team_spread", "tot_c", "opp_tgt_ratio"]),
    "rec_yards": (["E_tgt", "est_ypt"], ["team_spread", "tot_c", "opp_tgt_ratio"]),
    "carries": (["E_car0"], ["team_spread", "tot_c", "opp_car_ratio"]),
    "rush_yards": (["E_car0", "est_ypc"], ["team_spread", "tot_c", "opp_car_ratio", "opp_ypc_ratio"]),
    "pass_yards": (["E_pass0"], ["team_spread", "tot_c", "opp_att_ratio", "opp_ypa_ratio"]),
    "qb_carries": (["E_qcar"], ["team_spread", "tot_c"]),
    "qb_rush_yards": (["E_qcar", "est_qypc"], ["team_spread", "tot_c"]),
}


def apply(df: pd.DataFrame, market: str, coefs: list[float]) -> np.ndarray:
    base_cols, mcols = SPEC[market]
    base = np.prod([df[c].to_numpy(float) for c in base_cols], axis=0)
    return base * _mult(df, coefs, mcols)


def fit_coefs(df: pd.DataFrame, market: str, y: str) -> list[float]:
    from scipy.optimize import minimize
    base_cols, mcols = SPEC[market]
    d = df.dropna(subset=base_cols + mcols + [y])
    Y = d[y].to_numpy(float)

    def loss(z):
        m = np.maximum(apply(d, market, z), 1e-6)
        return np.mean(m - Y * np.log(m))
    return minimize(loss, np.zeros(len(mcols)), method="L-BFGS-B").x.tolist()


# ==========================================================================
# distributions
# ==========================================================================
def p_over(spec: dict, m, L):
    z = spec["z"]; m = np.asarray(m, float); L = np.asarray(L, float)
    m = np.maximum(m, 0.05)
    t = spec["type"]
    if t == "nb":
        th = np.exp(z[0] + z[1] * np.log(m)); return stats.nbinom.sf(np.floor(L), th, th / (th + m))
    if t == "dnorm":
        sd = np.exp(z[0]) * m ** z[1]; return stats.norm.sf(np.floor(L) + 0.5, m, sd)
    if t == "normal":
        return stats.norm.sf(L, m, np.exp(z[0]) * m ** z[1])
    a, b, k0, k1 = z  # zero-inflated gamma, mean tied to projection
    pi = 1 / (1 + np.exp(-(a + b * np.log(m)))); k = np.exp(k0 + k1 * np.log(m)); mean = m / (1 - pi)
    return np.where(L < 0, 1.0, (1 - pi) * stats.gamma.sf(np.maximum(L, 0), k, scale=mean / k))


def fair_line(spec: dict, m: float, count: bool) -> float:
    """Line (x.5) closest to a 50/50 split."""
    if count:
        grid = np.arange(-0.5, max(m * 3, 6) + 1, 1.0)
    else:
        grid = np.arange(-0.5, max(m * 3, 30) + 1, 1.0)
    p = p_over(spec, np.full(len(grid), m), grid)
    return float(grid[np.argmin(np.abs(p - 0.5))])


def distribution(spec: dict, m: float, count: bool, hi: float | None = None):
    """(x, probability) for charting: pmf for counts, density-ish 5-yd bins for yards."""
    hi = hi or max(m * 2.6, 10)
    if count:
        x = np.arange(0, int(hi) + 1)
        sf_prev = p_over(spec, np.full(len(x), m), x - 0.5)
        sf_next = p_over(spec, np.full(len(x), m), x + 0.5)
        return x, sf_prev - sf_next
    edges = np.arange(0, hi + 5, 5.0)
    sf = p_over(spec, np.full(len(edges), m), edges)
    return edges[:-1], sf[:-1] - sf[1:]


# ==========================================================================
# top-level
# ==========================================================================
def build(seasons: list[int], include_upcoming: bool, manual_out: tuple = (), raw: dict | None = None) -> dict:
    raw = raw or load_raw(seasons, current=max(seasons) if include_upcoming else None)
    raw = {**raw, "manual_out": tuple(manual_out)}
    u, G = _universe(raw, include_upcoming)
    T = _team_context(G)
    P = _player_features(u, T)
    Q = _qb_features(raw, G, T)
    return {"players": P, "qbs": Q, "games": G, "T": T, "excluded": raw.get("excluded", pd.DataFrame())}


def project(seasons: list[int], manual_out: tuple = (), raw: dict | None = None) -> dict:
    """Projections + distribution specs for each team's next game.
    manual_out: player ids you've marked as ruled out (on top of the injury report)."""
    coefs = json.loads(COEFS.read_text())
    dist = json.loads(DIST.read_text())
    B = build(seasons, include_upcoming=True, manual_out=manual_out, raw=raw)
    P, Q = B["players"], B["qbs"]
    up = P[P["upcoming"]].copy()
    for mk in ["receptions", "rec_yards", "carries", "rush_yards"]:
        up["mu_" + mk] = apply(up, mk, coefs[mk])
    uq = Q[~Q["played"]].copy()
    uq["mu_pass_yards"] = apply(uq, "pass_yards", coefs["pass_yards"])
    uq["mu_carries"] = apply(uq, "qb_carries", coefs["qb_carries"]) if "qb_carries" in coefs else uq["E_qcar"]
    uq["mu_rush_yards"] = (apply(uq, "qb_rush_yards", coefs["qb_rush_yards"]) if "qb_rush_yards" in coefs
                           else uq["E_qcar"] * uq["est_qypc"])
    # teammates ruled out (Out/Doubtful) who carried real volume this season
    ex = B["excluded"]
    if len(ex):
        cur = P["season"].max()
        h = P[(~P["upcoming"]) & (P["season"] == cur)]
        a = h.groupby(["player_id", "team"])[["targets", "team_targets", "carries", "team_carries"]].sum().reset_index()
        a["tshare"] = a["targets"] / a["team_targets"].clip(lower=1)
        a["cshare"] = a["carries"] / a["team_carries"].clip(lower=1)
        ex = ex.merge(a[["player_id", "team", "tshare", "cshare"]], on=["player_id", "team"], how="left")
        ex = ex[(ex["tshare"] >= 0.12) | (ex["cshare"] >= 0.25)]
    else:
        ex = pd.DataFrame(columns=["player_id", "name", "position", "inj_status", "team", "tshare", "cshare"])
    cur = P["season"].max()
    # QB rushing allowed per game by each defense this season (starters)
    qh = Q[Q["played"] & (Q["season"] == cur)]
    qdef = qh.groupby("opp").agg(qb_car_pg=("carries", "mean"), qb_rush_pg=("rushing_yards", "mean"))
    return {"players": up, "qbs": uq, "dist": dist, "history": P, "qb_history": Q, "games": B["games"],
            "T": B["T"], "out": ex, "def_role": defense_by_role(P, B["games"], cur), "qb_def": qdef}
