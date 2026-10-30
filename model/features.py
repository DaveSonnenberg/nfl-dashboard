"""Pre-game features for player receiving projections.

Rule: every feature for game g uses only (a) earlier games of the same season,
(b) prior seasons, (c) the betting lines, and (d) who is active for game g
(inactives are public ~90 min before kickoff). No same-game outcomes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = Path("/home/claude/nfl-dashboard")
K_SHARE_PRIOR = 50      # team targets of weight on last season's share (tuned 2023-24)
DECAY = 0.8             # each older game counts 0.8x the next one (tuned)
K_CR, K_YPT, K_ADOT = 80, 120, 120  # targets of shrinkage to position means (tuned)
W_PRIOR_EFF = 0.5       # last season's efficiency counts half (tuned)


def _pos(p):
    return "RB" if p in ("RB", "FB") else p


def load():
    pg = pd.read_parquet(D / "pg.parquet")
    tg = pd.read_parquet(D / "tg.parquet")
    mz = pd.read_parquet(D / "mz.parquet")
    pg["pos"] = pg["position"].map(_pos)
    pg = pg.merge(tg[["game_id", "team", "team_targets", "dropbacks", "plays", "home", "opp",
                      "team_spread", "total_line", "implied_total"]], on=["game_id", "team"], how="left")
    pg["team_targets"] = pg["team_targets"].fillna(0)
    return pg, tg, mz


# --------------------------------------------------------------------------
def player_history(pg):
    """Season-to-date (shifted) + prior-season aggregates per player-game."""
    pg = pg.sort_values(["player_id", "season", "week"]).copy()
    g = pg.groupby(["player_id", "season"])
    for c, src in [("c_tgt", "targets"), ("c_rec", "receptions"), ("c_yds", "receiving_yards"),
                   ("c_air", "receiving_air_yards"), ("c_tt", "team_targets"), ("c_snap", "offense_pct")]:
        pg[c] = g[src].cumsum() - pg[src]
    pg["c_g"] = g.cumcount()
    tv, ttv = pg["targets"].to_numpy(float), pg["team_targets"].to_numpy(float)
    dt, dtt = np.zeros(len(pg)), np.zeros(len(pg))
    for idx in g.indices.values():
        a = b = 0.0
        for i in idx:
            dt[i], dtt[i] = a, b
            a, b = a * DECAY + tv[i], b * DECAY + ttv[i]
    pg["d_tgt"], pg["d_tt"] = dt, dtt
    # last 3 games (shifted)
    for c, src in [("l3_tgt", "targets"), ("l3_tt", "team_targets"), ("l3_snap", "offense_pct")]:
        pg[c] = g[src].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    pg["l3_n"] = g["targets"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).count())
    pg["last_snap"] = g["offense_pct"].shift(1)

    ps = pg.groupby(["player_id", "season"]).agg(
        p_g=("game_id", "size"), p_tgt=("targets", "sum"), p_rec=("receptions", "sum"),
        p_yds=("receiving_yards", "sum"), p_air=("receiving_air_yards", "sum"),
        p_tt=("team_targets", "sum"), p_snap=("offense_pct", "mean")).reset_index()
    ps["season"] += 1
    pg = pg.merge(ps, on=["player_id", "season"], how="left")
    pg[[c for c in ps.columns if c.startswith("p_")]] = pg[[c for c in ps.columns if c.startswith("p_")]].fillna(0)
    return pg


def rate_estimates(pg):
    """Shrunk share / catch rate / yards per target / aDOT."""
    # position priors from all data (stable league constants)
    base = pg[pg["season"] < 2023]  # league constants from pre-backtest seasons only
    pos_cr = base.groupby("pos")["receptions"].sum() / base.groupby("pos")["targets"].sum()
    pos_ypt = base.groupby("pos")["receiving_yards"].sum() / base.groupby("pos")["targets"].sum()
    pos_adot = base.groupby("pos")["receiving_air_yards"].sum() / base.groupby("pos")["targets"].sum()

    p_share = np.where(pg["p_tt"] > 0, pg["p_tgt"] / pg["p_tt"].replace(0, np.nan), np.nan)
    pg["p_share"] = p_share
    pg["c_share"] = pg["c_tgt"] / pg["c_tt"].replace(0, np.nan)
    pg["l3_share"] = pg["l3_tgt"] / pg["l3_tt"].replace(0, np.nan)
    # fallback share for players with no history: by snap share in last game (or position low prior)
    fallback = np.where(pg["pos"] == "WR", 0.08, np.where(pg["pos"] == "TE", 0.06, 0.06))
    prior = np.where(pd.notna(p_share) & (pg["p_g"] >= 4), p_share, fallback)
    kp = np.where(pd.notna(p_share) & (pg["p_g"] >= 4), K_SHARE_PRIOR, 25)
    pg["est_share"] = (pg["d_tgt"] + kp * prior) / (pg["d_tt"] + kp)

    w = W_PRIOR_EFF
    tgt_all = pg["c_tgt"] + w * pg["p_tgt"]
    pg["est_cr"] = (pg["c_rec"] + w * pg["p_rec"] + K_CR * pg["pos"].map(pos_cr)) / (tgt_all + K_CR)
    pg["est_ypt"] = (pg["c_yds"] + w * pg["p_yds"] + K_YPT * pg["pos"].map(pos_ypt)) / (tgt_all + K_YPT)
    pg["est_adot"] = (pg["c_air"] + w * pg["p_air"] + K_ADOT * pg["pos"].map(pos_adot)) / (tgt_all + K_ADOT)
    pg["hist_tgt"] = tgt_all
    return pg


def team_volume(pg, tg):
    """Expected team targets for game g from team + opponent history and lines."""
    tg = tg.sort_values(["team", "season", "week"]).copy()
    g = tg.groupby(["team", "season"])
    tg["t_c_tt"] = g["team_targets"].cumsum() - tg["team_targets"]
    tg["t_c_g"] = g.cumcount()
    prev = tg.groupby(["team", "season"])["team_targets"].mean().rename("t_p_tpg").reset_index()
    prev["season"] += 1
    tg = tg.merge(prev, on=["team", "season"], how="left")
    lg = tg.groupby("season")["team_targets"].mean()
    tg["lg_tpg"] = tg["season"].map(lg.shift(1)).fillna(lg.mean())
    base_prior = tg["t_p_tpg"].fillna(tg["lg_tpg"]) * 0.5 + tg["lg_tpg"] * 0.5
    tg["est_team_tpg"] = (tg["t_c_tt"] + 4 * base_prior) / (tg["t_c_g"] + 4)
    # opponent: targets allowed per game, shrunk
    opp = tg[["game_id", "team", "team_targets", "season", "week"]].rename(
        columns={"team": "opp_def", "team_targets": "allowed"})
    tg2 = tg.merge(opp, left_on=["game_id", "opp"], right_on=["game_id", "opp_def"], how="left",
                   suffixes=("", "_o"))
    a = tg2.sort_values(["opp", "season", "week"]).copy()
    # defensive allowed by the *opponent* team, cumulative over its earlier games
    dfa = tg[["game_id", "season", "week", "team", "opp", "team_targets"]].rename(
        columns={"opp": "defteam"}).sort_values(["defteam", "season", "week"])
    gd = dfa.groupby(["defteam", "season"])
    dfa["d_c_allowed"] = gd["team_targets"].cumsum() - dfa["team_targets"]
    dfa["d_c_g"] = gd.cumcount()
    dfa = dfa[["game_id", "defteam", "d_c_allowed", "d_c_g"]]
    tg = tg.merge(dfa, left_on=["game_id", "opp"], right_on=["game_id", "defteam"], how="left")
    tg["est_opp_allow"] = (tg["d_c_allowed"] + 6 * tg["lg_tpg"]) / (tg["d_c_g"] + 6)
    tg["opp_allow_ratio"] = tg["est_opp_allow"] / tg["lg_tpg"]
    keep = ["game_id", "team", "est_team_tpg", "lg_tpg", "opp_allow_ratio"]
    return pg.merge(tg[keep], on=["game_id", "team"], how="left")


def active_context(pg):
    """Renormalize shares over the players active in game g; vacated share; role rank."""
    tot = pg.groupby(["game_id", "team"])["est_share"].transform("sum")
    pg["active_share_sum"] = tot
    pg["renorm_share"] = pg["est_share"] / tot * 0.995
    # vacated: teammates with season-to-date share who played earlier but are not active now
    rows = []
    for (s, team), g in pg.groupby(["season", "team"]):
        weeks = sorted(g["week"].unique())
        hist = g[["week", "player_id", "targets", "team_targets"]]
        for w in weeks:
            before = hist[hist["week"] < w]
            if before.empty:
                rows.append((s, team, w, 0.0)); continue
            sh = before.groupby("player_id")["targets"].sum() / max(
                before.drop_duplicates("week")["team_targets"].sum(), 1)
            active = set(g.loc[g["week"] == w, "player_id"])
            rows.append((s, team, w, float(sh[~sh.index.isin(active)].sum())))
    vac = pd.DataFrame(rows, columns=["season", "team", "week", "vacated"])
    pg = pg.merge(vac, on=["season", "team", "week"], how="left")
    pg["pos_rank"] = pg.groupby(["game_id", "team", "pos"])["est_share"].rank(ascending=False, method="first")
    pg["role"] = pg["pos"] + pg["pos_rank"].clip(upper=3).astype(int).astype(str)
    return pg


def opponent_adjust(pg):
    """Defense: share allowed to each role vs league, catch rate & YPT allowed by position.
    Season-to-date (shifted) shrunk toward prior season, then toward league."""
    pg = pg.sort_values(["season", "week"]).copy()
    pg["defteam"] = pg["opp"]
    out = {}
    # role share allowed
    r = pg.groupby(["season", "week", "game_id", "defteam", "role"]).agg(
        t=("targets", "sum"), tt=("team_targets", "first"), rec=("receptions", "sum"),
        yds=("receiving_yards", "sum")).reset_index()
    lg_role = r.groupby(["season", "role"]).agg(t=("t", "sum"), tt=("tt", "sum")).reset_index()
    lg_role["lg_share"] = lg_role["t"] / lg_role["tt"]
    r = r.sort_values(["defteam", "role", "season", "week"])
    g = r.groupby(["defteam", "role", "season"])
    r["d_t"] = g["t"].cumsum() - r["t"]
    r["d_tt"] = g["tt"].cumsum() - r["tt"]
    prev = r.groupby(["defteam", "role", "season"]).agg(pt=("t", "sum"), ptt=("tt", "sum")).reset_index()
    prev["season"] += 1
    r = r.merge(prev, on=["defteam", "role", "season"], how="left").fillna({"pt": 0, "ptt": 0})
    r = r.merge(lg_role[["season", "role", "lg_share"]], on=["season", "role"], how="left")
    # observed allowed share, prior season counted at 1/3 weight, shrink with 150 team targets
    num = r["d_t"] + r["pt"] / 3 + 150 * r["lg_share"]
    den = r["d_tt"] + r["ptt"] / 3 + 150
    r["opp_role_mult"] = (num / den) / r["lg_share"]
    pg = pg.merge(r[["game_id", "defteam", "role", "opp_role_mult"]], on=["game_id", "defteam", "role"], how="left")

    # catch rate & YPT allowed by position
    p = pg.groupby(["season", "week", "game_id", "defteam", "pos"]).agg(
        t=("targets", "sum"), rec=("receptions", "sum"), yds=("receiving_yards", "sum")).reset_index()
    lg = p.groupby(["season", "pos"]).agg(t=("t", "sum"), rec=("rec", "sum"), yds=("yds", "sum")).reset_index()
    lg["lg_cr"], lg["lg_ypt"] = lg["rec"] / lg["t"], lg["yds"] / lg["t"]
    p = p.sort_values(["defteam", "pos", "season", "week"])
    g = p.groupby(["defteam", "pos", "season"])
    for c in ["t", "rec", "yds"]:
        p["d_" + c] = g[c].cumsum() - p[c]
    prev = p.groupby(["defteam", "pos", "season"]).agg(pt=("t", "sum"), prec=("rec", "sum"), pyds=("yds", "sum")).reset_index()
    prev["season"] += 1
    p = p.merge(prev, on=["defteam", "pos", "season"], how="left").fillna(0)
    p = p.merge(lg[["season", "pos", "lg_cr", "lg_ypt"]], on=["season", "pos"])
    T = p["d_t"] + p["pt"] / 3
    p["opp_cr_mult"] = ((p["d_rec"] + p["prec"] / 3 + 80 * p["lg_cr"]) / (T + 80)) / p["lg_cr"]
    p["opp_ypt_mult"] = ((p["d_yds"] + p["pyds"] / 3 + 150 * p["lg_ypt"]) / (T + 150)) / p["lg_ypt"]
    pg = pg.merge(p[["game_id", "defteam", "pos", "opp_cr_mult", "opp_ypt_mult"]],
                  on=["game_id", "defteam", "pos"], how="left")
    for c in ["opp_role_mult", "opp_cr_mult", "opp_ypt_mult"]:
        pg[c] = pg[c].fillna(1.0)
    return pg


def man_zone(pg, mz):
    """Player man/zone splits from PRIOR seasons + opponent play-caller's man lean."""
    mz = mz.copy()
    rows = []
    for s in sorted(pg["season"].unique()):
        h = mz[mz["season"].between(s - 2, s - 1)]
        if h.empty:
            continue
        a = h.groupby(["pid", "cov"])[["db_on_field", "tgt", "rec", "yds"]].sum().unstack("cov").fillna(0)
        a.columns = [f"{x}_{y}" for x, y in a.columns]
        a = a.reset_index()
        a["season"] = s
        rows.append(a)
    A = pd.concat(rows)
    # league rates per route-proxy for shrinkage
    lgm = A["tgt_man"].sum() / A["db_on_field_man"].sum()
    lgz = A["tgt_zone"].sum() / A["db_on_field_zone"].sum()
    k = 250  # on-field dropbacks of shrinkage
    tot_rate = (A["tgt_man"] + A["tgt_zone"]) / (A["db_on_field_man"] + A["db_on_field_zone"]).replace(0, np.nan)
    # player's man rate relative to his own overall rate (shrunk toward 'no split' = league ratio)
    man_rate = (A["tgt_man"] + k * tot_rate * lgm / ((lgm + lgz) / 2)) / (A["db_on_field_man"] + k)
    zone_rate = (A["tgt_zone"] + k * tot_rate * lgz / ((lgm + lgz) / 2)) / (A["db_on_field_zone"] + k)
    A["mz_tgt_ratio"] = np.log((man_rate / lgm) / (zone_rate / lgz))
    kt = 40
    ypt_all = (A["yds_man"] + A["yds_zone"]) / (A["tgt_man"] + A["tgt_zone"]).replace(0, np.nan)
    ypt_m = (A["yds_man"] + kt * ypt_all) / (A["tgt_man"] + kt)
    ypt_z = (A["yds_zone"] + kt * ypt_all) / (A["tgt_zone"] + kt)
    A["mz_ypt_ratio"] = np.log(ypt_m / ypt_z)
    A["mz_db"] = A["db_on_field_man"] + A["db_on_field_zone"]
    pg = pg.merge(A[["pid", "season", "mz_tgt_ratio", "mz_ypt_ratio", "mz_db"]].rename(columns={"pid": "player_id"}),
                  on=["player_id", "season"], how="left")
    pg[["mz_tgt_ratio", "mz_ypt_ratio"]] = pg[["mz_tgt_ratio", "mz_ypt_ratio"]].fillna(0)
    pg["mz_db"] = pg["mz_db"].fillna(0)

    # opponent play-caller man lean from seasons before s
    cov = pd.read_csv(REPO / "coverage_team_seasons.csv")
    co = pd.read_csv(REPO / "coordinators.csv")
    co["caller"] = co["def_playcaller"].str.split(";").str[0].str.strip()
    h = cov.merge(co[["season", "team", "caller"]], on=["season", "team"])
    lean = []
    for s in sorted(pg["season"].unique()):
        cur = co[co["season"] == s][["team", "caller"]]
        hh = h[h["season"] < s].groupby("caller")["man_z"].mean()
        c = cur.assign(opp_man_z=cur["caller"].map(hh), season=s)
        lean.append(c[["season", "team", "opp_man_z"]])
    L = pd.concat(lean).rename(columns={"team": "opp"})
    pg = pg.merge(L, on=["season", "opp"], how="left")
    pg["opp_man_z"] = pg["opp_man_z"].fillna(0)
    pg["mz_tgt_x"] = pg["mz_tgt_ratio"] * pg["opp_man_z"]
    pg["mz_ypt_x"] = pg["mz_ypt_ratio"] * pg["opp_man_z"]
    return pg


def build_features():
    pg, tg, mz = load()
    pg = player_history(pg)
    pg = rate_estimates(pg)
    pg = team_volume(pg, tg)
    pg = active_context(pg)
    pg = opponent_adjust(pg)
    pg = man_zone(pg, mz)
    pg.to_parquet(D / "features.parquet")
    return pg


if __name__ == "__main__":
    f = build_features()
    print(f.shape)
