"""QB rushing data: QB profiles, defense profiles, splits.

Definitions (all from nflverse play-by-play + FTN charting):
- Starter: the QB with the most dropbacks for his team in that game. Profiles
  and defense numbers use starters only, so mop-up and trick plays are ignored.
- Dropback: qb_dropback == 1 (passes, sacks, scrambles).
- Scramble: qb_scramble == 1. Designed run: any other QB rush except kneels.
- Sneak: FTN's is_qb_sneak flag; if a play isn't charted yet, a designed run
  on 3rd/4th down with 1 yard or less to go.
- Official rush yards include kneels (that's how props settle).
  Designed/scramble numbers exclude kneels.
- Blitz: FTN n_blitzers >= 1, on charted dropbacks only.
"""

from __future__ import annotations

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl

PBP_COLS = [
    "season", "season_type", "week", "game_id", "play_id", "posteam", "defteam",
    "home_team", "spread_line", "qb_dropback", "qb_scramble", "qb_kneel",
    "rush_attempt", "rushing_yards", "two_point_attempt", "sack", "qb_hit",
    "passer_id", "passer_player_id", "rusher_player_id", "down", "ydstogo", "epa",
    "success", "rush_touchdown",
]


# --------------------------------------------------------------------------
# Load + classify plays
# --------------------------------------------------------------------------

def load_plays(season: int, include_post: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (plays, players). plays = every non-2pt play with QB flags + FTN."""
    types = ["REG", "POST"] if include_post else ["REG"]
    pbp = (nfl.load_pbp([season]).select(PBP_COLS)
           .filter(pl.col("season_type").is_in(types)
                   & pl.col("posteam").is_not_null()
                   & (pl.col("two_point_attempt").fill_null(0) == 0))
           .with_columns(pl.col("play_id").cast(pl.Int32)))
    try:
        ftn = (nfl.load_ftn_charting([season])
               .select(["nflverse_game_id", "nflverse_play_id", "n_blitzers",
                        "n_pass_rushers", "is_qb_sneak", "n_defense_box"])
               .with_columns(pl.col("nflverse_play_id").cast(pl.Int32)))
        pbp = pbp.join(ftn, left_on=["game_id", "play_id"],
                       right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
    except Exception:  # FTN missing (older seasons) -> blitz/sneak fall back
        pbp = pbp.with_columns(pl.lit(None, pl.Int32).alias("n_blitzers"),
                               pl.lit(None, pl.Int32).alias("n_pass_rushers"),
                               pl.lit(None, pl.Boolean).alias("is_qb_sneak"),
                               pl.lit(None, pl.Int32).alias("n_defense_box"))
    p = pbp.to_pandas()

    for c in ["qb_dropback", "qb_scramble", "qb_kneel", "rush_attempt", "sack", "qb_hit"]:
        p[c] = p[c].fillna(0).astype(int)
    p["rushing_yards"] = p["rushing_yards"].fillna(0.0)
    p["dropback_qb"] = p["passer_id"].fillna(p["passer_player_id"]).fillna(p["rusher_player_id"])
    p.loc[p["qb_dropback"] == 0, "dropback_qb"] = np.nan

    players = (nfl.load_players().select(["gsis_id", "display_name", "position"])
               .filter(pl.col("gsis_id").is_not_null()).to_pandas()
               .drop_duplicates("gsis_id"))
    return p, players


def starters(p: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: the starting QB (most dropbacks)."""
    db = (p[p["qb_dropback"] == 1].groupby(["game_id", "posteam", "dropback_qb"])
          .size().rename("dropbacks").reset_index()
          .sort_values("dropbacks", ascending=False)
          .drop_duplicates(["game_id", "posteam"])
          .rename(columns={"dropback_qb": "qb_id"}))
    return db.merge(players.rename(columns={"gsis_id": "qb_id", "display_name": "qb"})[["qb_id", "qb"]],
                    on="qb_id", how="left")


# --------------------------------------------------------------------------
# QB game log (the base table everything else is built from)
# --------------------------------------------------------------------------

def qb_games(p: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    st = starters(p, players)
    game_info = (p.groupby(["game_id", "posteam"])
                 .agg(week=("week", "first"), defteam=("defteam", "first"),
                      home_team=("home_team", "first"), spread_line=("spread_line", "first"))
                 .reset_index())
    g = st.merge(game_info, on=["game_id", "posteam"])

    # Plays involving the starter
    p = p.merge(st[["game_id", "posteam", "qb_id"]], on=["game_id", "posteam"], how="inner")
    qb_rush = p[(p["rush_attempt"] == 1) & (p["rusher_player_id"] == p["qb_id"])].copy()
    qb_rush["kind"] = np.select(
        [qb_rush["qb_kneel"] == 1, qb_rush["qb_scramble"] == 1], ["kneel", "scramble"], "designed")
    fallback_sneak = qb_rush["down"].isin([3, 4]) & (qb_rush["ydstogo"] <= 1)
    qb_rush["sneak"] = ((qb_rush["kind"] == "designed")
                        & qb_rush["is_qb_sneak"].fillna(fallback_sneak).astype(bool))

    def agg(mask, prefix):
        s = qb_rush[mask].groupby(["game_id", "posteam"]).agg(
            **{f"{prefix}_att": ("rushing_yards", "size"),
               f"{prefix}_yds": ("rushing_yards", "sum")})
        return s

    parts = [
        agg(qb_rush["kind"] == "designed", "des"),
        agg(qb_rush["sneak"], "sneak"),
        agg(qb_rush["kind"] == "scramble", "scr"),
        agg(qb_rush["kind"] == "kneel", "kneel"),
        agg(qb_rush["kind"] != "kneel", "rush"),
        qb_rush.groupby(["game_id", "posteam"]).agg(
            official_att=("rushing_yards", "size"), official_yds=("rushing_yards", "sum"),
            rush_td=("rush_touchdown", lambda s: int(np.nansum(s)))),
    ]
    for part in parts:
        g = g.merge(part.reset_index(), on=["game_id", "posteam"], how="left")
    num = [c for c in g.columns if c.endswith(("_att", "_yds")) or c == "rush_td"]
    g[num] = g[num].fillna(0)

    # Pressure faced on his dropbacks
    mine = p[(p["qb_dropback"] == 1) & (p["dropback_qb"] == p["qb_id"])]
    pr = mine.groupby(["game_id", "posteam"]).agg(
        sacks=("sack", "sum"),
        sack_or_hit=("sack", lambda s: int(((s == 1) | (mine.loc[s.index, "qb_hit"] == 1)).sum())),
        charted_db=("n_pass_rushers", lambda s: int((s > 0).sum())),
        blitzed=("n_blitzers", lambda s: int((s > 0).sum())))
    g = g.merge(pr.reset_index(), on=["game_id", "posteam"], how="left")

    g["home"] = g["posteam"] == g["home_team"]
    # spread_line is from the home team's view (positive = home favored)
    g["team_spread"] = np.where(g["home"], -g["spread_line"], g["spread_line"])
    g["favored"] = g["team_spread"] < 0
    return g.sort_values(["qb", "week"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Defense profile
# --------------------------------------------------------------------------

def add_norms(qg: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out QB norms: that QB's rates in his OTHER starts in the window."""
    qg = qg.copy()
    cols = ["des_att", "des_yds", "scr_att", "scr_yds", "rush_yds", "dropbacks"]
    tot = qg.groupby("qb_id")[cols].transform("sum")
    other = qg.groupby("qb_id")["game_id"].transform("size") - 1
    o = other.replace(0, np.nan)
    qg["has_norm"] = other > 0
    qg["norm_games"] = other
    qg["exp_des_att"] = (tot["des_att"] - qg["des_att"]) / o
    qg["exp_rush_yds"] = (tot["rush_yds"] - qg["rush_yds"]) / o
    loo_rate = (tot["scr_att"] - qg["scr_att"]) / (tot["dropbacks"] - qg["dropbacks"]).replace(0, np.nan)
    qg["exp_scr_att"] = loo_rate * qg["dropbacks"]
    return qg


def defense_profile(p: pd.DataFrame, qg: pd.DataFrame) -> pd.DataFrame:
    """Per defense: QB rushing allowed, opponent-adjusted, plus context."""
    qg = add_norms(qg)
    adj = qg[qg["has_norm"]].groupby("defteam").agg(
        adj_games=("game_id", "size"),
        des_vs_norm=("des_att", "sum"), des_exp=("exp_des_att", "sum"),
        scr_vs_norm=("scr_att", "sum"), scr_exp=("exp_scr_att", "sum"),
        yds_vs_norm=("rush_yds", "sum"), yds_exp=("exp_rush_yds", "sum"))
    adj["Designed vs norm /g"] = (adj["des_vs_norm"] - adj["des_exp"]) / adj["adj_games"]
    adj["Scrambles vs norm /g"] = (adj["scr_vs_norm"] - adj["scr_exp"]) / adj["adj_games"]
    adj["Rush yds vs norm /g"] = (adj["yds_vs_norm"] - adj["yds_exp"]) / adj["adj_games"]

    base = qg.groupby("defteam").agg(
        G=("game_id", "size"),
        des_att=("des_att", "sum"), des_yds=("des_yds", "sum"),
        scr_att=("scr_att", "sum"), scr_yds=("scr_yds", "sum"),
        dropbacks=("dropbacks", "sum"), official=("official_yds", "median"),
        rush_yds=("rush_yds", "sum"))
    out = pd.DataFrame(index=base.index)
    out["G"] = base["G"]
    out["Designed /g"] = base["des_att"] / base["G"]
    out["Designed YPC"] = base["des_yds"] / base["des_att"].replace(0, np.nan)
    out["Scramble %"] = base["scr_att"] / base["dropbacks"]
    out["Yds/scramble"] = base["scr_yds"] / base["scr_att"].replace(0, np.nan)
    out["QB rush yds /g"] = base["rush_yds"] / base["G"]
    out["Median official"] = base["official"]
    out = out.join(adj[["adj_games", "Designed vs norm /g", "Scrambles vs norm /g",
                        "Rush yds vs norm /g"]])

    # Context: pressure, blitz, run D vs RBs, pass D
    db = p[p["qb_dropback"] == 1]
    ch = db[db["n_pass_rushers"].fillna(0) > 0]
    ctx = pd.DataFrame({
        "Blitz %": ch.groupby("defteam")["n_blitzers"].apply(lambda s: (s > 0).mean()),
        "Charted DB": ch.groupby("defteam").size(),
        "Sack+hit %": db.assign(x=(db["sack"] == 1) | (db["qb_hit"] == 1))
                        .groupby("defteam")["x"].mean(),
        "Pass EPA/db": db.groupby("defteam")["epa"].mean(),
    })
    out = out.join(ctx)
    return out


def rb_run_defense(p: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    rbs = set(players.loc[players["position"].isin(["RB", "FB"]), "gsis_id"])
    r = p[(p["rush_attempt"] == 1) & p["rusher_player_id"].isin(rbs)]
    return pd.DataFrame({
        "RB YPC allowed": r.groupby("defteam")["rushing_yards"].mean(),
        "RB rush EPA": r.groupby("defteam")["epa"].mean(),
        "RB success %": r.groupby("defteam")["success"].mean(),
    })


def add_ranks(dp: pd.DataFrame) -> pd.DataFrame:
    """Adds opponent-trait buckets used for QB splits (top/bottom half)."""
    t = pd.DataFrame(index=dp.index)
    med = dp.median(numeric_only=True)
    t["pass_d"] = np.where(dp["Pass EPA/db"] > med["Pass EPA/db"], "Weak pass D", "Strong pass D")
    t["blitz"] = np.where(dp["Blitz %"] > med["Blitz %"], "High blitz", "Low blitz")
    t["pressure"] = np.where(dp["Sack+hit %"] > med["Sack+hit %"], "High pressure", "Low pressure")
    t["run_d"] = np.where(dp["RB rush EPA"] > med["RB rush EPA"], "Weak run D", "Strong run D")
    return t


# --------------------------------------------------------------------------
# QB summaries
# --------------------------------------------------------------------------

def summarize(g: pd.DataFrame, line: float) -> dict:
    n = len(g)
    if n == 0:
        return {"G": 0}
    db = g["dropbacks"].sum()
    return {
        "G": n,
        "Designed /g": g["des_att"].sum() / n,
        "Sneaks": int(g["sneak_att"].sum()),
        "YPC ex-sneak": ((g["des_yds"].sum() - g["sneak_yds"].sum())
                         / (g["des_att"].sum() - g["sneak_att"].sum())
                         if g["des_att"].sum() > g["sneak_att"].sum() else np.nan),
        "Dropbacks /g": db / n,
        "Scramble %": g["scr_att"].sum() / db if db else np.nan,
        "Yds/scramble": g["scr_yds"].sum() / g["scr_att"].sum() if g["scr_att"].sum() else np.nan,
        "Kneel yds /g": g["kneel_yds"].sum() / n,
        "Avg official": g["official_yds"].mean(),
        "Median official": g["official_yds"].median(),
        f"% over {line:g}": (g["official_yds"] > line).mean(),
    }


def qb_table(qg: pd.DataFrame, line: float, min_games: int) -> pd.DataFrame:
    rows = []
    for (qid, qb), g in qg.groupby(["qb_id", "qb"]):
        if len(g) < min_games:
            continue
        g = g.sort_values("week")
        s = summarize(g, line)
        l4 = summarize(g.tail(4), line)
        rows.append({"QB": qb, "Team": g["posteam"].iloc[-1], **s,
                     "L4 designed /g": l4["Designed /g"], "L4 scramble %": l4["Scramble %"],
                     "L4 median": l4["Median official"]})
    return pd.DataFrame(rows)


SPLITS = {
    "Opp pass D": "pass_d",
    "Opp blitz rate": "blitz",
    "Opp pressure": "pressure",
    "Opp run D vs RBs": "run_d",
    "Home / away": "home_lbl",
    "Favored / underdog": "fav_lbl",
}


def splits(qg_traits: pd.DataFrame, line: float) -> pd.DataFrame:
    rows = []
    for name, col in SPLITS.items():
        for val, g in qg_traits.groupby(col):
            rows.append({"Split": name, "Bucket": val, **summarize(g, line)})
    return pd.DataFrame(rows)


def with_traits(qg: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    out = qg.merge(traits, left_on="defteam", right_index=True, how="left")
    out["home_lbl"] = np.where(out["home"], "Home", "Away")
    out["fav_lbl"] = np.where(out["team_spread"].isna(), "No line",
                              np.where(out["favored"], "Favored", "Underdog"))
    return out
