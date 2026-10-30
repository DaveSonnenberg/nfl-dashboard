"""Rushing + play-calling tables, REG season 2021-2025.

rpg.parquet  player-game (RB/FB/WR/TE with >=1 offensive snap): carries, rush yds
rtg.parquet  team-game: non-QB carries, plays, pass attempts, neutral-situation
             pass rate & pass-rate-over-expected (PROE), rush/pass EPA, lines
"""
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl

OUT = Path(__file__).parent
SEASONS = [2021, 2022, 2023, 2024, 2025]


def build():
    players = nfl.load_players().select(["gsis_id", "pfr_id"]).to_pandas()
    pfr2gsis = players.dropna().drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    R, T = [], []
    for s in SEASONS:
        ps = (nfl.load_player_stats([s]).filter(pl.col("season_type") == "REG")
              .select(["player_id", "game_id", "position", "team", "carries", "rushing_yards"]).to_pandas())
        nonqb = (ps[ps["position"] != "QB"].groupby(["game_id", "team"])
                 .agg(team_carries=("carries", "sum"), team_rush_yds=("rushing_yards", "sum")).reset_index())
        sn = nfl.load_snap_counts([s]).filter(pl.col("game_type") == "REG").to_pandas()
        sn = sn[sn["position"].isin(["RB", "FB", "WR", "TE"]) & (sn["offense_snaps"] > 0)].copy()
        sn["player_id"] = sn["pfr_player_id"].map(pfr2gsis)
        sn = sn.dropna(subset=["player_id"])[["game_id", "season", "week", "team", "opponent", "player_id",
                                              "player", "position", "offense_pct"]]
        r = sn.merge(ps[["player_id", "game_id", "carries", "rushing_yards"]], on=["player_id", "game_id"], how="left")
        r[["carries", "rushing_yards"]] = r[["carries", "rushing_yards"]].fillna(0)
        r = r.merge(nonqb, on=["game_id", "team"], how="left")
        R.append(r)

        pbp = (nfl.load_pbp([s]).filter(pl.col("season_type") == "REG")
               .select(["game_id", "week", "posteam", "defteam", "play_type", "pass", "rush", "xpass", "pass_oe",
                        "wp", "qtr", "down", "half_seconds_remaining", "qb_kneel", "qb_spike", "epa",
                        "pass_attempt", "sack", "qb_scramble", "two_point_attempt"]).to_pandas())
        pbp = pbp[pbp["posteam"].notna() & pbp["play_type"].isin(["pass", "run"])
                  & (pbp["two_point_attempt"] != 1) & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)]
        neutral = (pbp["down"].isin([1, 2]) & pbp["wp"].between(0.2, 0.8) & (pbp["qtr"] <= 3)
                   & pbp["xpass"].notna())
        g = pbp.groupby(["game_id", "posteam"])
        tg = pd.DataFrame({
            "defteam": g["defteam"].first(), "week": g["week"].first(),
            "plays": g.size(), "pass_rate": g["pass"].mean(),
            "pass_att": pbp[(pbp["pass_attempt"] == 1) & (pbp["sack"] == 0)].groupby(["game_id", "posteam"]).size(),
            "rush_epa_play": pbp[pbp["rush"] == 1].groupby(["game_id", "posteam"])["epa"].mean(),
            "pass_epa_play": pbp[pbp["pass"] == 1].groupby(["game_id", "posteam"])["epa"].mean(),
            "rush_plays": pbp[pbp["rush"] == 1].groupby(["game_id", "posteam"]).size(),
            "pass_plays": pbp[pbp["pass"] == 1].groupby(["game_id", "posteam"]).size(),
            "n_neutral": pbp[neutral].groupby(["game_id", "posteam"]).size(),
            "neutral_pass_rate": pbp[neutral].groupby(["game_id", "posteam"])["pass"].mean(),
            "neutral_proe": pbp[neutral].groupby(["game_id", "posteam"])["pass_oe"].mean(),
            "neutral_xpass": pbp[neutral].groupby(["game_id", "posteam"])["xpass"].mean(),
        }).reset_index().rename(columns={"posteam": "team"})
        sch = nfl.load_schedules([s]).filter(pl.col("game_type") == "REG").to_pandas()[
            ["game_id", "home_team", "spread_line", "total_line"]]
        tg = tg.merge(sch, on="game_id")
        tg["home"] = tg["team"] == tg["home_team"]
        tg["team_spread"] = np.where(tg["home"], -tg["spread_line"], tg["spread_line"])
        tg = tg.merge(nonqb, on=["game_id", "team"], how="left")
        tg["season"] = s
        T.append(tg.drop(columns=["home_team", "spread_line"]))
        print("done", s, flush=True)
    pd.concat(R).to_parquet(OUT / "rpg.parquet")
    pd.concat(T).to_parquet(OUT / "rtg.parquet")


if __name__ == "__main__":
    build()
