"""Build raw player-game, team-game and man/zone tables for 2021-2025 (REG season).

Outputs (parquet, in this folder):
  pg.parquet    one row per player per game he played >=1 offensive snap (WR/TE/RB/FB)
                with official targets/receptions/yards/air yards, snap %, team, opp.
  tg.parquet    one row per team per game: team targets, dropbacks, plays, spread,
                total, implied team total.
  mz.parquet    per player-season: on-field dropbacks, targets, rec, yards vs man
                and vs zone (participation; 2022-2025).
"""
import sys
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl

OUT = Path(__file__).parent
SEASONS = [2021, 2022, 2023, 2024, 2025]
POS = ["WR", "TE", "RB", "FB"]


def build():
    players = nfl.load_players().select(["gsis_id", "pfr_id", "display_name", "position"]).to_pandas()
    pfr2gsis = players.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]

    pg_all, tg_all, mz_all = [], [], []
    for s in SEASONS:
        # ---- official player stats
        ps = (nfl.load_player_stats([s]).filter(pl.col("season_type") == "REG")
              .select(["player_id", "player_display_name", "position", "season", "week", "game_id",
                       "team", "opponent_team", "targets", "receptions", "receiving_yards",
                       "receiving_air_yards", "receiving_yards_after_catch", "carries"])
              .to_pandas())
        # ---- snaps (participation universe)
        sn = nfl.load_snap_counts([s]).filter(pl.col("game_type") == "REG").to_pandas()
        sn = sn[sn["position"].isin(POS) & (sn["offense_snaps"] > 0)].copy()
        sn["player_id"] = sn["pfr_player_id"].map(pfr2gsis)
        sn = sn.dropna(subset=["player_id"])
        sn = sn[["game_id", "season", "week", "team", "opponent", "player_id", "player", "position",
                 "offense_snaps", "offense_pct"]]
        pg = sn.merge(ps.drop(columns=["position", "team", "opponent_team", "season", "week"]),
                      on=["game_id", "player_id"], how="left")
        for c in ["targets", "receptions", "receiving_yards", "receiving_air_yards",
                  "receiving_yards_after_catch", "carries"]:
            pg[c] = pg[c].fillna(0)
        pg["name"] = pg["player_display_name"].fillna(pg["player"])
        pg_all.append(pg.drop(columns=["player_display_name", "player"]))

        # ---- team-game volume from pbp
        pbp = (nfl.load_pbp([s]).filter(pl.col("season_type") == "REG")
               .select(["game_id", "posteam", "play_id", "pass_attempt", "sack", "two_point_attempt",
                        "qb_dropback", "rush_attempt", "play_type", "receiver_player_id"])
               .to_pandas())
        pbp = pbp[pbp["posteam"].notna()]
        tgt = pbp[(pbp["pass_attempt"] == 1) & (pbp["sack"] == 0) & (pbp["two_point_attempt"] == 0)
                  & pbp["receiver_player_id"].notna()]
        tg = pd.DataFrame({
            "team_targets": tgt.groupby(["game_id", "posteam"]).size(),
            "dropbacks": pbp[pbp["qb_dropback"] == 1].groupby(["game_id", "posteam"]).size(),
            "plays": pbp[pbp["play_type"].isin(["pass", "run"])].groupby(["game_id", "posteam"]).size(),
        }).fillna(0).reset_index().rename(columns={"posteam": "team"})
        sch = nfl.load_schedules([s]).filter(pl.col("game_type") == "REG").to_pandas()
        sch = sch[["game_id", "week", "home_team", "away_team", "spread_line", "total_line"]]
        tg = tg.merge(sch, on="game_id")
        tg["home"] = tg["team"] == tg["home_team"]
        tg["opp"] = np.where(tg["home"], tg["away_team"], tg["home_team"])
        # spread_line: home perspective, positive = home favored
        tg["team_spread"] = np.where(tg["home"], -tg["spread_line"], tg["spread_line"])  # neg = favored
        tg["implied_total"] = tg["total_line"] / 2 - tg["team_spread"] / 2
        tg["season"] = s
        tg_all.append(tg.drop(columns=["home_team", "away_team", "spread_line"]))

        # ---- man/zone per player-season (participation)
        if s >= 2022:
            part = (nfl.load_participation([s])
                    .select(["nflverse_game_id", "play_id", "offense_players", "defense_man_zone_type"])
                    .with_columns(pl.col("play_id").cast(pl.Float64)).to_pandas())
            db = pbp[pbp["qb_dropback"] == 1][["game_id", "play_id", "receiver_player_id",
                                               "pass_attempt", "sack", "two_point_attempt"]]
            st = (nfl.load_pbp([s]).filter((pl.col("season_type") == "REG") & (pl.col("qb_dropback") == 1))
                  .select(["game_id", "play_id", "complete_pass", "receiving_yards"]).to_pandas())
            db = db.merge(st, on=["game_id", "play_id"], how="left")
            j = db.merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"])
            j["cov"] = j["defense_man_zone_type"].map({"MAN_COVERAGE": "man", "ZONE_COVERAGE": "zone"})
            j = j[j["cov"].notna() & (j["offense_players"] != "")]
            j["is_tgt"] = (j["pass_attempt"] == 1) & (j["sack"] == 0) & (j["two_point_attempt"] == 0) & j["receiver_player_id"].notna()
            on = j[["game_id", "play_id", "cov", "offense_players"]].copy()
            on["pid"] = on["offense_players"].str.split(";")
            on = on.explode("pid")
            onf = on.groupby(["pid", "cov"]).size().rename("db_on_field")
            t = j[j["is_tgt"]]
            tt = t.groupby(["receiver_player_id", "cov"]).agg(
                tgt=("play_id", "size"), rec=("complete_pass", "sum"),
                yds=("receiving_yards", lambda x: x.fillna(0).sum()))
            tt.index.names = ["pid", "cov"]
            mz = pd.concat([onf, tt], axis=1).fillna(0).reset_index()
            mz["season"] = s
            mz_all.append(mz)
        print("done", s, flush=True)

    pd.concat(pg_all).to_parquet(OUT / "pg.parquet")
    pd.concat(tg_all).to_parquet(OUT / "tg.parquet")
    pd.concat(mz_all).to_parquet(OUT / "mz.parquet")


if __name__ == "__main__":
    build()
