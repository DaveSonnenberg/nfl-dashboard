"""Man/zone coverage history and defensive play-callers.

Two parts:
1. build(): one-time (per offseason) precompute from nflverse participation +
   play-by-play into coverage_team_seasons.csv and coverage_league.csv. Those
   seasons never change, so the app reads the small CSVs instead of pulling
   ~4 seasons of data on every load.
   Rebuild after nflverse publishes a new season's participation file:
       python coverage.py 2022 2023 2024 2025 2026
2. caller_view(): joins coordinators.csv to that history so each defense shows
   its current play-caller and the man rate of defenses he called before.

Man % is measured against that season's league (rank and z-score) because
league-wide tagging shifts a lot between seasons (41% -> 49% -> 31%).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
TEAM_FILE = HERE / "coverage_team_seasons.csv"
LEAGUE_FILE = HERE / "coverage_league.csv"
COORD_FILE = HERE / "coordinators.csv"

TOP_SCRAMBLERS = ["J.Daniels", "D.Maye", "J.Fields", "L.Jackson", "J.Hurts", "J.Allen",
                  "P.Mahomes", "C.Williams", "K.Murray", "J.Herbert", "B.Nix", "B.Young"]


# --------------------------------------------------------------------------
# Build (offline)
# --------------------------------------------------------------------------

def _season_plays(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    import polars as pl
    part = (nfl.load_participation([season])
            .select(["nflverse_game_id", "play_id", "defense_man_zone_type", "was_pressure"])
            .with_columns(pl.col("play_id").cast(pl.Int32)))
    pbp = (nfl.load_pbp([season])
           .filter((pl.col("season_type") == "REG") & (pl.col("qb_dropback") == 1))
           .select(["season", "game_id", "play_id", "defteam", "passer", "qb_scramble",
                    "rush_attempt", "rushing_yards"])
           .with_columns(pl.col("play_id").cast(pl.Int32)))
    j = pbp.join(part, left_on=["game_id", "play_id"],
                 right_on=["nflverse_game_id", "play_id"], how="left").to_pandas()
    j["cov"] = j["defense_man_zone_type"].map({"MAN_COVERAGE": "Man", "ZONE_COVERAGE": "Zone"})
    j["scr"] = (j["qb_scramble"] == 1) & (j["rush_attempt"] == 1)
    return j


def build(seasons: list[int]) -> None:
    plays = pd.concat([_season_plays(s) for s in seasons], ignore_index=True)

    rows = []
    for (season, team), g in plays.groupby(["season", "defteam"]):
        tagged = g[g["cov"].notna()]
        pr = g["was_pressure"].dropna()
        s = g[g["scr"]]
        rows.append({"season": season, "team": team, "dropbacks": len(g),
                     "tagged": len(tagged), "man_pct": (tagged["cov"] == "Man").mean(),
                     "pressure_pct": pr.astype(bool).mean() if len(pr) else np.nan,
                     "scrambles": len(s), "scr_yds": s["rushing_yards"].sum()})
    t = pd.DataFrame(rows)
    grp = t.groupby("season")["man_pct"]
    t["man_z"] = (t["man_pct"] - grp.transform("mean")) / grp.transform("std")
    t["man_rank"] = grp.rank(ascending=False, method="min").astype(int)
    pgrp = t.groupby("season")["pressure_pct"]
    t["pressure_rank"] = pgrp.rank(ascending=False, method="min")
    t.round(4).to_csv(TEAM_FILE, index=False)

    # League-wide: yards per scramble vs man and zone
    s = plays[plays["scr"] & plays["cov"].notna()].copy()
    s["group"] = np.where(s["passer"].isin(TOP_SCRAMBLERS), "Top-12 scramblers", "Other")
    out = []
    for label, sub in [("All QBs", s), ("Top-12 scramblers", s[s["group"] == "Top-12 scramblers"])]:
        for cov, g in sub.groupby("cov"):
            y = g["rushing_yards"]
            out.append({"group": label, "coverage": cov, "scrambles": len(y),
                        "yds_per_scramble": y.mean(), "median": y.median(),
                        "pct_10plus": (y >= 10).mean(), "pct_20plus": (y >= 20).mean()})
    db = plays[plays["cov"].notna()]
    rate = db.groupby("cov")["scr"].mean()
    lg = pd.DataFrame(out)
    lg["scramble_rate_all"] = lg["coverage"].map(rate)
    lg.attrs["seasons"] = seasons
    lg.round(4).to_csv(LEAGUE_FILE, index=False)
    print(f"wrote {TEAM_FILE.name} ({len(t)} rows) and {LEAGUE_FILE.name}")


# --------------------------------------------------------------------------
# App side
# --------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cov = pd.read_csv(TEAM_FILE)
    lg = pd.read_csv(LEAGUE_FILE)
    co = pd.read_csv(COORD_FILE)
    co["caller"] = co["def_playcaller"].str.split(";").str[0].str.strip()
    return cov, lg, co


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def caller_view(season: int, cov: pd.DataFrame, co: pd.DataFrame) -> pd.DataFrame:
    """One row per defense for `season`: play-caller + man history before `season`."""
    cur = co[co["season"] == season][["team", "caller", "def_playcaller", "confidence"]]
    if cur.empty:  # season not in the coordinator file yet -> use the latest season we have
        cur = co[co["season"] == co["season"].max()][["team", "caller", "def_playcaller", "confidence"]]
    prev = co[co["season"] == season - 1][["team", "caller"]].rename(columns={"caller": "prev_caller"})
    hist = cov.merge(co[["season", "team", "caller"]], on=["season", "team"], how="left")
    hist = hist[hist["season"] < season]
    last_cov = cov["season"].max()
    n = cov.groupby("season").size()

    rows = []
    for r in cur.merge(prev, on="team", how="left").itertuples(index=False):
        h = hist[hist["caller"] == r.caller].sort_values("season")
        team_last = cov[(cov["team"] == r.team) & (cov["season"] == min(season - 1, last_cov))]
        row = {
            "team": r.team, "Play-caller": r.def_playcaller,
            "New caller": (r.prev_caller != r.caller) if isinstance(r.prev_caller, str) else np.nan,
            "Caller seasons": len(h),
            "Caller history": ", ".join(
                f"{x.team} {str(x.season)[2:]}: {x.man_pct:.0%} ({_ordinal(int(x.man_rank))})"
                for x in h.itertuples()) or "no prior play-calling data",
            "Caller man z": h["man_z"].mean() if len(h) else np.nan,
            "Caller last man %": h["man_pct"].iloc[-1] if len(h) else np.nan,
            "Caller last rank": int(h["man_rank"].iloc[-1]) if len(h) else np.nan,
            "Caller pressure %": h["pressure_pct"].mean() if len(h) else np.nan,
            "Team prior man %": team_last["man_pct"].iloc[0] if len(team_last) else np.nan,
            "Team prior rank": int(team_last["man_rank"].iloc[0]) if len(team_last) else np.nan,
            "Team prior season": int(team_last["season"].iloc[0]) if len(team_last) else np.nan,
            "confidence": r.confidence,
        }
        rows.append(row)
    out = pd.DataFrame(rows).set_index("team")
    out.attrs["teams_per_season"] = int(n.max()) if len(n) else 32
    return out


def man_label(z: float) -> str:
    if pd.isna(z):
        return "Unknown"
    if z >= 0.5:
        return "Man-heavy"
    if z <= -0.5:
        return "Zone-heavy"
    return "Balanced"


if __name__ == "__main__":
    build([int(a) for a in sys.argv[1:]] or [2022, 2023, 2024, 2025])
