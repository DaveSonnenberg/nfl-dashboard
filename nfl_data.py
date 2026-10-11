"""Data loading and role logic for the NFL defense-vs-role dashboard.

Everything is pulled straight from nflverse release files, so nothing here
needs a database or a scheduled job. nflverse refreshes nightly in season.
"""

from __future__ import annotations

import re
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import polars as pl

ROLE_ORDER = ["WR1", "WR2", "WR3", "TE", "RB"]
WR_SLOTS = ["WR1", "WR2", "WR3"]
STATS = ["targets", "receptions", "yards", "tds", "air_yards", "ppr"]
PRIOR_WEIGHT = 70  # team targets of weight given to last season's share

PBP_COLS = [
    "season", "season_type", "week", "game_id", "posteam", "defteam",
    "pass_attempt", "sack", "two_point_attempt", "receiver_player_id",
    "receiver_player_name", "complete_pass", "receiving_yards", "air_yards",
    "pass_touchdown", "td_player_id",
]


# --------------------------------------------------------------------------
# Raw loads
# --------------------------------------------------------------------------

def _targets_from_pbp(season: int, include_post: bool) -> pl.DataFrame:
    """One row per targeted pass (sacks, scrambles and 2-pt tries removed)."""
    pbp = nfl.load_pbp([season]).select(PBP_COLS)
    types = ["REG", "POST"] if include_post else ["REG"]
    return (
        pbp.filter(
            pl.col("season_type").is_in(types)
            & (pl.col("pass_attempt") == 1)
            & (pl.col("sack") == 0)
            & (pl.col("two_point_attempt") == 0)
            & pl.col("receiver_player_id").is_not_null()
        )
        .with_columns(
            pl.lit(1).alias("targets"),
            pl.col("complete_pass").fill_null(0).cast(pl.Int32).alias("receptions"),
            pl.col("receiving_yards").fill_null(0).alias("yards"),
            ((pl.col("pass_touchdown") == 1)
             & (pl.col("td_player_id") == pl.col("receiver_player_id")))
            .cast(pl.Int32).alias("tds"),
            pl.col("air_yards").fill_null(0).alias("air_yards"),
        )
        .rename({"receiver_player_id": "gsis_id", "posteam": "team"})
        .select(["season", "week", "game_id", "team", "defteam", "gsis_id",
                 "receiver_player_name", "targets", "receptions", "yards",
                 "tds", "air_yards"])
    )


def load_season(season: int, include_post: bool = False) -> dict[str, pl.DataFrame]:
    targets = _targets_from_pbp(season, include_post)
    try:
        prior = _targets_from_pbp(season - 1, include_post=False)
    except Exception:  # prior season unavailable -> just skip that fallback
        prior = targets.head(0)

    players = (
        nfl.load_players()
        .select(["gsis_id", "pfr_id", "display_name", "position"])
        .filter(pl.col("gsis_id").is_not_null())
    )
    snaps = nfl.load_snap_counts([season])
    if not include_post:
        snaps = snaps.filter(pl.col("game_type") == "REG")
    schedules = nfl.load_schedules([season])
    return {"targets": targets, "prior": prior, "players": players,
            "snaps": snaps, "schedules": schedules}


# --------------------------------------------------------------------------
# Player-game table with roles
# --------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _participation(raw: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Every pass-catcher who played an offensive snap, plus anyone targeted."""
    players = raw["players"]
    snaps = (
        raw["snaps"]
        .filter(pl.col("position").is_in(["WR", "TE", "RB", "FB"])
                & (pl.col("offense_snaps") > 0))
        .join(players.select(["pfr_id", "gsis_id"]).drop_nulls().unique("pfr_id"),
              left_on="pfr_player_id", right_on="pfr_id", how="left")
        .filter(pl.col("gsis_id").is_not_null())
        .select(["game_id", "week", "team", "gsis_id",
                 pl.col("player").alias("snap_name"),
                 pl.col("position").alias("snap_pos"), "offense_pct"])
    )
    tgt_players = raw["targets"].select(
        ["game_id", "week", "team", "defteam", "gsis_id"]).unique()

    games = raw["targets"].select(["game_id", "team", "defteam"]).unique()
    part = (
        pl.concat([snaps.select(["game_id", "week", "team", "gsis_id"]),
                   tgt_players.select(["game_id", "week", "team", "gsis_id"])])
        .unique(["game_id", "team", "gsis_id"])
        .join(games, on=["game_id", "team"], how="inner")
        .join(snaps.select(["game_id", "gsis_id", "snap_name", "snap_pos", "offense_pct"]),
              on=["game_id", "gsis_id"], how="left")
        .join(players.select(["gsis_id", "display_name", "position"]).unique("gsis_id"),
              on="gsis_id", how="left")
        .with_columns(
            pl.coalesce(["display_name", "snap_name"]).alias("player"),
            pl.coalesce(["snap_pos", "position"]).alias("pos"),
            pl.col("offense_pct").fill_null(0.0),
        )
        .select(["game_id", "week", "team", "defteam", "gsis_id", "player",
                 "pos", "offense_pct"])
    )
    return part


def _prior_share(prior: pl.DataFrame) -> pl.DataFrame:
    if prior.height == 0:
        return pl.DataFrame({"gsis_id": [], "prior_share": []},
                            schema={"gsis_id": pl.Utf8, "prior_share": pl.Float64})
    team_game = prior.group_by(["game_id", "team"]).agg(pl.col("targets").sum().alias("tt"))
    pg = (prior.group_by(["game_id", "team", "gsis_id"]).agg(pl.col("targets").sum())
          .join(team_game, on=["game_id", "team"]))
    return (pg.group_by("gsis_id")
              .agg((pl.col("targets").sum() / pl.col("tt").sum()).alias("prior_share")))


def load_overrides(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["season", "week", "team", "player", "role"])
    o = pd.read_csv(p, dtype=str, comment="#").fillna("")
    o.columns = [c.strip().lower() for c in o.columns]
    o = o[o.get("player", "").astype(str).str.strip() != ""]
    return o


def build_player_games(raw: dict[str, pl.DataFrame], overrides: pd.DataFrame) -> pd.DataFrame:
    """One row per player per game with stats, entering target share and role."""
    stats = raw["targets"].group_by(["game_id", "team", "gsis_id"]).agg(
        [pl.col(s).sum() for s in ["targets", "receptions", "yards", "tds", "air_yards"]])
    team_tgts = raw["targets"].group_by(["game_id", "team"]).agg(
        pl.col("targets").sum().alias("team_targets"))

    pg = (
        _participation(raw)
        .join(stats, on=["game_id", "team", "gsis_id"], how="left")
        .join(team_tgts, on=["game_id", "team"], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in
                       ["targets", "receptions", "yards", "tds", "air_yards", "team_targets"]])
        .sort(["team", "gsis_id", "week"])
        # Season-to-date target share ENTERING the game, over games he played
        # for this team. Shifted so a game's own result never sets its role.
        .with_columns(
            pl.col("targets").cum_sum().shift(1).over(["team", "gsis_id"]).alias("prev_t"),
            pl.col("team_targets").cum_sum().shift(1).over(["team", "gsis_id"]).alias("prev_tt"),
        )
        .with_columns(
            pl.when(pl.col("prev_tt") > 0)
              .then(pl.col("prev_t") / pl.col("prev_tt"))
              .otherwise(None).alias("entering_share"))
        .join(_prior_share(raw["prior"]), on="gsis_id", how="left")
        .with_columns(
            pl.when(pl.col("entering_share").is_not_null() & pl.col("prior_share").is_not_null())
              .then(pl.lit("blend"))
              .when(pl.col("entering_share").is_not_null()).then(pl.lit("season"))
              .when(pl.col("prior_share").is_not_null()).then(pl.lit("prior season"))
              .otherwise(pl.lit("snaps only")).alias("role_basis"),
            # Early-season shares are noisy: blend this season with last season,
            # weighting last season like PRIOR_WEIGHT team targets (~2 games).
            pl.when(pl.col("entering_share").is_not_null() & pl.col("prior_share").is_not_null())
              .then((pl.col("prev_t") + PRIOR_WEIGHT * pl.col("prior_share"))
                    / (pl.col("prev_tt") + PRIOR_WEIGHT))
              .otherwise(pl.coalesce(["entering_share", "prior_share", pl.lit(0.0)]))
              .alias("rank_share"),
            (pl.col("receptions") + pl.col("yards") / 10 + 6 * pl.col("tds")).alias("ppr"),
        )
        .to_pandas()
    )

    pg["role"] = "Other"
    pg.loc[pg["pos"].isin(["TE"]), "role"] = "TE"
    pg.loc[pg["pos"].isin(["RB", "FB"]), "role"] = "RB"
    pg["overridden"] = False

    ov = overrides.copy()
    if len(ov):
        ov["key"] = ov["player"].map(_norm)
        ov["team"] = ov["team"].str.upper().str.strip()
        ov["role"] = ov["role"].str.upper().str.strip()

    season = int(raw["targets"]["season"][0]) if raw["targets"].height else 0
    wr = pg[pg["pos"] == "WR"].copy()
    wr["key"] = wr["player"].map(_norm)
    roles = {}
    for (gid, team), grp in wr.groupby(["game_id", "team"]):
        grp = grp.sort_values(["rank_share", "offense_pct"], ascending=False)
        week = int(grp["week"].iloc[0])
        forced = {}
        if len(ov):
            m = ov[(ov["team"] == team)
                   & (ov["season"].isin(["", str(season)]))
                   & (ov["week"].isin(["", str(week)]))]
            for _, r in m.iterrows():
                hit = grp[grp["key"] == r["key"]]
                if len(hit):
                    forced[hit.index[0]] = r["role"]
        taken = {v for v in forced.values() if v in WR_SLOTS}
        open_slots = [s for s in WR_SLOTS if s not in taken]
        for idx in grp.index:
            if idx in forced:
                roles[idx] = (forced[idx] if forced[idx] in WR_SLOTS else "WR4+", True)
            elif open_slots:
                roles[idx] = (open_slots.pop(0), False)
            else:
                roles[idx] = ("WR4+", False)
    for idx, (role, flag) in roles.items():
        pg.at[idx, "role"] = role
        pg.at[idx, "overridden"] = flag
    return pg


# --------------------------------------------------------------------------
# Aggregations for the views
# --------------------------------------------------------------------------

def defense_games(pg: pd.DataFrame, weeks: tuple[int, int]) -> pd.DataFrame:
    g = pg[(pg["week"] >= weeks[0]) & (pg["week"] <= weeks[1])]
    return g[["game_id", "week", "defteam", "team"]].drop_duplicates()


def defense_vs_role(pg: pd.DataFrame, weeks: tuple[int, int]) -> pd.DataFrame:
    """Per defense x role: totals and per-game rates (missing slot = 0)."""
    g = pg[(pg["week"] >= weeks[0]) & (pg["week"] <= weeks[1])
           & pg["role"].isin(ROLE_ORDER)]
    n_games = defense_games(pg, weeks).groupby("defteam").size().rename("games")
    tot = g.groupby(["defteam", "role"])[STATS].sum()
    idx = pd.MultiIndex.from_product([n_games.index, ROLE_ORDER], names=["defteam", "role"])
    tot = tot.reindex(idx, fill_value=0).reset_index().merge(n_games.reset_index(), on="defteam")
    for s in STATS:
        tot[f"{s}_pg"] = tot[s] / tot["games"]
    return tot


def league_avg(dvr: pd.DataFrame) -> pd.DataFrame:
    t = dvr.groupby("role")[STATS + ["games"]].sum()
    for s in STATS:
        t[f"{s}_pg"] = t[s] / t["games"]
    return t


def next_opponents(schedules: pl.DataFrame) -> pd.DataFrame:
    s = schedules.filter(pl.col("result").is_null() & (pl.col("game_type") == "REG")).to_pandas()
    if s.empty:
        return pd.DataFrame(columns=["team", "next_opp", "next_week"])
    rows = []
    for _, r in s.iterrows():
        rows.append((r["home_team"], r["away_team"], r["week"]))
        rows.append((r["away_team"], "@" + r["home_team"], r["week"]))
    df = pd.DataFrame(rows, columns=["team", "next_opp", "next_week"])
    return df.sort_values("next_week").drop_duplicates("team")


def defense_game_log(pg: pd.DataFrame, defteam: str, weeks: tuple[int, int]) -> pd.DataFrame:
    g = pg[(pg["defteam"] == defteam) & (pg["week"] >= weeks[0]) & (pg["week"] <= weeks[1])
           & pg["role"].isin(ROLE_ORDER)].copy()
    g["line"] = (g["player"] + " " + g["receptions"].astype(int).astype(str) + "/"
                 + g["targets"].astype(int).astype(str) + ", "
                 + g["yards"].round(0).astype(int).astype(str) + " yds"
                 + g["tds"].map(lambda t: f", {int(t)} TD" if t else ""))
    out = []
    for (week, opp), grp in g.groupby(["week", "team"]):
        row = {"Week": week, "Opp": opp}
        for role in ROLE_ORDER:
            r = grp[grp["role"] == role].sort_values("targets", ascending=False)
            r = r[(r["targets"] > 0) | (role in WR_SLOTS)]
            row[role] = "; ".join(r["line"]) if len(r) else "—"
        out.append(row)
    return pd.DataFrame(out).sort_values("Week")


def current_roles(pg: pd.DataFrame, overrides: pd.DataFrame | None = None,
                  next_week: dict | None = None) -> pd.DataFrame:
    """Projected roles for each team's NEXT game.

    Uses everyone who played in the team's most recent game, ranked by
    target share through that game (blended with last season the same way).
    Missed the last game = not listed, so check injuries before acting.
    """
    rows = []
    last_week = pg.groupby("team")["week"].max()
    for team, wk in last_week.items():
        hist = pg[pg["team"] == team]
        totals = hist.groupby("gsis_id").agg(t=("targets", "sum"), tt=("team_targets", "sum"))
        recent = hist[hist["week"] == wk].drop_duplicates("gsis_id").set_index("gsis_id")
        recent = recent.join(totals)
        prior = recent["prior_share"]
        recent["share_now"] = (recent["t"] + PRIOR_WEIGHT * prior.fillna(0)) / (
            recent["tt"] + PRIOR_WEIGHT * prior.notna())
        recent["share_now"] = recent["share_now"].fillna(0)
        wr = recent[recent["pos"] == "WR"].sort_values(["share_now", "offense_pct"], ascending=False)
        forced = {}
        if overrides is not None and len(overrides):
            nw = str((next_week or {}).get(team, ""))
            m = overrides[(overrides["team"].str.upper().str.strip() == team)
                          & overrides["week"].isin(["", nw])]
            keys = wr["player"].map(_norm)
            for _, o in m.iterrows():
                hit = wr[keys == _norm(o["player"])]
                if len(hit):
                    forced[hit.iloc[0]["player"]] = o["role"].upper().strip()
        open_slots = [s for s in WR_SLOTS if s not in forced.values()]
        for _, r in wr.iterrows():
            slot = forced.get(r["player"]) or (open_slots.pop(0) if open_slots else None)
            if slot in WR_SLOTS:
                rows.append((team, slot, r["player"], r["share_now"]))
        for role, pos in [("TE", ["TE"]), ("RB", ["RB", "FB"])]:
            p = recent[recent["pos"].isin(pos)].sort_values("share_now", ascending=False)
            if len(p):
                rows.append((team, role, p.iloc[0]["player"], p.iloc[0]["share_now"]))
    return pd.DataFrame(rows, columns=["team", "role", "player", "share"])


def team_roles(pg: pd.DataFrame, team: str) -> pd.DataFrame:
    g = pg[(pg["team"] == team) & pg["role"].isin(WR_SLOTS)].copy()
    g["label"] = (g["player"] + " (" + (g["rank_share"] * 100).round(0).astype(int).astype(str)
                  + "%" + g["role_basis"].map({"season": "", "blend": "",
                                                "prior season": " prior",
                                                "snaps only": " snaps"})
                  + (")" ) + g["overridden"].map({True: " ✎", False: ""}))
    t = g.pivot_table(index=["week", "defteam"], columns="role", values="label",
                      aggfunc="first").reset_index()
    t = t.rename(columns={"week": "Week", "defteam": "Opp"})
    return t[["Week", "Opp"] + [c for c in WR_SLOTS if c in t.columns]]
