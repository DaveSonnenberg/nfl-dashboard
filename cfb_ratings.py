"""College football matchups: opponent-adjusted rush/pass ratings, same model as game_ratings.py.

Every play: yards = avg + offense + defense + home field, ridge (LAMBDA = 400 plays), per kind.
Garbage time = score margin past 43/37/27/21 points in Q1-Q4. (The NFL tab's win-probability rule
drops ~half of college plays, because ESPN's win prob starts from the spread: a 30-point favorite is
"garbage time" from kickoff. Score rule tracked the market better in the backtest.)
Difference from the NFL tab: each team starts at RHO x last season's rating instead of at average
(about 12 games a season is too few). Backtest 2022-26 (model/cfb_backtest.py): 400 / 0.5 was best.
Data: sportsdataverse ESPN play-by-play, schedules and teams (GitHub releases, free).
Lines for upcoming games: The Odds API, DraftKings spreads + totals, 2 credits per pull, by button.
Played games show ESPN's line.
"""
from __future__ import annotations

import difflib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import scipy.sparse as sp
import streamlit as st

LAMBDA, RHO = 400, 0.5
KINDS = {"rush": "Run", "pass": "Pass"}
UNIT = {"rush": "yds/carry", "pass": "yds/dropback"}
REL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
PBP_COLS = ["week", "game_id", "pos_team", "def_pos_team", "pass", "rush", "kneel_down", "scrimmage_play",
            "statYardage", "wp_before", "start.is_home", "type.text", "seasonType", "homeTeamSpread", "overUnder", "period", "pos_score_diff_start"]
GT = {1: 43, 2: 37, 3: 27, 4: 21}   # garbage time: lead bigger than this by quarter (Connelly); OT never
SKIP = {"Penalty", "Timeout", "Two Point Pass", "Two Point Rush", "Defensive 2pt Conversion", "Unknown"}
ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
CACHE = Path(__file__).parent / ".cfb_odds_cache.json"
MAX_AGE_H = 12
POST = 99  # week code for bowls / playoff


def _parquet(url: str, columns=None) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    f = pq.ParquetFile(io.BytesIO(r.content))
    cols = [c for c in (columns or f.schema_arrow.names) if c in f.schema_arrow.names]
    return f.read(columns=cols).to_pandas()


@st.cache_data(ttl=6 * 3600, show_spinner="Loading college schedule…")
def load_schedule(season: int) -> pd.DataFrame:
    s = _parquet(f"{REL}/cfb_schedules/cfb_schedules_{season}.parquet")
    if s.empty:
        return s
    s["wk"] = np.where(s["season_type"] == "regular", s["week"], POST).astype(int)
    s["kick"] = pd.to_datetime(s["start_date"], utc=True).dt.tz_convert("America/Chicago")
    t = load_teams(season)
    full = dict(zip(t["team_id"], t["display_name"])) if len(t) else {}
    for side in ("home", "away"):   # play-by-play uses full names ("Alabama Crimson Tide")
        s[f"{side}_name"] = s[f"{side}_id"].map(full).fillna(s[f"{side}_team"])
    return s[s["season_type"].isin(["regular", "post", "postseason"])]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_teams(season: int) -> pd.DataFrame:
    t = _parquet(f"{REL}/espn_cfb_teams/cfb_teams_{season}.parquet",
                 ["team_id", "display_name", "location", "school", "mascot", "is_fbs", "conference_short_name"])
    if t.empty and season > 2021:
        return load_teams(season - 1)
    return t


@st.cache_data(ttl=6 * 3600, show_spinner="Loading college play-by-play…")
def _raw_pbp(season: int) -> pd.DataFrame:
    return _parquet(f"{REL}/espn_cfb_pbp/play_by_play_{season}.parquet", PBP_COLS)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def espn_lines(season: int) -> pd.DataFrame:
    """ESPN's stored line per played game: sp = home margin expected (+ = home favored), ou = total."""
    p = _raw_pbp(season)
    if p.empty:
        return pd.DataFrame(columns=["sp", "ou"])
    return p.groupby("game_id").agg(sp=("homeTeamSpread", "first"), ou=("overUnder", "first")).astype(float)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_plays(season: int) -> pd.DataFrame:
    p = _raw_pbp(season)
    if p.empty:
        return p
    p = p[(p["seasonType"] == 2) | (p["seasonType"] == 3)]
    p = p[(p["scrimmage_play"] == True) & (p["kneel_down"] != True) & p["statYardage"].notna()
          & p["pos_team"].notna() & ~p["type.text"].isin(SKIP)
          & (p["pos_score_diff_start"].abs() <= p["period"].map(GT).fillna(999))
          & ~p["type.text"].str.contains("Field Goal|Punt|Kickoff", na=False)]
    p["kind"] = np.where(p["pass"] == 1, "pass", np.where(p["rush"] == 1, "rush", ""))
    p = p[p["kind"] != ""]
    sch = load_schedule(season)
    neutral = set(sch.loc[sch["neutral_site"] == True, "game_id"]) if len(sch) else set()
    wk = sch.set_index("game_id")["wk"] if len(sch) else pd.Series(dtype=float)
    p["wk"] = p["game_id"].map(wk).fillna(p["week"])
    p["home"] = np.where(p["game_id"].isin(neutral), 0.0, np.where(p["start.is_home"] == True, 1.0, -1.0))
    p["y"] = p["statYardage"].astype(float)
    return p[["wk", "game_id", "pos_team", "def_pos_team", "home", "kind", "y"]].reset_index(drop=True)


def _fit(d: pd.DataFrame, prior=None):
    teams = sorted(set(d["pos_team"]) | set(d["def_pos_team"]) | (set(prior[0].index) if prior else set()))
    ix = {t: i for i, t in enumerate(teams)}
    k, n = len(teams), len(d)
    r = np.arange(n)
    X = sp.csr_matrix((np.r_[np.ones(2 * n), d["home"].to_numpy()],
                       (np.r_[r, r, r], np.r_[d["pos_team"].map(ix).to_numpy(),
                                             k + d["def_pos_team"].map(ix).to_numpy(), np.full(n, 2 * k)])),
                      shape=(n, 2 * k + 1))
    y = d["y"].to_numpy(float)
    mu = y.mean() if n else 0.0
    b0 = np.zeros(2 * k + 1)
    if prior:
        b0[:k] = RHO * prior[0].reindex(teams).fillna(0).to_numpy()
        b0[k:2 * k] = RHO * prior[1].reindex(teams).fillna(0).to_numpy()
    if n == 0:
        return mu, pd.Series(b0[:k], teams), pd.Series(b0[k:2 * k], teams), 0.0
    b = np.linalg.solve((X.T @ X).toarray() + LAMBDA * np.eye(2 * k + 1), X.T @ (y - mu) + LAMBDA * b0)
    return mu, pd.Series(b[:k], teams), pd.Series(b[k:2 * k], teams), b[-1]


@st.cache_data(ttl=24 * 3600, show_spinner="Rating last season…")
def _prior(season: int) -> dict:
    P = load_plays(season)
    out = {}
    for kind in KINDS:
        d = P[P["kind"] == kind] if len(P) else P
        if len(d):
            mu, o, dfn, h = _fit(d)
            out[kind] = (o, dfn, mu)
    return out


@st.cache_data(ttl=6 * 3600, show_spinner="Rating offenses and defenses…")
def ratings(season: int, before_week: int, fbs: tuple) -> dict:
    P = load_plays(season)
    P = P[P["wk"] < before_week] if len(P) else P
    pri = _prior(season - 1)
    out = {}
    for kind in KINDS:
        d = P[P["kind"] == kind] if len(P) else P
        pr = pri.get(kind)
        if d.empty and pr is None:
            continue
        mu, off, dfn, home = _fit(d, pr[:2] if pr else None)
        if d.empty:            # week 1: last season's average and home edge
            mu = pr[2]
            home = _fit(load_plays(season - 1).query("kind == @kind"))[3]
        o = d.groupby("pos_team")["y"].agg(["mean", "size"])
        a = d.groupby("def_pos_team")["y"].agg(["mean", "size"])
        T = pd.DataFrame({"off_adj": mu + off, "def_adj": mu + dfn}).join(
            pd.DataFrame({"off_raw": o["mean"], "off_n": o["size"], "def_raw": a["mean"], "def_n": a["size"]}))
        F = T[T.index.isin(fbs)]   # ranks among FBS teams
        T["off_rank"] = F["off_adj"].rank(ascending=False, method="min")   # 1 = best offense
        T["def_rank"] = F["def_adj"].rank(ascending=True, method="min")    # 1 = toughest defense
        T["off_rank_raw"] = F["off_raw"].rank(ascending=False, method="min")
        T["def_rank_raw"] = F["def_raw"].rank(ascending=True, method="min")
        out[kind] = {"mu": mu, "home": home, "T": T, "n_fbs": len(F)}
    return out


# --------------------------------------------------------------------------
# odds (upcoming games)
# --------------------------------------------------------------------------
def _api_key() -> str | None:
    try:
        k = st.secrets.get("ODDS_API_KEY")
    except Exception:
        k = None
    return k or os.environ.get("ODDS_API_KEY")


def fetch_odds(api_key: str) -> dict:
    """All upcoming college games, DraftKings spreads + totals: 2 markets x 1 book = 2 credits."""
    r = requests.get(ODDS_URL, params={"apiKey": api_key, "bookmakers": "draftkings", "markets": "spreads,totals",
                                       "oddsFormat": "american"}, timeout=20)
    if r.status_code >= 400:
        try:
            b = r.json()
            msg = f"{b.get('error_code', '')}: {b.get('message', '')}"
        except ValueError:
            msg = r.text[:200]
        raise RuntimeError(f"Odds API error {r.status_code} {msg}")
    payload = {"pulled_at": time.time(), "events": r.json(),
               "remaining": r.headers.get("x-requests-remaining"), "last": r.headers.get("x-requests-last")}
    try:
        CACHE.write_text(json.dumps(payload))
    except OSError:
        pass
    return payload


def cached_odds() -> dict | None:
    try:
        p = json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return None
    return p if time.time() - p.get("pulled_at", 0) <= MAX_AGE_H * 3600 else None


def parse_odds(payload: dict) -> pd.DataFrame:
    """One row per event: home/away full names, home spread (+ = home favored, ESPN convention), total."""
    rows = []
    for e in (payload or {}).get("events", []):
        row = {"home_name": e["home_team"], "away_name": e["away_team"], "commence": e.get("commence_time"),
               "sp": np.nan, "ou": np.nan}
        for bk in e.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    if mk["key"] == "spreads" and oc["name"] == e["home_team"]:
                        row["sp"] = -float(oc["point"])
                    elif mk["key"] == "totals" and oc["name"] == "Over":
                        row["ou"] = float(oc["point"])
        rows.append(row)
    return pd.DataFrame(rows)


def _match_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Attach Odds API lines to schedule rows by full team name (ESPN display name), fuzzy fallback."""
    if odds.empty:
        return games.assign(o_sp=np.nan, o_ou=np.nan)
    pool = sorted(set(odds["home_name"]) | set(odds["away_name"]))

    def norm(n):
        if n in pool:
            return n
        m = difflib.get_close_matches(n, pool, n=1, cutoff=0.85)
        return m[0] if m else None

    key = {(h, a): (s, o) for h, a, s, o in zip(odds["home_name"], odds["away_name"], odds["sp"], odds["ou"])}
    sps, ous = [], []
    for _, g in games.iterrows():
        h, a = norm(g["home_name"]), norm(g["away_name"])
        v = key.get((h, a))
        if v is None and (a, h) in key:      # neutral sites can list teams the other way round
            s, o = key[(a, h)]
            v = (-s if pd.notna(s) else s, o)
        sps.append(v[0] if v else np.nan)
        ous.append(v[1] if v else np.nan)
    return games.assign(o_sp=sps, o_ou=ous)


# --------------------------------------------------------------------------
# view
# --------------------------------------------------------------------------
def _ord(n):
    if pd.isna(n):
        return "FCS"
    n = int(n)
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _spread_text(sp_, h, a):
    if pd.isna(sp_):
        return "—"
    if sp_ == 0:
        return "PK"
    return f"{h} -{abs(sp_):g}" if sp_ > 0 else f"{a} -{abs(sp_):g}"


def _games(g: pd.DataFrame, R: dict, dated: bool = False) -> pd.DataFrame:
    rows = []
    day = "%a %-m/%-d" if dated else "%a"
    for _, x in g.iterrows():
        a, h = x["away_name"], x["home_name"]
        sa, sh = x["away_team"], x["home_team"]
        hs = 0.0 if x["neutral_site"] else 1.0
        tag = lambda t, rk: f"#{int(rk)} {t}" if pd.notna(rk) and rk <= 25 else t
        sp_ = x["sp"] if pd.notna(x.get("sp")) else x.get("o_sp")
        ou = x["ou"] if pd.notna(x.get("ou")) else x.get("o_ou")
        row = {"Game": f"{tag(sa, x['away_rank'])} {'@' if hs else 'v'} {tag(sh, x['home_rank'])}",
               "Kickoff": f"{x['kick']:{day} %-I:%M %p}" if not x.get("start_time_tbd") else f"{x['kick']:{day}} TBD",
               "Spread": _spread_text(sp_, sh, sa), "Total": ou, "away": a, "home": h, "sa": sa, "sh": sh,
               "conf": {x["home_conference"], x["away_conference"]}}
        for kind in KINDS:
            if kind not in R:
                continue
            k, T = R[kind], R[kind]["T"]
            for side, team, opp, sign in [("Away", a, h, -1), ("Home", h, a, 1)]:
                o_ = T.at[team, "off_adj"] if team in T.index else k["mu"]
                d_ = T.at[opp, "def_adj"] if opp in T.index else k["mu"]
                exp = o_ + d_ - k["mu"] + sign * hs * k["home"]
                row[f"{side} {KINDS[kind].lower()}"] = exp
                row[f"{side} {KINDS[kind].lower()} d"] = exp - k["mu"]
        rows.append(row)
    return pd.DataFrame(rows)


def _detail(R: dict, kind: str, team: str, opp: str, exp: float, tn: str, on: str) -> pd.DataFrame:
    T = R[kind]["T"]
    nan = pd.Series(np.nan, index=T.columns)
    o = T.loc[team] if team in T.index else nan
    d = T.loc[opp] if opp in T.index else nan
    f = lambda v: "—" if pd.isna(v) else f"{v:.2f}"
    return pd.DataFrame({
        "": [f"{tn} {KINDS[kind].lower()} offense", f"{on} {KINDS[kind].lower()} defense", "Expected this game"],
        "Raw": [f(o["off_raw"]), f(d["def_raw"]), ""],
        "Adjusted": [f(o["off_adj"]), f(d["def_adj"]), f"{exp:.2f}"],
        "Rank (adj)": [_ord(o["off_rank"]), _ord(d["def_rank"]), ""],
        "Rank (raw)": [_ord(o["off_rank_raw"]), _ord(d["def_rank_raw"]), ""],
        "Plays": ["0" if pd.isna(o["off_n"]) else f"{o['off_n']:.0f}",
                  "0" if pd.isna(d["def_n"]) else f"{d['def_n']:.0f}", ""],
    }).set_index("")


def render(seasons: list[int]):
    c0, c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1, 4.4], vertical_alignment="bottom")
    season = c0.selectbox("Season", seasons, key="cfb_season")
    s = load_schedule(season)
    if s.empty:
        st.info("No college schedule for this season yet.")
        return
    s = s[(s["home_division"] == "fbs") | (s["away_division"] == "fbs")]
    upcoming = s[~s["completed"].astype(bool)]
    weeks = sorted(s["wk"].unique())
    default_wk = int(upcoming["wk"].min()) if len(upcoming) else int(max(weeks))
    confs = sorted({c for c in pd.concat([s.loc[s.home_division == "fbs", "home_conference"],
                                           s.loc[s.away_division == "fbs", "away_conference"]]).dropna()})
    week = c1.selectbox("Week", weeks, index=weeks.index(default_wk), key="cfb_week",
                        format_func=lambda w: "Bowls" if w == POST else str(w))
    conf = c2.selectbox("Conference", ["All FBS", "Top 25"] + confs, key="cfb_conf")

    key = _api_key()
    payload = cached_odds()
    if key:
        left = (payload or {}).get("remaining", "?")
        if c3.button("Lines", icon=":material/refresh:", width="stretch",
                     help=f"Pull DraftKings spreads and totals for all upcoming games (2 credits). Credits left: {left}."):
            with st.spinner("Pulling lines…"):
                try:
                    payload = fetch_odds(key)
                except Exception as e:
                    st.error(f"Lines pull failed: {e}")
    else:
        c3.button("Lines", icon=":material/refresh:", width="stretch", disabled=True,
                  help="No ODDS_API_KEY in the app's Secrets.")

    fbs = tuple(sorted(set(s.loc[s.home_division == "fbs", "home_name"]) | set(s.loc[s.away_division == "fbs", "away_name"])))
    R = ratings(season, int(week), fbs)
    if not R:
        st.info("No play-by-play yet.")
        return
    g = s[s["wk"] == week].sort_values("kick").copy()
    L = espn_lines(season)
    g = g.join(L, on="game_id") if len(L) else g.assign(sp=np.nan, ou=np.nan)
    g = _match_odds(g, parse_odds(payload) if payload else pd.DataFrame())
    if conf == "Top 25":
        g = g[(g["home_rank"] <= 25) | (g["away_rank"] <= 25)]
    elif conf != "All FBS":
        g = g[(g["home_conference"] == conf) | (g["away_conference"] == conf)]
    if g.empty:
        st.info("No games for this filter.")
        return
    G = _games(g, R, dated=week == POST)
    pulled = f" Lines pulled {pd.to_datetime(payload['pulled_at'], unit='s', utc=True).tz_convert('America/Chicago'):%a %-I:%M %p}." if payload else ""
    thru = "last season's ratings" if week <= 1 else ("the full regular season" if week == POST else f"weeks 1–{week - 1}")
    c4.caption(f"Expected yards per play for each unit, from opponent-adjusted ratings ({thru}, garbage time removed; "
               f"teams start at half of last season's rating). Run = yds/carry, pass = yds/dropback. (±) = vs FBS-game "
               f"average: run {R['rush']['mu']:.2f}, pass {R['pass']['mu']:.2f}. Spread/total: ESPN for played "
               f"games, DraftKings for upcoming.{pulled}")

    cells = ["Away run", "Home run", "Away pass", "Home pass"]
    show = G[["Game", "Kickoff", "Spread", "Total"]].copy()
    for c in cells:
        show[c] = [f"{v:.2f} ({dv:+.2f})" if pd.notna(v) else "—" for v, dv in zip(G[c], G[c + " d"])]

    def shade(col):
        lim = 0.8 if "run" in col.name else 1.2
        out = []
        for v in G[col.name + " d"]:
            if pd.isna(v):
                out.append("")
                continue
            a = min(abs(v) / lim, 1) * 0.55
            out.append(f"background-color: rgba({'46,160,67' if v > 0 else '218,54,51'},{a:.2f})")
        return out

    sty = show.style.apply(shade, subset=cells).format({"Total": "{:.1f}"}, na_rep="—")
    ev = st.dataframe(sty, hide_index=True, width="stretch", key=f"cfb_table_{week}_{conf}",
                      on_select="rerun", selection_mode="single-row",
                      height=min(36 * (len(show) + 1) + 3, 600),
                      column_config={c: st.column_config.TextColumn(
                          help=f"{c.split()[0]} offense's expected {UNIT['rush' if 'run' in c else 'pass']} "
                               "(vs average)") for c in cells})
    sel = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
    if not sel:
        return
    x = G.iloc[sel[0]]
    st.markdown(f"**{x['Game']}** · {x['Spread']}" + (f" · total {x['Total']:.1f}" if pd.notna(x["Total"]) else ""))
    cols = st.columns(2)
    for col, kind in zip(cols, ["rush", "pass"]):
        with col:
            for side, team, opp, tn, on in [("Away", x["away"], x["home"], x["sa"], x["sh"]),
                                            ("Home", x["home"], x["away"], x["sh"], x["sa"])]:
                exp = x[f"{side} {KINDS[kind].lower()}"]
                if pd.isna(exp):
                    continue
                st.caption(f"{tn} {KINDS[kind].lower()} vs {on} ({UNIT[kind]}; ranks of {R[kind]['n_fbs']} FBS)")
                st.dataframe(_detail(R, kind, team, opp, exp, tn, on), width="stretch")
