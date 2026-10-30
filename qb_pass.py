"""QB passing yards projection + backtest.

Universe: the starting QB listed in nflverse schedules (known pre-game), every
REG-season game 2021-2025. Outcome: his official passing yards (includes games
he left injured, since props settle on those too).

E[yards] = E[team pass attempts] x QB share of attempts x E[QB yards/attempt]
  team attempts: team history (recency weighted) shrunk to prior season/league,
                 x spread/total x opponent attempts allowed
  QB YPA:        QB's own history (this season decayed + prior seasons), shrunk
                 to a prior that depends on how much history he has, x opponent
                 YPA allowed
"""
import sys
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize

D = Path(__file__).parent
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]


def build_raw():
    rows, teams = [], []
    for s in SEASONS:
        ps = (nfl.load_player_stats([s]).filter(pl.col("season_type") == "REG")
              .select(["player_id", "player_display_name", "position", "game_id", "season", "week", "team",
                       "opponent_team", "attempts", "completions", "passing_yards", "sacks_suffered"]).to_pandas())
        tm = ps.groupby(["game_id", "team"]).agg(team_att=("attempts", "sum"), team_pyds=("passing_yards", "sum"),
                                                 team_sacks=("sacks_suffered", "sum")).reset_index()
        sch = nfl.load_schedules([s]).filter(pl.col("game_type") == "REG").to_pandas()
        st = pd.concat([
            sch[["game_id", "week", "home_team", "away_team", "home_qb_id", "home_qb_name", "spread_line", "total_line"]]
              .rename(columns={"home_team": "team", "away_team": "opp", "home_qb_id": "qb_id", "home_qb_name": "qb"})
              .assign(home=True),
            sch[["game_id", "week", "away_team", "home_team", "away_qb_id", "away_qb_name", "spread_line", "total_line"]]
              .rename(columns={"away_team": "team", "home_team": "opp", "away_qb_id": "qb_id", "away_qb_name": "qb"})
              .assign(home=False)])
        st["team_spread"] = np.where(st["home"], -st["spread_line"], st["spread_line"])
        st["season"] = s
        st = st.merge(tm, on=["game_id", "team"], how="left")
        q = ps[["player_id", "game_id", "attempts", "completions", "passing_yards", "sacks_suffered"]].rename(columns={"player_id": "qb_id"})
        st = st.merge(q, on=["qb_id", "game_id"], how="left")
        st[["attempts", "completions", "passing_yards", "sacks_suffered"]] = st[["attempts", "completions", "passing_yards", "sacks_suffered"]].fillna(0)
        rows.append(st.drop(columns=["spread_line"]))
        # all QB passing rows (for QB history even when not the listed starter)
        ps["season"] = s
        teams.append(ps[ps["attempts"] > 0][["player_id", "game_id", "season", "week", "team", "attempts", "passing_yards"]])
        print("done", s, flush=True)
    G = pd.concat(rows, ignore_index=True)
    G = G[G["team_att"].notna()]
    G.to_parquet(D / "qbg.parquet")
    pd.concat(teams).to_parquet(D / "qb_all.parquet")


def decayed(df, col, lam, keys):
    out = np.zeros(len(df)); v = df[col].to_numpy(float)
    for idx in df.groupby(keys).indices.values():
        a = 0.0
        for i in idx:
            out[i] = a; a = a * lam + v[i]
    return out


def features(lam_t=0.8, k_team=4, lam_q=0.85, k_ypa=150, w_prior=0.5, k_share=6):
    G = pd.read_parquet(D / "qbg.parquet")
    G = G[G["attempts"] > 0].sort_values(["season", "week"]).reset_index(drop=True)  # listed starter who never threw = prop voided
    A = pd.read_parquet(D / "qb_all.parquet")
    lg_att = G.groupby("season")["team_att"].mean(); lg_ypa = G.groupby("season")["team_pyds"].sum() / G.groupby("season")["team_att"].sum()
    G["lg_att"] = G["season"].map(lg_att.shift(1)).fillna(lg_att.mean())
    G["lg_ypa"] = G["season"].map(lg_ypa.shift(1)).fillna(lg_ypa.mean())

    # ---- team attempts: decayed this season, shrunk to 50/50 prior season & league
    G = G.sort_values(["team", "season", "week"]).reset_index(drop=True)
    G["d_att"] = decayed(G, "team_att", lam_t, ["team", "season"])
    G["d_n"] = decayed(G.assign(one=1.0), "one", lam_t, ["team", "season"])
    pt = G.groupby(["team", "season"])["team_att"].mean().rename("p_att").reset_index(); pt["season"] += 1
    G = G.merge(pt, on=["team", "season"], how="left")
    prior = 0.5 * G["p_att"].fillna(G["lg_att"]) + 0.5 * G["lg_att"]
    G["est_att"] = (G["d_att"] + k_team * prior) / (G["d_n"] + k_team)

    # ---- opponent: attempts allowed and YPA allowed (season-to-date + prior season at 1/3, shrunk)
    dd = G[["game_id", "season", "week", "opp", "team_att", "team_pyds"]].sort_values(["opp", "season", "week"])
    gd = dd.groupby(["opp", "season"])
    dd["c_att"] = gd["team_att"].cumsum() - dd["team_att"]; dd["c_n"] = gd.cumcount()
    dd["c_yds"] = gd["team_pyds"].cumsum() - dd["team_pyds"]
    pv = dd.groupby(["opp", "season"]).agg(pa=("team_att", "sum"), pn=("team_att", "size"), py=("team_pyds", "sum")).reset_index(); pv["season"] += 1
    dd = dd.merge(pv, on=["opp", "season"], how="left").fillna({"pa": 0, "pn": 0, "py": 0})
    G = G.merge(dd[["game_id", "opp", "c_att", "c_n", "c_yds", "pa", "pn", "py"]], on=["game_id", "opp"], how="left")
    G["opp_att_ratio"] = ((G["c_att"] + G["pa"] / 3 + 6 * G["lg_att"]) / (G["c_n"] + G["pn"] / 3 + 6)) / G["lg_att"]
    G["opp_ypa_ratio"] = ((G["c_yds"] + G["py"] / 3 + 250 * G["lg_ypa"]) / (G["c_att"] + G["pa"] / 3 + 250)) / G["lg_ypa"]

    # ---- QB history (all his passing, any team): this season decayed, prior 2 seasons
    A = A.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    A["q_yds"] = decayed(A, "passing_yards", lam_q, ["player_id", "season"])
    A["q_att"] = decayed(A, "attempts", lam_q, ["player_id", "season"])
    gq = A.groupby(["player_id", "season"])
    A["c_att_q"] = gq["attempts"].cumsum() - A["attempts"]
    A = A[["player_id", "game_id", "q_yds", "q_att", "c_att_q"]]
    pq = (pd.read_parquet(D / "qb_all.parquet").groupby(["player_id", "season"])
          .agg(y=("passing_yards", "sum"), a=("attempts", "sum")).reset_index())
    p1 = pq.assign(season=pq["season"] + 1).rename(columns={"y": "y1", "a": "a1"})
    p2 = pq.assign(season=pq["season"] + 2).rename(columns={"y": "y2", "a": "a2"})
    G = G.merge(A.rename(columns={"player_id": "qb_id"}), on=["qb_id", "game_id"], how="left")
    G = G.merge(p1, left_on=["qb_id", "season"], right_on=["player_id", "season"], how="left").drop(columns="player_id")
    G = G.merge(p2, left_on=["qb_id", "season"], right_on=["player_id", "season"], how="left").drop(columns="player_id")
    for c in ["q_yds", "q_att", "c_att_q", "y1", "a1", "y2", "a2"]:
        G[c] = G[c].fillna(0)
    hist_att = G["c_att_q"] + G["a1"] + G["a2"]
    # prior for QBs with little history is below league average (backups/rookies)
    G["ypa_prior"] = G["lg_ypa"] * np.where(hist_att >= 300, 1.0, 0.93)
    num = G["q_yds"] + w_prior * G["y1"] + w_prior / 2 * G["y2"] + k_ypa * G["ypa_prior"]
    den = G["q_att"] + w_prior * G["a1"] + w_prior / 2 * G["a2"] + k_ypa
    G["est_ypa"] = num / den
    G["hist_att"] = hist_att
    # starter's share of team attempts (injuries/benchings): league-wide constant from prior seasons
    G["share"] = G["attempts"] / G["team_att"].replace(0, np.nan)
    sh = G.groupby("season")["share"].mean()
    G["est_share"] = G["season"].map(sh.shift(1)).fillna(sh.mean())
    G["E0"] = G["est_att"] * G["est_share"] * G["est_ypa"]
    G["tot_c"] = G["total_line"] - 44.5
    # baselines: QB's own passing yards/game this season (fallback prior season, league)
    gqb = G.sort_values(["qb_id", "season", "week"]).groupby(["qb_id", "season"])
    G = G.sort_values(["qb_id", "season", "week"])
    G["c_g"] = gqb.cumcount(); G["c_y"] = gqb["passing_yards"].cumsum() - G["passing_yards"]
    pg1 = G.groupby(["qb_id", "season"]).agg(pgn=("game_id", "size"), pgy=("passing_yards", "sum")).reset_index(); pg1["season"] += 1
    G = G.merge(pg1, on=["qb_id", "season"], how="left")
    lg_ypg = G.groupby("season")["passing_yards"].mean()
    prev = (G["pgy"] / G["pgn"]).where(G["pgn"] >= 4)
    G["B0"] = (G["c_y"] / G["c_g"].replace(0, np.nan)).fillna(prev).fillna(G["season"].map(lg_ypg.shift(1)).fillna(210))
    G["B1"] = (G["c_y"] + 3 * prev.fillna(G["season"].map(lg_ypg.shift(1)).fillna(210) * 0.95)) / (G["c_g"] + 3)
    return G


VARS = {"M0 attempts×YPA": [], "+ lines": [0, 1], "+ lines + opp attempts allowed": [0, 1, 2],
        "+ lines + opp attempts + opp YPA allowed": [0, 1, 2, 3]}


def fit_eval(G, train, test, verbose=True):
    f = G[G["season"].isin(train + [test])].copy()
    tr = f["season"].isin(train).to_numpy(); te = ~tr
    X = {c: f[c].to_numpy(float) for c in ["E0", "team_spread", "tot_c", "opp_att_ratio", "opp_ypa_ratio", "passing_yards"]}

    def mu(p):
        return X["E0"] * np.exp(p[0] * X["team_spread"] + p[1] * X["tot_c"] + p[2] * np.log(X["opp_att_ratio"])
                                + p[3] * np.log(X["opp_ypa_ratio"]))
    rows = []
    for name, act in VARS.items():
        p = np.zeros(4)
        if act:
            Y = X["passing_yards"]
            def loss(z):
                q = p.copy(); q[act] = z; m = np.maximum(mu(q), 1e-6)
                return np.mean((m - Y * np.log(m))[tr])
            p[act] = minimize(loss, np.zeros(len(act)), method="L-BFGS-B").x
        f[name] = mu(p)
        if verbose and name.startswith("+ lines + opp attempts + opp"):
            print("   coefs [spread, total, opp_att, opp_ypa]:", np.round(p, 3))
    for m in ["B0", "B1"] + list(VARS):
        e = (f[m] - f["passing_yards"])[te]
        rows.append({"model": m, "RMSE": np.sqrt((e ** 2).mean()), "MAE": e.abs().mean(), "bias": e.mean(),
                     "corr": np.corrcoef(f[m][te], f["passing_yards"][te])[0, 1]})
    out = pd.DataFrame(rows)
    if verbose:
        print(f"TEST {test} (train {train}); starter games {te.sum()}")
        print(out.round(2).to_string(index=False))
    return out, f


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    if sys.argv[1:2] == ["build"]:
        build_raw()
    elif sys.argv[1:2] == ["tune"]:
        res = []
        for lam_q in [1.0, 0.85, 0.7]:
            for k_ypa in [75, 150, 300, 500]:
                for w_prior in [0.5, 1.0]:
                    G = features(lam_q=lam_q, k_ypa=k_ypa, w_prior=w_prior)
                    a, _ = fit_eval(G, [2021], 2022, False); b, _ = fit_eval(G, [2022], 2023, False)
                    res.append(dict(lam_q=lam_q, k_ypa=k_ypa, w_prior=w_prior, rmse=(a.iloc[2].RMSE + b.iloc[2].RMSE) / 2))
        print(pd.DataFrame(res).sort_values("rmse").head(6).round(2).to_string(index=False))
        res = []
        for lam_t in [1.0, 0.8, 0.6]:
            for k_team in [2, 4, 8]:
                G = features(lam_t=lam_t, k_team=k_team)
                a, _ = fit_eval(G, [2021], 2022, False); b, _ = fit_eval(G, [2022], 2023, False)
                res.append(dict(lam_t=lam_t, k_team=k_team, rmse=(a.iloc[2].RMSE + b.iloc[2].RMSE) / 2))
        print(pd.DataFrame(res).sort_values("rmse").head(4).round(2).to_string(index=False))
    else:
        kw = dict(a.split("=") for a in sys.argv[1:])
        G = features(**{k: float(v) for k, v in kw.items()})
        fit_eval(G, [2023], 2024)
        _, f = fit_eval(G, [2023, 2024], 2025)
        G.to_parquet(D / "qb_features.parquet")
