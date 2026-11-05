"""Opponent-adjusted defense ratings (research + backtest).

Each play: yards = league avg + offense effect + defense effect + home, fit by weighted ridge
(recency decay by week). Rush = designed runs; pass = dropbacks (incl. sacks, scrambles).
Backtest: ratings through week N-1 -> predict each team-game's yards/play in week N.
"""
import sys
import numpy as np
import pandas as pd
import nflreadpy as nfl
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge


def plays(seasons):
    p = nfl.load_pbp(seasons).to_pandas()
    p = p[(p["season_type"] == "REG") & p["play_type"].isin(["run", "pass"])
          & (p["two_point_attempt"] != 1) & (p["qb_kneel"] != 1) & (p["qb_spike"] != 1)
          & p["yards_gained"].notna() & p["posteam"].notna()]
    p["kind"] = np.where(p["pass"] == 1, "pass", np.where(p["rush"] == 1, "rush", None))
    p = p[p["kind"].notna()]
    p["garbage"] = (p["wp"] < 0.10) | (p["wp"] > 0.90)
    p["home"] = (p["posteam"] == p["home_team"]).astype(float) * 2 - 1
    return p[["season", "week", "game_id", "posteam", "defteam", "home", "kind", "yards_gained", "garbage"]]


def fit(d, lam, decay, cur_week):
    """Weighted ridge on one season's plays of one kind. Returns (mu, off, def, home)."""
    teams = sorted(set(d["posteam"]) | set(d["defteam"]))
    ix = {t: i for i, t in enumerate(teams)}
    n, k = len(d), len(teams)
    rows = np.arange(n)
    Xo = csr_matrix((np.ones(n), (rows, d["posteam"].map(ix))), shape=(n, k))
    Xd = csr_matrix((np.ones(n), (rows, d["defteam"].map(ix))), shape=(n, k))
    X = hstack([Xo, Xd, csr_matrix(d["home"].to_numpy()[:, None])]).tocsr()
    w = decay ** (cur_week - d["week"].to_numpy())
    y = d["yards_gained"].to_numpy(float)
    mu = np.average(y, weights=w)
    m = Ridge(alpha=lam, fit_intercept=False).fit(X, y - mu, sample_weight=w)
    c = m.coef_
    return mu, pd.Series(c[:k], teams), pd.Series(c[k:2 * k], teams), c[-1]


def backtest(P, kind, lam, decay, drop_garbage=True, start=5):
    errs = []
    for s, S in P[P["kind"] == kind].groupby("season"):
        for wk in range(start, S["week"].max() + 1):
            train = S[S["week"] < wk]
            if drop_garbage:
                train = train[~train["garbage"]]
            test = S[(S["week"] == wk) & ~S["garbage"]]
            if test.empty:
                continue
            g = test.groupby(["posteam", "defteam", "home"])["yards_gained"].agg(["mean", "size"]).reset_index()
            # raw: defense's per-play allowed (same filter), unweighted
            raw = train.groupby("defteam")["yards_gained"].mean()
            mu, off, dfn, h = fit(train, lam, decay, wk) if lam is not None else (None,) * 4
            g["raw"] = g["defteam"].map(raw)
            if lam is not None:
                g["adj_def"] = mu + g["defteam"].map(dfn)
                g["adj_full"] = mu + g["posteam"].map(off) + g["defteam"].map(dfn) + h * g["home"]
            g["season"], g["week"] = s, wk
            errs.append(g)
    E = pd.concat(errs)
    out = {}
    for c in [c for c in ["raw", "adj_def", "adj_full"] if c in E]:
        e = (E["mean"] - E[c]) ** 2
        out[c] = np.sqrt(np.average(e, weights=E["size"]))
    return out, E


if __name__ == "__main__":
    P = plays([2023, 2024, 2025])
    P.to_parquet("/tmp/def_plays.parquet")
    for kind in ["rush", "pass"]:
        base, _ = backtest(P, kind, None, 1.0)
        print(kind, "raw RMSE", round(base["raw"], 4))
        res = []
        for lam in [50, 150, 400, 1000, 2500]:
            for decay in [1.0, 0.95, 0.9, 0.85]:
                r, _ = backtest(P, kind, lam, decay)
                res.append((lam, decay, round(r["adj_def"], 4), round(r["adj_full"], 4)))
        R = pd.DataFrame(res, columns=["lam", "decay", "adj_def", "adj_full"]).sort_values("adj_full")
        print(R.head(8).to_string(index=False)); sys.stdout.flush()
