"""Backtest the rain adjustment (outdoor games only), then save production coefficients.

Rain = precipitation in the play-by-play weather text at an outdoors/open stadium (1/0). Wind stays
at its production coefficients. Rain coefficients fit on 2023-24 -> tested on 2025; 2023-25 -> 2026.
Note: the backtest uses OBSERVED rain; live use has only the forecast chance, so expect less.
    python model/rain_backtest.py [--save]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import odds as od
import props_model as pm
import weather

SEASONS = [2023, 2024, 2025, 2026]
TESTS = {2025: [2023, 2024], 2026: [2023, 2024, 2025]}
PLAYER = {"receptions": "receptions", "rec_yards": "receiving_yards", "carries": "carries", "rush_yards": "rushing_yards"}
QB = {"pass_yards": "passing_yards", "qb_carries": "carries", "qb_rush_yards": "rushing_yards"}


def data(cache="/tmp/rain_build.pkl"):
    p = Path(cache)
    if p.exists():
        return pd.read_pickle(p)
    raw = pm.load_raw(SEASONS)
    w = nfl.load_pbp(SEASONS).select(["game_id", "weather", "roof"]).unique(subset=["game_id"]).to_pandas()
    raw["sch"]["rain_obs"] = raw["sch"]["game_id"].map(weather.played_rain(w)).fillna(0.0)
    B = pm.build(SEASONS, include_upcoming=False, raw=raw)
    out = (B["players"][~B["players"]["upcoming"]], B["qbs"][B["qbs"]["played"]])
    pd.to_pickle(out, p)
    return out


if __name__ == "__main__":
    P, Q = data()
    pm.RAIN.clear()
    print("rainy share of player-games:", round((P["rain_x"] > 0).mean(), 3),
          "| by season:", P.groupby("season")["rain_x"].mean().round(3).to_dict())
    res, bs = [], {}
    for test, train in TESTS.items():
        for mk, y in {**PLAYER, **QB}.items():
            df = Q if mk in QB else P
            gcol = "c_qg" if mk.startswith("qb_") else "c_g"
            tr = df[df["season"].isin(train) & (df[gcol] >= 1)]
            c = pm.fit_coefs(tr, mk, y)
            b = pm.fit_wind(tr, mk, y, c, col="rain_x")
            bs[(test, mk)] = b
            te = df[df["season"] == test].dropna(subset=pm.SPEC[mk][0] + [y])
            if mk in PLAYER:
                te = te[(te["E_tgt"] >= 3) if mk.startswith("rec") else (te["E_car0"] >= 5)]
            m0 = pm.apply(te, mk, c)
            m1 = m0 * np.exp(b * te["rain_x"].to_numpy(float))
            yy = te[y].to_numpy(float)
            for nm, m in [("all", np.ones(len(te), bool)), ("rain", te["rain_x"].to_numpy(float) > 0)]:
                for v, mu in [("no rain", m0), ("rain", m1)]:
                    e = mu[m] - yy[m]
                    res.append(dict(test=test, market=mk, set=nm, variant=v, n=int(m.sum()),
                                    rmse=np.sqrt(np.mean(e ** 2)) if m.sum() else np.nan,
                                    bias=e.mean() if m.sum() else np.nan))
    R = pd.DataFrame(res)
    pd.set_option("display.width", 220)
    print("\nRain coefficient (log multiplier when it rains):")
    print(pd.Series(bs).unstack(0).round(4).assign(**{"2025 %": lambda d: (np.exp(d[2025]) - 1).round(3),
                                                       "2026 %": lambda d: (np.exp(d[2026]) - 1).round(3)}).to_string())
    print("\nRMSE / bias (projection - actual), rainy games and all:")
    t = R.pivot_table(index=["market", "set"], columns=["test", "variant"], values=["rmse", "bias"])
    n = R[R.variant == "rain"].pivot_table(index=["market", "set"], columns="test", values="n")
    print(pd.concat([t.round(2), n.add_prefix("n_")], axis=1).to_string())

    g = Path("/tmp/graded.csv")
    if g.exists():
        G = pd.read_csv(g).dropna(subset=["line", "model_p", "actual", "over", "under"])
        W = pd.concat([P[["game_id", "wind_x", "rain_x"]], Q[["game_id", "wind_x", "rain_x"]]]).drop_duplicates("game_id")
        G = G.merge(W, on="game_id", how="left").fillna({"wind_x": 0, "rain_x": 0})
        dist = json.loads((ROOT / "model" / "dist_specs.json").read_text())
        mk_of = lambda mk, pos: "qb_rush_yards" if (pos == "QB" and mk == "rush_yards") else mk
        bw = [pm.WIND.get(mk_of(m, p), 0) * x for m, p, x in zip(G["market"], G["pos"], G["wind_x"])]
        br = [bs.get((s, mk_of(m, p)), 0) * x for s, m, p, x in zip(G["season"], G["market"], G["pos"], G["rain_x"])]
        G["mu0"] = G["mu"] * np.exp(bw)
        G["mu1"] = G["mu0"] * np.exp(br)
        out = []
        for v, col in [("wind only", "mu0"), ("wind + rain", "mu1")]:
            p = np.array([float(pm.p_over(dist[d], m, l)) for d, m, l in zip(G["dist"], G[col], G["line"])])
            e = p - G["book_p"].to_numpy()
            pick = np.where(e >= 0, "over", "under")
            dec = od.american_to_decimal(np.where(pick == "over", G["over"], G["under"]))
            prof = np.where(G["result"] == "push", 0.0, np.where(G["result"] == pick, dec - 1, -1.0))
            out.append(G.assign(v=v, edge=np.abs(e), pick=pick, profit=prof,
                                win=np.where(G["result"] == "push", np.nan, (G["result"] == pick).astype(float))))
        D = pd.concat(out)
        D["wx"] = np.where(D["rain_x"] > 0, "rain", "dry/dome")
        for thr in [0.0, 0.05, 0.20]:
            x = D[D["edge"] >= thr]
            print(f"\nDK lines, edge >= {thr:.0%}: n / hit / ROI")
            print(x.groupby(["market", "wx", "v"]).agg(n=("profit", "size"), hit=("win", "mean"), roi=("profit", "mean"))
                  .round(3).unstack("v").to_string())

    C = json.loads((ROOT / "model" / "coefs.json").read_text())
    prod = {}
    for mk, y in {**PLAYER, **QB}.items():
        df = Q if mk in QB else P
        gcol = "c_qg" if mk.startswith("qb_") else "c_g"
        tr = df[df["season"].isin([2023, 2024, 2025]) & (df[gcol] >= 1)]
        prod[mk] = round(pm.fit_wind(tr, mk, y, C[mk], col="rain_x"), 5)
    print("\nProduction rain coefficients (2023-25):", prod, {k: round(np.exp(v) - 1, 3) for k, v in prod.items()})
    if "--save" in sys.argv:
        path = ROOT / "model" / "weather.json"
        wx = json.loads(path.read_text())
        wx["rain"] = prod
        path.write_text(json.dumps(wx, indent=1))
        print("saved rain coefficients to model/weather.json")
