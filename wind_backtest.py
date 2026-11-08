"""Backtest the wind adjustment, then save production coefficients to model/weather.json.

Out of sample: wind coefficients fit on 2023-24 -> tested on 2025; fit on 2023-25 -> tested on 2026.
Accuracy (RMSE, bias) on all games and on windy games (wind > 10 mph outdoors), and the DraftKings
line grading (graded.csv from grade_lines.py) with and without wind.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import props_model as pm
import odds as od

pm.WIND.clear()
SEASONS = [2023, 2024, 2025, 2026]
TESTS = {2025: [2023, 2024], 2026: [2023, 2024, 2025]}
PLAYER = {"receptions": "receptions", "rec_yards": "receiving_yards", "carries": "carries", "rush_yards": "rushing_yards"}
QB = {"pass_yards": "passing_yards", "qb_carries": "carries", "qb_rush_yards": "rushing_yards"}


def data(cache="/tmp/wind_build.pkl"):
    p = Path(cache)
    if p.exists():
        return pd.read_pickle(p)
    B = pm.build(SEASONS, include_upcoming=False, raw=pm.load_raw(SEASONS))
    P = B["players"][~B["players"]["upcoming"]]
    Q = B["qbs"][B["qbs"]["played"]]
    pd.to_pickle((P, Q), p)
    return P, Q


def rows(df, mk, y, filt):
    d = df[filt(df)]
    return d.dropna(subset=pm.SPEC[mk][0] + [y])


if __name__ == "__main__":
    P, Q = data()
    calm = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--calm=")), pm.weather.CALM))
    pm.weather.CALM = calm
    pm.weather.CAP = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--cap=")), pm.weather.CAP))
    print("cap:", pm.weather.CAP)
    P = P.assign(wind_x=pm.weather.wind_x(P["wind"])); Q = Q.assign(wind_x=pm.weather.wind_x(Q["wind"]))
    print("calm threshold:", calm)
    print("windy share (wind_x>0):", round((P["wind_x"] > 0).mean(), 3), "| games with wind known:",
          round(P["wind"].notna().mean(), 3))
    res, bs = [], {}
    for test, train in TESTS.items():
        for mk, y in {**PLAYER, **QB}.items():
            df = Q if mk in QB else P
            gcol = "c_qg" if mk.startswith("qb_") else "c_g"
            tr = df[df["season"].isin(train) & (df[gcol] >= 1)]
            te = df[df["season"] == test]
            c = pm.fit_coefs(tr, mk, y)
            b = pm.fit_wind(tr, mk, y, c)
            bs[(test, mk)] = b
            te = te.dropna(subset=pm.SPEC[mk][0] + [y])
            if mk in PLAYER:
                te = te[(te["E_tgt"] >= 3) if mk.startswith("rec") else (te["E_car0"] >= 5)]
            m0 = pm.apply(te, mk, c)
            m1 = m0 * np.exp(b * te["wind_x"].fillna(0).to_numpy(float))
            yy = te[y].to_numpy(float)
            for nm, m in [("all", np.ones(len(te), bool)), ("windy", te["wind_x"].to_numpy(float) > 0),
                          ("wind 15+", te["wind"].to_numpy(float) >= 15)]:
                for v, mu in [("no wind", m0), ("wind", m1)]:
                    e = mu[m] - yy[m]
                    res.append(dict(test=test, market=mk, set=nm, variant=v, n=int(m.sum()),
                                    rmse=np.sqrt(np.mean(e ** 2)) if m.sum() else np.nan,
                                    bias=e.mean() if m.sum() else np.nan))
    R = pd.DataFrame(res)
    pd.set_option("display.width", 220)
    print("\nWind coefficient by fit (per mph over 10):")
    print(pd.Series(bs).unstack(0).round(4).to_string())
    print("\nRMSE (no wind -> wind), n:")
    t = R.pivot_table(index=["market", "set"], columns=["test", "variant"], values="rmse")
    n = R[R.variant == "wind"].pivot_table(index=["market", "set"], columns="test", values="n")
    print(pd.concat([t.round(3), n.add_prefix("n_")], axis=1).to_string())
    print("\nBias (projection - actual):")
    print(R.pivot_table(index=["market", "set"], columns=["test", "variant"], values="bias").round(2).to_string())

    # ---- DraftKings line grading with and without wind (rec/rush/pass yards)
    g = Path("/tmp/graded.csv")
    if g.exists():
        G = pd.read_csv(g).dropna(subset=["line", "model_p", "actual", "over", "under"])
        W = pd.concat([P[["game_id", "wind", "wind_x"]], Q[["game_id", "wind", "wind_x"]]]).drop_duplicates("game_id")
        G = G.merge(W, on="game_id", how="left")
        dist = json.loads((ROOT / "model" / "dist_specs.json").read_text())
        key = {"rec_yards": "rec_yards", "rush_yards": "rush_yards", "pass_yards": "pass_yards"}
        b_of = lambda s, mk, qb: bs.get((s, "qb_rush_yards" if (qb and mk == "rush_yards") else mk), 0.0)
        G["mu_w"] = G["mu"] * np.exp([b_of(s, mk, p == "QB") * (x if x == x else 0)
                                       for s, mk, p, x in zip(G["season"], G["market"], G["pos"], G["wind_x"])])
        G["p_w"] = [float(pm.p_over(dist[d], m, l)) for d, m, l in zip(G["dist"], G["mu_w"], G["line"])]
        out = []
        for v, pcol in [("no wind", "model_p"), ("wind", "p_w")]:
            e = G[pcol] - G["book_p"]
            pick = np.where(e >= 0, "over", "under")
            price = np.where(pick == "over", G["over"], G["under"])
            dec = od.american_to_decimal(price)
            prof = np.where(G["result"] == "push", 0.0, np.where(G["result"] == pick, dec - 1, -1.0))
            d = G.assign(edge=np.abs(e), pick=pick, profit=prof, v=v,
                         win=np.where(G["result"] == "push", np.nan, (G["result"] == pick).astype(float)))
            out.append(d)
        D = pd.concat(out)
        D["windy"] = np.where(D["wind"] >= 15, "wind 15+", "other")
        for thr in [0.0, 0.05, 0.20]:
            x = D[D["edge"] >= thr]
            print(f"\nDK lines, edge >= {thr:.0%} (all picks): hit / ROI / n")
            print(x.groupby(["market", "windy", "v"]).agg(n=("profit", "size"), hit=("win", "mean"),
                                                         roi=("profit", "mean")).round(3).unstack("v").to_string())
            u = x[x["pick"] == "under"]
            print("  unders only:", u.groupby(["market", "v"]).agg(n=("profit", "size"), hit=("win", "mean"),
                                                                    roi=("profit", "mean")).round(3).unstack("v").to_string())

    # ---- production coefficients: fit on 2023-25 with the production coefs
    C = json.loads((ROOT / "model" / "coefs.json").read_text())
    prod = {}
    for mk, y in {**PLAYER, **QB}.items():
        df = Q if mk in QB else P
        gcol = "c_qg" if mk.startswith("qb_") else "c_g"
        tr = df[df["season"].isin([2023, 2024, 2025]) & (df[gcol] >= 1)]
        prod[mk] = round(pm.fit_wind(tr, mk, y, C[mk]), 5)
    print("\nProduction wind coefficients:", prod)
    if "--save" in sys.argv:
        (ROOT / "model" / "weather.json").write_text(json.dumps(prod, indent=1))
        print("saved model/weather.json")
