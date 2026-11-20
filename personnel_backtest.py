"""Fit and backtest missing-starter adjustments (all units, both sides of the ball).

For each market (and position group), multiplier = exp(sum b_u * count_u) on top of the current
projection, where count_u = missing starters at unit u (own T/IOL, opposing EDGE/IDL/LB/CB/S, capped
at 2). b fit by ridge-penalized Poisson loss with the projection as offset (pulls weak effects to 0).
Out of sample: fit 2023-24 -> test 2025-26. Then production fit on everything -> model/personnel.json.
    python model/personnel_backtest.py [--save]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import odds as od
import personnel as pe
import props_model as pm

C = json.loads((ROOT / "model" / "coefs.json").read_text())
DIST = json.loads((ROOT / "model" / "dist_specs.json").read_text())
F = pe.FEATURES
# (key, market for apply, actual column, row filter)
SPECS = [
    ("receptions|WR/TE", "receptions", "receptions", lambda d: d["pos"].isin(["WR", "TE"]) & (d["E_tgt"] >= 1)),
    ("receptions|RB", "receptions", "receptions", lambda d: d["pos"].eq("RB") & (d["E_tgt"] >= 1)),
    ("rec_yards|WR/TE", "rec_yards", "receiving_yards", lambda d: d["pos"].isin(["WR", "TE"]) & (d["E_tgt"] >= 1)),
    ("rec_yards|RB", "rec_yards", "receiving_yards", lambda d: d["pos"].eq("RB") & (d["E_tgt"] >= 1)),
    ("carries|RB", "carries", "carries", lambda d: d["pos"].eq("RB") & (d["E_car0"] >= 2)),
    ("rush_yards|RB", "rush_yards", "rushing_yards", lambda d: d["pos"].eq("RB") & (d["E_car0"] >= 2)),
    ("pass_yards|QB", "pass_yards", "passing_yards", lambda d: pd.Series(True, index=d.index)),
]


def fit(m0, X, y, lam):
    def f(b):
        z = X @ b
        return np.mean(m0 * np.exp(z) - y * z) + lam * np.sum(b ** 2)
    def g(b):
        z = X @ b
        return X.T @ (m0 * np.exp(z) - y) / len(y) + 2 * lam * b
    return minimize(f, np.zeros(X.shape[1]), jac=g, method="L-BFGS-B").x


def data():
    P = pd.read_pickle("/tmp/featP_v2.pkl")
    Q = pd.read_pickle("/tmp/featQ_all.pkl")
    M = pd.read_pickle("/tmp/pers_counts.pkl")
    P = P.merge(M, on=["game_id", "team"], how="left")
    Q = Q.merge(M, on=["game_id", "team"], how="left")
    for d in (P, Q):
        d[F] = d[F].fillna(0)
    return P, Q


def rows(P, Q, key, mk, y, filt):
    d = (Q.assign(pos="QB") if key.endswith("QB") else P)
    d = d.dropna(subset=pm.SPEC[mk][0] + [y])
    d = d[filt(d)].copy()
    d["m0"] = np.maximum(pm.apply(d, mk, C[mk]), 1e-3)
    return d


if __name__ == "__main__":
    P, Q = data()
    pd.set_option("display.width", 250)
    lams = [0.0, 0.002, 0.01, 0.05]
    res, coefs = [], {}
    for key, mk, y, filt in SPECS:
        d = rows(P, Q, key, mk, y, filt)
        tr, te = d[d["season"] <= 2024], d[d["season"] >= 2025]
        has = te[F].sum(axis=1) > 0
        base = np.abs(te[y] - te["m0"])
        r = {"market": key, "n_test": len(te), "n_test_affected": int(has.sum()),
             "MAE before": base.mean(), "MAE before (affected)": base[has].mean()}
        for lam in lams:
            b = fit(tr["m0"].to_numpy(), tr[F].to_numpy(float), tr[y].to_numpy(float), lam)
            m1 = te["m0"] * np.exp(te[F].to_numpy(float) @ b)
            e = np.abs(te[y] - m1)
            r[f"MAE lam={lam}"] = e.mean(); r[f"aff lam={lam}"] = e[has].mean()
            coefs[(key, lam)] = b
        res.append(r)
    R = pd.DataFrame(res).set_index("market")
    print("Out-of-sample MAE (fit 2023-24, test 2025-26): all rows, and rows with any starter missing")
    print(R.round(3).to_string())
    for lam in [0.002, 0.01]:
        print(f"\nFitted effects (fit 2023-24, lambda={lam}), % change per missing starter:")
        print(pd.DataFrame({k: (np.exp(coefs[(k, lam)]) - 1) * 100 for k, *_ in SPECS}, index=F).round(1).to_string())

    # production fit on all seasons (lambda from argv or 0.01)
    lam = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--lam=")), 0.01))
    prod = {}
    for key, mk, y, filt in SPECS:
        d = rows(P, Q, key, mk, y, filt)
        prod[key] = dict(zip(F, np.round(fit(d["m0"].to_numpy(), d[F].to_numpy(float), d[y].to_numpy(float), lam), 5)))
    print(f"\nProduction (all seasons, lambda={lam}) % per missing starter:")
    print(pd.DataFrame({k: {f: round((np.exp(v) - 1) * 100, 1) for f, v in b.items()} for k, b in prod.items()}).to_string())

    # DraftKings lines, 2025-26 out of sample (fit 2023-24, same lambda)
    gp = Path("/tmp/graded.csv")
    if gp.exists():
        G = pd.read_csv(gp)
        G = G[G["result"] != "push"].dropna(subset=["line", "over", "under"])
        out = []
        for key, mk, y, filt in SPECS:
            gmk = {"rec_yards": "rec_yards", "rush_yards": "rush_yards", "pass_yards": "pass_yards"}.get(mk)
            if gmk is None:
                continue
            d = rows(P, Q, key, mk, y, filt)
            tr = d[d["season"] <= 2024]
            b = fit(tr["m0"].to_numpy(), tr[F].to_numpy(float), tr[y].to_numpy(float), lam)
            d = d[d["season"] >= 2025].assign(mult=np.exp(d.loc[d["season"] >= 2025, F].to_numpy(float) @ b))
            idc = "qb_id" if key.endswith("QB") else "player_id"
            x = G[G["market"] == gmk].merge(d[["game_id", idc, "mult"] + F].rename(columns={idc: "player_id"}),
                                            on=["game_id", "player_id"])
            for v, mu in [("before", x["mu"]), ("after", x["mu"] * x["mult"])]:
                p = np.array([float(pm.p_over(DIST.get(dd, DIST["rush_yards"]), m, l))
                              for dd, m, l in zip(x["dist"], mu, x["line"])])
                e = p - x["book_p"].to_numpy()
                pick = np.where(e >= 0, "over", "under")
                dec = od.american_to_decimal(np.where(pick == "over", x["over"], x["under"]))
                prof = np.where(x["result"] == pick, dec - 1, -1.0)
                aff = x[F].sum(axis=1).to_numpy() > 0
                for nm, m in [("all", np.ones(len(x), bool)), ("starter missing", aff)]:
                    for thr in [0.05, 0.20]:
                        k = m & (np.abs(e) >= thr)
                        out.append(dict(market=key, set=nm, edge=f">={thr:.0%}", v=v, n=int(k.sum()),
                                        roi=prof[k].mean() if k.sum() else np.nan))
        O = pd.DataFrame(out).pivot_table(index=["market", "set", "edge"], columns="v", values=["n", "roi"])
        print("\nDK lines 2025-26 (bet the model's side):"); print(O.round(3).to_string())

    if "--save" in sys.argv:
        (ROOT / "model" / "personnel.json").write_text(json.dumps({"lambda": lam, "coefs": prod}, indent=1))
        print("saved model/personnel.json")
