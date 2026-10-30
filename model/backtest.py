"""Backtest: train on 2023-2024, test on 2025 (every regular-season week).

Compares naive averages, a hand-built volume x share x efficiency model, that
model with fitted opponent / man-zone / line multipliers, and gradient-boosted
models with feature-group ablations.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

D = Path(__file__).parent
f = pd.read_parquet(D / "features.parquet")
f = f[f["season"].between(2023, 2025)].copy()
TRAIN = f["season"] < 2025
TEST = ~TRAIN

# ---------------- baselines
pos_rec = f[TRAIN].groupby("pos")["receptions"].mean()
pos_yds = f[TRAIN].groupby("pos")["receiving_yards"].mean()


def naive(stat, pstat, pos_mean):
    cur = f["c_" + stat] / f["c_g"].replace(0, np.nan)
    prev = f["p_" + stat] / f["p_g"].replace(0, np.nan)
    out = cur.where(f["c_g"] > 0, prev)
    return out.fillna(f["pos"].map(pos_mean))


def blended(stat, pos_mean, k=3):
    prev = (f["p_" + stat] / f["p_g"].replace(0, np.nan)).where(f["p_g"] >= 4)
    prior = prev.fillna(f["pos"].map(pos_mean) * 0.6)
    return (f["c_" + stat] + k * prior) / (f["c_g"] + k)


f["B0_rec"], f["B0_yds"] = naive("rec", None, pos_rec), naive("yds", None, pos_yds)
f["B1_rec"], f["B1_yds"] = blended("rec", pos_rec), blended("yds", pos_yds)

# ---------------- decomposition
f["E_tgt0"] = f["renorm_share"] * f["est_team_tpg"]
f["M1_rec"] = f["E_tgt0"] * f["est_cr"]
f["M1_yds"] = f["E_tgt0"] * f["est_ypt"]

# fitted multipliers on TRAIN (Poisson-style loss for rec; squared error for yds)
f["tot_c"] = f["total_line"] - 44.5


A = {c: f[c].to_numpy(float) for c in ["E_tgt0", "team_spread", "tot_c", "opp_allow_ratio",
     "opp_role_mult", "mz_tgt_x", "est_cr", "opp_cr_mult", "est_ypt", "opp_ypt_mult", "mz_ypt_x",
     "receptions", "receiving_yards"]}
TR = TRAIN.to_numpy()


def mult(p, which):
    a_sp, a_tot, a_role, a_cr, a_ypt, a_mzt, a_mzy, a_opp = p
    tg = (A["E_tgt0"] * np.exp(a_sp * A["team_spread"] + a_tot * A["tot_c"]
          + a_opp * np.log(A["opp_allow_ratio"]) + a_role * np.log(A["opp_role_mult"])
          + a_mzt * A["mz_tgt_x"]))
    if which == "rec":
        return tg * A["est_cr"] * A["opp_cr_mult"] ** a_cr
    return tg * A["est_ypt"] * A["opp_ypt_mult"] ** a_ypt * np.exp(a_mzy * A["mz_ypt_x"])


def fit(active):
    p0 = np.zeros(8)
    if not active:
        return p0, p0
    out = []
    for which in ["rec", "yds"]:
        y = A["receptions" if which == "rec" else "receiving_yards"][TR]
        def loss(z):
            p = p0.copy(); p[active] = z
            mu = np.maximum(mult(p, which)[TR], 1e-6)
            return np.mean(mu - y * np.log(mu))  # Poisson deviance (up to constants); fine for yards too as quasi-Poisson
        res = minimize(loss, np.zeros(len(active)), method="L-BFGS-B")
        p = p0.copy(); p[active] = res.x
        out.append(p)
    return out[0], out[1]


VARIANTS = {
    "M1 volume×share×efficiency": [],
    "M1 + game lines": [0, 1],
    "M1 + lines + opp volume": [0, 1, 7],
    "M1 + lines + opp funnel/eff": [0, 1, 7, 2, 3, 4],
    "M1 + lines + opp + man/zone": [0, 1, 7, 2, 3, 4, 5, 6],
}
params = {}
for name, act in VARIANTS.items():
    pr, py = fit(act)
    params[name] = (pr, py)
    f[name + "|rec"] = mult(pr, "rec")
    f[name + "|yds"] = mult(py, "yds")

# ---------------- gradient boosting with feature groups
HIST = ["est_share", "renorm_share", "c_share", "l3_share", "p_share", "c_g", "p_g", "hist_tgt",
        "est_cr", "est_ypt", "est_adot", "last_snap", "p_snap", "est_team_tpg", "active_share_sum",
        "vacated", "pos_rank"]
LINES = ["team_spread", "total_line", "implied_total", "home"]
OPP = ["opp_allow_ratio", "opp_role_mult", "opp_cr_mult", "opp_ypt_mult"]
MZ = ["mz_tgt_ratio", "mz_ypt_ratio", "opp_man_z", "mz_tgt_x", "mz_ypt_x", "mz_db"]
f["pos_code"] = f["pos"].map({"WR": 0, "TE": 1, "RB": 2})
f["home"] = f["home"].astype(float)
GROUPS = {"GBM history": HIST, "GBM + lines": HIST + LINES, "GBM + lines + opp": HIST + LINES + OPP,
          "GBM + lines + opp + man/zone": HIST + LINES + OPP + MZ}
for name, cols in GROUPS.items():
    X = f[cols + ["pos_code"]].astype(float)
    for tgt, loss in [("rec", "poisson"), ("yds", "squared_error")]:
        y = f["receptions" if tgt == "rec" else "receiving_yards"]
        m = HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.04, max_leaf_nodes=24,
                                          min_samples_leaf=80, l2_regularization=1.0, random_state=0)
        m.fit(X[TRAIN], y[TRAIN])
        f[f"{name}|{tgt}"] = m.predict(X)

# ---------------- scoring
f["relevant"] = (f["E_tgt0"] >= 3.5)  # pre-game: projected >= 3.5 targets
models = ["B0 season avg", "B1 blended avg"] + list(VARIANTS) + list(GROUPS)
f["B0 season avg|rec"], f["B0 season avg|yds"] = f["B0_rec"], f["B0_yds"]
f["B1 blended avg|rec"], f["B1 blended avg|yds"] = f["B1_rec"], f["B1_yds"]


def score(mask):
    rows = []
    for m in models:
        r, y = f.loc[mask, m + "|rec"], f.loc[mask, "receptions"]
        a, b = f.loc[mask, m + "|yds"], f.loc[mask, "receiving_yards"]
        rows.append({"model": m, "rec MAE": (r - y).abs().mean(), "rec RMSE": np.sqrt(((r - y) ** 2).mean()),
                     "rec bias": (r - y).mean(), "yds MAE": (a - b).abs().mean(),
                     "yds RMSE": np.sqrt(((a - b) ** 2).mean()), "yds bias": (a - b).mean(),
                     "rec corr": np.corrcoef(r, y)[0, 1], "yds corr": np.corrcoef(a, b)[0, 1]})
    return pd.DataFrame(rows)


pd.set_option("display.width", 250)
print("TEST 2025, all player-games:", int(TEST.sum()))
print(score(TEST).round(3).to_string(index=False))
print("\nTEST 2025, prop-relevant (projected >= 3.5 targets):", int((TEST & f["relevant"]).sum()))
print(score(TEST & f["relevant"]).round(3).to_string(index=False))
print("\nfitted multipliers [spread, total, role, cr, ypt, mz_tgt, mz_ypt, opp_vol]:")
for k, v in params.items():
    print(f"  {k}: rec {np.round(v[0], 3)} | yds {np.round(v[1], 3)}")
f.to_parquet(D / "preds.parquet")
