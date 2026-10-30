"""Rushing projections (carries, rush yards) for RBs, backtested 2024 and 2025.

E[carries] = E[team non-QB carries] x carry share (recency-weighted, shrunk,
             partially renormalized over the active backfield)
E[yards]   = E[carries] x shrunk YPC
Team carries adjusted by spread/total and (optionally) opponent run/pass defense.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

D = Path(__file__).parent
r = pd.read_parquet(D / "rpg.parquet")
t = pd.read_parquet(D / "rtg.parquet")
x = pd.read_parquet(D / "runpass_x.parquet")[["game_id", "team", "d_rush_z", "d_pass_z"]]
r["pos"] = np.where(r["position"].isin(["RB", "FB"]), "RB", r["position"])
r = r.merge(t[["game_id", "team", "team_spread", "total_line", "defteam"]], on=["game_id", "team"], how="left")
r = r.merge(x, on=["game_id", "team"], how="left")
r[["d_rush_z", "d_pass_z"]] = r[["d_rush_z", "d_pass_z"]].fillna(0)
r = r.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def decayed(df, col, lam, keys):
    out = np.zeros(len(df)); v = df[col].to_numpy(float)
    for idx in df.groupby(keys).indices.values():
        a = 0.0
        for i in idx:
            out[i] = a; a = a * lam + v[i]
    return out


# ---- team expected non-QB carries (team history shrunk to prior season / league) + opp allowed
t = t.sort_values(["team", "season", "week"]).reset_index(drop=True)
g = t.groupby(["team", "season"])
t["c_car"] = g["team_carries"].cumsum() - t["team_carries"]; t["c_g"] = g.cumcount()
prev = t.groupby(["team", "season"])["team_carries"].mean().rename("p_cpg").reset_index(); prev["season"] += 1
t = t.merge(prev, on=["team", "season"], how="left")
lg = t.groupby("season")["team_carries"].mean()
t["lg_cpg"] = t["season"].map(lg.shift(1)).fillna(lg.mean())
t["est_team_cpg"] = (t["c_car"] + 4 * (0.5 * t["p_cpg"].fillna(t["lg_cpg"]) + 0.5 * t["lg_cpg"])) / (t["c_g"] + 4)
dd = t[["game_id", "season", "week", "defteam", "team_carries", "team_rush_yds"]].sort_values(["defteam", "season", "week"])
gd = dd.groupby(["defteam", "season"])
dd["d_car"] = gd["team_carries"].cumsum() - dd["team_carries"]; dd["d_g"] = gd.cumcount()
dd["d_yds"] = gd["team_rush_yds"].cumsum() - dd["team_rush_yds"]
t = t.merge(dd[["game_id", "defteam", "d_car", "d_g", "d_yds"]], on=["game_id", "defteam"], how="left")
t["opp_car_ratio"] = ((t["d_car"] + 6 * t["lg_cpg"]) / (t["d_g"] + 6)) / t["lg_cpg"]
lg_ypc = (t.groupby("season")["team_rush_yds"].sum() / t.groupby("season")["team_carries"].sum())
t["lg_ypc"] = t["season"].map(lg_ypc.shift(1)).fillna(lg_ypc.mean())
t["opp_ypc_ratio"] = ((t["d_yds"] + 150 * t["lg_ypc"]) / (t["d_car"] + 150)) / t["lg_ypc"]
r = r.merge(t[["game_id", "team", "est_team_cpg", "opp_car_ratio", "opp_ypc_ratio"]], on=["game_id", "team"], how="left")


def build(lam, K, Kypc, alpha, wprior=0.5):
    f = r.copy()
    gp = f.groupby(["player_id", "season"])
    f["c_g"] = gp.cumcount()
    for c in ["carries", "rushing_yards", "team_carries"]:
        f["c_" + c] = gp[c].cumsum() - f[c]
    f["d_car_p"] = decayed(f, "carries", lam, ["player_id", "season"])
    f["d_car_t"] = decayed(f, "team_carries", lam, ["player_id", "season"])
    ps = f.groupby(["player_id", "season"]).agg(p_g=("game_id", "size"), p_car=("carries", "sum"),
                                                p_yds=("rushing_yards", "sum"), p_tc=("team_carries", "sum")).reset_index()
    ps["season"] += 1
    f = f.merge(ps, on=["player_id", "season"], how="left").fillna({"p_g": 0, "p_car": 0, "p_yds": 0, "p_tc": 0})
    has = f["p_g"] >= 4
    p_share = (f["p_car"] / f["p_tc"].replace(0, np.nan)).where(has)
    fb = np.where(f["pos"] == "RB", 0.10, 0.005)
    prior = p_share.fillna(pd.Series(fb, index=f.index))
    kp = np.where(has, K, 20)
    f["est_share"] = (f["d_car_p"] + kp * prior) / (f["d_car_t"] + kp)
    tot = f.groupby(["game_id", "team"])["est_share"].transform("sum")
    f["share_adj"] = f["est_share"] * (1 + alpha * (1.0 / tot - 1))
    base = f[f["season"] < 2023]
    lg_ypc = base["rushing_yards"].sum() / base["carries"].sum()
    rb_ypc = base[base.pos == "RB"]["rushing_yards"].sum() / base[base.pos == "RB"]["carries"].sum()
    n = f["c_carries"] + wprior * f["p_car"]
    f["est_ypc"] = (f["c_rushing_yards"] + wprior * f["p_yds"] + Kypc * rb_ypc) / (n + Kypc)
    f["E_car0"] = f["share_adj"] * f["est_team_cpg"]
    # baselines
    prevpg = lambda c: (f["p_" + c] / f["p_g"].replace(0, np.nan)).where(has)
    for c, src, pc in [("car", "carries", "car"), ("yds", "rushing_yards", "yds")]:
        cur = f["c_" + src] / f["c_g"].replace(0, np.nan)
        f["B0|" + c] = cur.where(f["c_g"] > 0, prevpg(pc)).fillna(0)
        f["B1|" + c] = (f["c_" + src] + 3 * prevpg(pc).fillna(0)) / (f["c_g"] + 3)
    return f


def evaluate(f, train, test, verbose=True):
    f = f[f["season"].isin(train + [test]) & (f["pos"] == "RB")].copy()
    tr = f["season"].isin(train).to_numpy(); te = ~tr
    f["tot_c"] = f["total_line"] - 44.5
    A = {c: f[c].to_numpy(float) for c in ["E_car0", "team_spread", "tot_c", "opp_car_ratio", "d_rush_z", "d_pass_z",
                                           "est_ypc", "opp_ypc_ratio", "carries", "rushing_yards"]}
    variants = {"M0 share×team carries": [], "+ lines": [0, 1], "+ lines + opp carries allowed": [0, 1, 2],
                "+ lines + opp + run/pass D quality": [0, 1, 2, 3, 4],
                "+ all + opp YPC allowed": [0, 1, 2, 3, 4, 5]}

    def mu(p, which):
        car = A["E_car0"] * np.exp(p[0] * A["team_spread"] + p[1] * A["tot_c"] + p[2] * np.log(A["opp_car_ratio"])
                                   + p[3] * A["d_rush_z"] + p[4] * A["d_pass_z"])
        return car if which == "car" else car * A["est_ypc"] * A["opp_ypc_ratio"] ** p[5]
    for name, act in variants.items():
        for which, y in [("car", "carries"), ("yds", "rushing_yards")]:
            p = np.zeros(6)
            if act:
                Y = A[y]
                def loss(z):
                    q = p.copy(); q[act] = z; m = np.maximum(mu(q, which), 1e-6)
                    return np.mean((m - Y * np.log(m))[tr])
                p[act] = minimize(loss, np.zeros(len(act)), method="L-BFGS-B").x
            f[f"{name}|{which}"] = mu(p, which)
            if verbose and name.startswith("+ all") and which == "yds":
                print("   coefs [spread, total, opp_car, d_rush_z, d_pass_z, opp_ypc]:", np.round(p, 3))
    rel = te & (f["E_car0"] >= 8).to_numpy()
    rows = []
    for m in ["B0", "B1"] + list(variants):
        row = {"model": m}
        for nm, mask in [("all", te), ("rel", rel)]:
            for c, y in [("car", "carries"), ("yds", "rushing_yards")]:
                e = (f[f"{m}|{c}"] - f[y])[mask]
                row[f"{nm} {c} RMSE"] = np.sqrt((e ** 2).mean()); row[f"{nm} {c} MAE"] = e.abs().mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    if verbose:
        print(f"TEST {test} (train {train}); RB games {te.sum()}, relevant (proj >= 8 carries) {rel.sum()}")
        print(out.round(3).to_string(index=False))
    return out, f


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    if sys.argv[1:] == ["tune"]:
        res = []
        for lam in [1.0, 0.8, 0.7, 0.6]:
            for K in [20, 40, 70]:
                for alpha in [0.0, 0.5, 1.0]:
                    f = build(lam, K, 150, alpha)
                    o, _ = evaluate(f, [2023], 2024, verbose=False)
                    o2, _ = evaluate(f, [2022], 2023, verbose=False)
                    res.append(dict(lam=lam, K=K, alpha=alpha,
                                    car_rmse=(o.iloc[2]["rel car RMSE"] + o2.iloc[2]["rel car RMSE"]) / 2))
        print(pd.DataFrame(res).sort_values("car_rmse").head(8).round(4).to_string(index=False))
        for Kypc in [50, 150, 300, 600]:
            f = build(0.7, 40, Kypc, 0.5)
            o, _ = evaluate(f, [2023], 2024, verbose=False); o2, _ = evaluate(f, [2022], 2023, verbose=False)
            print("Kypc", Kypc, round((o.iloc[2]["rel yds RMSE"] + o2.iloc[2]["rel yds RMSE"]) / 2, 3))
    else:
        lam, K, Kypc, alpha = [float(a) for a in sys.argv[1:5]]
        f = build(lam, K, Kypc, alpha)
        evaluate(f, [2023], 2024)
        _, ff = evaluate(f, [2023, 2024], 2025)
        ff.to_parquet(D / "rush_preds.parquet")
