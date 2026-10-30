import pandas as pd, numpy as np
from scipy.optimize import minimize
F = pd.read_parquet("features.parquet")
F["E_tgt"] = F.est_share*(1 + 0.5*(0.995/F.active_share_sum - 1))*F.est_team_tpg
F["tot_c"] = F.total_line - 44.5
pos_mean = lambda d, c: d.groupby("pos")[c].mean()
def run(train_seasons, test_season):
    f = F[F.season.isin(train_seasons+[test_season])].copy(); tr = f.season.isin(train_seasons).to_numpy(); te = ~tr
    # baselines
    prev = lambda c: (f["p_"+c]/f.p_g.replace(0,np.nan)).where(f.p_g>=4)
    for c,y in [("rec","receptions"),("yds","receiving_yards")]:
        pm = f[tr].groupby("pos")[y].mean()
        f["B0|"+c] = (f["c_"+c]/f.c_g.replace(0,np.nan)).where(f.c_g>0, prev(c)).fillna(f.pos.map(pm))
        f["B1|"+c] = (f["c_"+c] + 3*prev(c).fillna(f.pos.map(pm)*0.6))/(f.c_g+3)
    X = {k: f[k].to_numpy(float) for k in ["E_tgt","team_spread","tot_c","opp_allow_ratio","est_cr","est_ypt","receptions","receiving_yards"]}
    for c, eff, y in [("rec","est_cr","receptions"),("yds","est_ypt","receiving_yards")]:
        f["M0|"+c] = X["E_tgt"]*X[eff]
        for name, use_opp in [("FINAL lines", False), ("FINAL lines+oppvol", True)]:
            def mu(z): return X["E_tgt"]*X[eff]*np.exp(z[0]*X["team_spread"]+z[1]*X["tot_c"]+(z[2] if use_opp else 0)*np.log(X["opp_allow_ratio"]))
            Y = X[y]
            z = minimize(lambda z: np.mean((mu(z)-Y*np.log(np.maximum(mu(z),1e-6)))[tr]), np.zeros(3), method="L-BFGS-B").x
            f[f"{name}|{c}"] = mu(z)
            if c=="yds" and name=="FINAL lines+oppvol": print("  coefs (yds, lines+opp):", np.round(z,4))
    rel = te & (f.E_tgt>=3.5).to_numpy()
    rows=[]
    for m in ["B0","B1","M0","FINAL lines","FINAL lines+oppvol"]:
        r={}
        for nm, mask in [("all",te),("rel",rel)]:
            for c,y in [("rec","receptions"),("yds","receiving_yards")]:
                e=(f[f"{m}|{c}"]-f[y])[mask]; r[f"{nm} {c} RMSE"]=np.sqrt((e**2).mean()); r[f"{nm} {c} MAE"]=e.abs().mean()
        rows.append(dict(model=m, **r))
    print(f"TEST {test_season} (train {train_seasons}) relevant n={rel.sum()}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return f
pd.set_option("display.width",250)
run([2023],2024); f = run([2023,2024],2025)
f.to_parquet("final_preds.parquet")
