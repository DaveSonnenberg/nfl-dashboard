"""QB rushing projection backtest: starters (nflverse listed), official carries + rush yards.
E[carries] = QB carries/g (recency-weighted this season, shrunk to prior seasons / QB-type prior)
E[yards]   = E[carries] x QB YPC (shrunk)
x exp(spread, total, opponent QB-rush allowed)."""
import sys; sys.path.insert(0, "/home/claude/nfl-dashboard")
import numpy as np, pandas as pd
from scipy.optimize import minimize
import props_model as pm
raw = pm.load_raw([2020,2021,2022,2023,2024,2025])
ps = raw["ps"]; G = pm._games(raw["sch"])
q = ps[ps.position=="QB"][["player_id","game_id","season","week","team","carries","rushing_yards","attempts"]]
S = G[G.played][["game_id","season","week","team","opp","qb_id","team_spread","total_line"]].merge(
    q.rename(columns={"player_id":"qb_id"})[["qb_id","game_id","carries","rushing_yards","attempts"]], on=["qb_id","game_id"], how="left")
S = S[S.attempts.fillna(0)>0].fillna({"carries":0,"rushing_yards":0}).sort_values(["qb_id","season","week"]).reset_index(drop=True)
# QB history from ALL his games (any role) for priors
A = q.copy()
tot = A.groupby(["player_id","season"]).agg(g=("game_id","size"), car=("carries","sum"), yds=("rushing_yards","sum")).reset_index()
def feats(lam, kg, kypc, w1):
    f = S.copy(); gq = f.groupby(["qb_id","season"])
    f["c_g"] = gq.cumcount(); f["c_car"] = gq.carries.cumsum()-f.carries; f["c_yds"] = gq.rushing_yards.cumsum()-f.rushing_yards
    f["d_car"] = pm._decayed(f,"carries",lam,["qb_id","season"]); f["d_n"] = pm._decayed(f.assign(one=1.0),"one",lam,["qb_id","season"])
    for k in (1,2):
        t = tot.assign(season=tot.season+k).rename(columns={"player_id":"qb_id","g":f"g{k}","car":f"car{k}","yds":f"yds{k}"})
        f = f.merge(t, on=["qb_id","season"], how="left")
    for c in ["g1","car1","yds1","g2","car2","yds2"]: f[c]=f[c].fillna(0)
    lg_cpg = S.groupby("season").carries.mean().shift(1).reindex(f.season).fillna(S.carries.mean()).to_numpy()
    lg_ypc = (S.groupby("season").rushing_yards.sum()/S.groupby("season").carries.sum()).shift(1).reindex(f.season).fillna(4.5).to_numpy()
    prior_cpg = np.where(f.g1+f.g2>=6, (w1*f.car1+0.5*w1*f.car2)/(w1*f.g1+0.5*w1*f.g2).replace(0,np.nan), np.nan)
    prior_cpg = np.where(np.isnan(prior_cpg), lg_cpg, prior_cpg)
    pw = np.minimum(w1*f.g1+0.5*w1*f.g2, 8)  # prior worth up to 8 games
    f["est_cpg"] = (f.d_car + (kg+pw)*prior_cpg)/(f.d_n + kg + pw)
    n = f.c_car + w1*f.car1 + 0.5*w1*f.car2
    f["est_ypc"] = (f.c_yds + w1*f.yds1 + 0.5*w1*f.yds2 + kypc*lg_ypc)/(n + kypc)
    # opponent: QB carries & yards allowed per game, season to date, shrunk
    d = f[["game_id","season","week","opp","carries","rushing_yards"]].sort_values(["opp","season","week"])
    gd = d.groupby(["opp","season"]); d["dc"]=gd.carries.cumsum()-d.carries; d["dy"]=gd.rushing_yards.cumsum()-d.rushing_yards; d["dn"]=gd.cumcount()
    f = f.merge(d[["game_id","opp","dc","dy","dn"]], on=["game_id","opp"], how="left")
    f["opp_qbcar_ratio"] = ((f.dc + 6*lg_cpg)/(f.dn+6))/lg_cpg
    f["opp_qbypc_ratio"] = ((f.dy + 60*lg_ypc*1)/(f.dc+60))/lg_ypc
    f["tot_c"] = f.total_line-44.5
    f["B0_car"] = (f.c_car/f.c_g.replace(0,np.nan)).fillna(pd.Series(prior_cpg)); f["B0_yds"] = (f.c_yds/f.c_g.replace(0,np.nan)).fillna(pd.Series(prior_cpg*lg_ypc))
    return f
def evaluate(f, train, test, show=False):
    tr = f.season.isin(train).to_numpy(); te = (f.season==test).to_numpy(); out={}
    for st, y, base, cols in [("car","carries", f.est_cpg, ["team_spread","tot_c","opp_qbcar_ratio"]),
                              ("yds","rushing_yards", f.est_cpg*f.est_ypc, ["team_spread","tot_c","opp_qbcar_ratio","opp_qbypc_ratio"])]:
        B = base.to_numpy(); Y = f[y].to_numpy(float)
        X = np.column_stack([np.log(f[c]) if c.endswith("_ratio") else f[c] for c in cols])
        res = {"B0": f["B0_"+st].to_numpy(), "M0": B}
        for lab, k in [("+lines",2),("+lines+opp",len(cols))]:
            mu = lambda z: B*np.exp(X[:,:k]@z)
            z = minimize(lambda z: np.mean((np.maximum(mu(z),1e-6)-Y*np.log(np.maximum(mu(z),1e-6)))[tr]), np.zeros(k), method="L-BFGS-B").x
            res[lab] = mu(z); out[st+"_coef_"+lab]=np.round(z,3)
        for k,v in res.items(): out[f"{st} {k}"] = np.sqrt(((v-Y)**2)[te].mean())
    return out
if sys.argv[1:]==["tune"]:
    rows=[]
    for lam in [1.0,0.85,0.7]:
        for kg in [1,3,6]:
            for kypc in [20,50,100]:
                f=feats(lam,kg,kypc,1.0); a=evaluate(f,[2021],2022); b=evaluate(f,[2022],2023)
                rows.append(dict(lam=lam,kg=kg,kypc=kypc,car=(a["car M0"]+b["car M0"])/2, yds=(a["yds M0"]+b["yds M0"])/2))
    R=pd.DataFrame(rows); print(R.sort_values("car").head(4).round(3).to_string(index=False)); print(R.sort_values("yds").head(4).round(3).to_string(index=False))
else:
    lam,kg,kypc = [float(x) for x in sys.argv[1:4]]
    f=feats(lam,kg,kypc,1.0)
    for tr,te in [([2023],2024),([2023,2024],2025)]:
        o=evaluate(f,tr,te); print(te, {k:(round(v,3) if not isinstance(v,np.ndarray) else v) for k,v in o.items()})
    f.to_parquet("qb_rush_feats.parquet")
