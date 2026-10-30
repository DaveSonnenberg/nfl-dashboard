import pandas as pd, numpy as np, statsmodels.formula.api as smf
t = pd.read_parquet("rtg.parquet").sort_values(["season","week"]).reset_index(drop=True)
# defense allowed per game (from the offense rows)
d = t[["game_id","season","week","defteam","rush_epa_play","pass_epa_play","rush_plays","pass_plays","team_carries","pass_att","neutral_proe","n_neutral"]].rename(columns={"defteam":"dteam"})
d["rush_epa_sum"]=d.rush_epa_play*d.rush_plays; d["pass_epa_sum"]=d.pass_epa_play*d.pass_plays
d["proe_sum"]=d.neutral_proe*d.n_neutral

def pregame(df, key, cols, k):
    """season-to-date (shifted) sums for team `key`, + prior season at 1/3 weight; returns rate = sum/count with shrink k to league."""
    df = df.sort_values([key,"season","week"]).copy(); g = df.groupby([key,"season"])
    out = {}
    for num, den in cols:
        cn = g[num].cumsum()-df[num]; cd = g[den].cumsum()-df[den]
        prev = df.groupby([key,"season"]).agg(pn=(num,"sum"), pd_=(den,"sum")).reset_index(); prev["season"]+=1
        tmp = df[[key,"season"]].merge(prev, on=[key,"season"], how="left").fillna(0)
        lg = df.groupby("season")[num].sum()/df.groupby("season")[den].sum()
        lgv = df["season"].map(lg)
        out[num] = ((cn.values + tmp.pn.values/3 + k*lgv.values)/(cd.values + tmp.pd_.values/3 + k)) - lgv.values
        # leave-one-out full season (hindsight, excludes this game)
        tn = g[num].transform("sum"); td = g[den].transform("sum")
        out[num+"_loo"] = ((tn-df[num])/(td-df[den])).values - lgv.values
    res = pd.DataFrame(out, index=df.index); res["game_id"]=df.game_id.values; res[key]=df[key].values
    return res

D = pregame(d, "dteam", [("rush_epa_sum","rush_plays"),("pass_epa_sum","pass_plays")], k=150)
D.columns = ["d_rush","d_rush_loo","d_pass","d_pass_loo","game_id","defteam"]
O = pregame(t.assign(proe_sum=t.neutral_proe*t.n_neutral), "team", [("proe_sum","n_neutral")], k=60)
O.columns = ["o_proe","o_proe_loo","game_id","team"]
x = t.merge(D, on=["game_id","defteam"]).merge(O, on=["game_id","team"])
x = x[x.season>=2022].copy()
for c in ["d_rush","d_pass","d_rush_loo","d_pass_loo"]:
    x[c+"_z"] = x.groupby("season")[c].transform(lambda s: (s-s.mean())/s.std())
x["d_rel_z"] = x.d_rush_z - x.d_pass_z; x["d_rel_loo_z"] = x.d_rush_loo_z - x.d_pass_loo_z
x["tot_c"] = x.total_line - 44.5
print("team-games:", len(x))
pd.set_option("display.width",220)
for outcome in ["neutral_proe","pass_att","team_carries","pass_rate"]:
    print(f"\n=== {outcome} (mean {x[outcome].mean():.2f}, sd {x[outcome].std():.2f})")
    for lab, terms in [("pre-game, separate", "d_rush_z + d_pass_z"), ("pre-game, relative", "d_rel_z"),
                       ("hindsight LOO, separate", "d_rush_loo_z + d_pass_loo_z")]:
        f = f"{outcome} ~ {terms} + o_proe + team_spread + tot_c"
        m = smf.ols(f, data=x, missing="drop").fit(cov_type="cluster", cov_kwds={"groups": x.dropna(subset=[outcome])["game_id"].astype("category").cat.codes})
        keep = [c for c in m.params.index if c.startswith("d_")]
        print(f"  {lab:24s} " + " | ".join(f"{c}: {m.params[c]:+.3f} (se {m.bse[c]:.3f}, p {m.pvalues[c]:.3f})" for c in keep) + f" | spread {m.params['team_spread']:+.3f} | R2 {m.rsquared:.3f}")
# intuitive bucket table: opponent run-D quintile (hindsight LOO) vs offense deviation from own norm
x["proe_dev"] = x.neutral_proe - x.o_proe_loo
x["carries_dev"] = x.team_carries - x.groupby(["team","season"]).team_carries.transform(lambda s: (s.sum()-s)/(len(s)-1))
x["att_dev"] = x.pass_att - x.groupby(["team","season"]).pass_att.transform(lambda s: (s.sum()-s)/(len(s)-1))
x["q"] = pd.qcut(x.d_rush_loo_z, 5, labels=["Strongest run D","2","3","4","Weakest run D"])
print("\nBy opponent run defense (full-season, excluding this game), offense vs its own season norm:")
print(x.groupby("q", observed=True).agg(games=("game_id","size"), neutral_PROE_vs_norm=("proe_dev","mean"), carries_vs_norm=("carries_dev","mean"), pass_att_vs_norm=("att_dev","mean"), avg_spread=("team_spread","mean")).round(2).to_string())
x.to_parquet("runpass_x.parquet")
