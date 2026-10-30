"""Does a missing player's volume go mostly to teammates at his position?
Backtest: tune on 2023-24, test 2025. Targets (WR/TE/RB) and carries (RB)."""
import numpy as np, pandas as pd
f = pd.read_parquet("features.parquet")
f["pos"] = f["pos"].fillna("RB")
# team season-to-date share of targets by position (earlier games), shrunk to league position share
tp = f.groupby(["season","week","game_id","team","pos"]).targets.sum().unstack("pos").fillna(0)
tt = f.groupby(["season","week","game_id","team"]).team_targets.first()
tp = tp.join(tt).reset_index().sort_values(["team","season","week"])
lg = {p: tp[p].sum()/tp.team_targets.sum() for p in ["WR","TE","RB"]}
g = tp.groupby(["team","season"])
for p in ["WR","TE","RB"]:
    cum = g[p].cumsum()-tp[p]; ctt = g["team_targets"].cumsum()-tp["team_targets"]
    tp["S_"+p] = (cum + 60*lg[p])/(ctt + 60)
S = tp.melt(id_vars=["game_id","team"], value_vars=["S_WR","S_TE","S_RB"], var_name="pos", value_name="S_pos")
S["pos"] = S.pos.str[2:]
f = f.merge(S, on=["game_id","team","pos"], how="left")
f["act_pos"] = f.groupby(["game_id","team","pos"]).est_share.transform("sum")
f["act_tot"] = f.groupby(["game_id","team"]).est_share.transform("sum")
f["miss_pos"] = (f.S_pos - f.act_pos).clip(lower=0)
f["miss_tot"] = (0.995 - f.act_tot).clip(lower=0)
f["over_tot"] = (f.act_tot - 0.995).clip(lower=0)   # actives sum above 1 -> scale down as before
def E(alpha, gamma):
    pos_part = gamma * f.miss_pos * f.est_share/f.act_pos.replace(0,np.nan)
    # whatever isn't claimed by position-mates is spread across everyone
    rest = (f.miss_tot - gamma*f.groupby(["game_id","team"]).miss_pos.transform(lambda s: s.drop_duplicates().sum() if False else s.groupby(f.loc[s.index,"pos"]).first().sum())).clip(lower=0)
    all_part = rest * f.est_share/f.act_tot
    share = f.est_share + alpha*(pos_part.fillna(0) + all_part) - f.est_share*(f.over_tot/f.act_tot)*alpha
    return share * f.est_team_tpg
y = f.targets
big = f.groupby(["game_id","team"]).miss_tot.transform("max") >= 0.12
rel = f.renorm_share*f.est_team_tpg >= 3.5
rows=[]
for a in [0.5, 0.75, 1.0]:
    for gm in [0.0, 0.5, 1.0]:
        e = E(a, gm) - y
        r = {"alpha":a,"gamma":gm}
        for nm, m in [("train",f.season.between(2023,2024)),("test",f.season==2025)]:
            r[nm+"_rel"] = np.sqrt((e[m&rel]**2).mean()); r[nm+"_teammate_out"] = np.sqrt((e[m&rel&big]**2).mean())
            r[nm+"_bias_out"] = e[m&rel&big].mean()
        rows.append(r)
pd.set_option("display.width",200)
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
print("teammate-out games (relevant):", int((f.season.between(2023,2025)&rel&big).sum()))
