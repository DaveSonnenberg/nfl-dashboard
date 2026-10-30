"""Out-of-sample projections: fold A (train 2023 -> test 2024), fold B (train 2023-24 -> test 2025).
For each fold, also keep the TRAIN-season projections from that fold's fit (needed to fit distributions)."""
import pandas as pd, numpy as np, io, contextlib
import final as fm
import rush_model as rm
out = []
for train, test, fold in [([2023], 2024, "A"), ([2023, 2024], 2025, "B")]:
    with contextlib.redirect_stdout(io.StringIO()):
        f = fm.run(train, test)
        rb = rm.build(0.7, 10, 40, 0.5); _, r = rm.evaluate(rb, train, test, verbose=False)
    a = f[["season","week","game_id","player_id","name","pos","team","opp","E_tgt","est_adot","receptions","receiving_yards","targets"]].copy()
    a["mu_rec"] = f["FINAL lines+oppvol|rec"]; a["mu_ryd"] = f["FINAL lines+oppvol|yds"]
    b = r[["season","week","game_id","player_id","E_car0","carries","rushing_yards"]].copy()
    b["mu_car"] = r["+ lines + opp carries allowed|car"]; b["mu_rush"] = r["+ all + opp YPC allowed|yds"]
    m = a.merge(b, on=["season","week","game_id","player_id"], how="outer")
    m["name"] = m["name"].fillna(m.groupby("player_id")["name"].transform("first"))
    m["fold"] = fold; m["split"] = np.where(m.season == test, "test", "train")
    out.append(m)
P = pd.concat(out, ignore_index=True)
P.to_parquet("oos_preds.parquet")
print(P.groupby(["fold","split"]).size()); print(P[["mu_rec","mu_ryd","mu_car","mu_rush"]].describe().round(2))
