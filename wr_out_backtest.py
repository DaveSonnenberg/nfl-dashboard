"""Backtest receiver redistribution v2 against v1 (production).

v2 (props_model._redistribute_v2): missing WR targets split WR/TE/RB, WR part by share + snaps,
snap floor for WRs at 40%+ snaps, receiving efficiency cut when key WRs are out.
Accuracy on 2024-26 receivers (projected 1+ targets): all, teammates in games with a key WR out,
two out, and snap-surge players. Then DraftKings receiving-yards lines (2025-26).
    python model/wr_out_backtest.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import odds as od
import props_model as pm

SEASONS = [2023, 2024, 2025, 2026]
C = json.loads((ROOT / "model" / "coefs.json").read_text())
DIST = json.loads((ROOT / "model" / "dist_specs.json").read_text())


def base(cache="/tmp/wr_base.pkl"):
    p = Path(cache)
    if p.exists():
        return pd.read_pickle(p)
    raw = {**pm.load_raw(SEASONS), "manual_out": ()}
    u, G = pm._universe(raw, False)
    out = (u, pm._team_context(G))
    pd.to_pickle(out, p)
    return out


def run(u, T, **tm):
    old = dict(pm.TM)
    pm.TM.update(tm)
    try:
        P = pm._player_features(u.copy(), T)
    finally:
        pm.TM.clear(); pm.TM.update(old)
    P = P[P["season"] >= 2024].copy()
    P["mu_rec"] = pm.apply(P, "receptions", C["receptions"])
    P["mu_yds"] = pm.apply(P, "rec_yards", C["rec_yards"])
    return P


def score(P, sets):
    out = {}
    for nm, m in sets.items():
        d = P[m.reindex(P.index).fillna(False).to_numpy()]
        out[(nm, "n")] = len(d)
        out[(nm, "tgt MAE")] = (d["targets"] - d["E_tgt"]).abs().mean()
        out[(nm, "tgt bias")] = (d["targets"] - d["E_tgt"]).mean()
        out[(nm, "yds MAE")] = (d["receiving_yards"] - d["mu_yds"]).abs().mean()
        out[(nm, "yds bias")] = (d["receiving_yards"] - d["mu_yds"]).mean()
        out[(nm, "rec MAE")] = (d["receptions"] - d["mu_rec"]).abs().mean()
    return out


if __name__ == "__main__":
    u, T = base()
    variants = {
        "v1 (production)": dict(rec_v2=False),
        "v2 full (floor+cut)": dict(rec_v2=True),
        "v2 lean (no floor, no cut)": dict(rec_v2=True, snap_floor=0.0, eff_cut=(0, 0), cr_cut=(0, 0)),
        "v2 lean, snap wt 0.4": dict(rec_v2=True, snap_floor=0.0, eff_cut=(0, 0), cr_cut=(0, 0), snap_beta=0.4),
        "v2 lean, snap wt 0.4, TE 0.15": dict(rec_v2=True, snap_floor=0.0, eff_cut=(0, 0), cr_cut=(0, 0), snap_beta=0.4,
                                              wr_split=(0.82, 0.15, 0.03)),
    }
    runs = {k: run(u, T, **v) for k, v in variants.items()}
    ref = runs["v2 full (floor+cut)"]
    base_rows = ref["E_tgt"] >= 1
    wm = ref.groupby(["game_id", "team"])["wr_missing"].transform("max")
    g = ref.groupby(["player_id", "season"])["offense_pct"]
    early = g.transform(lambda x: x.shift(3).expanding().mean())
    sets = {
        "all": base_rows,
        "1 key WR out": base_rows & wm.between(0.12, 0.30, inclusive="left"),
        "2+ key WRs out": base_rows & (wm >= 0.30),
        "snap surge WR/TE": (ref["snap_l2"] >= 0.40) & (early <= 0.20) & ref["pos"].isin(["WR", "TE"]),
    }
    res = pd.DataFrame({k: score(P.loc[ref.index] if len(P) == len(ref) else P, sets) for k, P in runs.items()}).T
    pd.set_option("display.width", 250)
    for nm in sets:
        print(f"\n=== {nm} (n={int(res.iloc[0][(nm, 'n')])})")
        print(res[[c for c in res.columns if c[0] == nm and c[1] != "n"]].droplevel(0, axis=1).round(3).to_string())
    for s in [2024, 2025, 2026]:
        print(f"\n{s} 2+ out / surge tgt MAE:", {k: (round((P[(P.season == s) & sets['2+ key WRs out'].reindex(P.index).fillna(False)]['targets'] - P[(P.season == s) & sets['2+ key WRs out'].reindex(P.index).fillna(False)]['E_tgt']).abs().mean(), 3)) for k, P in runs.items()})

    # ---- DraftKings receiving-yards lines
    gpath = Path("/tmp/graded.csv")
    if gpath.exists():
        Gd = pd.read_csv(gpath)
        Gd = Gd[(Gd["market"] == "rec_yards") & (Gd["result"] != "push")].dropna(subset=["line", "over", "under"])
        rows = []
        wmap = ref.groupby(["game_id", "team"])["wr_missing"].max().rename("wm").reset_index()
        for k in ["v1 (production)", "v2 lean, snap wt 0.4", "v2 lean, snap wt 0.4, TE 0.15"]:
            P = runs[k][["game_id", "team", "player_id", "mu_yds", "snap_l2"]].merge(wmap, on=["game_id", "team"])
            P = P.rename(columns={"wm": "wr_missing"}).drop(columns="team")
            d = Gd.merge(P, on=["game_id", "player_id"], how="inner")
            d["p"] = [float(pm.p_over(DIST["rec_yards"], m, l)) for m, l in zip(d["mu_yds"], d["line"])]
            e = d["p"] - d["book_p"]
            pick = np.where(e >= 0, "over", "under")
            dec = od.american_to_decimal(np.where(pick == "over", d["over"], d["under"]))
            d["profit"] = np.where(d["result"] == pick, dec - 1, -1.0)
            d["edge_u"] = -e
            for nm, m in [("all", np.ones(len(d), bool)),
                          ("key WR out", d["wr_missing"] >= 0.12),
                          ("snap 40%+", d["snap_l2"] >= 0.40)]:
                for thr in [0.0, 0.05, 0.20]:
                    x = d[m & (np.abs(e) >= thr)]
                    xu = d[m & (d["edge_u"] >= 0.20)]
                    rows.append(dict(variant=k, set=nm, edge=f">={thr:.0%}", n=len(x), roi=x["profit"].mean(),
                                     u20_n=len(xu), u20_roi=xu["profit"].mean()))
        print("\nDK receiving-yards lines (ROI betting the model's side):")
        print(pd.DataFrame(rows).pivot_table(index=["set", "edge"], columns="variant", values=["n", "roi"]).round(3).to_string())
