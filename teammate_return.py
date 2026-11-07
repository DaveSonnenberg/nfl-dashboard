"""Backtest: teammate-return adjustment to target/carry shares.

Train coefs on earlier seasons, test on the next (2023->2024, 2023-24->2025), with the production
feature pipeline. Reports RMSE/MAE overall and on 'affected' rows: games where a teammate active now
was absent in at least one of the player's earlier games this season with D >= 0.15.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import props_model as pm

SEASONS = [2023, 2024, 2025]
MK = {"receptions": "receptions", "rec_yards": "receiving_yards", "carries": "carries", "rush_yards": "rushing_yards"}


def base(cache="/tmp/tr_base.pkl"):
    p = Path(cache)
    if p.exists():
        return pd.read_pickle(p)
    raw = pm.load_raw(SEASONS)
    raw = {**raw, "manual_out": ()}
    u, G = pm._universe(raw, False)
    T = pm._team_context(G)
    pd.to_pickle((u, T), p)
    return u, T


def features(u, T, **tm):
    old = dict(pm.TM)
    pm.TM.update(tm)
    try:
        return pm._player_features(u.copy(), T)
    finally:
        pm.TM.clear(); pm.TM.update(old)


def exposure(P):
    """Max D across the player's earlier games (same-position, current shares), per market."""
    out = {}
    for share, col in [("est_tshare", "rec"), ("est_cshare", "rush")]:
        P = P.reset_index(drop=True)
        act = {}
        for i, (g, t, pid, ps_, s) in enumerate(zip(P.game_id, P.team, P.player_id, P.pos.fillna("RB"), P[share])):
            act.setdefault((g, t), {})[pid] = (ps_, s)
        D = np.zeros(len(P))
        for idx in P.groupby(["player_id", "season"], sort=False).indices.values():
            idx = sorted(idx, key=lambda i: P.at[i, "week"])
            for a, i in enumerate(idx):
                cur = act[(P.at[i, "game_id"], P.at[i, "team"])]
                best = 0.0
                for h in idx[:a]:
                    if P.at[h, "team"] != P.at[i, "team"]:
                        continue
                    prev = act.get((P.at[h, "game_id"], P.at[h, "team"]), {})
                    d = sum(sj for j, (pj, sj) in cur.items()
                            if j != P.at[i, "player_id"] and j not in prev and pj == P.at[i, "pos"])
                    best = max(best, d)
                D[i] = best
        out[col] = D
    return out


MASK = {}


def evaluate(P, expo):
    """Evaluation rows fixed by the BASELINE run (first call), so variants are compared on the same games."""
    P = P.reset_index(drop=True)
    rows = []
    for train, test in [([2023], 2024), ([2023, 2024], 2025)]:
        tr = P[P["season"].isin(train) & (P["c_g"] >= 1)]
        te_mask = (P["season"] == test) & (P["c_g"] >= 1)
        for mk, y in MK.items():
            c = pm.fit_coefs(tr, mk, y)
            mu = pm.apply(P[te_mask], mk, c)
            yy = P.loc[te_mask, y].to_numpy(float)
            key = (test, mk)
            if key not in MASK:
                MASK[key] = ((P.loc[te_mask, "E_tgt"] >= 3) if mk.startswith("rec")
                             else (P.loc[te_mask, "E_car0"] >= 5)).to_numpy()
            rel = pd.Series(MASK[key])
            D = (expo["rec"] if mk.startswith("rec") else expo["rush"])[te_mask.to_numpy()]
            for name, m in [("all", rel.to_numpy()), ("affected", rel.to_numpy() & (D >= 0.15)),
                            ("big return", rel.to_numpy() & (D >= 0.35))]:
                e = mu[m] - yy[m]
                rows.append({"test": test, "market": mk, "set": name, "n": int(m.sum()),
                             "rmse": np.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e)), "bias": e.mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    u, T = base()
    P0 = features(u, T)
    expo = exposure(P0)
    print(f"base built {time.time() - t0:.0f}s")
    R0 = evaluate(P0, expo).assign(variant="baseline")
    out = [R0]
    grid = [dict(snap_rec=x, snap_rush=x, snap_win=w) for w in [1, 2] for x in [0.25, 0.5, 1.0]]
    for tm in grid:
        P = features(u, T, **tm)
        out.append(evaluate(P, expo).assign(variant=str(tm)))
        print(tm, f"{time.time() - t0:.0f}s"); sys.stdout.flush()
    R = pd.concat(out)
    R.to_csv("/tmp/tr_results.csv", index=False)
    piv = R.pivot_table(index=["variant"], columns=["market", "set"], values="rmse", aggfunc="mean")
    print(R.pivot_table(index=["variant"], columns=["market", "set"], values="bias", aggfunc="mean").round(2).to_string())
    pd.set_option("display.width", 250)
    print(piv.round(3).to_string())
