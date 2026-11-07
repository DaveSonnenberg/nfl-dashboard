"""How much do teammates' projections suffer when a player returning from 3+ missed games is invisible
(production candidate rule = snaps in the team's last 3 games)? Variant A drops returners from the
active set on their return game (current production); variant B keeps them (auto-include fix)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import teammate_return as tr
import props_model as pm


def returner_rows(u):
    u = u.sort_values(["team", "season", "week"])
    keys = []
    for (t, s), d in u.groupby(["team", "season"]):
        wks = sorted(d["week"].unique()); pos = {w: i for i, w in enumerate(wks)}; seen = {}
        for w in wks:
            g = d[d["week"] == w]
            for pid, gid in zip(g["player_id"], g["game_id"]):
                if pid in seen and pos[w] - seen[pid] >= 4:
                    keys.append((gid, pid))
            for pid in g["player_id"]:
                seen[pid] = pos[w]
    return set(keys)


if __name__ == "__main__":
    pm.TM.update(dict(rush_over=1.0, rec_over=0.75))
    u, T = tr.base()
    ret = returner_rows(u)
    is_ret = np.array([(g, p) in ret for g, p in zip(u["game_id"], u["player_id"])])
    PB = pm._player_features(u.copy(), T)
    rr = PB[[(g, p) in ret for g, p in zip(PB.game_id, PB.player_id)]]
    big = rr[(rr.est_tshare >= 0.12) | ((rr.pos == "RB") & (rr.est_cshare >= 0.25))]
    games_rec = set(big.loc[big.est_tshare >= 0.12, "game_id"] + "|" + big.loc[big.est_tshare >= 0.12, "team"])
    games_rush = set(big.loc[(big.pos == "RB") & (big.est_cshare >= 0.25), "game_id"]
                     + "|" + big.loc[(big.pos == "RB") & (big.est_cshare >= 0.25), "team"])
    games = {g for g, _ in ret}
    PA = pm._player_features(u[~is_ret].copy(), T)
    out = []
    for train, test in [([2023], 2024), ([2023, 2024], 2025)]:
        for mk, y in tr.MK.items():
            c = pm.fit_coefs(PB[PB.season.isin(train) & (PB.c_g >= 1)], mk, y)
            for name, P in [("A: returner invisible", PA), ("B: returner included", PB)]:
                gk = games_rec if mk.startswith("rec") else games_rush
                m = (P.season == test) & (P.c_g >= 1) & (P.game_id + "|" + P.team).isin(gk)
                m &= ~np.array([(g, p) in ret for g, p in zip(P.game_id, P.player_id)])  # teammates only
                m &= (P.E_tgt >= 2) if mk.startswith("rec") else (P.pos == "RB")
                mu = pm.apply(P[m], mk, c); yy = P.loc[m, y].to_numpy(float)
                out.append(dict(test=test, market=mk, variant=name, n=int(m.sum()),
                                rmse=np.sqrt(np.mean((mu - yy) ** 2)), bias=np.mean(mu - yy)))
    R = pd.DataFrame(out)
    print(f"return games: {len(games)}")
    print(R.pivot_table(index="market", columns="variant", values=["rmse", "bias"]).round(2).to_string())
