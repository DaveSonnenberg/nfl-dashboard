"""Grade the model against real pre-game prop lines (history/props_lines.csv.gz).

Out of sample only: 2025 projections use coefs fit on 2023-24; 2026 uses coefs fit on 2023-25.
For each player-game-market: DraftKings main line (and a consensus line), model P(over) at that
line, the book's no-vig P(over), the pick (side with the larger model-minus-book gap), and the result.

    python model/grade_lines.py history/props_lines.csv.gz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import odds as od
import props_model as pm

SEASONS = [2023, 2024, 2025, 2026]
TEST = {2025: [2023, 2024], 2026: [2023, 2024, 2025]}
SPEC_KEY = {"rec_yards": "rec_yards", "rush_yards": "rush_yards", "pass_yards": "pass_yards"}


def projections(cache="/tmp/grade_proj.pkl"):
    p = Path(cache)
    if p.exists():
        return pd.read_pickle(p)
    raw = pm.load_raw(SEASONS)
    B = pm.build(SEASONS, include_upcoming=False, raw=raw)
    P, Q = B["players"], B["qbs"]
    P = P[~P["upcoming"]].copy()
    Q = Q[Q["played"]].copy()
    import nflreadpy as nfl
    inj = nfl.load_injuries(list(TEST)).to_pandas()
    st = inj.drop_duplicates(["season", "week", "gsis_id"], keep="last").set_index(["season", "week", "gsis_id"])["report_status"]
    P["inj_status"] = [st.get((s, w, i)) for s, w, i in zip(P["season"], P["week"], P["player_id"])]
    out = []
    for test, train in TEST.items():
        tr = P[P["season"].isin(train) & (P["c_g"] >= 1)]
        te = P[P["season"] == test].copy()
        q = te["inj_status"].eq("Questionable").to_numpy()
        for mk, y in [("rec_yards", "receiving_yards"), ("rush_yards", "rushing_yards")]:
            c = pm.fit_coefs(tr, mk, y)
            mult = np.where(q, pm.QUESTIONABLE_MULT[mk], 1.0)
            out.append(pd.DataFrame({"season": test, "game_id": te["game_id"], "player_id": te["player_id"],
                                     "name": te["name"], "team": te["team"], "pos": te["pos"], "market": mk,
                                     "mu": pm.apply(te, mk, c) * mult, "actual": te[y], "inj": te["inj_status"],
                                     "dist": mk}))
        qtr = Q[Q["season"].isin(train)]
        qte = Q[Q["season"] == test]
        c = pm.fit_coefs(qtr[qtr["c_g"] >= 1], "pass_yards", "passing_yards")
        out.append(pd.DataFrame({"season": test, "game_id": qte["game_id"], "player_id": qte["qb_id"], "name": qte["name"],
                                 "team": qte["team"], "pos": "QB", "market": "pass_yards",
                                 "mu": pm.apply(qte, "pass_yards", c), "actual": qte["passing_yards"], "inj": None,
                                 "dist": "pass_yards"}))
        c = pm.fit_coefs(qtr[qtr["c_qg"] >= 1], "qb_rush_yards", "rushing_yards")
        out.append(pd.DataFrame({"season": test, "game_id": qte["game_id"], "player_id": qte["qb_id"], "name": qte["name"],
                                 "team": qte["team"], "pos": "QB", "market": "rush_yards",
                                 "mu": pm.apply(qte, "qb_rush_yards", c), "actual": qte["rushing_yards"], "inj": None,
                                 "dist": "qb_rush_yards"}))
    R = pd.concat(out, ignore_index=True)
    R.to_pickle(p)
    return R


def main_line(g: pd.DataFrame) -> pd.Series:
    """One book's line for a player: the one priced closest to even."""
    g = g.dropna(subset=["over", "under"])
    if g.empty:
        return pd.Series(dtype=float)
    po, _ = od.implied_no_vig(g["over"], g["under"])
    r = g.iloc[int(np.argmin(np.abs(np.asarray(po) - 0.5)))]
    return pd.Series({"line": r["line"], "over": r["over"], "under": r["under"]})


def lines_table(L: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    who = R.drop_duplicates(["player_id", "team"])[["player_id", "name", "team"]]
    L = od.match_players(L, who, pm.norm_name)
    L = L.dropna(subset=["player_id"])
    dk = (L[L["book"] == "DraftKings"].groupby(["game_id", "market", "player_id"]).apply(main_line, include_groups=False)
          .reset_index())
    # consensus: most common line across books, median no-vig P(over) there
    L = L.dropna(subset=["over", "under"]).copy()
    L["p_over_book"], _ = od.implied_no_vig(L["over"], L["under"])
    mode = L.groupby(["game_id", "market", "player_id"])["line"].agg(lambda s: s.mode().iloc[0]).rename("c_line")
    L = L.join(mode, on=["game_id", "market", "player_id"])
    cons = (L[L["line"] == L["c_line"]].groupby(["game_id", "market", "player_id"])
            .agg(c_line=("line", "first"), c_p_over=("p_over_book", "median"), n_books=("book", "nunique")).reset_index())
    return dk.merge(cons, on=["game_id", "market", "player_id"], how="outer")


def grade(T: pd.DataFrame, dist: dict, edge_min=0.05) -> pd.DataFrame:
    T = T.copy()
    T["model_p"] = [float(pm.p_over(dist[d], m, l)) if pd.notna(l) else np.nan
                    for d, m, l in zip(T["dist"], T["mu"], T["line"])]
    T["book_p"], _ = od.implied_no_vig(T["over"], T["under"])
    T["edge_over"] = T["model_p"] - T["book_p"]
    T["pick"] = np.where(T["edge_over"] >= 0, "over", "under")
    T["edge"] = T["edge_over"].abs()
    T["result"] = np.where(T["actual"] > T["line"], "over", np.where(T["actual"] < T["line"], "under", "push"))
    price = np.where(T["pick"] == "over", T["over"], T["under"])
    dec = od.american_to_decimal(price)
    T["profit"] = np.where(T["result"] == "push", 0.0, np.where(T["result"] == T["pick"], dec - 1, -1.0))
    T["win"] = np.where(T["result"] == "push", np.nan, (T["result"] == T["pick"]).astype(float))
    return T


def report(T: pd.DataFrame):
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    d = T.dropna(subset=["line", "model_p", "actual"])
    print(f"\nDraftKings lines graded: {len(d):,}  (seasons {sorted(d['season'].unique())})")
    # calibration: who predicts over/under better?
    y = (d["actual"] > d["line"]).astype(float)
    m = d["result"] != "push"
    print("\nOver rate vs predictions (non-push):")
    print(pd.DataFrame({"actual over rate": y[m].groupby(d.loc[m, "market"]).mean(),
                        "book avg P(over)": d.loc[m, "book_p"].groupby(d.loc[m, "market"]).mean(),
                        "model avg P(over)": d.loc[m, "model_p"].groupby(d.loc[m, "market"]).mean(),
                        "book Brier": ((d.loc[m, "book_p"] - y[m]) ** 2).groupby(d.loc[m, "market"]).mean(),
                        "model Brier": ((d.loc[m, "model_p"] - y[m]) ** 2).groupby(d.loc[m, "market"]).mean()}).round(4))
    d = d.assign(bucket=pd.cut(d["edge"], [0, 0.03, 0.05, 0.08, 0.12, 1], labels=["0-3", "3-5", "5-8", "8-12", "12+"]))
    g = d.groupby(["market", "bucket"], observed=True).agg(bets=("profit", "size"), win=("win", "mean"),
                                                           roi=("profit", "mean"))
    print("\nBy edge bucket (bet every pick at DK price):"); print(g.round(3).to_string())
    b = d[d["edge"] >= 0.05]
    for col in ["market", "pick", "season"]:
        print(f"\nEdge >= 5 pts by {col}:")
        print(b.groupby(col).agg(bets=("profit", "size"), win=("win", "mean"), roi=("profit", "mean")).round(3).to_string())
    qq = d[d["inj"].eq("Questionable")]
    if len(qq):
        print("\nQuestionable players: line vs model vs actual")
        print(qq.groupby("market").agg(n=("line", "size"), line=("line", "mean"), model=("mu", "mean"),
                                       actual=("actual", "mean"), over_rate=("result", lambda s: (s == "over").mean())).round(2))


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "history" / "props_lines.csv.gz"
    L = pd.read_csv(path)
    L["market"] = L["market"].astype(str)
    import json
    dist = json.loads((ROOT / "model" / "dist_specs.json").read_text())
    R = projections()
    T = lines_table(L, R)
    T = T.merge(R, on=["game_id", "market", "player_id"], how="inner")
    G = grade(T, dist)
    G.to_csv("/tmp/graded.csv", index=False)
    unmatched = 1 - len(T) / max(L.groupby(["game_id", "market", "player_raw"]).ngroups, 1)
    print(f"player-markets with a line: {L.groupby(['game_id', 'market', 'player_raw']).ngroups:,}; "
          f"matched to projections: {len(T):,} ({unmatched:.0%} unmatched)")
    report(G)
