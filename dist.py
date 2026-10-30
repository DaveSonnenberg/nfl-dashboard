"""Turn projections into P(over line). Compare distribution methods out of sample.

Methods
  naive : Poisson (counts) / Normal with SD fitted by projection level (yards)
  param : Negative binomial (counts) / zero-inflated gamma (yards)
  emp   : empirical conditional — historical games with a similar projection;
          counts use raw outcomes, yards scale outcomes by (mu / mu_i)
Evaluated on test seasons (2024 via fold A, 2025 via fold B), prop-relevant rows,
at several x.5 lines around each projection.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

D = Path(__file__).parent
P = pd.read_parquet(D / "oos_preds.parquet")
STATS = {  # stat: (mu col, outcome col, relevance col, relevance cut, is_count)
    "receptions": ("mu_rec", "receptions", "E_tgt", 3.5, True),
    "rec_yards": ("mu_ryd", "receiving_yards", "E_tgt", 3.5, False),
    "carries": ("mu_car", "carries", "E_car0", 8.0, True),
    "rush_yards": ("mu_rush", "rushing_yards", "E_car0", 8.0, False),
}


def lines_for(mu, is_count):
    if is_count:
        k = np.round(mu)
        return np.stack([k - 1.5, k - 0.5, k + 0.5, k + 1.5], 1)
    return np.stack([np.round(mu * m) + 0.5 for m in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)], 1)


# ---------------------------------------------------------------- fitters
def fit_naive(mu, y, is_count):
    if is_count:
        return lambda m, L: stats.poisson.sf(np.floor(L), m)
    # SD = a * mu^b
    def nll(z):
        sd = np.exp(z[0]) * mu ** z[1]
        return -np.sum(stats.norm.logpdf(y, mu, sd))
    z = minimize(nll, [1.0, 0.5], method="Nelder-Mead").x
    return lambda m, L: stats.norm.sf(L, m, np.exp(z[0]) * m ** z[1])


def fit_param(mu, y, is_count):
    if is_count:
        # var = mu + mu^2/theta ; theta = exp(z)
        # theta varies with the projection: log theta = z0 + z1 log mu
        def nll(z):
            th = np.exp(z[0] + z[1] * np.log(mu)); p = th / (th + mu)
            return -np.sum(stats.nbinom.logpmf(y, th, p))
        z = minimize(nll, [2.0, 0.0], method="Nelder-Mead").x

        def sf(m, L):
            th = np.exp(z[0] + z[1] * np.log(m))
            return stats.nbinom.sf(np.floor(L), th, th / (th + m))
        return sf
    # zero-inflated gamma: P(y<=0) = logistic(a + b log mu); y>0 ~ Gamma(shape=k(mu), mean=c*mu/(1-pi))
    pos = y > 0
    def nll(z):
        a, b, k0, k1, c = z
        pi = 1 / (1 + np.exp(-(a + b * np.log(mu))))
        k = np.exp(k0 + k1 * np.log(mu)); mean = np.exp(c) * mu / (1 - pi)
        ll = np.where(pos, np.log(1 - pi) + stats.gamma.logpdf(np.maximum(y, 1e-9), k, scale=mean / k), np.log(pi))
        return -np.sum(ll)
    z = minimize(nll, [0, -1, 0.5, 0, 0], method="Nelder-Mead", options={"maxiter": 6000, "maxfev": 6000}).x
    a, b, k0, k1, c = z

    def sf(m, L):
        pi = 1 / (1 + np.exp(-(a + b * np.log(m))))
        k = np.exp(k0 + k1 * np.log(m)); mean = np.exp(c) * m / (1 - pi)
        return np.where(L < 0, 1 - 0.0, (1 - pi) * stats.gamma.sf(np.maximum(L, 0), k, scale=mean / k))
    return sf


def fit_emp(mu, y, is_count, bw=0.12):
    lm, Y, MU = np.log(mu), y.astype(float), mu

    def sf(m, L):
        out = np.empty(L.shape)
        for i in range(len(m)):
            w = np.exp(-0.5 * ((lm - np.log(m[i])) / bw) ** 2)
            vals = Y if is_count else Y * (m[i] / MU)
            for j in range(L.shape[1]):
                out[i, j] = np.sum(w * (vals > L[i, j])) / w.sum()
        return out
    return sf


# ---------------------------------------------------------------- evaluation
def evaluate():
    res, calib, curves = [], [], {}
    for st, (mc, yc, rc, cut, isc) in STATS.items():
        for fold in ["A", "B"]:
            d = P[(P["fold"] == fold) & P[mc].notna() & (P[mc] > 0.05)]
            tr = d[(d["split"] == "train") & (d[rc] >= cut * 0.6)]
            te = d[(d["split"] == "test") & (d[rc] >= cut)]
            mu_te, y_te = te[mc].to_numpy(), te[yc].to_numpy()
            L = lines_for(mu_te, isc)
            hit = (y_te[:, None] > L).astype(float)
            for name, fitter in [("naive", fit_naive), ("param", fit_param), ("emp", fit_emp)]:
                sf = fitter(tr[mc].to_numpy(), tr[yc].to_numpy(), isc)
                if name == "emp":
                    p = sf(mu_te, L)
                else:
                    p = sf(mu_te[:, None] * np.ones_like(L), L)
                p = np.clip(p, 1e-4, 1 - 1e-4)
                brier = np.mean((p - hit) ** 2)
                ll = -np.mean(hit * np.log(p) + (1 - hit) * np.log(1 - p))
                mid = (p > 0.35) & (p < 0.65)
                res.append({"stat": st, "fold": fold, "method": name, "n_games": len(te), "brier": brier,
                            "logloss": ll, "brier_35_65": np.mean((p[mid] - hit[mid]) ** 2),
                            "mean_p": p.mean(), "hit_rate": hit.mean()})
                for lo in np.arange(0, 1, 0.1):
                    m = (p >= lo) & (p < lo + 0.1)
                    if m.sum():
                        calib.append({"stat": st, "fold": fold, "method": name, "bin": f"{lo:.1f}-{lo + 0.1:.1f}",
                                      "pred": p[m].mean(), "actual": hit[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(res), pd.DataFrame(calib)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    R, C = evaluate()
    print(R.round(4).to_string(index=False))
    R.to_csv(D / "dist_scores.csv", index=False); C.to_csv(D / "dist_calib.csv", index=False)
