"""¿Un ajuste de empate por CONFEDERACIÓN mejora el modelo, sin leak?

Hipótesis (del análisis de patrones): equipos africanos empatan más de lo
que predice el modelo. Test honesto:
  - estima un multiplicador de empate por confederación en partidos
    NO-Mundial y ANTIGUOS (< 2014),
  - lo aplica a datos que NO vio: partidos generales >= 2014 Y los
    Mundiales (cross-confederación, como 2026),
  - mide log-loss / RPS / accuracy con y sin ajuste.

Si mejora en Mundiales -> sirve para 2026 y se promueve. Si no, se descarta.

Uso: python scripts/experiment_draw_adjust.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.historical import build_historical_dataset, load_results
from src.models.poisson_goals import PoissonGoalsModel
from src.simulation.monte_carlo import _dixon_coles_matrix
from src.utils.metrics import ModelMetrics

CLASSES = ["1", "X", "2"]
IDX = {c: i for i, c in enumerate(CLASSES)}
_CONF = {
    "UEFA": ["uefa", "euro", "nordic", "baltic", "british home"],
    "CONMEBOL": ["copa am", "conmebol"],
    "CONCACAF": ["concacaf", "gold cup", "cfu", "caribbean", "uncaf",
                 "central american"],
    "CAF": ["african", "africa cup", "caf ", "cosafa", "cecafa", "wafu",
            "cemac"],
    "AFC": ["afc", "asian", "gulf cup", "aff ", "saff", "asean"],
    "OFC": ["ofc", "oceania", "pacific games"],
}


def infer_confed():
    df = load_results()
    treg = {}
    for t in df["tournament"].unique():
        tl = t.lower()
        treg[t] = next((c for c, pats in _CONF.items()
                        if any(p in tl for p in pats)), None)
    cnt = {}
    for r in df.itertuples():
        c = treg.get(r.tournament)
        if c is None:
            continue
        for tm in (r.home_team, r.away_team):
            cnt.setdefault(tm, {})[c] = cnt.setdefault(tm, {}).get(c, 0) + 1
    return {tm: max(d, key=d.get) for tm, d in cnt.items() if d}


def walk_forward(first=2002, last=2024):
    data = build_historical_dataset(cutoff=None)
    rd = data["row_dates"].dt.year.to_numpy()
    my = data["match_dates"].dt.year.to_numpy()
    confed = infer_confed()
    out = []
    for year in range(first, last + 1):
        tr, te = rd < year, my == year
        if te.sum() == 0:
            continue
        m = PoissonGoalsModel("poisson")
        m.fit(data["X"][tr], data["y"][tr], sample_weight=data["w"][tr])
        la = m.predict_lambda(data["X_match"][te])
        lb = m.predict_lambda(data["X_match_away"][te])
        sub = pd.DataFrame({
            "year": year,
            "home": data["match_home"][te].to_numpy(),
            "away": data["match_away"][te].to_numpy(),
            "result": data["y_result"][te].to_numpy(),
            "is_wc": data["match_is_wc"][te],
        })
        mats = [_dixon_coles_matrix(a, b) for a, b in zip(la, lb)]
        sub["p1"] = [float(np.tril(x, -1).sum()) for x in mats]
        sub["pX"] = [float(np.trace(x)) for x in mats]
        sub["p2"] = [float(np.triu(x, 1).sum()) for x in mats]
        out.append(sub)
    df = pd.concat(out, ignore_index=True)
    df["ch"] = df["home"].map(confed)
    df["ca"] = df["away"].map(confed)
    df["is_draw"] = (df["result"] == "X").astype(int)
    return df


def fit_mult(train):
    """Multiplicador de empate por confederación (partidos intra)."""
    mult = {}
    for c in _CONF:
        s = train[(train["ch"] == c) & (train["ca"] == c)]
        mult[c] = (s["is_draw"].mean() / s["pX"].mean()
                   if len(s) >= 100 else 1.0)
    return mult


def apply_mult(df, mult):
    m = np.sqrt(df["ch"].map(lambda c: mult.get(c, 1.0)).to_numpy()
                * df["ca"].map(lambda c: mult.get(c, 1.0)).to_numpy())
    m = np.clip(m, 0.7, 1.6)
    pX2 = np.clip(df["pX"].to_numpy() * m, 1e-4, 0.95)
    scale = (1 - pX2) / (1 - df["pX"].to_numpy())
    return np.c_[df["p1"].to_numpy() * scale, pX2, df["p2"].to_numpy() * scale]


def metrics(proba, results):
    idx = np.array([IDX[r] for r in results])
    return (float((proba.argmax(1) == idx).mean()),
            ModelMetrics.multiclass_logloss(list(results), proba, CLASSES),
            ModelMetrics.rps(idx, proba))


def report(name, df, mult):
    base = df[["p1", "pX", "p2"]].to_numpy()
    adj = apply_mult(df, mult)
    res = list(df["result"])
    a0 = metrics(base, res)
    a1 = metrics(adj, res)
    print(f"\n{name}  (n={len(df)})")
    print(f"  {'':>10} {'acc':>7} {'logloss':>9} {'RPS':>8}")
    print(f"  {'sin ajuste':>10} {a0[0]:>7.3f} {a0[1]:>9.3f} {a0[2]:>8.4f}")
    print(f"  {'con ajuste':>10} {a1[0]:>7.3f} {a1[1]:>9.3f} {a1[2]:>8.4f}"
          f"   {'MEJORA' if a1[2] < a0[2] else 'no mejora'}")


def main():
    print("Prediciendo walk-forward 2002-2024 (sin leak)...", flush=True)
    df = walk_forward()
    train = df[(~df["is_wc"]) & (df["year"] < 2014)]
    mult = fit_mult(train)
    print("\nMultiplicador de empate por confederación (estimado en <2014):")
    for c, v in mult.items():
        print(f"  {c:>10}: x{v:.3f}")

    report("TEST general (no-Mundial >= 2014)",
           df[(~df["is_wc"]) & (df["year"] >= 2014)], mult)
    report("TEST MUNDIALES (cross-confed, lo que importa para 2026)",
           df[df["is_wc"]], mult)


if __name__ == "__main__":
    main()
