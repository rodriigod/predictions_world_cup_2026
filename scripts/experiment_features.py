"""¿Las features candidatas de la Parte 2 mejoran el modelo? (leak-free)

Mide, con EXACTAMENTE el mismo protocolo del backtest multi-Mundial
(entrenar solo con datos previos a cada torneo, predecir con Poisson +
Dixon-Coles), el RPS/log-loss de:

  - base                       (FEATURE_NAMES de producción)
  - base + cada candidata sola (aislar el aporte de cada una)
  - base + todas las candidatas

Conserva solo las que BAJAN el RPS. No toca producción: solo reporta.

Uso:
    python scripts/experiment_features.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.data.historical import (WC_BACKTEST_YEARS, WC_START,
                                 build_historical_dataset, wc_backtest_rows)
from core.data.wc_schema import (FEATURE_NAMES, CANDIDATE_NAMES,
                                match_features_frame)
from core.models.poisson_goals import PoissonGoalsModel
from core.simulation.monte_carlo import _dixon_coles_matrix
from core.utils.metrics import ModelMetrics

CLASSES = ["1", "X", "2"]
IDX = {c: i for i, c in enumerate(CLASSES)}


def _probs(la, lb):
    m = _dixon_coles_matrix(la, lb)
    return (float(np.tril(m, -1).sum()), float(np.trace(m)),
            float(np.triu(m, 1).sum()))


def evaluate(cols: list[str]) -> dict:
    """RPS/log-loss/acc global sobre los 7 Mundiales con este feature set."""
    recs = []
    for year in WC_BACKTEST_YEARS:
        data = build_historical_dataset(cutoff=WC_START[year], feature_names=cols)
        poisson = PoissonGoalsModel(backend="poisson", feature_names=cols)
        poisson.fit(data["X"], data["y"], sample_weight=data["w"])
        rows = wc_backtest_rows(year, data["snapshots"])
        n = len(rows)
        lam = poisson.predict_lambda(match_features_frame(
            [r["feat_a"] for r in rows] + [r["feat_b"] for r in rows],
            columns=cols))
        for i, r in enumerate(rows):
            p1, pX, p2 = _probs(float(lam[i]), float(lam[n + i]))
            recs.append({"result": r["result"], "p1": p1, "pX": pX, "p2": p2})
    df = pd.DataFrame(recs)
    proba = df[["p1", "pX", "p2"]].to_numpy()
    idx = df["result"].map(IDX).to_numpy()
    return {
        "n": len(df),
        "acc": float((proba.argmax(1) == idx).mean()),
        "logloss": ModelMetrics.multiclass_logloss(df["result"], proba, CLASSES),
        "rps": ModelMetrics.rps(idx, proba),
    }


def main() -> None:
    print("Midiendo feature sets (Poisson+DC, 7 Mundiales, leak-free)...\n")
    base = evaluate(FEATURE_NAMES)
    print(f"  {'base (producción)':<28} RPS={base['rps']:.4f}  "
          f"logloss={base['logloss']:.4f}  acc={base['acc']:.3f}")

    results = {}
    for cand in CANDIDATE_NAMES:
        m = evaluate(FEATURE_NAMES + [cand])
        results[cand] = m
        delta = m["rps"] - base["rps"]
        flag = "✅ mejora" if delta < 0 else "✗ no"
        print(f"  +{cand:<27} RPS={m['rps']:.4f}  Δ={delta:+.4f}  {flag}")

    allm = evaluate(FEATURE_NAMES + CANDIDATE_NAMES)
    print(f"  {'+ TODAS':<28} RPS={allm['rps']:.4f}  "
          f"Δ={allm['rps'] - base['rps']:+.4f}")

    winners = [c for c in CANDIDATE_NAMES if results[c]["rps"] < base["rps"]]
    # set combinado de ganadoras
    if winners:
        wm = evaluate(FEATURE_NAMES + winners)
        print(f"\n  ganadoras juntas {winners}: RPS={wm['rps']:.4f} "
              f"(Δ={wm['rps'] - base['rps']:+.4f})")
    print("\n" + "=" * 60)
    if winners and evaluate(FEATURE_NAMES + winners)["rps"] < base["rps"]:
        print(f"RECOMENDACIÓN: agregar a FEATURE_NAMES -> {winners}")
    else:
        print("RECOMENDACIÓN: NINGUNA candidata mejora el RPS. No promover.")
    print("=" * 60)


if __name__ == "__main__":
    main()
