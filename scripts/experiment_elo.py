"""Tuneo HONESTO de los hiperparámetros del ELO/decaimiento — nunca se
habían optimizado (son valores de convención). Como el ELO es la señal
dominante, es el lugar con más chance de ganancia real.

Validación: partidos NO-Mundial desde 2019 (entrenando con lo anterior),
método Poisson+Dixon-Coles (el de producción). Sin leak.

Uso: python scripts/experiment_elo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.data.historical import build_historical_dataset
from core.models.poisson_goals import PoissonGoalsModel
from core.simulation.monte_carlo import _dixon_coles_matrix
from core.utils.metrics import ModelMetrics

CLASSES = ["1", "X", "2"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SPLIT = np.datetime64("2019-01-01")


def evaluate(**elo_kw):
    data = build_historical_dataset(cutoff=None, **elo_kw)
    rd = data["row_dates"].to_numpy()
    md = data["match_dates"].to_numpy()
    tr = rd < SPLIT
    val = (md >= SPLIT) & (~data["match_is_wc"])
    model = PoissonGoalsModel("poisson")
    model.fit(data["X"][tr], data["y"][tr], sample_weight=data["w"][tr])
    la = model.predict_lambda(data["X_match"][val])
    lb = model.predict_lambda(data["X_match_away"][val])
    proba = np.array([(float(np.tril(m := _dixon_coles_matrix(a, b), -1).sum()),
                       float(np.trace(m)), float(np.triu(m, 1).sum()))
                      for a, b in zip(la, lb)])
    yv = list(data["y_result"][val])
    idx = np.array([IDX[v] for v in yv])
    return (float((proba.argmax(1) == idx).mean()),
            ModelMetrics.multiclass_logloss(yv, proba, CLASSES),
            ModelMetrics.rps(idx, proba))


def sweep(name, key, values, base):
    print(f"\n{name} (resto en baseline):")
    print(f"  {'valor':>10} {'acc':>7} {'logloss':>9} {'RPS':>8}")
    best = None
    for v in values:
        acc, ll, rps = evaluate(**{**base, key: v})
        flag = ""
        if best is None or rps < best[1]:
            best = (v, rps); flag = " <-"
        tag = " (actual)" if v == base.get(key, BASELINE[key]) else ""
        print(f"  {v:>10} {acc:>7.3f} {ll:>9.3f} {rps:>8.4f}{flag}{tag}")
    return best[0]


BASELINE = {"home_adv_elo": 100.0, "k_mult": 1.0, "half_life": 8.0}


def main():
    acc, ll, rps = evaluate(**BASELINE)
    print("=" * 60)
    print(f"BASELINE (producción): acc={acc:.3f} logloss={ll:.3f} RPS={rps:.4f}")
    print("=" * 60)

    bh = sweep("1) Ventaja de localía (ELO)", "home_adv_elo",
               [50, 75, 100, 125, 150], BASELINE)
    bk = sweep("2) Escala del factor K", "k_mult",
               [0.5, 0.75, 1.0, 1.5, 2.0], BASELINE)
    bhl = sweep("3) Vida media del decaimiento (años)", "half_life",
                [3, 5, 8, 12, 30], BASELINE)

    print("\n" + "=" * 60)
    combo = {"home_adv_elo": bh, "k_mult": bk, "half_life": bhl}
    acc2, ll2, rps2 = evaluate(**combo)
    print(f"BASELINE        : acc={acc:.3f} logloss={ll:.3f} RPS={rps:.4f}")
    print(f"MEJOR combinado : {combo}")
    print(f"                  acc={acc2:.3f} logloss={ll2:.3f} RPS={rps2:.4f}")


if __name__ == "__main__":
    main()
