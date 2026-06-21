"""Campaña de experimentos para mejorar el modelo, medida en el backtest
multi-Mundial (1998-2022, 448 partidos), sin data leak.

Cachea los datasets por Mundial UNA vez y evalúa muchas variantes:
backend del modelo de goles, regularización alpha, rho de Dixon-Coles, y
una calibración por "temperatura" de las probabilidades 1X2. Reporta
RPS / log-loss / accuracy para decidir, con datos, qué se promueve.

Uso: python scripts/experiment_models.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.data.historical import (WC_BACKTEST_YEARS, WC_START,
                                 build_historical_dataset, wc_backtest_rows)
from core.data.wc_schema import match_features_frame
from core.models.poisson_goals import PoissonGoalsModel
from core.simulation.monte_carlo import _dixon_coles_matrix
from core.utils.metrics import ModelMetrics

CLASSES = ["1", "X", "2"]
IDX = {c: i for i, c in enumerate(CLASSES)}


def _probs(la, lb, rho):
    m = _dixon_coles_matrix(la, lb, rho)
    return (float(np.tril(m, -1).sum()), float(np.trace(m)),
            float(np.triu(m, 1).sum()))


def _temper(proba, t):
    """Temperature scaling: p^(1/t) renormalizado. t>1 suaviza, t<1 afila."""
    p = np.power(np.clip(proba, 1e-12, 1), 1.0 / t)
    return p / p.sum(axis=1, keepdims=True)


def _metrics(proba, results):
    idx = np.array([IDX[r] for r in results])
    acc = float((proba.argmax(1) == idx).mean())
    return (acc, ModelMetrics.multiclass_logloss(results, proba, CLASSES),
            ModelMetrics.rps(idx, proba))


def general_tuning():
    """Tunea alpha y temperatura en un set de VALIDACIÓN independiente:
    partidos NO-Mundial desde 2019 (amistosos/eliminatorias/continentales),
    entrenando con todo lo anterior. Sin leak hacia el backtest de Mundiales.
    """
    print("=" * 64)
    print("0) TUNEO HONESTO en validación general (no-Mundial >= 2019)")
    print("=" * 64)
    data = build_historical_dataset(cutoff=None)
    rd = data["row_dates"].to_numpy()
    md = data["match_dates"].to_numpy()
    split = np.datetime64("2019-01-01")
    tr = rd < split
    val = (md >= split) & (~data["match_is_wc"])
    Xtr, ytr, wtr = data["X"][tr], data["y"][tr], data["w"][tr]
    Xm, Xa = data["X_match"][val], data["X_match_away"][val]
    yv = list(data["y_result"][val])
    print(f"  train team-rows: {tr.sum():,} | val matches (no-WC): {val.sum():,}")

    def val_lams(alpha):
        m = PoissonGoalsModel("poisson", alpha=alpha)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m.predict_lambda(Xm), m.predict_lambda(Xa)

    print(f"\n  {'alpha':>10} {'acc':>7} {'logloss':>9} {'RPS':>8}")
    best_a = None
    cache_a = {}
    for alpha in [1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0]:
        la, lb = val_lams(alpha)
        cache_a[alpha] = (la, lb)
        proba = np.array([_probs(a, b, -0.08) for a, b in zip(la, lb)])
        acc, ll, rps = _metrics(proba, yv)
        flag = ""
        if best_a is None or rps < best_a[1]:
            best_a = (alpha, rps); flag = " <-"
        print(f"  {alpha:>10} {acc:>7.3f} {ll:>9.3f} {rps:>8.4f}{flag}")

    la, lb = cache_a[best_a[0]]
    print(f"\n  {'T':>10} {'acc':>7} {'logloss':>9} {'RPS':>8}  (alpha={best_a[0]})")
    best_t = None
    for t in [0.75, 0.85, 0.95, 1.0, 1.1]:
        proba = np.array([_probs(a, b, -0.08) for a, b in zip(la, lb)])
        proba = _temper(proba, t)
        acc, ll, rps = _metrics(proba, yv)
        flag = ""
        if best_t is None or rps < best_t[1]:
            best_t = (t, rps); flag = " <-"
        print(f"  {t:>10} {acc:>7.3f} {ll:>9.3f} {rps:>8.4f}{flag}")
    print(f"\n  >>> elegidos en validación general: alpha={best_a[0]}, T={best_t[0]}\n")
    return best_a[0], best_t[0]


def main():
    tuned_alpha, tuned_t = general_tuning()
    print("Cacheando datasets por Mundial (una vez)...", flush=True)
    cache = {}
    for year in WC_BACKTEST_YEARS:
        data = build_historical_dataset(cutoff=WC_START[year])
        rows = wc_backtest_rows(year, data["snapshots"])
        cache[year] = (data, rows)
    print("listo.\n", flush=True)

    def lambdas_for(make_model):
        """Devuelve (lam_a, lam_b, results) concatenados sobre los 7 Mundiales."""
        LA, LB, RES = [], [], []
        for year, (data, rows) in cache.items():
            model = make_model()
            model.fit(data["X"], data["y"], sample_weight=data["w"])
            n = len(rows)
            lam = model.predict_lambda(match_features_frame(
                [r["feat_a"] for r in rows] + [r["feat_b"] for r in rows]))
            LA.extend(lam[:n]); LB.extend(lam[n:])
            RES.extend(r["result"] for r in rows)
        return np.array(LA), np.array(LB), RES

    def eval_lambdas(LA, LB, RES, rho=-0.08, t=1.0):
        proba = np.array([_probs(a, b, rho) for a, b in zip(LA, LB)])
        if t != 1.0:
            proba = _temper(proba, t)
        return _metrics(proba, RES)

    # ---------- 1) backend + alpha (modelo de goles) ----------
    print("=" * 64)
    print("1) BACKEND / REGULARIZACIÓN del modelo de goles (rho=-0.08)")
    print("=" * 64)
    print(f"  {'config':>22} {'acc':>7} {'logloss':>9} {'RPS':>8}")
    configs = {
        "poisson alpha=1e-4": lambda: PoissonGoalsModel("poisson", alpha=1e-4),
        "poisson alpha=1e-3*": lambda: PoissonGoalsModel("poisson", alpha=1e-3),
        "poisson alpha=1e-2": lambda: PoissonGoalsModel("poisson", alpha=1e-2),
        "poisson alpha=1e-1": lambda: PoissonGoalsModel("poisson", alpha=1e-1),
        "poisson alpha=1.0": lambda: PoissonGoalsModel("poisson", alpha=1.0),
        "gbm": lambda: PoissonGoalsModel("gbm"),
        "xgb": lambda: PoissonGoalsModel("xgb"),
    }
    best = None
    cached_lams = {}
    for name, mk in configs.items():
        try:
            LA, LB, RES = lambdas_for(mk)
        except Exception as e:
            print(f"  {name:>22}  (omitido: {e})")
            continue
        cached_lams[name] = (LA, LB, RES)
        acc, ll, rps = eval_lambdas(LA, LB, RES)
        flag = ""
        if best is None or rps < best[1]:
            best = (name, rps); flag = " <-"
        print(f"  {name:>22} {acc:>7.3f} {ll:>9.3f} {rps:>8.4f}{flag}")
    print(f"\n  mejor por RPS: {best[0]}")

    # ---------- 2) rho de Dixon-Coles (sobre el mejor backend) ----------
    LA, LB, RES = cached_lams[best[0]]
    print("\n" + "=" * 64)
    print(f"2) rho de Dixon-Coles (backend = {best[0]})")
    print("=" * 64)
    print(f"  {'rho':>8} {'acc':>7} {'logloss':>9} {'RPS':>8}")
    best_rho = None
    for rho in [-0.16, -0.12, -0.08, -0.04, 0.0, 0.04]:
        acc, ll, rps = eval_lambdas(LA, LB, RES, rho=rho)
        flag = ""
        if best_rho is None or rps < best_rho[1]:
            best_rho = (rho, rps); flag = " <-"
        print(f"  {rho:>8.2f} {acc:>7.3f} {ll:>9.3f} {rps:>8.4f}{flag}")
    print(f"\n  mejor rho por RPS: {best_rho[0]}")

    # ---------- 3) calibración por temperatura ----------
    print("\n" + "=" * 64)
    print(f"3) Temperature scaling (backend={best[0]}, rho={best_rho[0]})")
    print("=" * 64)
    print(f"  {'T':>8} {'acc':>7} {'logloss':>9} {'RPS':>8}")
    best_t = None
    for t in [0.80, 0.90, 1.00, 1.10, 1.25]:
        acc, ll, rps = eval_lambdas(LA, LB, RES, rho=best_rho[0], t=t)
        flag = ""
        if best_t is None or rps < best_t[1]:
            best_t = (t, rps); flag = " <-"
        print(f"  {t:>8.2f} {acc:>7.3f} {ll:>9.3f} {rps:>8.4f}{flag}")
    print(f"\n  mejor T por RPS: {best_t[0]}")

    # ---------- CONFIRMACIÓN HONESTA en el backtest de Mundiales ----------
    # con la config elegida en la validación GENERAL (no en el test de WC)
    print("\n" + "=" * 64)
    print("CONFIRMACIÓN: config de validación general aplicada al test de WC")
    print("=" * 64)
    LAt, LBt, RESt = lambdas_for(
        lambda: PoissonGoalsModel("poisson", alpha=tuned_alpha))
    base_acc, base_ll, base_rps = eval_lambdas(*cached_lams["poisson alpha=1e-3*"])
    fin_acc, fin_ll, fin_rps = eval_lambdas(LAt, LBt, RESt, t=tuned_t)
    print(f"  ACTUAL (alpha=1e-3, T=1.0):        acc={base_acc:.3f} "
          f"logloss={base_ll:.3f} RPS={base_rps:.4f}")
    print(f"  TUNED  (alpha={tuned_alpha}, T={tuned_t}): acc={fin_acc:.3f} "
          f"logloss={fin_ll:.3f} RPS={fin_rps:.4f}")


if __name__ == "__main__":
    main()
