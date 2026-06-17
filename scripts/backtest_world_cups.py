"""Backtest del modelo sobre Mundiales pasados (1998-2022).

Para cada Mundial: entrena SOLO con partidos previos a ese torneo (mismo
protocolo pre-torneo que se usa para 2026), predice sus partidos con
Poisson + Dixon-Coles (el método real que mueve las predicciones) y mide
la calidad probabilística con métricas estándar de forecasting de fútbol.

Responde, con evidencia: ¿el modelo está calibrado? ¿degrada en
eliminación directa? ¿la heurística "partido parejo = empate" acierta?

Uso:
    python scripts/backtest_world_cups.py

Salida: results/reports/backtest_world_cups.csv + resumen por consola.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.historical import (WC_BACKTEST_YEARS, WC_START,
                                 build_historical_dataset, wc_backtest_rows)
from src.data.wc_schema import match_features_frame
from src.models.poisson_goals import PoissonGoalsModel
from src.models.result_classifier import _make_models
from src.models.stacked_classifier import StackedResultClassifier
from src.simulation.monte_carlo import _dixon_coles_matrix
from src.utils.metrics import ModelMetrics

CLASSES = ["1", "X", "2"]          # orden de las columnas de probabilidad
IDX = {c: i for i, c in enumerate(CLASSES)}
DRAW_MARGIN_PP = 3                  # misma regla "pareja = empate" de la polla
METHODS = ["poisson_dc", "logistic", "stack"]


def _probs_from_lambdas(la: float, lb: float) -> tuple[float, float, float]:
    m = _dixon_coles_matrix(la, lb)
    return (float(np.tril(m, -1).sum()), float(np.trace(m)),
            float(np.triu(m, 1).sum()))


def _proba_in_1x2(model, X) -> np.ndarray:
    """predict_proba reordenado a columnas [1, X, 2]."""
    p = model.predict_proba(X)
    pos = {c: i for i, c in enumerate(model.classes_)}
    return p[:, [pos["1"], pos["X"], pos["2"]]]


def _metrics_block(df: pd.DataFrame) -> dict:
    proba = df[["p1", "pX", "p2"]].to_numpy()
    idx = df["result"].map(IDX).to_numpy()
    pred = proba.argmax(axis=1)
    return {
        "n": len(df),
        "accuracy": float((pred == idx).mean()),
        "log_loss": ModelMetrics.multiclass_logloss(df["result"], proba, CLASSES),
        "rps": ModelMetrics.rps(idx, proba),
        "brier": ModelMetrics.brier_multiclass(idx, proba),
    }


def main() -> None:
    records = []
    for year in WC_BACKTEST_YEARS:
        cutoff = WC_START[year]
        print(f"[{year}] entrenando (Poisson + logística + stacking) ...",
              flush=True)
        data = build_historical_dataset(cutoff=cutoff)
        poisson = PoissonGoalsModel(backend="poisson")
        poisson.fit(data["X"], data["y"], sample_weight=data["w"])
        Xm, ym, wm = data["X_match"], data["y_result"], data["w_match"]
        logit = _make_models(42)["logistic_baseline"]
        logit.fit(Xm, ym, clf__sample_weight=wm)
        stack = StackedResultClassifier().fit(Xm, ym, wm)

        rows = wc_backtest_rows(year, data["snapshots"])
        n = len(rows)
        fm = match_features_frame([r["feat_a"] for r in rows])
        lam = poisson.predict_lambda(match_features_frame(
            [r["feat_a"] for r in rows] + [r["feat_b"] for r in rows]))
        p_logit = _proba_in_1x2(logit, fm)
        p_stack = stack.predict_proba_1x2(fm).to_numpy()

        for i, r in enumerate(rows):
            meta = {k: r[k] for k in ("year", "date", "home", "away",
                                      "stage", "city", "country",
                                      "gh", "ga", "result")}
            probs = {
                "poisson_dc": _probs_from_lambdas(float(lam[i]),
                                                  float(lam[n + i])),
                "logistic": tuple(p_logit[i]),
                "stack": tuple(p_stack[i]),
            }
            for method, (p1, pX, p2) in probs.items():
                records.append({**meta, "method": method,
                                "p1": p1, "pX": pX, "p2": p2})
        print(f"      {n} partidos × {len(METHODS)} métodos", flush=True)

    out = pd.DataFrame(records)
    out_path = ROOT / "results/reports/backtest_world_cups.csv"
    out.to_csv(out_path, index=False)
    prim = out[out["method"] == "poisson_dc"].reset_index(drop=True)

    # ---------------- resumen ----------------
    print("\n" + "=" * 70)
    print(f"BACKTEST — {len(prim)} partidos de {prim['year'].nunique()} "
          "Mundiales (1998-2022)")
    print("=" * 70)

    # baseline ingenuo: tasa base fija de cada resultado
    base = prim["result"].value_counts(normalize=True)
    base_proba = np.tile([base.get("1", 0), base.get("X", 0),
                          base.get("2", 0)], (len(prim), 1))
    base_rps = ModelMetrics.rps(prim["result"].map(IDX).to_numpy(), base_proba)
    base_ll = ModelMetrics.multiclass_logloss(prim["result"], base_proba,
                                              CLASSES)

    print("\nCOMPARACIÓN DE MÉTODOS (global, 448 partidos):")
    print(f"  {'método':>12} {'acc':>7} {'log-loss':>9} {'RPS':>8} {'Brier':>7}")
    for method in METHODS:
        m = _metrics_block(out[out["method"] == method])
        print(f"  {method:>12} {m['accuracy']:>7.3f} {m['log_loss']:>9.3f} "
              f"{m['rps']:>8.4f} {m['brier']:>7.3f}")
    print(f"  {'baseline':>12} {'—':>7} {base_ll:>9.3f} {base_rps:>8.4f} "
          f"{'—':>7}   <- todos deben ganarle")

    print("\nPor Mundial (método poisson_dc):")
    print(f"  {'año':>5} {'n':>4} {'acc':>6} {'logloss':>8} {'RPS':>7}")
    for year, sub in prim.groupby("year"):
        m = _metrics_block(sub)
        print(f"  {year:>5} {m['n']:>4} {m['accuracy']:>6.3f} "
              f"{m['log_loss']:>8.3f} {m['rps']:>7.4f}")

    print("\nPor etapa (método poisson_dc):")
    for stage, sub in prim.groupby("stage"):
        m = _metrics_block(sub)
        print(f"  {stage:>9}: n={m['n']:>3}  acc={m['accuracy']:.3f}  "
              f"log-loss={m['log_loss']:.3f}  RPS={m['rps']:.4f}")

    # ---- validación de la heurística "pareja = empate" ----
    prim["diff_pp"] = (prim["p1"] * 100).round() - (prim["p2"] * 100).round()
    even = prim[prim["diff_pp"].abs() <= DRAW_MARGIN_PP]
    groups = prim[prim["stage"] == "group"]
    print(f"\nHeurística 'pareja = empate' (|P1-P2| <= {DRAW_MARGIN_PP}pp):")
    print(f"  partidos parejos: {len(even)}  ->  "
          f"terminaron en empate: {(even['result'] == 'X').mean():.1%}")
    print(f"  tasa base de empate (fase de grupos): "
          f"{(groups['result'] == 'X').mean():.1%}")

    # ---- calibración: P(gana team_a) predicha vs observada ----
    rel = ModelMetrics.reliability_table(
        (prim["result"] == "1").to_numpy(), prim["p1"].to_numpy(), n_bins=10)
    print("\nCalibración de P(gana local/team_a)  [bien calibrado: pred≈obs]:")
    print(f"  {'p_pred':>7} {'p_obs':>7} {'n':>5}")
    for r in rel.itertuples():
        print(f"  {r.p_pred_mean:>7.2f} {r.p_obs:>7.2f} {r.n:>5}")

    print(f"\nCSV: {out_path}")

    # ---- plot de calibración opcional (--plots) ----
    if "--plots" in sys.argv:
        idx = prim["result"].map(IDX).to_numpy()
        png = ROOT / "results/reports/calibration_poisson_dc.png"
        ModelMetrics.plot_reliability_curve(
            idx, prim[["p1", "pX", "p2"]].to_numpy(), str(png),
            model_name="Poisson+Dixon-Coles")
        print(f"Reliability plot: {png}")


if __name__ == "__main__":
    main()
