#!/usr/bin/env python3
"""Backtest comparativo del ensemble sobre los 7 Mundiales (1998-2022).

Responde con NÚMEROS la pregunta del encargo: ¿el stacking realmente gana, o es
ruido? Compara 4 enfoques sobre los MISMOS partidos, con accuracy y RPS:

  1. solo core/        (Poisson + Dixon-Coles)
  2. solo microsim/    (fuerza de plantel desde ELO, retroactivo)
  3. promedio simple   (core + microsim, renormalizado)
  4. stacking          (meta-modelo logístico, OOF temporal leak-free)

El stacking se evalúa OUT-OF-FOLD con TimeSeriesSplit (entrena en el pasado,
predice el futuro) para no hacer trampa. Además imprime los COEFICIENTES del
meta-modelo (cuánto pesa cada input).

Uso:
    python scripts/ensemble_backtest.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.utils.metrics import ModelMetrics
from ensemble.calibration_methods import cv_calibrated_oof
from ensemble.dataset import X_y, build_backtest_dataset
from ensemble.evaluate import oof_predictions
from ensemble.meta_model import CLASSES, IDX, StackingMetaModel
from ensemble.pooling import cv_extremized_oof, log_pool

REPORT = ROOT / "results/reports/ensemble_backtest.md"

# orden de presentación: enfoques + calibración (A) + pooling (E) + baselines.
ORDER = ["solo core", "solo microsim", "promedio core+micro", "stacking (OOF)",
         "core + isotonic", "core + Platt",
         "log-pooling", "log-pooling extremizado",
         "baseline FIFA ranking", "baseline uniforme"]


def _metrics(y: pd.Series, proba: np.ndarray) -> dict:
    idx = y.map(IDX).to_numpy()
    pred = proba.argmax(axis=1)
    return {"acc": float((pred == idx).mean()),
            "rps": ModelMetrics.rps(idx, proba),
            "brier": ModelMetrics.brier_multiclass(idx, proba),
            "logloss": ModelMetrics.multiclass_logloss(y, proba, CLASSES)}


def _augment_probas(probas: dict, mask: np.ndarray, y: pd.Series) -> float:
    """Añade a `probas` (sobre las filas OOF enmascaradas) los métodos nuevos:
    calibración de core (isotónica/Platt, CV-OOF) y log-pooling (+extremizado).
    Devuelve el `a` global del extremizing para reportarlo. Los métodos que
    requieren ajuste se calculan OUT-OF-FOLD para no auto-engañarse."""
    idx = y.map(IDX).to_numpy()
    core_m = probas["solo core"][mask]
    micro_m = probas["solo microsim"][mask]
    n = mask.sum()

    def _full(rows_masked):
        full = np.full((len(mask), 3), np.nan)
        full[mask] = rows_masked
        return full

    probas["core + isotonic"] = _full(cv_calibrated_oof(core_m, idx, "isotonic"))
    probas["core + Platt"] = _full(cv_calibrated_oof(core_m, idx, "platt"))
    pooled = log_pool(core_m, micro_m)
    probas["log-pooling"] = _full(pooled)
    extr, a_glob = cv_extremized_oof(pooled, idx)
    probas["log-pooling extremizado"] = _full(extr)
    return a_glob


def main() -> None:
    df = build_backtest_dataset()
    X, y = X_y(df)
    print(f"\nDataset del meta-modelo: {len(df)} partidos de "
          f"{df['year'].nunique()} Mundiales\n")

    meta_df, probas, mask = oof_predictions(df)
    ys = meta_df["result"][mask].reset_index(drop=True)
    a_glob = _augment_probas(probas, mask, ys)

    print(f"COMPARACIÓN sobre {int(mask.sum())} partidos OOF "
          "(menor RPS/Brier/logloss = mejor):")
    print(f"  {'enfoque':>26} {'acc':>7} {'RPS':>8} {'Brier':>8} {'logloss':>9}")
    lines = []
    for name in ORDER:
        m = _metrics(ys, probas[name][mask])
        print(f"  {name:>26} {m['acc']:>7.3f} {m['rps']:>8.4f} "
              f"{m['brier']:>8.4f} {m['logloss']:>9.3f}")
        lines.append((name, m))

    # --- meta-modelo final (todo el dataset) + coeficientes ---
    meta_full = StackingMetaModel().fit(X, y)
    print(f"\nMeta-modelo final: C={meta_full.C_} "
          f"(elegido por CV temporal, RPS={meta_full.cv_rps_})")
    coef = meta_full.coefficients()
    print("\nCOEFICIENTES (escala estandarizada; |peso| = importancia):")
    print(f"  {'feature':>22} {'grupo':>9} {'coef_1':>8} {'coef_X':>8} "
          f"{'coef_2':>8} {'|peso|':>7}")
    for r in coef.itertuples():
        print(f"  {r.feature:>22} {r.group:>9} {r.coef_1:>8.3f} {r.coef_X:>8.3f} "
              f"{r.coef_2:>8.3f} {r.abs_mean:>7.3f}")

    # importancia agregada por grupo
    grp = coef.groupby("group")["abs_mean"].sum().sort_values(ascending=False)
    print("\nImportancia agregada por grupo (suma de |peso|):")
    for g, v in grp.items():
        print(f"  {g:>10}: {v:.3f}")

    _write_report(int(mask.sum()), lines, meta_full, coef, grp, a_glob)
    print(f"\nReporte: {REPORT}")


def _write_report(n_oof, lines, meta, coef, grp, a_glob) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Ensemble — backtest comparativo (7 Mundiales 1998-2022)\n",
          f"OOF sobre **{n_oof} partidos**. Menor RPS/Brier/logloss = mejor. "
          "Brier = multiclase (Σ por clase, promedio por partido).\n",
          "Calibración de core (isotónica/Platt) y pooling extremizado se miden "
          "OUT-OF-FOLD (el calibrador/`a` no ve la fila que evalúa).\n",
          "| Enfoque | Accuracy | RPS | Brier | Log-loss |",
          "|---|:-:|:-:|:-:|:-:|"]
    for name, m in lines:
        md.append(f"| {name} | {m['acc']:.3f} | {m['rps']:.4f} | "
                  f"{m['brier']:.4f} | {m['logloss']:.3f} |")
    md.append(f"\n_Log-pooling extremizado: a global = {a_glob:.2f} "
              f"({'afila' if a_glob > 1 else 'suaviza' if a_glob < 1 else 'neutro'}, "
              "elegido por CV minimizando RPS)._")
    md.append(f"\nMeta-modelo: logística multinomial L2, "
              f"**C={meta.C_}** (CV temporal).\n")
    md.append("## Coeficientes (escala estandarizada)\n")
    md.append("| Feature | Grupo | coef_1 | coef_X | coef_2 | \\|peso\\| |")
    md.append("|---|:-:|:-:|:-:|:-:|:-:|")
    for r in coef.itertuples():
        md.append(f"| {r.feature} | {r.group} | {r.coef_1:.3f} | {r.coef_X:.3f} "
                  f"| {r.coef_2:.3f} | {r.abs_mean:.3f} |")
    md.append("\n## Importancia por grupo (suma \\|peso\\|)\n")
    for g, v in grp.items():
        md.append(f"- **{g}**: {v:.3f}")
    md.append("\n> Nota honesta: las señales del LLM entran NEUTRAS en el "
              "backtest histórico (no hay forma leak-free de reconstruir "
              "lesiones/consenso de 1998-2022), así que sus coeficientes salen "
              "~0 por FALTA DE SEÑAL EN ENTRENAMIENTO, no por medirse inútiles. "
              "Quedan cableadas para la inferencia 2026 (datos reales).")
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
