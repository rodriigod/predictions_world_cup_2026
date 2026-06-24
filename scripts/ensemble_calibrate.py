#!/usr/bin/env python3
"""Ajusta y APLICA temperature scaling (calibración) sobre las predicciones OOF.

A diferencia de scripts/ensemble_calibration.py (que solo DIAGNOSTICA), este
ajusta el parámetro T y mide la mejora real de RPS/log-loss, con un chequeo
held-out temporal (T ajustado en el pasado, evaluado en el futuro) para que la
mejora no sea sobreajuste. Guarda el T del ensemble en models/ y reporta.

Uso: python scripts/ensemble_calibrate.py
Salidas: models/ensemble_temperature.json
         results/reports/calibration_applied.md
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.utils.metrics import ModelMetrics
from ensemble.calibrate import TemperatureScaler
from ensemble.evaluate import oof_predictions
from ensemble.meta_model import CLASSES, IDX

T_PATH = ROOT / "models/ensemble_temperature.json"
REPORT = ROOT / "results/reports/calibration_applied.md"


def _metrics(proba, y) -> tuple[float, float]:
    idx = y.map(IDX).to_numpy() if hasattr(y, "map") else np.asarray(y)
    return (ModelMetrics.rps(idx, proba),
            ModelMetrics.multiclass_logloss(
                [CLASSES[i] for i in idx], proba, CLASSES))


def _fit_report(name, proba, idx, order) -> dict:
    """Ajusta T sobre TODO el OOF y, aparte, valida held-out temporal (60/40)."""
    T_all = TemperatureScaler().fit(proba, idx).T_
    rps0, ll0 = ModelMetrics.rps(idx, proba), \
        ModelMetrics.multiclass_logloss([CLASSES[i] for i in idx], proba, CLASSES)
    cal = TemperatureScaler(T_all).transform(proba)
    rps1, ll1 = ModelMetrics.rps(idx, cal), \
        ModelMetrics.multiclass_logloss([CLASSES[i] for i in idx], cal, CLASSES)

    # held-out temporal: T en el 60% más antiguo, evaluar en el 40% reciente
    k = int(0.6 * len(order))
    tr, va = order[:k], order[k:]
    T_tr = TemperatureScaler().fit(proba[tr], idx[tr]).T_
    base_rps = ModelMetrics.rps(idx[va], proba[va])
    cal_rps = ModelMetrics.rps(idx[va], TemperatureScaler(T_tr).transform(proba[va]))
    return {"T_all": T_all, "rps0": rps0, "ll0": ll0, "rps1": rps1, "ll1": ll1,
            "T_holdout": T_tr, "ho_base": base_rps, "ho_cal": cal_rps}


def main() -> None:
    meta_df, probas, mask = oof_predictions()
    idx = meta_df["result"].map(IDX).to_numpy()[mask]
    # orden cronológico DENTRO de las filas OOF (para el held-out temporal)
    dates = meta_df["date"][mask].reset_index(drop=True)
    order = np.argsort(dates.values, kind="stable")
    n = int(mask.sum())
    print(f"\nCalibración (temperature scaling) sobre {n} partidos OOF\n")

    results = {}
    for name in ["solo core", "stacking (OOF)"]:
        r = _fit_report(name, probas[name][mask], idx, order)
        results[name] = r
        print(f"{name}:")
        print(f"  T*={r['T_all']:.3f}  RPS {r['rps0']:.4f} -> {r['rps1']:.4f}  "
              f"logloss {r['ll0']:.3f} -> {r['ll1']:.3f}")
        print(f"  held-out (T={r['T_holdout']:.3f}): RPS {r['ho_base']:.4f} -> "
              f"{r['ho_cal']:.4f}  ({'mejora' if r['ho_cal'] < r['ho_base'] else 'no mejora'})")

    # guardar el T del ENSEMBLE (lo aplica predict_final a su salida)
    T_ens = results["stacking (OOF)"]["T_all"]
    TemperatureScaler(T_ens).save(T_PATH)
    print(f"\nGuardado T del ensemble = {T_ens:.3f} -> {T_PATH}")

    _write_report(n, results, T_ens)
    print(f"Reporte: {REPORT}")


def _write_report(n, results, T_ens) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# Calibración aplicada — temperature scaling ({n} partidos OOF)\n",
          "T<1 = afilar (más confianza). Ajustado por NLL sobre OOF leak-free.\n",
          "| Modelo | T* | RPS antes | RPS después | logloss antes | logloss después |",
          "|---|:-:|:-:|:-:|:-:|:-:|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['T_all']:.3f} | {r['rps0']:.4f} | "
                  f"{r['rps1']:.4f} | {r['ll0']:.3f} | {r['ll1']:.3f} |")
    md.append("\n## Chequeo held-out temporal (T ajustado en 60% antiguo, "
              "evaluado en 40% reciente)\n")
    md.append("| Modelo | T (train) | RPS sin calibrar | RPS calibrado |")
    md.append("|---|:-:|:-:|:-:|")
    for name, r in results.items():
        md.append(f"| {name} | {r['T_holdout']:.3f} | {r['ho_base']:.4f} | "
                  f"{r['ho_cal']:.4f} |")
    core = results["solo core"]
    verdict = ("AFILAR (T<1): core está sub-confiado, calibrar ayuda."
               if core["T_all"] < 0.97 else
               "SUAVIZAR (T>1): core está sobre-confiado."
               if core["T_all"] > 1.03 else
               "core ya está prácticamente calibrado (T≈1); la corrección es mínima.")
    md.append(f"\n## Lectura\n\n- core: T*={core['T_all']:.3f} -> {verdict}")
    md.append(f"- El T del ENSEMBLE ({T_ens:.3f}) se guardó en "
              "`models/ensemble_temperature.json` y `predict_final` lo aplica a "
              "su salida (apply_calibration=True).")
    md.append("- La mejora de RPS suele ser pequeña (la calibración no cambia el "
              "argmax, solo la confianza); el valor está en log-loss/Brier y en "
              "que las probabilidades reflejen mejor la frecuencia real.")
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
