#!/usr/bin/env python3
"""A. Compara calibración de core: SIN calibrar vs isotónica vs Platt.

Sobre las predicciones OOF de core (leak-free: cada Mundial usa solo el pasado),
mide RPS + Brier + log-loss de:
  - core sin calibrar
  - core + isotónica (one-vs-rest, CV-OOF)
  - core + Platt    (logística multinomial sobre el logit, CV-OOF)

La calibración se evalúa OUT-OF-FOLD (StratifiedKFold) para no sobre-ajustar la
calibración misma. DECISIÓN honesta: se queda con un método SOLO si mejora de
forma CONSISTENTE (RPS y Brier a la vez) frente a core sin calibrar; si ninguno
mejora, no se guarda nada y predict_final NO calibra core (flag por defecto OFF).

Si gana alguno, se reajusta sobre TODO el OOF y se guarda en
models/core_calibrator.json para que predict_final lo aplique (calibrate_core=True).

Uso: python scripts/calibration_compare.py
Salidas: results/reports/calibration_compare.md  [+ models/core_calibrator.json si gana]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.utils.metrics import ModelMetrics
from ensemble.calibration_methods import (cv_calibrated_oof, make_calibrator,
                                          save_calibrator)
from ensemble.evaluate import oof_predictions
from ensemble.meta_model import CLASSES, IDX

REPORT = ROOT / "results/reports/calibration_compare.md"
CAL_PATH = ROOT / "models/core_calibrator.json"


def _m(idx, proba) -> dict:
    return {"rps": ModelMetrics.rps(idx, proba),
            "brier": ModelMetrics.brier_multiclass(idx, proba),
            "logloss": ModelMetrics.multiclass_logloss(
                [CLASSES[i] for i in idx], proba, CLASSES),
            "acc": float((proba.argmax(1) == idx).mean())}


def main() -> None:
    meta_df, probas, mask = oof_predictions()
    idx = meta_df["result"].map(IDX).to_numpy()[mask]
    core = probas["solo core"][mask]
    n = int(mask.sum())
    print(f"\nCalibración de core sobre {n} partidos OOF (CV-OOF, leak-free)\n")

    rows = {"core sin calibrar": _m(idx, core)}
    iso = cv_calibrated_oof(core, idx, "isotonic")
    platt = cv_calibrated_oof(core, idx, "platt")
    rows["core + isotónica"] = _m(idx, iso)
    rows["core + Platt"] = _m(idx, platt)

    for name, r in rows.items():
        print(f"  {name:>20}  RPS {r['rps']:.4f}  Brier {r['brier']:.4f}  "
              f"logloss {r['logloss']:.3f}  acc {r['acc']:.3f}")

    base = rows["core sin calibrar"]
    # CONSISTENTE = mejora RPS y Brier a la vez (umbral pequeño para no premiar ruido)
    winners = {name: r for name, r in rows.items() if name != "core sin calibrar"
               and r["rps"] < base["rps"] - 1e-4
               and r["brier"] < base["brier"] - 1e-4}
    if winners:
        best = min(winners, key=lambda k: winners[k]["rps"])
        method = "isotonic" if "isot" in best else "platt"
        cal = make_calibrator(method).fit(core, idx)   # reajuste sobre TODO el OOF
        save_calibrator(cal, CAL_PATH)
        verdict = (f"**Gana {best}** (mejora RPS y Brier). Guardado en "
                   f"`{CAL_PATH.name}`; predict_final lo aplica con "
                   "`calibrate_core=True`.")
        print(f"\n{best} gana -> guardado {CAL_PATH}")
    else:
        if CAL_PATH.exists():
            CAL_PATH.unlink()
        verdict = ("**No gana ninguno**: ni isotónica ni Platt mejoran RPS y "
                   "Brier a la vez frente a core sin calibrar sobre held-out OOF. "
                   "No se guarda calibrador; predict_final deja core SIN calibrar "
                   "(`calibrate_core=False`, por defecto). Coherente con que el "
                   "temperature scaling ya había dado T≈1 y mejoras marginales: "
                   "calibrar core no aporta señal real en este conjunto.")
        print("\nNingún método mejora de forma consistente -> no se guarda calibrador")

    _write(n, rows, verdict)
    print(f"Reporte: {REPORT}")


def _write(n, rows, verdict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# A. Calibración de core — isotónica vs Platt ({n} partidos OOF)\n",
          "Evaluación OUT-OF-FOLD (StratifiedKFold): el calibrador no ve la fila "
          "que mide, así no se sobre-ajusta la calibración. Menor RPS/Brier/"
          "logloss = mejor.\n",
          "| Método | RPS | Brier | Log-loss | Accuracy |", "|---|:-:|:-:|:-:|:-:|"]
    for name, r in rows.items():
        md.append(f"| {name} | {r['rps']:.4f} | {r['brier']:.4f} | "
                  f"{r['logloss']:.3f} | {r['acc']:.3f} |")
    md.append(f"\n## Decisión\n\n{verdict}")
    md.append("\n> Nota metodológica: Platt = logística multinomial sobre el "
              "logit de las probs (generaliza temperature scaling, que es el caso "
              "de 1 parámetro). Isotónica = monótona no paramétrica por clase + "
              "renormalización; con ~344 partidos es la más expuesta a overfit, "
              "por eso la evaluación CV-OOF es imprescindible.")
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
