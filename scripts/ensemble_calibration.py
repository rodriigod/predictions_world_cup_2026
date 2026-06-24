#!/usr/bin/env python3
"""C. Reliability diagram / calibración de core sobre las predicciones OOF.

DIAGNÓSTICO (no aplica calibración): muestra, por clase 1/X/2, la probabilidad
predicha vs la frecuencia real observada por bins, y resume dónde core está
sobre/sub-confiado. Sirve para decidir DESPUÉS qué método de calibración usar.

Uso: python scripts/ensemble_calibration.py
Salidas: results/reports/calibration_core_oof.png
         results/reports/calibration_core_oof.md
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.utils.metrics import ModelMetrics
from ensemble.evaluate import oof_predictions
from ensemble.meta_model import IDX

PNG = ROOT / "results/reports/calibration_core_oof.png"
MD = ROOT / "results/reports/calibration_core_oof.md"
CLASS_NAMES = ["1 (gana local)", "X (empate)", "2 (gana visita)"]
N_BINS = 8


def main() -> None:
    meta_df, probas, mask = oof_predictions()
    core = probas["solo core"][mask]
    idx = meta_df["result"].map(IDX).to_numpy()[mask]
    n = len(idx)
    print(f"\nCalibración de core sobre {n} partidos OOF\n")

    PNG.parent.mkdir(parents=True, exist_ok=True)
    ModelMetrics.plot_reliability_curve(idx, core, str(PNG),
                                        model_name="core (OOF)", n_bins=N_BINS)

    md = [f"# C. Calibración de core (OOF, {n} partidos)\n",
          f"Reliability diagram: `{PNG.name}`. Por clase, prob. predicha vs "
          "frecuencia real por bins. `gap = pred − obs`: **gap>0 = "
          "sobre-confianza** (predice más de lo que ocurre), gap<0 = "
          "sub-confianza.\n"]
    overall = []
    for c, name in enumerate(CLASS_NAMES):
        tab = ModelMetrics.reliability_table((idx == c).astype(float),
                                             core[:, c], n_bins=N_BINS)
        print(f"Clase {name}:")
        print(f"  {'p_pred':>7} {'p_obs':>7} {'gap':>7} {'n':>5}")
        md += [f"\n## Clase {name}\n",
               "| p_pred | p_obs | gap (pred−obs) | n |", "|:-:|:-:|:-:|:-:|"]
        for r in tab.itertuples():
            gap = r.p_pred_mean - r.p_obs
            overall.append(abs(gap) * r.n)
            print(f"  {r.p_pred_mean:>7.2f} {r.p_obs:>7.2f} {gap:>+7.2f} "
                  f"{r.n:>5}")
            md.append(f"| {r.p_pred_mean:.2f} | {r.p_obs:.2f} | {gap:+.3f} | "
                      f"{r.n} |")
        # resumen de la clase: ¿sobre o sub confiada en general?
        hi = tab[tab["p_pred_mean"] >= 0.5]
        tilt = ("sobre-confianza en prob. altas"
                if not hi.empty and (hi["p_pred_mean"] - hi["p_obs"]).mean() > 0.02
                else "sub-confianza en prob. altas"
                if not hi.empty and (hi["p_pred_mean"] - hi["p_obs"]).mean() < -0.02
                else "razonablemente calibrada")
        md.append(f"\n_Resumen {name}: {tilt}._")

    ece = float(np.sum(overall) / (3 * n))     # error de calibración promedio
    print(f"\nECE aproximado (promedio |gap| ponderado): {ece:.4f}")
    md.append(f"\n## Resumen global\n\n- **ECE aproximado** (|gap| medio "
              f"ponderado por bin/clase): **{ece:.4f}**.")
    md.append("- Diagnóstico, sin aplicar calibración todavía. Si la "
              "sobre-confianza en probabilidades altas es marcada, "
              "*temperature scaling* o *isotónica por clase* serían los "
              "candidatos naturales (el backtest de core ya reportó T≈1.0, "
              "así que se espera poca corrección).")
    MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nPNG: {PNG}\nMD:  {MD}")


if __name__ == "__main__":
    main()
