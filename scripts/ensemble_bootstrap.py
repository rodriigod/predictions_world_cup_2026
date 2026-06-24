#!/usr/bin/env python3
"""A. Significancia estadística del backtest del ensemble (bootstrap).

Sobre los mismos partidos OOF, remuestrea con reemplazo N=2000 veces y mide el
intervalo de confianza al 95% de las diferencias de RPS frente a 'solo core'.
Si el IC de una diferencia INCLUYE 0, la diferencia observada puede ser ruido
de muestra (no evidencia sólida) — el reporte lo dice explícitamente.

Uso: python scripts/ensemble_bootstrap.py
Salida: results/reports/bootstrap_significance.md
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ensemble.evaluate import oof_predictions
from ensemble.meta_model import IDX
from scripts.ensemble_backtest import _augment_probas

N_BOOT = 2000
SEED = 42
REPORT = ROOT / "results/reports/bootstrap_significance.md"


def _per_match_rps(proba: np.ndarray, idx: np.ndarray) -> np.ndarray:
    cum_p = np.cumsum(proba, axis=1)
    cum_o = np.cumsum(np.eye(3)[idx], axis=1)
    return np.sum((cum_p - cum_o) ** 2, axis=1) / 2.0


def main() -> None:
    meta_df, probas, mask = oof_predictions()
    # añade los métodos NUEVOS (calibración A + pooling E) al mismo conjunto OOF
    ys = meta_df["result"][mask].reset_index(drop=True)
    _augment_probas(probas, mask, ys)
    idx = meta_df["result"].map(IDX).to_numpy()[mask]
    n = int(mask.sum())
    # RPS por partido de cada enfoque (alineado a las filas OOF)
    rps_match = {name: _per_match_rps(p[mask], idx) for name, p in probas.items()}
    point = {name: float(v.mean()) for name, v in rps_match.items()}

    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))
    boot_rps = {name: v[boot_idx].mean(axis=1) for name, v in rps_match.items()}

    base = boot_rps["solo core"]
    comparisons = ["core + isotonic", "core + Platt", "log-pooling",
                   "log-pooling extremizado", "promedio core+micro",
                   "stacking (OOF)", "solo microsim",
                   "baseline FIFA ranking", "baseline uniforme"]
    diffs = {}
    for name in comparisons:
        d = boot_rps[name] - base                  # RPS_X - RPS_core
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs[name] = (float(d.mean()), float(lo), float(hi),
                       bool(lo <= 0 <= hi))

    # ---------------- consola ----------------
    print(f"\nBootstrap de significancia — {n} partidos OOF, N={N_BOOT}\n")
    print("RPS puntual por enfoque:")
    for name, v in sorted(point.items(), key=lambda kv: kv[1]):
        print(f"  {name:>24}: {v:.4f}")
    print("\nDiferencia de RPS vs 'solo core'  (positivo = peor que core):")
    print(f"  {'enfoque':>24} {'Δmedia':>8} {'IC95% bajo':>11} "
          f"{'IC95% alto':>11} {'incluye 0?':>11}")
    for name, (mu, lo, hi, zero) in diffs.items():
        print(f"  {name:>24} {mu:>+8.4f} {lo:>+11.4f} {hi:>+11.4f} "
              f"{('SÍ' if zero else 'no'):>11}")

    _write_report(n, point, diffs)
    print(f"\nReporte: {REPORT}")


def _verdict(name: str, mu: float, lo: float, hi: float, zero: bool) -> str:
    if zero:
        return (f"El IC95% de ({name} − core) **incluye 0** "
                f"([{lo:+.4f}, {hi:+.4f}]): la diferencia observada NO es "
                "estadísticamente concluyente — puede ser ruido de muestra.")
    sign = "PEOR" if mu > 0 else "MEJOR"
    return (f"El IC95% de ({name} − core) **no incluye 0** "
            f"([{lo:+.4f}, {hi:+.4f}]): {name} es {sign} que core de forma "
            "estadísticamente significativa al 95%.")


def _write_report(n, point, diffs) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# A. Significancia del backtest (bootstrap, N={N_BOOT})\n",
          f"Sobre **{n} partidos OOF** del backtest de Mundiales. Remuestreo con "
          "reemplazo; menor RPS = mejor.\n",
          "## RPS puntual por enfoque\n",
          "| Enfoque | RPS |", "|---|:-:|"]
    for name, v in sorted(point.items(), key=lambda kv: kv[1]):
        md.append(f"| {name} | {v:.4f} |")
    md += ["\n## Diferencia de RPS vs «solo core» (IC95%)\n",
           "Positivo = peor que core. Si el IC incluye 0, no es concluyente.\n",
           "| Enfoque | Δ media | IC95% | ¿incluye 0? |", "|---|:-:|:-:|:-:|"]
    for name, (mu, lo, hi, zero) in diffs.items():
        md.append(f"| {name} | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                  f"{'SÍ' if zero else 'no'} |")
    md.append("\n## Veredicto\n")
    for name, (mu, lo, hi, zero) in diffs.items():
        md.append(f"- {_verdict(name, mu, lo, hi, zero)}")
    md.append("\n> Lectura honesta: con solo unos cientos de partidos, las "
              "diferencias de RPS entre enfoques basados en fuerza de equipo "
              "suelen ser pequeñas frente al ruido de muestra. Un IC que incluye "
              "0 significa que NO podemos afirmar con rigor que el stacking sea "
              "peor que core — solo que no lo mejora.")
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
