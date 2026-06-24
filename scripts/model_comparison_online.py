#!/usr/bin/env python3
"""Tabla comparativa tipo "model comparison" para los 6 modelos en los partidos
ya jugados: core/microsim/ensemble × SIN datos (pre-torneo) y CON datos (online).

Para cada partido jugado calcula 1X2, marcador modal consistente y puntos 5/3/0,
y agrega accuracy + RPS por modelo. Salida: results/reports/model_comparison_online.md

OJO (leakage): "con datos" sobre la 1ª ronda es IN-SAMPLE (los ratings ya
incorporan ese resultado). Sirve para ver el ajuste, no como predicción honesta;
la comparación limpia sin/con será en las rondas que aún no se juegan.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.simulation.monte_carlo import consistent_modal_score
from online_learning.dataset import load_results
from online_learning.panels import MODELS, REGIMES, _variant
from online_learning.priors import to_es

REPORT = ROOT / "results/reports/model_comparison_online.md"
VARIANTS = [(m, r) for m in MODELS for r in REGIMES]


def _modal(v) -> str:
    m = np.asarray(v["matrix"])
    i, j = consistent_modal_score(m, *v["probs"])
    return f"{i}-{j}"


def _points(score, gh, ga) -> int:
    pi, pj = (int(x) for x in score.split("-"))
    ar = "1" if gh > ga else ("2" if gh < ga else "X")
    pr = "1" if pi > pj else ("2" if pi < pj else "X")
    return 5 if (pi, pj) == (gh, ga) else (3 if pr == ar else 0)


def _rps(probs, gh, ga) -> float:
    idx = 0 if gh > ga else (2 if gh < ga else 1)
    cp = np.cumsum(probs); co = np.cumsum(np.eye(3)[idx])
    return float(np.sum((cp - co) ** 2) / 2)


def main() -> None:
    from ensemble.predict import _context
    ctx = _context()
    res = load_results()
    rows, tot = [], {vk: {"pts": 0, "hit": 0, "rps": []} for vk in VARIANTS}
    for r in res.itertuples():
        h, a = to_es(r.home_team), to_es(r.away_team)
        gh, ga = int(r.home_goals), int(r.away_goals)
        rec = {"match": f"{h} – {a}", "real": f"{gh}-{ga}"}
        for vk in VARIANTS:
            v = _variant(ctx, h, a, "2026-06-15", *vk)
            score = _modal(v)
            pts = _points(score, gh, ga)
            rec[f"{vk[0][:4]}_{vk[1]}"] = (
                f"{v['probs'][0]*100:.0f}/{v['probs'][1]*100:.0f}/"
                f"{v['probs'][2]*100:.0f} {score} ({pts})")
            tot[vk]["pts"] += pts
            tot[vk]["hit"] += 1 if pts >= 3 else 0
            tot[vk]["rps"].append(_rps(v["probs"], gh, ga))
        rows.append(rec)

    _write(rows, tot, len(res))
    print(f"Reporte: {REPORT}")
    for vk in VARIANTS:
        t = tot[vk]
        print(f"  {vk[0]:>9}/{vk[1]}: {t['pts']} pts · {t['hit']}/{len(res)} "
              f"resultados · RPS {np.mean(t['rps']):.4f}")


def _write(rows, tot, n) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"{m[:4]}_{r}" for m, r in VARIANTS]
    md = ["# Comparación de modelos — SIN vs CON datos 2026 (partidos jugados)\n",
          f"{n} partidos jugados. Cada celda: P(1)/P(X)/P(2) + marcador modal "
          "consistente + (puntos 5/3/0).\n",
          "⚠️ **'con' es in-sample en la 1ª ronda** (los ratings ya incluyen ese "
          "resultado): sirve para ver ajuste, no como predicción honesta.\n",
          "## Totales\n",
          "| Modelo | Puntos (5/3/0) | Aciertos resultado | RPS |",
          "|---|:-:|:-:|:-:|"]
    for (m, r) in VARIANTS:
        t = tot[(m, r)]
        md.append(f"| {m} · {r} datos | {t['pts']} | {t['hit']}/{n} "
                  f"({t['hit']/n:.0%}) | {np.mean(t['rps']):.4f} |")
    md += ["\n## Partido a partido\n",
           "| Partido | Real | " + " | ".join(cols) + " |",
           "|---|:-:|" + "---|" * len(cols)]
    for rec in rows:
        md.append(f"| {rec['match']} | {rec['real']} | "
                  + " | ".join(rec[c] for c in cols) + " |")
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
