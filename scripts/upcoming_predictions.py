#!/usr/bin/env python3
"""Tabla consolidada de predicciones para los PRÓXIMOS partidos de grupos
(los que aún no se juegan), para los 6 modelos: core/microsim/ensemble × sin/con.

Cada celda: P(1)/P(X)/P(2) + marcador modal consistente. Salida:
  results/reports/proximos_partidos_predicciones.md
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
from online_learning.priors import canon

FIXTURES = ROOT / "files/f0_raw/fixtures_2026.csv"
REPORT = ROOT / "results/reports/proximos_partidos_predicciones.md"
VARIANTS = [(m, r) for m in MODELS for r in REGIMES]


def _cell(v) -> str:
    p1, pX, p2 = v["probs"]
    i, j = consistent_modal_score(np.asarray(v["matrix"]), p1, pX, p2)
    return f"{p1*100:.0f}/{pX*100:.0f}/{p2*100:.0f} {i}-{j}"


def main() -> None:
    from ensemble.predict import _context
    ctx = _context()
    fx = pd.read_csv(FIXTURES)
    played = load_results()
    done = {(canon(r.home_team), canon(r.away_team)) for r in played.itertuples()}
    done |= {(b, a) for a, b in done}                  # ignora orden local/visita

    rows = []
    for r in fx.itertuples():
        if (canon(r.team_a), canon(r.team_b)) in done:
            continue
        rec = {"J": int(r.matchday), "G": r.group,
               "match": f"{r.team_a} – {r.team_b}", "date": str(r.date)}
        for vk in VARIANTS:
            rec[f"{vk[0][:4]}_{vk[1]}"] = _cell(
                _variant(ctx, r.team_a, r.team_b, str(r.date), *vk))
        rows.append(rec)
    rows.sort(key=lambda d: (d["J"], d["G"]))

    cols = [f"{m[:4]}_{r}" for m, r in VARIANTS]
    md = ["# Predicciones — próximos partidos de fase de grupos\n",
          f"{len(rows)} partidos sin jugar. Cada celda: **P(1)/P(X)/P(2) + "
          "marcador**. Modelos: core/microsim/ensemble × sin/con datos 2026.\n",
          "| J | G | Partido | " + " | ".join(cols) + " |",
          "|:-:|:-:|---|" + "---|" * len(cols)]
    for d in rows:
        md.append(f"| {d['J']} | {d['G']} | {d['match']} | "
                  + " | ".join(d[c] for c in cols) + " |")
    md.append("\n> Imágenes (heatmap + barras + top-10) de cada uno en "
              "`results/match_panels/<Local_vs_Visita>/`.")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"{len(rows)} próximos partidos -> {REPORT}")


if __name__ == "__main__":
    main()
