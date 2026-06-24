#!/usr/bin/env python3
"""Simula el torneo completo para los 6 modelos (core/microsim/ensemble × sin/con
datos 2026) y compara la probabilidad de campeón y de alcanzar cada ronda.

Salidas:
  results/reports/tournament_sim/<model>_<regime>.csv   (por equipo, todas las rondas)
  results/reports/tournament_comparison.md              (campeón % lado a lado)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from online_learning.panels import MODELS, REGIMES
from online_learning.tournament import MatrixProvider, simulate

OUTDIR = ROOT / "results/reports/tournament_sim"
REPORT = ROOT / "results/reports/tournament_comparison.md"
N_SIMS = 1500


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    champ = {}
    full = {}
    for model in MODELS:
        for regime in REGIMES:
            prov = MatrixProvider(model, regime)
            df = simulate(prov, n_sims=N_SIMS, seed=0)
            df.to_csv(OUTDIR / f"{model}_{regime}.csv", index=False)
            champ[(model, regime)] = df.set_index("team")["champion"]
            full[(model, regime)] = df
            print(f"  {model}/{regime}: campeón -> "
                  + ", ".join(f"{t} {p:.0%}"
                              for t, p in df.set_index('team')['champion']
                              .head(3).items()))
    _write(champ, full)
    print(f"\nReporte: {REPORT}")


def _col(model, regime):
    return f"{model[:4]}·{regime}"


def _write(champ, full) -> None:
    cols = [(m, r) for m in MODELS for r in REGIMES]
    cdf = pd.DataFrame({_col(m, r): champ[(m, r)] for m, r in cols}).fillna(0)
    cdf["orden"] = cdf.max(axis=1)
    cdf = cdf.sort_values("orden", ascending=False).drop(columns="orden").head(16)

    md = ["# Simulación del torneo — 6 modelos (campeón %)\n",
          f"Monte Carlo, {N_SIMS} torneos por modelo. Cuadro oficial 2026 "
          "(12 grupos, avanzan 1º/2º + 8 mejores 3º). `sin` = ratings pre-torneo; "
          "`con` = ratings online-updated con la 1ª ronda real.\n",
          "## Probabilidad de ser CAMPEÓN (top 16)\n",
          "| Equipo | " + " | ".join(_col(m, r) for m, r in cols) + " |",
          "|---|" + ":-:|" * len(cols)]
    for team, row in cdf.iterrows():
        md.append(f"| {team} | "
                  + " | ".join(f"{row[_col(m, r)]*100:.1f}" for m, r in cols)
                  + " |")
    md.append("\n## Cómo leerlo\n")
    md.append("- Cada columna es un modelo. `core·sin` vs `core·con` = efecto de "
              "incorporar la 1ª ronda real en core; igual para microsim y ensemble.")
    md.append("- microsim suele ser más extremo (favorece más a los fuertes); "
              "ensemble modera; core está en medio.")
    md.append("- Las tablas por ronda (16avos→final) están en "
              "`results/reports/tournament_sim/<modelo>_<regimen>.csv`.")
    md.append(f"\n> {N_SIMS} simulaciones: hay ruido de muestreo de ~±1pp en las "
              "probabilidades de campeón. Para diferencias finas, subir N_SIMS.")
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
