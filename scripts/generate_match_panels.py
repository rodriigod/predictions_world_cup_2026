#!/usr/bin/env python3
"""Genera los paneles comparativos de los 72 partidos de fase de grupos.

Para CADA partido, 6 imágenes (core/microsim/ensemble × sin/con datos 2026), cada
una con matriz de marcadores 0-5×0-5, barras 1X2 y top-10 marcadores.

Uso:
  python scripts/generate_match_panels.py            # todos (resumible)
  python scripts/generate_match_panels.py --only "Brasil" "Haití" 2026-06-20
Salida: results/match_panels/<Local_vs_Visita>/<modelo>_<regimen>_datos.png
        results/match_panels/INDEX.md
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from online_learning.panels import OUTDIR, _safe, generate_match

FIXTURES = ROOT / "files/f0_raw/fixtures_2026.csv"


def _done(home, away) -> bool:
    folder = OUTDIR / f"{_safe(home)}_vs_{_safe(away)}"
    return folder.exists() and len(list(folder.glob("*.png"))) >= 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs=3, metavar=("HOME", "AWAY", "DATE"))
    ap.add_argument("--force", action="store_true", help="regenera aunque existan")
    args = ap.parse_args()

    if args.only:
        generate_match(*args.only)
        print(f"OK: {args.only[0]} vs {args.only[1]}")
        return

    fx = pd.read_csv(FIXTURES)
    n = len(fx)
    index = ["# Paneles por partido — fase de grupos (2026)\n",
             "6 modelos por partido: core/microsim/ensemble × sin/con datos 2026. "
             "Cada panel: matriz de marcadores 0-5×0-5 (casilla resaltada = más "
             "probable) + barras 1X2 + top-10 marcadores.\n"]
    for k, r in enumerate(fx.itertuples(), 1):
        home, away, date = str(r.team_a), str(r.team_b), str(r.date)
        tag = f"[{k}/{n}] G{r.group} J{r.matchday} {home} vs {away}"
        if not args.force and _done(home, away):
            print(f"  {tag}  (ya existe, skip)")
        else:
            try:
                generate_match(home, away, date)
                print(f"  {tag}  ✅")
            except Exception as e:
                print(f"  {tag}  ⚠️ {type(e).__name__}: {e}")
                continue
        folder = f"{_safe(home)}_vs_{_safe(away)}"
        index.append(f"- **{home} vs {away}** (G{r.group} J{r.matchday}, {date}): "
                     + " · ".join(f"[{m}_{rg}](./{folder}/{m}_{rg}_datos.png)"
                                  for m in ("core", "microsim", "ensemble")
                                  for rg in ("sin", "con")))
    (OUTDIR).mkdir(parents=True, exist_ok=True)
    (OUTDIR / "INDEX.md").write_text("\n".join(index), encoding="utf-8")
    print(f"\nIndex: {OUTDIR / 'INDEX.md'}")


if __name__ == "__main__":
    main()
