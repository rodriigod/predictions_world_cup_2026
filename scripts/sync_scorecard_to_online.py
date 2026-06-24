#!/usr/bin/env python3
"""Sincroniza los partidos JUGADOS del scorecard del README hacia online_learning.

Lee la tabla scorecard de README.md, toma las filas con resultado REAL y agrega a
online_learning/data/results_2026.csv las que falten (sin duplicar, comparando por
par de equipos sin importar el orden local/visita). Así el online learning y la
validación usan TODOS los partidos jugados, no solo la 1ª ronda.

Uso: python scripts/sync_scorecard_to_online.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from online_learning.dataset import RESULTS_CSV, append_result, load_results
from online_learning.priors import canon

README = ROOT / "README.md"
ROW = re.compile(r"^\|\s*(\d\d)/(\d\d)\s*\|\s*([A-L])\s*\|\s*(.+?)\s*\|"
                 r"\s*(\d+-\d+)\s*\|\s*(\d+-\d+)\s*\|")


def main() -> None:
    existing = load_results()
    have = {frozenset((r.home_team, r.away_team)) for r in existing.itertuples()}

    added = 0
    for ln in README.read_text(encoding="utf-8").splitlines():
        m = ROW.match(ln)
        if not m or "–" not in m.group(4):
            continue
        dd, mm, grp, match, _pred, real = m.groups()
        home, away = [s.strip() for s in match.split("–")]
        try:
            ch, ca = canon(home), canon(away)
        except KeyError as e:
            print(f"  ⚠️ nombre no reconocido, salto: {e}")
            continue
        if frozenset((ch, ca)) in have:
            continue
        gh, ga = (int(x) for x in real.split("-"))
        date = f"2026-{mm}-{dd}"
        append_result(date, home, away, gh, ga, "group")
        have.add(frozenset((ch, ca)))
        added += 1
        print(f"  + {date} {home} {gh}-{ga} {away} (G{grp})")

    total = len(load_results())
    print(f"\nAgregados: {added} | total jugados en online_learning: {total}")
    print(f"CSV: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
