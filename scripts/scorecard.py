#!/usr/bin/env python3
"""Genera un scorecard limpio en su propio .md: Fecha · Grupo · Partido ·
Predicho · Real · Puntos (5 exacto / 3 resultado / 0), con totales.

Lee los marcadores PREDICHOS y los resultados REALES desde el scorecard que ya
está en README.md (los que cargaste con update_scorecard.py), recalcula los
puntos y los vuelca a results/reports/scorecard.md para verlo cómodo.

Uso:
    python scripts/scorecard.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT = ROOT / "results/reports/scorecard.md"

# | 11/06 | A | México – Sudáfrica | 2-0 | 2-0 | ✅ 5 |   (real/pts pueden estar vacíos)
ROW = re.compile(r"^\|\s*(\d\d/\d\d)\s*\|\s*([A-L])\s*\|\s*(.+?)\s*\|"
                 r"\s*(\d+-\d+)\s*\|\s*([^|]*?)\s*\|\s*[^|]*?\s*\|$")


def pts(pred: str, real: str):
    if not real:
        return None
    if pred == real:
        return 5
    pa, pb = map(int, pred.split("-")); ra, rb = map(int, real.split("-"))
    so = (pa > pb) - (pa < pb); ro = (ra > rb) - (ra < rb)
    return 3 if so == ro else 0


def main() -> None:
    rows = []
    for ln in README.read_text(encoding="utf-8").splitlines():
        m = ROW.match(ln)
        if m and "–" in m.group(3):
            date, grp, match, pred, real = (m.group(1), m.group(2),
                                            m.group(3).strip(), m.group(4),
                                            m.group(5).strip())
            rows.append([date, grp, match, pred, real])
    if not rows:
        sys.exit("No encontré filas de scorecard en README.md. "
                 "Cargá resultados con scripts/update_scorecard.py primero.")

    out = ["# 📝 Scorecard — Mundial 2026 (mi polla)\n",
           "Predicho = marcador del modelo · Real = resultado verdadero · "
           "Puntos: **5** marcador exacto, **3** solo resultado, **0** falla.\n",
           "| Fecha | Grupo | Partido | Predicho | Real | Puntos |",
           "|:-:|:-:|---|:-:|:-:|:-:|"]
    tot = played = exact = outcome = 0
    for date, grp, match, pred, real in rows:
        p = pts(pred, real)
        if p is None:
            cell = "— | —"
        else:
            played += 1; tot += p
            exact += (pred == real)
            outcome += (p >= 3)
            cell = f"{real} | {'✅' if p > 0 else '❌'} {p}"
        out.append(f"| {date} | {grp} | {match} | {pred} | {cell} |")

    acc = f"{outcome}/{played}" if played else "0/0"
    out.append(f"\n**Total: {tot} pts** en {played} jugados · "
               f"marcador exacto: {exact}/{played} · "
               f"resultado acertado: {acc} "
               f"({(outcome/played*100) if played else 0:.0f}%) · "
               f"faltan: {len(rows) - played}/72")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"✅ {played} jugados, {tot} pts → {OUT}")


if __name__ == "__main__":
    main()
