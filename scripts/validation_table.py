#!/usr/bin/env python3
"""Validación de predicciones contra resultados reales (✅/❌) + tabla "con datos".

Sobre los partidos ya jugados (online_learning/data/results_2026.csv):
  - Tabla A: ✅/❌ de acierto de RESULTADO por modelo (los 6: core/microsim/
    ensemble × sin/con), con 🎯 si además acertó el marcador exacto, + totales.
  - Tabla B: predicciones de los modelos CON DATOS (core/micro/ensemble con) con
    su 1X2 + marcador + ✅/❌.

Salida: results/reports/validacion_modelos.md
        + inserta/actualiza la sección en README.md (entre marcadores HTML).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.simulation.monte_carlo import consistent_modal_score
from online_learning.dataset import load_results
from online_learning.panels import _variant
from online_learning.priors import to_es

REPORT = ROOT / "results/reports/validacion_modelos.md"
README = ROOT / "README.md"
MARK_A, MARK_B = "<!-- VALIDACION_ONLINE_START -->", "<!-- VALIDACION_ONLINE_END -->"

VARIANTS = [("core", "sin"), ("core", "con"), ("microsim", "sin"),
            ("microsim", "con"), ("ensemble", "sin"), ("ensemble", "con")]
CON = [("core", "con"), ("microsim", "con"), ("ensemble", "con")]
COL = {("core", "sin"): "core·sin", ("core", "con"): "core·con",
       ("microsim", "sin"): "micro·sin", ("microsim", "con"): "micro·con",
       ("ensemble", "sin"): "ens·sin", ("ensemble", "con"): "ens·con"}


def _pred(v):
    m = np.asarray(v["matrix"])
    i, j = consistent_modal_score(m, *v["probs"])
    pr = "1" if i > j else ("2" if i < j else "X")
    return pr, f"{i}-{j}"


def _build():
    from ensemble.predict import _context
    ctx = _context()
    res = load_results()
    matches = []
    totals = {vk: {"hit": 0, "exact": 0} for vk in VARIANTS}
    for r in res.itertuples():
        h, a = to_es(r.home_team), to_es(r.away_team)
        gh, ga = int(r.home_goals), int(r.away_goals)
        ar = "1" if gh > ga else ("2" if gh < ga else "X")
        cells = {}
        for vk in VARIANTS:
            v = _variant(ctx, h, a, "2026-06-15", *vk)
            pr, score = _pred(v)
            ok = pr == ar
            exact = score == f"{gh}-{ga}"
            mark = ("🎯" if exact else "✅") if ok else "❌"
            cells[vk] = {"probs": v["probs"], "score": score, "mark": mark,
                         "ok": ok, "exact": exact}
            totals[vk]["hit"] += ok
            totals[vk]["exact"] += exact
        matches.append({"home": h, "away": a, "real": f"{gh}-{ga}",
                        "cells": cells})
    return matches, totals, len(res)


def _table_validation(matches, totals, n) -> str:
    head = "| Partido | Real | " + " | ".join(COL[vk] for vk in VARIANTS) + " |"
    sep = "|---|:-:|" + ":-:|" * len(VARIANTS)
    lines = [head, sep]
    for m in matches:
        row = [f"{m['home']} – {m['away']}", m["real"]]
        for vk in VARIANTS:
            c = m["cells"][vk]
            row.append(f"{c['score']} {c['mark']}")
        lines.append("| " + " | ".join(row) + " |")
    tot = ["**Aciertos resultado**", f"**/{n}**"]
    for vk in VARIANTS:
        tot.append(f"**{totals[vk]['hit']}/{n}**")
    lines.append("| " + " | ".join(tot) + " |")
    ex = ["Marcador exacto 🎯", ""]
    for vk in VARIANTS:
        ex.append(f"{totals[vk]['exact']}")
    lines.append("| " + " | ".join(ex) + " |")
    return "\n".join(lines)


def _table_con(matches) -> str:
    head = "| Partido | Real | " + " | ".join(COL[vk] for vk in CON) + " |"
    sep = "|---|:-:|" + ":-:|" * len(CON)
    lines = [head, sep]
    for m in matches:
        row = [f"{m['home']} – {m['away']}", m["real"]]
        for vk in CON:
            c = m["cells"][vk]
            p1, pX, p2 = c["probs"]
            row.append(f"{p1*100:.0f}/{pX*100:.0f}/{p2*100:.0f} {c['score']} {c['mark']}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _upcoming_table(ctx) -> tuple[str, int]:
    """Tabla de los próximos partidos: marcador predicho por cada uno de los 6
    modelos (sin/con datos). Marcador = modal consistente con el 1X2."""
    import pandas as pd
    fx = pd.read_csv(ROOT / "files/f0_raw/fixtures_2026.csv")
    played = load_results()
    done = {frozenset((r.home_team, r.away_team)) for r in played.itertuples()}
    head = "| J | G | Partido | " + " | ".join(COL[vk] for vk in VARIANTS) + " |"
    lines = [head, "|:-:|:-:|---|" + ":-:|" * len(VARIANTS)]
    rows = []
    from online_learning.priors import canon
    for r in fx.itertuples():
        if frozenset((canon(r.team_a), canon(r.team_b))) in done:
            continue
        cells = []
        for vk in VARIANTS:
            v = _variant(ctx, r.team_a, r.team_b, str(r.date), *vk)
            _, score = _pred(v)
            cells.append(score)
        rows.append((int(r.matchday), r.group,
                     f"{r.team_a} – {r.team_b}", cells))
    rows.sort(key=lambda x: (x[0], x[1]))
    for md, g, match, cells in rows:
        lines.append(f"| {md} | {g} | {match} | " + " | ".join(cells) + " |")
    return "\n".join(lines), len(rows)


def main() -> None:
    matches, totals, n = _build()
    val = _table_validation(matches, totals, n)
    con = _table_con(matches)
    from ensemble.predict import _context
    upcoming, n_up = _upcoming_table(_context())

    md = [f"# Validación de modelos — {n} partidos jugados (✅/❌)\n",
          "✅ = acertó el resultado (1X2) · 🎯 = acertó el marcador exacto · "
          "❌ = falló el resultado. Marcador = modal consistente con el 1X2.\n",
          "## Tabla A — acierto de resultado por modelo (sin vs con datos)\n",
          val,
          "\n## Tabla B — predicciones de los modelos CON DATOS\n",
          "Cada celda: P(1)/P(X)/P(2) + marcador + ✅/❌.\n",
          con,
          f"\n## Tabla C — predicciones de los PRÓXIMOS {n_up} partidos\n",
          "Marcador predicho por cada modelo (sin/con datos).\n",
          upcoming,
          "\n> 'con datos' en partidos jugados es in-sample (los ratings ya "
          "incluyen ese resultado): es para ver el ajuste. La validación limpia "
          "out-of-sample llega con las jornadas que faltan.\n"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"Reporte: {REPORT}")
    for vk in VARIANTS:
        print(f"  {COL[vk]:>10}: {totals[vk]['hit']}/{n} resultados, "
              f"{totals[vk]['exact']} exactos")

    _update_readme(val, con, upcoming, n, n_up)


def _update_readme(val, con, upcoming, n, n_up) -> None:
    section = "\n".join([
        MARK_A,
        "## 📊 Modelos SIN datos vs CON datos (online learning)\n",
        "El sistema corre en **dos regímenes**:",
        "- **SIN datos** — ratings **pre-torneo** (core, microsim, ensemble; el "
        "LLM entra dentro del ensemble). Es la predicción de producción.",
        "- **CON datos** — los **mismos** modelos pero con los ratings "
        "**re-estimados con los resultados reales** del Mundial a medida que se "
        "juegan (módulo `online_learning/`, en paralelo, sin tocar producción).\n",
        f"### Validación sobre {n} partidos jugados (✅ resultado · 🎯 exacto · ❌ falló)\n",
        val,
        "\n### Predicciones de los modelos CON DATOS (partidos jugados)\n",
        con,
        f"\n### Predicciones de los PRÓXIMOS {n_up} partidos (marcador por modelo)\n",
        upcoming,
        f"\n_Detalle, 1X2 e imágenes por partido: `results/reports/validacion_modelos.md`, "
        "`proximos_partidos_predicciones.md` y `results/match_panels/`. "
        "'con datos' en partidos jugados es in-sample._",
        MARK_B])
    text = README.read_text(encoding="utf-8")
    if MARK_A in text and MARK_B in text:
        pre = text.split(MARK_A)[0]
        post = text.split(MARK_B)[1]
        text = pre + section + post
    else:
        text = text.rstrip() + "\n\n---\n\n" + section + "\n"
    README.write_text(text, encoding="utf-8")
    print(f"README actualizado: sección de validación ({n} partidos)")


if __name__ == "__main__":
    main()
