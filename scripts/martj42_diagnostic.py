#!/usr/bin/env python3
"""C. Diagnóstico de integración del dataset martj42 (international_results).

NO mezcla nada automáticamente: solo COMPARA la fuente recién descargada
(files/f0_raw/martj42_results_latest.csv, bajada de raw.githubusercontent.com)
contra el dataset que YA usa core/ (files/f0_raw/international_results.csv) y
reporta:
  - solapamiento y partidos NETOS nuevos (anti-join por fecha+equipos+marcador);
  - rango temporal y torneos aportados por los netos nuevos;
  - nomenclatura de selecciones históricas/desaparecidas (West Germany, USSR,
    Yugoslavia, ...) que podrían romper el NAME_MAP/IDs de equipo del sistema.

Hallazgo esperado (documentado): el archivo que core/ ya usa ES martj42 (su
docstring lo dice), así que el delta suele ser ~0. El valor del diagnóstico es
confirmarlo con números y dejar la fuente candidata lista para una ampliación
controlada del set de entrenamiento (que NO se hace aquí).

Uso: python scripts/martj42_diagnostic.py
Salida: results/reports/martj42_integration.md
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.data.historical import NAME_MAP, TRAIN_FROM, WARMUP_FROM

CURRENT = ROOT / "files/f0_raw/international_results.csv"
LATEST = ROOT / "files/f0_raw/martj42_results_latest.csv"
REPORT = ROOT / "results/reports/martj42_integration.md"

# selecciones históricas/desaparecidas con nomenclatura distinta a la actual
HISTORICAL = [
    "West Germany", "East Germany", "Yugoslavia", "Soviet Union", "USSR",
    "Czechoslovakia", "Serbia and Montenegro", "Zaire", "Netherlands Antilles",
    "Dutch Guyana", "Yemen DPR", "North Yemen", "South Yemen", "Burma",
    "Ceylon", "Dahomey", "Upper Volta", "Zaïre", "Bohemia",
]


def _key(df: pd.DataFrame) -> pd.Series:
    f = df.copy()
    for c in ["home_score", "away_score"]:
        f[c] = f[c].astype("string").fillna("NA")
    cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    return f[cols].astype(str).agg("|".join, axis=1)


def main() -> None:
    cur = pd.read_csv(CURRENT)
    new = pd.read_csv(LATEST)
    ck, nk = _key(cur), _key(new)
    cset, nset = set(ck), set(nk)
    net_new = new[~nk.isin(cset)].copy()
    only_cur = cur[~ck.isin(nset)].copy()

    cur_teams = set(cur["home_team"]) | set(cur["away_team"])
    new_teams = set(new["home_team"]) | set(new["away_team"])
    hist_present = sorted(t for t in HISTORICAL if t in cur_teams | new_teams)
    # ¿qué históricas NO están en el NAME_MAP (no se mapean a una selección 2026)?
    mapped = set(NAME_MAP.values())
    hist_unmapped = [t for t in hist_present if t not in mapped]
    teams_2026 = sorted(NAME_MAP.values())
    missing_2026 = [t for t in teams_2026 if t not in new_teams]

    print(f"current: {len(cur)} filas | latest martj42: {len(new)} filas")
    print(f"NETOS nuevos (en latest, no en current): {len(net_new)}")
    print(f"solo en current (no en latest): {len(only_cur)}")
    print(f"selecciones históricas presentes: {hist_present}")
    print(f"selecciones 2026 (NAME_MAP) ausentes del dataset: {missing_2026}")

    _write(cur, new, net_new, only_cur, hist_present, hist_unmapped, missing_2026)
    print(f"\nReporte: {REPORT}")


def _tourn_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "(ninguno)"
    vc = df["tournament"].value_counts().head(6)
    return ", ".join(f"{k}: {v}" for k, v in vc.items())


def _write(cur, new, net_new, only_cur, hist_present, hist_unmapped,
           missing_2026) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    dr_new = (f"{net_new['date'].min()} → {net_new['date'].max()}"
              if not net_new.empty else "—")
    md = [
        "# C. Integración del dataset martj42 — diagnóstico (sin merge)\n",
        "Fuente: github.com/martj42/international_results "
        "(`results.csv`, bajado a `files/f0_raw/martj42_results_latest.csv`). "
        "Comparado contra el dataset que core/ ya usa "
        "(`files/f0_raw/international_results.csv`).\n",
        "## Conteos\n",
        f"- Dataset actual de core/: **{len(cur):,} partidos** "
        f"(1872→{cur['date'].max()}). core/ entrena desde {TRAIN_FROM} y "
        f"calienta ELO/forma desde {WARMUP_FROM}.",
        f"- martj42 (descarga fresca): **{len(new):,} partidos**.",
        f"- **Partidos NETOS nuevos** (anti-join fecha+equipos+marcador): "
        f"**{len(net_new):,}**.",
        f"- Solo en el actual y no en la descarga: **{len(only_cur):,}**.",
    ]
    if not net_new.empty:
        md += [f"- Rango temporal de los netos nuevos: {dr_new}.",
               f"- Torneos de los netos nuevos: {_tourn_summary(net_new)}."]
    md += [
        "\n## Lectura\n",
        ("El archivo que core/ ya usa **es** martj42 (lo dice su docstring), por "
         "lo que el delta es prácticamente nulo: la descarga **confirma** que el "
         "set está al día, no aporta partidos relevantes nuevos."
         if len(net_new) < 50 else
         f"La descarga aporta **{len(net_new):,} partidos nuevos** — fuente "
         "candidata para AMPLIAR el entrenamiento (no se mezcla aquí)."),
        f"\nLa copia fresca queda en `{LATEST.relative_to(ROOT)}` como **fuente "
        "candidata** (NO se mezcló al pipeline). Para adoptarla bastaría "
        "reemplazar `international_results.csv` y reconstruir la caché del "
        "dataset (`files/cache/ensemble/backtest_dataset.csv`).",
        "\n## Nomenclatura de selecciones (riesgo para IDs)\n",
        f"- Selecciones históricas/desaparecidas presentes: "
        f"{', '.join(hist_present) if hist_present else '(ninguna)'}.",
        ("- Ninguna de ellas está en `NAME_MAP` (que mapea ES→EN solo para las 48 "
         "del Mundial 2026), así que **no rompen los IDs**: core/ las usa solo "
         "para calcular ELO/forma histórica (por nombre de cadena, sin ID "
         "numérico), y nunca se cruzan con el fixture 2026."
         if hist_unmapped == hist_present else
         f"- ⚠️ Algunas históricas SÍ chocan con NAME_MAP: {hist_unmapped}."),
        (f"- Selecciones 2026 ausentes del dataset martj42: {missing_2026}."
         if missing_2026 else
         "- Las 48 selecciones del Mundial 2026 (vía NAME_MAP) están todas "
         "presentes en el dataset por su nombre actual."),
        "\n> El ELO/forma de core/ se acumula por **nombre actual** de la "
        "selección; las entidades sucesoras (p.ej. Germany hereda de West "
        "Germany solo si comparten nombre) NO se fusionan automáticamente — es "
        "una limitación conocida y honesta de usar el nombre como clave.",
    ]
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
