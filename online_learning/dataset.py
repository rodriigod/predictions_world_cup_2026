"""Carga y mantenimiento de results_2026.csv (resultados reales del Mundial).

El ORDEN cronológico importa (ELO/pi son recursivos), así que siempre se devuelve
ordenado por fecha y, dentro del día, por orden de inserción.

NOTA HONESTA sobre la data: el prompt mencionaba "los 24 partidos de la primera
ronda", pero solo 2 están disponibles como resultado REAL en el dataset del repo
(martj42): México-Sudáfrica y Corea del Sur-Rep. Checa (2026-06-11). El CSV se
siembra con esos 2 reales; los demás se agregan con `update_results.py --add`
a medida que se jueguen. No se inventan marcadores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from online_learning.priors import canon

DATA = Path(__file__).resolve().parent / "data"
RESULTS_CSV = DATA / "results_2026.csv"
COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals", "stage"]
STAGES = {"group", "r32", "r16", "qf", "sf", "final"}
KNOCKOUT = {"r32", "r16", "qf", "sf", "final"}


def load_results(*, canonical: bool = True) -> pd.DataFrame:
    """Resultados 2026 ordenados cronológicamente. Si `canonical`, normaliza los
    nombres de equipo a inglés (claves de los snapshots de core/)."""
    if not RESULTS_CSV.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    if canonical:
        df["home_team"] = df["home_team"].map(canon)
        df["away_team"] = df["away_team"].map(canon)
    return df


def append_result(date: str, home: str, away: str, hg: int, ag: int,
                  stage: str = "group") -> pd.DataFrame:
    """Agrega un partido al CSV (sin recomputar nada; eso lo hace el orquestador).
    Valida stage y que el partido no esté duplicado. Devuelve el CSV completo."""
    if stage not in STAGES:
        raise ValueError(f"stage inválido {stage!r}; usa uno de {sorted(STAGES)}")
    canon(home); canon(away)            # valida nombres (lanza si desconocido)
    DATA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else \
        pd.DataFrame(columns=COLUMNS)
    mid = f"{date}|{home}|{away}"
    existing = (df["date"].astype(str) + "|" + df["home_team"] + "|"
                + df["away_team"]) if len(df) else pd.Series([], dtype=str)
    if (existing == mid).any():
        raise ValueError(f"partido ya registrado: {mid}")
    row = {"date": date, "home_team": home, "away_team": away,
           "home_goals": int(hg), "away_goals": int(ag), "stage": stage}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    return df
