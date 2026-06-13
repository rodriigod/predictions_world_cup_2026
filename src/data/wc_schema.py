"""Esquema de features para el modelo de goles esperados (lambda).

Espejo programático de DATA_DICTIONARY.md. Cada fila del dataset de
modelado representa UN equipo en UN partido (perspectiva propia vs rival);
las features son contrastes equipo-rival, según la nota de diseño #1 del
diccionario.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str          # float | int | bool | categorical
    category: str       # 1_rendimiento | 2_contexto | 3_plantel | 4_torneo
    description: str
    normalization: str


FEATURES: list[FeatureSpec] = [
    # ---- Categoría 1: rendimiento deportivo ----
    FeatureSpec("elo_diff_scaled", "float", "1_rendimiento",
                "(ELO propio - ELO rival) / 400",
                "división por 400 (escala logística ELO)"),
    FeatureSpec("elo_win_expectancy", "float", "1_rendimiento",
                "1 / (1 + 10^(-elo_diff/400))", "ya en [0,1]"),
    FeatureSpec("xg_for_own", "float", "1_rendimiento",
                "xG a favor por partido, últimos 10 oficiales (propio)",
                "z-score (fit en train)"),
    FeatureSpec("xg_against_opp", "float", "1_rendimiento",
                "xG en contra por partido del RIVAL (debilidad defensiva "
                "que enfrenta el ataque propio)", "z-score"),
    FeatureSpec("form_diff", "float", "1_rendimiento",
                "form_last5_points_pct propio - rival", "ya en [-1,1]"),
    # ---- Categoría 2: contexto y logística ----
    FeatureSpec("is_host_own", "bool", "2_contexto",
                "el equipo juega en su propio país", "0/1"),
    FeatureSpec("is_host_opp", "bool", "2_contexto",
                "el rival juega en su propio país", "0/1"),
    FeatureSpec("distance_diff_log", "float", "2_contexto",
                "log1p(km propio a sede) - log1p(km rival a sede)",
                "diferencia de log1p"),
    FeatureSpec("rest_days_diff", "float", "2_contexto",
                "días de descanso propio - rival", "clipping [-4,4] / 4"),
    # ---- Categoría 3: plantel ----
    FeatureSpec("market_value_log_diff", "float", "3_plantel",
                "log(valor plantel M EUR) propio - rival", "diferencia de logs"),
    FeatureSpec("caps_log_diff", "float", "3_plantel",
                "log1p(caps promedio) propio - rival", "diferencia de log1p"),
    FeatureSpec("wc_experience_diff", "float", "3_plantel",
                "(jugadores con mundial previo propio - rival) / 26",
                "ya en [-1,1]"),
    FeatureSpec("injury_impact_diff", "float", "3_plantel",
                "injury_impact_index rival - propio (positivo = ventaja)",
                "ya en [-0.5,0.5]"),
    # ---- Categoría 4: torneo (las dinámicas se aplican DENTRO del Monte
    # Carlo como multiplicadores de lambda; aquí solo la jornada) ----
    FeatureSpec("matchday_2", "bool", "4_torneo", "es fecha 2", "one-hot"),
    FeatureSpec("matchday_3", "bool", "4_torneo", "es fecha 3", "one-hot"),
]

FEATURE_NAMES = [f.name for f in FEATURES]

# Columnas requeridas en el CSV de equipos (files/f0_raw/teams_*.csv)
TEAM_COLUMNS = [
    "team", "group", "confed", "elo", "xg_for_last10", "xg_against_last10",
    "form_last5_points_pct", "market_value_meur", "avg_caps",
    "players_with_wc_experience", "injury_impact_index", "is_host",
    "distance_avg_km",
]


def build_match_features(own: pd.Series, opp: pd.Series,
                         matchday: int = 1,
                         rest_days_diff: float = 0.0) -> dict:
    """Construye el vector de features de UN equipo contra UN rival.

    `own` y `opp` son filas del DataFrame de equipos con TEAM_COLUMNS.
    """
    elo_diff = own["elo"] - opp["elo"]
    return {
        "elo_diff_scaled": elo_diff / 400.0,
        "elo_win_expectancy": 1.0 / (1.0 + 10 ** (-elo_diff / 400.0)),
        "xg_for_own": own["xg_for_last10"],
        "xg_against_opp": opp["xg_against_last10"],
        "form_diff": own["form_last5_points_pct"] - opp["form_last5_points_pct"],
        "is_host_own": float(own["is_host"]),
        "is_host_opp": float(opp["is_host"]),
        "distance_diff_log": (np.log1p(own["distance_avg_km"])
                              - np.log1p(opp["distance_avg_km"])),
        "rest_days_diff": float(np.clip(rest_days_diff, -4, 4)) / 4.0,
        "market_value_log_diff": (np.log(own["market_value_meur"])
                                  - np.log(opp["market_value_meur"])),
        "caps_log_diff": np.log1p(own["avg_caps"]) - np.log1p(opp["avg_caps"]),
        "wc_experience_diff": (own["players_with_wc_experience"]
                               - opp["players_with_wc_experience"]) / 26.0,
        "injury_impact_diff": (opp["injury_impact_index"]
                               - own["injury_impact_index"]),
        "matchday_2": float(matchday == 2),
        "matchday_3": float(matchday == 3),
    }


def match_features_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def schema_as_dataframe() -> pd.DataFrame:
    """Exporta el esquema (útil para documentar f2_intermedia)."""
    return pd.DataFrame([vars(f) for f in FEATURES])
