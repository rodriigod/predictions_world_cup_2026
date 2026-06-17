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
    FeatureSpec("pi_attack_own", "float", "1_rendimiento",
                "pi-rating OFENSIVO propio (Constantinou&Fenton 2013): fuerza "
                "de ataque latente, actualizada online con el error de goles",
                "z-score (fit en train)"),
    FeatureSpec("pi_defense_opp", "float", "1_rendimiento",
                "pi-rating DEFENSIVO del RIVAL (+ = concede menos): debilidad/"
                "solidez defensiva que enfrenta el ataque propio",
                "z-score (fit en train)"),
    FeatureSpec("xg_for_own", "float", "1_rendimiento",
                "xG a favor por partido, últimos 10 oficiales (propio)",
                "z-score (fit en train)"),
    FeatureSpec("xg_against_opp", "float", "1_rendimiento",
                "xG en contra por partido del RIVAL (debilidad defensiva "
                "que enfrenta el ataque propio)", "z-score"),
    FeatureSpec("form_diff", "float", "1_rendimiento",
                "form_last5_points_pct propio - rival", "ya en [-1,1]"),
    FeatureSpec("form_competitive_diff", "float", "1_rendimiento",
                "forma en partidos NO amistosos (últimos 5) propia - rival. "
                "Validada leak-free (experiment_features.py): baja el RPS del "
                "backtest 0.1982->0.1971; la única candidata de la Parte 2 que "
                "aporta señal real (los amistosos engañan a la forma simple).",
                "ya en [-1,1]"),
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

# Features CANDIDATAS (Parte 2): se computan siempre en build_match_features
# pero NO entran en producción. Medidas leak-free (scripts/experiment_features.py):
# momentum_diff y form10_diff dan Δ−0.0001 (ruido) y EMPEORAN al combinarse con
# form_competitive_diff; form_std_diff da Δ+0.0001 (peor). Solo
# form_competitive_diff aportó señal real y FUE PROMOVIDA a FEATURES arriba.
# Estas quedan documentadas como descartadas, inertes para producción.
CANDIDATE_FEATURES: list[FeatureSpec] = [
    FeatureSpec("momentum_diff", "float", "1_rendimiento",
                "tendencia (forma5 − forma10) propia − rival: + = subiendo "
                "[descartada: Δ−0.0001, ruido]", "ya en [-1,1]"),
    FeatureSpec("form10_diff", "float", "1_rendimiento",
                "forma (pts%) últimos 10 propia − rival "
                "[descartada: Δ−0.0001, ruido]", "ya en [-1,1]"),
    FeatureSpec("form_std_diff", "float", "1_rendimiento",
                "consistencia: std de pts últimos 10, propia − rival "
                "[descartada: Δ+0.0001, peor]", "ya en [0,1] aprox"),
]
CANDIDATE_NAMES = [f.name for f in CANDIDATE_FEATURES]
ALL_FEATURE_NAMES = FEATURE_NAMES + CANDIDATE_NAMES

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
        # pi-ratings: si la fila no los trae (modo sintético) caen a neutro 0
        "pi_attack_own": float(own.get("pi_attack", 0.0)),
        "pi_defense_opp": float(opp.get("pi_defense", 0.0)),
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
        # ---- features candidatas (inertes hasta validarse, ver schema) ----
        **_candidate_features(own, opp),
    }


def _candidate_features(own: pd.Series, opp: pd.Series) -> dict:
    """Features candidatas de la Parte 2. Usan .get con defaults para no
    romper el modo sintético (filas sin estos campos)."""
    def g(s, k, default=0.0):
        try:
            v = s.get(k, default)
        except AttributeError:
            v = s[k] if k in s else default
        return float(default if v is None else v)
    o5, a5 = g(own, "form_last5_points_pct", 0.5), g(opp, "form_last5_points_pct", 0.5)
    o10 = g(own, "form_last10_points_pct", o5)
    a10 = g(opp, "form_last10_points_pct", a5)
    return {
        "momentum_diff": (o5 - o10) - (a5 - a10),
        "form10_diff": o10 - a10,
        "form_std_diff": g(own, "form_std10") - g(opp, "form_std10"),
        "form_competitive_diff": (g(own, "form_comp5_points_pct", o5)
                                  - g(opp, "form_comp5_points_pct", a5)),
    }


def match_features_frame(rows: list[dict],
                         columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns or FEATURE_NAMES)


def schema_as_dataframe() -> pd.DataFrame:
    """Exporta el esquema (útil para documentar f2_intermedia)."""
    return pd.DataFrame([vars(f) for f in FEATURES])
