"""Generador de datos de entrenamiento sintéticos.

PROPÓSITO: validar el pipeline end-to-end mientras no se cargan los
~7000 partidos internacionales reales 2010-2025 en files/f0_raw (ver
nota de diseño #3 del DATA_DICTIONARY.md). Genera un universo de
selecciones con fuerzas latentes de ataque/defensa, deriva de ellas las
features observables (con ruido, como ocurre con datos reales) y
muestrea los goles de la distribución de Poisson verdadera.

CALIBRACIÓN: la escala está anclada al ELO real (media 1500, sd 170 a
nivel mundial), de modo que los contrastes extremos del Mundial (p.ej.
España 2157 vs Catar 1421) caigan dentro del rango de entrenamiento.
Con k=0.35 por unidad de fuerza, una diferencia de 400 puntos ELO
produce lambdas ~2.9 vs ~0.55 (margen esperado ~2.3 goles), consistente
con la expectativa de victoria de 91% que implica el ELO.

Cuando existan datos reales, reemplazar `make_training_data` por un
loader que construya las mismas columnas de wc_schema.FEATURE_NAMES
desde partidos históricos.
"""

import numpy as np
import pandas as pd

from .wc_schema import FEATURE_NAMES

BASE_GOALS = 1.25   # media de goles por equipo en fútbol de selecciones
K_STRENGTH = 0.35   # efecto de 1 unidad de fuerza (=170 ELO) sobre log-lambda
ELO_PER_STRENGTH = 170.0


def _synthetic_universe(n_teams: int, rng: np.random.Generator) -> pd.DataFrame:
    strength = rng.normal(0.0, 1.0, n_teams)            # fuerza latente
    attack = strength + rng.normal(0, 0.25, n_teams)     # ataque latente
    defense = strength + rng.normal(0, 0.25, n_teams)    # defensa latente
    elo = 1500 + strength * ELO_PER_STRENGTH + rng.normal(0, 25, n_teams)
    return pd.DataFrame({
        "attack": attack,
        "defense": defense,
        "elo": elo,
        # observables ruidosos correlacionados con las fuerzas latentes
        # (misma forma funcional que scripts/update_team_data.py)
        "xg_for_last10": np.clip(
            BASE_GOALS * np.exp(0.25 * attack) + rng.normal(0, 0.12, n_teams),
            0.5, 3.2),
        "xg_against_last10": np.clip(
            BASE_GOALS * np.exp(-0.25 * defense) + rng.normal(0, 0.12, n_teams),
            0.5, 3.2),
        "form_last5_points_pct": np.clip(
            0.5 + 0.13 * strength + rng.normal(0, 0.12, n_teams), 0, 1),
        "market_value_meur": np.clip(
            np.exp(3.45 + 0.95 * strength + rng.normal(0, 0.35, n_teams)),
            5, 2000),
        "avg_caps": np.clip(
            34 + 6 * strength + rng.normal(0, 6, n_teams), 5, 90),
        "players_with_wc_experience": np.clip(
            (9 + 4 * strength + rng.normal(0, 3, n_teams)).round(), 0, 26),
        "injury_impact_index": np.clip(
            rng.exponential(0.05, n_teams), 0, 0.5),
        "is_host": 0.0,
        "distance_avg_km": rng.uniform(500, 14000, n_teams),
    })


def make_training_data(n_matches: int = 8000, n_teams: int = 150,
                       seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) con X[FEATURE_NAMES] y y = goles del equipo."""
    from .wc_schema import build_match_features

    rng = np.random.default_rng(seed)
    universe = _synthetic_universe(n_teams, rng)

    rows, targets = [], []
    for _ in range(n_matches):
        i, j = rng.choice(n_teams, size=2, replace=False)
        a, b = universe.iloc[i], universe.iloc[j]
        matchday = int(rng.integers(1, 4))
        # localía ocasional (amistosos/eliminatorias tienen local real)
        host_a = float(rng.random() < 0.45)
        a = a.copy(); a["is_host"] = host_a
        b = b.copy(); b["is_host"] = float(rng.random() < 0.45) * (1 - host_a)

        # lambdas verdaderos del proceso generador
        lam_a = BASE_GOALS * np.exp(
            K_STRENGTH * (a["attack"] - b["defense"])
            + 0.18 * a["is_host"] - 0.10 * b["is_host"]
            - 0.07 * (matchday == 1))
        lam_b = BASE_GOALS * np.exp(
            K_STRENGTH * (b["attack"] - a["defense"])
            + 0.18 * b["is_host"] - 0.10 * a["is_host"]
            - 0.07 * (matchday == 1))

        rows.append(build_match_features(a, b, matchday))
        targets.append(rng.poisson(lam_a))
        rows.append(build_match_features(b, a, matchday))
        targets.append(rng.poisson(lam_b))

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(targets, name="goals_scored")
    return X, y
