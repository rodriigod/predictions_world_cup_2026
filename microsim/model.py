"""Motor de microsimulación basado en fuerza de plantel.

Toma los índices ataque/defensa de `strength.py` (derivados del valor de
mercado, ver `ingest.py`), los mapea a goles esperados (λ) con la estructura
clásica de Maher (1982) — λ = base · ataque_propio / defensa_rival — y simula
el partido vía Monte Carlo reusando la matriz Dixon-Coles de `core/`.

La salida cumple el contrato común `schema.MatchPrediction`, igual que el
pipeline de producción, así que el `ensemble/` puede combinar este modelo con
los demás sin conocer sus tripas.

LIMITACIÓN CENTRAL (honestidad, nota #5 del encargo)
----------------------------------------------------
Esto NO es una microsimulación jugador-a-jugador tipo videojuego (no simula
pases, posiciones ni eventos). Es un modelo de FUERZA DE PLANTEL: usa el VALOR
DE MERCADO como PROXY de calidad, lo separa en ataque/defensa ponderando por
posición, y de ahí saca λ. El valor de mercado correlaciona con la calidad,
pero arrastra sesgos conocidos (infravalora a jugadores de ligas menores y a
selecciones fuera de las 5 grandes ligas; reacciona con retraso a la forma y a
las lesiones). Por defecto los planteles son SINTÉTICOS (el total del equipo
repartido por una plantilla típica), no listas reales por jugador — ver
`ingest.synthetic_squad_from_total`. Tómalo como una segunda opinión barata
para el ensemble, no como una verdad de simulación física.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.simulation.monte_carlo import (DC_RHO, LAMBDA_JITTER, MAX_GOALS,
                                         _dixon_coles_matrix, _sample_score)
from core.simulation.monte_carlo import dc_1x2
from microsim.ingest import load_all_squads
from microsim.strength import (TeamStrength, compute_strengths,
                               strengths_from_elo)
from schema import MatchPrediction

MODEL_NAME = "microsim_market_value"
MODEL_VERSION = "1.0.0"

# Tasa base de goles por equipo en un cruce promedio (índices ataque=defensa=1).
# ~1.35 goles/equipo es la media histórica internacional (ver core/historical).
BASE_GOALS = 1.35
# Exponentes de elasticidad: cuánto responde λ a la ventaja de ataque propio y
# a la (debilidad de) defensa rival. 1.0 = proporcional puro (Maher).
ATTACK_ELASTICITY = 1.0
DEFENSE_ELASTICITY = 1.0
# Localía: en el Mundial las sedes son neutrales -> 1.0 por defecto. Se expone
# por si se quiere modelar la ventaja del anfitrión (México/EEUU/Canadá).
HOME_ADVANTAGE = 1.0
LAMBDA_CLIP = (0.25, 3.5)


@dataclass
class MarketValueMicroSim:
    """Modelo de microsimulación por fuerza de plantel.

    `strengths`: {equipo: TeamStrength} ya normalizado (media de liga = 1.0).
    Constrúyelo a mano o con `MarketValueMicroSim.from_teams(teams_df)`.
    """
    strengths: dict[str, TeamStrength]
    base_goals: float = BASE_GOALS
    home_advantage: float = HOME_ADVANTAGE
    rho: float = DC_RHO
    lambda_jitter: float = LAMBDA_JITTER
    seed: int = 42

    # ----------------------------- construcción ------------------------------
    @classmethod
    def from_teams(cls, teams_df: pd.DataFrame, *, allow_network: bool = False,
                   transfermarkt_urls: dict[str, str] | None = None,
                   **kwargs) -> "MarketValueMicroSim":
        """Carga planteles (ingest) -> índices (strength) -> modelo, en un paso.
        Por defecto offline (allow_network=False): usa el fallback sintético
        desde `market_value_meur`, que existe para las 48 selecciones."""
        squads = load_all_squads(teams_df, allow_network=allow_network,
                                 transfermarkt_urls=transfermarkt_urls)
        return cls(strengths=compute_strengths(squads), **kwargs)

    @classmethod
    def from_elo(cls, elo_by_team: dict[str, float], *, scale: float = 400.0,
                 **kwargs) -> "MarketValueMicroSim":
        """Construye la microsim desde ELO (no valor de mercado). Para correrla
        RETROACTIVAMENTE en el backtest, donde no hay valores históricos de
        plantel. Ver `strength.strengths_from_elo` (honestidad/limitación)."""
        return cls(strengths=strengths_from_elo(elo_by_team, scale=scale),
                   **kwargs)

    # ------------------------------- λ desde fuerza --------------------------
    def lambdas(self, team_home: str, team_away: str,
                neutral: bool = True) -> tuple[float, float]:
        """(λ_home, λ_away) a partir de los índices de fuerza (Maher).

        λ_home = base · ataque_home^a / defensa_away^d · (localía si no neutral).
        Si un equipo no tiene fuerza calculada, usa el promedio de liga (1.0)
        para no romper — el ensemble decidirá cuánto confiar."""
        sh = self.strengths.get(team_home)
        sa = self.strengths.get(team_away)
        att_h = sh.attack if sh else 1.0
        def_h = sh.defense if sh else 1.0
        att_a = sa.attack if sa else 1.0
        def_a = sa.defense if sa else 1.0

        home_factor = 1.0 if neutral else self.home_advantage
        lam_h = (self.base_goals * att_h ** ATTACK_ELASTICITY
                 / max(def_a, 1e-3) ** DEFENSE_ELASTICITY * home_factor)
        lam_a = (self.base_goals * att_a ** ATTACK_ELASTICITY
                 / max(def_h, 1e-3) ** DEFENSE_ELASTICITY)
        lo, hi = LAMBDA_CLIP
        return float(np.clip(lam_h, lo, hi)), float(np.clip(lam_a, lo, hi))

    def probs_analytic(self, team_home: str, team_away: str,
                       neutral: bool = True) -> tuple[float, float, float]:
        """1X2 ANALÍTICO (P(1),P(X),P(2)) de la matriz Dixon-Coles sobre las λ
        de fuerza — sin Monte Carlo. Determinista y rápido: lo usa el ensemble
        para puntuar los 448 partidos del backtest sin ruido de muestreo."""
        lam_h, lam_a = self.lambdas(team_home, team_away, neutral=neutral)
        return dc_1x2(lam_h, lam_a, self.rho)

    # ------------------------------- predicción ------------------------------
    def predict_match(self, team_home: str, team_away: str, match_date: str,
                      *, n_sims: int = 10000, neutral: bool = True,
                      rng: np.random.Generator | None = None) -> MatchPrediction:
        """Simula el partido n_sims veces y devuelve un `MatchPrediction`.

        Monte Carlo: muestrea marcadores de la matriz Dixon-Coles con ruido
        lognormal por iteración (mismas condiciones-del-día que `core/`); el 1X2
        sale de la frecuencia muestreada. `score_matrix` es la matriz DC
        analítica (la distribución 'propia' del modelo), igual que `core/adapter`.
        """
        rng = rng or np.random.default_rng(self.seed)
        lam_h, lam_a = self.lambdas(team_home, team_away, neutral=neutral)

        win_h = draw = win_a = 0
        gh_sum = ga_sum = 0
        jitter = np.exp(rng.normal(0.0, self.lambda_jitter, (n_sims, 2)))
        for k in range(n_sims):
            m = _dixon_coles_matrix(lam_h * jitter[k, 0],
                                    lam_a * jitter[k, 1], self.rho)
            gh, ga = _sample_score(m, rng)
            if gh > ga:
                win_h += 1
            elif gh < ga:
                win_a += 1
            else:
                draw += 1
            gh_sum += gh
            ga_sum += ga

        p_home = win_h / n_sims
        p_draw = draw / n_sims
        p_away = win_a / n_sims
        score_matrix = _dixon_coles_matrix(lam_h, lam_a, self.rho)

        # confianza: dispersión del valor de plantel del cruce como proxy de
        # cuánta señal tiene el modelo (planteles muy parejos -> menos seguro).
        sh, sa = self.strengths.get(team_home), self.strengths.get(team_away)
        if sh and sa:
            edge = abs(sh.attack - sa.attack) + abs(sh.defense - sa.defense)
            confidence = float(np.clip(edge / 2.0, 0.0, 1.0))
        else:
            confidence = None

        return MatchPrediction(
            team_home=team_home, team_away=team_away, match_date=str(match_date),
            prob_home=p_home, prob_draw=p_draw, prob_away=p_away,
            lambda_home=lam_h, lambda_away=lam_a,
            model_name=MODEL_NAME, model_version=MODEL_VERSION,
            score_matrix=score_matrix, confidence=confidence)

    def predict_fixtures(self, fixtures: pd.DataFrame, *, n_sims: int = 10000,
                         neutral: bool = True) -> list[MatchPrediction]:
        """Predice todos los partidos de un fixture (cols team_a, team_b, date).
        Reusa un único RNG sembrado para reproducibilidad."""
        rng = np.random.default_rng(self.seed)
        preds: list[MatchPrediction] = []
        for fx in fixtures.itertuples(index=False):
            d = fx._asdict()
            preds.append(self.predict_match(
                str(d["team_a"]), str(d["team_b"]), str(d["date"]),
                n_sims=n_sims, neutral=neutral, rng=rng))
        return preds
