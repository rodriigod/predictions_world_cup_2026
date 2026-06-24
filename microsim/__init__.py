"""microsim: modelo de microsimulación por FUERZA DE PLANTEL.

Predice partidos a partir del VALOR DE MERCADO del plantel (proxy de calidad),
separado en fuerza ofensiva/defensiva ponderando por posición, y simulado vía
Monte Carlo con la matriz Dixon-Coles de `core/`. La salida cumple
`schema.MatchPrediction`, igual que `core/`, para entrar al `ensemble/`.

NO es microsimulación jugador-a-jugador tipo videojuego: usa valor de mercado
como proxy y, por defecto, planteles sintéticos derivados del total del equipo.
Ver la limitación detallada en `microsim/model.py` y el README.

Pipeline:
  ingest.load_all_squads  -> {equipo: Squad}        (valores de mercado)
  strength.compute_strengths -> {equipo: TeamStrength}  (ataque/defensa)
  model.MarketValueMicroSim  -> list[MatchPrediction]   (simulación MC)

Uso rápido:
    import pandas as pd
    from microsim import MarketValueMicroSim
    teams = pd.read_csv("files/f0_raw/teams_2026.csv")
    sim = MarketValueMicroSim.from_teams(teams)        # offline por defecto
    pred = sim.predict_match("Brasil", "Argentina", "2026-06-20")
"""

from microsim.ingest import (PlayerValue, RespectfulFetcher, Squad,
                             load_all_squads, load_squad)
from microsim.model import (MODEL_NAME, MODEL_VERSION, MarketValueMicroSim)
from microsim.strength import TeamStrength, compute_strengths

__all__ = [
    "PlayerValue", "Squad", "RespectfulFetcher", "load_squad", "load_all_squads",
    "TeamStrength", "compute_strengths",
    "MarketValueMicroSim", "MODEL_NAME", "MODEL_VERSION",
]
