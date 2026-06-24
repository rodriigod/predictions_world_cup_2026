# microsim/ — microsimulación por FUERZA DE PLANTEL

Modelo de predicción **independiente** que estima λ (goles esperados) y el 1X2
a partir del **valor de mercado del plantel** como proxy de calidad, separado
en fuerza **ofensiva/defensiva** ponderando por posición, y simulado vía
**Monte Carlo** con la matriz Dixon-Coles de `core/`. No toca el pipeline de
producción.

## Pipeline

```
ingest.load_all_squads      -> {equipo: Squad}         # valores de mercado
strength.compute_strengths  -> {equipo: TeamStrength}  # ataque/defensa (media liga=1)
model.MarketValueMicroSim   -> list[MatchPrediction]   # simulación Monte Carlo
```

```python
import pandas as pd
from microsim import MarketValueMicroSim
teams = pd.read_csv("files/f0_raw/teams_2026.csv")
sim = MarketValueMicroSim.from_teams(teams)              # offline por defecto
pred = sim.predict_match("Brasil", "Argentina", "2026-06-20")
```

## Contrato

`predict_match` / `predict_fixtures` devuelven [`MatchPrediction`](../schema.py),
igual que el resto del repo, para entrar al [`ensemble/`](../ensemble/).

## Limitaciones (honestidad)

- **NO es microsimulación jugador-a-jugador** tipo videojuego: usa valor de
  mercado como proxy, no simula eventos/posiciones.
- **Transfermarkt no es scrapeable de forma confiable** (403, HTML cambiante):
  `ingest.py` trae un scraper *respetuoso* (UA real, rate-limit, caché, back-off)
  y un stub de **FotMob**, pero el camino por defecto es el **fallback offline**:
  el total `market_value_meur` (que viene de Transfermarkt; un número por equipo)
  repartido por una **plantilla canónica** → planteles **sintéticos**, no listas
  reales por jugador. Con solo el total, ataque y defensa de un equipo se mueven
  juntos; el desglose por posición solo "muerde" si das datos reales por jugador.
- El valor de mercado **infravalora ligas menores** y **reacciona con retraso** a
  forma/lesiones. Úsalo como segunda opinión barata para el ensemble.

Detalle completo en el docstring de `model.py` y en el README raíz.
