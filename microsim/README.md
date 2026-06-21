# microsim/ — modelo de microsimulación (vacío por ahora)

Modelo de predicción **independiente** que estimará las lambdas (goles
esperados) y el 1X2 simulando el partido a nivel de **plantilla**: jugadores,
posiciones y stats (npxG/90, PSxG, etc.). Hoy existe material relacionado en
`core/simulation/match_engine.py` y en `scripts/microsim_groupstage.py`; este
paquete será su hogar definitivo como modelo de primera clase.

## Contrato

Como todo modelo del repo, su punto de entrada debe devolver una lista de
[`MatchPrediction`](../schema.py) (mismas `team_home/team_away/match_date`,
`prob_home/draw/away`, `lambda_home/away`, `model_name`, `model_version`).
Así el [`ensemble/`](../ensemble/) lo combina con `core/` sin acoplarse a su
implementación.

## Estado

Vacío. Solo `__init__.py` + este README. No romper `core/`: este modelo se
desarrolla en paralelo y nunca modifica el pipeline en producción.
