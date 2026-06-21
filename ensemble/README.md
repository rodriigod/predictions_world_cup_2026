# ensemble/ — combinación de modelos (vacío por ahora)

Punto de unión del repo. Toma las predicciones de cada modelo —todas en el
idioma común [`MatchPrediction`](../schema.py)— y produce la predicción final
combinada.

Como todos los modelos casan por `MatchPrediction.match_key`
(`match_date, team_home, team_away`), el ensemble combina sin saber cómo se
generó cada predicción. Estrategias previstas:

- Mezcla de probabilidades 1X2 con pesos (lineal o en log-odds), opcionalmente
  usando `confidence` por modelo.
- Mezcla a nivel de `score_matrix` cuando todos los modelos la aportan.
- Calibración / blend con mercado (ya existe en `core/`: `blend_with_market`).

## Contrato

Entrada: una o varias `list[MatchPrediction]` (una por modelo).
Salida: idealmente otra `list[MatchPrediction]` con `model_name="ensemble"`,
para que el resto del pipeline (reportes, polla) la consuma igual que a un
modelo individual.

## Estado

Vacío. Solo `__init__.py` + este README.
