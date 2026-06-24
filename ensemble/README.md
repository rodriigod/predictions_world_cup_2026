# ensemble/ — stacking de core + microsim + llm_features

Combina los tres modelos en una predicción final. **Stacking**, no promedio:
las **probabilidades 1X2** de `core/` y `microsim/` se apilan y un **meta-modelo
logístico multinomial (L2)** aprende cuánto pesar cada uno; las **señales del
LLM** entran como **columnas extra** al mismo meta-modelo (no se promedian con
nada). Antes, las extracciones del LLM pasan por **validación de nómina**.

## Pipeline

```
roster.validate_features   # descarta nombres que no son de esa selección + log
features.build_feature_row # arma la fila: probs core/micro + señales LLM
dataset.build_backtest_dataset  # 448 partidos: core real + microsim retroactivo
meta_model.StackingMetaModel    # logística L2, C por TimeSeriesSplit (RPS)
predict.predict_final      # core+micro+llm -> meta-modelo -> MatchPrediction
```

```python
from ensemble import predict_final, train_meta_model
train_meta_model()                                   # entrena sobre el backtest
pred = predict_final("Brasil", "Argentina", "2026-06-20")   # MatchPrediction
```

Reproducir la comparación: `python scripts/ensemble_backtest.py`
(→ `results/reports/ensemble_backtest.md`).

## Resultado del backtest (7 Mundiales, 384 partidos OOF)

| Enfoque | Accuracy | RPS | Log-loss |
|---|:-:|:-:|:-:|
| **solo core** | **0.547** | **0.1994** | **0.971** |
| solo microsim | 0.536 | 0.2054 | 0.994 |
| promedio core+micro | 0.536 | 0.2005 | 0.974 |
| stacking (OOF) | 0.521 | 0.2047 | 0.998 |

**El stacking NO le gana a `core/` solo** — 4ª confirmación del mismo patrón en
este repo: la señal del fútbol internacional es casi lineal en la fuerza de
equipo (ELO), así que apilar modelos correlacionados agrega varianza, no skill.
Coeficientes aprendidos (importancia agregada \|peso\|):
**core 0.43 · microsim 0.08 · llm 0.00**.

## Honestidad / limitaciones

- **microsim retroactivo** se alimenta de **ELO** (no hay valores de mercado
  históricos de plantel), así que es un Dixon-Coles de ELO **correlacionado con
  core** → su coeficiente sale pequeño. En inferencia 2026 también se usa ELO,
  por coherencia con el entrenamiento (el microsim de valor de mercado queda
  como herramienta standalone).
- **señales del LLM**: no hay forma leak-free de reconstruir lesiones/consenso
  de 1998-2022, así que entran **neutras** en el backtest → coeficiente **0 por
  falta de señal de entrenamiento**, NO por medirse inútiles. Quedan cableadas
  para 2026 (datos reales), pero **el meta-modelo no puede ponderarlas** hasta
  tener partidos etiquetados con señales LLM reales (reentrenar sobre partidos
  2026 ya jugados con sus features LLM).
- **validación de nómina** (`roster.py`): cruza cada nombre extraído contra la
  nómina real (caché Transfermarkt → FIFA-24 por país). Descarta los que no
  pertenecen (caso 'Carvajal'→Argentina) y los registra en
  `results/ensemble/discarded_extractions.csv`. Caveat: FIFA-24 es incompleto,
  así que el log distingue `cross_contamination` (alta confianza) de
  `not_in_roster` (puede ser hueco de datos); equipos sin nómina conocida quedan
  `unverified` (no se descarta a ciegas).

## Contrato

`predict_final(home, away, date)` devuelve un [`MatchPrediction`](../schema.py)
(`model_name='ensemble_stacking'`) con λ y `score_matrix` re-derivadas del 1X2
final, para que el resto del repo lo consuma como cualquier otro modelo.
