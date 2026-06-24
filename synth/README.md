# synth — sintetizador LLM acotado (F, experimental)

Toma la predicción de **core**, la de **microsim** y el **JSON de señales
factuales ya validado** por `ensemble/roster.py`, y pide a un LLM local
(LM Studio) una decisión ESTRUCTURADA y de rango limitado. Nunca una probabilidad
libre.

## Contrato de salida (JSON del LLM)

```json
{"accion": "sin_cambio|ajuste_leve|ajuste_fuerte|marcar_revision",
 "magnitud": -0.15 .. 0.15,
 "justificacion": "cita la señal exacta que motivó la decisión"}
```

## Límites DUROS (en código, no solo en el prompt)

`clamp_decision` recorta SIEMPRE, pase lo que pase el LLM:

| acción | |magnitud| máx |
|---|:-:|
| `sin_cambio` | 0 (forzado) |
| `ajuste_leve` | 0.05 |
| `ajuste_fuerte` | 0.15 |
| `marcar_revision` | 0 (forzado) |

Tope global absoluto: ±0.15. Acción inválida → `marcar_revision`. Magnitud no
numérica/NaN → 0. Verificado en `tests/test_synth.py` (incluye un LLM "malicioso"
que pide magnitud 5.0 → recortado a 0.15).

El ajuste se aplica a la probabilidad del **resultado más probable** y se
redistribuye proporcionalmente en los otros dos (`apply_adjustment`),
preservando una distribución válida.

## Validación

NO se puede backtestear (no hay contexto factual leak-free de partidos pasados).
Su única validación es el **log en vivo** (`scripts/log_friendlies.py`), que
guarda la predicción ajustada (`p_*_adj`, `pred_result_adj`, `points_5_3_0_adj`)
junto a la sin ajustar, para compararlas contra el resultado real con el tiempo.

> Nota honesta: con qwen2.5-7b local, el LLM a veces acierta la señal pero
> equivoca el SIGNO del ajuste. Por eso es experimental y se mide en vivo antes
> de confiar en él; los límites duros acotan el daño de cualquier error.

## Uso

```python
from synth import synthesize, apply_adjustment
d = synthesize(core_probs, micro_probs, señales_validadas)   # LM Studio por defecto
probs_ajustadas = apply_adjustment(prediccion_final_1x2, d)
```
