# llm_features/ — señales derivadas de LLM (vacío por ahora)

Modelo/proveedor de señales generadas con un **LLM**: contexto que el modelo
estadístico no ve (forma narrativa, lesiones y bajas, moral, declaraciones,
clima mediático). Dos posibles roles:

1. **Generador de features** que alimente a otros modelos.
2. **Modelo de predicción** por derecho propio, devolviendo 1X2 + lambdas.

## Contrato

Si actúa como modelo de predicción, debe devolver
[`MatchPrediction`](../schema.py) como cualquier otro modelo del repo, para
entrar al [`ensemble/`](../ensemble/). Usar `confidence` para reflejar la
incertidumbre del LLM es especialmente recomendable aquí.

## Estado

Vacío. Solo `__init__.py` + este README.
