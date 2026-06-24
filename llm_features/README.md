# llm_features/ — señales ESTRUCTURADAS de LLM + búsqueda web

Extrae **hechos objetivos y verificables** por partido vía un LLM con
**búsqueda web real**, para alimentar al [`ensemble/`](../ensemble/) como
columnas extra. **El LLM NO predice el resultado**: solo rellena un schema de
señales. Lo que no se confirma queda en `None` (nunca se inventa).

## Features extraídas (por partido)

| Campo | Qué es |
|---|---|
| `lesionados_clave` | titulares/figuras confirmados FUERA por lesión, por equipo |
| `cambio_dt_reciente` | ¿cambio de entrenador en los últimos ~3 meses? |
| `consenso_expertos_pct` | % de analistas/medios reconocidos que favorecen a c/equipo (de predicciones PUBLICADAS) |
| `dead_rubber` | ¿partido intrascendente (un equipo ya clasificado/eliminado)? |
| `fatiga_viaje_km` / `fatiga_husos_horarios` | viaje desde el último partido |

```python
from llm_features import get_match_features
feats = get_match_features("Brasil", "Argentina", "2026-06-20")  # dict; cacheado
```

## Diseño

- **Schema** (`schema.py`): `MatchFeatures` con todos los campos *nullable* +
  `to_dict()` / `to_flat_dict()` (columnas planas para el ensemble).
- **Caché** (`cache.py`): agresivo, por `(home, away, date)` en
  `files/cache/llm_features/`. No repite búsquedas entre corridas. Un resultado
  null (fallo transitorio) **no** se cachea.
- **Extracción** (`extract.py`): proveedor y buscador pluggables.
  - **Default = LM Studio (Qwen local) + DuckDuckGo.** El LLM local NO tiene
    internet, así que NO se confía en su memoria: `ddg_match_search` trae
    snippets REALES de la web (sin API key) y Qwen SOLO los lee para rellenar
    el JSON. Autodetecta el modelo cargado en LM Studio (`LMSTUDIO_URL`,
    `LMSTUDIO_MODEL`). Mismo patrón que `scripts/match_dossier.py`.
  - **Alternativa = Gemini grounded** (`google_search`, `GEMINI_API_KEY`): hace
    la búsqueda web por su cuenta. Pásale `searcher=None` para no inyectar DDG.
  - JSON forzado (`response_format` json_object) + parseo robusto + clamps.

## Limitaciones (honestidad)

- **Sin clave / sin red** → esqueleto con todo en `None` (degradación elegante).
- El LLM puede malinterpretar una fuente: estos campos son **inputs a verificar**,
  no verdad de terreno; `consenso_expertos` vale lo que valgan las predicciones
  publicadas que encuentre.
- Wired para el ensemble pero **aún sin validar** que mejoren alguna métrica.
