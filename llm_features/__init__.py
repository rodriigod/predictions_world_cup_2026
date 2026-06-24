"""llm_features: señales ESTRUCTURADAS por partido extraídas con un LLM + web.

NO predice resultados: extrae hechos objetivos y verificables (lesionados
clave, cambio de DT reciente, consenso de expertos publicado, dead rubber,
fatiga de viaje) buscándolos en la web, para alimentar al `ensemble/` como
columnas extra. Lo que no se encuentra queda en None (no se inventa).

Uso rápido:
    from llm_features import get_match_features
    feats = get_match_features("Brasil", "Argentina", "2026-06-20")
    # -> dict anidado (schema.MatchFeatures.to_dict())

Por defecto usa Gemini con google_search grounding (GEMINI_API_KEY) y cachea
agresivamente en disco. Sin clave / sin red devuelve un esqueleto con todo None.
"""

from llm_features.extract import (build_prompt, ddg_match_search,
                                  default_provider, gemini_provider,
                                  get_features_for_fixtures, get_match_features,
                                  lmstudio_provider, parse_features)
from llm_features.schema import MatchFeatures, TeamSignals

__all__ = [
    "get_match_features", "get_features_for_fixtures", "parse_features",
    "build_prompt", "default_provider", "gemini_provider", "lmstudio_provider",
    "ddg_match_search", "MatchFeatures", "TeamSignals",
]
