"""synth: sintetizador LLM ACOTADO (experimental, separado de producción).

Recibe la predicción de core (o su versión calibrada si hubiera ganado en A), la
de microsim y el JSON de señales factuales YA validado por `ensemble/roster.py`,
y pide a un LLM local (LM Studio) una decisión ESTRUCTURADA y de rango limitado:
nunca una probabilidad libre, solo una de cuatro acciones y una magnitud acotada
a [-0.15, 0.15] con límites DUROS en código (no solo en el prompt).

No se puede backtestear con datos históricos (no hay contexto factual leak-free
de partidos pasados): su única validación es el log en vivo (sección G), donde se
guarda la predicción ajustada junto a la de core SIN ajustar para comparar ambas
contra el resultado real.
"""

from synth.synthesize import (ACTIONS, MAGNITUDE_HARD_CAP, SynthDecision,
                              apply_adjustment, clamp_decision, synthesize)

__all__ = ["synthesize", "apply_adjustment", "clamp_decision", "SynthDecision",
           "ACTIONS", "MAGNITUDE_HARD_CAP"]
