"""online_learning: ratings "2026-ajustados" en PARALELO a producción.

Actualiza la fuerza de cada selección con los resultados reales del Mundial 2026
a medida que se juegan, SIN tocar core/, microsim/, llm_features/ ni ensemble/.
Produce ratings alternativos (ELO online, pi-ratings online, fuerza Bayesiana)
usables como reemplazo drop-in de los ratings pre-torneo de core/.

Autocontenido: nada de producción importa este paquete; borrar `online_learning/`
deja el resto del repo intacto.

API principal:
    get_elo_2026(team)                 # A
    get_pi_2026(team) -> (att, dfn)    # B
    get_strength_posterior(team)       # C  (Gamma-Poisson, media + IC80%)
    build_surprise_log()               # D
    predict_final_updated(h, a, date)  # E
"""

from online_learning.bayes_strength import (compute_posteriors,
                                            get_strength_posterior)
from online_learning.elo_online import elo_movements, get_elo_2026
from online_learning.pi_online import get_pi_2026, pi_movements
from online_learning.predict_updated import predict_final_updated, updated_1x2
from online_learning.surprise import (build_surprise_log, detect_favorite_bias)

__all__ = [
    "get_elo_2026", "elo_movements",
    "get_pi_2026", "pi_movements",
    "get_strength_posterior", "compute_posteriors",
    "build_surprise_log", "detect_favorite_bias",
    "predict_final_updated", "updated_1x2",
]
