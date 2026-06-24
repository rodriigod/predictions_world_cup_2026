"""Agregación de fuerza de plantel: valores de mercado -> índices ataque/defensa.

A partir del `Squad` (lista de jugadores con valor y posición) que entrega
`ingest.py`, construye dos índices por selección:

  - `attack`  : fuerza OFENSIVA (capacidad de generar gol)
  - `defense` : fuerza DEFENSIVA (capacidad de evitar gol); MAYOR = mejor defensa

La clave del diseño (nota #2 del encargo): no todo el valor del plantel pesa
igual para cada faceta. El valor de un delantero estrella alimenta sobre todo
el ataque; el de un central, sobre todo la defensa. Por eso cada grupo de
posición contribuye con un PESO distinto a cada índice (OFF_WEIGHTS / DEF_WEIGHTS).

Saturación: el dinero se convierte en calidad de forma CÓNCAVA (rendimientos
decrecientes) — duplicar el valor no duplica los goles. Aplicamos `sqrt` al
agregado por equipo, coherente con que el resto del repo trata el valor de
mercado en escala log (`market_value_log_diff` en `core/data/wc_schema.py`).

Normalización: ambos índices se escalan para que la MEDIA de las selecciones
sea 1.0. Así "1.0" = equipo promedio del torneo, y `model.py` puede mapear los
índices a λ con una tasa base de goles realista.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microsim.ingest import POSITION_GROUPS, Squad

# Cuánto contribuye el VALOR de cada grupo de posición a cada índice.
# (No suman 1: son pesos relativos; la normalización global los reescala.)
OFF_WEIGHTS = {"GK": 0.00, "DEF": 0.10, "MID": 0.55, "ATT": 1.00}
DEF_WEIGHTS = {"GK": 1.00, "DEF": 1.00, "MID": 0.45, "ATT": 0.05}


@dataclass
class TeamStrength:
    """Índices normalizados (media de la liga = 1.0) de una selección."""
    team: str
    attack: float
    defense: float
    source: str        # procedencia de los datos del plantel (ver ingest.Squad)


def _weighted_raw(squad: Squad, weights: dict[str, float]) -> float:
    """Agregado crudo: sum(valor_jugador * peso_de_su_posición), con saturación
    cóncava (sqrt) para reflejar rendimientos decrecientes del valor."""
    total = 0.0
    by_group = squad.value_by_group()
    for group in POSITION_GROUPS:
        total += by_group[group] * weights[group]
    return float(np.sqrt(max(total, 0.0)))


def strengths_from_elo(elo_by_team: dict[str, float], *, scale: float = 400.0
                       ) -> dict[str, TeamStrength]:
    """Construye índices de fuerza desde el ELO (NO desde valor de mercado).

    Uso: correr la microsim RETROACTIVAMENTE sobre Mundiales pasados, donde NO
    existe valor de mercado histórico de plantel — el único proxy de fuerza
    leak-free disponible en el snapshot pre-torneo es el ELO. Mapea
    s = exp((elo - media)/scale) y normaliza a media 1.0; ataque = defensa = s
    (un equipo más fuerte ataca más y concede menos por igual).

    HONESTO: así la microsim queda alimentada por ELO, así que se comporta como
    un modelo Dixon-Coles de ELO — CORRELACIONADO con `core/` (que también usa
    ELO). El meta-modelo del ensemble lo revelará: si su coeficiente ≈ 0, no
    aporta señal sobre `core/`. Es la consecuencia inevitable de no tener
    valores de mercado históricos (ver microsim/model.py y README)."""
    if not elo_by_team:
        return {}
    mean = float(np.mean(list(elo_by_team.values())))
    raw = {t: float(np.exp((e - mean) / scale)) for t, e in elo_by_team.items()}
    mraw = float(np.mean(list(raw.values()))) or 1.0
    return {t: TeamStrength(t, attack=v / mraw, defense=v / mraw, source="elo")
            for t, v in raw.items()}


def compute_strengths(squads: dict[str, Squad]) -> dict[str, TeamStrength]:
    """Convierte {equipo: Squad} en {equipo: TeamStrength} normalizado.

    Cada índice se divide por la media de todos los equipos, de modo que 1.0 =
    promedio del torneo. Robusto a un solo equipo (media = su propio valor).
    """
    if not squads:
        return {}
    raw_att = {t: _weighted_raw(s, OFF_WEIGHTS) for t, s in squads.items()}
    raw_def = {t: _weighted_raw(s, DEF_WEIGHTS) for t, s in squads.items()}

    mean_att = float(np.mean(list(raw_att.values()))) or 1.0
    mean_def = float(np.mean(list(raw_def.values()))) or 1.0

    out: dict[str, TeamStrength] = {}
    for t, s in squads.items():
        out[t] = TeamStrength(
            team=t,
            attack=raw_att[t] / mean_att,
            defense=raw_def[t] / mean_def,
            source=s.source,
        )
    return out
