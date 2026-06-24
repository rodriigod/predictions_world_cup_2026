"""A. Update ELO online con los resultados reales del Mundial 2026.

Arranca de los ELO pre-torneo de core/ (NO de cero) y aplica el update estándar
de ELO partido a partido, en orden cronológico:

    E_a = 1 / (1 + 10^(-(elo_a - elo_b + ventaja_local) / 400))
    elo_a += K · (S_a − E_a)         (S_a = 1 gana / 0.5 empata / 0 pierde)

K-factor: core/ usa K=60 para Mundial dentro de un esquema con multiplicador de
goleada y ventaja local 100 (su ELO de entrenamiento). Para un actualizador
ONLINE de un solo torneo eso es agresivo, así que por defecto se usa el esquema
conservador que pide el encargo: **K=30 grupos, K=40 eliminatorias** (configurable).
La ventaja local es 0 por defecto (sede neutral, lo normal en un Mundial salvo
anfitriones); ajustable.

`get_elo_2026(team)` es el reemplazo drop-in del ELO pre-torneo.
"""

from __future__ import annotations

import numpy as np

from online_learning.dataset import KNOCKOUT, load_results
from online_learning.priors import canon, pretournament_priors

K_GROUP = 30.0
K_KNOCKOUT = 40.0
HOME_ADV = 0.0          # sede neutral por defecto


def _k(stage: str, k_group: float, k_ko: float) -> float:
    return k_ko if stage in KNOCKOUT else k_group


def compute_elo(*, k_group: float = K_GROUP, k_ko: float = K_KNOCKOUT,
                home_adv: float = HOME_ADV, with_trace: bool = False):
    """Recorre los resultados 2026 y devuelve {equipo_en: elo}. Si `with_trace`,
    devuelve también la lista de updates (para reportes antes/después)."""
    priors = pretournament_priors()
    elo = {en: p["elo"] for en, p in priors.items()}
    trace = []
    for r in load_results().itertuples():
        a, b = r.home_team, r.away_team
        if a not in elo or b not in elo:
            continue
        ea, eb = elo[a], elo[b]
        we = 1.0 / (1.0 + 10 ** (-(ea + home_adv - eb) / 400.0))
        gh, ga = int(r.home_goals), int(r.away_goals)
        sa = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        k = _k(r.stage, k_group, k_ko)
        delta = k * (sa - we)
        elo[a] = ea + delta
        elo[b] = eb - delta
        if with_trace:
            trace.append({"date": r.date, "home": a, "away": b,
                          "gh": gh, "ga": ga, "we_home": we,
                          "delta_home": delta, "elo_home_after": elo[a],
                          "elo_away_after": elo[b]})
    return (elo, trace) if with_trace else elo


# --------- caché invalidada por mtime del CSV de resultados -----------------
_CACHE = {"key": None, "elo": None}


def _state():
    from online_learning.dataset import RESULTS_CSV
    key = RESULTS_CSV.stat().st_mtime if RESULTS_CSV.exists() else 0
    if _CACHE["key"] != key:
        _CACHE["elo"] = compute_elo()
        _CACHE["key"] = key
    return _CACHE["elo"]


def get_elo_2026(team: str) -> float:
    """ELO actualizado con los partidos 2026 jugados hasta ahora (drop-in del ELO
    pre-torneo). Acepta nombre español o inglés."""
    return float(_state()[canon(team)])


def elo_movements() -> "list[dict]":
    """Movimiento de cada equipo: elo_pre, elo_now, delta. Ordenado por |delta|."""
    priors = pretournament_priors()
    now = _state()
    rows = [{"team_en": en, "elo_pre": p["elo"], "elo_now": now[en],
             "delta": now[en] - p["elo"]} for en, p in priors.items()]
    rows.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return rows
