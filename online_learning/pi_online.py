"""B. Update online de pi-ratings (ataque/defensa) con resultados 2026.

Mismo update online que core/ (Constantinou & Fenton 2013, idénticas constantes
PI_MU/PI_HOME/PI_LR/PI_CLIP), pero arrancando de los pi-ratings PRE-TORNEO de
core/ y alimentándolo con los goles reales 2026:

    pred_a = exp(PI_MU + att_a − dfn_b [+ PI_HOME si local])
    att_a += PI_LR · (gf − pred_a)        ;  dfn_b −= PI_LR · (gf − pred_a)
    (y simétrico para el equipo b)

Con PI_LR=0.06 y 1 partido jugado por equipo, el movimiento es MODERADO a propósito:
1 partido no es señal suficiente para mover mucho un rating. `pi_movements()`
documenta cuánto se movió cada equipo.
"""

from __future__ import annotations

import numpy as np

from core.data.historical import PI_CLIP, PI_HOME, PI_LR, PI_MU
from online_learning.dataset import load_results
from online_learning.priors import canon, pretournament_priors


def compute_pi(*, home_adv: bool = False):
    """Recorre los resultados 2026 y devuelve {equipo_en: (att, dfn)}.
    `home_adv`: aplica PI_HOME al local (False por defecto: sede neutral)."""
    priors = pretournament_priors()
    att = {en: p["att"] for en, p in priors.items()}
    dfn = {en: p["dfn"] for en, p in priors.items()}
    for r in load_results().itertuples():
        a, b = r.home_team, r.away_team
        if a not in att or b not in att:
            continue
        gh, ga = int(r.home_goals), int(r.away_goals)
        ph = PI_HOME if home_adv else 0.0
        pred_a = min(np.exp(PI_MU + att[a] - dfn[b] + ph), 6.0)
        pred_b = min(np.exp(PI_MU + att[b] - dfn[a]), 6.0)
        err_a, err_b = gh - pred_a, ga - pred_b
        att[a] = float(np.clip(att[a] + PI_LR * err_a, -PI_CLIP, PI_CLIP))
        dfn[b] = float(np.clip(dfn[b] - PI_LR * err_a, -PI_CLIP, PI_CLIP))
        att[b] = float(np.clip(att[b] + PI_LR * err_b, -PI_CLIP, PI_CLIP))
        dfn[a] = float(np.clip(dfn[a] - PI_LR * err_b, -PI_CLIP, PI_CLIP))
    return att, dfn


_CACHE = {"key": None, "att": None, "dfn": None}


def _state():
    from online_learning.dataset import RESULTS_CSV
    key = RESULTS_CSV.stat().st_mtime if RESULTS_CSV.exists() else 0
    if _CACHE["key"] != key:
        _CACHE["att"], _CACHE["dfn"] = compute_pi()
        _CACHE["key"] = key
    return _CACHE["att"], _CACHE["dfn"]


def get_pi_2026(team: str) -> tuple[float, float]:
    """(att, dfn) pi-ratings actualizados con 2026. Drop-in de los pre-torneo."""
    att, dfn = _state()
    en = canon(team)
    return float(att[en]), float(dfn[en])


def pi_movements() -> "list[dict]":
    """Movimiento de att/dfn por equipo (pre vs ahora). Ordenado por |Δatt|+|Δdfn|."""
    priors = pretournament_priors()
    att, dfn = _state()
    rows = []
    for en, p in priors.items():
        d_att = att[en] - p["att"]
        d_dfn = dfn[en] - p["dfn"]
        rows.append({"team_en": en, "att_pre": p["att"], "att_now": att[en],
                     "d_att": d_att, "dfn_pre": p["dfn"], "dfn_now": dfn[en],
                     "d_dfn": d_dfn, "move": abs(d_att) + abs(d_dfn)})
    rows.sort(key=lambda d: d["move"], reverse=True)
    return rows
