"""E. predict_final_updated: el ensemble de producción pero con ratings 2026.

Idéntico a `ensemble.predict.predict_final`, salvo que el ELO y los pi-ratings
pre-torneo se REEMPLAZAN por los online-updated (A y B). El resto del pipeline
(modelo Poisson de core/, Dixon-Coles, meta-modelo, temperature scaling) es el
mismo y se usa SOLO LECTURA — no se modifica producción.

Tanto el leg de core como el de microsim pasan a usar el ELO/pi actualizado, de
modo que el cambio refleja toda la fuerza re-estimada con los partidos reales.
"""

from __future__ import annotations

import copy

import pandas as pd

from core.data.wc_schema import build_match_features, match_features_frame
from core.simulation.monte_carlo import (_dixon_coles_matrix, dc_1x2,
                                         lambdas_from_1x2)
from online_learning.elo_online import get_elo_2026
from online_learning.pi_online import get_pi_2026
from online_learning.priors import canon, teams_2026, to_es
from schema import MatchPrediction

MODEL_NAME = "ensemble_stacking_online"
MODEL_VERSION = "1.0.0-online"


def _series_override(ctx, en: str, is_host: float):
    """Serie de features de core/ con elo/att/dfn reemplazados por los 2026."""
    from core.data.historical import _neutral_series
    snap = ctx.snapshots[en]
    proxy = copy.copy(snap)                 # _TeamHistory (__slots__) -> copia
    proxy.elo = get_elo_2026(en)
    proxy.att, proxy.dfn = get_pi_2026(en)
    return _neutral_series(en, proxy, is_host=is_host)


def core_lambdas_updated(ctx, home: str, away: str) -> tuple[float, float]:
    """λ (goles esperados) de core con ratings 2026, para construir la matriz."""
    en_h, en_a = canon(home), canon(away)
    host = float(ctx.is_host.get(to_es(en_h), 0.0))
    sa = _series_override(ctx, en_h, host)
    sb = _series_override(ctx, en_a, 0.0)
    lam = ctx.poisson.predict_lambda(match_features_frame(
        [build_match_features(sa, sb, 1, 0.0),
         build_match_features(sb, sa, 1, 0.0)]))
    return float(lam[0]), float(lam[1])


def _core_probs_updated(ctx, home: str, away: str):
    lam_h, lam_a = core_lambdas_updated(ctx, home, away)
    return dc_1x2(lam_h, lam_a)


def micro_updated(ctx):
    """Accesor público del microsim reconstruido con ELO 2026."""
    return _micro_updated(ctx)


_MICRO_CACHE = {"key": None, "micro": None}


def _micro_updated(ctx):
    """microsim reconstruido desde el ELO 2026 (keyed por nombre español, como
    en producción). Cacheado mientras el ELO 2026 no cambie."""
    from microsim.model import MarketValueMicroSim
    key = tuple(round(get_elo_2026(en), 2) for en in teams_2026())
    if _MICRO_CACHE["key"] != key:
        elo_by = {to_es(en): get_elo_2026(en) for en in teams_2026()}
        _MICRO_CACHE["micro"] = MarketValueMicroSim.from_elo(elo_by)
        _MICRO_CACHE["key"] = key
    return _MICRO_CACHE["micro"]


def updated_1x2(home: str, away: str):
    """(core_updated, micro_updated, ensemble_updated) en 1X2 [1,X,2]."""
    from ensemble.features import build_feature_row
    from ensemble.predict import (TEMP_PATH, _context, _get_meta)
    ctx = _context()
    core_p = _core_probs_updated(ctx, home, away)
    micro = _micro_updated(ctx)
    micro_p = micro.probs_analytic(to_es(canon(home)), to_es(canon(away)))
    meta = _get_meta(None)
    row = build_feature_row(core_p, micro_p, None)
    probs = meta.predict_proba_1x2(pd.DataFrame([row]))[0]
    if TEMP_PATH.exists():
        from ensemble.calibrate import TemperatureScaler
        probs = TemperatureScaler.load(TEMP_PATH).transform(probs.reshape(1, -1))[0]
    return core_p, micro_p, tuple(float(x) for x in probs)


def predict_final_updated(team_home: str, team_away: str, match_date: str
                          ) -> MatchPrediction:
    """Predicción final con ratings 2026 online (drop-in de predict_final)."""
    _, _, probs = updated_1x2(team_home, team_away)
    p1, pX, p2 = (float(v) for v in probs)
    lam_h, lam_a = lambdas_from_1x2(p1, pX, p2)
    score_matrix = _dixon_coles_matrix(lam_h, lam_a)
    return MatchPrediction(
        team_home=team_home, team_away=team_away, match_date=str(match_date),
        prob_home=p1, prob_draw=pX, prob_away=p2,
        lambda_home=lam_h, lambda_away=lam_a,
        model_name=MODEL_NAME, model_version=MODEL_VERSION,
        score_matrix=score_matrix)
