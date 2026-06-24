"""Predicción final del ensemble: predict_final(home, away, date).

Orquesta los tres modelos y el meta-modelo:
  1. core/     -> probs 1X2 (Poisson+Dixon-Coles, snapshot pre-torneo leak-free)
  2. microsim/ -> probs 1X2 (fuerza de plantel; en 2026 desde ELO actual, para
     ser COHERENTE con cómo se entrenó el meta-modelo — ver dataset.py)
  3. llm_features/ -> señales estructuradas, con VALIDACIÓN DE NÓMINA (roster.py)
  4. meta-modelo (stacking) -> probs 1X2 finales

Devuelve un `schema.MatchPrediction` (model_name='ensemble_stacking'), con λ y
score_matrix re-derivadas del 1X2 final (vía `lambdas_from_1x2`) para que el
resto del repo lo consuma igual que cualquier otro modelo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.data.historical import (NAME_MAP, _neutral_series,
                                  build_historical_dataset)
from core.data.wc_schema import build_match_features, match_features_frame
from core.models.poisson_goals import PoissonGoalsModel
from core.simulation.monte_carlo import (_dixon_coles_matrix, dc_1x2,
                                         lambdas_from_1x2)
from ensemble.features import build_feature_row
from ensemble.meta_model import StackingMetaModel
from microsim.model import MarketValueMicroSim
from schema import MatchPrediction

ROOT = Path(__file__).resolve().parents[1]
TEAMS_CSV = ROOT / "files/f0_raw/teams_2026.csv"
META_PATH = ROOT / "models/ensemble_meta.pkl"
TEMP_PATH = ROOT / "models/ensemble_temperature.json"   # calibración (opcional)
DEFAULT_CUTOFF = "2026-06-11"          # foto pre-torneo (leak-free)

MODEL_NAME = "ensemble_stacking"
MODEL_VERSION = "1.0.0"


# ----------------------------- contexto (lazy) ------------------------------
class _Context:
    """core entrenado + snapshots + microsim(ELO) + tabla 2026, cacheado."""

    def __init__(self, cutoff: str = DEFAULT_CUTOFF):
        self.teams = pd.read_csv(TEAMS_CSV)
        data = build_historical_dataset(cutoff=cutoff)
        self.snapshots = data["snapshots"]
        self.poisson = PoissonGoalsModel(backend="poisson")
        self.poisson.fit(data["X"], data["y"], sample_weight=data["w"])
        self.is_host = dict(zip(self.teams["team"], self.teams["is_host"]))
        elo_by_team = {}
        for t in self.teams["team"]:
            en = NAME_MAP.get(t, t)
            if en in self.snapshots:
                elo_by_team[t] = float(self.snapshots[en].elo)
        self.micro = MarketValueMicroSim.from_elo(elo_by_team)

    def _series(self, team_es: str, is_host: float):
        en = NAME_MAP.get(team_es, team_es)
        if en not in self.snapshots:
            raise ValueError(f"'{team_es}' ({en}) sin snapshot histórico")
        return _neutral_series(en, self.snapshots[en], is_host=is_host)

    def core_probs(self, home: str, away: str) -> tuple[float, float, float]:
        sa = self._series(home, float(self.is_host.get(home, 0.0)))
        sb = self._series(away, 0.0)
        lam = self.poisson.predict_lambda(match_features_frame(
            [build_match_features(sa, sb, 1, 0.0),
             build_match_features(sb, sa, 1, 0.0)]))
        return dc_1x2(float(lam[0]), float(lam[1]))

    def micro_probs(self, home: str, away: str) -> tuple[float, float, float]:
        return self.micro.probs_analytic(home, away)


@lru_cache(maxsize=4)
def _context(cutoff: str = DEFAULT_CUTOFF) -> _Context:
    return _Context(cutoff)


# ------------------------------ meta-modelo ---------------------------------
def train_meta_model(*, save: bool = True, verbose: bool = True
                     ) -> StackingMetaModel:
    """Construye el dataset del backtest y entrena el meta-modelo (CV temporal).
    Lo guarda en models/ensemble_meta.pkl."""
    from ensemble.dataset import X_y, build_backtest_dataset
    df = build_backtest_dataset(verbose=verbose)
    X, y = X_y(df)
    meta = StackingMetaModel().fit(X, y)
    if save:
        meta.save(META_PATH)
        if verbose:
            print(f"[meta] entrenado C={meta.C_} (CV RPS {meta.cv_rps_}) "
                  f"-> {META_PATH}")
    return meta


def _get_meta(meta: Optional[StackingMetaModel]) -> StackingMetaModel:
    if meta is not None:
        return meta
    if META_PATH.exists():
        return StackingMetaModel.load(META_PATH)
    return train_meta_model(verbose=False)


# ------------------------------ predict_final -------------------------------
def predict_final(team_home: str, team_away: str, match_date: str, *,
                  use_llm: bool = True, cutoff: str = DEFAULT_CUTOFF,
                  meta_model: Optional[StackingMetaModel] = None,
                  llm_provider=None, apply_calibration: bool = True
                  ) -> MatchPrediction:
    """Predicción final combinada (core + microsim + llm) vía el meta-modelo.

    `use_llm`: si False (o si la extracción falla) las señales del LLM entran en
    su valor neutro — el meta-modelo igual produce su 1X2 con core+microsim.
    Nunca lanza por la red: la capa LLM degrada con elegancia.
    `apply_calibration`: si hay un T guardado (models/ensemble_temperature.json),
    aplica temperature scaling al 1X2 final (no cambia el ganador, solo afina la
    confianza). Ajústalo con scripts/ensemble_calibrate.py.
    """
    ctx = _context(cutoff)
    meta = _get_meta(meta_model)

    core_p = ctx.core_probs(team_home, team_away)
    micro_p = ctx.micro_probs(team_home, team_away)

    llm_dict = None
    if use_llm:
        try:
            from ensemble.roster import validate_features
            from llm_features import get_match_features
            raw = get_match_features(team_home, team_away, match_date,
                                     provider=llm_provider)
            llm_dict, _ = validate_features(
                raw, teams=list(ctx.teams["team"]), log=True)
        except Exception:
            llm_dict = None

    row = build_feature_row(core_p, micro_p, llm_dict)
    X = pd.DataFrame([row])
    probs = meta.predict_proba_1x2(X)[0]

    if apply_calibration and TEMP_PATH.exists():
        from ensemble.calibrate import TemperatureScaler
        probs = TemperatureScaler.load(TEMP_PATH).transform(probs.reshape(1, -1))[0]
    p1, pX, p2 = (float(v) for v in probs)

    # λ y score_matrix coherentes con el 1X2 final (para cumplir el schema)
    lam_h, lam_a = lambdas_from_1x2(p1, pX, p2)
    score_matrix = _dixon_coles_matrix(lam_h, lam_a)

    return MatchPrediction(
        team_home=team_home, team_away=team_away, match_date=str(match_date),
        prob_home=p1, prob_draw=pX, prob_away=p2,
        lambda_home=lam_h, lambda_away=lam_a,
        model_name=MODEL_NAME, model_version=MODEL_VERSION,
        score_matrix=score_matrix)
