"""Tests del contrato común `MatchPrediction` y del adaptador de `core/`."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.adapter import matches_to_predictions
from schema import MatchPrediction, predictions_to_dataframe


def _fake_matches():
    """Una fila futura (distribución real) y una jugada (1-0-0)."""
    return pd.DataFrame([
        {"team_a": "España", "team_b": "Cabo Verde", "date": "2026-06-20",
         "lambda_a": 1.9, "lambda_b": 0.6,
         "p_win_a": 0.72, "p_draw": 0.18, "p_win_b": 0.10,
         "p_win_a_dc": 0.70, "p_draw_dc": 0.20, "p_win_b_dc": 0.10},
        {"team_a": "México", "team_b": "Sudáfrica", "date": "2026-06-11",
         "lambda_a": 2.0, "lambda_b": 0.5,
         "p_win_a": 1.0, "p_draw": 0.0, "p_win_b": 0.0,
         "p_win_a_dc": 0.65, "p_draw_dc": 0.22, "p_win_b_dc": 0.13},
    ])


def test_schema_rejects_non_simplex():
    with pytest.raises(ValueError):
        MatchPrediction("A", "B", "2026-06-11", 0.5, 0.5, 0.5, 1.0, 1.0, "m", "1")
    with pytest.raises(ValueError):
        MatchPrediction("A", "B", "2026-06-11", -0.1, 0.6, 0.5, 1.0, 1.0, "m", "1")


def test_schema_accepts_valid_and_exposes_helpers():
    p = MatchPrediction("A", "B", "2026-06-11", 0.6, 0.25, 0.15, 1.4, 0.9, "m", "1")
    assert np.isclose(p.probs.sum(), 1.0)
    assert p.match_key == ("2026-06-11", "A", "B")
    assert p.confidence is None and p.score_matrix is None


def test_adapter_preserves_numbers_exactly():
    matches = _fake_matches()
    preds = matches_to_predictions(matches)
    assert len(preds) == 2
    for p, row in zip(preds, matches.itertuples(index=False)):
        # ni un decimal cambia respecto a la tabla del pipeline
        assert p.prob_home == row.p_win_a
        assert p.prob_draw == row.p_draw
        assert p.prob_away == row.p_win_b
        assert p.lambda_home == row.lambda_a
        assert p.lambda_away == row.lambda_b
        assert p.model_name == "poisson_dc_montecarlo"
        # score_matrix reconstruida es una distribución
        assert p.score_matrix.shape == (10, 10)
        assert np.isclose(p.score_matrix.sum(), 1.0)


def test_adapter_dc_source_and_dataframe_view():
    matches = _fake_matches()
    preds = matches_to_predictions(matches, prob_source="dc", include_score_matrix=False)
    assert preds[0].prob_home == matches.iloc[0]["p_win_a_dc"]
    assert preds[0].score_matrix is None
    df = predictions_to_dataframe(preds)
    assert list(df["team_home"]) == ["España", "México"]
    with pytest.raises(ValueError):
        matches_to_predictions(matches, prob_source="bogus")
