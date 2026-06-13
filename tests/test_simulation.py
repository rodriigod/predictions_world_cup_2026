"""Tests del pipeline de predicción de fase de grupos."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import make_training_data
from src.data.wc_schema import FEATURE_NAMES
from src.models.poisson_goals import PoissonGoalsModel
from src.simulation import GroupStageSimulator
from src.simulation.monte_carlo import _dixon_coles_matrix


@pytest.fixture(scope="module")
def model():
    X, y = make_training_data(n_matches=1500, seed=1)
    m = PoissonGoalsModel(backend="poisson")
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def teams():
    return pd.read_csv(ROOT / "files/f0_raw/teams_2026.csv")


@pytest.fixture(scope="module")
def fixtures():
    return pd.read_csv(ROOT / "files/f0_raw/fixtures_2026.csv")


def test_training_data_shape():
    X, y = make_training_data(n_matches=100, seed=0)
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == 200  # dos filas espejo por partido
    assert (y >= 0).all()


def test_fixture_integrity(teams, fixtures):
    assert len(fixtures) == 72  # 12 grupos x 6 partidos
    assert len(teams) == 48
    # cada equipo juega exactamente 3 partidos, uno por jornada
    played = pd.concat([
        fixtures[["team_a", "matchday"]].rename(columns={"team_a": "team"}),
        fixtures[["team_b", "matchday"]].rename(columns={"team_b": "team"}),
    ])
    counts = played.groupby("team")["matchday"].agg(["count", "nunique"])
    assert (counts["count"] == 3).all()
    assert (counts["nunique"] == 3).all()
    # los equipos del fixture coinciden con los del CSV de equipos
    assert set(played["team"]) == set(teams["team"])


def test_dixon_coles_matrix_is_distribution():
    m = _dixon_coles_matrix(1.4, 1.1)
    assert m.shape == (10, 10)
    assert np.isclose(m.sum(), 1.0)
    assert (m >= 0).all()


def test_lambda_monotonic_in_strength(model, teams):
    """Un equipo fuerte vs débil debe tener lambda mayor que el inverso."""
    from src.data.wc_schema import build_match_features, match_features_frame
    spain = teams[teams.team == "España"].iloc[0]
    cpv = teams[teams.team == "Cabo Verde"].iloc[0]
    X = match_features_frame([
        build_match_features(spain, cpv, 1),
        build_match_features(cpv, spain, 1),
    ])
    lam = model.predict_lambda(X)
    assert lam[0] > lam[1]


def test_historical_dataset_and_fixed_results(teams, fixtures):
    """Dataset real: integridad básica + partidos jugados quedan fijos."""
    from src.data.historical import (build_historical_dataset,
                                     played_results_es,
                                     teams_table_from_history)
    data = build_historical_dataset()
    assert len(data["X"]) == 2 * len(data["X_match"])
    assert (data["y"] >= 0).all()
    assert (data["w"] > 0).all()
    assert set(data["y_result"]) == {"1", "X", "2"}

    m = PoissonGoalsModel(backend="poisson")
    m.fit(data["X"].sample(6000, random_state=0),
          data["y"].sample(6000, random_state=0))
    hist_teams = teams_table_from_history(data["snapshots"], teams)
    played = played_results_es(data["played_wc"])
    sim = GroupStageSimulator(hist_teams, fixtures, m,
                              played_results=played, seed=3)
    _, matches = sim.run(n_sims=50)
    fixed = matches[matches.status == "JUGADO"]
    assert len(fixed) == len(played)
    # un partido jugado tiene probabilidad 1 en su resultado real
    assert ((fixed[["p_win_a", "p_draw", "p_win_b"]].max(axis=1)) == 1.0).all()


def test_simulation_probabilities_consistent(model, teams, fixtures):
    sim = GroupStageSimulator(teams, fixtures, model, seed=7)
    standings, matches = sim.run(n_sims=400)
    assert len(standings) == 48
    # cada grupo reparte exactamente 1 ganador, 1 segundo, 1 tercero
    by_group = standings.groupby("group")[
        ["p_win_group", "p_runner_up", "p_third_place"]].sum()
    assert np.allclose(by_group, 1.0)
    # avanzan 32 equipos en total (24 directos + 8 terceros)
    assert np.isclose(standings["p_advance"].sum(), 32.0)
    assert standings["exp_points"].between(0, 9).all()
    # probabilidades 1X2 de cada partido suman 1
    probs = matches[["p_win_a", "p_draw", "p_win_b"]].sum(axis=1)
    assert np.allclose(probs, 1.0)
