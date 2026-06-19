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
from src.simulation.monte_carlo import (_demargin, _dixon_coles_matrix,
                                        blend_with_market, dc_1x2,
                                        lambdas_from_1x2)


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


def test_demargin_removes_overround():
    p = _demargin(2.0, 3.5, 4.0)
    assert np.isclose(sum(p), 1.0)
    assert p[0] > p[2]                       # menor odd -> mayor probabilidad


def test_lambdas_from_1x2_roundtrip():
    """Re-derivar λ de un 1X2 objetivo debe reproducir ese 1X2 (matriz DC)."""
    la, lb = lambdas_from_1x2(0.30, 0.30, 0.40)
    p1, pX, p2 = dc_1x2(la, lb)
    assert np.allclose([p1, pX, p2], [0.30, 0.30, 0.40], atol=0.01)


def test_blend_with_market_alpha_extremes():
    model_probs = pd.DataFrame([{
        "home_team": "A", "away_team": "B",
        "p_home": 0.7, "p_draw": 0.2, "p_away": 0.1}])
    odds = pd.DataFrame([{
        "home_team": "A", "away_team": "B",
        "odds_home": 3.0, "odds_draw": 3.0, "odds_away": 2.3}])
    mkt = _demargin(3.0, 3.0, 2.3)
    only_model = blend_with_market(model_probs, odds, alpha=1.0)
    only_market = blend_with_market(model_probs, odds, alpha=0.0)
    assert bool(only_model.at[0, "blended"])
    assert np.isclose(only_model.at[0, "p_home"], 0.7)        # alpha=1 -> modelo
    assert np.isclose(only_market.at[0, "p_home"], mkt[0], atol=1e-6)  # alpha=0
    # un partido sin odds queda intacto
    other = pd.DataFrame([{"home_team": "C", "away_team": "D",
                           "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}])
    res = blend_with_market(other, odds, alpha=0.3)
    assert not bool(res.at[0, "blended"])
    assert np.isclose(res.at[0, "p_home"], 0.5)


def test_match_engine_micro_and_blend():
    """Motor pre-partido (V3): probabilidades válidas, equipo fuerte favorito,
    blend en log-odds que suma 1 y respeta los extremos de alpha."""
    from src.simulation.match_engine import (Player, Squad, blend_predictions,
                                             simulate_match)

    def squad(name, atk):
        ps = [Player("GK", "GK", psxg_save_pct=0.74)]
        ps += [Player(f"{name}{i}", pos, npxg_p90=a)
               for i, (pos, a) in enumerate(
                   zip(["FB", "CB", "CB", "FB", "MF", "MF", "AM", "WF", "FW", "WF"],
                       atk))]
        return Squad(name, ps)

    strong = squad("S", [.05, .04, .04, .05, .08, .10, .20, .30, .45, .28])
    weak = squad("W", [.03, .02, .02, .03, .05, .05, .08, .10, .15, .10])
    res = simulate_match(strong, weak, n_sims=2000)
    assert np.isclose(res["p_home_win"] + res["p_draw"] + res["p_away_win"], 1.0)
    assert res["lambda_home"] > res["lambda_away"]      # el fuerte genera más
    assert res["p_home_win"] > res["p_away_win"]

    b = blend_predictions([0.5, 0.3, 0.2], [0.3, 0.3, 0.4], alpha=0.65,
                          p_market=[0.45, 0.30, 0.25], beta=0.2)
    assert np.isclose(sum(b), 1.0)
    only_stat = blend_predictions([0.5, 0.3, 0.2], [0.3, 0.3, 0.4], alpha=1.0)
    assert np.allclose(only_stat, [0.5, 0.3, 0.2], atol=1e-6)


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
