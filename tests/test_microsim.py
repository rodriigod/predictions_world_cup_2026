"""Tests del modelo de microsimulación por fuerza de plantel (microsim/).

Todo corre OFFLINE: el fallback sintético desde `market_value_meur` existe para
las 48 selecciones, así que no se toca la red.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from microsim import MarketValueMicroSim, compute_strengths
from microsim.ingest import (PlayerValue, Squad, load_all_squads, load_squad,
                             normalize_position, synthetic_squad_from_total)
from schema import MatchPrediction


@pytest.fixture(scope="module")
def teams():
    return pd.read_csv(ROOT / "files/f0_raw/teams_2026.csv")


@pytest.fixture(scope="module")
def fixtures():
    return pd.read_csv(ROOT / "files/f0_raw/fixtures_2026.csv")


@pytest.fixture(scope="module")
def sim(teams):
    return MarketValueMicroSim.from_teams(teams)


# ------------------------------- ingest -------------------------------------
def test_normalize_position_maps_to_groups():
    assert normalize_position("Centre-Back") == "DEF"
    assert normalize_position("Left Winger") == "ATT"
    assert normalize_position("Goalkeeper") == "GK"
    assert normalize_position("attacking midfield") == "MID"
    assert normalize_position("cosa rara") == "MID"   # default


def test_synthetic_squad_preserves_total(teams):
    total = float(teams.iloc[0]["market_value_meur"])
    squad = synthetic_squad_from_total(teams.iloc[0]["team"], total)
    assert squad.source == "synthetic_total"
    assert np.isclose(squad.total_value_meur, total, rtol=1e-6)
    # tiene jugadores de los cuatro grupos
    assert {p.position_group for p in squad.players} == {"GK", "DEF", "MID", "ATT"}


def test_load_squad_falls_back_offline(teams):
    squad = load_squad("España", teams, allow_network=False)
    assert isinstance(squad, Squad)
    assert squad.source == "synthetic_total"
    assert squad.total_value_meur > 0


def test_load_squad_unknown_team_raises(teams):
    with pytest.raises(ValueError):
        load_squad("Atlantis", teams, allow_network=False)


def test_load_all_squads_covers_all_teams(teams):
    squads = load_all_squads(teams, allow_network=False)
    assert len(squads) == len(teams) == 48


# ------------------------------ strength ------------------------------------
def test_strengths_normalized_to_league_mean(teams):
    strengths = compute_strengths(load_all_squads(teams, allow_network=False))
    att = np.array([s.attack for s in strengths.values()])
    dfn = np.array([s.defense for s in strengths.values()])
    assert np.isclose(att.mean(), 1.0, atol=1e-6)
    assert np.isclose(dfn.mean(), 1.0, atol=1e-6)


def test_strong_team_has_more_attack_than_weak(sim):
    assert sim.strengths["España"].attack > sim.strengths["Cabo Verde"].attack
    assert sim.strengths["Brasil"].attack > sim.strengths["Sudáfrica"].attack


def test_position_weights_separate_attack_and_defense():
    """Con un plantel real por jugador, un equipo cargado al ataque debe tener
    más índice ofensivo que defensivo (y viceversa)."""
    attacking = Squad("ATK", [
        PlayerValue("a", "ATT", 100), PlayerValue("b", "ATT", 100),
        PlayerValue("c", "MID", 50), PlayerValue("d", "DEF", 5),
        PlayerValue("g", "GK", 5)], source="test")
    defending = Squad("DEF", [
        PlayerValue("a", "ATT", 5), PlayerValue("b", "MID", 50),
        PlayerValue("c", "DEF", 100), PlayerValue("d", "DEF", 100),
        PlayerValue("g", "GK", 60)], source="test")
    st = compute_strengths({"ATK": attacking, "DEF": defending})
    assert st["ATK"].attack > st["DEF"].attack
    assert st["DEF"].defense > st["ATK"].defense


# ------------------------------- model --------------------------------------
def test_predict_match_conforms_to_schema(sim):
    pred = sim.predict_match("España", "Cabo Verde", "2026-06-15", n_sims=2000)
    assert isinstance(pred, MatchPrediction)
    assert np.isclose(pred.probs.sum(), 1.0)          # simplex (lo valida schema)
    assert pred.model_name == "microsim_market_value"
    assert pred.score_matrix is not None
    assert np.isclose(pred.score_matrix.sum(), 1.0)


def test_lambda_monotonic_in_strength(sim):
    lam_h, lam_a = sim.lambdas("España", "Cabo Verde")
    assert lam_h > lam_a                                # el fuerte genera más


def test_favorite_is_symmetric(sim):
    """Invertir local/visitante invierte el favorito."""
    a = sim.predict_match("España", "Cabo Verde", "2026-06-15", n_sims=2000)
    b = sim.predict_match("Cabo Verde", "España", "2026-06-15", n_sims=2000)
    assert a.prob_home > a.prob_away
    assert b.prob_away > b.prob_home


def test_predict_fixtures_returns_all(sim, fixtures):
    preds = sim.predict_fixtures(fixtures.head(6), n_sims=500)
    assert len(preds) == 6
    assert all(np.isclose(p.probs.sum(), 1.0) for p in preds)


def test_reproducible_with_seed(teams):
    s1 = MarketValueMicroSim.from_teams(teams, seed=7)
    s2 = MarketValueMicroSim.from_teams(teams, seed=7)
    p1 = s1.predict_match("Brasil", "Japón", "2026-06-15", n_sims=1500)
    p2 = s2.predict_match("Brasil", "Japón", "2026-06-15", n_sims=1500)
    assert p1.prob_home == p2.prob_home and p1.prob_away == p2.prob_away
