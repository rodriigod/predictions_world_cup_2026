"""Tests del módulo ensemble (stacking core + microsim + llm)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble.features import (CORE_COLS, LLM_COLS, META_COLUMNS,
                              build_feature_row)
from ensemble.meta_model import StackingMetaModel
from ensemble.roster import (looks_like_name, name_in_roster, real_roster,
                            validate_features)
from schema import MatchPrediction


# ------------------------------- roster -------------------------------------
def test_roster_conservative_keeps_real_unverified():
    """Conservador (default): un jugador real ausente de FIFA-24 NO se descarta
    (se conserva como unverified) — evita borrar lesiones reales."""
    import pandas as pd
    teams = pd.read_csv(ROOT / "files/f0_raw/teams_2026.csv")["team"].tolist()
    feats = {"team_home": "España", "team_away": "Uruguay",
             "match_date": "2026-06-21",
             "home": {"lesionados_clave": ["Carvajal", "Lamine"]},
             "away": {"lesionados_clave": ["Darwin Núñez"]}}
    clean, _ = validate_features(feats, teams=teams, log=False)
    assert clean["home"]["lesionados_clave"] == ["Carvajal", "Lamine"]
    assert clean["away"]["lesionados_clave"] == ["Darwin Núñez"]


def test_roster_discards_clear_cross_contamination():
    """'Lionel Messi' en Uruguay (nombre completo de Argentina) -> descartado."""
    import pandas as pd
    teams = pd.read_csv(ROOT / "files/f0_raw/teams_2026.csv")["team"].tolist()
    feats = {"team_home": "Uruguay", "team_away": "Cabo Verde",
             "match_date": "2026-06-21",
             "home": {"lesionados_clave": ["Lionel Messi"]}, "away": {}}
    clean, disc = validate_features(feats, teams=teams, log=False)
    assert "Lionel Messi" not in (clean["home"]["lesionados_clave"] or [])
    assert any(d["reason"].startswith("cross_contamination") for d in disc)


def test_roster_discards_junk_not_a_name():
    """Un titular de noticia (no un nombre) se descarta como not_a_name."""
    feats = {"team_home": "Irán", "team_away": "Bélgica",
             "match_date": "2026-06-21",
             "home": {"lesionados_clave": ["Lesiones de última hora 2026: bajas"]},
             "away": {}}
    clean, disc = validate_features(feats, log=False)
    assert not clean["home"]["lesionados_clave"]
    assert any(d["reason"] == "not_a_name" for d in disc)


def test_roster_strict_mode_discards_unverified():
    """En modo strict, un nombre ausente de la nómina propia SÍ se descarta."""
    import pandas as pd
    teams = pd.read_csv(ROOT / "files/f0_raw/teams_2026.csv")["team"].tolist()
    feats = {"team_home": "España", "team_away": "Uruguay",
             "match_date": "2026-06-21",
             "home": {"lesionados_clave": ["Carvajal"]}, "away": {}}
    clean, disc = validate_features(feats, teams=teams, mode="strict", log=False)
    assert not clean["home"]["lesionados_clave"]
    assert any(d["reason"] == "not_in_roster" for d in disc)


def test_looks_like_name():
    assert looks_like_name("Lionel Messi")
    assert not looks_like_name("Lesiones 2026: bajas")   # dígitos + ':'
    assert not looks_like_name("una frase larga con demasiadas palabras aqui")


def test_name_in_roster_token_match():
    roster = {"messi", "lionel", "munoz", "daniel"}
    assert name_in_roster("Daniel Muñoz", roster)      # acentos + token
    assert not name_in_roster("Cristiano Ronaldo", roster)


# ------------------------------- features -----------------------------------
def test_build_feature_row_counts_and_missing():
    core = (0.6, 0.25, 0.15)
    micro = (0.5, 0.3, 0.2)
    llm = {"home": {"lesionados_clave": ["a", "b"], "cambio_dt_reciente": True,
                    "consenso_expertos_pct": 70.0},
           "away": {"lesionados_clave": None, "consenso_expertos_pct": None},
           "dead_rubber": True}
    row = build_feature_row(core, micro, llm)
    assert set(row) == set(META_COLUMNS)
    assert row["lesionados_home"] == 2.0 and row["lesionados_away"] == 0.0
    assert row["cambio_dt_home"] == 1.0
    assert row["dead_rubber"] == 1.0
    assert row["consenso_home"] == 70.0 and row["consenso_home_missing"] == 0.0
    assert row["consenso_away"] == 0.0 and row["consenso_away_missing"] == 1.0


def test_build_feature_row_handles_none_llm():
    row = build_feature_row((0.4, 0.3, 0.3), (0.4, 0.3, 0.3), None)
    assert row["lesionados_home"] == 0.0
    assert row["consenso_home_missing"] == 1.0    # None -> missing


# ------------------------------ meta-modelo ---------------------------------
def _synthetic_meta_df(n=300, seed=0):
    """Dataset sintético donde el resultado depende SOLO de core (microsim/llm
    son ruido/constantes), para verificar que el meta-modelo lo detecta."""
    rng = np.random.default_rng(seed)
    rows, ys = [], []
    for _ in range(n):
        p = rng.dirichlet([3, 2, 3])
        core = tuple(p)
        micro = tuple(rng.dirichlet([2, 2, 2]))     # ruido
        row = build_feature_row(core, micro, None)  # llm constante/neutro
        rows.append(row)
        ys.append(rng.choice(["1", "X", "2"], p=p))
    return pd.DataFrame(rows)[META_COLUMNS], pd.Series(ys)


def test_meta_model_fit_predict_simplex():
    X, y = _synthetic_meta_df()
    meta = StackingMetaModel(C=1.0).fit(X, y)
    proba = meta.predict_proba_1x2(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_meta_model_coefficients_core_dominates_llm_zero():
    X, y = _synthetic_meta_df()
    meta = StackingMetaModel(C=1.0).fit(X, y)
    coef = meta.coefficients()
    by_group = coef.groupby("group")["abs_mean"].sum()
    # core debe pesar más que microsim; llm constante -> coef ~0
    assert by_group["core"] > by_group.get("microsim", 0)
    assert by_group.get("llm", 0) < 1e-6


def test_meta_model_cv_selects_C():
    X, y = _synthetic_meta_df()
    meta = StackingMetaModel().fit(X, y)        # C=None -> CV temporal
    assert meta.C_ in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


# --------------------------- predict_final (integración) --------------------
def test_predict_final_returns_valid_prediction():
    """Integración: predict_final sin LLM (offline) -> MatchPrediction válido."""
    from ensemble import predict_final
    pred = predict_final("Brasil", "Argentina", "2026-06-20", use_llm=False)
    assert isinstance(pred, MatchPrediction)
    assert np.isclose(pred.probs.sum(), 1.0)
    assert pred.model_name == "ensemble_stacking"
    assert pred.score_matrix is not None
