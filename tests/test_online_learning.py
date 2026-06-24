"""Tests del módulo online_learning (paralelo a producción)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import online_learning as ol
from online_learning.priors import canon, pretournament_priors


def test_priors_cover_all_48_teams():
    from online_learning.priors import teams_2026
    priors = pretournament_priors()
    for en in teams_2026():
        assert en in priors
        assert priors[en]["elo"] > 0


def test_elo_initialized_from_prior_and_zero_sum():
    priors = pretournament_priors()
    # México ganó todos sus partidos -> su ELO sube respecto al prior
    d_mex = ol.get_elo_2026("México") - priors[canon("México")]["elo"]
    assert d_mex > 0
    # invariante global (robusto al nº de partidos): el ELO es zero-sum ->
    # la suma de todos los Δ se conserva en 0
    total_delta = sum(ol.get_elo_2026(en) - p["elo"]
                      for en, p in priors.items())
    assert total_delta == pytest.approx(0.0, abs=1e-6)


def test_pi_moves_in_right_direction():
    priors = pretournament_priors()
    att0 = priors[canon("México")]["att"]
    att1, _ = ol.get_pi_2026("México")     # buen ataque -> no se desploma
    assert att1 >= att0 - 0.2


def test_bayes_posterior_between_prior_and_obs_with_interval():
    s = ol.get_strength_posterior("Sudáfrica")["off"]
    assert s["lo"] < s["mean"] < s["hi"]                 # IC80 ordenado
    assert s["n"] >= 1 and s["mean"] > 0
    # con prior fuerte (κ=5), la media se mueve poco respecto al prior
    assert abs(s["mean"] - s["prior"]) < 1.0


def test_surprise_log_sign_and_columns():
    df = ol.build_surprise_log(save=False)
    assert set(["date", "equipos", "resultado_real", "prob_core_pre",
                "surprise_score"]).issubset(df.columns)
    # identidad: surprise == 1/3 - prob_core_pre, en TODAS las filas
    for _, row in df.iterrows():
        assert row["surprise_score"] == pytest.approx(
            1 / 3 - row["prob_core_pre"], abs=1e-3)
    # un resultado al que core daba poca prob debe salir como sorpresa (>0)
    assert (df["surprise_score"] > 0).any()


def test_predict_final_updated_valid_distribution():
    p = ol.predict_final_updated("México", "Sudáfrica", "2026-06-20")
    s = p.prob_home + p.prob_draw + p.prob_away
    assert s == pytest.approx(1.0, abs=1e-6)
    assert p.prob_home > p.prob_away          # México claramente favorito
    assert p.model_name.endswith("online")


def test_module_is_self_contained():
    """Ningún archivo de producción importa online_learning."""
    import subprocess
    out = subprocess.run(
        ["grep", "-rn", "online_learning", "core", "ensemble", "microsim",
         "llm_features", "schema.py"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"producción importa online_learning:\n{out.stdout}"
