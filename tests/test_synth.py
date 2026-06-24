"""Tests del sintetizador LLM acotado (F). El foco son los LÍMITES DUROS: pase lo
que pase el LLM, la magnitud debe quedar dentro de rango (no se confía en el prompt).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from synth.synthesize import (ACTION_CAP, MAGNITUDE_HARD_CAP, apply_adjustment,
                              clamp_decision, synthesize)


@pytest.mark.parametrize("accion,raw,expected", [
    ("ajuste_fuerte", 0.95, 0.15),     # se pasa del tope global -> 0.15
    ("ajuste_fuerte", -9.0, -0.15),
    ("ajuste_leve", 0.50, 0.05),       # tope de la acción
    ("ajuste_leve", -0.50, -0.05),
    ("sin_cambio", 0.10, 0.0),         # sin_cambio fuerza 0
    ("marcar_revision", 0.13, 0.0),    # revision fuerza 0
    ("ajuste_leve", 0.03, 0.03),       # válido, sin recorte
])
def test_magnitude_hard_caps(accion, raw, expected):
    d = clamp_decision(accion, raw, "x")
    assert d.magnitud == pytest.approx(expected)
    assert abs(d.magnitud) <= MAGNITUDE_HARD_CAP + 1e-12
    assert abs(d.magnitud) <= ACTION_CAP[d.accion] + 1e-12


def test_invalid_action_falls_to_revision():
    d = clamp_decision("invent", 0.04, "x")
    assert d.accion == "marcar_revision" and d.magnitud == 0.0 and d.clamped


def test_garbage_magnitude_is_zero():
    d = clamp_decision("ajuste_leve", "no-numero", "x")
    assert d.magnitud == 0.0
    d = clamp_decision("ajuste_fuerte", float("nan"), "x")
    assert d.magnitud == 0.0


def test_clamped_flag():
    assert clamp_decision("ajuste_leve", 0.5, "x").clamped is True
    assert clamp_decision("ajuste_leve", 0.03, "x").clamped is False


def test_apply_adjustment_keeps_valid_distribution():
    p = [0.55, 0.25, 0.20]
    out = apply_adjustment(p, clamp_decision("ajuste_fuerte", 0.15, "x"))
    assert out.sum() == pytest.approx(1.0)
    assert (out > 0).all()
    assert out[0] > p[0]                # subió el favorito
    # magnitud 0 -> sin cambio
    out0 = apply_adjustment(p, clamp_decision("sin_cambio", 0.0, "x"))
    assert np.allclose(out0, np.array(p) / sum(p))


def test_synthesize_degrades_without_provider():
    # provider que siempre falla -> sin_cambio, sin tocar nada
    d = synthesize([0.5, 0.3, 0.2], [0.5, 0.3, 0.2], {}, provider=lambda _: None)
    assert d.accion == "sin_cambio" and d.magnitud == 0.0


def test_synthesize_clamps_malicious_llm():
    # un LLM que intenta inyectar una magnitud enorme: debe quedar acotado
    bad = lambda _: '{"accion":"ajuste_fuerte","magnitud":5.0,"justificacion":"hack"}'
    d = synthesize([0.5, 0.3, 0.2], [0.5, 0.3, 0.2], {}, provider=bad)
    assert d.magnitud == pytest.approx(0.15) and d.clamped
