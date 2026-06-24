"""Tests de las features de LLM (llm_features/).

Todo corre OFFLINE con proveedores FALSOS (callables): nunca se llama a la red
ni se necesita GEMINI_API_KEY. La caché se aísla en un tmp_path por test.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_features import (MatchFeatures, TeamSignals, get_match_features,
                         parse_features)
from llm_features import cache
from llm_features.extract import _extract_json


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """Redirige la caché a un directorio temporal para no tocar files/cache."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "llm_features")


GOOD_JSON = """```json
{"home": {"lesionados_clave": ["Pedri", "Gavi"], "cambio_dt_reciente": false,
          "consenso_expertos_pct": 70, "fatiga_viaje_km": 800,
          "fatiga_husos_horarios": 1},
 "away": {"lesionados_clave": null, "cambio_dt_reciente": true,
          "consenso_expertos_pct": 20},
 "dead_rubber": false, "notes": "fuentes: Marca, ESPN"}
```"""


def fake_provider(_prompt: str) -> str:
    return GOOD_JSON


# ------------------------------- schema -------------------------------------
def test_null_skeleton_all_none():
    f = MatchFeatures.null("A", "B", "2026-06-20")
    d = f.to_dict()
    assert d["source"] == "unavailable"
    assert d["dead_rubber"] is None
    assert d["home"]["lesionados_clave"] is None
    assert d["away"]["consenso_expertos_pct"] is None


def test_roundtrip_to_from_dict():
    f = MatchFeatures("A", "B", "2026-06-20",
                      home=TeamSignals(lesionados_clave=["x"], consenso_expertos_pct=55.0))
    f2 = MatchFeatures.from_dict(f.to_dict())
    assert f2.home.lesionados_clave == ["x"]
    assert f2.home.consenso_expertos_pct == 55.0


def test_flat_dict_summarizes_lists():
    f = MatchFeatures("A", "B", "2026-06-20",
                      home=TeamSignals(lesionados_clave=["x", "y"]))
    flat = f.to_flat_dict()
    assert flat["home_lesionados_clave_n"] == 2
    assert flat["team_home"] == "A"


# ------------------------------- parsing ------------------------------------
def test_extract_json_tolerates_fences_and_prose():
    d = _extract_json("texto\n```json\n{\"a\": 1}\n``` fin")
    assert d == {"a": 1}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("no hay json aqui") is None


def test_parse_features_clamps_and_types():
    raw = '{"home": {"consenso_expertos_pct": 150, "cambio_dt_reciente": "sí"}, ' \
          '"away": {}, "dead_rubber": true}'
    f = parse_features(raw, "A", "B", "2026-06-20")
    assert f.source == "llm_web_search"
    assert f.home.consenso_expertos_pct == 100.0       # clamp a [0,100]
    assert f.home.cambio_dt_reciente is True           # "sí" -> True
    assert f.dead_rubber is True


def test_parse_features_none_on_unparseable():
    assert parse_features("nada de json", "A", "B", "2026-06-20") is None


# ------------------------------- pipeline -----------------------------------
def test_offline_returns_null_skeleton():
    d = get_match_features("A", "B", "2026-06-20", allow_network=False,
                           use_cache=False)
    assert d["source"] == "unavailable"


def test_no_provider_returns_null_skeleton(monkeypatch):
    # sin proveedor disponible -> esqueleto (forzamos default_provider=None)
    import llm_features.extract as ex
    monkeypatch.setattr(ex, "default_provider", lambda: None)
    d = get_match_features("A", "B", "2026-06-20", provider=None,
                           allow_network=True, use_cache=False, searcher=None)
    assert d["source"] == "unavailable"


def test_fake_provider_extracts_and_caches():
    d = get_match_features("España", "Marruecos", "2026-06-20",
                           provider=fake_provider, provider_name="fake",
                           searcher=None)
    assert d["source"] == "llm_web_search"
    assert d["home"]["lesionados_clave"] == ["Pedri", "Gavi"]
    assert d["away"]["cambio_dt_reciente"] is True

    # segunda llamada SIN proveedor y sin red -> debe venir de la caché
    d2 = get_match_features("España", "Marruecos", "2026-06-20",
                            provider=None, allow_network=False, searcher=None)
    assert d2["source"] == "cache"
    assert d2["home"]["lesionados_clave"] == ["Pedri", "Gavi"]


def test_searcher_context_reaches_provider():
    """El contexto del buscador debe inyectarse en el prompt que ve el LLM."""
    seen = {}

    def capturing_provider(prompt):
        seen["prompt"] = prompt
        return GOOD_JSON

    get_match_features("A", "B", "2026-06-20", provider=capturing_provider,
                       searcher=lambda h, a, d: "NOTICIA: lesión de X",
                       use_cache=False)
    assert "NOTICIA: lesión de X" in seen["prompt"]
    assert "única fuente" in seen["prompt"].lower()


def test_garbage_provider_does_not_invent():
    d = get_match_features("A", "B", "2026-06-21",
                           provider=lambda p: "sin datos confiables",
                           use_cache=False, searcher=None)
    assert d["source"] == "unavailable"
    assert d["home"]["lesionados_clave"] is None
    assert d["dead_rubber"] is None


def test_null_result_is_not_cached():
    """Un fallo transitorio (esqueleto null) NO debe quedar fijado en caché."""
    get_match_features("X", "Y", "2026-06-22",
                       provider=lambda p: "garbage", use_cache=True,
                       searcher=None)
    assert cache.load("X", "Y", "2026-06-22") is None


def test_get_features_for_fixtures():
    from llm_features import get_features_for_fixtures
    fx = pd.DataFrame([{"team_a": "A", "team_b": "B", "date": "2026-06-20"},
                       {"team_a": "C", "team_b": "D", "date": "2026-06-21"}])
    out = get_features_for_fixtures(fx, provider=fake_provider, use_cache=False,
                                    searcher=None)
    assert len(out) == 2
    assert all(o["source"] == "llm_web_search" for o in out)
