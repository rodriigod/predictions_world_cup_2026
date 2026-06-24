"""Tests de los diagnósticos del ensemble: evaluación OOF, baselines y el
pre-registro de amistosos (logging en vivo)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------- A/B: evaluación + baselines --------------------
def test_oof_predictions_sum_to_one_and_have_baselines():
    from ensemble.evaluate import oof_predictions
    meta_df, probas, mask = oof_predictions()
    assert {"solo core", "stacking (OOF)", "baseline FIFA ranking",
            "baseline uniforme"} <= set(probas)
    # uniforme = 1/3 exacto
    assert np.allclose(probas["baseline uniforme"], 1 / 3)
    # cada enfoque suma 1 donde está definido (en filas OOF)
    for name, p in probas.items():
        pm = p[mask]
        assert np.allclose(pm.sum(axis=1), 1.0), name
    assert mask.sum() > 0 and mask.sum() <= len(meta_df)


# --------------------------- calibración (temperature) ----------------------
def test_temperature_sharpen_and_soften():
    from ensemble.calibrate import TemperatureScaler
    p = np.array([[0.5, 0.3, 0.2]])
    sharp = TemperatureScaler(0.5).transform(p)[0]
    soft = TemperatureScaler(2.0).transform(p)[0]
    assert sharp[0] > p[0, 0]                       # T<1 afila el máximo
    assert soft[0] < p[0, 0]                        # T>1 lo suaviza
    assert np.isclose(sharp.sum(), 1.0) and np.isclose(soft.sum(), 1.0)
    assert np.argmax(sharp) == np.argmax(soft) == 0  # no cambia el ganador


def test_temperature_fit_recovers_underconfidence():
    """Si el modelo es sub-confiado, fit debe devolver T<1 (afilar)."""
    from ensemble.calibrate import TemperatureScaler
    rng = np.random.default_rng(0)
    # genera probs sub-confiadas: la clase real ocurre más que lo predicho
    n = 2000
    p = np.full((n, 3), [0.5, 0.3, 0.2])
    y = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1])  # local gana más que 0.5
    T = TemperatureScaler().fit(p, y).T_
    assert T < 1.0


def test_temperature_save_load(tmp_path):
    from ensemble.calibrate import TemperatureScaler
    path = tmp_path / "t.json"
    TemperatureScaler(0.85).save(path)
    assert abs(TemperatureScaler.load(path).T_ - 0.85) < 1e-9


# --------------------------- E: logging de amistosos ------------------------
@pytest.fixture
def log_path(tmp_path, monkeypatch):
    import scripts.log_friendlies as lf
    p = tmp_path / "amistosos.csv"
    monkeypatch.setattr(lf, "LOG", p)
    return lf


def _seed_prediction(lf, home="España", away="Alemania", date="2026-03-25"):
    """Inserta una fila pre-registrada SIN llamar a predict_final (rápido)."""
    row = {c: None for c in lf.ALL_COLS}
    row.update({"match_id": lf._match_id(home, away, date), "date": date,
                "home": home, "away": away, "p_home": 0.64, "p_draw": 0.19,
                "p_away": 0.17, "pred_result": "1", "pred_score": "2-1",
                "status": "pending"})
    lf._save(pd.DataFrame([row]))


def test_record_result_preserves_prediction(log_path):
    lf = log_path
    _seed_prediction(lf)
    before = lf._load().iloc[0].to_dict()
    lf.record_result("España", "Alemania", "2026-03-25", 2, 1)
    after = lf._load().iloc[0].to_dict()
    # la predicción NO se toca
    for col in ("p_home", "p_draw", "p_away", "pred_result", "pred_score"):
        assert after[col] == before[col]
    # el resultado se cargó y los puntos son 5 (marcador exacto 2-1)
    assert after["status"] == "played"
    assert int(after["actual_home_goals"]) == 2
    assert str(after["actual_result"]) == "1"        # CSV puede inferir int
    assert int(after["points_5_3_0"]) == 5


def test_record_result_points_outcome_only(log_path):
    lf = log_path
    _seed_prediction(lf)
    lf.record_result("España", "Alemania", "2026-03-25", 3, 0)   # gana local, no exacto
    after = lf._load().iloc[0].to_dict()
    assert int(after["points_5_3_0"]) == 3                        # acierta resultado


def test_record_result_requires_pre_registration(log_path):
    lf = log_path
    with pytest.raises(ValueError):
        lf.record_result("Brasil", "Chile", "2026-03-26", 1, 0)  # nunca pre-registrado


def test_match_id_stable(log_path):
    lf = log_path
    assert lf._match_id("A", "B", "2026-01-01") == "2026-01-01|A|B"
