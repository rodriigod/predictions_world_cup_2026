"""Closing Line Value (CLV): matemática pura, sin red ni I/O.

CLV mide si tu predicción tenía VALOR contra el precio de cierre del mercado —
el mejor proxy disponible de "edge real". La idea: el cierre es el pronóstico más
afilado del mercado (máxima información, mínimo margen); si tu probabilidad, hecha
cuota, le habría ganado al cierre de forma sistemática, tienes ventaja genuina,
no solo accuracy aparente.

Dos lecturas complementarias (se guardan ambas):
  - CLV de PRECIO: open_odds / close_odds − 1, para el resultado que tu modelo
    favorece. >0 = el mercado se movió HACIA tu pick tras la apertura (cerraste
    "por delante de la línea"): señal clásica de edge.
  - EDGE vs cierre: tu_prob − prob_cierre_sin_margen. >0 = tu modelo da más
    probabilidad al resultado que el consenso de cierre (des-marginado).

Sin dependencias de red: las cuotas se inyectan ya obtenidas (ver
scripts/clv_tracking.py para la fuente real, The Odds API).
"""

from __future__ import annotations

import numpy as np

from core.data.odds_tools import demargin

OUTCOMES = ("home", "draw", "away")


def implied_prob(odds: float) -> float:
    """Probabilidad implícita CRUDA (con margen) de una cuota decimal."""
    return 1.0 / float(odds)


def fair_odds(prob: float) -> float:
    """Cuota 'justa' (sin margen) de una probabilidad del modelo."""
    p = min(max(float(prob), 1e-9), 1.0)
    return 1.0 / p


def clv_price(open_odds: float, close_odds: float) -> float:
    """CLV de precio: si hubieras tomado `open_odds` y el cierre fue `close_odds`.
    >0 = el precio se acortó (el mercado vino hacia tu pick). open/close − 1."""
    return float(open_odds) / float(close_odds) - 1.0


def closing_fair_probs(close_odds_1x2, method: str = "shin") -> np.ndarray:
    """Probabilidades de cierre SIN margen (des-marginadas) en orden [1,X,2]."""
    return demargin(close_odds_1x2, method=method)


def compute_clv(model_probs, open_odds_1x2, close_odds_1x2, *,
                method: str = "shin") -> dict:
    """CLV de UN partido. `model_probs`/`*_odds_1x2`: (home, draw, away).

    Elige el resultado que el modelo más favorece y reporta, para ese resultado:
    tu_prob, cuota de apertura/cierre, CLV de precio y edge contra el cierre
    des-marginado. Devuelve un dict listo para una fila del CSV.
    """
    model_probs = [float(x) for x in model_probs]
    pick = int(np.argmax(model_probs))
    close_fair = closing_fair_probs(close_odds_1x2, method=method)
    open_odd = float(open_odds_1x2[pick])
    close_odd = float(close_odds_1x2[pick])
    return {
        "pick": OUTCOMES[pick],
        "model_prob": round(model_probs[pick], 4),
        "open_odds": round(open_odd, 3),
        "close_odds": round(close_odd, 3),
        "open_implied": round(implied_prob(open_odd), 4),
        "close_fair_prob": round(float(close_fair[pick]), 4),
        "clv_price": round(clv_price(open_odd, close_odd), 4),
        "edge_vs_close": round(model_probs[pick] - float(close_fair[pick]), 4),
    }


def summarize_clv(clv_prices) -> dict:
    """Resumen de una serie de CLV de precio: media y % de picks con CLV>0.
    Un CLV medio >0 sostenido es el indicador de edge real (no la accuracy)."""
    arr = np.asarray([c for c in clv_prices if c is not None], float)
    if arr.size == 0:
        return {"n": 0, "clv_mean": None, "beat_close_rate": None}
    return {"n": int(arr.size), "clv_mean": float(arr.mean()),
            "beat_close_rate": float((arr > 0).mean())}
