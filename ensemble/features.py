"""Esquema de features del meta-modelo (stacking) y construcción de filas.

Una fila del meta-modelo combina:
  - las PROBABILIDADES 1X2 de core/ y de microsim/ (se apilan: el meta-modelo
    aprende cuánto pesar cada uno);
  - las SEÑALES discretas del LLM (no son probabilidades; entran como columnas
    extra, NO se promedian con nada).

Imputación de faltantes: `consenso_expertos` suele no estar disponible -> se
imputa con 0 y se añade una columna indicadora `*_missing` (1 = era NA), para
que el meta-modelo distinga "no hay consenso" de "consenso = 0%". El resto de
señales del LLM faltantes caen a su valor neutro (0 / False).
"""

from __future__ import annotations

from typing import Optional

# --- columnas, en orden estable (lo usa el meta-modelo y el reporte) ---
CORE_COLS = ["p_home_core", "p_draw_core", "p_away_core"]
MICRO_COLS = ["p_home_micro", "p_draw_micro", "p_away_micro"]
LLM_COLS = [
    "lesionados_home", "lesionados_away",            # conteos (post-validación)
    "cambio_dt_home", "cambio_dt_away",              # 0/1
    "dead_rubber",                                   # 0/1
    "consenso_home", "consenso_away",                # % (0-100), imputado 0 si NA
    "consenso_home_missing", "consenso_away_missing",  # 1 = era NA
    "fatiga_home", "fatiga_away",                    # husos horarios (0 si NA)
]
META_COLUMNS = CORE_COLS + MICRO_COLS + LLM_COLS

# etiquetas legibles para el reporte de coeficientes
GROUP_OF = ({c: "core" for c in CORE_COLS}
            | {c: "microsim" for c in MICRO_COLS}
            | {c: "llm" for c in LLM_COLS})


def _count(v) -> float:
    return float(len(v)) if isinstance(v, (list, tuple)) else 0.0


def _bool01(v) -> float:
    return 1.0 if v is True else 0.0


def _num(v, default: float = 0.0) -> float:
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def build_feature_row(core_probs: tuple[float, float, float],
                      micro_probs: tuple[float, float, float],
                      llm: Optional[dict]) -> dict:
    """Arma UNA fila de features del meta-modelo.

    `core_probs`/`micro_probs`: (P(1), P(X), P(2)).
    `llm`: dict del schema de `llm_features` YA validado por `roster.py`
    (o None / esqueleto si no hay features del LLM para este partido).
    """
    row = {
        "p_home_core": core_probs[0], "p_draw_core": core_probs[1],
        "p_away_core": core_probs[2],
        "p_home_micro": micro_probs[0], "p_draw_micro": micro_probs[1],
        "p_away_micro": micro_probs[2],
    }
    home = (llm or {}).get("home") or {}
    away = (llm or {}).get("away") or {}
    row["lesionados_home"] = _count(home.get("lesionados_clave"))
    row["lesionados_away"] = _count(away.get("lesionados_clave"))
    row["cambio_dt_home"] = _bool01(home.get("cambio_dt_reciente"))
    row["cambio_dt_away"] = _bool01(away.get("cambio_dt_reciente"))
    row["dead_rubber"] = _bool01((llm or {}).get("dead_rubber"))
    ch, ca = home.get("consenso_expertos_pct"), away.get("consenso_expertos_pct")
    row["consenso_home"] = _num(ch, 0.0)
    row["consenso_away"] = _num(ca, 0.0)
    row["consenso_home_missing"] = 1.0 if ch is None else 0.0
    row["consenso_away_missing"] = 1.0 if ca is None else 0.0
    row["fatiga_home"] = _num(home.get("fatiga_husos_horarios"), 0.0)
    row["fatiga_away"] = _num(away.get("fatiga_husos_horarios"), 0.0)
    return row
