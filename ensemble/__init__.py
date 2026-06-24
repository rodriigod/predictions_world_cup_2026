"""ensemble: combina core/ + microsim/ + llm_features/ en una predicción final.

Stacking: las PROBABILIDADES 1X2 de core/ y microsim/ se apilan y un
meta-modelo (logística multinomial L2) aprende su peso; las SEÑALES del LLM
entran como columnas extra al mismo meta-modelo (no se promedian). Antes, las
extracciones del LLM pasan por VALIDACIÓN DE NÓMINA (roster.py).

Uso rápido:
    from ensemble import predict_final
    pred = predict_final("Brasil", "Argentina", "2026-06-20")   # MatchPrediction

    # (re)entrenar el meta-modelo sobre el backtest de Mundiales:
    from ensemble import train_meta_model
    meta = train_meta_model()
    print(meta.coefficients())     # cuánto pesa core vs microsim vs cada señal
"""

from ensemble.features import META_COLUMNS, build_feature_row
from ensemble.meta_model import StackingMetaModel
from ensemble.predict import (MODEL_NAME, MODEL_VERSION, predict_final,
                              train_meta_model)
from ensemble.roster import real_roster, validate_features

__all__ = [
    "predict_final", "train_meta_model", "StackingMetaModel",
    "validate_features", "real_roster", "build_feature_row", "META_COLUMNS",
    "MODEL_NAME", "MODEL_VERSION",
]
