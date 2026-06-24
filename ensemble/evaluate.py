"""Predicciones OUT-OF-FOLD de todos los enfoques + baselines, centralizadas.

Genera, para los partidos del backtest, las probabilidades 1X2 de cada enfoque
sobre las MISMAS filas, de forma leak-free:
  - solo core / solo microsim / promedio core+micro  (de las columnas del dataset)
  - stacking (OOF, TimeSeriesSplit)
  - baseline FIFA ranking (regla simple sobre puntos FIFA, sin entrenar)
  - baseline uniforme (1/3, 1/3, 1/3)

Lo consumen: el reporte comparativo, el bootstrap de significancia y el
reliability diagram. Así todos miran exactamente el mismo conjunto OOF.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ensemble.dataset import X_y, build_backtest_dataset
from ensemble.features import CORE_COLS, MICRO_COLS
from ensemble.meta_model import StackingMetaModel

ROOT = Path(__file__).resolve().parents[1]
FIFA_CSV = ROOT / "files/f0_raw/fifa_ranking.csv"
FIFA_URL = ("https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/"
            "master/ranking_fifa_historical.csv")
FIFA_NAME_FIX = {
    "Korea Republic": "South Korea", "USA": "United States", "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast", "China PR": "China",
    "Czechia": "Czech Republic", "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo", "Türkiye": "Turkey",
}

# baseline FIFA: pendiente FIJA (no entrenada) sobre el log-cociente de puntos,
# y empate fijado a la tasa base típica de fase de grupos. Es una REGLA SIMPLE.
FIFA_BETA = 1.5
FIFA_DRAW = 0.25


def _renorm(p: np.ndarray) -> np.ndarray:
    return p / p.sum(axis=1, keepdims=True)


def _fifa_points(home, away, dates) -> tuple[np.ndarray, np.ndarray]:
    """Puntos FIFA más recientes ANTES de cada partido (merge_asof, sin leak).
    NaN donde no hay dato/equipo. Descarga el CSV histórico si falta."""
    if not FIFA_CSV.exists():
        try:
            urllib.request.urlretrieve(FIFA_URL, FIFA_CSV)
        except Exception:
            n = len(list(home))
            return np.full(n, np.nan), np.full(n, np.nan)
    fifa = pd.read_csv(FIFA_CSV, parse_dates=["date"])
    fifa["team"] = fifa["team"].replace(FIFA_NAME_FIX)
    fifa = fifa[["team", "date", "total_points"]].sort_values("date")
    m = pd.DataFrame({"home": list(home), "away": list(away),
                      "date": pd.to_datetime(list(dates))}).reset_index()
    m = m.sort_values("date")

    def pts(col):
        j = pd.merge_asof(m, fifa.rename(columns={"team": col}),
                          on="date", by=col, direction="backward")
        return j.set_index("index")["total_points"].reindex(range(len(m))).to_numpy()

    return pts("home"), pts("away")


def _fifa_baseline(df: pd.DataFrame) -> np.ndarray:
    ph, pa = _fifa_points(df["home"], df["away"], df["date"])
    out = np.full((len(df), 3), 1 / 3)         # default uniforme si no hay dato
    ok = np.isfinite(ph) & np.isfinite(pa) & (ph > 0) & (pa > 0)
    logdiff = np.zeros(len(df))
    logdiff[ok] = np.log(ph[ok]) - np.log(pa[ok])
    we = 1.0 / (1.0 + np.exp(-FIFA_BETA * logdiff))   # expectativa de victoria local
    out[ok, 0] = we[ok] * (1 - FIFA_DRAW)
    out[ok, 1] = FIFA_DRAW
    out[ok, 2] = (1 - we[ok]) * (1 - FIFA_DRAW)
    return _renorm(out)


def oof_predictions(df: pd.DataFrame | None = None, *, n_splits: int = 6
                    ) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Devuelve (meta_df, probas, mask).

    meta_df: filas del backtest con result/date/home/away.
    probas: {enfoque: ndarray (n,3) en orden [1,X,2]}. El stacking trae NaN en
            el primer fold (sin OOF); el resto está completo.
    mask:   booleano de filas con predicción de stacking (las comparables OOF).
    """
    df = build_backtest_dataset() if df is None else df
    X, y = X_y(df)
    n = len(df)
    core = df[CORE_COLS].to_numpy()
    micro = df[MICRO_COLS].to_numpy()

    oof = np.full((n, 3), np.nan)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(X):
        if len(np.unique(y.iloc[tr])) < 3:
            continue
        meta = StackingMetaModel(C=1.0).fit(X.iloc[tr], y.iloc[tr], n_splits=3)
        oof[va] = meta.predict_proba_1x2(X.iloc[va])

    probas = {
        "solo core": core,
        "solo microsim": micro,
        "promedio core+micro": _renorm(0.5 * core + 0.5 * micro),
        "stacking (OOF)": oof,
        "baseline FIFA ranking": _fifa_baseline(df),
        "baseline uniforme": np.full((n, 3), 1 / 3),
    }
    mask = ~np.isnan(oof[:, 0])
    meta_df = df[["result", "date", "home", "away"]].reset_index(drop=True)
    return meta_df, probas, mask
