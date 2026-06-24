"""Dataset de entrenamiento del meta-modelo, sobre el backtest de Mundiales.

Para CADA partido de los 7 Mundiales del backtest (1998-2022) genera una fila
con las probabilidades 1X2 de core/ y de microsim/ + las señales del LLM + el
resultado real. Usa EXACTAMENTE el mismo protocolo pre-torneo leak-free que
`scripts/backtest_world_cups.py`: por cada Mundial, core y la fuerza de equipo
se calculan SOLO con partidos previos a ese torneo.

microsim RETROACTIVO (lo que pide el encargo): como NO existen valores de
mercado históricos de plantel, la microsim se alimenta del ELO pre-torneo del
snapshot (`MarketValueMicroSim.from_elo`). HONESTO: así la microsim histórica es
un Dixon-Coles de ELO, correlacionado con core/ — el meta-modelo dirá si aporta.

LLM en el backtest: las señales del LLM (lesionados, cambio_dt, consenso,
fatiga) se obtienen por búsqueda web ACTUAL (2026); NO hay forma leak-free de
reconstruirlas para partidos de 1998-2022 sin fabricar datos. Por eso en el
dataset histórico entran en su valor NEUTRO (0 / NA). Consecuencia esperada y
declarada: sus coeficientes saldrán ~0 por FALTA DE SEÑAL EN ENTRENAMIENTO (no
porque se midan inútiles) — quedan cableadas para la inferencia 2026, donde sí
tienen valores reales. El reporte de coeficientes lo marca explícitamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.data.historical import (WC_BACKTEST_YEARS, WC_START,
                                  build_historical_dataset, wc_backtest_rows)
from core.data.wc_schema import match_features_frame
from core.models.poisson_goals import PoissonGoalsModel
from core.simulation.monte_carlo import dc_1x2
from ensemble.features import META_COLUMNS, build_feature_row
from microsim.model import MarketValueMicroSim

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "files/cache/ensemble/backtest_dataset.csv"


def build_backtest_dataset(*, use_cache: bool = True,
                           verbose: bool = True) -> pd.DataFrame:
    """Construye (o lee de caché) el dataset del meta-modelo.

    Columnas: META_COLUMNS + [result, date, year, home, away, stage].
    `result` ∈ {1, X, 2} es el target.
    """
    if use_cache and CACHE.exists():
        df = pd.read_csv(CACHE, parse_dates=["date"])
        if verbose:
            print(f"[dataset] leído de caché: {len(df)} filas ({CACHE})")
        return df

    records: list[dict] = []
    for year in WC_BACKTEST_YEARS:
        cutoff = WC_START[year]
        if verbose:
            print(f"[dataset] {year}: entrenando core (Poisson) leak-free ...",
                  flush=True)
        data = build_historical_dataset(cutoff=cutoff)
        poisson = PoissonGoalsModel(backend="poisson")
        poisson.fit(data["X"], data["y"], sample_weight=data["w"])

        rows = wc_backtest_rows(year, data["snapshots"])
        if not rows:
            continue
        n = len(rows)
        lam = poisson.predict_lambda(match_features_frame(
            [r["feat_a"] for r in rows] + [r["feat_b"] for r in rows]))

        # microsim retroactivo: fuerza desde el ELO pre-torneo del snapshot
        teams_in = {r["home"] for r in rows} | {r["away"] for r in rows}
        elo_by_team = {t: float(data["snapshots"][t].elo) for t in teams_in
                       if t in data["snapshots"]}
        sim = MarketValueMicroSim.from_elo(elo_by_team)

        for i, r in enumerate(rows):
            core_probs = dc_1x2(float(lam[i]), float(lam[n + i]))
            micro_probs = sim.probs_analytic(r["home"], r["away"])
            feat = build_feature_row(core_probs, micro_probs, llm=None)
            records.append({**feat, "result": r["result"], "date": r["date"],
                            "year": year, "home": r["home"], "away": r["away"],
                            "stage": r["stage"]})
        if verbose:
            print(f"           {n} partidos", flush=True)

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    if verbose:
        print(f"[dataset] {len(df)} filas -> {CACHE}")
    return df


def X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de features (META_COLUMNS) y el target `result`."""
    return df[META_COLUMNS].copy(), df["result"].copy()
