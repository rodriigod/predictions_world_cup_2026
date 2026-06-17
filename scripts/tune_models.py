"""Tuning leak-free con Optuna (Parte 2). Reporta si los hiperparámetros
óptimos baten a los defaults de producción; NO los promueve solo.

- Clasificadores 1X2 (XGBoost, logística): TimeSeriesSplit con gap sobre
  partidos GENERALES (no-Mundial). El test de los 7 Mundiales queda intacto.
- half-life del decaimiento: grid leak-free con holdout cronológico (Poisson
  -> Dixon-Coles -> RPS) sobre partidos generales.

Uso:
    python scripts/tune_models.py --n-trials 40
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.historical import build_historical_dataset
from src.data.wc_schema import FEATURE_NAMES
from src.models.poisson_goals import PoissonGoalsModel
from src.simulation.monte_carlo import _dixon_coles_matrix
from src.tuning.optuna_tuner import (LABEL, default_cv_rps, rps,
                                     tune_classifier)

CUTOFF = "2026-06-01"   # excluye el Mundial 2026 (fixture sin resultado igual)


def _dc_rps(lam_h, lam_a, y_idx) -> float:
    proba = []
    for lh, la in zip(lam_h, lam_a):
        m = _dixon_coles_matrix(float(lh), float(la))
        proba.append([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
    return rps(y_idx, np.array(proba))


def tune_half_life(grid) -> tuple[float, dict]:
    """Holdout cronológico (80/20) sobre partidos generales, por cada hl."""
    out = {}
    for hl in grid:
        data = build_historical_dataset(cutoff=CUTOFF, half_life=hl)
        wc = data["match_is_wc"]
        gen = ~wc
        n = int(gen.sum())
        split = int(n * 0.8)
        order = np.where(gen)[0]                 # ya en orden cronológico
        tr, te = order[:split], order[split:]
        Xa, Xb = data["X_match"], data["X_match_away"]
        # entrenar Poisson sobre filas equipo-partido del train (ambos lados)
        m = PoissonGoalsModel(backend="poisson")
        Xfit = data["X"]; yfit = data["y"]; wfit = data["w"]
        # filtrar X (equipo-partido) por fecha de train: usar row_dates
        rd = data["row_dates"].to_numpy()
        cut_date = data["match_dates"].to_numpy()[te[0]]
        fit_mask = rd < cut_date
        m.fit(Xfit[fit_mask], yfit[fit_mask], sample_weight=wfit[fit_mask])
        y_idx = data["y_result"].iloc[te].map(LABEL).to_numpy()
        lam_h = m.predict_lambda(Xa.iloc[te])
        lam_a = m.predict_lambda(Xb.iloc[te])
        out[hl] = _dc_rps(lam_h, lam_a, y_idx)
    best = min(out, key=out.get)
    return best, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=40)
    args = ap.parse_args()

    print(f"[1] Construyendo dataset (cutoff {CUTOFF})...")
    data = build_historical_dataset(cutoff=CUTOFF)
    gen = ~data["match_is_wc"]
    X = data["X_match"][gen].reset_index(drop=True)
    y_idx = data["y_result"][gen].map(LABEL).to_numpy()
    w = data["w_match"][gen]
    print(f"    {len(X):,} partidos generales (no-Mundial) para tuning leak-free")

    results = {}
    for kind in ["logistic", "xgb"]:
        print(f"\n[2] Tuneando {kind} ({args.n_trials} trials, TimeSeriesSplit)...")
        default = default_cv_rps(kind, X, y_idx, w)
        study = tune_classifier(kind, X, y_idx, w, n_trials=args.n_trials)
        best = study.best_value
        verdict = "MEJORA" if best < default - 1e-4 else "≈ igual / no mejora"
        print(f"    default RPS={default:.4f}  |  tuneado RPS={best:.4f}  -> {verdict}")
        print(f"    best params: {study.best_params}")
        results[kind] = {"default_rps": default, "tuned_rps": best,
                         "best_params": study.best_params, "verdict": verdict}

    print("\n[3] half-life del decaimiento (grid leak-free)...")
    grid = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    best_hl, hl_scores = tune_half_life(grid)
    for hl in grid:
        tag = " <- best" if hl == best_hl else (" (producción)" if hl == 3.0 else "")
        print(f"    hl={hl:>4}: RPS={hl_scores[hl]:.4f}{tag}")
    results["half_life"] = {"best": best_hl, "scores": hl_scores,
                            "production": 3.0}

    out_dir = ROOT / "results/optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_params.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nGuardado: {out_dir / 'best_params.json'}")
    print("\nNOTA: si 'no mejora', los defaults actuales ya son óptimos en "
          "validación honesta (consistente con experimentos previos).")


if __name__ == "__main__":
    main()
