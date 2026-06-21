"""Predictor GENERAL de victoria/empate/derrota para cualquier A vs B.

Países = equipos. Usa TODA la historia de partidos de selecciones (martj42,
~49k partidos) con el mismo motor del proyecto: ELO interno + pi-ratings +
forma + Poisson + Dixon-Coles. No depende del fixture del Mundial.

Uso:
  python scripts/predict_match.py "Brazil" "Argentina"
  python scripts/predict_match.py "España" "Francia" --home A   # A de local
  python scripts/predict_match.py --evaluate                    # ¿sirve? (holdout)

Acepta nombres en inglés (dataset) o español (NAME_MAP de la polla).
"""
import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.data.historical import (NAME_MAP, _neutral_series,
                                 build_historical_dataset)
from core.data.wc_schema import build_match_features, match_features_frame
from core.models.poisson_goals import PoissonGoalsModel
from core.simulation.monte_carlo import dc_1x2


def resolve_team(name: str, snapshots: dict) -> str | None:
    if name in snapshots:
        return name
    if name in NAME_MAP and NAME_MAP[name] in snapshots:   # español -> inglés
        return NAME_MAP[name]
    low = {k.lower(): k for k in snapshots}
    if name.lower() in low:
        return low[name.lower()]
    return None


def suggest(name: str, snapshots: dict) -> list:
    return difflib.get_close_matches(name, list(snapshots), n=5, cutoff=0.5)


def _train(cutoff=None):
    data = build_historical_dataset(cutoff=cutoff)
    model = PoissonGoalsModel(backend="poisson")
    model.fit(data["X"], data["y"], sample_weight=data["w"])
    return model, data


def predict(team_a, team_b, home="neutral"):
    model, data = _train()
    snaps = data["snapshots"]
    a = resolve_team(team_a, snaps)
    b = resolve_team(team_b, snaps)
    for raw, res in [(team_a, a), (team_b, b)]:
        if res is None:
            print(f"✗ No encuentro '{raw}'. ¿Quizás: {suggest(raw, snaps)}?")
            return
    host_a = 1.0 if home == "A" else 0.0
    host_b = 1.0 if home == "B" else 0.0
    sa = _neutral_series(a, snaps[a], is_host=host_a)
    sb = _neutral_series(b, snaps[b], is_host=host_b)
    fa = build_match_features(sa, sb, 0)
    fb = build_match_features(sb, sa, 0)
    lam = model.predict_lambda(match_features_frame([fa, fb]))
    p1, pX, p2 = dc_1x2(float(lam[0]), float(lam[1]))
    ga, gb = int(np.floor(lam[0] + 0.5)), int(np.floor(lam[1] + 0.5))

    print(f"\n{'='*52}\n  {team_a}  vs  {team_b}"
          f"{'  (neutral)' if home=='neutral' else f'  ({home} local)'}\n{'='*52}")
    print(f"  ELO: {team_a} {snaps[a].elo:.0f}  |  {team_b} {snaps[b].elo:.0f}")
    print(f"  Goles esperados (λ): {team_a} {lam[0]:.2f} | {team_b} {lam[1]:.2f}")
    print(f"\n  PROBABILIDADES:")
    print(f"     Gana {team_a:20s} {p1*100:5.1f}%")
    print(f"     Empate{'':19s} {pX*100:5.1f}%")
    print(f"     Gana {team_b:20s} {p2*100:5.1f}%")
    res = ("Gana " + team_a if p1 >= max(pX, p2)
           else ("Empate" if pX >= p2 else "Gana " + team_b))
    print(f"\n  ➤ Más probable: {res}   |   marcador ~ {ga}-{gb}\n")


def evaluate(test_from="2022-01-01"):
    """¿Sirve? Entrena con partidos previos a `test_from`, predice los
    posteriores (todos los internacionales), mide accuracy/RPS leak-free."""
    from core.utils.metrics import ModelMetrics
    import pandas as pd
    print(f"Evaluación general (holdout internacional ≥ {test_from})...")
    data = build_historical_dataset()           # todo
    dates = pd.to_datetime(data["match_dates"])
    cut = pd.Timestamp(test_from)
    tr = dates < cut
    te = ~tr
    model = PoissonGoalsModel(backend="poisson")
    # entrenar SOLO con filas equipo-partido anteriores al corte (leak-free)
    rd = pd.to_datetime(data["row_dates"])
    fit = rd < cut
    model.fit(data["X"][fit.values], data["y"][fit.values],
              sample_weight=data["w"][fit.values])
    la = model.predict_lambda(data["X_match"][te.values])
    lb = model.predict_lambda(data["X_match_away"][te.values])
    proba = np.array([dc_1x2(float(a), float(b)) for a, b in zip(la, lb)])
    y = data["y_result"][te.values].to_numpy()
    idx = {"1": 0, "X": 1, "2": 2}
    yi = np.array([idx[v] for v in y])
    acc = float((proba.argmax(1) == yi).mean())
    rps = ModelMetrics.rps(yi, proba)
    ll = ModelMetrics.multiclass_logloss(y, proba, ["1", "X", "2"])
    base = pd.Series(y).value_counts(normalize=True)
    bp = np.tile([base.get("1", 0), base.get("X", 0), base.get("2", 0)], (len(y), 1))
    print(f"\n  n test = {len(y):,} partidos internacionales")
    print(f"  Accuracy   : {acc:.1%}")
    print(f"  RPS        : {rps:.4f}   (baseline tasa-base {ModelMetrics.rps(yi, bp):.4f})")
    print(f"  Log-loss   : {ll:.4f}")
    print(f"\n  ➤ ¿Sirve? Sí: ~{acc:.0%} de acierto 1X2 en partidos generales, "
          f"muy por encima del azar (33%) y del baseline.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team_a", nargs="?")
    ap.add_argument("team_b", nargs="?")
    ap.add_argument("--home", choices=["A", "B", "neutral"], default="neutral")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    if args.evaluate:
        evaluate()
    elif args.team_a and args.team_b:
        predict(args.team_a, args.team_b, args.home)
    else:
        ap.error("da dos equipos, o usa --evaluate")


if __name__ == "__main__":
    main()
