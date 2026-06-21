"""Análisis de patrones en las probabilidades predichas vs la realidad.

Predice OUT-OF-SAMPLE (walk-forward por año: cada partido con un modelo
entrenado solo con datos previos, features pre-partido -> sin leak) todos
los partidos 2010-2024, y busca patrones sistemáticos:

  1. Calibración del empate por forma de la distribución (¿los partidos
     parejos terminan en empate más de lo que dice el modelo?).
  2. La hipótesis del usuario: favorito ~50% con empate ≈ victoria rival.
  3. Equipos que el modelo sub/sobre-estima, y equipos "empatadores".

Uso: python scripts/analyze_probabilities.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.data.historical import build_historical_dataset
from core.models.poisson_goals import PoissonGoalsModel
from core.simulation.monte_carlo import _dixon_coles_matrix


def build_predictions(first_year=2010, last_year=2024) -> pd.DataFrame:
    data = build_historical_dataset(cutoff=None)
    rd = data["row_dates"].dt.year.to_numpy()
    my = data["match_dates"].dt.year.to_numpy()
    recs = []
    for year in range(first_year, last_year + 1):
        tr = rd < year
        te = my == year
        if te.sum() == 0:
            continue
        model = PoissonGoalsModel("poisson")
        model.fit(data["X"][tr], data["y"][tr], sample_weight=data["w"][tr])
        la = model.predict_lambda(data["X_match"][te])
        lb = model.predict_lambda(data["X_match_away"][te])
        sub = pd.DataFrame({
            "home": data["match_home"][te].to_numpy(),
            "away": data["match_away"][te].to_numpy(),
            "result": data["y_result"][te].to_numpy(),
            "is_wc": data["match_is_wc"][te],
        })
        p1, pX, p2 = [], [], []
        for a, b in zip(la, lb):
            m = _dixon_coles_matrix(a, b)
            p1.append(float(np.tril(m, -1).sum()))
            pX.append(float(np.trace(m)))
            p2.append(float(np.triu(m, 1).sum()))
        sub["p1"], sub["pX"], sub["p2"] = p1, pX, p2
        recs.append(sub)
    return pd.concat(recs, ignore_index=True)


def section(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def main():
    print("Prediciendo 2010-2024 walk-forward (sin leak)...", flush=True)
    df = build_predictions()
    df["fav"] = df[["p1", "p2"]].max(axis=1)        # prob del favorito a ganar
    df["is_draw"] = (df["result"] == "X").astype(int)
    print(f"  {len(df):,} partidos analizados | empates reales: "
          f"{df['is_draw'].mean():.1%}")

    # ---- 1) ¿Los partidos parejos terminan en empate más de lo predicho? ----
    section("1) EMPATE vs FUERZA DEL FAVORITO  (clave para la hipótesis)")
    print(f"  {'P(favorito gana)':>18} {'n':>5} {'pX modelo':>10} "
          f"{'empate real':>12} {'dif':>7}")
    bins = [0.33, 0.40, 0.45, 0.50, 0.55, 0.65, 0.80, 1.01]
    for lo, hi in zip(bins, bins[1:]):
        s = df[(df["fav"] >= lo) & (df["fav"] < hi)]
        if len(s) < 20:
            continue
        pm, pr = s["pX"].mean(), s["is_draw"].mean()
        print(f"  {f'{lo:.2f}-{hi:.2f}':>18} {len(s):>5} {pm:>10.1%} "
              f"{pr:>12.1%} {pr - pm:>+7.1%}")

    # ---- 2) Hipótesis exacta del usuario ----
    section("2) HIPÓTESIS: favorito ~50% y empate ≈ victoria del otro")
    fav = df[["p1", "p2"]].max(axis=1)
    und = df[["p1", "p2"]].min(axis=1)
    m = (fav.between(0.42, 0.55)) & ((df["pX"] - und).abs() <= 0.07)
    s = df[m]
    print(f"  partidos que matchean el patrón: {len(s)}")
    if len(s):
        fav_wins = (((s["p1"] >= s["p2"]) & (s["result"] == "1")) |
                    ((s["p2"] > s["p1"]) & (s["result"] == "2"))).mean()
        draw_rate = (s["result"] == "X").mean()
        print(f"    -> gana el favorito : {fav_wins:.1%}")
        print(f"    -> EMPATE           : {draw_rate:.1%}  "
              f"(el modelo decía {s['pX'].mean():.1%})")
        print(f"    -> gana el otro     : {1 - fav_wins - draw_rate:.1%}")

    # ---- 3) Calibración del empate por nivel de pX ----
    section("3) CALIBRACIÓN DEL EMPATE  (pX predicho vs empate real)")
    print(f"  {'pX predicho':>14} {'n':>5} {'empate real':>12} {'dif':>7}")
    for lo, hi in [(0, .20), (.20, .24), (.24, .27), (.27, .30), (.30, 1)]:
        s = df[(df["pX"] >= lo) & (df["pX"] < hi)]
        if len(s) < 20:
            continue
        pr = s["is_draw"].mean()
        print(f"  {f'{lo:.2f}-{hi:.2f}':>14} {len(s):>5} {pr:>12.1%} "
              f"{pr - s['pX'].mean():>+7.1%}")

    # ---- 4) Patrones por EQUIPO ----
    section("4) EQUIPOS: realidad vs lo que esperaba el modelo (>=40 partidos)")
    rows = []
    for _, r in df.iterrows():
        rows.append((r["home"], r["p1"], r["pX"], r["result"] == "1",
                     r["result"] == "X"))
        rows.append((r["away"], r["p2"], r["pX"], r["result"] == "2",
                     r["result"] == "X"))
    t = pd.DataFrame(rows, columns=["team", "p_win", "p_draw", "won", "drew"])
    g = t.groupby("team").agg(n=("won", "size"),
                              exp_win=("p_win", "mean"),
                              act_win=("won", "mean"),
                              exp_draw=("p_draw", "mean"),
                              act_draw=("drew", "mean"))
    g = g[g["n"] >= 40]
    g["win_delta"] = g["act_win"] - g["exp_win"]
    g["draw_delta"] = g["act_draw"] - g["exp_draw"]

    def show(col, title):
        print(f"\n  {title}")
        print(f"  {'equipo':>16} {'n':>4} {'esperado':>9} {'real':>7} {'dif':>7}")
        top = g.sort_values(col, ascending=False)
        for name, r in pd.concat([top.head(6), top.tail(6)]).iterrows():
            base = "exp_win" if "win" in col else "exp_draw"
            act = "act_win" if "win" in col else "act_draw"
            print(f"  {name:>16} {int(r['n']):>4} {r[base]:>9.1%} "
                  f"{r[act]:>7.1%} {r[col]:>+7.1%}")

    show("win_delta", "Sub-estimados (arriba) / sobre-estimados (abajo) en VICTORIAS:")
    show("draw_delta", "Empatan MÁS (arriba) / MENOS (abajo) de lo predicho:")


if __name__ == "__main__":
    main()
