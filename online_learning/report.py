"""E.2/E.3. Reporte por ronda + anotación de los live logs.

- `generate_round_report`: ELO antes vs después de los partidos cargados, top-5
  que más subieron/bajaron, y accuracy/RPS del modelo ACTUALIZADO vs el ORIGINAL
  sobre esos partidos.
- `annotate_live_logs`: agrega la columna `prob_core_updated` a los logs de
  results/live_log/ para comparar partido a partido estático vs actualizado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from online_learning.bayes_strength import compute_posteriors
from online_learning.dataset import load_results
from online_learning.elo_online import elo_movements
from online_learning.pi_online import pi_movements
from online_learning.priors import canon, to_es
from online_learning.predict_updated import predict_final_updated, updated_1x2
from online_learning.surprise import build_surprise_log, detect_favorite_bias

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/reports/online_learning_round.md"
LIVE_LOG_DIR = ROOT / "results/live_log"
IDX = {"1": 0, "X": 1, "2": 2}


def _rps(probs, actual_idx) -> float:
    cp = np.cumsum(probs)
    co = np.cumsum(np.eye(3)[actual_idx])
    return float(np.sum((cp - co) ** 2) / 2.0)


def _orig_probs(home_en, away_en, date):
    """1X2 del ensemble ORIGINAL (pre-torneo), sin red."""
    from ensemble.predict import predict_final
    p = predict_final(to_es(home_en), to_es(away_en), date, use_llm=False)
    return (p.prob_home, p.prob_draw, p.prob_away)


def generate_round_report(*, save: bool = True) -> str:
    res = load_results()
    n = len(res)
    # --- accuracy/RPS original vs actualizado sobre los partidos jugados ---
    rows, acc_o, acc_u, rps_o, rps_u = [], [], [], [], []
    for r in res.itertuples():
        a, b = r.home_team, r.away_team
        gh, ag = int(r.home_goals), int(r.away_goals)
        ridx = 0 if gh > ag else (2 if gh < ag else 1)
        try:
            po = _orig_probs(a, b, str(pd.Timestamp(r.date).date()))
            _, _, pu = updated_1x2(a, b)
        except Exception:
            continue
        acc_o.append(int(np.argmax(po) == ridx))
        acc_u.append(int(np.argmax(pu) == ridx))
        rps_o.append(_rps(po, ridx)); rps_u.append(_rps(pu, ridx))
        rows.append((to_es(a), to_es(b), f"{gh}-{ag}", po, pu))

    elo_mov = elo_movements()
    ups = [m for m in elo_mov if m["delta"] > 0][:5]
    downs = sorted([m for m in elo_mov if m["delta"] < 0],
                   key=lambda d: d["delta"])[:5]
    md = _render(n, rows, acc_o, acc_u, rps_o, rps_u, ups, downs)
    if save:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(md, encoding="utf-8")
    return md


def _render(n, rows, acc_o, acc_u, rps_o, rps_u, ups, downs) -> str:
    m = len(rows)
    L = ["# E. Online learning — reporte por ronda\n",
         f"Partidos 2026 cargados: **{n}** (evaluados: {m}).\n"]
    if m:
        L += ["## Original vs Actualizado (sobre los partidos jugados)\n",
              "| Modelo | Accuracy | RPS |", "|---|:-:|:-:|",
              f"| ensemble ORIGINAL (pre-torneo) | {np.mean(acc_o):.3f} | "
              f"{np.mean(rps_o):.4f} |",
              f"| ensemble ACTUALIZADO (2026) | {np.mean(acc_u):.3f} | "
              f"{np.mean(rps_u):.4f} |",
              f"\n> Con {m} partido(s), estas cifras son anecdóticas (no "
              "concluyentes): el valor del online learning se mide a lo largo del "
              "torneo, no con la primera ronda."]
    L += ["\n## ELO: top movimientos tras la ronda\n",
          "| ↑ Subieron | Δ ELO | | ↓ Bajaron | Δ ELO |",
          "|---|:-:|---|---|:-:|"]
    for i in range(max(len(ups), len(downs))):
        u = ups[i] if i < len(ups) else None
        d = downs[i] if i < len(downs) else None
        ut = f"{to_es(u['team_en'])} ({u['elo_pre']:.0f}→{u['elo_now']:.0f})" if u else ""
        ud = f"+{u['delta']:.1f}" if u else ""
        dt = f"{to_es(d['team_en'])} ({d['elo_pre']:.0f}→{d['elo_now']:.0f})" if d else ""
        dd = f"{d['delta']:.1f}" if d else ""
        L.append(f"| {ut} | {ud} | | {dt} | {dd} |")
    L += ["\n## Partidos: original vs actualizado\n",
          "| Partido | Real | ORIG (1/X/2) | ACTUALIZADO (1/X/2) |",
          "|---|:-:|:-:|:-:|"]
    for ho, aw, sc, po, pu in rows:
        L.append(f"| {ho} – {aw} | {sc} | "
                 f"{po[0]:.2f}/{po[1]:.2f}/{po[2]:.2f} | "
                 f"{pu[0]:.2f}/{pu[1]:.2f}/{pu[2]:.2f} |")
    L.append("\n> Reporte autogenerado por `online_learning`. El módulo es "
             "paralelo: NO modifica core/microsim/ensemble.")
    return "\n".join(L)


def annotate_live_logs() -> list[str]:
    """Agrega `prob_core_updated` (prob del ensemble actualizado para el resultado
    originalmente predicho) a cada CSV de results/live_log/. Aditivo y reversible."""
    touched = []
    if not LIVE_LOG_DIR.exists():
        return touched
    for csv in sorted(LIVE_LOG_DIR.glob("*.csv")):
        df = pd.read_csv(csv)
        if not {"home", "away", "pred_result"} <= set(df.columns):
            continue
        vals = []
        for _, row in df.iterrows():
            try:
                _, _, pu = updated_1x2(str(row["home"]), str(row["away"]))
                pr = str(row["pred_result"])
                vals.append(round(float(pu[IDX.get(pr, 0)]), 4))
            except Exception:
                vals.append(None)
        df["prob_core_updated"] = vals
        df.to_csv(csv, index=False)
        touched.append(csv.name)
    return touched
