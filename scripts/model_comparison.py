#!/usr/bin/env python3
"""Dossier comparativo: qué dice CADA modelo por partido + aciertos.

Una sola tabla, estilo dossier, con columnas core | microsim | llm | ensemble
para los 72 partidos del Mundial, más el resultado real y los puntos (5/3/0)
de cada modelo en los ya jugados — para ver de un vistazo quién va acertando.

Fuentes (todo offline, sin re-correr LLM):
  - core      : files/f3_output/match_predictions.csv (1X2 + marcador ya calculados)
  - microsim  : MarketValueMicroSim.from_teams (valor de plantel, analítico)
  - llm       : señales del log results/live_log/proximos_mundial_2026.csv (si está)
  - ensemble  : meta-modelo sobre core+microsim+llm (+ calibración), recomputado
  - real      : columna 'Real' del scorecard (results/reports/scorecard.md)

Uso: python scripts/model_comparison.py
Salida: results/reports/model_comparison.md
"""

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.simulation.monte_carlo import (_dixon_coles_matrix,
                                         consistent_modal_score,
                                         lambdas_from_1x2)
from ensemble.calibrate import TemperatureScaler
from ensemble.features import build_feature_row
from ensemble.meta_model import StackingMetaModel
from microsim.model import MarketValueMicroSim

CORE_CSV = ROOT / "files/f3_output/match_predictions.csv"
LOG_CSV = ROOT / "results/live_log/proximos_mundial_2026.csv"
SCORECARD = ROOT / "results/reports/scorecard.md"
META_PATH = ROOT / "models/ensemble_meta.pkl"
TEMP_PATH = ROOT / "models/ensemble_temperature.json"
TEAMS_CSV = ROOT / "files/f0_raw/teams_2026.csv"
OUT = ROOT / "results/reports/model_comparison.md"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _score_from_probs(p1: float, pX: float, p2: float) -> str:
    """Marcador modal CONSISTENTE con el 1X2 (no el argmax global, que para
    equipos parejos colapsa a empate aunque el 1X2 favorezca a un ganador)."""
    la, lb = lambdas_from_1x2(p1, pX, p2)
    m = _dixon_coles_matrix(la, lb)
    i, j = consistent_modal_score(m, p1, pX, p2)
    return f"{i}-{j}"


def _res(p1: float, pX: float, p2: float) -> str:
    return ["1", "X", "2"][int(np.argmax([p1, pX, p2]))]


def _parse_scorecard() -> dict:
    """{ 'local|visitante' (norm) : (gh, ga) } desde la columna Real."""
    out = {}
    if not SCORECARD.exists():
        return out
    for line in SCORECARD.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "Partido" in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        match, real = cells[2], cells[4]
        if "–" not in match and "-" not in match:
            continue
        sep = "–" if "–" in match else "-"
        parts = match.split(sep)
        if len(parts) != 2:
            continue
        m = re.match(r"(\d+)\s*-\s*(\d+)", real)
        if not m:
            continue
        key = f"{_norm(parts[0])}|{_norm(parts[1])}"
        out[key] = (int(m.group(1)), int(m.group(2)))
    return out


def _points(pred_score: str, actual: tuple[int, int]) -> int:
    if actual is None:
        return None
    gh, ga = actual
    ar = "1" if gh > ga else ("2" if gh < ga else "X")
    try:
        pi, pj = (int(x) for x in pred_score.split("-"))
    except (ValueError, AttributeError):
        return 0
    pr = "1" if pi > pj else ("2" if pi < pj else "X")
    return 5 if (pi, pj) == (gh, ga) else (3 if pr == ar else 0)


def _llm_nested(logrow) -> dict | None:
    if logrow is None:
        return None
    def n(v):
        return None if pd.isna(v) else v
    return {
        "home": {"lesionados_clave": ["x"] * int(logrow.get("home_lesionados_n", 0) or 0),
                 "cambio_dt_reciente": bool(logrow.get("home_cambio_dt")),
                 "consenso_expertos_pct": n(logrow.get("home_consenso_pct")),
                 "fatiga_husos_horarios": n(logrow.get("home_fatiga_husos"))},
        "away": {"lesionados_clave": ["x"] * int(logrow.get("away_lesionados_n", 0) or 0),
                 "cambio_dt_reciente": bool(logrow.get("away_cambio_dt")),
                 "consenso_expertos_pct": n(logrow.get("away_consenso_pct")),
                 "fatiga_husos_horarios": n(logrow.get("away_fatiga_husos"))},
        "dead_rubber": bool(logrow.get("dead_rubber")),
    }


def _llm_summary(logrow) -> str:
    if logrow is None:
        return "—"
    lh = int(logrow.get("home_lesionados_n", 0) or 0)
    la = int(logrow.get("away_lesionados_n", 0) or 0)
    bits = [f"lesion {lh}/{la}"]
    if bool(logrow.get("home_cambio_dt")) or bool(logrow.get("away_cambio_dt")):
        bits.append("ΔDT")
    if bool(logrow.get("dead_rubber")):
        bits.append("dead")
    return " · ".join(bits)


def main() -> None:
    fx = pd.read_csv(ROOT / "files/f0_raw/fixtures_2026.csv")
    core = pd.read_csv(CORE_CSV)
    teams = pd.read_csv(TEAMS_CSV)
    actuals = _parse_scorecard()
    log = (pd.read_csv(LOG_CSV) if LOG_CSV.exists() else pd.DataFrame())
    log_by = {f"{_norm(r['home'])}|{_norm(r['away'])}": r
              for _, r in log.iterrows()} if len(log) else {}

    micro = MarketValueMicroSim.from_teams(teams)
    meta = (StackingMetaModel.load(META_PATH) if META_PATH.exists()
            else None)
    temp = TemperatureScaler.load(TEMP_PATH) if TEMP_PATH.exists() else None

    core_idx = {(r.team_a, r.team_b): r for r in core.itertuples()}
    rows, totals = [], {"core": [0, 0], "microsim": [0, 0], "ensemble": [0, 0]}
    n_played = 0

    for f in fx.itertuples():
        a, b = f.team_a, f.team_b
        c = core_idx.get((a, b))
        if c is None:
            continue
        core_p = (float(c.p_win_a), float(c.p_draw), float(c.p_win_b))
        core_score = _score_from_probs(*core_p)   # consistente con el 1X2

        mp = micro.probs_analytic(a, b)
        micro_score = _score_from_probs(*mp)

        key = f"{_norm(a)}|{_norm(b)}"
        logrow = log_by.get(key)
        ens_p, ens_score = None, "—"
        if meta is not None:
            row = build_feature_row(core_p, mp, _llm_nested(logrow))
            ep = meta.predict_proba_1x2(pd.DataFrame([row]))[0]
            if temp is not None:
                ep = temp.transform(ep.reshape(1, -1))[0]
            ens_p = tuple(float(x) for x in ep)
            ens_score = _score_from_probs(*ens_p)

        actual = actuals.get(key)
        rows.append({
            "date": f.date, "match": f"{a} – {b}",
            "core": f"{core_p[0]:.2f}/{core_p[1]:.2f}/{core_p[2]:.2f} {core_score}",
            "microsim": f"{mp[0]:.2f}/{mp[1]:.2f}/{mp[2]:.2f} {micro_score}",
            "llm": _llm_summary(logrow),
            "ensemble": (f"{ens_p[0]:.2f}/{ens_p[1]:.2f}/{ens_p[2]:.2f} {ens_score}"
                         if ens_p else "—"),
            "real": (f"{actual[0]}-{actual[1]}" if actual else ""),
            "pts": "",
        })
        if actual is not None:
            n_played += 1
            pc = _points(core_score, actual)
            pm = _points(micro_score, actual)
            pe = _points(ens_score, actual) if ens_p else 0
            rows[-1]["pts"] = f"C{pc} M{pm} E{pe}"
            for k, p in (("core", pc), ("microsim", pm), ("ensemble", pe)):
                totals[k][0] += p
                totals[k][1] += 1 if p >= 3 else 0

    _write(rows, totals, n_played)
    print(f"\n{len(rows)} partidos · {n_played} jugados")
    print("Puntos (5/3/0) y aciertos de resultado en los jugados:")
    for k, (pts, hits) in totals.items():
        acc = hits / n_played if n_played else 0
        print(f"  {k:>9}: {pts} pts · {hits}/{n_played} resultados ({acc:.0%})")
    print(f"\nTabla completa: {OUT}")


def _write(rows, totals, n_played) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Dossier comparativo — core · microsim · llm · ensemble\n",
          "1X2 = P(local)/P(empate)/P(visita) + marcador modal **consistente con "
          "el 1X2** (se fija primero el resultado más probable y dentro de él el "
          "marcador más probable; ya NO el argmax global que colapsaba a empate). "
          "`pts`: C=core M=microsim E=ensemble (5 exacto / 3 resultado / 0).\n",
          "| Fecha | Partido | CORE | MICROSIM | LLM (señales) | ENSEMBLE | Real | Pts |",
          "|:-:|---|---|---|---|---|:-:|:-:|"]
    for r in rows:
        md.append(f"| {r['date']} | {r['match']} | {r['core']} | {r['microsim']} "
                  f"| {r['llm']} | {r['ensemble']} | {r['real']} | {r['pts']} |")
    md.append(f"\n## Aciertos en los {n_played} partidos jugados\n")
    md.append("| Modelo | Puntos (5/3/0) | Aciertos de resultado |")
    md.append("|---|:-:|:-:|")
    for k, (pts, hits) in totals.items():
        acc = hits / n_played if n_played else 0
        md.append(f"| {k} | {pts} | {hits}/{n_played} ({acc:.0%}) |")
    md.append("\n> microsim y ensemble se recomputan aquí de forma retrospectiva "
              "para los jugados (no fueron pre-registrados antes de esos "
              "partidos); core es el pick congelado del scorecard. El LLM no "
              "predice resultado: aporta señales (lesionados, ΔDT, dead rubber).")
    OUT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
