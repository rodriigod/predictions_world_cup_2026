#!/usr/bin/env python3
"""E. Pre-registro en vivo de predicciones de amistosos de selecciones.

Corre el pipeline COMPLETO (predict_final = core + microsim + llm_features con
validación de nómina, en tiempo real) sobre amistosos próximos y guarda la
predicción ANTES de que se jueguen — un PRE-REGISTRO honesto. Después, otra
función agrega el resultado real SIN tocar la predicción ya guardada.

Doble objetivo: (a) validar que el pipeline corre en producción (no backtest);
(b) acumular el dataset 2026 real de (features LLM en tiempo real -> resultado)
para algún día reentrenar el coeficiente del LLM en el meta-modelo con señal
genuina, en vez de los ceros forzados que dejó el backtest retroactivo.

Uso:
  # pre-registrar (antes del partido):
  python scripts/log_friendlies.py --predict "España" "Alemania" 2026-03-25
  python scripts/log_friendlies.py --from-csv files/f0_raw/amistosos.csv
  # cargar el resultado (después del partido), sin tocar la predicción:
  python scripts/log_friendlies.py --result "España" "Alemania" 2026-03-25 2-1
  python scripts/log_friendlies.py --list
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

LOG = ROOT / "results/live_log/amistosos_2026.csv"

# columnas de PREDICCIÓN (pre-registro, NUNCA se sobreescriben tras guardarse)
PRED_COLS = [
    "match_id", "predicted_at", "date", "home", "away",
    "p_home", "p_draw", "p_away", "pred_result", "pred_score",
    "lambda_home", "lambda_away",
    # features del LLM USADAS (post-validación de nómina) — clave para reentrenar
    "llm_source", "llm_discarded_n",
    "home_lesionados_n", "away_lesionados_n",
    "home_cambio_dt", "away_cambio_dt", "dead_rubber",
    "home_consenso_pct", "away_consenso_pct",
    "home_fatiga_husos", "away_fatiga_husos",
    # F. sintetizador LLM acotado: predicción AJUSTADA junto a la sin ajustar
    "synth_accion", "synth_magnitud", "synth_clamped", "synth_justificacion",
    "p_home_adj", "p_draw_adj", "p_away_adj", "pred_result_adj",
    # B. CLV: cuota de apertura capturada al predecir (o 'pending' si no hay fuente)
    "clv_open_source", "clv_cuota_apertura",
    "model_version",
]
# columnas de RESULTADO (se completan después del partido)
RESULT_COLS = ["status", "actual_home_goals", "actual_away_goals",
               "actual_result", "points_5_3_0", "points_5_3_0_adj",
               "result_recorded_at"]
ALL_COLS = PRED_COLS + RESULT_COLS


def _match_id(home: str, away: str, date: str) -> str:
    return f"{str(date).strip()}|{home.strip()}|{away.strip()}"


def _load() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG, dtype={"match_id": str})
    return pd.DataFrame(columns=ALL_COLS)


def _save(df: pd.DataFrame) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG, index=False)


def _most_likely_score(score_matrix, probs=None) -> str:
    """Marcador modal CONSISTENTE con el 1X2 si se pasan las probs (recomendado);
    si no, cae al argmax global (compatibilidad). El consistente evita el empate
    fantasma cuando el 1X2 favorece a un ganador."""
    if score_matrix is None:
        return ""
    if probs is not None:
        from core.simulation.monte_carlo import consistent_modal_score
        i, j = consistent_modal_score(score_matrix, *probs)
        return f"{i}-{j}"
    i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    return f"{i}-{j}"


# ------------------------------ pre-registro --------------------------------
def log_prediction(home: str, away: str, date: str, *, use_llm: bool = True,
                   use_synth: bool = True, capture_odds: bool = True,
                   overwrite: bool = False) -> dict:
    """Pre-registra la predicción de un amistoso. Idempotente: si ya existe el
    partido NO lo re-predice (preserva la integridad del pre-registro), salvo
    overwrite=True. Devuelve la fila escrita (o la existente).

    `use_synth`: aplica el sintetizador LLM acotado (F) y guarda la predicción
    AJUSTADA junto a la sin ajustar, para compararlas contra el resultado real.
    `capture_odds`: captura la cuota de APERTURA (B) vía scripts/clv_tracking."""
    from ensemble.predict import MODEL_VERSION, predict_final
    from ensemble.roster import validate_features
    from llm_features import MatchFeatures, get_match_features

    df = _load()
    mid = _match_id(home, away, date)
    if not overwrite and (df["match_id"] == mid).any():
        print(f"  ya pre-registrado: {mid} (usa --overwrite para forzar)")
        return df[df["match_id"] == mid].iloc[0].to_dict()

    # 1) features del LLM en tiempo real + validación de nómina (para el log)
    llm_flat = {}
    llm_signals = None                    # dict anidado validado (para el synth)
    discarded = 0
    if use_llm:
        try:
            teams = pd.read_csv(ROOT / "files/f0_raw/teams_2026.csv")["team"].tolist()
            raw = get_match_features(home, away, date)
            clean, disc = validate_features(raw, teams=teams, log=True)
            llm_signals = clean
            llm_flat = MatchFeatures.from_dict(clean).to_flat_dict()
            discarded = sum(1 for d in disc if not d.get("kept", False))
        except Exception as e:
            print(f"  ⚠️ llm_features no disponible ({type(e).__name__}); "
                  "se registra sin señales LLM")

    # 2) predicción final del ensemble (usa la caché del LLM ya poblada)
    pred = predict_final(home, away, date, use_llm=use_llm)

    # 3) F. sintetizador LLM acotado sobre (core, microsim, señales validadas)
    synth = _synth_decision(home, away, date, pred, llm_signals,
                            use_synth=use_synth)

    # 4) B. captura de cuota de apertura (CLV) — nunca rompe el pre-registro
    clv_src, clv_open = "skipped", None
    if capture_odds:
        clv_src, clv_open = _capture_open_odds(
            home, away, date, [pred.prob_home, pred.prob_draw, pred.prob_away])

    row = {
        "match_id": mid, "predicted_at": datetime.now(timezone.utc).isoformat(),
        "date": date, "home": home, "away": away,
        "p_home": round(pred.prob_home, 4), "p_draw": round(pred.prob_draw, 4),
        "p_away": round(pred.prob_away, 4),
        "pred_result": ["1", "X", "2"][int(np.argmax(pred.probs))],
        "pred_score": _most_likely_score(
            pred.score_matrix,
            (pred.prob_home, pred.prob_draw, pred.prob_away)),
        "lambda_home": round(pred.lambda_home, 3),
        "lambda_away": round(pred.lambda_away, 3),
        "llm_source": llm_flat.get("features_source", "none"),
        "llm_discarded_n": discarded,
        "home_lesionados_n": llm_flat.get("home_lesionados_clave_n", 0),
        "away_lesionados_n": llm_flat.get("away_lesionados_clave_n", 0),
        "home_cambio_dt": llm_flat.get("home_cambio_dt_reciente"),
        "away_cambio_dt": llm_flat.get("away_cambio_dt_reciente"),
        "dead_rubber": llm_flat.get("dead_rubber"),
        "home_consenso_pct": llm_flat.get("home_consenso_expertos_pct"),
        "away_consenso_pct": llm_flat.get("away_consenso_expertos_pct"),
        "home_fatiga_husos": llm_flat.get("home_fatiga_husos_horarios"),
        "away_fatiga_husos": llm_flat.get("away_fatiga_husos_horarios"),
        "synth_accion": synth["accion"], "synth_magnitud": synth["magnitud"],
        "synth_clamped": synth["clamped"],
        "synth_justificacion": synth["justificacion"],
        "p_home_adj": synth["p_adj"][0], "p_draw_adj": synth["p_adj"][1],
        "p_away_adj": synth["p_adj"][2], "pred_result_adj": synth["pred_result"],
        "clv_open_source": clv_src, "clv_cuota_apertura": clv_open,
        "model_version": MODEL_VERSION,
        "status": "pending", "actual_home_goals": None,
        "actual_away_goals": None, "actual_result": None,
        "points_5_3_0": None, "points_5_3_0_adj": None,
        "result_recorded_at": None,
    }
    df = df[df["match_id"] != mid]                      # si overwrite, quita la vieja
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    print(f"  ✅ pre-registrado {mid}: {row['p_home']}/{row['p_draw']}/"
          f"{row['p_away']} -> {row['pred_score']} (llm={row['llm_source']}, "
          f"descartados={discarded}, synth={synth['accion']}/{synth['magnitud']:+.2f}"
          f", clv={clv_src})")
    return row


# ----------------------- helpers F (synth) y B (CLV) ------------------------
def _synth_decision(home, away, date, pred, llm_signals, *, use_synth) -> dict:
    """Aplica el sintetizador LLM acotado (F) a la predicción del ensemble.
    Devuelve accion/magnitud/clamped/justificacion + probs ajustadas. La base
    a ajustar es la predicción del ensemble (core+microsim+meta); el synth recibe
    core+microsim+señales validadas. Degrada a 'sin_cambio' ante cualquier fallo."""
    base = [float(pred.prob_home), float(pred.prob_draw), float(pred.prob_away)]
    if not use_synth:
        return {"accion": "skipped", "magnitud": 0.0, "clamped": False,
                "justificacion": "", "p_adj": [round(x, 4) for x in base],
                "pred_result": ["1", "X", "2"][int(np.argmax(base))]}
    try:
        from ensemble.predict import _context
        from synth import apply_adjustment, synthesize
        ctx = _context()
        core_p = ctx.core_probs(home, away)
        micro_p = ctx.micro_probs(home, away)
        d = synthesize(core_p, micro_p, llm_signals)
        adj = apply_adjustment(base, d)
        return {"accion": d.accion, "magnitud": round(d.magnitud, 4),
                "clamped": d.clamped,
                "justificacion": (d.justificacion or "")[:300],
                "p_adj": [round(float(x), 4) for x in adj],
                "pred_result": ["1", "X", "2"][int(np.argmax(adj))]}
    except Exception as e:
        print(f"  ⚠️ sintetizador no disponible ({type(e).__name__}); "
              "se registra sin ajuste")
        return {"accion": "sin_cambio", "magnitud": 0.0, "clamped": False,
                "justificacion": f"error: {type(e).__name__}",
                "p_adj": [round(x, 4) for x in base],
                "pred_result": ["1", "X", "2"][int(np.argmax(base))]}


def _capture_open_odds(home, away, date, model_probs) -> tuple:
    """Captura la cuota de APERTURA (B) en results/clv_tracking.csv. Devuelve
    (source, cuota_apertura_del_pick) o ('error'/'pending', None). Nunca rompe."""
    try:
        from scripts.clv_tracking import record_open
        row = record_open(home, away, date, model_probs)
        return row.get("odds_source", "pending"), row.get("cuota_apertura")
    except Exception as e:
        return f"error:{type(e).__name__}", None


# ------------------------------ cargar resultado ----------------------------
def record_result(home: str, away: str, date: str, home_goals: int,
                  away_goals: int) -> dict:
    """Agrega el resultado real SIN modificar la predicción pre-registrada.
    Falla si el partido no fue pre-registrado antes (no se permite registrar
    'a posteriori' una predicción que no existía)."""
    df = _load()
    mid = _match_id(home, away, date)
    sel = df["match_id"] == mid
    if not sel.any():
        raise ValueError(f"{mid} no está pre-registrado: primero --predict "
                         "ANTES de que se juegue (no se permite post-hoc).")
    i = df[sel].index[0]
    # las columnas de resultado se releen como float64 (NaN); pásalas a object
    # para poder escribir enteros/strings sin conflicto de dtype.
    for col in RESULT_COLS:
        if df[col].dtype != object:
            df[col] = df[col].astype(object)
    hg, ag = int(home_goals), int(away_goals)
    actual = "1" if hg > ag else ("2" if hg < ag else "X")
    pred_result = str(df.at[i, "pred_result"])      # tras roundtrip CSV puede ser int
    pred_score = str(df.at[i, "pred_score"])
    exact = pred_score == f"{hg}-{ag}"
    pts = 5 if exact else (3 if pred_result == actual else 0)
    # puntos de la predicción AJUSTADA por el sintetizador (F.4): mismo score
    # (el synth solo mueve el 1X2), distinto resultado si el argmax cambió.
    pred_result_adj = str(df.at[i, "pred_result_adj"]) \
        if "pred_result_adj" in df.columns and pd.notna(df.at[i, "pred_result_adj"]) \
        else pred_result
    pts_adj = 5 if (exact and pred_result_adj == actual) \
        else (3 if pred_result_adj == actual else 0)
    # SOLO se tocan columnas de resultado; las de predicción quedan intactas
    df.at[i, "status"] = "played"
    df.at[i, "actual_home_goals"] = hg
    df.at[i, "actual_away_goals"] = ag
    df.at[i, "actual_result"] = actual
    df.at[i, "points_5_3_0"] = pts
    df.at[i, "points_5_3_0_adj"] = pts_adj
    df.at[i, "result_recorded_at"] = datetime.now(timezone.utc).isoformat()
    _save(df)
    print(f"  ✅ resultado {mid}: {hg}-{ag} (real {actual}, "
          f"predicho {pred_result}/{pred_score}) -> {pts} pts "
          f"| ajustado {pred_result_adj} -> {pts_adj} pts")
    return df.loc[i].to_dict()


# ----------------------------------- CLI ------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-registro de amistosos 2026")
    ap.add_argument("--predict", nargs=3, metavar=("HOME", "AWAY", "DATE"))
    ap.add_argument("--from-csv", metavar="CSV",
                    help="CSV con columnas home,away,date para pre-registrar en lote")
    ap.add_argument("--result", nargs=4, metavar=("HOME", "AWAY", "DATE", "SCORE"),
                    help="cargar resultado, SCORE como '2-1'")
    ap.add_argument("--no-llm", action="store_true", help="no usar llm_features")
    ap.add_argument("--no-synth", action="store_true",
                    help="no aplicar el sintetizador LLM acotado (F)")
    ap.add_argument("--no-odds", action="store_true",
                    help="no capturar cuota de apertura para CLV (B)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--log", metavar="PATH", help="archivo de log alternativo "
                    "(p.ej. para partidos de Mundial, no amistosos)")
    args = ap.parse_args()

    if args.log:                       # permite separar amistosos vs Mundial
        global LOG
        LOG = Path(args.log)

    if args.predict:
        log_prediction(*args.predict, use_llm=not args.no_llm,
                       use_synth=not args.no_synth, capture_odds=not args.no_odds,
                       overwrite=args.overwrite)
    if args.from_csv:
        rows = pd.read_csv(args.from_csv)
        for r in rows.itertuples():
            log_prediction(str(r.home), str(r.away), str(r.date),
                           use_llm=not args.no_llm, use_synth=not args.no_synth,
                           capture_odds=not args.no_odds, overwrite=args.overwrite)
    if args.result:
        h, a, d, score = args.result
        hg, ag = score.split("-")
        record_result(h, a, d, int(hg), int(ag))
    if args.list or not any([args.predict, args.from_csv, args.result]):
        df = _load()
        if df.empty:
            print("(log vacío — usa --predict para pre-registrar)")
        else:
            cols = ["date", "home", "away", "p_home", "p_draw", "p_away",
                    "pred_score", "llm_source", "status", "points_5_3_0"]
            print(df[cols].to_string(index=False))
            played = df[df["status"] == "played"]
            if len(played):
                print(f"\njugados: {len(played)}  | pts totales: "
                      f"{played['points_5_3_0'].sum():.0f}  | acierto resultado: "
                      f"{(played['points_5_3_0'] >= 3).mean():.0%}")


if __name__ == "__main__":
    main()
