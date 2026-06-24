#!/usr/bin/env python3
"""F. Mantenimiento: agrega un partido y re-corre todo el online learning.

Al agregar un resultado real 2026, recalcula ELO + pi-ratings + Bayesiano +
surprise log y regenera el reporte por ronda. Las cachés se invalidan solas por
el mtime del CSV.

Uso:
  # agregar un partido y recomputar todo:
  python online_learning/update_results.py --add 2026-06-12 España "Cabo Verde" 1 2 group
  # recomputar reportes sin agregar nada:
  python online_learning/update_results.py --rebuild
  # ver el estado actual (ELO/pi/bayes de los equipos con datos):
  python online_learning/update_results.py --state
  # además, anotar prob_core_updated en results/live_log/:
  python online_learning/update_results.py --rebuild --annotate-logs
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_learning.bayes_strength import compute_posteriors
from online_learning.dataset import append_result, load_results
from online_learning.elo_online import elo_movements
from online_learning.pi_online import pi_movements
from online_learning.priors import to_es
from online_learning.report import annotate_live_logs, generate_round_report
from online_learning.surprise import build_surprise_log, detect_favorite_bias


def _recompute(*, annotate: bool) -> None:
    res = load_results()
    print(f"\nResultados 2026 cargados: {len(res)}")
    sl = build_surprise_log()
    if len(sl):
        top = sl.iloc[0]
        print(f"Sorpresa mayor: {top['equipos']} {top['resultado_real']} "
              f"(P_core={top['prob_core_pre']}, surprise={top['surprise_score']:+})")
    bias = detect_favorite_bias()
    if bias["n"]:
        print(f"Sesgo favoritos (n={bias['n']}): core daba "
              f"{bias['pred_mean']:.2f} al favorito, ganó {bias['won_rate']:.2f} "
              f"-> gap {bias['gap']:+.2f}")
    generate_round_report()
    print("Reporte: results/reports/online_learning_round.md")
    print("Surprise log: online_learning/data/surprise_log.csv")
    if annotate:
        touched = annotate_live_logs()
        print(f"Live logs anotados (prob_core_updated): {touched or 'ninguno'}")


def _state() -> None:
    res = load_results()
    teams = sorted(set(res["home_team"]) | set(res["away_team"]))
    post = compute_posteriors()
    elo = {m["team_en"]: m for m in elo_movements()}
    pim = {m["team_en"]: m for m in pi_movements()}
    print(f"\nEstado tras {len(res)} partido(s) — equipos con datos:\n")
    print(f"  {'equipo':>16} {'ELO pre→now (Δ)':>22} {'Δatt':>7} {'Δdfn':>7} "
          f"{'λof post[IC80]':>22}")
    for en in teams:
        e, p = elo[en], pim[en]
        o = post[en]["off"]
        print(f"  {to_es(en):>16} {e['elo_pre']:.0f}→{e['elo_now']:.0f} "
              f"({e['delta']:+.1f})  {p['d_att']:+.3f} {p['d_dfn']:+.3f}  "
              f"{o['mean']:.2f} [{o['lo']:.2f},{o['hi']:.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Online learning — mantenimiento (F)")
    ap.add_argument("--add", nargs="+",
                    metavar="DATE HOME AWAY HG AG [STAGE]",
                    help="agrega un partido: fecha local visita golesL golesV [stage]")
    ap.add_argument("--rebuild", action="store_true",
                    help="recomputa reportes desde el CSV actual")
    ap.add_argument("--state", action="store_true",
                    help="muestra ELO/pi/bayes de los equipos con datos")
    ap.add_argument("--annotate-logs", action="store_true",
                    help="anota prob_core_updated en results/live_log/")
    args = ap.parse_args()

    if args.add:
        if len(args.add) not in (5, 6):
            ap.error("--add requiere: DATE HOME AWAY HG AG [STAGE]")
        date, home, away, hg, ag = args.add[:5]
        stage = args.add[5] if len(args.add) == 6 else "group"
        append_result(date, home, away, int(hg), int(ag), stage)
        print(f"  ✅ agregado {date} {home} {hg}-{ag} {away} ({stage})")
        _recompute(annotate=args.annotate_logs)
    elif args.rebuild:
        _recompute(annotate=args.annotate_logs)
    elif args.state:
        _state()
    else:
        _state()


if __name__ == "__main__":
    main()
