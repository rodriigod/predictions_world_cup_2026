"""Pipeline end-to-end: entrena con partidos internacionales REALES
(1995-2026, con decaimiento temporal de Dixon-Coles) y simula la fase
de grupos del Mundial 2026 (fixture de la polla mundialera) con Monte
Carlo. Los partidos ya jugados del Mundial quedan fijos.

Uso:
    python scripts/run_groups_simulation.py [--sims 50000]
        [--backend poisson|gbm|xgb] [--train historical|synthetic]
        [--no-classifier]

Salidas: files/f3_output/*.csv y results/reports/prediccion_fase_grupos.md
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.wc_schema import TEAM_COLUMNS, schema_as_dataframe
from src.models.poisson_goals import PoissonGoalsModel
from src.simulation import (GroupStageSimulator, console_summary,
                            full_report, polla_sheet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50000)
    ap.add_argument("--backend", default="poisson",
                    choices=["poisson", "gbm", "xgb"])
    ap.add_argument("--train", default="historical",
                    choices=["historical", "synthetic"])
    ap.add_argument("--cutoff", default="2026-06-11",
                    help="ignora todo partido desde esta fecha (default: "
                         "inicio del Mundial = predicción pre-torneo). "
                         "Usar --cutoff none para incluir lo ya jugado.")
    ap.add_argument("--no-classifier", action="store_true",
                    help="omite la comparación con el clasificador 1X2")
    ap.add_argument("--half-life", type=float, default=3.0,
                    help="vida media (años) del decaimiento temporal "
                         "Dixon-Coles (default 3.0)")
    ap.add_argument("--noise-sigma", type=float, default=0.10,
                    help="sigma del ruido lognormal por partido en el Monte "
                         "Carlo (componente de 'suerte'; default 0.10)")
    ap.add_argument("--teams",
                    default=str(ROOT / "files/f0_raw/teams_2026.csv"))
    ap.add_argument("--fixtures",
                    default=str(ROOT / "files/f0_raw/fixtures_2026.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / "files/f3_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "results/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    schema_as_dataframe().to_csv(
        ROOT / "files/f2_intermedia/feature_schema.csv", index=False)

    teams_csv = pd.read_csv(args.teams)
    missing = set(TEAM_COLUMNS) - set(teams_csv.columns)
    if missing:
        raise SystemExit(f"Faltan columnas en {args.teams}: {missing}")
    fixtures = pd.read_csv(args.fixtures)
    played = None

    # ---- 1. Datos de entrenamiento ----
    if args.train == "historical":
        from src.data.historical import (build_historical_dataset,
                                         played_results_es,
                                         teams_table_from_history)
        cutoff = None if args.cutoff.lower() == "none" else args.cutoff
        print("[1/5] Construyendo dataset histórico real "
              "(ELO + forma + decaimiento temporal)...")
        if cutoff:
            print(f"      Modo PRE-TORNEO: se ignora todo desde {cutoff}")
        print(f"      Vida media del decaimiento: {args.half_life} años")
        data = build_historical_dataset(cutoff=cutoff,
                                        half_life=args.half_life)
        X, y, w = data["X"], data["y"], data["w"]
        teams = teams_table_from_history(data["snapshots"], teams_csv)
        played = played_results_es(data["played_wc"])
        print(f"      {len(X):,} filas de entrenamiento "
              f"({len(X) // 2:,} partidos reales)")
        if not played.empty:
            print("      Partidos del Mundial YA JUGADOS (quedan fijos):")
            for p in played.itertuples():
                print(f"        {p.team_a} {p.goals_a}-{p.goals_b} {p.team_b}")
    else:
        from src.data.synthetic import make_training_data
        print("[1/5] Generando datos de entrenamiento sintéticos...")
        X, y = make_training_data(n_matches=8000, seed=args.seed)
        w = None
        teams = teams_csv

    # ---- 2. Modelo de lambdas (Poisson) ----
    print(f"[2/5] Entrenando PoissonGoalsModel (backend={args.backend})...")
    model = PoissonGoalsModel(backend=args.backend, random_state=args.seed)
    metrics = model.fit(X, y, sample_weight=w)
    for k, v in metrics.items():
        print(f"      {k}: {v}")
    model.save(str(ROOT / "models" / f"poisson_goals_{args.backend}.pkl"))

    # ---- 3. Clasificador 1X2 + stacking (verificación estilo ML moderno) ----
    if not args.no_classifier and args.train == "historical":
        from src.models.result_classifier import ResultClassifier
        from src.models.stacked_classifier import StackedResultClassifier
        from src.data.wc_schema import build_match_features, match_features_frame
        print("[3/5] Entrenando clasificador 1X2 "
              "(XGBoost vs RF vs logística baseline) + STACKING...")
        clf = ResultClassifier(random_state=args.seed)
        clf_metrics = clf.fit(data["X_match"], data["y_result"],
                              sample_weight=data["w_match"])
        print(clf_metrics.to_string(index=False))
        print(f"      mejor modelo 1X2: {clf.best_name}")
        clf_metrics.to_csv(out_dir / "classifier_comparison.csv", index=False)

        # Stacking out-of-fold (logística+RF+XGBoost -> meta-logística).
        # NOTA HONESTA: el backtest multi-Mundial muestra que el stacking
        # rinde PEOR que Poisson+Dixon-Coles en TODAS las métricas (los
        # modelos base comparten señal lineal en ELO -> muy correlacionados).
        # Se ejecuta y reporta por pedido explícito, pero NO mueve la
        # predicción de marcadores: esa la sigue dando el modelo de goles.
        print("      Entrenando STACKING (se reporta como verificación; "
              "el backtest lo marca peor que Poisson+DC)...")
        stack = StackedResultClassifier(random_state=args.seed).fit(
            data["X_match"], data["y_result"], data["w_match"])
        feat_rows = []
        for fx in fixtures.itertuples():
            ta = teams.loc[teams["team"] == fx.team_a].iloc[0]
            tb = teams.loc[teams["team"] == fx.team_b].iloc[0]
            feat_rows.append(build_match_features(ta, tb, int(fx.matchday)))
        stack_p = stack.predict_proba_1x2(match_features_frame(feat_rows))
        stack_out = fixtures[["group", "matchday", "team_a", "team_b"]].copy()
        stack_out[["p_win_a_stack", "p_draw_stack", "p_win_b_stack"]] = \
            stack_p.round(4).to_numpy()
        stack_out.to_csv(out_dir / "stacking_1x2_predictions.csv", index=False)
        print(f"      stacking 1X2 -> {out_dir}/stacking_1x2_predictions.csv")
    else:
        print("[3/5] (clasificador 1X2 omitido)")

    # ---- 4. Monte Carlo (grupos + eliminación directa) ----
    print(f"[4/5] Simulando el torneo COMPLETO {args.sims:,} veces "
          "(grupos + 16avos -> final)...")
    sim = GroupStageSimulator(teams, fixtures, model,
                              lambda_jitter=args.noise_sigma,
                              played_results=played, seed=args.seed)
    standings, match_results = sim.run(n_sims=args.sims, knockout=True)

    # ---- 5. Salidas ----
    print("[5/5] Guardando resultados...")
    standings.to_csv(out_dir / "group_stage_probabilities.csv", index=False)
    match_results.to_csv(out_dir / "match_predictions.csv", index=False)
    report_path = report_dir / "prediccion_fase_grupos.md"
    report_path.write_text(full_report(match_results, standings, args.sims),
                           encoding="utf-8")
    polla_path = report_dir / "mi_polla.md"
    polla_path.write_text(polla_sheet(match_results), encoding="utf-8")

    # torneo determinista completo (camino más probable, con goles)
    from src.simulation.knockout import deterministic_tournament
    tour = deterministic_tournament(sim, match_results, standings)
    bracket_path = report_dir / "torneo_completo.md"
    bracket_path.write_text(
        bracket_markdown(tour, standings), encoding="utf-8")
    print(console_summary(match_results, standings))
    print(f"\n🏆 CAMPEÓN PRONOSTICADO: {tour['champion']}")
    print(f"\nReporte por grupos : {report_path}")
    print(f"Polla para llenar  : {polla_path}")
    print(f"Torneo completo    : {bracket_path}")
    print(f"CSVs               : {out_dir}/")


def bracket_markdown(tour: dict, standings: pd.DataFrame) -> str:
    """Markdown del torneo completo: cuadro con goles + P(campeón)."""
    lines = ["# 🏆 Mundial 2026 — Torneo completo pronosticado", ""]
    lines.append("## Cuadro de eliminación directa (camino más probable)")
    lines.append("")
    cur = None
    for m in tour["ko_matches"]:
        if m.round_name != cur:
            cur = m.round_name
            lines.append(f"### {cur}")
            lines.append("")
        lines.append(
            f"- {m.date} · {m.team_a} **{m.score}** {m.team_b} → "
            f"**{m.winner}** ({m.p_win_a:.0%}/{m.p_draw:.0%}/{m.p_win_b:.0%})")
    lines.append("")
    lines.append(f"## 🏆 Campeón pronosticado: **{tour['champion']}**")
    lines.append("")
    lines.append("## Probabilidades Monte Carlo (top 16)")
    lines.append("")
    top = standings.nlargest(16, "p_champion")
    lines.append("| Equipo | P(octavos) | P(cuartos) | P(semis) | "
                 "P(final) | P(CAMPEÓN) |")
    lines.append("|---|:-:|:-:|:-:|:-:|:-:|")
    for r in top.itertuples():
        lines.append(f"| {r.team} | {r.p_r16:.0%} | {r.p_qf:.0%} "
                     f"| {r.p_sf:.0%} | {r.p_final:.0%} "
                     f"| **{r.p_champion:.1%}** |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
