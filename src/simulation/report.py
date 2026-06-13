"""Reporte legible de la fase de grupos, organizado grupo por grupo."""

import pandas as pd

DAYS = {0: "LUN", 1: "MAR", 2: "MIÉ", 3: "JUE", 4: "VIE", 5: "SÁB", 6: "DOM"}


def _fmt_date(date_str: str, time_str: str) -> str:
    d = pd.Timestamp(date_str)
    return f"{DAYS[d.dayofweek]} {d.strftime('%d/%m')} {time_str}"


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _pred_text(fx) -> str:
    if getattr(fx, "status", "") == "JUGADO":
        return f"✅ JUGADO {fx.pred_score}"
    if fx.pred_result == "1":
        return f"Gana {fx.team_a} {fx.pred_score}"
    if fx.pred_result == "2":
        return f"Gana {fx.team_b} {fx.pred_score}"
    return f"Empate {fx.pred_score}"


def group_report(g: str, matches: pd.DataFrame,
                 standings: pd.DataFrame) -> str:
    """Sección de markdown para un grupo: partidos + tabla esperada."""
    lines = [f"## GRUPO {g}", ""]

    lines.append("### Partidos")
    lines.append("")
    lines.append("| Jornada | Fecha | Partido | 🎯 Pronóstico | "
                 "P(1) | P(X) | P(2) | Goles esperados |")
    lines.append("|:-:|---|---|---|:-:|:-:|:-:|:-:|")
    for fx in matches.itertuples():
        name_a = f"**{fx.team_a}**" if fx.pred_result == "1" else fx.team_a
        name_b = f"**{fx.team_b}**" if fx.pred_result == "2" else fx.team_b
        lines.append(
            f"| J{fx.matchday} | {_fmt_date(fx.date, fx.time)} "
            f"| {name_a} vs {name_b} "
            f"| {_pred_text(fx)} "
            f"| {_fmt_pct(fx.p_win_a)} | {_fmt_pct(fx.p_draw)} "
            f"| {_fmt_pct(fx.p_win_b)} "
            f"| {fx.exp_goals_a:.1f} - {fx.exp_goals_b:.1f} |")
    lines.append("")
    lines.append("P(1) = gana el primer equipo · P(X) = empate · "
                 "P(2) = gana el segundo equipo.")

    lines.append("")
    lines.append("### Tabla esperada del grupo")
    lines.append("")
    lines.append("| Pos | Equipo | Pts esp. | GF-GC esp. | P(1°) | P(2°) | "
                 "P(3° clasif.) | P(clasificar) |")
    lines.append("|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|")
    for pos, row in enumerate(standings.itertuples(), start=1):
        team = f"**{row.team}**" if row.p_advance >= 0.5 else row.team
        lines.append(
            f"| {pos} | {team} | {row.exp_points:.1f} "
            f"| {row.exp_gf:.1f}-{row.exp_ga:.1f} "
            f"| {_fmt_pct(row.p_win_group)} | {_fmt_pct(row.p_runner_up)} "
            f"| {_fmt_pct(row.p_advance_as_third)} "
            f"| **{_fmt_pct(row.p_advance)}** |")
    lines.append("")
    return "\n".join(lines)


def full_report(matches: pd.DataFrame, standings: pd.DataFrame,
                n_sims: int) -> str:
    header = [
        "# 🏆 Mundial 2026 — Predicción de la Fase de Grupos",
        "",
        f"**Método:** Regresión de Poisson (goles esperados λ por equipo) "
        f"+ corrección Dixon-Coles + Simulación de Monte Carlo: "
        f"**{n_sims:,} iteraciones del torneo completo** — cada uno de los "
        f"72 partidos se simula {n_sims:,} veces con perturbación "
        "estocástica de las condiciones del día (clima, estado físico, "
        "campo) e incentivos dinámicos en la jornada 3 (rotaciones de "
        "clasificados, empates que sirven a ambos, urgencias).",
        "",
        "**Clasifican a 16avos:** los 2 primeros de cada grupo + los 8 "
        "mejores terceros de los 12 grupos.",
        "",
        "El 🎯 Pronóstico es el resultado 1X2 más probable con su marcador "
        "más frecuente (útil para llenar la polla). Las probabilidades "
        "muestran cuán confiable es cada pronóstico.",
        "",
    ]
    parts = [
        group_report(g,
                     matches[matches.group == g],
                     standings[standings.group == g])
        for g in sorted(matches["group"].unique())
    ]
    return "\n".join(header) + "\n" + "\n---\n\n".join(parts)


def polla_sheet(matches: pd.DataFrame) -> str:
    """La polla lista para llenar: los 72 marcadores pronosticados,
    en el mismo orden del PDF (por grupo y jornada)."""
    lines = [
        "# ✍️ MI POLLA — Mundial 2026, Fase de Grupos",
        "",
        "Marcador pronosticado por el modelo para cada casilla "
        "(el más probable condicionado al resultado 1X2 predicho).",
        "",
    ]
    for g in sorted(matches["group"].unique()):
        lines.append(f"## Grupo {g}")
        lines.append("")
        for fx in matches[matches.group == g].itertuples():
            ga, gb = fx.pred_score.split("-")
            lines.append(
                f"- {_fmt_date(fx.date, fx.time)} · "
                f"{fx.team_a} **[{ga}] - [{gb}]** {fx.team_b}")
        lines.append("")
    return "\n".join(lines)


def console_summary(matches: pd.DataFrame, standings: pd.DataFrame) -> str:
    """Versión compacta para terminal, grupo por grupo."""
    out = []
    for g in sorted(matches["group"].unique()):
        out.append(f"\n{'=' * 84}\nGRUPO {g}\n{'=' * 84}")
        for fx in matches[matches.group == g].itertuples():
            vs = f"{fx.team_a} vs {fx.team_b}"
            out.append(
                f"  J{fx.matchday} {_fmt_date(fx.date, fx.time):<16} "
                f"{vs:<34} -> {_pred_text(fx):<24} "
                f"({_fmt_pct(fx.p_win_a)}/{_fmt_pct(fx.p_draw)}/"
                f"{_fmt_pct(fx.p_win_b)})")
        out.append("  " + "-" * 80)
        out.append(f"  {'Equipo':<18}{'Pts esp.':>9} {'P(1°)':>7} {'P(2°)':>7} "
                   f"{'P(3°clasif)':>12} {'P(CLASIFICAR)':>14}")
        for row in standings[standings.group == g].itertuples():
            out.append(
                f"  {row.team:<18}{row.exp_points:>9.1f} "
                f"{_fmt_pct(row.p_win_group):>7} {_fmt_pct(row.p_runner_up):>7} "
                f"{_fmt_pct(row.p_advance_as_third):>12} "
                f"{_fmt_pct(row.p_advance):>14}")
    return "\n".join(out)
