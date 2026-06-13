"""Cuadro de eliminación directa del Mundial 2026 (de la polla, Hoja2).

Formato: 16avos (32 equipos: 12 ganadores + 12 segundos + 8 mejores
terceros) → octavos → cuartos → semis → final (19/07). Los slots de
terceros (ej. "3ABCDF") restringen de qué grupos puede venir el tercero
de cada llave; la asignación se resuelve por backtracking.

Dos usos:
- `deterministic_tournament`: el torneo "más probable" con marcador
  concreto en cada partido (para el README / la polla).
- Las constantes y helpers que usa el Monte Carlo de monte_carlo.py
  para estimar P(campeón) por equipo.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.wc_schema import build_match_features, match_features_frame

# (match_id, slot_a, slot_b, fecha) — slots: 1A/2A, 3XXXXX, W##/L##
ROUND_OF_32 = [
    (73, "2A", "2B", "2026-06-28"),
    (74, "1E", "3ABCDF", "2026-06-29"),
    (75, "1F", "2C", "2026-06-29"),
    (76, "1C", "2F", "2026-06-29"),
    (77, "1I", "3CDFGH", "2026-06-30"),
    (78, "2E", "2I", "2026-06-30"),
    (79, "1A", "3CEFHI", "2026-06-30"),
    (80, "1L", "3EHIJK", "2026-07-01"),
    (81, "1D", "3BEFIJ", "2026-07-01"),
    (82, "1G", "3AEHIJ", "2026-07-01"),
    (83, "2K", "2L", "2026-07-02"),
    (84, "1H", "2J", "2026-07-02"),
    (85, "1B", "3EFGIJ", "2026-07-02"),
    (86, "1J", "2H", "2026-07-03"),
    (87, "1K", "3DEIJL", "2026-07-03"),
    (88, "2D", "2G", "2026-07-03"),
]
ROUND_OF_16 = [
    (89, "W74", "W77", "2026-07-04"), (90, "W73", "W75", "2026-07-04"),
    (91, "W76", "W78", "2026-07-05"), (92, "W79", "W80", "2026-07-05"),
    (93, "W83", "W84", "2026-07-06"), (94, "W81", "W82", "2026-07-06"),
    (95, "W86", "W88", "2026-07-07"), (96, "W85", "W87", "2026-07-07"),
]
QUARTERS = [
    (97, "W89", "W90", "2026-07-09"), (98, "W93", "W94", "2026-07-10"),
    (99, "W91", "W92", "2026-07-11"), (100, "W95", "W96", "2026-07-11"),
]
SEMIS = [(101, "W97", "W98", "2026-07-14"), (102, "W99", "W100", "2026-07-15")]
THIRD_PLACE = (103, "L101", "L102", "2026-07-18")
FINAL = (104, "W101", "W102", "2026-07-19")

KO_ROUNDS = [("Dieciseisavos", ROUND_OF_32), ("Octavos", ROUND_OF_16),
             ("Cuartos", QUARTERS), ("Semifinales", SEMIS),
             ("Tercer puesto", [THIRD_PLACE]), ("FINAL", [FINAL])]

THIRD_SLOTS = {mid: set(sb[1:]) for mid, _, sb, _ in ROUND_OF_32
               if sb.startswith("3")}


def allocate_thirds(qualified_groups: list[str]) -> dict[int, str]:
    """Asigna los 8 terceros clasificados (letras de grupo, mejor
    primero) a los 8 slots, respetando las restricciones; backtracking
    priorizando el slot más restringido."""
    slots = sorted(THIRD_SLOTS,
                   key=lambda m: len(THIRD_SLOTS[m] & set(qualified_groups)))
    assign: dict[int, str] = {}

    def bt(k: int, remaining: set[str]) -> bool:
        if k == len(slots):
            return True
        mid = slots[k]
        for g in sorted(THIRD_SLOTS[mid] & remaining):
            assign[mid] = g
            if bt(k + 1, remaining - {g}):
                return True
            del assign[mid]
        return False

    if not bt(0, set(qualified_groups)):
        raise RuntimeError(
            f"Sin asignación válida de terceros para {qualified_groups}")
    return assign


# ---------------------------------------------------------------------
# Torneo determinista (el camino más probable, con goles)
# ---------------------------------------------------------------------

@dataclass
class KoMatch:
    round_name: str
    match_id: int
    date: str
    team_a: str
    team_b: str
    score: str          # "2-1" o "1-1 (pen)"
    winner: str
    p_win_a: float
    p_draw: float
    p_win_b: float


def _group_tables_from_predictions(matches: pd.DataFrame,
                                   standings: pd.DataFrame) -> tuple[dict, list, dict]:
    """Tablas de grupo aplicando los marcadores pronosticados.

    Devuelve (slotmap 1X/2X -> equipo, terceros clasificados [grupo
    mejor primero], tablas por grupo para el reporte)."""
    exp_pts = standings.set_index("team")["exp_points"]
    slotmap: dict[str, str] = {}
    thirds: list[tuple] = []
    tables: dict[str, pd.DataFrame] = {}
    for g, gdf in matches.groupby("group"):
        stats = {t: [0, 0, 0] for t in
                 set(gdf.team_a) | set(gdf.team_b)}  # pts, gf, ga
        for fx in gdf.itertuples():
            ga, gb = map(int, fx.pred_score.split("-"))
            stats[fx.team_a][1] += ga; stats[fx.team_a][2] += gb
            stats[fx.team_b][1] += gb; stats[fx.team_b][2] += ga
            if ga > gb:
                stats[fx.team_a][0] += 3
            elif gb > ga:
                stats[fx.team_b][0] += 3
            else:
                stats[fx.team_a][0] += 1; stats[fx.team_b][0] += 1
        order = sorted(stats,
                       key=lambda t: (stats[t][0], stats[t][1] - stats[t][2],
                                      stats[t][1], exp_pts[t]),
                       reverse=True)
        slotmap["1" + g] = order[0]
        slotmap["2" + g] = order[1]
        t3 = order[2]
        thirds.append((g, t3, stats[t3][0], stats[t3][1] - stats[t3][2],
                       stats[t3][1], exp_pts[t3]))
        tables[g] = pd.DataFrame(
            [{"team": t, "pts": stats[t][0], "gf": stats[t][1],
              "ga": stats[t][2]} for t in order])
    thirds.sort(key=lambda x: x[2:], reverse=True)
    qualified = [x[0] for x in thirds[:8]]
    third_team = {g: t for g, t, *_ in thirds}
    return slotmap, qualified, {"tables": tables, "third_team": third_team}


def _predict_ko_match(sim, team_a: str, team_b: str) -> tuple[str, str, float, float, float]:
    """(ganador, marcador, p_a, p_x, p_b) del cruce, vía el modelo."""
    from .monte_carlo import _dixon_coles_matrix
    a = sim.teams.loc[sim._idx[team_a]]
    b = sim.teams.loc[sim._idx[team_b]]
    lams = sim.model.predict_lambda(match_features_frame([
        build_match_features(a, b, 0), build_match_features(b, a, 0)]))
    m = _dixon_coles_matrix(float(lams[0]), float(lams[1]), sim.rho)
    p_a = float(np.tril(m, -1).sum())
    p_x = float(np.trace(m))
    p_b = float(np.triu(m, 1).sum())
    if p_x >= max(p_a, p_b):           # empate como desenlace modal
        diag = np.diag(m)
        k = int(diag.argmax())
        winner = team_a if p_a >= p_b else team_b
        return winner, f"{k}-{k} (pen)", p_a, p_x, p_b
    if p_a >= p_b:                      # gana A en 90'
        mask = np.tril(m, -1)
    else:
        mask = np.triu(m, 1)
    i, j = np.unravel_index(mask.argmax(), mask.shape)
    winner = team_a if p_a >= p_b else team_b
    return winner, f"{i}-{j}", p_a, p_x, p_b


def deterministic_tournament(sim, matches: pd.DataFrame,
                             standings: pd.DataFrame) -> dict:
    """Camino más probable del torneo completo, con goles en cada
    partido. Devuelve {group_tables, third_team, ko_matches, champion}."""
    slotmap, qualified, extra = _group_tables_from_predictions(
        matches, standings)
    alloc = allocate_thirds(qualified)
    third_team = extra["third_team"]

    resolved: dict[str, str] = dict(slotmap)
    winners: dict[str, str] = {}
    losers: dict[str, str] = {}
    ko_matches: list[KoMatch] = []

    def resolve(slot: str, mid: int) -> str:
        if slot.startswith("W"):
            return winners[slot[1:]]
        if slot.startswith("L"):
            return losers[slot[1:]]
        if slot.startswith("3"):
            return third_team[alloc[mid]]
        return resolved[slot]

    for round_name, matches_list in KO_ROUNDS:
        for mid, sa, sb, date in matches_list:
            ta, tb = resolve(sa, mid), resolve(sb, mid)
            winner, score, p_a, p_x, p_b = _predict_ko_match(sim, ta, tb)
            winners[str(mid)] = winner
            losers[str(mid)] = tb if winner == ta else ta
            ko_matches.append(KoMatch(round_name, mid, date, ta, tb,
                                      score, winner, p_a, p_x, p_b))

    return {
        "group_tables": extra["tables"],
        "third_team": third_team,
        "thirds_qualified": qualified,
        "ko_matches": ko_matches,
        "champion": winners["104"],
    }
