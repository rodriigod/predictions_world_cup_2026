"""¿La microsimulación 11v11 tiene señal real o es ruido?

Valida el motor de partido (src/simulation/match_engine.py) sobre alineaciones
REALES de los Mundiales 2014/2018/2022 (Fjelstul DB) midiendo si el λ que
produce correlaciona con el resultado real (equipos con mayor λ_micro ¿ganan
más?). NO mide accuracy exacta; mide si hay señal.

Dos escenarios de stats por jugador (FBref está bloqueado, así que probamos):
  A) DEFAULTS por posición — lo que REALMENTE tendrías sin stats: el motor solo
     ve la formación. (Esperado: sin diferenciación -> sin señal.)
  B) PROXY de ataque desde el historial de goles de Mundial (leak-free: solo
     goles ANTES de la fecha del partido) — para ver si, alimentando calidad
     ofensiva real por jugador, el MECANISMO del motor produce señal.

Uso:
  python scripts/validate_microsim.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.simulation.match_engine import Player, Squad, squad_lambdas

CACHE = ROOT / "files/cache"
YEARS = {"WC-2014", "WC-2018", "WC-2022"}
POS = {"GK": "GK", "DF": "CB", "MF": "MF", "FW": "FW"}


def load():
    m = pd.read_csv(CACHE / "wc_matches.csv", parse_dates=["match_date"])
    m = m[m["tournament_id"].isin(YEARS)]
    pa = pd.read_csv(CACHE / "wc_player_appearances.csv", parse_dates=["match_date"])
    g = pd.read_csv(CACHE / "wc_goals.csv", parse_dates=["match_date"])
    return m, pa, g


def goal_proxy_table(goals: pd.DataFrame, apps: pd.DataFrame) -> dict:
    """player_id -> lista ordenada de fechas de gol (no penal, no en contra),
    para contar goles ANTES de cada partido (leak-free)."""
    g = goals[(goals["own_goal"] == 0)].sort_values("match_date")
    by_player: dict = {}
    for r in g.itertuples():
        by_player.setdefault(r.player_id, []).append(r.match_date)
    return by_player


def player_npxg(pid, date, gp: dict) -> float:
    """proxy de npxG/90 ~ goles de Mundial ANTES de esta fecha / 5 (escala),
    cap 0.9. 0 -> el motor usa el default de posición."""
    dates = gp.get(pid, [])
    n = sum(1 for d in dates if d < date)
    return min(n / 5.0, 0.9) if n else 0.0


def build_squad(team, rows, date, gp, use_proxy):
    players = []
    for r in rows.itertuples():
        pos = POS.get(r.position_code, "MF")
        npxg = player_npxg(r.player_id, date, gp) if use_proxy else 0.0
        players.append(Player(name=str(r.player_id), position=pos, npxg_p90=npxg))
    return Squad(team, players)


def evaluate(use_proxy: bool, m, pa, gp) -> dict:
    lam_diff, margin, home_win = [], [], []
    starters = pa[pa["starter"] == 1]
    for mt in m.itertuples():
        hs = starters[(starters["match_id"] == mt.match_id)
                      & (starters["team_id"] == mt.home_team_id)]
        as_ = starters[(starters["match_id"] == mt.match_id)
                       & (starters["team_id"] == mt.away_team_id)]
        if len(hs) < 11 or len(as_) < 11:
            continue
        H = build_squad(mt.home_team_name, hs, mt.match_date, gp, use_proxy)
        A = build_squad(mt.away_team_name, as_, mt.match_date, gp, use_proxy)
        lh, la = squad_lambdas(H, A, is_neutral=True)
        lam_diff.append(lh - la)
        margin.append(mt.home_team_score - mt.away_team_score)
        home_win.append(1 if mt.home_team_score > mt.away_team_score else 0)
    lam_diff, margin, home_win = map(np.array, (lam_diff, margin, home_win))

    nondraw = margin != 0
    dir_acc = float((np.sign(lam_diff[nondraw]) == np.sign(margin[nondraw])).mean())
    pear = float(np.corrcoef(lam_diff, margin)[0, 1]) if lam_diff.std() > 1e-9 else 0.0
    # tasa de victoria local por terciles de lam_diff
    order = np.argsort(lam_diff)
    t = len(order) // 3
    wr_low = home_win[order[:t]].mean()
    wr_high = home_win[order[-t:]].mean()
    return {"n": len(lam_diff), "lam_diff_std": float(lam_diff.std()),
            "pearson": pear, "dir_acc": dir_acc,
            "winrate_low_third": float(wr_low), "winrate_high_third": float(wr_high)}


def main():
    m, pa, gp_raw = load()
    gp = goal_proxy_table(gp_raw, pa)
    print(f"Partidos 2014/2018/2022 con alineaciones completas...\n")
    for label, proxy in [("A) DEFAULTS por posición (sin stats reales)", False),
                         ("B) PROXY de goles WC (leak-free)", True)]:
        r = evaluate(proxy, m, pa, gp)
        print(f"{label}")
        print(f"   n={r['n']}  std(λ_diff)={r['lam_diff_std']:.3f}")
        print(f"   correlación λ_diff vs margen real (Pearson): {r['pearson']:+.3f}")
        print(f"   acierto direccional (no-empates): {r['dir_acc']:.1%}")
        print(f"   win-rate local: tercil λ bajo {r['winrate_low_third']:.1%} "
              f"-> tercil λ alto {r['winrate_high_third']:.1%}\n")
    print("Lectura: Pearson≈0 y win-rates planos = SIN señal. Pearson>0 y "
          "win-rate sube del tercil bajo al alto = el motor SÍ ordena.")


if __name__ == "__main__":
    main()
