"""¿La microsimulación mejora la polla en la fase de grupos?

Para cada uno de los 72 partidos:
  - "actual"    = marcador de tu modelo de producción (Poisson+DC+mercado),
                  leído de files/f3_output/match_predictions.csv
  - "+microsim" = blend de las λ del modelo con las λ de la microsim 11v11,
                  donde el XI de cada país se ARMA AUTOMÁTICAMENTE con los
                  mejores jugadores del dataset FIFA-24 (no hay alineaciones
                  reales por partido). Peso de la micro = MICRO_W.
  - "real"      = resultado real (parseado del scorecard del README)
Y calcula los puntos de la polla (5 exacto / 3 resultado / 0) de cada enfoque
sobre los partidos ya jugados, para ver si la micro SUMA o no.

Uso:  python scripts/microsim_groupstage.py [--micro-w 0.3]
Salida: results/reports/microsim_vs_modelo.md + resumen por consola.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.data.fifa_ratings import NPXG_CAL, POS_SCALE, _f, _norm
from core.data.historical import NAME_MAP
from core.simulation.match_engine import Player, Squad, simulate_match

FIFA = pd.read_csv(ROOT / "files/cache/fifa24_players.csv")
PRED = pd.read_csv(ROOT / "files/f3_output/match_predictions.csv")
COUNTRY_ALIAS = {   # nombre dataset (EN martj42) -> país tal cual en FIFA-24
    "South Korea": "Korea Republic",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
}   # el resto (United States 113, Ivory Coast 34, etc.) ya coinciden por nombre
FORMATION = [("GK", "GK"), ("CB", "def"), ("CB", "def"), ("FB", "def"),
             ("FB", "def"), ("MF", "mid"), ("MF", "mid"), ("MF", "mid"),
             ("FW", "atk"), ("WF", "atk"), ("WF", "atk")]

# índice de jugadores por país normalizado
FIFA["_c"] = FIFA["country"].map(_norm)
_BY_COUNTRY = {c: g for c, g in FIFA.groupby("_c")}


def _team_to_fifa(team_es: str) -> str:
    en = NAME_MAP.get(team_es, team_es)
    return _norm(COUNTRY_ALIAS.get(en, en))


def _scores(row):
    atk = np.nanmean([_f(row.finishing), _f(row.shot_power), _f(row.att_position)])
    dfn = np.nanmean([_f(row.stand_tackle), _f(row.marking), _f(row.interceptions)])
    gk = np.nanmean([_f(row.gk_reflexes), _f(row.gk_diving), _f(row.gk_handling),
                     _f(row.gk_positioning)])
    ovr = np.nanmean([_f(row.reactions), _f(row.ball_control), _f(row.short_pass),
                      _f(row.sprint_speed), _f(row.strength), atk, dfn])
    return atk, dfn, gk, ovr


def select_xi(team_es: str):
    """XI automático del país desde FIFA (o None si no hay datos)."""
    g = _BY_COUNTRY.get(_team_to_fifa(team_es))
    if g is None or len(g) < 11:
        return None
    rows = []
    for r in g.itertuples():
        atk, dfn, gk, ovr = _scores(r)
        rows.append({"row": r, "atk": atk, "dfn": dfn, "gk": gk, "ovr": ovr})
    gk_p = max(rows, key=lambda x: (x["gk"] if not np.isnan(x["gk"]) else 0))
    pool = sorted([x for x in rows if x is not gk_p],
                  key=lambda x: (x["ovr"] if not np.isnan(x["ovr"]) else 0),
                  reverse=True)[:16]
    fwd = sorted(pool, key=lambda x: (x["atk"] if not np.isnan(x["atk"]) else 0),
                 reverse=True)[:3]
    rem = [x for x in pool if x not in fwd]
    dfd = sorted(rem, key=lambda x: (x["dfn"] if not np.isnan(x["dfn"]) else 0),
                 reverse=True)[:4]
    mid = [x for x in rem if x not in dfd][:3]
    chosen = [gk_p] + dfd + mid + fwd          # orden no importa; asigno por FORMATION
    buckets = {"GK": [gk_p], "def": dfd, "mid": mid, "atk": fwd}
    players = []
    used = {"GK": 0, "def": 0, "mid": 0, "atk": 0}
    for pos, bucket in FORMATION:
        cand = buckets[bucket][used[bucket]]
        used[bucket] += 1
        r = cand["row"]
        scale = POS_SCALE.get(pos, 0.45)
        atk = cand["atk"]
        npxg = float(np.clip((atk / 99.) ** 1.5 * scale * NPXG_CAL, 0.02, 1.0)) \
            if not np.isnan(atk) else 0.0
        save = float(np.clip(0.60 + 0.20 * (cand["gk"] / 99.), 0.58, 0.82)) \
            if (pos == "GK" and not np.isnan(cand["gk"])) else 0.72
        dfn = cand["dfn"]
        press = float(np.clip(4 + 8 * (dfn / 99.), 3, 13)) if not np.isnan(dfn) else 0.0
        players.append(Player(name=str(r.player), position=pos, npxg_p90=npxg,
                              pressures_p90=press, psxg_save_pct=save))
    return Squad(team_es, players)


def parse_actuals() -> dict:
    rx = re.compile(r"^\|\s*\d\d/\d\d\s*\|\s*[A-L]\s*\|\s*(.+?)\s*\|\s*\d+-\d+\s*"
                    r"\|\s*(\d+-\d+)\s*\|")
    out = {}
    for ln in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        m = rx.match(ln)
        if m and "–" in m.group(1):
            h, a = [s.strip() for s in m.group(1).split("–")]
            out[(h, a)] = m.group(2)
    return out


def pts(pred, real):
    if not real:
        return None
    if pred == real:
        return 5
    pa, pb = map(int, pred.split("-")); ra, rb = map(int, real.split("-"))
    return 3 if ((pa > pb) - (pa < pb)) == ((ra > rb) - (ra < rb)) else 0


def rhu(x):
    return int(np.floor(x + 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro-w", type=float, default=0.3,
                    help="peso de la microsim al blendear las λ (default 0.3)")
    args = ap.parse_args()
    w = args.micro_w
    actuals = parse_actuals()
    squads = {}

    rows, miss = [], set()
    tot_cur = tot_mic = exact_cur = exact_mic = played = 0
    for r in PRED.itertuples():
        a, b = r.team_a, r.team_b
        cur = r.pred_score
        sa = squads.setdefault(a, select_xi(a))
        sb = squads.setdefault(b, select_xi(b))
        if sa is None or sb is None:
            miss.add(a if sa is None else b)
            mic = cur                                   # sin datos: cae al modelo
        else:
            res = simulate_match(sa, sb, n_sims=3000, is_neutral=True)
            la = (1 - w) * r.exp_goals_a + w * res["lambda_home"]
            lb = (1 - w) * r.exp_goals_b + w * res["lambda_away"]
            mic = f"{rhu(la)}-{rhu(lb)}"
        real = actuals.get((a, b), "")
        pc, pm = pts(cur, real), pts(mic, real)
        if real:
            played += 1; tot_cur += pc; tot_mic += pm
            exact_cur += (cur == real); exact_mic += (mic == real)
        rows.append((r.group, a, b, cur, mic, real, pc, pm))

    # ---- reporte ----
    out = ["# 🤖 Microsim vs Modelo — fase de grupos\n",
           f"XI automático desde FIFA-24 · peso microsim = {w:.0%} · "
           f"micro = blend de λ(modelo) y λ(microsim).\n",
           "| Grp | Partido | Modelo | +Microsim | Real | Pts mod | Pts mic |",
           "|:-:|---|:-:|:-:|:-:|:-:|:-:|"]
    for g, a, b, cur, mic, real, pc, pm in rows:
        flag = "" if cur == mic else " ⚠️"
        out.append(f"| {g} | {a} – {b} | {cur} | {mic}{flag} | {real or '—'} "
                   f"| {pc if pc is not None else '—'} "
                   f"| {pm if pm is not None else '—'} |")
    out.append(f"\n**Jugados: {played}** · "
               f"Puntos MODELO: **{tot_cur}** (exactos {exact_cur}) · "
               f"Puntos +MICROSIM: **{tot_mic}** (exactos {exact_mic})")
    diff = tot_mic - tot_cur
    out.append(f"\n**Veredicto:** la microsim {'SUMA' if diff>0 else ('EMPATA' if diff==0 else 'RESTA')} "
               f"{diff:+d} puntos vs el modelo solo.")
    if miss:
        out.append(f"\n_Países sin datos FIFA (micro = modelo): {sorted(miss)}_")
    path = ROOT / "results/reports/microsim_vs_modelo.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[-6:]))
    print(f"\nReporte: {path}")


if __name__ == "__main__":
    main()
