"""Dataset de entrenamiento REAL desde partidos internacionales históricos.

Fuente: files/f0_raw/international_results.csv (github.com/martj42/
international_results, ~49k partidos 1872-2026, incluye el fixture del
Mundial 2026 con los partidos ya jugados).

Qué hace este módulo (papers Maher 1982 / Dixon-Coles 1997):
1. Calcula un ELO propio partido a partido (K según importancia del
   torneo, bonus de localía, multiplicador por goleada) — las "fuerzas
   de ataque/defensa" latentes de Maher quedan capturadas por el ELO y
   las medias móviles.
2. Features rodantes anti-leakage: para cada partido usa SOLO partidos
   anteriores (forma últimos 5, goles a favor/en contra últimos 10 como
   proxy de xG, días de descanso).
3. Decaimiento temporal de Dixon-Coles: cada fila lleva un peso
   w = exp(-años_transcurridos / HALF_LIFE) * peso_torneo, que el
   modelo usa como sample_weight (partidos recientes e importantes
   pesan más).

NOTA de features: valor de mercado, caps, experiencia mundialista y
lesiones no existen históricamente, así que se fijan en neutro (0) en
entrenamiento E inferencia del modo histórico — el modelo real se apoya
en ELO + forma + goles, que es lo que la literatura reporta como las
variables de mayor importancia.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .wc_schema import FEATURE_NAMES, build_match_features

RESULTS_CSV = Path(__file__).resolve().parents[2] / "files/f0_raw/international_results.csv"

TRAIN_FROM = "1995-01-01"   # las filas de entrenamiento parten aquí
WARMUP_FROM = "1980-01-01"  # el ELO/forma se calculan desde aquí
DECAY_HALF_LIFE_YEARS = 3.0  # peso 0.5 a los ~2 años (exp decay /3): prioriza
#                              forma reciente sobre historial largo. El barrido
#                              (experiment_elo.py) mostró que 3-30a es plano en
#                              el RPS del backtest, pero 3a es la convención para
#                              selecciones y no degrada, así que se adopta.

# --- pi-ratings ofensivo/defensivo online (Constantinou & Fenton 2013) ---
# Cada equipo lleva un rating de ATAQUE (att, +=marca más) y DEFENSA (dfn,
# +=concede menos), actualizados partido a partido con el error de goles
# esperados. Capturan ataque y defensa por separado, complementando al ELO
# (que es un escalar de fuerza total).
PI_MU = 0.30          # log de la media de goles de referencia (~1.35)
PI_HOME = 0.20        # bonus ofensivo de localía en escala log
PI_LR = 0.06          # tasa de aprendizaje del update online
PI_CLIP = 1.2         # cota de los ratings para evitar divergencia

# Nombre en español (polla) -> nombre del dataset
NAME_MAP = {
    "México": "Mexico", "Sudáfrica": "South Africa",
    "Corea del Sur": "South Korea", "Rep. Checa": "Czech Republic",
    "Canadá": "Canada", "Bosnia y Her.": "Bosnia and Herzegovina",
    "Catar": "Qatar", "Suiza": "Switzerland", "Brasil": "Brazil",
    "Marruecos": "Morocco", "Haití": "Haiti", "Escocia": "Scotland",
    "EEUU": "United States", "Paraguay": "Paraguay",
    "Australia": "Australia", "Turquía": "Turkey", "Alemania": "Germany",
    "Curazao": "Curaçao", "Costa de Marfil": "Ivory Coast",
    "Ecuador": "Ecuador", "Países Bajos": "Netherlands", "Japón": "Japan",
    "Suecia": "Sweden", "Túnez": "Tunisia", "Bélgica": "Belgium",
    "Egipto": "Egypt", "Irán": "Iran", "Nueva Zelanda": "New Zealand",
    "España": "Spain", "Cabo Verde": "Cape Verde",
    "Arabia S.": "Saudi Arabia", "Uruguay": "Uruguay",
    "Francia": "France", "Senegal": "Senegal", "Irak": "Iraq",
    "Noruega": "Norway", "Argentina": "Argentina", "Argelia": "Algeria",
    "Austria": "Austria", "Jordania": "Jordan", "Portugal": "Portugal",
    "RD Congo": "DR Congo", "Uzbekistán": "Uzbekistan",
    "Colombia": "Colombia", "Inglaterra": "England", "Croacia": "Croatia",
    "Ghana": "Ghana", "Panamá": "Panama",
}


def _k_factor(tournament: str) -> float:
    t = tournament.lower()
    if t == "fifa world cup":
        return 60.0
    if "qualification" in t:
        return 40.0
    if t == "friendly":
        return 20.0
    return 40.0  # continentales, Nations League, etc.


def _tournament_weight(tournament: str) -> float:
    t = tournament.lower()
    if t == "fifa world cup":
        return 1.5
    if t == "friendly":
        return 0.7
    return 1.2


def _goal_mult(diff: int) -> float:
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return 1.75 + (diff - 3) / 8.0


class _TeamHistory:
    __slots__ = ("elo", "att", "dfn", "gf10", "ga10", "pts5", "pts10",
                 "ptsC5", "last_date", "n_matches", "wc_matches_in")

    def __init__(self):
        self.elo = 1500.0
        self.att = 0.0                  # pi-rating ofensivo (log-goles)
        self.dfn = 0.0                  # pi-rating defensivo (+ = concede menos)
        self.gf10: deque = deque(maxlen=10)
        self.ga10: deque = deque(maxlen=10)
        self.pts5: deque = deque(maxlen=5)
        self.pts10: deque = deque(maxlen=10)    # forma/consistencia (cand.)
        self.ptsC5: deque = deque(maxlen=5)     # forma en NO amistosos (cand.)
        self.last_date: pd.Timestamp | None = None
        self.n_matches = 0
        self.wc_matches_in: dict = {}   # año WC -> partidos jugados en él


def _neutral_series(name_es_or_team: str, h: "_TeamHistory",
                    is_host: float = 0.0) -> pd.Series:
    """Serie con TEAM_COLUMNS para build_match_features, con los campos
    sin dato histórico en neutro."""
    return pd.Series({
        "elo": h.elo,
        "pi_attack": h.att,
        "pi_defense": h.dfn,
        "xg_for_last10": float(np.mean(h.gf10)) if h.gf10 else 1.25,
        "xg_against_last10": float(np.mean(h.ga10)) if h.ga10 else 1.25,
        "form_last5_points_pct": (float(np.mean(h.pts5)) / 3.0
                                  if h.pts5 else 0.5),
        "form_last10_points_pct": (float(np.mean(h.pts10)) / 3.0
                                   if h.pts10 else 0.5),
        "form_std10": (float(np.std(h.pts10)) / 3.0
                       if len(h.pts10) >= 3 else 0.0),
        "form_comp5_points_pct": (float(np.mean(h.ptsC5)) / 3.0
                                  if h.ptsC5 else 0.5),
        "market_value_meur": 1.0,            # log(1)=0 -> neutro
        "avg_caps": 0.0,
        "players_with_wc_experience": 0.0,
        "injury_impact_index": 0.0,
        "is_host": is_host,
        "distance_avg_km": 0.0,
    })


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df = df[df["date"] >= WARMUP_FROM].reset_index(drop=True)
    return df


def build_historical_dataset(cutoff: str | None = None, *,
                             home_adv_elo: float = 100.0,
                             k_mult: float = 1.0,
                             half_life: float = DECAY_HALF_LIFE_YEARS,
                             feature_names: list[str] | None = None) -> dict:
    """Recorre la historia una sola vez y devuelve:

    - X, y, w           : filas equipo-partido para el modelo de Poisson
    - X_match, y_result, w_match : filas partido para el clasificador 1X2
    - snapshots         : estado actual (_TeamHistory) por equipo
    - played_wc         : partidos del Mundial 2026 ya jugados

    `cutoff`: si se entrega (ej. "2026-06-11"), los partidos desde esa
    fecha se IGNORAN por completo (ni entrenan, ni actualizan el ELO,
    ni quedan fijos) — simula la foto pre-torneo.

    `home_adv_elo`, `k_mult`, `half_life`: hiperparámetros del ELO/decaimiento
    (defaults = los de producción), expuestos para tuneo leak-free.
    """
    cols = feature_names if feature_names is not None else FEATURE_NAMES
    df = load_results()
    if cutoff is not None:
        df = df[df["date"] < pd.Timestamp(cutoff)].reset_index(drop=True)
    hist: dict[str, _TeamHistory] = {}
    rows, goals, weights, row_dates = [], [], [], []
    m_rows, m_rows_b, m_labels, m_weights, m_dates, m_is_wc = \
        [], [], [], [], [], []
    m_home, m_away = [], []
    played_wc = []

    train_from = pd.Timestamp(TRAIN_FROM)
    t_max = df.loc[df["home_score"].notna(), "date"].max()

    for r in df.itertuples():
        ha = hist.setdefault(r.home_team, _TeamHistory())
        hb = hist.setdefault(r.away_team, _TeamHistory())
        is_wc = r.tournament == "FIFA World Cup"

        if pd.isna(r.home_score):
            continue  # fixture futuro (Mundial 2026): no entrena
        gh, ga_ = int(r.home_score), int(r.away_score)

        if is_wc and r.date >= pd.Timestamp("2026-06-01"):
            played_wc.append((r.home_team, r.away_team, gh, ga_))

        # ---- jornada del grupo en mundiales (1-3; 0 = no aplica) ----
        if is_wc:
            year = r.date.year
            md_a = min(ha.wc_matches_in.get(year, 0) + 1, 3)
            md_b = min(hb.wc_matches_in.get(year, 0) + 1, 3)
            matchday = min(md_a, md_b) if max(md_a, md_b) <= 3 else 0
        else:
            matchday = 0

        # ---- features ANTES de actualizar el estado (anti-leakage) ----
        usable = (r.date >= train_from
                  and ha.n_matches >= 10 and hb.n_matches >= 10)
        if usable:
            host_a = 0.0 if r.neutral else 1.0
            rest_a = (min((r.date - ha.last_date).days, 60)
                      if ha.last_date is not None else 30)
            rest_b = (min((r.date - hb.last_date).days, 60)
                      if hb.last_date is not None else 30)
            sa = _neutral_series(r.home_team, ha, is_host=host_a)
            sb = _neutral_series(r.away_team, hb, is_host=0.0)
            fa = build_match_features(sa, sb, matchday, rest_a - rest_b)
            fb = build_match_features(sb, sa, matchday, rest_b - rest_a)
            w = (np.exp(-(t_max - r.date).days / 365.25 / half_life)
                 * _tournament_weight(r.tournament))
            rows.extend([fa, fb]); goals.extend([gh, ga_])
            weights.extend([w, w])
            row_dates.extend([r.date, r.date])
            m_rows.append(fa)
            m_rows_b.append(fb)
            m_labels.append("1" if gh > ga_ else ("2" if gh < ga_ else "X"))
            m_weights.append(w)
            m_dates.append(r.date)
            m_is_wc.append(is_wc)
            m_home.append(r.home_team)
            m_away.append(r.away_team)

        # ---- actualizar ELO y rodantes ----
        k = k_mult * _k_factor(r.tournament) * _goal_mult(abs(gh - ga_))
        home_adv = 0.0 if r.neutral else home_adv_elo
        we_a = 1.0 / (1.0 + 10 ** (-(ha.elo + home_adv - hb.elo) / 400.0))
        score_a = 1.0 if gh > ga_ else (0.5 if gh == ga_ else 0.0)
        delta = k * (score_a - we_a)
        ha.elo += delta; hb.elo -= delta

        # ---- update online de pi-ratings ataque/defensa ----
        # goles esperados con los ratings ACTUALES (pre-update); el error
        # respecto al marcado real ajusta ataque propio y defensa rival.
        pi_home = 0.0 if r.neutral else PI_HOME
        pred_a = min(np.exp(PI_MU + ha.att - hb.dfn + pi_home), 6.0)
        pred_b = min(np.exp(PI_MU + hb.att - ha.dfn), 6.0)
        err_a, err_b = gh - pred_a, ga_ - pred_b
        ha.att = float(np.clip(ha.att + PI_LR * err_a, -PI_CLIP, PI_CLIP))
        hb.dfn = float(np.clip(hb.dfn - PI_LR * err_a, -PI_CLIP, PI_CLIP))
        hb.att = float(np.clip(hb.att + PI_LR * err_b, -PI_CLIP, PI_CLIP))
        ha.dfn = float(np.clip(ha.dfn - PI_LR * err_b, -PI_CLIP, PI_CLIP))
        ha.gf10.append(gh); ha.ga10.append(ga_)
        hb.gf10.append(ga_); hb.ga10.append(gh)
        pa_pts = 3 if gh > ga_ else (1 if gh == ga_ else 0)
        pb_pts = 3 if ga_ > gh else (1 if gh == ga_ else 0)
        ha.pts5.append(pa_pts); hb.pts5.append(pb_pts)
        ha.pts10.append(pa_pts); hb.pts10.append(pb_pts)
        if r.tournament.lower() != "friendly":        # forma competitiva
            ha.ptsC5.append(pa_pts); hb.ptsC5.append(pb_pts)
        ha.last_date = r.date; hb.last_date = r.date
        ha.n_matches += 1; hb.n_matches += 1
        if is_wc:
            year = r.date.year
            ha.wc_matches_in[year] = ha.wc_matches_in.get(year, 0) + 1
            hb.wc_matches_in[year] = hb.wc_matches_in.get(year, 0) + 1

    return {
        "X": pd.DataFrame(rows, columns=cols),
        "y": pd.Series(goals, name="goals_scored"),
        "w": np.array(weights),
        "row_dates": pd.Series(row_dates, name="date"),
        "X_match": pd.DataFrame(m_rows, columns=cols),
        "X_match_away": pd.DataFrame(m_rows_b, columns=cols),
        "y_result": pd.Series(m_labels, name="result_1x2"),
        "w_match": np.array(m_weights),
        "match_dates": pd.Series(m_dates, name="date"),
        "match_is_wc": np.array(m_is_wc),
        "match_home": pd.Series(m_home, name="home"),
        "match_away": pd.Series(m_away, name="away"),
        "snapshots": hist,
        "played_wc": played_wc,
    }


# ---------------------------------------------------------------------
# Backtesting multi-Mundial (validación con torneos pasados)
# ---------------------------------------------------------------------

# Mundiales con formato comparable (32 equipos) disponibles en el dataset,
# y el día ANTES de su primer partido (cutoff pre-torneo).
WC_BACKTEST_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
WC_START = {
    1998: "1998-06-09", 2002: "2002-05-30", 2006: "2006-06-08",
    2010: "2010-06-10", 2014: "2014-06-11", 2018: "2018-06-13",
    2022: "2022-11-19",
}


def matches_of_wc(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Partidos jugados (con resultado) del Mundial `year`, por fecha."""
    return df[(df["tournament"] == "FIFA World Cup")
              & (df["date"].dt.year == year)
              & df["home_score"].notna()].sort_values("date")


def wc_backtest_rows(year: int, snapshots: dict) -> list[dict]:
    """Filas de features para los partidos del Mundial `year`, usando el
    estado (ELO/forma) CONGELADO justo antes del torneo — exactamente el
    mismo protocolo pre-torneo con el que se predice 2026, sin leakage.

    `snapshots` debe venir de build_historical_dataset(cutoff=WC_START[year]).
    Replica la MISMA fórmula de matchday del entrenamiento (los partidos de
    eliminación quedan como matchday=3, igual que en el fit) para no crear
    desajuste train/test; el campo `stage` separa grupo/eliminación aparte.
    """
    df = load_results()
    test = matches_of_wc(df, year)
    rows: list[dict] = []
    seen: dict[str, int] = {}   # equipo -> partidos jugados en este Mundial
    for r in test.itertuples():
        a, b = r.home_team, r.away_team
        if a not in snapshots or b not in snapshots:
            continue
        ha, hb = snapshots[a], snapshots[b]
        cnt_a, cnt_b = seen.get(a, 0), seen.get(b, 0)
        md_a = min(cnt_a + 1, 3)
        md_b = min(cnt_b + 1, 3)
        matchday = min(md_a, md_b) if max(md_a, md_b) <= 3 else 0
        stage = "knockout" if (cnt_a >= 3 or cnt_b >= 3) else "group"
        host_a = 0.0 if r.neutral else 1.0
        rest_a = (min((r.date - ha.last_date).days, 60)
                  if ha.last_date is not None else 30)
        rest_b = (min((r.date - hb.last_date).days, 60)
                  if hb.last_date is not None else 30)
        sa = _neutral_series(a, ha, is_host=host_a)
        sb = _neutral_series(b, hb, is_host=0.0)
        gh, ga_ = int(r.home_score), int(r.away_score)
        rows.append({
            "year": year, "date": r.date, "home": a, "away": b,
            "stage": stage, "city": getattr(r, "city", ""),
            "country": getattr(r, "country", ""),
            "gh": gh, "ga": ga_,
            "result": "1" if gh > ga_ else ("2" if gh < ga_ else "X"),
            "feat_a": build_match_features(sa, sb, matchday, rest_a - rest_b),
            "feat_b": build_match_features(sb, sa, matchday, rest_b - rest_a),
        })
        seen[a] = cnt_a + 1
        seen[b] = cnt_b + 1
    return rows


def teams_table_from_history(snapshots: dict,
                             teams_csv: pd.DataFrame) -> pd.DataFrame:
    """Tabla de equipos 2026 con el estado calculado desde la historia
    (ELO propio, goles/forma reales) y campos sin histórico en neutro,
    consistente con el entrenamiento."""
    rows = []
    for r in teams_csv.itertuples():
        en = NAME_MAP[r.team]
        h = snapshots[en]
        s = _neutral_series(en, h, is_host=float(r.is_host))
        rows.append({"team": r.team, "group": r.group, "confed": r.confed,
                     **s.to_dict()})
    return pd.DataFrame(rows)


def played_results_es(played_wc: list) -> pd.DataFrame:
    """Resultados ya jugados del Mundial, con nombres en español."""
    inv = {v: k for k, v in NAME_MAP.items()}
    return pd.DataFrame(
        [(inv[a], inv[b], ga, gb) for a, b, ga, gb in played_wc
         if a in inv and b in inv],
        columns=["team_a", "team_b", "goals_a", "goals_b"])
