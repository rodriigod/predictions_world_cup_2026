#!/usr/bin/env python3
"""Análisis pre-partido con microsimulación 11v11 (flujo V3, independiente).

Toma las alineaciones reales, baja stats de FBref, simula el partido y lo
blendea con la probabilidad base del modelo histórico (y el mercado si hay).

Uso:
  python scripts/pre_match.py                                  # interactivo
  python scripts/pre_match.py --lineup files/lineups/arg_esp.json
  python scripts/pre_match.py --lineup ... --alpha 0.5         # más peso a la micro
  python scripts/pre_match.py --lineup ... --alpha 0.0         # solo micro
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.match_engine import (POSITION_KEYS, Squad,
                                         blend_predictions, simulate_match)


def _stats_fn(source: str):
    """Selecciona la fuente de stats por jugador."""
    if source == "fifa":
        from src.data.fifa_ratings import get_squad_stats_fifa
        return lambda lineup, delay: get_squad_stats_fifa(lineup)
    from src.data.fbref_scraper import get_squad_stats
    return get_squad_stats

PRED_CSV = ROOT / "files/f3_output/match_predictions.csv"
ODDS_CSV = ROOT / "files/f0_raw/odds_2026.csv"


def load_historical_probs(home: str, away: str):
    """Prob base del modelo histórico desde match_predictions.csv (columnas
    reales team_a/team_b/p_win_a/p_draw/p_win_b). Devuelve [pH,pX,pA] o None."""
    if not PRED_CSV.exists():
        return None
    import pandas as pd
    df = pd.read_csv(PRED_CSV)
    m = df[(df["team_a"].str.lower() == home.lower())
           & (df["team_b"].str.lower() == away.lower())]
    if m.empty:   # probar orientación invertida
        m = df[(df["team_a"].str.lower() == away.lower())
               & (df["team_b"].str.lower() == home.lower())]
        if not m.empty:
            r = m.iloc[0]
            return [float(r["p_win_b"]), float(r["p_draw"]), float(r["p_win_a"])]
        return None
    r = m.iloc[0]
    return [float(r["p_win_a"]), float(r["p_draw"]), float(r["p_win_b"])]


def load_market_probs(home: str, away: str):
    """Odds de mercado desde odds_2026.csv (si existen) -> [pH,pX,pA] sin margen."""
    if not ODDS_CSV.exists():
        return None
    import pandas as pd
    df = pd.read_csv(ODDS_CSV)
    m = df[(df["home_team"].str.lower() == home.lower())
           & (df["away_team"].str.lower() == away.lower())]
    if m.empty:
        return None
    r = m.iloc[0]
    try:
        raw = [1 / float(r["odds_home"]), 1 / float(r["odds_draw"]),
               1 / float(r["odds_away"])]
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    s = sum(raw)
    return [x / s for x in raw]


def interactive_lineup(team: str) -> list:
    print(f"\n{'='*50}\nAlineación de {team}\n{'='*50}")
    print(f"Posiciones: {', '.join(sorted(POSITION_KEYS))}")
    print("Formato por línea: Nombre, POS   (ej: Lionel Messi, AM)\n")
    lineup = []
    for i in range(11):
        while True:
            entry = input(f"  Jugador {i+1:2d}: ").strip()
            if "," not in entry:
                print("    Formato: Nombre, POS"); continue
            name, pos = [x.strip() for x in entry.split(",", 1)]
            pos = pos.upper()
            if pos not in POSITION_KEYS:
                print(f"    Posición inválida. Usa: {', '.join(sorted(POSITION_KEYS))}")
                continue
            lineup.append({"name": name, "position": pos}); break
    return lineup


def run(args):
    p_market = None
    if args.lineup:
        data = json.loads(Path(args.lineup).read_text(encoding="utf-8"))
        home, away = data["home_team"], data["away_team"]
        is_neutral = data.get("is_neutral", True)
        home_lineup, away_lineup = data["home_lineup"], data["away_lineup"]
        if data.get("home_odds") and data.get("draw_odds") and data.get("away_odds"):
            raw = [1 / data["home_odds"], 1 / data["draw_odds"], 1 / data["away_odds"]]
            p_market = [x / sum(raw) for x in raw]
    else:
        print("\n⚽ ANÁLISIS PRE-PARTIDO — Microsimulación 11v11")
        home = input("\nEquipo local/A: ").strip()
        away = input("Equipo visitante/B: ").strip()
        is_neutral = input("¿Campo neutral? (s/n): ").lower().startswith("s")
        home_lineup = interactive_lineup(home)
        away_lineup = interactive_lineup(away)

    # mercado: si no vino en el JSON, intentar odds_2026.csv
    if p_market is None:
        p_market = load_market_probs(home, away)
        if p_market:
            print(f"\n💰 Mercado tomado de {ODDS_CSV.name}")

    get_stats = _stats_fn(args.stats)
    print(f"\n📊 {home} (stats: {args.stats})...")
    home_players = get_stats(home_lineup, args.delay)
    print(f"\n📊 {away} (stats: {args.stats})...")
    away_players = get_stats(away_lineup, args.delay)
    hs = Squad(home, home_players); as_ = Squad(away, away_players)

    print(f"\n🎲 Simulando {args.n_sims:,} partidos...")
    micro = simulate_match(hs, as_, n_sims=args.n_sims,
                           noise_sigma=args.noise_sigma, is_neutral=is_neutral)
    p_micro = [micro["p_home_win"], micro["p_draw"], micro["p_away_win"]]

    p_stat = load_historical_probs(home, away)
    if p_stat is None:
        print(f"\n⚠ Sin prob histórica para {home} vs {away}; uso solo micro.")
        p_final, alpha = p_micro, 0.0
    else:
        alpha = args.alpha
        p_final = blend_predictions(p_stat, p_micro, alpha=alpha,
                                    p_market=p_market,
                                    beta=args.beta if p_market else 0.0)

    # ---------- output ----------
    print(f"\n{'='*55}\n  {home.upper()}  vs  {away.upper()}\n{'='*55}")
    if p_stat:
        print(f"  📈 Modelo histórico:   {home} {p_stat[0]*100:.1f}% | "
              f"Empate {p_stat[1]*100:.1f}% | {away} {p_stat[2]*100:.1f}%")
    print(f"  🤖 Microsimulación:    {home} {p_micro[0]*100:.1f}% | "
          f"Empate {p_micro[1]*100:.1f}% | {away} {p_micro[2]*100:.1f}%")
    print(f"     λ: {home} {micro['lambda_home']} | {away} {micro['lambda_away']}")
    if p_stat:
        dh, da = (p_micro[0]-p_stat[0])*100, (p_micro[2]-p_stat[2])*100
        print(f"     Ajuste vs histórico: {home} {dh:+.1f}pp | {away} {da:+.1f}pp")
    if p_market:
        print(f"  💰 Mercado:            {home} {p_market[0]*100:.1f}% | "
              f"Empate {p_market[1]*100:.1f}% | {away} {p_market[2]*100:.1f}%")
    print(f"\n  ✅ PROBABILIDAD FINAL (blend α={alpha:.2f}"
          f"{', β=%.2f' % args.beta if p_market else ''}):")
    print(f"     {home:22s} {p_final[0]*100:.1f}%")
    print(f"     {'Empate':22s} {p_final[1]*100:.1f}%")
    print(f"     {away:22s} {p_final[2]*100:.1f}%")
    ml = micro["most_likely_score"]
    print(f"\n  ⚽ Marcador más probable: {ml[0]}-{ml[1]}")
    print("  Top 5 marcadores:")
    for (gh, ga), f in sorted(micro["score_matrix"].items(), key=lambda x: -x[1])[:5]:
        print(f"     {gh}-{ga}  {f*100:4.1f}%  {'█'*int(f*60)}")
    print(f"\n  📊 Equipos: {home} xG/90={hs.team_xg_p90():.2f} "
          f"pres={hs.team_pressure_p90():.0f} save%={hs.gk_save_rate()*100:.0f}  |  "
          f"{away} xG/90={as_.team_xg_p90():.2f} pres={as_.team_pressure_p90():.0f} "
          f"save%={as_.gk_save_rate()*100:.0f}")
    print(f"{'='*55}\n")

    out_dir = ROOT / "results/pre_match"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = out_dir / f"{ts}_{home}_vs_{away}.json"
    out.write_text(json.dumps({
        "timestamp": ts, "home_team": home, "away_team": away,
        "is_neutral": is_neutral, "alpha": alpha,
        "p_statistical": p_stat, "p_micro": p_micro, "p_market": p_market,
        "p_final": p_final, "micro_details": {
            "lambda_home": micro["lambda_home"], "lambda_away": micro["lambda_away"],
            "most_likely_score": list(ml)},
        "home_lineup": home_lineup, "away_lineup": away_lineup,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Guardado: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pre-partido 11v11")
    ap.add_argument("--lineup", default=None, help="JSON de alineaciones")
    ap.add_argument("--n-sims", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.65,
                    help="peso del modelo estadístico (1=solo stat, 0=solo micro)")
    ap.add_argument("--beta", type=float, default=0.20, help="peso del mercado")
    ap.add_argument("--noise-sigma", type=float, default=0.12)
    ap.add_argument("--delay", type=float, default=6.0,
                    help="segundos entre requests a FBref (rate limit)")
    ap.add_argument("--stats", choices=["fifa", "fbref"], default="fifa",
                    help="fuente de stats por jugador (default fifa: ratings "
                         "FIFA-24 accesibles; fbref suele dar 403)")
    run(ap.parse_args())
