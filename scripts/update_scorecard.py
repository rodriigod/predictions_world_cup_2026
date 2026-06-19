"""Actualiza el scorecard de README.md con resultados REALES.

Rellena las columnas "Actual" y "Pts" de la tabla scorecard usando el
marcador pronosticado que ya está en la columna "Predicted" (no re-predice),
y recalcula el pie "Running total".

Puntaje de la polla: 5 si el marcador exacto coincide, 3 si acierta el
resultado 1X2, 0 si falla.

Uso:
    python scripts/update_scorecard.py --match "Francia vs Senegal" --score 3-1
    python scripts/update_scorecard.py --csv results/played_matches.csv
        # CSV con columnas: home,away,score   (score como "3-1")
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# | 16/06 | I | Francia – Senegal | 1-0 | 3-1 | ✅ 3 |
ROW = re.compile(r"^\|\s*(\d\d/\d\d)\s*\|\s*([A-L])\s*\|\s*(.+?)\s*\|"
                 r"\s*(\d+-\d+)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|$")


def _norm(s: str) -> str:
    return s.replace("–", "-").replace(" vs ", " - ").strip().lower()


def points(pred: str, actual: str) -> int:
    if pred == actual:
        return 5
    pa, pb = map(int, pred.split("-")); aa, ab = map(int, actual.split("-"))
    so = (pa > pb) - (pa < pb); ao = (aa > ab) - (aa < ab)
    return 3 if so == ao else 0


def parse_match(text: str) -> tuple[str, str]:
    t = text.replace("–", "-").replace(" vs ", " - ")
    parts = [p.strip() for p in t.split(" - ")]
    if len(parts) != 2:
        raise SystemExit(f"No pude separar local/visita en: {text!r}")
    return parts[0], parts[1]


def load_updates(args) -> dict:
    """{(home_lower, away_lower): 'g-g'}"""
    ups = {}
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        for r in df.itertuples():
            ups[(_norm(str(r.home)), _norm(str(r.away)))] = str(r.score).strip()
    if args.match:
        if not args.score:
            raise SystemExit("--match requiere --score")
        h, a = parse_match(args.match)
        ups[(_norm(h), _norm(a))] = args.score.strip()
    if not ups:
        raise SystemExit("Nada que actualizar: usa --match/--score o --csv")
    return ups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", help='ej: "Francia vs Senegal"')
    ap.add_argument("--score", help='ej: 3-1')
    ap.add_argument("--csv", help="CSV con columnas home,away,score")
    args = ap.parse_args()
    ups = load_updates(args)

    lines = README.read_text(encoding="utf-8").splitlines()
    changed = 0
    out = []
    for ln in lines:
        m = ROW.match(ln)
        if m:
            date, grp, match, pred, actual_old, _pts_old = m.groups()
            h, a = parse_match(match)
            key = (_norm(h), _norm(a))
            if key in ups:
                actual = ups[key]
                p = points(pred, actual)
                mark = "✅" if p > 0 else "❌"
                ln = (f"| {date} | {grp} | {match} | {pred} "
                      f"| {actual} | {mark} {p} |")
                changed += 1
        out.append(ln)

    if not changed:
        print("⚠️  No se encontró ningún partido del scorecard que coincida.")
        sys.exit(1)

    # recomputar el pie "Running total"
    tot = played = exact = outc = 0
    res_row = re.compile(r"^\|\s*\d\d/\d\d\s*\|\s*[A-L]\s*\|\s*.+?\s*\|"
                         r"\s*(\d+-\d+)\s*\|\s*(\d+-\d+)\s*\|.*?(\d+)\s*\|$")
    for ln in out:
        rm = res_row.match(ln)
        if rm:
            pred, actual, pts = rm.group(1), rm.group(2), int(rm.group(3))
            played += 1; tot += pts
            if pred == actual:
                exact += 1
            if pts >= 3:
                outc += 1
    footer = (f"**Running total: {tot} pts** · exact scores: {exact}/{played}"
              f" · outcomes (≥3pts): {outc}/{played} · played: {played}/72")
    out = [footer if ln.startswith("**Running total:") else ln for ln in out]

    README.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"✅ {changed} partido(s) actualizado(s).")
    print(f"   {footer}")


if __name__ == "__main__":
    main()
