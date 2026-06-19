"""Genera la sección PREDICTED RESULTS del README desde los artefactos del
run actual (50k). Preserva los resultados REALES ya anotados en el README
(no los inventa) y recalcula los puntos de la polla con las predicciones
nuevas. Salida por stdout."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "files/f3_output/match_predictions.csv")
st = pd.read_csv(ROOT / "files/f3_output/group_stage_probabilities.csv")

# predicción nueva por par de equipos
pred = {(r.team_a, r.team_b): (r.pred_score, r.pred_result)
        for r in m.itertuples()}

# ---- resultados REALES anotados en el README actual (no se inventan) ----
actual = {}
readme = (ROOT / "README.md").read_text(encoding="utf-8")
row_re = re.compile(r"^\|\s*(\d\d/\d\d)\s*\|\s*([A-L])\s*\|\s*(.+?)\s*\|"
                    r"\s*([\d]+-[\d]+)\s*\|\s*([\d]+-[\d]+)?\s*\|")
for ln in readme.splitlines():
    mt = row_re.match(ln)
    if mt and mt.group(5):
        home, away = [s.strip() for s in mt.group(3).split("–")]
        actual[(home, away)] = mt.group(5)


def pts(ps, acs):
    if not acs:
        return None
    if ps == acs:
        return 5
    pa, pb = map(int, ps.split("-")); aa, ab = map(int, acs.split("-"))
    so = (pa > pb) - (pa < pb); ao = (aa > ab) - (aa < ab)
    return 3 if so == ao else 0


out = []
# ====================== GROUP STAGE ======================
out.append("## Group stage — all 72 matches\n")
out.append("**Bold** = predicted scoreline. Tables show the simulated final "
           "standing (top 2 + best thirds ✦ advance).\n")

GROUP_TEAMS = {g: list(dict.fromkeys(
    list(sub["team_a"]) + list(sub["team_b"])))
    for g, sub in m.groupby("group")}

# los 8 mejores terceros (por P(clasificar como tercero) del Monte Carlo)
qual_thirds = set(st.sort_values("p_advance_as_third", ascending=False)
                  .head(8)["team"])

for g in sorted(m["group"].unique()):
    sub = m[m["group"] == g]
    # standings simulados (exp_points redondeo no; usar pred para tabla)
    stand = {t: {"pts": 0, "gf": 0, "ga": 0} for t in GROUP_TEAMS[g]}
    for r in sub.itertuples():
        ga, gb = map(int, r.pred_score.split("-"))
        stand[r.team_a]["gf"] += ga; stand[r.team_a]["ga"] += gb
        stand[r.team_b]["gf"] += gb; stand[r.team_b]["ga"] += ga
        if ga > gb:
            stand[r.team_a]["pts"] += 3
        elif ga < gb:
            stand[r.team_b]["pts"] += 3
        else:
            stand[r.team_a]["pts"] += 1; stand[r.team_b]["pts"] += 1
    order = sorted(GROUP_TEAMS[g], key=lambda t: (
        stand[t]["pts"], stand[t]["gf"] - stand[t]["ga"], stand[t]["gf"]),
        reverse=True)
    def tag(t):
        return f"{t} ✦" if t in qual_thirds else t
    head = ", ".join(tag(t) for t in order[:3] if order.index(t) < 2
                     or t in qual_thirds)
    out.append(f"### Group {g} — {head}")
    out.append("| Match | Score |\n|---|:-:|")
    for r in sub.itertuples():
        out.append(f"| {r.team_a} – {r.team_b} | **{r.pred_score}** |")
    def cell(i, t):
        s = f"{t} {stand[t]['pts']}"
        if i < 2:
            return f"**{s}**"
        if t in qual_thirds:
            return f"**{s} ✦**"
        return s
    line = " · ".join(cell(i, t) for i, t in enumerate(order))
    out.append(f"\nTable: {line}\n")

# ====================== KNOCKOUT ======================
out.append("## Knockout stage — most likely path\n")
tour = (ROOT / "results/reports/torneo_completo.md").read_text(encoding="utf-8")
# reusar el bloque del torneo (ya viene en español, lo dejamos textual)
ko_lines = []
capture = False
for ln in tour.splitlines():
    if ln.startswith("### "):
        capture = True
    if ln.startswith("## 🏆 Campeón"):
        capture = False
    if capture:
        ko_lines.append(ln)
out.append("\n".join(ko_lines))

champ = st.sort_values("p_champion", ascending=False).iloc[0]
out.append(f"\n### 🏆 PREDICTED CHAMPION: **{champ.team}** "
           f"({champ.p_champion:.1%})\n")

# ====================== TITLE PROBABILITIES ======================
out.append("## Title probabilities (50,000 Monte Carlo simulations)\n")
out.append("| Team | Reach R16 | Quarters | Semis | Final | **CHAMPION** |")
out.append("|---|:-:|:-:|:-:|:-:|:-:|")
top = st.sort_values("p_champion", ascending=False).head(12)
for r in top.itertuples():
    out.append(f"| {r.team} | {r.p_r16:.0%} | {r.p_qf:.0%} | {r.p_sf:.0%} "
               f"| {r.p_final:.0%} | **{r.p_champion:.1%}** |")
out.append("")

# ====================== SCORECARD ======================
out.append("---\n\n## 📝 Scorecard — model hits & misses\n")
out.append("*Real results are facts entered as the tournament unfolds; the "
           "Predicted column is the frozen pre-tournament pick of the CURRENT "
           "model (pi-ratings + 3-yr decay, 50k sims). Points recomputed.*\n")
out.append("| Date | Grp | Match | Predicted | Actual | Pts (5/3/0) |")
out.append("|---|:-:|---|:-:|:-:|:-:|")

order_m = m.copy()
order_m["d"] = pd.to_datetime(order_m["date"])
order_m = order_m.sort_values(["d", "time"])
tot = exact_n = exact_d = out_n = out_d = played = 0
for r in order_m.itertuples():
    ps = pred[(r.team_a, r.team_b)][0]
    acs = actual.get((r.team_a, r.team_b), "")
    p = pts(ps, acs)
    dd = pd.to_datetime(r.date).strftime("%d/%m")
    if p is None:
        cell = "  |  "
    else:
        played += 1
        mark = "✅" if p > 0 else "❌"
        cell = f"{acs} | {mark} {p}"
        tot += p
        if ps == acs:
            exact_n += 1
        exact_d += 1
        if p >= 3:
            out_n += 1
        out_d += 1
    out.append(f"| {dd} | {r.group} | {r.team_a} – {r.team_b} "
               f"| {ps} | {cell} |")
out.append(f"\n**Running total: {tot} pts** · exact scores: {exact_n}/{played}"
           f" · outcomes (≥3pts): {out_n}/{played} · played: {played}/72\n")

sys.stdout.write("\n".join(out))
