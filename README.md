# ⚽ World Cup 2026 — Group Stage & Tournament Predictor

Machine-learning pipeline that predicts **every match of the 2026 FIFA World Cup** — all 72 group-stage games and the full knockout bracket (Round of 32 → Final) — with concrete scorelines, built for a prediction pool ("polla mundialera": 5 pts for the exact score, 3 pts for the correct outcome).

**Method in one line:** Poisson regression of expected goals (λ) trained on **28,500+ real international matches (1995–2026)**, corrected with Dixon-Coles, and run through a **10,000-iteration Monte Carlo simulation** of the whole tournament.

---

## 🧠 How it works

### 1. Data

| Source | What it provides |
|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | ~49,000 international matches (1872–2026). Training uses 1995+ → **28,564 matches / 57,128 team-match rows** |
| [eloratings.net](https://www.eloratings.net) | Live ELO ratings (reference; the pipeline also computes its own ELO) |
| [Transfermarkt](https://www.transfermarkt.com) | Squad market values for the 48 qualified teams |
| `Polla_Mundial_2026_Fase_de_Grupos.pdf` / `Polla Mundial 2026.xlsx` | The official pool: real groups, fixture and knockout bracket |

### 2. Feature engineering (`src/data/historical.py`)

The pipeline replays history match by match and builds **leak-free rolling features** — every match only sees data from *before* it was played:

- **Internal ELO rating**, updated per match (K-factor by tournament importance: World Cup 60, qualifiers/continentals 40, friendlies 20; +100 home bonus; goal-difference multiplier). This captures the latent attack/defence strengths (α, β) of **Maher (1982)**.
- **Rolling form**: points % over the last 5 matches, goals for/against over the last 10 (xG proxy).
- **Context**: home/host flag, real rest-days difference between the teams.
- **ELO win expectancy**: `1 / (1 + 10^(−elo_diff/400))` — the single most predictive feature, per the literature.

The full feature taxonomy (60+ variables incl. climate, altitude, travel fatigue, squad fatigue, betting odds) is documented in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

### 3. Models (`src/models/`)

| Model | Role |
|---|---|
| **Poisson GLM** (log-link) | Main model. Predicts λ (expected goals) per team per match. Generalises Maher's α×β model: λ = exp(β·x) |
| **HistGradientBoosting / XGBoost** (`count:poisson`) | Alternative non-linear backends (`--backend gbm` / `--backend xgb`) |
| **1X2 multiclass classifier** | Cross-check in the modern ML formulation: XGBoost (`multi:softprob`) vs Random Forest vs **multinomial Logistic Regression baseline** |

Two classic corrections from **Dixon & Coles (1997)** are implemented:

1. **Temporal decay (φ):** every training row is weighted `w = exp(−years/8) × tournament_weight`, so recent and important matches dominate the fit.
2. **Low-score adjustment (τ):** independent Poissons under-predict 0-0 and 1-1; the score-matrix probabilities for {0-0, 1-0, 0-1, 1-1} are corrected with ρ = −0.08.

> 📊 An honest finding, consistent with the literature: on these features the **logistic baseline beats XGBoost and Random Forest** at 1X2 log-loss (0.864 vs 0.868 vs 0.873). International football signal is mostly linear in ELO + form — which is why the Poisson GLM (which also yields exact scorelines) drives the predictions.

### 4. Monte Carlo simulation (`src/simulation/`)

The **full tournament is simulated 10,000 times** (~19 s):

- Each match samples a score from its Dixon-Coles probability matrix.
- **Per-iteration stochastic noise** (±10% lognormal on λ) models day-of-match conditions: heat, pitch, fitness.
- **Matchday-3 incentives** are recomputed *inside each iteration* from the live group table: qualified teams rotate (−12% attack), teams that both benefit from a draw play cagey (−8%), must-win teams open up (+6%).
- 2026 format: 12 groups of 4 → top 2 + **8 best third-placed teams** advance to a Round of 32. Third-place slots are allocated to the bracket with constraint-respecting backtracking. FIFA tiebreakers (points → GD → GF → head-to-head → drawing of lots).
- Knockout draws are resolved by relative strength (extra time / penalties proxy).

Outputs: per-team probabilities (win group, advance, reach each round, **win the title**), per-match 1X2 probabilities and the most likely scoreline consistent with the predicted outcome.

---

## 🚀 Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# refresh team data (live ELO download) — optional
python scripts/update_team_data.py

# train + simulate the whole tournament (pre-tournament mode)
python scripts/run_groups_simulation.py --sims 10000
```

Useful flags: `--backend poisson|gbm|xgb` · `--train historical|synthetic` · `--cutoff 2026-06-11` (ignore everything from this date on; `--cutoff none` to include matches already played) · `--no-classifier`.

Outputs land in:

- `results/reports/mi_polla.md` — **the filled-in pool** (72 scorelines)
- `results/reports/prediccion_fase_grupos.md` — group-by-group detail with probabilities
- `results/reports/torneo_completo.md` — full bracket to the champion
- `files/f3_output/*.csv` — raw probabilities

## 📁 Project structure

```
src/
├── data/
│   ├── wc_schema.py      # feature schema (programmatic data dictionary)
│   ├── historical.py     # real-data trainer: ELO replay + rolling features
│   └── synthetic.py      # calibrated synthetic generator (pipeline testing)
├── models/
│   ├── poisson_goals.py  # λ model: Poisson GLM / GBM / XGBoost
│   └── result_classifier.py  # 1X2 multiclass cross-check
└── simulation/
    ├── monte_carlo.py    # group stage + knockout MC, Dixon-Coles, incentives
    ├── knockout.py       # 2026 bracket, third-place allocation, deterministic run
    └── report.py         # markdown/console reports
scripts/
├── update_team_data.py   # refresh ELO (live) + market values
└── run_groups_simulation.py  # end-to-end pipeline
files/f0_raw/             # datasets (results, teams, fixtures)
```

---

# 🏆 PREDICTED RESULTS — FULL TOURNAMENT

*Pre-tournament prediction (trained only on data before 2026-06-11, no played matches used). Scores are the most likely scoreline consistent with the predicted 1X2 outcome. Regenerate with `python scripts/run_groups_simulation.py --sims 10000`.*

## Group stage — all 72 matches

**Bold** = predicted to advance (top 2 + best thirds ✦).

### Group A — México, Corea del Sur, Rep. Checa ✦
| Match | Score |
|---|:-:|
| México – Sudáfrica | **2-0** |
| Corea del Sur – Rep. Checa | **1-0** |
| Rep. Checa – Sudáfrica | **1-0** |
| México – Corea del Sur | **1-0** |
| Rep. Checa – México | **0-1** |
| Sudáfrica – Corea del Sur | **0-2** |

Table: **México 9** · **Corea del Sur 6** · **Rep. Checa 3 ✦** · Sudáfrica 0

### Group B — Canadá, Suiza
| Match | Score |
|---|:-:|
| Canadá – Bosnia y Her. | **2-0** |
| Catar – Suiza | **0-2** |
| Suiza – Bosnia y Her. | **2-0** |
| Canadá – Catar | **2-0** |
| Suiza – Canadá | **0-1** |
| Bosnia y Her. – Catar | **1-0** |

Table: **Canadá 9** · **Suiza 6** · Bosnia y Her. 3 · Catar 0

### Group C — Brasil, Marruecos
| Match | Score |
|---|:-:|
| Brasil – Marruecos | **1-0** |
| Haití – Escocia | **0-1** |
| Escocia – Marruecos | **0-2** |
| Brasil – Haití | **2-0** |
| Escocia – Brasil | **0-2** |
| Marruecos – Haití | **2-0** |

Table: **Brasil 9** · **Marruecos 6** · Escocia 3 · Haití 0

### Group D — Turquía, Australia, EEUU ✦
| Match | Score |
|---|:-:|
| EEUU – Paraguay | **1-0** |
| Australia – Turquía | **0-1** |
| EEUU – Australia | **0-1** |
| Turquía – Paraguay | **1-0** |
| Turquía – EEUU | **2-1** |
| Paraguay – Australia | **0-1** |

Table: **Turquía 9** · **Australia 6** · **EEUU 3 ✦** · Paraguay 0

### Group E — Ecuador, Alemania, Costa de Marfil ✦
| Match | Score |
|---|:-:|
| Alemania – Curazao | **2-0** |
| Costa de Marfil – Ecuador | **0-1** |
| Alemania – Costa de Marfil | **2-0** |
| Ecuador – Curazao | **2-0** |
| Curazao – Costa de Marfil | **0-2** |
| Ecuador – Alemania | **1-0** |

Table: **Ecuador 9** · **Alemania 6** · **Costa de Marfil 3 ✦** · Curazao 0

### Group F — Países Bajos, Japón
| Match | Score |
|---|:-:|
| Países Bajos – Japón | **1-0** |
| Suecia – Túnez | **0-1** |
| Países Bajos – Suecia | **2-0** |
| Túnez – Japón | **0-2** |
| Japón – Suecia | **2-0** |
| Túnez – Países Bajos | **0-2** |

Table: **Países Bajos 9** · **Japón 6** · Túnez 3 · Suecia 0

### Group G — Bélgica, Irán, Egipto ✦
| Match | Score |
|---|:-:|
| Bélgica – Egipto | **1-0** |
| Irán – Nueva Zelanda | **1-0** |
| Bélgica – Irán | **2-1** |
| Nueva Zelanda – Egipto | **0-1** |
| Egipto – Irán | **0-1** |
| Nueva Zelanda – Bélgica | **0-2** |

Table: **Bélgica 9** · **Irán 6** · **Egipto 3 ✦** · Nueva Zelanda 0

### Group H — España, Uruguay, Arabia S. ✦
| Match | Score |
|---|:-:|
| España – Cabo Verde | **3-0** |
| Arabia S. – Uruguay | **0-1** |
| España – Arabia S. | **3-0** |
| Uruguay – Cabo Verde | **2-0** |
| Cabo Verde – Arabia S. | **1-2** |
| Uruguay – España | **0-1** |

Table: **España 9** · **Uruguay 6** · **Arabia S. 3 ✦** · Cabo Verde 0

### Group I — Francia, Noruega, Senegal ✦
| Match | Score |
|---|:-:|
| Francia – Senegal | **1-0** |
| Irak – Noruega | **0-2** |
| Francia – Irak | **2-0** |
| Noruega – Senegal | **1-0** |
| Noruega – Francia | **1-2** |
| Senegal – Irak | **1-0** |

Table: **Francia 9** · **Noruega 6** · **Senegal 3 ✦** · Irak 0

### Group J — Argentina, Argelia, Austria ✦
| Match | Score |
|---|:-:|
| Argentina – Argelia | **2-0** |
| Austria – Jordania | **1-0** |
| Argentina – Austria | **2-0** |
| Jordania – Argelia | **0-1** |
| Argelia – Austria | **1-0** |
| Jordania – Argentina | **0-2** |

Table: **Argentina 9** · **Argelia 6** · **Austria 3 ✦** · Jordania 0

### Group K — Colombia, Portugal
| Match | Score |
|---|:-:|
| Portugal – RD Congo | **2-0** |
| Uzbekistán – Colombia | **0-2** |
| Portugal – Uzbekistán | **2-0** |
| Colombia – RD Congo | **2-0** |
| Colombia – Portugal | **2-1** |
| RD Congo – Uzbekistán | **0-1** |

Table: **Colombia 9** · **Portugal 6** · Uzbekistán 3 · RD Congo 0

### Group L — Inglaterra, Croacia, Panamá ✦
| Match | Score |
|---|:-:|
| Inglaterra – Croacia | **1-0** |
| Ghana – Panamá | **0-1** |
| Inglaterra – Ghana | **2-0** |
| Panamá – Croacia | **0-1** |
| Panamá – Inglaterra | **0-2** |
| Croacia – Ghana | **2-0** |

Table: **Inglaterra 9** · **Croacia 6** · **Panamá 3 ✦** · Ghana 0

## Knockout stage — most likely path

### Round of 32 (Dieciseisavos)
| Date | Match | Score | Winner |
|---|---|:-:|---|
| 28/06 | Corea del Sur – Suiza | 0-1 | **Suiza** |
| 29/06 | Ecuador – Rep. Checa | 1-0 | **Ecuador** |
| 29/06 | Países Bajos – Marruecos | 0-1 | **Marruecos** |
| 29/06 | Brasil – Japón | 1-0 | **Brasil** |
| 30/06 | Francia – EEUU | 1-0 | **Francia** |
| 30/06 | Alemania – Noruega | 1-2 | **Noruega** |
| 30/06 | México – Costa de Marfil | 1-0 | **México** |
| 01/07 | Inglaterra – Arabia S. | 2-0 | **Inglaterra** |
| 01/07 | Turquía – Senegal | 1-0 | **Turquía** |
| 01/07 | Bélgica – Austria | 1-0 | **Bélgica** |
| 02/07 | Portugal – Croacia | 1-0 | **Portugal** |
| 02/07 | España – Argelia | 2-0 | **España** |
| 02/07 | Canadá – Egipto | 1-0 | **Canadá** |
| 03/07 | Argentina – Uruguay | 1-0 | **Argentina** |
| 03/07 | Colombia – Panamá | 2-0 | **Colombia** |
| 03/07 | Australia – Irán | 1-0 | **Australia** |

### Round of 16 (Octavos)
| Date | Match | Score | Winner |
|---|---|:-:|---|
| 04/07 | Ecuador – Francia | 0-1 | **Francia** |
| 04/07 | Suiza – Marruecos | 0-1 | **Marruecos** |
| 05/07 | Brasil – Noruega | 2-1 | **Brasil** |
| 05/07 | México – Inglaterra | 0-1 | **Inglaterra** |
| 06/07 | Portugal – España | 0-2 | **España** |
| 06/07 | Turquía – Bélgica | 1-2 | **Bélgica** |
| 07/07 | Argentina – Australia | 2-0 | **Argentina** |
| 07/07 | Canadá – Colombia | 0-1 | **Colombia** |

### Quarter-finals (Cuartos)
| Date | Match | Score | Winner |
|---|---|:-:|---|
| 09/07 | Francia – Marruecos | 1-0 | **Francia** |
| 10/07 | España – Bélgica | 2-0 | **España** |
| 11/07 | Brasil – Inglaterra | 0-1 | **Inglaterra** |
| 11/07 | Argentina – Colombia | 1-0 | **Argentina** |

### Semi-finals
| Date | Match | Score | Winner |
|---|---|:-:|---|
| 14/07 | Francia – España | 0-1 | **España** |
| 15/07 | Inglaterra – Argentina | 0-1 | **Argentina** |

### Third place — 18/07
Francia **1-0** Inglaterra → **Francia** 🥉

### 🏆 FINAL — 19/07, MetLife Stadium
# España 1-0 Argentina

## Title probabilities (10,000 Monte Carlo simulations)

| Team | Reach R16 | Quarters | Semis | Final | **CHAMPION** |
|---|:-:|:-:|:-:|:-:|:-:|
| España 🇪🇸 | 83% | 69% | 59% | 48% | **34.4%** |
| Argentina 🇦🇷 | 70% | 58% | 44% | 30% | **17.7%** |
| Francia 🇫🇷 | 76% | 54% | 35% | 17% | **8.5%** |
| Inglaterra 🇬🇧 | 74% | 46% | 29% | 16% | **7.8%** |
| México 🇲🇽 | 72% | 40% | 22% | 10% | **4.4%** |
| Colombia 🇨🇴 | 65% | 36% | 19% | 10% | **4.3%** |
| Brasil 🇧🇷 | 62% | 38% | 21% | 9% | **3.8%** |
| Portugal 🇵🇹 | 58% | 28% | 14% | 7% | **2.9%** |
| Bélgica 🇧🇪 | 63% | 34% | 11% | 5% | **1.7%** |
| Noruega 🇳🇴 | 54% | 30% | 14% | 5% | **1.7%** |

---

## 📝 Scorecard — model hits & misses

*(filled in as the real tournament unfolds — predictions above are frozen pre-tournament)*

| Date | Grp | Match | Predicted | Actual | Pts (5 exact / 3 outcome / 0 miss) |
|---|:-:|---|:-:|:-:|:-:|
| 11/06 | A | México – Sudáfrica | 2-0 | 2-0 | ✅ 5 |
| 11/06 | A | Corea del Sur – Rep. Checa | 1-0 | 2-1 | ✅ 3 |
| 12/06 | B | Canadá – Bosnia y Her. | 2-0 | 1-1 | ❌ 0 |
| 12/06 | D | EEUU – Paraguay | 1-0 | 2-0 | ✅ 3 |
| 13/06 | B | Catar – Suiza | 0-2 |  |  |
| 13/06 | C | Brasil – Marruecos | 1-0 |  |  |
| 13/06 | C | Haití – Escocia | 0-1 |  |  |
| 14/06 | D | Australia – Turquía | 0-1 |  |  |
| 14/06 | E | Alemania – Curazao | 2-0 |  |  |
| 14/06 | F | Países Bajos – Japón | 1-0 |  |  |
| 14/06 | E | Costa de Marfil – Ecuador | 0-1 |  |  |
| 14/06 | F | Suecia – Túnez | 0-1 |  |  |
| 15/06 | H | España – Cabo Verde | 3-0 |  |  |
| 15/06 | G | Bélgica – Egipto | 1-0 |  |  |
| 15/06 | H | Arabia S. – Uruguay | 0-1 |  |  |
| 15/06 | G | Irán – Nueva Zelanda | 1-0 |  |  |
| 16/06 | I | Francia – Senegal | 1-0 |  |  |
| 16/06 | I | Irak – Noruega | 0-2 |  |  |
| 16/06 | J | Argentina – Argelia | 2-0 |  |  |
| 17/06 | J | Austria – Jordania | 1-0 |  |  |
| 17/06 | K | Portugal – RD Congo | 2-0 |  |  |
| 17/06 | L | Inglaterra – Croacia | 1-0 |  |  |
| 17/06 | L | Ghana – Panamá | 0-1 |  |  |
| 17/06 | K | Uzbekistán – Colombia | 0-2 |  |  |
| 18/06 | A | Rep. Checa – Sudáfrica | 1-0 |  |  |
| 18/06 | B | Suiza – Bosnia y Her. | 2-0 |  |  |
| 18/06 | B | Canadá – Catar | 2-0 |  |  |
| 18/06 | A | México – Corea del Sur | 1-0 |  |  |
| 19/06 | D | EEUU – Australia | 0-1 |  |  |
| 19/06 | C | Escocia – Marruecos | 0-2 |  |  |
| 19/06 | C | Brasil – Haití | 2-0 |  |  |
| 19/06 | D | Turquía – Paraguay | 1-0 |  |  |
| 20/06 | F | Países Bajos – Suecia | 2-0 |  |  |
| 20/06 | E | Alemania – Costa de Marfil | 2-0 |  |  |
| 20/06 | E | Ecuador – Curazao | 2-0 |  |  |
| 21/06 | F | Túnez – Japón | 0-2 |  |  |
| 21/06 | H | España – Arabia S. | 3-0 |  |  |
| 21/06 | G | Bélgica – Irán | 2-1 |  |  |
| 21/06 | H | Uruguay – Cabo Verde | 2-0 |  |  |
| 21/06 | G | Nueva Zelanda – Egipto | 0-1 |  |  |
| 22/06 | J | Argentina – Austria | 2-0 |  |  |
| 22/06 | I | Francia – Irak | 2-0 |  |  |
| 22/06 | I | Noruega – Senegal | 1-0 |  |  |
| 22/06 | J | Jordania – Argelia | 0-1 |  |  |
| 23/06 | K | Portugal – Uzbekistán | 2-0 |  |  |
| 23/06 | L | Inglaterra – Ghana | 2-0 |  |  |
| 23/06 | L | Panamá – Croacia | 0-1 |  |  |
| 23/06 | K | Colombia – RD Congo | 2-0 |  |  |
| 24/06 | B | Suiza – Canadá | 0-1 |  |  |
| 24/06 | B | Bosnia y Her. – Catar | 1-0 |  |  |
| 24/06 | C | Marruecos – Haití | 2-0 |  |  |
| 24/06 | C | Escocia – Brasil | 0-2 |  |  |
| 24/06 | A | Sudáfrica – Corea del Sur | 0-2 |  |  |
| 24/06 | A | Rep. Checa – México | 0-1 |  |  |
| 25/06 | E | Curazao – Costa de Marfil | 0-2 |  |  |
| 25/06 | E | Ecuador – Alemania | 1-0 |  |  |
| 25/06 | F | Túnez – Países Bajos | 0-2 |  |  |
| 25/06 | F | Japón – Suecia | 2-0 |  |  |
| 25/06 | D | Paraguay – Australia | 0-1 |  |  |
| 25/06 | D | Turquía – EEUU | 2-1 |  |  |
| 26/06 | I | Noruega – Francia | 1-2 |  |  |
| 26/06 | I | Senegal – Irak | 1-0 |  |  |
| 26/06 | H | Uruguay – España | 0-1 |  |  |
| 26/06 | H | Cabo Verde – Arabia S. | 1-2 |  |  |
| 26/06 | G | Nueva Zelanda – Bélgica | 0-2 |  |  |
| 26/06 | G | Egipto – Irán | 0-1 |  |  |
| 27/06 | L | Panamá – Inglaterra | 0-2 |  |  |
| 27/06 | L | Croacia – Ghana | 2-0 |  |  |
| 27/06 | K | Colombia – Portugal | 2-1 |  |  |
| 27/06 | K | RD Congo – Uzbekistán | 0-1 |  |  |
| 27/06 | J | Argelia – Austria | 1-0 |  |  |
| 27/06 | J | Jordania – Argentina | 0-2 |  |  |

**Running total: 11 / 20** · exact scores: 1/4 · outcomes: 3/4 · played: 4/72

---

## ⚠️ Notes & limitations

- The "most likely path" bracket is the modal outcome of each match in sequence — the *probability of this exact bracket happening is tiny* (that's football). The Monte Carlo probabilities are the honest forecast.
- xG features are ELO-derived proxies (no public xG provider for all 48 national teams); squad market value, caps and injuries are neutralised in historical-training mode because they don't exist retroactively. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the full roadmap (betting-odds features are the highest-ROI upgrade).
- To re-predict mid-tournament with played matches locked in: refresh the results dataset and run with `--cutoff none`.

*Built with scikit-learn, XGBoost, pandas and 10,000 parallel universes.* 🎲
