# ⚽ World Cup 2026 — Group Stage & Tournament Predictor

Machine-learning pipeline that predicts **every match of the 2026 FIFA World Cup** — all 72 group-stage games and the full knockout bracket (Round of 32 → Final) — with concrete scorelines, built for a prediction pool ("polla mundialera": 5 pts for the exact score, 3 pts for the correct outcome).

**Method in one line:** Poisson regression of expected goals (λ) over ELO + **online pi-ratings (attack/defence)** + form (incl. competitive-only form), trained on **28,500+ real international matches (1995–2026)** with a 3-year recency decay, corrected with Dixon-Coles, calibration-checked, and run through a **50,000-iteration Monte Carlo simulation** of the whole tournament.

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

- **Internal ELO rating**, updated per match (K-factor by tournament importance: World Cup 60, qualifiers/continentals 40, friendlies 20; +100 home bonus; goal-difference multiplier). A single scalar of total strength.
- **Pi-ratings — attack & defence, separated** (**Constantinou & Fenton 2013**): each team carries an online *attack* rating and a *defence* rating, updated match by match from the expected-goals error (`λ = exp(μ + att_own − def_opp + home)`). These split offensive and defensive ability — exactly the latent α/β of **Maher (1982)** — which a single ELO scalar cannot. Implemented in `historical.py` (`PI_LR`, `PI_CLIP`).
- **Rolling form**: points % over the last 5 matches, goals for/against over the last 10 (xG proxy).
- **Competitive form**: points % over the last 5 **non-friendly** matches — friendlies flatter weak teams and mislead plain form. The only Part-2 candidate feature that survived a leak-free test (RPS 0.1982 → 0.1971); momentum, form-10 and form-variance were measured and dropped (no gain). See `scripts/experiment_features.py`.
- **Context**: home/host flag, real rest-days difference between the teams.
- **ELO win expectancy**: `1 / (1 + 10^(−elo_diff/400))` — among the most predictive features, per the literature.

The full feature taxonomy (60+ variables incl. climate, altitude, travel fatigue, squad fatigue, betting odds) is documented in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

### 3. Models (`src/models/`)

| Model | Role |
|---|---|
| **Poisson GLM** (log-link) | Main model. Predicts λ (expected goals) per team per match. Generalises Maher's α×β model: λ = exp(β·x) |
| **HistGradientBoosting / XGBoost** (`count:poisson`) | Alternative non-linear backends (`--backend gbm` / `--backend xgb`) |
| **1X2 multiclass classifier** | Cross-check in the modern ML formulation: XGBoost (`multi:softprob`) vs Random Forest vs **multinomial Logistic Regression baseline** |

Two classic corrections from **Dixon & Coles (1997)** are implemented:

1. **Temporal decay (φ):** every training row is weighted `w = exp(−years/3) × tournament_weight`, so recent and important matches dominate the fit. The half-life was shortened from 8 → **3 years** (the convention for national teams: prioritise recent form over long history).
2. **Low-score adjustment (τ):** independent Poissons under-predict 0-0 and 1-1; the score-matrix probabilities for {0-0, 1-0, 0-1, 1-1} are corrected with ρ = −0.08.

> 📊 An honest finding, consistent with the literature: on these features the **logistic baseline and Random Forest are within ~0.003 log-loss of each other** (≈0.865), and neither meaningfully beats Poisson+DC on the World-Cup backtest. International football signal is mostly linear in ELO + pi-ratings — which is why the Poisson GLM (which also yields exact scorelines) drives the predictions.

### 4. Monte Carlo simulation (`src/simulation/`)

The **full tournament is simulated 50,000 times** (~90 s):

- Each match samples a score from its Dixon-Coles probability matrix. The analytic 1X2 probabilities are also reported directly from that matrix (`p_*_dc` columns): **P(draw) = the trace (diagonal sum) of the score matrix** — the correct way to get a draw probability, not a heuristic.
- **Calibration before sampling:** λ pass through a temperature step (`lam' = mean·(lam/mean)^T`) before the MC. The backtest shows the model is already well calibrated, so the production temperature is **T = 1.0** (identity); the hook is exposed in case re-calibration is ever needed.
- **Per-iteration stochastic noise** (±10% lognormal on λ) models day-of-match conditions: heat, pitch, fitness.
- **Matchday-3 incentives** are recomputed *inside each iteration* from the live group table: qualified teams rotate (−12% attack), teams that both benefit from a draw play cagey (−8%), must-win teams open up (+6%).
- 2026 format: 12 groups of 4 → top 2 + **8 best third-placed teams** advance to a Round of 32. Third-place slots are allocated to the bracket with constraint-respecting backtracking. FIFA tiebreakers (points → GD → GF → head-to-head → drawing of lots).
- Knockout draws go to **extra time** (goals re-sampled at λ×0.33); if still level, a **penalty shootout** that is near a coin-flip (≤±5% lean to the higher-ELO side), reflecting the empirical near-randomness of shootouts rather than full 90' strength.

Outputs: per-team probabilities (win group, advance, reach each round, **win the title**), per-match 1X2 probabilities and the most likely scoreline consistent with the predicted outcome.

### 5. Backtest — does the model actually work? (`scripts/backtest_world_cups.py`)

The method is validated on **448 matches across 7 past World Cups (1998–2022)**. For each tournament every model is trained **only on matches played before it** (same pre-tournament protocol used for 2026), then predicts every match. Run with `python scripts/backtest_world_cups.py`.

Three forecasters are compared head-to-head (lower log-loss/RPS/Brier = better):

| Method | Accuracy | Log-loss | RPS | Brier |
|---|:-:|:-:|:-:|:-:|
| **Poisson + Dixon-Coles** (production) | 54.2% | **0.970** | **0.1971** | **0.574** |
| Logistic 1X2 | **54.9%** | 0.987 | 0.2014 | 0.583 |
| Stacking ensemble (logistic+RF+XGBoost, time-series OOF + isotonic) | 52.5% | 0.994 | 0.2037 | 0.590 |
| Naive baseline (fixed base rates) | — | 1.069 | 0.2325 | — |

All models crush the base-rate baseline, and Poisson+DC is **well calibrated** (predicted P(win) ≈ observed frequency across all bins). Adding pi-ratings, a 3-year decay and **competitive-form** (form in non-friendly matches only) **improved every probabilistic metric** vs the original 8-year/no-pi model (RPS 0.2009 → **0.1971**, log-loss 0.981 → **0.970**, acc 52.5% → **54.2%**). Group vs knockout performance is nearly identical — no degradation in the bracket. The hardest years remain **2002** (acc 43.8%) and **2022**, the upset tournaments — an honest limitation, not a bug.

> 📊 **Why stacking still didn't win (a measured negative result, re-confirmed).** Following the [cuML stacking playbook](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/), an ensemble of logistic + Random Forest + XGBoost is trained with **TimeSeriesSplit out-of-fold predictions (gap=10, no temporal leakage) + per-class isotonic calibration**, and reported in production (`files/f3_output/stacking_1x2_predictions.csv`). On the World-Cup backtest it is **still worse** than plain Poisson+DC on every probabilistic metric (RPS 0.2037 vs **0.1971**) — and slightly worse than the earlier random-KFold stack (0.2033), because the honest time-ordered OOF removes the mild optimism that random folds introduced. Reason: international-football signal is almost linear in ELO/pi-ratings, so the base learners (sharing features) are highly correlated; stacking correlated models adds variance, not skill. It is computed and surfaced by explicit request, but **does not drive the scoreline predictions** — those stay with the goals model.

#### Improvement experiments — all measured, all leak-free

Every idea below was validated on **held-out data the tuning never saw** (chronological split for hyper-params; the 7 World Cups as the final test). None beat the current model — a useful, honest map of dead ends:

| Experiment | Method | Result |
|---|---|:-:|
| **Pi-ratings (attack/defence)** | online α/β ratings added as features (`historical.py`) | **adopted** — RPS 0.2009→0.1982 |
| **Competitive-form feature** | form in non-friendly matches only (`experiment_features.py`) | **adopted** — RPS 0.1982→**0.1971** (only Part-2 candidate that helped) |
| Momentum / form-10 / form-variance | extra rolling features (`experiment_features.py`) | rejected — Δ≈±0.0001, no real signal |
| **Decay half-life 8 → 3 yr** | shorter recency window (`experiment_elo.py`) | **adopted** — neutral-to-slightly-positive; the national-team convention |
| Optuna hyper-param search | TimeSeriesSplit on general matches (`tune_models.py`) | half-life **flat 1.5–8 yr**; XGB/logistic gains on general data don't beat the production driver — defaults kept |
| Host-continent feature | inferred confederation, host-region advantage | RPS 0.2009→0.2011 — no gain (ELO already captures it) |
| Stacking ensemble | OOF logistic+RF+XGBoost (`stacked_classifier.py`) | worse than Poisson+DC on all metrics (re-confirmed: 0.2033 vs 0.1982) |
| Regularisation / rho / temperature | grid-searched on a general validation set | current defaults already optimal; WC-test gains were **overfitting** |
| ELO hyper-parameters | home-advantage, K-factor swept (`experiment_elo.py`) | flat — current convention values are already optimal |
| Confederation draw adjustment | per-confed draw multiplier fit pre-2014, tested out-of-sample (`experiment_draw_adjust.py`) | helps general matches by a hair but **hurts World Cups** — the African/CONMEBOL draw tendency is a regional-qualifier artifact that doesn't transfer to neutral cross-confed WC games |
| **FIFA world ranking** (new data, 1992–2024) | merge-asof join, `experiment_fifa.py` | **zero** marginal value — the internal ELO is a *better* predictor than the official ranking (solo RPS 0.171 vs 0.204) |

The reproducible harnesses are `scripts/experiment_models.py` and `scripts/experiment_fifa.py`. The takeaway is consistent across the experiments: **predictive signal in international football is dominated by team strength (ELO + pi-ratings).** General-match accuracy is **~60%** (the WC's 53.1% is the hard strong-vs-strong subset).

> 💸 **Bookmaker odds — the one real remaining lever, and why it's not here.** Closing 1X2 odds are the strongest single predictor (Groll et al. 2019). We investigated the source the standard advice points to, **Football-Data.co.uk** — it only publishes **club-league** CSVs (England, Spain, Italy, +16 leagues); it has **no national-team / World Cup odds feed**. No free, ready dataset of historical international 1X2 odds was found. Rather than build a loader with no data behind it, this is documented as the highest-ROI upgrade *if* a real odds source is sourced (paid APIs exist).

> ⚠️ **Honest finding on the "even match = draw" rule:** of the 38 backtest matches the model rated as near-ties (`|P1 − P2| ≤ 3 pts`), only **21.1%** ended in a draw — *below* the group-stage base draw rate of 24.7%. So an even forecast does **not** make a draw more likely; even games still usually have a winner. The draw rule in `mi_polla.md` is a deliberate scoring-pool choice (favoring the 3-pt outcome bet on a coin-flip game), **not** a data-backed prediction.

---

## 🚀 Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# refresh team data (live ELO download) — optional
python scripts/update_team_data.py

# train + simulate the whole tournament (pre-tournament mode)
python scripts/run_groups_simulation.py --sims 50000

# validate the method on past World Cups (1998-2022), + save calibration plot
python scripts/backtest_world_cups.py --plots

# log a real result into the README scorecard (recomputes the points total)
python scripts/update_scorecard.py --match "Francia vs Senegal" --score 3-1

# OPTIONAL — blend bookmaker odds into the forecast (the one real accuracy lever):
python scripts/fetch_odds_2026.py            # writes a template to fill by hand…
#   …or with an API key:  ODDS_API_KEY=xxx python scripts/fetch_odds_2026.py
python scripts/run_groups_simulation.py --sims 50000 --alpha 0.3   # 0.3·model + 0.7·market
```

Useful flags: `--backend poisson|gbm|xgb` · `--train historical|synthetic` · `--cutoff 2026-06-11` (ignore everything from this date on; `--cutoff none` to include matches already played) · `--no-classifier` · `--half-life 3.0` (temporal-decay half-life in years) · `--noise-sigma 0.10` (per-match lognormal "luck" noise in the Monte Carlo) · `--alpha 1.0` (market blend: 1.0 = model only, 0.0 = market only, 0.5 = half-and-half) · `--odds <csv>`.

> 💸 **Market blend (`blend_with_market`).** Bookmaker 1X2 odds are the strongest external predictor, so the simulator can fold them in: for each match it takes the model's Dixon-Coles 1X2, de-margins the market odds to probabilities, blends them as a **log-linear opinion pool** (`p ∝ p_model^α · p_market^(1−α)`), then **re-solves the (λ_a, λ_b)** that reproduce the blended 1X2 — so the Monte Carlo still samples scorelines, now anchored to the market. `--alpha` is the model weight (default `1.0` = unchanged). Get the odds with `scripts/fetch_odds_2026.py` (the-odds-api, or a hand-filled `files/f0_raw/odds_2026.csv`). This is the highest-ROI upgrade for the actual pool — and unlike historical odds, current 2026 odds are freely available right now.

Outputs land in:

- `results/reports/mi_polla.md` — **the filled-in pool** (72 scorelines)
- `results/reports/prediccion_fase_grupos.md` — group-by-group detail with probabilities
- `results/reports/torneo_completo.md` — full bracket to the champion
- `results/reports/backtest_world_cups.csv` — per-match backtest predictions vs reality (1998-2022)
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
│   ├── result_classifier.py  # 1X2 multiclass cross-check
│   └── stacked_classifier.py # time-series OOF stacking + isotonic (reported; worse)
├── utils/
│   └── metrics.py       # RPS, log-loss, Brier, reliability table + plot
├── tuning/
│   └── optuna_tuner.py  # leak-free Optuna search (TimeSeriesSplit, RPS objective)
└── simulation/
    ├── monte_carlo.py    # group stage + knockout MC, Dixon-Coles, pi-ratings, incentives
    ├── knockout.py       # 2026 bracket, third-place allocation, deterministic run
    └── report.py         # markdown/console reports
scripts/
├── update_team_data.py   # refresh ELO (live) + market values
├── fetch_odds_2026.py        # download/prepare bookmaker 1X2 odds for the blend
├── run_groups_simulation.py  # end-to-end pipeline (--half-life, --noise-sigma, --alpha)
├── update_scorecard.py       # log real results into the README scorecard
├── backtest_world_cups.py    # validation on past World Cups (1998-2022, --plots)
├── experiment_features.py    # measure Part-2 candidate features (leak-free)
├── tune_models.py            # Optuna + TimeSeriesSplit hyper-param search (leak-free)
├── experiment_models.py      # backend/alpha/rho/temperature sweep (leak-free)
├── experiment_elo.py         # ELO home-adv / K / decay tuning (leak-free)
├── experiment_fifa.py        # marginal-value test of FIFA ranking as a feature
├── experiment_draw_adjust.py # confederation draw-adjustment test (leak-free)
└── analyze_probabilities.py  # probability/team pattern analysis (2010-2024)
files/f0_raw/             # datasets (results, teams, fixtures)
```

---

# 🏆 PREDICTED RESULTS — FULL TOURNAMENT

*Pre-tournament prediction (trained only on data before 2026-06-11, no played matches used). Scores are the most likely scoreline consistent with the predicted 1X2 outcome; **evenly-matched games (|P(1) − P(2)| ≤ 3 pts) are called as draws**. Generated by the current model (pi-ratings + competitive-form + 3-yr decay, 50,000 sims). Regenerate with `python scripts/run_groups_simulation.py --sims 50000`.*

## Group stage — all 72 matches

**Bold** = predicted scoreline. Tables show the simulated final standing (top 2 + best thirds ✦ advance).

### Group A — México, Corea del Sur, Rep. Checa ✦
| Match | Score |
|---|:-:|
| México – Sudáfrica | **2-0** |
| Corea del Sur – Rep. Checa | **1-0** |
| Rep. Checa – Sudáfrica | **2-1** |
| México – Corea del Sur | **1-0** |
| Rep. Checa – México | **0-2** |
| Sudáfrica – Corea del Sur | **0-2** |

Table: **México 9** · **Corea del Sur 6** · **Rep. Checa 3 ✦** · Sudáfrica 0

### Group B — Canadá, Suiza, Bosnia y Her. ✦
| Match | Score |
|---|:-:|
| Canadá – Bosnia y Her. | **2-0** |
| Catar – Suiza | **0-2** |
| Suiza – Bosnia y Her. | **2-0** |
| Canadá – Catar | **2-0** |
| Suiza – Canadá | **0-1** |
| Bosnia y Her. – Catar | **2-1** |

Table: **Canadá 9** · **Suiza 6** · **Bosnia y Her. 3 ✦** · Catar 0

### Group C — Brasil, Marruecos, Escocia ✦
| Match | Score |
|---|:-:|
| Brasil – Marruecos | **1-0** |
| Haití – Escocia | **0-1** |
| Escocia – Marruecos | **0-1** |
| Brasil – Haití | **2-0** |
| Escocia – Brasil | **1-2** |
| Marruecos – Haití | **2-0** |

Table: **Brasil 9** · **Marruecos 6** · **Escocia 3 ✦** · Haití 0

### Group D — Turquía, EEUU
| Match | Score |
|---|:-:|
| EEUU – Paraguay | **1-1** |
| Australia – Turquía | **1-1** |
| EEUU – Australia | **1-1** |
| Turquía – Paraguay | **1-0** |
| Turquía – EEUU | **1-1** |
| Paraguay – Australia | **1-1** |

Table: **Turquía 5** · **EEUU 3** · Australia 3 · Paraguay 2

### Group E — Alemania, Ecuador, Costa de Marfil ✦
| Match | Score |
|---|:-:|
| Alemania – Curazao | **2-0** |
| Costa de Marfil – Ecuador | **0-1** |
| Alemania – Costa de Marfil | **1-0** |
| Ecuador – Curazao | **2-0** |
| Curazao – Costa de Marfil | **0-2** |
| Ecuador – Alemania | **1-1** |

Table: **Alemania 7** · **Ecuador 7** · **Costa de Marfil 3 ✦** · Curazao 0

### Group F — Países Bajos, Japón
| Match | Score |
|---|:-:|
| Países Bajos – Japón | **1-1** |
| Suecia – Túnez | **1-1** |
| Países Bajos – Suecia | **2-0** |
| Túnez – Japón | **0-1** |
| Japón – Suecia | **2-0** |
| Túnez – Países Bajos | **0-2** |

Table: **Países Bajos 7** · **Japón 7** · Túnez 1 · Suecia 1

### Group G — Bélgica, Irán
| Match | Score |
|---|:-:|
| Bélgica – Egipto | **1-0** |
| Irán – Nueva Zelanda | **1-0** |
| Bélgica – Irán | **2-1** |
| Nueva Zelanda – Egipto | **0-1** |
| Egipto – Irán | **0-1** |
| Nueva Zelanda – Bélgica | **0-2** |

Table: **Bélgica 9** · **Irán 6** · Egipto 3 · Nueva Zelanda 0

### Group H — España, Uruguay, Arabia S. ✦
| Match | Score |
|---|:-:|
| España – Cabo Verde | **2-0** |
| Arabia S. – Uruguay | **0-1** |
| España – Arabia S. | **2-0** |
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
| Senegal – Irak | **2-0** |

Table: **Francia 9** · **Noruega 6** · **Senegal 3 ✦** · Irak 0

### Group J — Argentina, Argelia, Austria ✦
| Match | Score |
|---|:-:|
| Argentina – Argelia | **1-0** |
| Austria – Jordania | **1-0** |
| Argentina – Austria | **2-0** |
| Jordania – Argelia | **1-2** |
| Argelia – Austria | **2-1** |
| Jordania – Argentina | **0-2** |

Table: **Argentina 9** · **Argelia 6** · **Austria 3 ✦** · Jordania 0

### Group K — Colombia, Portugal
| Match | Score |
|---|:-:|
| Portugal – RD Congo | **1-0** |
| Uzbekistán – Colombia | **0-1** |
| Portugal – Uzbekistán | **1-0** |
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
| Panamá – Croacia | **1-2** |
| Panamá – Inglaterra | **0-2** |
| Croacia – Ghana | **2-0** |

Table: **Inglaterra 9** · **Croacia 6** · **Panamá 3 ✦** · Ghana 0

## Knockout stage — most likely path

### Dieciseisavos

- 2026-06-28 · Corea del Sur **0-1** Suiza → **Suiza** (23%/27%/51%)
- 2026-06-29 · Alemania **2-0** Rep. Checa → **Alemania** (60%/23%/17%)
- 2026-06-29 · Países Bajos **0-1** Marruecos → **Marruecos** (34%/30%/35%)
- 2026-06-29 · Brasil **1-0** Japón → **Brasil** (46%/28%/26%)
- 2026-06-30 · Francia **2-0** Escocia → **Francia** (64%/22%/14%)
- 2026-06-30 · Ecuador **1-0** Noruega → **Ecuador** (36%/31%/33%)
- 2026-06-30 · México **1-0** Costa de Marfil → **México** (59%/27%/15%)
- 2026-07-01 · Inglaterra **1-0** Uzbekistán → **Inglaterra** (62%/25%/13%)
- 2026-07-01 · Turquía **1-0** Senegal → **Turquía** (37%/29%/34%)
- 2026-07-01 · Bélgica **1-0** Austria → **Bélgica** (48%/27%/25%)
- 2026-07-02 · Portugal **1-0** Croacia → **Portugal** (47%/28%/25%)
- 2026-07-02 · España **1-0** Argelia → **España** (61%/24%/16%)
- 2026-07-02 · Canadá **1-0** Egipto → **Canadá** (53%/30%/17%)
- 2026-07-03 · Argentina **1-0** Uruguay → **Argentina** (56%/27%/17%)
- 2026-07-03 · Colombia **1-0** Australia → **Colombia** (52%/27%/21%)
- 2026-07-03 · EEUU **0-1** Irán → **Irán** (36%/28%/36%)
### Octavos

- 2026-07-04 · Alemania **0-1** Francia → **Francia** (25%/27%/48%)
- 2026-07-04 · Suiza **0-1** Marruecos → **Marruecos** (34%/30%/36%)
- 2026-07-05 · Brasil **1-0** Ecuador → **Brasil** (44%/30%/27%)
- 2026-07-05 · México **0-1** Inglaterra → **Inglaterra** (33%/32%/36%)
- 2026-07-06 · Portugal **0-1** España → **España** (22%/27%/51%)
- 2026-07-06 · Turquía **1-2** Bélgica → **Bélgica** (29%/26%/44%)
- 2026-07-07 · Argentina **1-0** Irán → **Argentina** (60%/24%/15%)
- 2026-07-07 · Canadá **0-1** Colombia → **Colombia** (28%/31%/40%)
### Cuartos

- 2026-07-09 · Francia **1-0** Marruecos → **Francia** (45%/29%/26%)
- 2026-07-10 · España **2-0** Bélgica → **España** (57%/24%/19%)
- 2026-07-11 · Brasil **0-1** Inglaterra → **Inglaterra** (35%/29%/36%)
- 2026-07-11 · Argentina **1-0** Colombia → **Argentina** (46%/28%/26%)
### Semifinales

- 2026-07-14 · Francia **0-1** España → **España** (27%/28%/45%)
- 2026-07-15 · Inglaterra **0-1** Argentina → **Argentina** (27%/30%/43%)
### Tercer puesto

- 2026-07-18 · Francia **1-0** Inglaterra → **Francia** (37%/29%/34%)
### FINAL

- 2026-07-19 · España **1-0** Argentina → **España** (37%/29%/34%)


### 🏆 PREDICTED CHAMPION: **España** (16.4%)

## Title probabilities (50,000 Monte Carlo simulations)

| Team | Reach R16 | Quarters | Semis | Final | **CHAMPION** |
|---|:-:|:-:|:-:|:-:|:-:|
| España | 70% | 50% | 37% | 25% | **16.4%** |
| Argentina | 66% | 50% | 34% | 22% | **13.9%** |
| Francia | 67% | 44% | 27% | 15% | **8.4%** |
| Brasil | 64% | 41% | 24% | 13% | **7.1%** |
| Inglaterra | 66% | 38% | 23% | 13% | **6.8%** |
| México | 67% | 39% | 22% | 11% | **5.9%** |
| Colombia | 64% | 37% | 21% | 12% | **5.8%** |
| Portugal | 58% | 31% | 17% | 9% | **4.1%** |
| Canadá | 63% | 32% | 16% | 7% | **3.2%** |
| Suiza | 62% | 31% | 14% | 6% | **2.7%** |
| Ecuador | 55% | 28% | 14% | 6% | **2.7%** |
| Bélgica | 56% | 30% | 13% | 6% | **2.6%** |

---

## 📝 Scorecard — model hits & misses

*Real results are facts entered as the tournament unfolds; the Predicted column is the frozen pre-tournament pick of the CURRENT model (pi-ratings + 3-yr decay, 50k sims). Points recomputed.*

| Date | Grp | Match | Predicted | Actual | Pts (5/3/0) |
|---|:-:|---|:-:|:-:|:-:|
| 11/06 | A | México – Sudáfrica | 2-0 | 2-0 | ✅ 5 |
| 11/06 | A | Corea del Sur – Rep. Checa | 1-0 | 2-1 | ✅ 3 |
| 12/06 | B | Canadá – Bosnia y Her. | 2-0 | 1-1 | ❌ 0 |
| 12/06 | D | EEUU – Paraguay | 1-1 | 4-1 | ❌ 0 |
| 13/06 | B | Catar – Suiza | 0-2 | 1-1 | ❌ 0 |
| 13/06 | C | Brasil – Marruecos | 1-0 | 1-1 | ❌ 0 |
| 13/06 | C | Haití – Escocia | 0-1 | 0-2 | ✅ 3 |
| 14/06 | D | Australia – Turquía | 1-1 | 2-0 | ❌ 0 |
| 14/06 | E | Alemania – Curazao | 2-0 | 7-1 | ✅ 3 |
| 14/06 | F | Países Bajos – Japón | 1-1 | 2-2 | ✅ 3 |
| 14/06 | E | Costa de Marfil – Ecuador | 0-1 | 1-0 | ❌ 0 |
| 14/06 | F | Suecia – Túnez | 1-1 | 5-1 | ❌ 0 |
| 15/06 | H | España – Cabo Verde | 2-0 | 0-0 | ❌ 0 |
| 15/06 | G | Bélgica – Egipto | 1-0 | 1-1 | ❌ 0 |
| 15/06 | H | Arabia S. – Uruguay | 0-1 | 1-1 | ❌ 0 |
| 15/06 | G | Irán – Nueva Zelanda | 1-0 | 2-2 | ❌ 0 |
| 16/06 | I | Francia – Senegal | 1-0 | 3-1 | ✅ 3 |
| 16/06 | I | Irak – Noruega | 0-2 | 1-4 | ✅ 3 |
| 16/06 | J | Argentina – Argelia | 1-0 |   |   |
| 17/06 | J | Austria – Jordania | 1-0 |   |   |
| 17/06 | K | Portugal – RD Congo | 1-0 |   |   |
| 17/06 | L | Inglaterra – Croacia | 1-0 |   |   |
| 17/06 | L | Ghana – Panamá | 0-1 |   |   |
| 17/06 | K | Uzbekistán – Colombia | 0-1 |   |   |
| 18/06 | A | Rep. Checa – Sudáfrica | 2-1 |   |   |
| 18/06 | B | Suiza – Bosnia y Her. | 2-0 |   |   |
| 18/06 | B | Canadá – Catar | 2-0 |   |   |
| 18/06 | A | México – Corea del Sur | 1-0 |   |   |
| 19/06 | D | EEUU – Australia | 1-1 |   |   |
| 19/06 | C | Escocia – Marruecos | 0-1 |   |   |
| 19/06 | C | Brasil – Haití | 2-0 |   |   |
| 19/06 | D | Turquía – Paraguay | 1-0 |   |   |
| 20/06 | F | Países Bajos – Suecia | 2-0 |   |   |
| 20/06 | E | Alemania – Costa de Marfil | 1-0 |   |   |
| 20/06 | E | Ecuador – Curazao | 2-0 |   |   |
| 21/06 | F | Túnez – Japón | 0-1 |   |   |
| 21/06 | H | España – Arabia S. | 2-0 |   |   |
| 21/06 | G | Bélgica – Irán | 2-1 |   |   |
| 21/06 | H | Uruguay – Cabo Verde | 2-0 |   |   |
| 21/06 | G | Nueva Zelanda – Egipto | 0-1 |   |   |
| 22/06 | J | Argentina – Austria | 2-0 |   |   |
| 22/06 | I | Francia – Irak | 2-0 |   |   |
| 22/06 | I | Noruega – Senegal | 1-0 |   |   |
| 22/06 | J | Jordania – Argelia | 1-2 |   |   |
| 23/06 | K | Portugal – Uzbekistán | 1-0 |   |   |
| 23/06 | L | Inglaterra – Ghana | 2-0 |   |   |
| 23/06 | L | Panamá – Croacia | 1-2 |   |   |
| 23/06 | K | Colombia – RD Congo | 2-0 |   |   |
| 24/06 | B | Suiza – Canadá | 0-1 |   |   |
| 24/06 | B | Bosnia y Her. – Catar | 2-1 |   |   |
| 24/06 | C | Escocia – Brasil | 1-2 |   |   |
| 24/06 | C | Marruecos – Haití | 2-0 |   |   |
| 24/06 | A | Rep. Checa – México | 0-2 |   |   |
| 24/06 | A | Sudáfrica – Corea del Sur | 0-2 |   |   |
| 25/06 | E | Curazao – Costa de Marfil | 0-2 |   |   |
| 25/06 | E | Ecuador – Alemania | 1-1 |   |   |
| 25/06 | F | Japón – Suecia | 2-0 |   |   |
| 25/06 | F | Túnez – Países Bajos | 0-2 |   |   |
| 25/06 | D | Turquía – EEUU | 1-1 |   |   |
| 25/06 | D | Paraguay – Australia | 1-1 |   |   |
| 26/06 | I | Noruega – Francia | 1-2 |   |   |
| 26/06 | I | Senegal – Irak | 2-0 |   |   |
| 26/06 | H | Cabo Verde – Arabia S. | 1-2 |   |   |
| 26/06 | H | Uruguay – España | 0-1 |   |   |
| 26/06 | G | Egipto – Irán | 0-1 |   |   |
| 26/06 | G | Nueva Zelanda – Bélgica | 0-2 |   |   |
| 27/06 | L | Panamá – Inglaterra | 0-2 |   |   |
| 27/06 | L | Croacia – Ghana | 2-0 |   |   |
| 27/06 | K | Colombia – Portugal | 2-1 |   |   |
| 27/06 | K | RD Congo – Uzbekistán | 0-1 |   |   |
| 27/06 | J | Argelia – Austria | 2-1 |   |   |
| 27/06 | J | Jordania – Argentina | 0-2 |   |   |

**Running total: 23 pts** · exact scores: 1/18 · outcomes (≥3pts): 7/18 · played: 18/72

---

## ⚠️ Notes & limitations

- The "most likely path" bracket is the modal outcome of each match in sequence — the *probability of this exact bracket happening is tiny* (that's football). The Monte Carlo probabilities are the honest forecast.
- xG features are ELO-derived proxies (no public xG provider for all 48 national teams); squad market value, caps and injuries are neutralised in historical-training mode because they don't exist retroactively. **Bookmaker odds** would be the highest-ROI upgrade but no free national-team odds source exists (see backtest section).
- Model upgrades that were *measured and kept*: pi-ratings (attack/defence), 3-year decay, and **competitive-form** (form in non-friendly matches) — together RPS 0.2009 → **0.1971**. Upgrades *measured and rejected* (no RPS gain): momentum, form-10, form-variance, stacking, FIFA ranking, Optuna-tuned hyper-params (half-life is flat 1.5–8 yr).
- To re-predict mid-tournament with played matches locked in: refresh the results dataset and run with `--cutoff none`. Log real results into the scorecard with `scripts/update_scorecard.py`.

*Built with scikit-learn, XGBoost, Optuna, pandas and 50,000 parallel universes.* 🎲
