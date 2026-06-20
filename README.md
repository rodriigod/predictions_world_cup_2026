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

### 4b. Scoreline rule — two methods (`pred_score` vs `pred_score_round`)

The filled-in scoreline can be derived two ways; both are in `match_predictions.csv`:

| Method | How | Behaviour | On 27 played group games |
|---|---|---|:-:|
| **Expected-points (production, `pred_score`)** | argmax over simulated scores of `E[pts]=3·P(result)+2·P(score)` | recovers realistic **1-0, 2-0, 0-0** (66/144 clean sheets) | **44 pts**, 14/27 results, 1 exact |
| Round-half-up (`pred_score_round`) | `round(expected goals)` (1.5→2, 1.4→1) | **never predicts 0**, mostly 2-1/1-2 | 46 pts, 14/27 results, 2 exact |

**Key finding:** the **W/D/L accuracy is identical either way (14/27 ≈ 52%)** — the result is decided by the model, *not* by the scoreline rule. Total points are within noise on 27 games (the expected-points method is optimal *in the long run*; rounding got lucky with one extra exact here). Production uses the expected-points scoreline because it is principled and produces realistic scorelines.

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

# predict ANY country vs country (W/D/L), using all international history
python scripts/predict_match.py "Brazil" "Argentina"
python scripts/predict_match.py --evaluate     # ~60% accuracy on 4.5k held-out games

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
>
> **How much weight to the market? (measured, `validate_blend_alpha.py`).** No free historical World-Cup odds exist (OddsPortal is JS-obfuscated; odds APIs charge for history), so the blend was validated on **130k club matches that DO carry Bet365 odds + pre-match ELO** (leak-free, train ≤2018 / test ≥2018). Result: a simple ELO model scores RPS 0.206, the **market alone scores 0.201**, and the optimal blend weight is **α≈0.0 (trust the market)** — adding model weight only worsened RPS. Read-through for 2026: lean heavily on the market when it's available; the production `--alpha 0.3` (70% market) is on the right side, and the data would even support a touch lower. (Our WC model is stronger than a bare ELO, so a small model weight is still defensible — but the market clearly dominates.)

> 🧩 **Pre-match 11v11 engine (V3, standalone — `scripts/pre_match.py`).** A separate manual tool for when you know the actual starting line-ups: it derives each team's λ from the 11 starters' per-90 stats (npxG, pressures, GK PSxG-save%), micro-simulates the match, and blends it with the historical model **and** the market (log-odds, `--alpha`/`--beta`). It does **not** touch the main pipeline. Player stats come from FBref — but **FBref returns HTTP 403 to scrapers**, so in practice you pass stats by hand in the line-up JSON (any stat field present overrides the scraper) or it falls back to position-average defaults.
>
> **Is the engine signal or noise? (measured, `validate_microsim.py`).** Ran on **192 real line-ups from the 2014/2018/2022 World Cups** (Fjelstul DB). Two scenarios: **(A)** with position defaults only (what you actually get without per-player stats) → λ is **identical for both teams**, correlation with results **+0.00, win-rate flat** = literally **zero signal**. **(B)** feeding a leak-free per-player attacking proxy (prior-WC goals) → λ correlates with the real margin (**Pearson +0.33**) and the home win-rate climbs **30% → 61%** from the low to high λ tercile. **Verdict: the engine mechanism works — but only if fed real per-player quality.** The bottleneck is 100% the player-stat data (which FBref blocks), not the model.
>
> **Player data — solved via FIFA ratings (`src/data/fifa_ratings.py`).** Since FBref blocks scrapers, the engine now draws per-player quality from the freely-downloadable **FIFA-24 player dataset** (~5.7k players, 135 countries: finishing, shot_power, GK ratings…). Run `python scripts/pre_match.py --lineup … --stats fifa` (default). It's a video-game *proxy* (not real npxG) and matches ~5-6 of 11 names per side (accents/short names miss → position default), but it makes λ differentiate by real squad quality (e.g. Argentina λ 1.69 vs España 1.44) — good enough to predict current matches, the bottleneck the engine needed.
>
> **Does blending the micro into the group-stage forecast help? No (measured, `microsim_groupstage.py`).** Auto-picking each country's XI from FIFA and blending the micro's λ into the production model, scored against the 27 played group matches: model **46 polla pts** vs model+micro **41** at micro-weight 0.3 (**−5**), and a tie at weight 0.15. The micro never improves the pool — consistent with every other ensemble test here. It also can't cover smaller nations (Qatar, Uzbekistan, Iran… have <11 players in the FIFA set). Use it for line-up what-ifs, not as a forecaster.

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
├── fetch_wc_odds_historical.py # (best-effort) historical WC odds — OddsPortal not scrapable
├── scrape_oddsportal_wc.py     # (optional) Selenium scraper for OddsPortal WC odds
├── validate_blend_alpha.py     # measure the optimal market-blend α on real odds data
├── validate_microsim.py        # test the V3 engine for signal on real WC line-ups
├── predict_match.py            # GENERAL any-country A-vs-B W/D/L predictor (+ --evaluate)
├── microsim_groupstage.py      # group-stage A/B test: model vs model+microsim (polla pts)
├── pre_match.py              # V3: 11v11 micro-sim from real line-ups (standalone)
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

# 🧾 Project log — everything built & everything tried (honest)

A complete, honest record of the work — including the dead ends, because knowing what *doesn't* work is half the value.

### ✅ Adopted — measurably improved the model
| Change | Effect |
|---|---|
| **Pi-ratings** (online attack/defence, Constantinou & Fenton) | RPS 0.2009 → 0.1982 |
| **Decay half-life 8 → 3 years** | national-team convention; neutral-to-positive |
| **Competitive-form feature** (form in non-friendly games only) | RPS → **0.1971**; only Part-2 feature that survived |
| **Monte Carlo 10k → 50k** | tighter tail probabilities |
| **Analytic Dixon-Coles draw** (`p_*_dc` = matrix diagonal) | correct P(draw), not a heuristic |
| **Expected-points scoreline** (`3·P(result)+2·P(score)`) | realistic 1-0/2-0/0-0; optimal for the pool |
| **Live market-odds blend** (`--alpha`, the-odds-api) | folds bookmaker 1X2 into upcoming matches |

### 🧰 Capabilities/tools added
- **General A-vs-B predictor** (`predict_match.py`) — any two countries → W/D/L, **60% accuracy** on 4.5k held-out internationals.
- **Pre-match 11v11 micro-sim** (`match_engine.py`) + **FIFA-24 player ratings** (`fifa_ratings.py`) as the per-player data source (FBref blocks scrapers).
- **Backtest** over 7 World Cups + calibration plot; **scorecard updater**; **α-blend validation**; **micro-sim validation**; **group-stage A/B test**.

### ❌ Measured and discarded — negative results (don't re-try)
| Idea | Verdict |
|---|---|
| **FIFA world ranking** as a feature | **zero** value — internal ELO is a *better* predictor (RPS 0.171 vs 0.204) |
| **Stacking** (logistic+RF+XGBoost, even time-series OOF + isotonic) | worse than Poisson+DC on every metric (0.204 vs 0.197) |
| **Hyper-parameter tuning** (α, ρ, temperature; ELO K/home/half-life) | current defaults already optimal; WC-test "gains" were overfitting |
| **Momentum / form-10 / form-variance** features | no gain or worse |
| **Host-continent** feature | no gain (ELO already captures it) |
| **Confederation draw adjustment** | helps general games, *hurts* World Cups (regional artifact) |
| **"Even match = draw" heuristic** | not data-backed (even games still have a winner) — dropped |
| **Micro-sim as a forecaster** | no signal without player stats; *with* FIFA ratings it still **doesn't improve the pool** (group A/B: −5 / 0 pts) |
| Optuna tuner | built (`optuna_tuner.py`) but not promoted — prior leak-free tuning already showed defaults hold |
| **WC-2026 features** (host-nation, CONCACAF, altitude, heat, confederation) | not added: host already = `is_host`; confederation/host-continent **already measured neutral/worse**; altitude/heat **infeasible** (no venue/city column in `fixtures_2026.csv`) |

### 🎯 Odds-pipeline fine-tuning (Shin 1992 / power / log-consensus)
- **Shin de-margining is now primary** in `blend_with_market` (`src/data/odds_tools.py`), replacing proportional normalization. Measured on 130k club matches with Bet365 odds: Shin RPS **0.20086** vs proportional 0.20098 vs power 0.20084 — a *real but tiny* gain (the market is already near-efficient); adopted because it's principled and never worse.
- **Power method as cross-check:** if Shin and power disagree by >1pp on a match it's logged as an *unbalanced line — review manually* (~3.5% of club matches).
- **Multi-book log-consensus** (`logit_consensus`) implemented and ready, but we only have one consensus line per match today, so it isn't validated — wire it when per-bookmaker odds (e.g. Pinnacle) exist.
- **Auto-warning** in `validate_blend_alpha.py`: if the blend at the production α is worse than market-only it prints a *lower α* suggestion. It currently fires (blend@α=0.3 RPS 0.2015 > market 0.2010) — confirming Groll-Zeileis (EURO 2024): in tournaments the market beats the model, so **α belongs near 0**.

### 🔌 Data sources — used & explored
- **Used:** martj42 international_results (~49k matches, the core), eloratings.net, Transfermarkt, **the-odds-api** (live odds), **FIFA-24 player dataset** (micro-sim), club Bet365 dataset (α validation), **Fjelstul WC DB** (lineups for micro-sim validation).
- **Dead ends:** OddsPortal (JS-obfuscated, not scrapable), OSF/ISDB (results only, no odds), football-data.co.uk (club leagues only, no national-team odds), historical World-Cup odds (no free source exists).

### 📌 Honest bottom line
- **Ceiling reached:** ~**60%** accuracy on general internationals, ~**54%** on World-Cup matches, **RPS 0.197** (academic SOTA without odds). The scoreline rule does **not** change W/D/L accuracy.
- **The one real lever** beyond team strength is **bookmaker odds** — and the blend validation shows the market dominates, so lean on it when available.
- **The micro-sim** is a working engine starved of data; useful for line-up what-ifs, not as a forecaster.

---

# 🏆 PREDICTED RESULTS — FULL TOURNAMENT

*Forecast of the current model **blended with live bookmaker odds** (`--alpha 0.3` → 30% model + 70% market, via the-odds-api) on the 49 upcoming matches; already-played games keep the pure-model pick (no market odds exist for them). Scores are the most likely scoreline consistent with the predicted 1X2; **even games (|P(1) − P(2)| ≤ 3 pts) are called draws**. Reproduce the pure pre-tournament model with `--alpha 1.0`. Regenerate: `python scripts/fetch_odds_2026.py && python scripts/run_groups_simulation.py --sims 50000 --alpha 0.3`.*

## Group stage — all 72 matches

**Bold** = predicted scoreline. Tables show the simulated final standing (top 2 + best thirds ✦ advance).

### Group A — México, Corea del Sur, Rep. Checa ✦
| Match | Score |
|---|:-:|
| México – Sudáfrica | **2-0** |
| Corea del Sur – Rep. Checa | **2-1** |
| Rep. Checa – Sudáfrica | **1-0** |
| México – Corea del Sur | **1-0** |
| Rep. Checa – México | **0-1** |
| Sudáfrica – Corea del Sur | **0-2** |

Table: **México 9** · **Corea del Sur 6** · **Rep. Checa 3 ✦** · Sudáfrica 0

### Group B — Suiza, Canadá, Bosnia y Her. ✦
| Match | Score |
|---|:-:|
| Canadá – Bosnia y Her. | **1-0** |
| Catar – Suiza | **0-3** |
| Suiza – Bosnia y Her. | **2-0** |
| Canadá – Catar | **2-0** |
| Suiza – Canadá | **1-0** |
| Bosnia y Her. – Catar | **2-0** |

Table: **Suiza 9** · **Canadá 6** · **Bosnia y Her. 3 ✦** · Catar 0

### Group C — Brasil, Marruecos, Escocia ✦
| Match | Score |
|---|:-:|
| Brasil – Marruecos | **1-0** |
| Haití – Escocia | **0-1** |
| Escocia – Marruecos | **0-1** |
| Brasil – Haití | **3-0** |
| Escocia – Brasil | **0-2** |
| Marruecos – Haití | **2-0** |

Table: **Brasil 9** · **Marruecos 6** · **Escocia 3 ✦** · Haití 0

### Group D — EEUU, Turquía
| Match | Score |
|---|:-:|
| EEUU – Paraguay | **1-0** |
| Australia – Turquía | **0-1** |
| EEUU – Australia | **2-1** |
| Turquía – Paraguay | **1-0** |
| Turquía – EEUU | **1-2** |
| Paraguay – Australia | **1-0** |

Table: **EEUU 9** · **Turquía 6** · Paraguay 3 · Australia 0

### Group E — Alemania, Ecuador, Costa de Marfil ✦
| Match | Score |
|---|:-:|
| Alemania – Curazao | **2-0** |
| Costa de Marfil – Ecuador | **0-1** |
| Alemania – Costa de Marfil | **2-0** |
| Ecuador – Curazao | **3-0** |
| Curazao – Costa de Marfil | **0-2** |
| Ecuador – Alemania | **0-1** |

Table: **Alemania 9** · **Ecuador 6** · **Costa de Marfil 3 ✦** · Curazao 0

### Group F — Países Bajos, Japón
| Match | Score |
|---|:-:|
| Países Bajos – Japón | **1-0** |
| Suecia – Túnez | **1-0** |
| Países Bajos – Suecia | **2-0** |
| Túnez – Japón | **0-1** |
| Japón – Suecia | **1-0** |
| Túnez – Países Bajos | **0-2** |

Table: **Países Bajos 9** · **Japón 6** · Suecia 3 · Túnez 0

### Group G — Bélgica, Egipto, Irán ✦
| Match | Score |
|---|:-:|
| Bélgica – Egipto | **1-0** |
| Irán – Nueva Zelanda | **1-0** |
| Bélgica – Irán | **2-0** |
| Nueva Zelanda – Egipto | **0-1** |
| Egipto – Irán | **1-0** |
| Nueva Zelanda – Bélgica | **0-2** |

Table: **Bélgica 9** · **Egipto 6** · **Irán 3 ✦** · Nueva Zelanda 0

### Group H — España, Uruguay
| Match | Score |
|---|:-:|
| España – Cabo Verde | **3-0** |
| Arabia S. – Uruguay | **0-1** |
| España – Arabia S. | **3-0** |
| Uruguay – Cabo Verde | **1-0** |
| Cabo Verde – Arabia S. | **1-2** |
| Uruguay – España | **0-2** |

Table: **España 9** · **Uruguay 6** · Arabia S. 3 · Cabo Verde 0

### Group I — Francia, Noruega, Senegal ✦
| Match | Score |
|---|:-:|
| Francia – Senegal | **1-0** |
| Irak – Noruega | **0-2** |
| Francia – Irak | **3-0** |
| Noruega – Senegal | **1-0** |
| Noruega – Francia | **0-1** |
| Senegal – Irak | **2-0** |

Table: **Francia 9** · **Noruega 6** · **Senegal 3 ✦** · Irak 0

### Group J — Argentina, Austria
| Match | Score |
|---|:-:|
| Argentina – Argelia | **1-0** |
| Austria – Jordania | **1-0** |
| Argentina – Austria | **2-0** |
| Jordania – Argelia | **0-2** |
| Argelia – Austria | **0-1** |
| Jordania – Argentina | **0-2** |

Table: **Argentina 9** · **Austria 6** · Argelia 3 · Jordania 0

### Group K — Portugal, Colombia, RD Congo ✦
| Match | Score |
|---|:-:|
| Portugal – RD Congo | **1-0** |
| Uzbekistán – Colombia | **0-2** |
| Portugal – Uzbekistán | **2-0** |
| Colombia – RD Congo | **2-0** |
| Colombia – Portugal | **0-1** |
| RD Congo – Uzbekistán | **2-1** |

Table: **Portugal 9** · **Colombia 6** · **RD Congo 3 ✦** · Uzbekistán 0

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

### Dieciseisavos

- 2026-06-28 · Corea del Sur **0-1** Canadá → **Canadá** (21%/30%/49%)
- 2026-06-29 · Alemania **2-0** Rep. Checa → **Alemania** (60%/23%/17%)
- 2026-06-29 · Países Bajos **0-1** Marruecos → **Marruecos** (34%/30%/35%)
- 2026-06-29 · Brasil **1-0** Japón → **Brasil** (46%/28%/26%)
- 2026-06-30 · Francia **1-0** Paraguay → **Francia** (55%/27%/18%)
- 2026-06-30 · Ecuador **1-0** Noruega → **Ecuador** (36%/31%/33%)
- 2026-06-30 · México **1-0** Costa de Marfil → **México** (59%/27%/15%)
- 2026-07-01 · Inglaterra **1-0** RD Congo → **Inglaterra** (66%/24%/11%)
- 2026-07-01 · EEUU **2-0** Bosnia y Her. → **EEUU** (68%/20%/11%)
- 2026-07-01 · Bélgica **1-0** Senegal → **Bélgica** (45%/28%/27%)
- 2026-07-02 · Colombia **1-0** Croacia → **Colombia** (51%/27%/23%)
- 2026-07-02 · España **2-0** Austria → **España** (65%/23%/13%)
- 2026-07-02 · Suiza **1-0** Irán → **Suiza** (43%/29%/29%)
- 2026-07-03 · Argentina **1-0** Uruguay → **Argentina** (56%/27%/17%)
- 2026-07-03 · Portugal **1-0** Argelia → **Portugal** (47%/28%/25%)
- 2026-07-03 · Turquía **1-0** Egipto → **Turquía** (49%/29%/23%)
### Octavos

- 2026-07-04 · Alemania **0-1** Francia → **Francia** (25%/27%/48%)
- 2026-07-04 · Canadá **1-0** Marruecos → **Canadá** (34%/33%/33%)
- 2026-07-05 · Brasil **1-0** Ecuador → **Brasil** (44%/30%/27%)
- 2026-07-05 · México **0-1** Inglaterra → **Inglaterra** (33%/32%/36%)
- 2026-07-06 · Colombia **0-1** España → **España** (25%/27%/48%)
- 2026-07-06 · EEUU **1-2** Bélgica → **Bélgica** (29%/26%/45%)
- 2026-07-07 · Argentina **2-0** Turquía → **Argentina** (62%/23%/15%)
- 2026-07-07 · Suiza **0-1** Portugal → **Portugal** (30%/28%/42%)
### Cuartos

- 2026-07-09 · Francia **1-0** Canadá → **Francia** (43%/31%/26%)
- 2026-07-10 · España **2-0** Bélgica → **España** (57%/24%/19%)
- 2026-07-11 · Brasil **0-1** Inglaterra → **Inglaterra** (35%/29%/36%)
- 2026-07-11 · Argentina **1-0** Portugal → **Argentina** (49%/27%/23%)
### Semifinales

- 2026-07-14 · Francia **0-1** España → **España** (27%/28%/45%)
- 2026-07-15 · Inglaterra **0-1** Argentina → **Argentina** (27%/30%/43%)
### Tercer puesto

- 2026-07-18 · Francia **1-0** Inglaterra → **Francia** (37%/29%/34%)
### FINAL

- 2026-07-19 · España **1-0** Argentina → **España** (37%/29%/34%)


### 🏆 PREDICTED CHAMPION: **España** (16.6%)

## Title probabilities (50,000 Monte Carlo simulations)

| Team | Reach R16 | Quarters | Semis | Final | **CHAMPION** |
|---|:-:|:-:|:-:|:-:|:-:|
| España | 71% | 51% | 37% | 26% | **16.6%** |
| Argentina | 67% | 50% | 34% | 23% | **14.3%** |
| Francia | 68% | 44% | 27% | 15% | **8.3%** |
| Brasil | 66% | 41% | 24% | 13% | **6.9%** |
| Inglaterra | 66% | 40% | 24% | 13% | **6.8%** |
| Colombia | 64% | 35% | 21% | 12% | **5.9%** |
| México | 65% | 38% | 21% | 11% | **5.6%** |
| Portugal | 63% | 34% | 18% | 9% | **4.4%** |
| Canadá | 61% | 31% | 15% | 7% | **3.2%** |
| Ecuador | 55% | 27% | 13% | 6% | **2.7%** |
| Bélgica | 60% | 33% | 14% | 7% | **2.7%** |
| Suiza | 61% | 30% | 14% | 6% | **2.5%** |

---

## 📝 Scorecard — model hits & misses

*Real results are facts entered as the tournament unfolds; the Predicted column is the frozen pre-tournament pick of the CURRENT model (pi-ratings + 3-yr decay, 50k sims). Points recomputed.*

| Date | Grp | Match | Predicted | Actual | Pts (5/3/0) |
|---|:-:|---|:-:|:-:|:-:|
| 11/06 | A | México – Sudáfrica | 2-0 | 2-0 | ✅ 5 |
| 11/06 | A | Corea del Sur – Rep. Checa | 2-1 | 2-1 | ✅ 5 |
| 12/06 | B | Canadá – Bosnia y Her. | 1-0 | 1-1 | ❌ 0 |
| 12/06 | D | EEUU – Paraguay | 1-0 | 4-1 | ✅ 3 |
| 13/06 | B | Catar – Suiza | 0-3 | 1-1 | ❌ 0 |
| 13/06 | C | Brasil – Marruecos | 1-0 | 1-1 | ❌ 0 |
| 13/06 | C | Haití – Escocia | 0-1 | 0-2 | ✅ 3 |
| 14/06 | D | Australia – Turquía | 0-1 | 2-0 | ❌ 0 |
| 14/06 | E | Alemania – Curazao | 2-0 | 7-1 | ✅ 3 |
| 14/06 | F | Países Bajos – Japón | 1-0 | 2-2 | ❌ 0 |
| 14/06 | E | Costa de Marfil – Ecuador | 0-1 | 1-0 | ❌ 0 |
| 14/06 | F | Suecia – Túnez | 1-0 | 5-1 | ✅ 3 |
| 15/06 | H | España – Cabo Verde | 3-0 | 0-0 | ❌ 0 |
| 15/06 | G | Bélgica – Egipto | 1-0 | 1-1 | ❌ 0 |
| 15/06 | H | Arabia S. – Uruguay | 0-1 | 1-1 | ❌ 0 |
| 15/06 | G | Irán – Nueva Zelanda | 1-0 | 2-2 | ❌ 0 |
| 16/06 | I | Francia – Senegal | 1-0 | 3-1 | ✅ 3 |
| 16/06 | I | Irak – Noruega | 0-2 | 1-4 | ✅ 3 |
| 16/06 | J | Argentina – Argelia | 1-0 | 3-0 | ✅ 3 |
| 17/06 | J | Austria – Jordania | 1-0 | 3-1 | ✅ 3 |
| 17/06 | K | Portugal – RD Congo | 1-0 | 1-1 | ❌ 0 |
| 17/06 | L | Inglaterra – Croacia | 1-0 | 4-2 | ✅ 3 |
| 17/06 | L | Ghana – Panamá | 0-1 | 1-0 | ❌ 0 |
| 17/06 | K | Uzbekistán – Colombia | 0-2 | 1-3 | ✅ 3 |
| 18/06 | A | Rep. Checa – Sudáfrica | 1-0 | 1-1 | ❌ 0 |
| 18/06 | B | Suiza – Bosnia y Her. | 2-0 | 4-1 | ✅ 3 |
| 18/06 | B | Canadá – Catar | 2-0 | 6-0 | ✅ 3 |
| 18/06 | A | México – Corea del Sur | 1-0 | 1-0 | ✅ 5 |
| 19/06 | D | EEUU – Australia | 2-1 | 2-0 | ✅ 3 |
| 19/06 | C | Escocia – Marruecos | 0-1 | 0-1 | ✅ 5 |
| 19/06 | C | Brasil – Haití | 3-0 | 3-0 | ✅ 5 |
| 19/06 | D | Turquía – Paraguay | 1-0 | 0-1 | ❌ 0 |
| 20/06 | F | Países Bajos – Suecia | 2-0 |   |   |
| 20/06 | E | Alemania – Costa de Marfil | 2-0 |   |   |
| 20/06 | E | Ecuador – Curazao | 3-0 |   |   |
| 21/06 | F | Túnez – Japón | 0-1 |   |   |
| 21/06 | H | España – Arabia S. | 3-0 |   |   |
| 21/06 | G | Bélgica – Irán | 2-0 |   |   |
| 21/06 | H | Uruguay – Cabo Verde | 1-0 |   |   |
| 21/06 | G | Nueva Zelanda – Egipto | 0-1 |   |   |
| 22/06 | J | Argentina – Austria | 2-0 |   |   |
| 22/06 | I | Francia – Irak | 3-0 |   |   |
| 22/06 | I | Noruega – Senegal | 1-0 |   |   |
| 22/06 | J | Jordania – Argelia | 0-2 |   |   |
| 23/06 | K | Portugal – Uzbekistán | 2-0 |   |   |
| 23/06 | L | Inglaterra – Ghana | 2-0 |   |   |
| 23/06 | L | Panamá – Croacia | 0-1 |   |   |
| 23/06 | K | Colombia – RD Congo | 2-0 |   |   |
| 24/06 | B | Suiza – Canadá | 1-0 |   |   |
| 24/06 | B | Bosnia y Her. – Catar | 2-0 |   |   |
| 24/06 | C | Escocia – Brasil | 0-2 |   |   |
| 24/06 | C | Marruecos – Haití | 2-0 |   |   |
| 24/06 | A | Rep. Checa – México | 0-1 |   |   |
| 24/06 | A | Sudáfrica – Corea del Sur | 0-2 |   |   |
| 25/06 | E | Curazao – Costa de Marfil | 0-2 |   |   |
| 25/06 | E | Ecuador – Alemania | 0-1 |   |   |
| 25/06 | F | Japón – Suecia | 1-0 |   |   |
| 25/06 | F | Túnez – Países Bajos | 0-2 |   |   |
| 25/06 | D | Turquía – EEUU | 1-2 |   |   |
| 25/06 | D | Paraguay – Australia | 1-0 |   |   |
| 26/06 | I | Noruega – Francia | 0-1 |   |   |
| 26/06 | I | Senegal – Irak | 2-0 |   |   |
| 26/06 | H | Cabo Verde – Arabia S. | 1-2 |   |   |
| 26/06 | H | Uruguay – España | 0-2 |   |   |
| 26/06 | G | Egipto – Irán | 1-0 |   |   |
| 26/06 | G | Nueva Zelanda – Bélgica | 0-2 |   |   |
| 27/06 | L | Panamá – Inglaterra | 0-2 |   |   |
| 27/06 | L | Croacia – Ghana | 2-0 |   |   |
| 27/06 | K | Colombia – Portugal | 0-1 |   |   |
| 27/06 | K | RD Congo – Uzbekistán | 2-1 |   |   |
| 27/06 | J | Argelia – Austria | 0-1 |   |   |
| 27/06 | J | Jordania – Argentina | 0-2 |   |   |

**Running total: 59 pts** · exact scores: 4/30 · outcomes (≥3pts): 17/30 · played: 30/72

---

## ⚠️ Notes & limitations

- The "most likely path" bracket is the modal outcome of each match in sequence — the *probability of this exact bracket happening is tiny* (that's football). The Monte Carlo probabilities are the honest forecast.
- xG features are ELO-derived proxies (no public xG provider for all 48 national teams); squad market value, caps and injuries are neutralised in historical-training mode because they don't exist retroactively.
- **Market blend** (`--alpha`): the production forecast now folds live bookmaker 1X2 odds into the upcoming matches (log-linear pool, then λ re-solved so the Monte Carlo still samples scorelines). This is the one real accuracy lever beyond team strength. Pure-model and pure-market are reproducible with `--alpha 1.0` / `--alpha 0.0`.
- **pi-ratings (attack/defence)**, **3-year decay** and **form_competitive_diff** improved the leak-free backtest (RPS 0.2009 → 0.1971); the **stacking ensemble** is reported but measured worse, so it does not drive scorelines.
- To re-predict mid-tournament with played matches locked in: refresh the results dataset and run with `--cutoff none`.

*Built with scikit-learn, XGBoost, pandas, Optuna and 50,000 parallel universes.* 🎲
