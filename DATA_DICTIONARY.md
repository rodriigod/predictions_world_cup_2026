# 📖 Diccionario de Datos — Predicción Fase de Grupos Mundial 2026

Dataset a nivel **partido-equipo** (cada fila = un equipo en un partido; un partido genera 2 filas espejo, o bien una fila con features `_diff` equipo vs rival). El target del modelo de Poisson/XGBoost es `goals_scored` (goles anotados por el equipo en el partido), del cual se estima **λ (goles esperados)** para alimentar la Simulación de Monte Carlo.

**Convenciones de normalización usadas en este documento:**
- `z-score`: `(x - μ_train) / σ_train` — ajustado SOLO con datos de entrenamiento (evitar leakage).
- `min-max [0,1]`: `(x - min) / (max - min)` con límites fijados a priori (no del sample) cuando el rango es conocido.
- `log1p`: `log(1 + x)` para variables con cola derecha pesada antes de escalar.
- `diff`: variable relativa `valor_equipo - valor_rival` (recomendada: el fútbol es un juego de fuerzas relativas; reduce dimensionalidad y colinealidad).

---

## CATEGORÍA 1: RENDIMIENTO DEPORTIVO (EL CORE TÉCNICO)

| # | Variable | Tipo | Descripción | Normalización / Cálculo |
|---|----------|------|-------------|--------------------------|
| 1.1 | `elo_rating` | Float | Puntaje ELO actual del equipo (fuente: eloratings.net), actualizado al día previo al partido. Captura fuerza histórica ponderada por importancia del rival y del torneo. | z-score sobre el universo de selecciones FIFA (μ≈1500, σ≈170). Alternativa: min-max con rango fijo [1200, 2200]. |
| 1.2 | `elo_diff` | Float | `elo_rating - elo_rating_rival`. La variable individual más predictiva en fútbol de selecciones. | Dividir por 400 (escala logística clásica del ELO: `1/(1+10^(-diff/400))` ≈ prob. esperada de victoria). Usar tanto `elo_diff/400` como la probabilidad implícita `elo_win_expectancy`. |
| 1.3 | `elo_win_expectancy` | Float | Probabilidad de victoria implícita en el ELO: `1 / (1 + 10^(-elo_diff/400))`. | Ya está en [0,1]; no requiere escalado. Excelente feature "ancla" para el modelo. |
| 1.4 | `fifa_ranking_points` | Float | Puntos del ranking FIFA oficial (SUM formula). Redundante con ELO pero útil como segunda opinión. | z-score. Usar `fifa_rank_log = log1p(posición)` si se prefiere la posición ordinal (comprime diferencias entre top-10). |
| 1.5 | `xg_for_last10` | Float | Goles esperados (xG) generados, promedio por partido en los últimos 10 partidos oficiales. Mide capacidad ofensiva real, descontando suerte de definición. | z-score. Rango típico [0.5, 3.0]. Imputar con media de la confederación si faltan datos de proveedor (Opta/StatsBomb). |
| 1.6 | `xg_against_last10` | Float | xG concedido promedio por partido, últimos 10 oficiales. Mide solidez defensiva estructural. | z-score. Invertir signo o usar `xg_diff_last10` para interpretación monotónica. |
| 1.7 | `xg_for_last20` | Float | xG a favor promedio, últimos 20 oficiales. Ventana larga = señal más estable, menos "momento". | z-score. Combinar con la ventana 10 vía decaimiento exponencial: `xg_ewma = Σ xG_i · e^(-0.1·i)` (i = partidos hacia atrás). |
| 1.8 | `xg_against_last20` | Float | xG en contra promedio, últimos 20 oficiales. | z-score. |
| 1.9 | `xg_overperformance_last10` | Float | `(goles_reales - xG) / partidos` en últimos 10. Mide calidad de definición (positivo) o mala suerte/mal portero (negativo). Tiende a regresar a la media: úsese con regularización fuerte. | Clipping a [-1, +1] y luego z-score. |
| 1.10 | `qualifying_points_pct` | Float | % de puntos obtenidos en la fase clasificatoria: `puntos / (partidos·3)`. Para anfitriones sin eliminatorias (México, USA, Canadá), imputar con % de puntos en partidos oficiales del ciclo. | Ya en [0,1]. Marcar imputación con flag booleano `is_host_no_qualifying`. |
| 1.11 | `qualifying_goal_diff_pg` | Float | Diferencia de goles por partido en eliminatorias: `(GF - GC) / partidos`. | z-score POR CONFEDERACIÓN (una +2.0 en OFC ≠ +2.0 en UEFA). Alternativa superior: ajustar por ELO promedio de los rivales enfrentados (`strength_adjusted_gd`). |
| 1.12 | `confed_strength_factor` | Float | Fuerza media de la confederación del equipo (ELO promedio de sus miembros), para des-sesgar métricas de clasificatorias. | min-max [0,1] sobre las 6 confederaciones. Usar como interacción multiplicativa con 1.10 y 1.11, no como feature aislada. |
| 1.13 | `h2h_wins_4y` | Integer | Victorias sobre el rival directo en los últimos 4 años (todas las competiciones). | Convertir a tasa: `h2h_win_rate_4y = (W + 0.5·E) / partidos_h2h`; si `partidos_h2h < 2`, imputar 0.5 (no hay señal). Acompañar de `h2h_n_matches_4y` para que el modelo pondere confianza. |
| 1.14 | `h2h_score_8y_weighted` | Float | Historial directo de 8 años ponderado por recencia: `Σ resultado_i · w_i` con `w_i = 0.85^(años_transcurridos)`; resultado = 1 victoria, 0.5 empate, 0 derrota. | Normalizar por `Σ w_i` → queda en [0,1]. Si no hay enfrentamientos, imputar 0.5 + flag `h2h_no_history`. |
| 1.15 | `h2h_goal_diff_8y` | Float | Diferencia de goles acumulada vs el rival en 8 años, ponderada por recencia (mismo `w_i`). | Clipping a [-3, +3] por partido equivalente, luego z-score. |
| 1.16 | `form_last5_points_pct` | Float | "Forma": % de puntos en los últimos 5 partidos (oficiales + amistosos): `puntos/15`. | Ya en [0,1]. Ponderar amistosos ×0.5 en el numerador y denominador (señal más débil). |
| 1.17 | `form_last5_weighted` | Float | Forma ponderada por calidad del rival: `Σ resultado_i · (elo_rival_i / 1800)` en últimos 5. Ganarle a Brasil vale más que a una selección débil. | min-max [0,1] con máximo teórico = 5·(2100/1800). |
| 1.18 | `goals_scored_last5_pg` | Float | Goles anotados por partido, últimos 5. Señal de "momento ofensivo" (ruidosa: regularizar). | z-score. |
| 1.19 | `clean_sheets_last5` | Integer | Vallas invictas en los últimos 5 partidos. | Dividir por 5 → [0,1]. |
| 1.20 | `days_since_last_official` | Integer | Días desde el último partido oficial (detecta selecciones "oxidadas" o sin rodaje conjunto). | log1p + z-score. |

---

## CATEGORÍA 2: CONTEXTO, ENTORNO Y LOGÍSTICA

| # | Variable | Tipo | Descripción | Normalización / Cálculo |
|---|----------|------|-------------|--------------------------|
| 2.1 | `is_host` | Boolean | El equipo juega en su propio país (USA, México o Canadá en 2026). Localía real: histórico +0.3 a +0.5 goles de ventaja. | 0/1 directo. |
| 2.2 | `distance_home_to_venue_km` | Float | Distancia great-circle (Haversine) entre la capital del país del equipo y la ciudad sede del partido. Proxy de localía simulada y desgaste de aclimatación. | log1p y luego min-max [0,1] con máximo ≈ log1p(18000). La relación no es lineal: 500 km ≈ 0 efecto, >8000 km sí. |
| 2.3 | `timezone_shift_hours` | Integer | Diferencia horaria absoluta entre el país de origen y la sede. Jet lag: efecto medible con ≥5 husos. | min-max [0,1] con máx = 12. Considerar `tz_shift_east` (viajar al este penaliza más, según literatura de cronobiología). |
| 2.4 | `expected_fan_share` | Float | % estimado de la fanaticada del equipo en el estadio (venta de entradas por federación, diáspora local — clave en USA para México, Colombia, etc.). Estimable con datos de población migrante por ciudad sede (censo USA) + asignación FIFA. | Ya en [0,1]. Si no hay dato: modelo proxy `diáspora_ciudad / capacidad + cuota_federación`. Usar `fan_share_diff` vs rival. |
| 2.5 | `stadium_capacity` | Integer | Capacidad del estadio. Modula la presión ambiental (interactúa con 2.4). | min-max [0,1] con rango [45000, 95000] (estadios 2026). Feature de interacción: `crowd_pressure = expected_fan_share_diff · stadium_capacity_norm`. |
| 2.6 | `temperature_expected_c` | Float | Temperatura media esperada a la hora del kickoff (°C). Partidos de junio-julio en Texas/Florida pueden superar 35°C. Penaliza a equipos de climas fríos y ritmo alto. | z-score. Mejor aún: `temp_delta_vs_home = temp_sede - temp_media_país_origen_junio` (mide des-aclimatación, no calor absoluto). |
| 2.7 | `humidity_expected_pct` | Float | Humedad relativa esperada (%) el día del partido. Interactúa con temperatura: usar índice de calor (heat index) combinado. | Dividir por 100 → [0,1]. Feature derivada recomendada: `heat_index_norm` (fórmula NOAA con 2.6 y 2.7, luego min-max). |
| 2.8 | `is_indoor_stadium` | Boolean | Estadio techado/climatizado (AT&T, NRG, Mercedes-Benz, SoFi, BC Place). Anula el efecto de 2.6 y 2.7 → poner temperatura = 22°C, humedad = 50% si es 1. | 0/1. Aplicar como máscara sobre las variables de clima, no solo como feature. |
| 2.9 | `altitude_venue_m` | Float | Altitud del estadio (msnm). Estadio Azteca ≈ 2240 m, Guadalajara ≈ 1560 m: impacto aeróbico real para no aclimatados. | min-max [0,1] con máx = 2300. Feature clave derivada: `altitude_delta = altitude_venue - altitude_media_país_equipo` (Bolivia/Ecuador/México no sufren; Países Bajos sí). |
| 2.10 | `rest_days` | Integer | Días de descanso desde el último partido del torneo (en fecha 1: desde el último amistoso de preparación). | Clipping a [2, 8], min-max [0,1]. Lo que importa es `rest_days_diff` vs rival (ventajas de ±2 días son significativas). |
| 2.11 | `travel_km_cumulative` | Float | Km totales viajados desde el campamento base hasta cada sede, acumulados en el torneo. En 2026 (3 países, 16 ciudades) la varianza entre equipos será enorme. | log1p + z-score. También `travel_km_since_last_match` para el desgaste inmediato. |
| 2.12 | `kickoff_local_hour` | Categorical | Franja del kickoff en hora local: {`midday` (12-15h, calor máximo), `afternoon` (15-18h), `night` (18+)}. | One-hot encoding (3 dummies). Interactúa con heat index. |
| 2.13 | `pitch_type` | Categorical | Superficie: {`natural`, `hybrid`}. En 2026 los estadios NFL usarán césped natural instalado sobre superficies artificiales — calidad variable. | One-hot. Señal débil: candidata a eliminación si no aporta en validación. |

---

## CATEGORÍA 3: PLANTEL, MICRO-DATOS E INDIVIDUALIDADES

Agregados sobre los 26 convocados; el "11 ideal" se define como los 11 jugadores con más minutos en los últimos 10 partidos oficiales de la selección.

| # | Variable | Tipo | Descripción | Normalización / Cálculo |
|---|----------|------|-------------|--------------------------|
| 3.1 | `starters_season_minutes_total` | Integer | Minutos totales jugados en clubes por el 11 ideal en la temporada 2025-26 (liga + copas + internacional). Mide fatiga acumulada pre-Mundial. | z-score. Relación en U invertida: pocos minutos = falta de ritmo, demasiados (>4500 por jugador clave) = fatiga. Añadir `starters_minutes_squared` o binning para capturar no-linealidad. |
| 3.2 | `squad_market_value_log` | Float | `log(Σ valor de mercado de los 26 en M€, Transfermarkt)`. El log es esencial: la diferencia 1000→1200 M€ importa menos que 50→250 M€. | Tras el log, z-score. Usar `market_value_log_diff` vs rival como feature principal de esta familia. |
| 3.3 | `top3_value_concentration` | Float | `valor de los 3 más caros / valor total del plantel`. Mide dependencia de estrellas (riesgo si se lesionan; interactúa con 3.5). | Ya en [0,1]. |
| 3.4 | `squad_avg_age_weighted` | Float | Edad promedio ponderada por minutos esperados. Pico de rendimiento ≈ 27-28 años. | Transformar a `abs(edad - 27.5)` (distancia al pico) y luego min-max. Evita que el modelo asuma linealidad edad-rendimiento. |
| 3.5 | `injury_impact_index` | Float | Índice de bajas: `Σ (minutos_del_jugador_en_últimos_10 / minutos_totales_equipo) · valor_relativo_jugador` sobre lesionados/suspendidos para ESE partido. 0 = plantel completo; 0.25+ = pierde ~3 titulares clave. | Ya en [0,1] aprox; clipping a [0, 0.5]. Es la variable más dinámica del dataset: recalcular partido a partido. |
| 3.6 | `key_player_available` | Boolean | Disponibilidad del jugador de mayor valor de mercado (proxy: la "estrella"). Complementa 3.5 con señal binaria fuerte. | 0/1. |
| 3.7 | `suspended_players_n` | Integer | Jugadores suspendidos por acumulación de amarillas/roja para este partido (determinista: se sabe con certeza, a diferencia de lesiones). | Dividir por 3 y clipping → [0,1]. |
| 3.8 | `avg_caps` | Float | Promedio de partidos internacionales (caps) de los 26. Experiencia internacional bruta. | log1p + z-score (rendimientos decrecientes: 80→100 caps aporta poco vs 5→25). |
| 3.9 | `players_with_wc_experience` | Integer | Número de jugadores (de 26) que disputaron al menos 1 partido en mundiales previos. | Dividir por 26 → [0,1]. |
| 3.10 | `wc_minutes_total_squad` | Float | Minutos totales acumulados en mundiales previos por el plantel. Versión continua de 3.9. | log1p + z-score. |
| 3.11 | `squad_big5_league_pct` | Float | % de los 26 que juegan en las 5 grandes ligas europeas + % en clubes de Champions League (`squad_ucl_pct` como variable hermana). Proxy de roce al máximo nivel. | Ya en [0,1]. |
| 3.12 | `gk_quality_proxy` | Float | Calidad del portero titular: PSxG-GA (goles prevenidos vs esperados post-disparo) por 90' en su club, temporada 2025-26. El portero es ~30% de la varianza defensiva y los agregados de equipo lo diluyen. | Clipping a [-0.3, +0.3], luego z-score. Si no hay dato de proveedor: valor de mercado del portero, log + z-score. |
| 3.13 | `attack_value_log` / `defense_value_log` | Float | Valor de mercado desagregado por línea (DEL+MED ofensivos vs DEF+POR). Permite que el modelo de λ ofensivo use el ataque propio Y la defensa rival como inputs separados. | log1p + z-score. En el modelo de Poisson: λ_equipo ~ f(ataque_propio, defensa_rival, …). |
| 3.14 | `days_in_camp` | Integer | Días de concentración/entrenamiento conjunto antes del torneo. Mide cohesión táctica de corto plazo. | min-max [0,1] con rango [7, 30]. |

---

## CATEGORÍA 4: DINÁMICA E INCENTIVOS DEL TORNEO (FASE DE GRUPOS)

Estas variables se **recalculan dinámicamente dentro de la Simulación de Monte Carlo**: en cada iteración, el estado del grupo tras las fechas 1 y 2 determina los incentivos de la fecha 3.

| # | Variable | Tipo | Descripción | Normalización / Cálculo |
|---|----------|------|-------------|--------------------------|
| 4.1 | `matchday` | Categorical | Jornada del grupo: {1, 2, 3}. Fecha 1: conservadurismo histórico (−15% goles vs media); fecha 3: varianza máxima. | One-hot (2 dummies, fecha 1 como base). NO tratar como numérica ordinal. |
| 4.2 | `points_before_match` | Integer | Puntos del equipo en el grupo antes del partido (0-6). | Dividir por 6 → [0,1]. |
| 4.3 | `qualification_status` | Categorical | Estado matemático antes del partido: {`already_qualified` (clasificado), `draw_enough` (le sirve empatar), `must_win` (obligado a ganar), `must_win_big` (necesita ganar Y goleada/otros resultados), `eliminated` (eliminado matemáticamente), `open` (todo abierto, fechas 1-2)}. Calculado enumerando los desenlaces posibles del grupo. **En 2026 incluye la vía de mejores terceros**: 8 de 12 terceros avanzan, así que "eliminated" en fecha 3 es más raro y `third_place_alive` importa. | One-hot (5 dummies). Dentro del Monte Carlo se computa exacto por enumeración de resultados restantes. |
| 4.4 | `incentive_score` | Float | Versión continua de 4.3: `P(clasificar si gana) - P(clasificar si pierde)` estimada por sub-simulación. 0 = el partido no cambia nada (clasificado o eliminado); 1 = se juega la vida. | Ya en [0,1]. Modula λ: equipos sin incentivo rotan plantel (−10 a −20% λ y rendimiento, evidencia histórica fecha 3). |
| 4.5 | `goal_diff_incentive` | Boolean | En fecha 3: la diferencia de goles define posición relevante (1° vs 2° del grupo, o ranking de terceros). Activa el modo "ganar por goleada". | 0/1. En el simulador: si es 1, inflar λ del favorito +8% y del rival +5% (partidos abiertos). |
| 4.6 | `simultaneous_match_flag` | Boolean | Fecha 3 se juega en simultáneo con el otro partido del grupo (regla anti-biscotto FIFA). Reduce la probabilidad de empates "pactados" pero no la elimina (caso Alemania-Austria '82). | 0/1. Usar como interacción con `draw_enough` de ambos equipos: si a AMBOS les sirve el empate, inflar P(empate) +20-40% sobre lo que diga Poisson (ajuste post-hoc tipo Dixon-Coles contextual). |
| 4.7 | `opponent_rotation_risk` | Float | Probabilidad de que el RIVAL alinee suplentes (rival ya clasificado en fecha 3 con `incentive_score` ≈ 0). | Ya en [0,1]. Se aplica como descuento al λ defensivo y ofensivo del rival. |
| 4.8 | `coach_tournament_experience` | Integer | Torneos cortos internacionales (Mundial, Continental) dirigidos previamente por el DT. | log1p, min-max con máx ≈ log1p(8). |
| 4.9 | `coach_knockout_winrate` | Float | % de puntos del DT en partidos de torneos cortos (fase de grupos + eliminación directa) en su carrera con selecciones. Imputar 0.5 con <5 partidos + flag `coach_low_sample`. | Ya en [0,1]. Shrinkage bayesiano hacia 0.5: `(W + 5·0.5) / (N + 5)`. |
| 4.10 | `coach_tenure_months` | Integer | Meses del DT en el cargo. Proyectos consolidados (>24 meses) rinden mejor en mundiales que "bomberos" de última hora. | log1p + min-max. |
| 4.11 | `coach_style_cluster` | Categorical | Perfil táctico del DT: {`high_press`, `possession`, `low_block_counter`, `balanced`}, asignado por clustering de métricas de estilo (PPDA, posesión media, directness) de sus últimos 20 partidos. | One-hot. Úsese sobre todo en INTERACCIONES (ej. `low_block_counter` vs `possession` produce menos goles totales → ajusta λ de ambos). |
| 4.12 | `is_group_decider` | Boolean | El partido define directamente el 1° del grupo entre los dos contendientes (ambos con 6 pts en fecha 3, etc.). | 0/1. |

---

## CATEGORÍA EXTRA: CUOTAS DE CASAS DE APUESTAS (CONSENSO DE EXPERTOS)

La literatura moderna usa las cuotas (Pinnacle/Bet365) como features de entrada: resumen implícitamente información difícil de cuantificar (lesiones de último minuto, despidos de DT, clima). El modelo las usa como "consenso de expertos" y busca refinarlo.

| # | Variable | Tipo | Descripción | Normalización / Cálculo |
|---|----------|------|-------------|--------------------------|
| E.1 | `odds_implied_p1` / `odds_implied_px` / `odds_implied_p2` | Float | Probabilidades implícitas de las cuotas de cierre: `(1/cuota_i) / Σ(1/cuota_j)` — la división por la suma elimina el margen (overround) de la casa. | Ya en [0,1] y suman 1. Usar cuotas de CIERRE (más informativas que las de apertura). |
| E.2 | `odds_overround` | Float | Margen de la casa: `Σ(1/cuota) - 1`. Proxy de incertidumbre del mercado sobre ese partido. | Ya en escala [0.02, 0.10]; z-score. |
| E.3 | `odds_over25_implied` | Float | Probabilidad implícita de Over 2.5 goles. Informa el TOTAL de goles esperado del partido (calibra λ_a + λ_b). | Ya en [0,1]. |
| E.4 | `odds_movement_p1` | Float | Cambio de la probabilidad implícita entre apertura y cierre. Captura noticias de último minuto (lesiones, alineaciones). | Diferencia en [-0.3, 0.3]; clipping + z-score. |

**Estado en este proyecto:** definidas pero no integradas (requieren feed de odds histórico + actual, ej. football-data.co.uk para ligas u OddsAPI para selecciones). Son el upgrade de mayor retorno esperado si se consiguen los datos.

---

## 📚 Mapeo teoría → implementación en este repo

| Concepto (papers) | Estado | Dónde |
|---|---|---|
| Maher (1982): goles ~ Poisson(λ), fuerzas de ataque α y defensa β | ✅ | `core/models/poisson_goals.py` — el GLM con log-link generaliza a Maher: λ = exp(β·x) donde x incluye ataque propio (xG/goles a favor) y debilidad del rival. El ELO interno absorbe α·β. |
| Dixon-Coles (1997), corrección 1: decaimiento temporal φ | ✅ | `core/data/historical.py` (`DECAY_HALF_LIFE_YEARS`) — peso `w = exp(-años/8) × peso_torneo` pasado como `sample_weight` al modelo. |
| Dixon-Coles (1997), corrección 2: ajuste τ de marcadores bajos (0-0, 1-0, 0-1, 1-1) | ✅ | `core/simulation/monte_carlo.py` (`_dixon_coles_matrix`, ρ=-0.08). |
| ML moderno: clasificación multiclase [1, X, 2] | ✅ | `core/models/result_classifier.py` — XGBoost (multi:softprob), Random Forest y Regresión Logística Multinomial como baseline; se compara log-loss/accuracy en holdout. |
| Medias móviles de forma (pts, goles, xG últimos 3/5/10) | ✅ | `core/data/historical.py` — forma últimos 5, goles a favor/en contra últimos 10 (proxy de xG), anti-leakage (solo partidos anteriores). |
| Diferencia de ELO como feature top | ✅ | `elo_diff_scaled` + `elo_win_expectancy`; ELO propio calculado partido a partido con K por torneo y bonus de localía. |
| Efecto localía | ✅ | `is_host_own/opp` + bonus de 100 ELO para el local en el cálculo histórico. |
| Días de descanso | ✅ | `rest_days_diff` calculado de fechas reales (histórico y fixture 2026). |
| Valor de plantilla (Transfermarkt) | ⚠️ | En el diccionario y en `teams_2026.csv` (datos reales), pero el modo histórico lo neutraliza (no existe retroactivamente). Activo solo en modo sintético. |
| Cuotas de apuestas como features | ❌ | Definidas arriba (E.1-E.4); pendiente conseguir el feed de datos. |

---

## 🎯 Variables Target (para entrenamiento)

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `goals_scored` | Integer | Goles anotados por el equipo en el partido (90 minutos). Target de la Regresión de Poisson / XGBoost (`objective='count:poisson'`). |
| `match_result` | Categorical | {W, D, L} — solo para validar la calibración de las probabilidades derivadas de λ, no como target directo. |

## ⚙️ Notas de diseño para el pipeline

1. **Features relativas (`_diff`) sobre absolutas**: el modelo de λ debe recibir SIEMPRE el contraste equipo vs rival (ataque propio vs defensa rival). Reduce de ~60 a ~35 features efectivas.
2. **Anti-leakage**: toda ventana móvil (xG last10, forma, ELO) se calcula con datos estrictamente ANTERIORES a la fecha del partido. Los scalers se ajustan solo con el set de entrenamiento (partidos internacionales 2010-2025).
3. **Entrenamiento**: usar partidos oficiales de selecciones 2010-2025 (eliminatorias, mundiales, copas continentales, Nations League) — ~7000 partidos, 14000 filas. Ponderar por recencia (`w = e^(-años/4)`) y por importancia del torneo (mundial ×1.5).
4. **Dixon-Coles**: los conteos de Poisson independientes subestiman 0-0 y 1-1. Aplicar la corrección `τ(x, y, λ, μ, ρ)` con ρ ≈ -0.05 a -0.13 estimado por máxima verosimilitud.
5. **Monte Carlo**: N ≥ 20,000 iteraciones del torneo completo. Las variables de la Categoría 4 se recalculan DENTRO de cada iteración (el estado del grupo es estocástico). Reportar P(1° del grupo), P(top-2), P(mejor tercero), P(clasificar a 16avos), distribución de puntos.
